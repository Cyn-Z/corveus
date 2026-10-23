"""Transactional metadata registry for future service jobs.

This module is deliberately not connected to the running application.

It stores identifiers, request digests, timestamps, and job status.
It does not store prompts, answers, review payloads, or raw retry keys.

Only queued admission, cancellation, and expiry are implemented here.
Worker scheduling and inference resource management belong to later batches.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from service_contracts import (
    ErrorCode,
    JobState,
    JobStatus,
    JobSubmission,
    Principal,
    ServiceError,
    validate_transition,
)
from service_settings import ServiceSettings


SCHEMA_VERSION = 6
APPLICATION_ID = 0x43524A31

_OBJECT_ID = re.compile(r"[0-9a-f]{32}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}")

_SCHEMA = (
    """
    CREATE TABLE service_jobs (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL UNIQUE,
        tenant_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        idempotency_digest TEXT NOT NULL,
        request_fingerprint TEXT NOT NULL,
        execution_profile_id TEXT NOT NULL,
        conversation_id TEXT,
          state TEXT NOT NULL
            CHECK (state IN (
                'queued', 'preparing', 'awaiting_review', 'waiting_inference',
                'running', 'finalizing', 'completed', 'failed',
                'cancelled', 'expired', 'interrupted'
            )),
        version INTEGER NOT NULL CHECK (version >= 1),
        attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        cancel_requested INTEGER NOT NULL DEFAULT 0
            CHECK (cancel_requested IN (0, 1)),
        UNIQUE (tenant_id, owner_id, idempotency_digest),
        CHECK (length(job_id) = 32),
        CHECK (length(idempotency_digest) = 64),
        CHECK (length(request_fingerprint) = 64),
        CHECK (updated_at >= created_at),
        CHECK (expires_at > created_at)
    )
    """,
    """
    CREATE INDEX service_jobs_waiting
    ON service_jobs (state, sequence)
    """,
    """
    CREATE INDEX service_jobs_owner_waiting
    ON service_jobs (tenant_id, owner_id, state)
    """,
    """
    CREATE INDEX service_jobs_expiry
    ON service_jobs (state, expires_at)
    """,
)
_LEASE_SCHEMA = """
CREATE TABLE service_job_leases (
    job_id TEXT PRIMARY KEY NOT NULL
        REFERENCES service_jobs(job_id) ON DELETE CASCADE,
    worker_id TEXT NOT NULL,
    token TEXT NOT NULL CHECK (length(token) = 64),
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    lease_expires_at TEXT NOT NULL
)
"""
_HOLD_SCHEMA = """
CREATE TABLE service_inference_holds (
    job_id TEXT PRIMARY KEY NOT NULL
        REFERENCES service_jobs(job_id) ON DELETE CASCADE,
    worker_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    token TEXT NOT NULL CHECK (length(token) = 64),
    reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes >= 1),
    issued_at TEXT NOT NULL
)
"""
_REQUEST_SCHEMA = """
CREATE TABLE service_request_content (
    job_id TEXT PRIMARY KEY NOT NULL
        REFERENCES service_jobs(job_id) ON DELETE CASCADE,
    prompt TEXT NOT NULL CHECK (length(trim(prompt)) > 0),
    expires_at TEXT NOT NULL
)
"""

_REQUEST_INDEX = """
CREATE INDEX service_request_content_expiry
ON service_request_content(expires_at)
"""

_RESULT_SCHEMA = """
CREATE TABLE service_job_results (
    job_id TEXT PRIMARY KEY NOT NULL
        REFERENCES service_request_content(job_id) ON DELETE CASCADE,
    worker_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK(attempt >= 1),
    permit_digest TEXT NOT NULL CHECK(length(permit_digest) = 64),
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL CHECK(length(payload_digest) = 64),
    captured_at TEXT NOT NULL,
    persistence_state TEXT CHECK(
        persistence_state IS NULL OR persistence_state IN (
            'not_requested', 'saved',
            'memory_unavailable', 'conversation_deleted'
        )
    )
)
"""
_PREPARATION_SCHEMA = """
CREATE TABLE service_preparations (
    job_id TEXT PRIMARY KEY NOT NULL
        REFERENCES service_request_content(job_id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL CHECK(length(payload_digest) = 64),
    review_expires_at TEXT
)
"""

class UnsupportedJobSchemaError(RuntimeError):
    """The file is not a supported service-job database."""


@dataclass(frozen=True)
class Admission:
    status: JobStatus
    created: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _require_principal(principal: Principal) -> None:
    if not isinstance(principal, Principal):
        raise ValueError("principal must be a server-established Principal.")


def _require_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or _OBJECT_ID.fullmatch(job_id) is None:
        raise ValueError("job_id must contain 32 lowercase hexadecimal characters.")


def _require_version(version: int) -> None:
    if type(version) is not int or version < 1:
        raise ValueError("expected_version must be a positive integer.")


class JobStore:
    """Single-host SQLite job metadata registry.

    Each operation uses its own connection. BEGIN IMMEDIATE serializes
    admission and mutation across threads and processes using the same file.

    All service instances sharing this file must use the same queue settings.
    This class does not coordinate databases on different machines.
    """

    def __init__(
        self,
        path: str | Path,
        settings: ServiceSettings,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("path must identify a dedicated SQLite file.")
        if str(path) == ":memory:":
            raise ValueError("JobStore requires a persistent database file.")
        if not isinstance(settings, ServiceSettings):
            raise ValueError("settings must be ServiceSettings.")
        if not callable(clock):
            raise ValueError("clock must be callable.")

        self.path = Path(path).expanduser().resolve()
        self.settings = settings
        self._clock = clock

        if not self.path.parent.is_dir():
            raise ValueError("The database parent directory must already exist.")

        self._initialize()

    @contextmanager
    def _transaction(
        self,
        *,
        write: bool,
    ) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.path),
            timeout=10.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row

        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except sqlite3.OperationalError as exc:
            if connection.in_transaction:
                connection.rollback()

            error_code = getattr(exc, "sqlite_errorcode", None)
            if error_code is not None and (
                error_code & 0xFF
            ) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise ServiceError(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    retry_after_seconds=1,
                ) from None
            raise
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _migrate_5_to_6(connection: sqlite3.Connection) -> None:
        """Rebuild the parent state constraint while preserving child rows."""
        children = (
            ("service_job_leases", _LEASE_SCHEMA),
            ("service_inference_holds", _HOLD_SCHEMA),
            ("service_request_content", _REQUEST_SCHEMA),
            ("service_job_results", _RESULT_SCHEMA),
        )

        # Temporary SQL tables avoid loading all stored content into Python.
        for name, _ in children:
            connection.execute(
                f"CREATE TEMP TABLE migration_backup_{name} "
                f"AS SELECT * FROM {name}"
            )

        for name, _ in reversed(children):
            connection.execute(f"DROP TABLE {name}")

        connection.execute(
            _SCHEMA[0].replace(
                "CREATE TABLE service_jobs (",
                "CREATE TABLE service_jobs_v6 (",
                1,
            )
        )
        columns = (
            "sequence, job_id, tenant_id, owner_id, "
            "idempotency_digest, request_fingerprint, "
            "execution_profile_id, conversation_id, state, version, attempt, "
            "created_at, updated_at, expires_at, cancel_requested"
        )
        connection.execute(
            f"INSERT INTO service_jobs_v6 ({columns}) "
            f"SELECT {columns} FROM service_jobs"
        )
        connection.execute("DROP TABLE service_jobs")
        connection.execute(
            "ALTER TABLE service_jobs_v6 RENAME TO service_jobs"
        )
        for statement in _SCHEMA[1:]:
            connection.execute(statement)

        for name, statement in children:
            connection.execute(statement)
            connection.execute(
                f"INSERT INTO {name} "
                f"SELECT * FROM migration_backup_{name}"
            )
            connection.execute(f"DROP TABLE migration_backup_{name}")

        connection.execute(_REQUEST_INDEX)
        connection.execute(_PREPARATION_SCHEMA)

        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise UnsupportedJobSchemaError(
                "The migrated database has invalid references."
            )

    def _initialize(self) -> None:
        with self._transaction(write=True) as connection:
            application_id = connection.execute(
                "PRAGMA application_id"
            ).fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]

            if application_id == 0 and version == 0:
                existing = connection.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE name NOT LIKE 'sqlite_%'
                    LIMIT 1
                    """
                ).fetchone()
                if existing is not None:
                    raise UnsupportedJobSchemaError(
                        "Refusing to initialize an unrelated database."
                    )

                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute(_LEASE_SCHEMA)
                connection.execute(_HOLD_SCHEMA)
                connection.execute(_REQUEST_SCHEMA)
                connection.execute(_REQUEST_INDEX)
                connection.execute(_RESULT_SCHEMA)
                connection.execute(_PREPARATION_SCHEMA)
                connection.execute(
                    f"PRAGMA application_id = {APPLICATION_ID}"
                )
                connection.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}"
                )

            else:
                if application_id != APPLICATION_ID:
                    raise UnsupportedJobSchemaError(
                        "The database has an unsupported application identity."
                    )
                if version not in (1, 2, 3, 4, 5, SCHEMA_VERSION):
                    raise UnsupportedJobSchemaError(
                        "The database schema version is unsupported."
                    )

                if version in (1, 2):
                    # Preserve child rows before rebuilding their parent.
                    # Every statement remains inside the outer transaction.
                    leases = []
                    if version == 2:
                        leases = [
                            tuple(row)
                            for row in connection.execute(
                                """
                                SELECT job_id, worker_id, token,
                                       attempt, lease_expires_at
                                FROM service_job_leases
                                """
                            ).fetchall()
                        ]
                        connection.execute("DROP TABLE service_job_leases")

                    connection.execute(
                        _SCHEMA[0].replace(
                            "CREATE TABLE service_jobs (",
                            "CREATE TABLE service_jobs_next (",
                            1,
                        )
                    )

                    attempt_expression = "0" if version == 1 else "attempt"
                    connection.execute(
                        f"""
                        INSERT INTO service_jobs_next (
                            sequence, job_id, tenant_id, owner_id,
                            idempotency_digest, request_fingerprint,
                            execution_profile_id, conversation_id,
                            state, version, attempt,
                            created_at, updated_at, expires_at,
                            cancel_requested
                        )
                        SELECT
                            sequence, job_id, tenant_id, owner_id,
                            idempotency_digest, request_fingerprint,
                            execution_profile_id, conversation_id,
                            state, version, {attempt_expression},
                            created_at, updated_at, expires_at,
                            cancel_requested
                        FROM service_jobs
                        """
                    )
                    connection.execute("DROP TABLE service_jobs")
                    connection.execute(
                        "ALTER TABLE service_jobs_next RENAME TO service_jobs"
                    )

                    for statement in _SCHEMA[1:]:
                        connection.execute(statement)

                    connection.execute(_LEASE_SCHEMA)
                    if leases:
                        connection.executemany(
                            """
                            INSERT INTO service_job_leases (
                                job_id, worker_id, token,
                                attempt, lease_expires_at
                            )
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            leases,
                        )

                    connection.execute(_HOLD_SCHEMA)
                    connection.execute("PRAGMA user_version = 3")
                    version = 3

                if version == 3:
                    connection.execute(_REQUEST_SCHEMA)
                    connection.execute(_REQUEST_INDEX)
                    connection.execute("PRAGMA user_version = 4")
                    version = 4

                if version == 4:
                    connection.execute(_RESULT_SCHEMA)
                    connection.execute("PRAGMA user_version = 5")
                    version = 5

                if version == 5:
                    self._migrate_5_to_6(connection)
                    connection.execute(
                        f"PRAGMA user_version = {SCHEMA_VERSION}"
                    )

            connection.execute(
                """
                SELECT sequence, job_id, tenant_id, owner_id,
                       idempotency_digest, request_fingerprint,
                       execution_profile_id, conversation_id,
                       state, version, attempt, created_at, updated_at,
                       expires_at, cancel_requested
                FROM service_jobs
                LIMIT 0
                """
            )
            connection.execute(
                """
                SELECT job_id, worker_id, token, attempt, lease_expires_at
                FROM service_job_leases
                LIMIT 0
                """
            )
            connection.execute(
                """
                SELECT job_id, worker_id, attempt, token,
                       reserved_bytes, issued_at
                FROM service_inference_holds
                LIMIT 0
                """
            )
            connection.execute(
                """
                SELECT job_id, prompt, expires_at
                FROM service_request_content
                LIMIT 0
                """
            )
            connection.execute(
                """
                SELECT job_id, worker_id, attempt, permit_digest,
                       payload_json, payload_digest, captured_at,
                       persistence_state
                FROM service_job_results
                LIMIT 0
                """
            )
            connection.execute(
                """
                SELECT job_id, payload_json, payload_digest, review_expires_at
                FROM service_preparations
                LIMIT 0
                """
            )

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise ValueError("clock must return a timezone-aware datetime.")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime.")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _status(row: sqlite3.Row) -> JobStatus:
        return JobStatus(
            job_id=row["job_id"],
            principal=Principal(row["tenant_id"], row["owner_id"]),
            state=JobState(row["state"]),
            version=row["version"],
            created_at=_parse_timestamp(row["created_at"]),
            updated_at=_parse_timestamp(row["updated_at"]),
            expires_at=_parse_timestamp(row["expires_at"]),
            conversation_id=row["conversation_id"],
            cancel_requested=bool(row["cancel_requested"]),
            execution_profile_id=row["execution_profile_id"],
        )

    @staticmethod
    def _owned_row(
        connection: sqlite3.Connection,
        principal: Principal,
        job_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT *
            FROM service_jobs
            WHERE job_id = ? AND tenant_id = ? AND owner_id = ?
            """,
            (job_id, principal.tenant_id, principal.owner_id),
        ).fetchone()

        if row is None:
            raise ServiceError(ErrorCode.NOT_FOUND)
        return row
    
    @staticmethod
    def _require_request(
        connection: sqlite3.Connection,
        job_id: str,
        now: datetime,
    ) -> sqlite3.Row:
        """Read unexpired content after the caller establishes authority."""
        row = connection.execute(
            """
            SELECT job_id, prompt, expires_at
            FROM service_request_content
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

        if row is None or row["expires_at"] <= _timestamp(now):
            raise ServiceError(ErrorCode.CONTENT_EXPIRED)
        return row
    
    @staticmethod
    def _expire(
        connection: sqlite3.Connection,
        now: datetime,
    ) -> int:
        stamp = _timestamp(now)
        cursor = connection.execute(
            """
            UPDATE service_jobs
            SET state = 'expired',
                version = version + 1,
                updated_at = MAX(updated_at, ?)
            WHERE
                (state = 'queued' AND expires_at <= ?)
                OR (
                    state = 'awaiting_review'
                    AND (
                        expires_at <= ?
                        OR NOT EXISTS (
                            SELECT 1
                            FROM service_preparations AS preparation
                            JOIN service_request_content AS content
                              ON content.job_id = preparation.job_id
                            WHERE preparation.job_id = service_jobs.job_id
                              AND preparation.review_expires_at > ?
                              AND content.expires_at > ?
                        )
                    )
                )
            """,
            (stamp, stamp, stamp, stamp, stamp),
        )
        return cursor.rowcount

    def admit(
        self,
        principal: Principal,
        submission: JobSubmission,
        *,
        execution_profile_id: str = "default",
        store_request: bool = False,
        authorize_new: Callable[[], None] | None = None,
    ) -> Admission:
        """Admit metadata atomically or return the existing matching job.

        The caller must establish identity and select the execution profile.
        Conversation authorization must be checked by the future API adapter.

        A matching retry returns the original job, including terminal jobs.
        A changed payload or profile under the same key raises CONFLICT.

        authorize_new is an optional trusted local authorization callback,
        invoked inside the admission transaction only for a new job. Failure
        rolls back admission. It must not call inference or the network.
        """
        if type(store_request) is not bool:
            raise ValueError("store_request must be a boolean.")
        if authorize_new is not None and not callable(authorize_new):
            raise ValueError("authorize_new must be callable or None.")
        _require_principal(principal)
        if not isinstance(submission, JobSubmission):
            raise ValueError("submission must be a JobSubmission.")
        if len(submission.prompt) > self.settings.max_prompt_characters:
            raise ServiceError(ErrorCode.INVALID_REQUEST)
        if (
            not isinstance(execution_profile_id, str)
            or _IDENTIFIER.fullmatch(execution_profile_id) is None
        ):
            raise ValueError("execution_profile_id must be a valid identifier.")

        key_digest = hashlib.sha256(
            submission.idempotency_key.encode("utf-8")
        ).hexdigest()
        fingerprint = submission.fingerprint()

        with self._transaction(write=True) as connection:
            now = self._now()
            self._expire(connection, now)

            existing = connection.execute(
                """
                SELECT *
                FROM service_jobs
                WHERE tenant_id = ? AND owner_id = ?
                  AND idempotency_digest = ?
                """,
                (principal.tenant_id, principal.owner_id, key_digest),
            ).fetchone()

            if existing is not None:
                if (
                    existing["request_fingerprint"] != fingerprint
                    or existing["execution_profile_id"] != execution_profile_id
                ):
                    raise ServiceError(ErrorCode.CONFLICT)
                if store_request:
                    self._require_request(
                        connection,
                        existing["job_id"],
                        now,
                    )
                return Admission(self._status(existing), created=False)

            # Trusted local authorization runs only for new admissions. It
            # must not perform network/inference calls. Lock order is job
            # database then conversation database, matching finalization.
            if authorize_new is not None:
                authorize_new()

            # Preparation retains its queue reservation. Recovering an
            # expired preparation lease therefore cannot overflow the queue.
            waiting = connection.execute(
                """
                SELECT COUNT(*)
                FROM service_jobs
                WHERE state IN ('queued', 'preparing', 'awaiting_review', 'waiting_inference')
                """
            ).fetchone()[0]

            owner_waiting = connection.execute(
                """
                SELECT COUNT(*)
                FROM service_jobs
                WHERE state IN ('queued', 'preparing', 'awaiting_review', 'waiting_inference')
                  AND tenant_id = ? AND owner_id = ?
                """,
                (principal.tenant_id, principal.owner_id),
            ).fetchone()[0]

            if (
                waiting >= self.settings.max_waiting_jobs
                or owner_waiting >= self.settings.owner_max_waiting
            ):
                raise ServiceError(
                    ErrorCode.QUEUE_FULL,
                    retry_after_seconds=1,
                )

            job_id = uuid.uuid4().hex
            stamp = _timestamp(now)
            expires = _timestamp(
                now + timedelta(seconds=self.settings.job_deadline_seconds)
            )

            connection.execute(
                """
                INSERT INTO service_jobs (
                    job_id, tenant_id, owner_id,
                    idempotency_digest, request_fingerprint,
                    execution_profile_id, conversation_id,
                    state, version, created_at, updated_at,
                    expires_at, cancel_requested
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 1, ?, ?, ?, 0)
                """,
                (
                    job_id,
                    principal.tenant_id,
                    principal.owner_id,
                    key_digest,
                    fingerprint,
                    execution_profile_id,
                    submission.conversation_id,
                    stamp,
                    stamp,
                    expires,
                ),
            )
            if store_request:
                content_expires = now + timedelta(
                    seconds=self.settings.content_retention_seconds
                )
                connection.execute(
                    """
                    INSERT INTO service_request_content (
                        job_id, prompt, expires_at
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        job_id,
                        submission.prompt,
                        _timestamp(content_expires),
                    ),
                )
                
            row = self._owned_row(connection, principal, job_id)
            return Admission(self._status(row), created=True)

    def get(self, principal: Principal, job_id: str) -> JobStatus:
        """Read owned metadata; this operation does not advance job state."""
        _require_principal(principal)
        _require_job_id(job_id)

        with self._transaction(write=False) as connection:
            return self._status(self._owned_row(connection, principal, job_id))

    def cancel(
        self,
        principal: Principal,
        job_id: str,
        *,
        expected_version: int,
    ) -> JobStatus:
        """Cancel preparation or request cancellation of running inference.

        A running cancellation never releases a permit or worker lease.
        Finalization has already passed the cancellation boundary.
        """
        _require_principal(principal)
        _require_job_id(job_id)
        _require_version(expected_version)

        with self._transaction(write=True) as connection:
            status = self._status(
                self._owned_row(connection, principal, job_id)
            )

            if status.state.terminal:
                return status

            if status.state is JobState.FINALIZING:
                raise ServiceError(ErrorCode.CONFLICT)

            if (
                status.state is JobState.RUNNING
                and status.cancel_requested
            ):
                return status

            if status.version != expected_version:
                raise ServiceError(ErrorCode.CONFLICT)

            now = self._now()
            updated = _timestamp(max(now, status.updated_at))

            if status.state is JobState.RUNNING:
                connection.execute(
                    """
                    UPDATE service_jobs
                    SET cancel_requested = 1,
                        version = version + 1,
                        updated_at = ?
                    WHERE job_id = ? AND version = ?
                    """,
                    (updated, job_id, expected_version),
                )
                return self._status(
                    self._owned_row(connection, principal, job_id)
                )

            target = (
                JobState.EXPIRED
                if now >= status.expires_at
                else JobState.CANCELLED
            )
            validate_transition(status.state, target)

            cursor = connection.execute(
                """
                UPDATE service_jobs
                SET state = ?,
                    version = version + 1,
                    updated_at = ?,
                    cancel_requested = ?
                WHERE job_id = ? AND tenant_id = ? AND owner_id = ?
                  AND version = ?
                """,
                (
                    target.value,
                    updated,
                    int(target is JobState.CANCELLED),
                    job_id,
                    principal.tenant_id,
                    principal.owner_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise ServiceError(ErrorCode.CONFLICT)

            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (job_id,),
            )
            return self._status(
                self._owned_row(connection, principal, job_id)
            )

    def expire_queued(self) -> int:
        """Expire queued metadata atomically; return the number changed."""
        with self._transaction(write=True) as connection:
            return self._expire(connection, self._now())
