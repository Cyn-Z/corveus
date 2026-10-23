"""Single-host preparation scheduler.

This module grants authority to prepare requests, never to run inference.

All mutations use the JobStore transaction boundary. Worker identity and
capabilities must come from trusted service configuration.

A lease token authorizes one attempt for one worker. Expired, cancelled,
released, or superseded attempts cannot mutate the job.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from service_contracts import (
    ErrorCode,
    JobState,
    JobStatus,
    ServiceError,
    validate_transition,
)
from service_job_store import (
    JobStore,
    _parse_timestamp,
    _require_job_id,
    _timestamp,
)


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}")
_TOKEN = re.compile(r"[0-9a-f]{64}")

_PREPARATION_STATES = frozenset({
    JobState.PREPARING,
    JobState.WAITING_INFERENCE,
})

_LEASED_STATES = _PREPARATION_STATES | frozenset({
    JobState.RUNNING,
    JobState.FINALIZING,
})


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a valid service identifier.")


@dataclass(frozen=True)
class WorkerLease:
    """Internal worker credential; never serialize this into public status."""

    status: JobStatus
    worker_id: str
    attempt: int
    token: str = field(repr=False)
    lease_expires_at: datetime


class JobScheduler:
    """Coordinate preparation workers sharing one local job database.

    Global and owner limits are enforced transactionally in SQLite.
    Worker IDs must uniquely identify worker processes.

    Conversation ordering is scoped by tenant, owner, and conversation ID.
    Independent jobs can proceed while an earlier conversation is blocked.
    """

    def __init__(self, store: JobStore) -> None:
        if not isinstance(store, JobStore):
            raise ValueError("store must be a JobStore.")
        self.store = store

    @staticmethod
    def _job(
        connection: sqlite3.Connection,
        job_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM service_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise ServiceError(ErrorCode.CONFLICT)
        return row

    def _pack(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> WorkerLease:
        lease = connection.execute(
            "SELECT * FROM service_job_leases WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if lease is None:
            raise ServiceError(ErrorCode.CONFLICT)

        return WorkerLease(
            status=self.store._status(self._job(connection, job_id)),
            worker_id=lease["worker_id"],
            attempt=lease["attempt"],
            token=lease["token"],
            lease_expires_at=_parse_timestamp(lease["lease_expires_at"]),
        )

    def _checked(
            self,
            connection: sqlite3.Connection,
            claim: WorkerLease,
            now: datetime,
            *,
            allowed_states: frozenset[JobState] = _PREPARATION_STATES,
    ) -> sqlite3.Row:
        if not isinstance(claim, WorkerLease):
            raise ValueError("claim must be a WorkerLease.")
        if not isinstance(claim.status, JobStatus):
            raise ValueError("claim.status must be a JobStatus.")

        _require_job_id(claim.status.job_id)
        _identifier(claim.worker_id, "worker_id")

        if type(claim.attempt) is not int or claim.attempt < 1:
            raise ValueError("claim.attempt must be a positive integer.")
        if (
            not isinstance(claim.token, str)
            or _TOKEN.fullmatch(claim.token) is None
        ):
            raise ServiceError(ErrorCode.CONFLICT)

        row = connection.execute(
            """
            SELECT leases.*,
                   jobs.state,
                   jobs.expires_at AS job_expires_at
            FROM service_job_leases AS leases
            JOIN service_jobs AS jobs ON jobs.job_id = leases.job_id
            WHERE leases.job_id = ?
            """,
            (claim.status.job_id,),
        ).fetchone()

        if row is None:
            raise ServiceError(ErrorCode.CONFLICT)

        if (
            row["worker_id"] != claim.worker_id
            or row["attempt"] != claim.attempt
            or not secrets.compare_digest(row["token"], claim.token)
            or JobState(row["state"]) not in allowed_states
            or now >= _parse_timestamp(row["lease_expires_at"])
            or now >= _parse_timestamp(row["job_expires_at"])
        ):
            raise ServiceError(ErrorCode.CONFLICT)

        return row

    def _change(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        target: JobState,
        now: datetime,
        *,
        inference_stopped: bool = False,
    ) -> JobStatus:
        row = self._job(connection, job_id)
        current = self.store._status(row)
        validate_transition(
            current.state,
            target,
            inference_stopped=inference_stopped,
        )

        connection.execute(
            """
            UPDATE service_jobs
            SET state = ?, version = version + 1, updated_at = ?
            WHERE job_id = ?
            """,
            (
                target.value,
                _timestamp(max(now, current.updated_at)),
                job_id,
            ),
        )
        return self.store._status(self._job(connection, job_id))

    def _recover(
        self,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> int:
        changed = self.store._expire(connection, now)

        expired = connection.execute(
            """
            SELECT jobs.job_id, jobs.expires_at
            FROM service_jobs AS jobs
            JOIN service_job_leases AS leases
              ON leases.job_id = jobs.job_id
            WHERE jobs.state IN ('preparing', 'waiting_inference')
              AND (leases.lease_expires_at <= ? OR jobs.expires_at <= ?)
            ORDER BY jobs.sequence
            """,
            (_timestamp(now), _timestamp(now)),
        ).fetchall()

        for row in expired:
            target = (
                JobState.EXPIRED
                if now >= _parse_timestamp(row["expires_at"])
                else JobState.QUEUED
            )
            self._change(connection, row["job_id"], target, now)
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (row["job_id"],),
            )
            changed += 1

        uncertain = connection.execute(
            """
            SELECT jobs.job_id, jobs.state,
                   EXISTS (
                       SELECT 1
                       FROM service_inference_holds AS holds
                       WHERE holds.job_id = jobs.job_id
                   ) AS has_hold
            FROM service_jobs AS jobs
            JOIN service_job_leases AS leases
              ON leases.job_id = jobs.job_id
            WHERE jobs.state IN ('running', 'finalizing')
              AND (leases.lease_expires_at <= ? OR jobs.expires_at <= ?)
            ORDER BY jobs.sequence
            """,
            (_timestamp(now), _timestamp(now)),
        ).fetchall()

        for row in uncertain:
            if row["state"] == JobState.RUNNING.value and not row["has_hold"]:
                # An invariant violation must never silently free capacity.
                raise ServiceError(ErrorCode.INTERNAL_ERROR)

            self._change(
                connection,
                row["job_id"],
                JobState.INTERRUPTED,
                now,
            )

            # Running work retains both its inference hold and worker slot.
            # Finalization normally has no hold: inference already stopped.
            if not row["has_hold"]:
                connection.execute(
                    "DELETE FROM service_job_leases WHERE job_id = ?",
                    (row["job_id"],),
                )
            changed += 1
            
        abandoned = connection.execute(
            """
            SELECT jobs.job_id
            FROM service_jobs AS jobs
            WHERE jobs.state = 'finalizing'
              AND NOT EXISTS (
                  SELECT 1 FROM service_job_leases AS leases
                  WHERE leases.job_id = jobs.job_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM service_inference_holds AS holds
                  WHERE holds.job_id = jobs.job_id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM service_job_results AS results
                  JOIN service_request_content AS content
                    ON content.job_id = results.job_id
                  WHERE results.job_id = jobs.job_id
                    AND content.expires_at > ?
              )
            ORDER BY jobs.sequence
            """,
            (_timestamp(now),),
        ).fetchall()

        for row in abandoned:
            self._change(
                connection,
                row["job_id"],
                JobState.FAILED,
                now,
            )
            changed += 1

        return changed

    def recover(self) -> int:
        """Recover preparation and record interrupted running work.

        Running inference retains its permit and worker slot until a trusted
        adapter confirms that execution has stopped and cannot start later.
        """
        with self.store._transaction(write=True) as connection:
            return self._recover(connection, self.store._now())
        
    def claim(
        self,
        worker_id: str,
        *,
        profile_ids: tuple[str, ...] = ("default",),
    ) -> WorkerLease | None:
        """Claim the oldest eligible job within shared concurrency limits.

        profile_ids must describe server-configured worker capabilities.
        None means no eligible capacity or work is currently available.
        """
        _identifier(worker_id, "worker_id")
        if (
            type(profile_ids) is not tuple
            or not profile_ids
            or len(profile_ids) > 64
        ):
            raise ValueError("profile_ids must contain 1 to 64 profile IDs.")
        for profile_id in profile_ids:
            _identifier(profile_id, "profile_id")
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("profile_ids must not contain duplicates.")

        settings = self.store.settings

        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self._recover(connection, now)

            active = connection.execute(
                "SELECT COUNT(*) FROM service_job_leases"
            ).fetchone()[0]
            worker_active = connection.execute(
                """
                SELECT COUNT(*)
                FROM service_job_leases
                WHERE worker_id = ?
                """,
                (worker_id,),
            ).fetchone()[0]

            if (
                active >= settings.max_active_jobs
                or worker_active >= settings.worker_concurrency
            ):
                return None

            placeholders = ",".join("?" for _ in profile_ids)
            row = connection.execute(
                f"""
                SELECT jobs.*
                FROM service_jobs AS jobs
                WHERE jobs.state = 'queued'
                  AND jobs.execution_profile_id IN ({placeholders})
                  AND (
                      SELECT COUNT(*)
                      FROM service_jobs AS active_jobs
                      JOIN service_job_leases AS active_leases
                        ON active_leases.job_id = active_jobs.job_id
                      WHERE active_jobs.tenant_id = jobs.tenant_id
                        AND active_jobs.owner_id = jobs.owner_id
                  ) < ?
                                    AND (
                      jobs.conversation_id IS NULL
                      OR NOT EXISTS (
                          SELECT 1
                          FROM service_jobs AS earlier
                          WHERE earlier.tenant_id = jobs.tenant_id
                            AND earlier.owner_id = jobs.owner_id
                            AND earlier.conversation_id = jobs.conversation_id
                            AND earlier.sequence < jobs.sequence
                            AND (
                                earlier.state IN (
                                    'queued', 'preparing', 'awaiting_review', 'waiting_inference',
                                    'running', 'finalizing'
                                )
                                OR EXISTS (
                                    SELECT 1
                                    FROM service_inference_holds AS holds
                                    WHERE holds.job_id = earlier.job_id
                                )
                            )
                      )
                  )
                ORDER BY jobs.sequence
                LIMIT 1
                """,
                (*profile_ids, settings.owner_max_active),
            ).fetchone()

            if row is None:
                return None

            job_id = row["job_id"]
            status = self._change(
                connection, job_id, JobState.PREPARING, now
            )

            connection.execute(
                """
                UPDATE service_jobs
                SET attempt = attempt + 1
                WHERE job_id = ?
                """,
                (job_id,),
            )
            attempt = self._job(connection, job_id)["attempt"]

            lease_expires = min(
                status.expires_at,
                now + timedelta(seconds=settings.worker_lease_seconds),
            )
            connection.execute(
                """
                INSERT INTO service_job_leases (
                    job_id, worker_id, token, attempt, lease_expires_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    worker_id,
                    secrets.token_hex(32),
                    attempt,
                    _timestamp(lease_expires),
                ),
            )
            return self._pack(connection, job_id)

    def heartbeat(self, claim: WorkerLease) -> WorkerLease:
        """Renew a live attempt without changing the public job version."""
        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            row = self._checked(
                connection,
                claim,
                now,
                allowed_states=_LEASED_STATES,
            )

            # Clock rollback must not shorten a previously granted lease.
            lease_expires = min(
                _parse_timestamp(row["job_expires_at"]),
                max(
                    _parse_timestamp(row["lease_expires_at"]),
                    now + timedelta(
                        seconds=self.store.settings.worker_lease_seconds
                    ),
                ),
            )
            connection.execute(
                """
                UPDATE service_job_leases
                SET lease_expires_at = ?
                WHERE job_id = ?
                """,
                (_timestamp(lease_expires), claim.status.job_id),
            )
            return self._pack(connection, claim.status.job_id)

    def mark_waiting(self, claim: WorkerLease) -> WorkerLease:
        """Record preparation completion; grant no inference permission."""
        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self._checked(connection, claim, now)
            self._change(
                connection,
                claim.status.job_id,
                JobState.WAITING_INFERENCE,
                now,
            )
            return self._pack(connection, claim.status.job_id)

    def fail(self, claim: WorkerLease) -> JobStatus:
        """Fail a live preparation attempt without granting inference."""
        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self._checked(connection, claim, now)
            status = self._change(
                connection, claim.status.job_id, JobState.FAILED, now
            )
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (claim.status.job_id,),
            )
            return status

    def release(self, claim: WorkerLease) -> JobStatus:
        """Return a live preparation attempt to the queue and revoke it."""
        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self._checked(connection, claim, now)
            status = self._change(
                connection,
                claim.status.job_id,
                JobState.QUEUED,
                now,
            )
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (claim.status.job_id,),
            )
            return status
