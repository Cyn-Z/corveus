"""Durable result capture and recoverable finalization.

All result content is untrusted data.

Results share their request's temporary retention, including Incognito.
Conversation saves use durable job receipts to prevent duplicate exchanges.

No inference, HTTP endpoints, background timers, or deployment changes
are introduced by this module.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

from memory_manager import MemoryManager
from memory_store import ConversationNotFoundError
from service_contracts import ErrorCode, JobState, JobStatus, Principal, ServiceError
from service_inference import (
    InferenceGate,
    InferencePermit,
    _require_stop_confirmation,
)
from service_job_store import (
    _require_job_id,
    _require_principal,
    _timestamp,
)
from service_scheduler import _identifier, _TOKEN
from token_usage import TokenUsage


MAX_RESULT_BYTES = 4_000_000


@dataclass(frozen=True)
class ResultSnapshot:
    answer: str = field(repr=False)
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a nonempty string.")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be TokenUsage or None.")

        try:
            encoded = self.serialize().encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("Result content must contain valid Unicode.") from None

        if len(encoded) > MAX_RESULT_BYTES:
            raise ValueError("Result content exceeds the byte limit.")

    def serialize(self) -> str:
        return json.dumps(
            {
                "answer": self.answer,
                "usage": None if self.usage is None else asdict(self.usage),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


@dataclass(frozen=True)
class CompletedResult:
    status: JobStatus
    snapshot: ResultSnapshot = field(repr=False)
    persistence_state: str


class JobResults:
    """Capture inference results and finalize them without repeating inference.

    memory_for must be a trusted, local resolver from Principal to the
    correct configured MemoryManager. Never resolve a scope from client input.

    Finalization locks the job database before accessing conversation storage.
    The resolver must not perform network or inference operations.
    """

    def __init__(
        self,
        gate: InferenceGate,
        *,
        memory_for: Callable[[Principal], MemoryManager | None] | None = None,
    ) -> None:
        if not isinstance(gate, InferenceGate):
            raise ValueError("gate must be an InferenceGate.")
        if memory_for is not None and not callable(memory_for):
            raise ValueError("memory_for must be callable or None.")

        self.gate = gate
        self.scheduler = gate.scheduler
        self.store = gate.store
        self.memory_for = memory_for

    @staticmethod
    def _permit_digest(permit: InferencePermit) -> str:
        if not isinstance(permit, InferencePermit):
            raise ValueError("permit must be an InferencePermit.")
        if not isinstance(permit.status, JobStatus):
            raise ValueError("permit.status must be a JobStatus.")

        _require_job_id(permit.status.job_id)
        _identifier(permit.worker_id, "worker_id")

        if type(permit.attempt) is not int or permit.attempt < 1:
            raise ValueError("permit.attempt must be positive.")
        if (
            not isinstance(permit.token, str)
            or _TOKEN.fullmatch(permit.token) is None
        ):
            raise ServiceError(ErrorCode.CONFLICT)

        return hashlib.sha256(permit.token.encode("ascii")).hexdigest()

    def capture(
        self,
        permit: InferencePermit,
        snapshot: ResultSnapshot,
        *,
        inference_stopped: bool,
    ) -> JobStatus:
        """Store an answer and release inference capacity atomically.

        The adapter must verify the upstream request has stopped and cannot
        execute later before setting inference_stopped=True.

        Matching retries are accepted without changing the result or expiry.
        Cancellation stores no answer. Expired attempts cannot publish.
        """
        _require_stop_confirmation(inference_stopped)
        if not isinstance(snapshot, ResultSnapshot):
            raise ValueError("snapshot must be a ResultSnapshot.")

        permit_digest = self._permit_digest(permit)
        payload = snapshot.serialize()
        payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        job_id = permit.status.job_id

        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self.store._require_request(connection, job_id, now)

            existing = connection.execute(
                """
                SELECT worker_id, attempt, permit_digest, payload_digest
                FROM service_job_results
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

            if existing is not None:
                if (
                    existing["worker_id"] != permit.worker_id
                    or existing["attempt"] != permit.attempt
                    or not secrets.compare_digest(
                        existing["permit_digest"], permit_digest
                    )
                    or existing["payload_digest"] != payload_digest
                ):
                    raise ServiceError(ErrorCode.CONFLICT)

                status = self.store._status(
                    self.scheduler._job(connection, job_id)
                )
                if status.state not in (
                    JobState.FINALIZING,
                    JobState.COMPLETED,
                ):
                    raise ServiceError(ErrorCode.CONFLICT)
                return status

            status = self.gate._finish(connection, permit, now)

            if status.state is not JobState.FINALIZING:
                return status

            connection.execute(
                """
                INSERT INTO service_job_results (
                    job_id, worker_id, attempt, permit_digest,
                    payload_json, payload_digest, captured_at,
                    persistence_state
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    job_id,
                    permit.worker_id,
                    permit.attempt,
                    permit_digest,
                    payload,
                    payload_digest,
                    _timestamp(now),
                ),
            )

            # The answer is durable. Finalization now belongs to the
            # coordinator and no longer requires the generation worker.
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (job_id,),
            )
            return status

    def _read(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        now: datetime,
    ) -> tuple[sqlite3.Row, ResultSnapshot]:
        self.store._require_request(connection, job_id, now)
        row = connection.execute(
            "SELECT * FROM service_job_results WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise ServiceError(ErrorCode.CONFLICT)

        try:
            actual_digest = hashlib.sha256(
                row["payload_json"].encode("utf-8")
            ).hexdigest()
            if actual_digest != row["payload_digest"]:
                raise ValueError("Result checksum mismatch.")

            payload = json.loads(row["payload_json"])
            if type(payload) is not dict or set(payload) != {"answer", "usage"}:
                raise ValueError("Invalid result structure.")

            usage = (
                None
                if payload["usage"] is None
                else TokenUsage(**payload["usage"])
            )
            snapshot = ResultSnapshot(payload["answer"], usage)
        except (ValueError, TypeError, UnicodeError):
            raise ServiceError(ErrorCode.INTERNAL_ERROR) from None

        return row, snapshot

    def _completed(
        self,
        connection: sqlite3.Connection,
        status: JobStatus,
        now: datetime,
    ) -> CompletedResult:
        row, snapshot = self._read(connection, status.job_id, now)
        if status.state is not JobState.COMPLETED:
            raise ServiceError(ErrorCode.CONFLICT)
        if row["persistence_state"] is None:
            raise ServiceError(ErrorCode.INTERNAL_ERROR)

        return CompletedResult(
            status=status,
            snapshot=snapshot,
            persistence_state=row["persistence_state"],
        )

    def get(self, principal: Principal, job_id: str) -> CompletedResult:
        """Read an owned, completed, unexpired result."""
        _require_principal(principal)
        _require_job_id(job_id)

        with self.store._transaction(write=False) as connection:
            status = self.store._status(
                self.store._owned_row(connection, principal, job_id)
            )
            return self._completed(connection, status, self.store._now())

    def _mark_completed(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        persistence_state: str,
        now: datetime,
    ) -> JobStatus:
        connection.execute(
            """
            UPDATE service_job_results
            SET persistence_state = ?
            WHERE job_id = ?
            """,
            (persistence_state, job_id),
        )
        return self.scheduler._change(
            connection,
            job_id,
            JobState.COMPLETED,
            now,
        )

    def finalize(self, principal: Principal, job_id: str) -> CompletedResult:
        """Save the conversation if requested, then complete the result.

        A transient storage failure leaves the result finalizing.
        Retrying the conversation write uses the same job receipt.

        The two databases do not share a transaction. The durable receipt
        makes recovery safe after conversation commit but before job commit.
        """
        _require_principal(principal)
        _require_job_id(job_id)

        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            status = self.store._status(
                self.store._owned_row(connection, principal, job_id)
            )
            if status.state is JobState.COMPLETED:
                return self._completed(connection, status, now)
            if status.state is not JobState.FINALIZING:
                raise ServiceError(ErrorCode.CONFLICT)

            content = self.store._require_request(connection, job_id, now)
            _, snapshot = self._read(connection, job_id, now)

            hold = connection.execute(
                """
                SELECT 1 FROM service_inference_holds WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if hold is not None:
                raise ServiceError(ErrorCode.INTERNAL_ERROR)

            persistence_state = "not_requested"
            if status.conversation_id is not None:
                manager = (
                    None if self.memory_for is None
                    else self.memory_for(principal)
                )
                if manager is not None and not isinstance(manager, MemoryManager):
                    raise ValueError(
                        "memory_for must return MemoryManager or None."
                    )

                if manager is None or not manager.enabled:
                    persistence_state = "memory_unavailable"
                else:
                    try:
                        pair = manager.record_completed_exchange(
                            status.conversation_id,
                            user_content=content["prompt"],
                            assistant_content=snapshot.answer,
                            usage=snapshot.usage,
                            job_id=job_id,
                        )
                        persistence_state = (
                            "saved" if pair is not None
                            else "memory_unavailable"
                        )
                    except ConversationNotFoundError:
                        persistence_state = "conversation_deleted"

            changed = self._mark_completed(
                connection,
                job_id,
                persistence_state,
                now,
            )
            return self._completed(connection, changed, now)

    def recover(self, *, limit: int = 100) -> int:
        """Finalize a bounded snapshot of pending results without inference.

        Stops and propagates storage errors. Unfinished results remain
        available for another recovery attempt until their content expires.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer from 1 to 1000.")

        with self.store._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT jobs.job_id, jobs.tenant_id, jobs.owner_id
                FROM service_jobs AS jobs
                JOIN service_job_results AS results
                  ON results.job_id = jobs.job_id
                JOIN service_request_content AS content
                  ON content.job_id = jobs.job_id
                WHERE jobs.state = 'finalizing'
                  AND content.expires_at > ?
                ORDER BY jobs.sequence
                LIMIT ?
                """,
                (_timestamp(self.store._now()), limit),
            ).fetchall()

        processed = 0
        for row in rows:
            self.finalize(
                Principal(row["tenant_id"], row["owner_id"]),
                row["job_id"],
            )
            processed += 1
        return processed