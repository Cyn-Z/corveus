"""Inference admission contracts backed by the local job registry.

This module does not call the model or measure physical GPU memory.

All profiles share one global inference resource budget in this version.
Reservations must be calculated by trusted server-side configuration or
estimation code, never supplied directly by an external client.

A permit reserves capacity. It does not enforce exactly-once execution at
the inference backend. The future adapter must prevent duplicate dispatch.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from service_contracts import ErrorCode, JobState, JobStatus, ServiceError
from service_job_store import _require_job_id, _timestamp
from service_scheduler import (
    JobScheduler,
    WorkerLease,
    _identifier,
    _TOKEN,
)
from service_job_store import _parse_timestamp


_MAX_SQLITE_INTEGER = (1 << 63) - 1


@dataclass(frozen=True)
class InferencePermit:
    """Internal credential; never expose through public job endpoints."""

    status: JobStatus
    worker_id: str
    attempt: int
    token: str = field(repr=False)
    reserved_bytes: int


def _require_stop_confirmation(inference_stopped: bool) -> None:
    if inference_stopped is not True:
        raise ValueError(
            "A trusted adapter must confirm inference has stopped."
        )


class InferenceGate:
    """Coordinate inference reservations sharing one local job database."""

    def __init__(self, scheduler: JobScheduler) -> None:
        if not isinstance(scheduler, JobScheduler):
            raise ValueError("scheduler must be a JobScheduler.")
        self.scheduler = scheduler
        self.store = scheduler.store

    def _permit(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> InferencePermit:
        row = connection.execute(
            """
            SELECT *
            FROM service_inference_holds
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ServiceError(ErrorCode.CONFLICT)

        return InferencePermit(
            status=self.store._status(
                self.scheduler._job(connection, job_id)
            ),
            worker_id=row["worker_id"],
            attempt=row["attempt"],
            token=row["token"],
            reserved_bytes=row["reserved_bytes"],
        )

    def _hold(
        self,
        connection: sqlite3.Connection,
        permit: InferencePermit,
    ) -> sqlite3.Row:
        if not isinstance(permit, InferencePermit):
            raise ValueError("permit must be an InferencePermit.")
        if not isinstance(permit.status, JobStatus):
            raise ValueError("permit.status must be a JobStatus.")

        _require_job_id(permit.status.job_id)
        _identifier(permit.worker_id, "worker_id")

        if type(permit.attempt) is not int or permit.attempt < 1:
            raise ValueError("permit.attempt must be a positive integer.")
        if (
            not isinstance(permit.token, str)
            or _TOKEN.fullmatch(permit.token) is None
        ):
            raise ServiceError(ErrorCode.CONFLICT)

        row = connection.execute(
            """
            SELECT *
            FROM service_inference_holds
            WHERE job_id = ?
            """,
            (permit.status.job_id,),
        ).fetchone()

        if row is None or (
            row["worker_id"] != permit.worker_id
            or row["attempt"] != permit.attempt
            or not secrets.compare_digest(row["token"], permit.token)
        ):
            raise ServiceError(ErrorCode.CONFLICT)

        return row

    def start(
        self,
        claim: WorkerLease,
        *,
        reserved_bytes: int,
    ) -> InferencePermit | None:
        """Reserve capacity and change waiting_inference to running.

        None means shared capacity is currently unavailable.

        A duplicate start conflicts, rather than granting a second permit.
        Workers must not dispatch inference without a successful permit.
        """
        if (
            type(reserved_bytes) is not int
            or not 1 <= reserved_bytes <= _MAX_SQLITE_INTEGER
        ):
            raise ValueError(
                "reserved_bytes must be a positive SQLite-sized integer."
            )

        budget = self.store.settings.inference_memory_budget_bytes
        if budget is not None and reserved_bytes > budget:
            raise ValueError(
                "The request reservation exceeds the configured memory budget."
            )

        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            row = self.scheduler._checked(connection, claim, now)

            if JobState(row["state"]) is not JobState.WAITING_INFERENCE:
                raise ServiceError(ErrorCode.CONFLICT)

            reservations = connection.execute(
                "SELECT reserved_bytes FROM service_inference_holds"
            ).fetchall()

            if len(reservations) >= self.store.settings.max_inference_requests:
                return None

            used_bytes = sum(row["reserved_bytes"] for row in reservations)
            if budget is not None and used_bytes + reserved_bytes > budget:
                return None

            self.scheduler._change(
                connection,
                claim.status.job_id,
                JobState.RUNNING,
                now,
            )
            connection.execute(
                """
                INSERT INTO service_inference_holds (
                    job_id, worker_id, attempt, token,
                    reserved_bytes, issued_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.status.job_id,
                    claim.worker_id,
                    claim.attempt,
                    secrets.token_hex(32),
                    reserved_bytes,
                    _timestamp(now),
                ),
            )
            return self._permit(connection, claim.status.job_id)

    def finish(
        self,
        permit: InferencePermit,
        *,
        inference_stopped: bool,
        failed: bool = False,
    ) -> JobStatus:
        """Record confirmed termination without storing an answer.

        Future successful worker requests must use JobResults.capture.
        This method remains available for confirmed inference failures
        and existing metadata-only callers.
        """
        _require_stop_confirmation(inference_stopped)
        if type(failed) is not bool:
            raise ValueError("failed must be a boolean.")

        with self.store._transaction(write=True) as connection:
            return self._finish(
                connection,
                permit,
                self.store._now(),
                failed=failed,
            )

    def _finish(
        self,
        connection: sqlite3.Connection,
        permit: InferencePermit,
        now: datetime,
        *,
        failed: bool = False,
    ) -> JobStatus:
        """Internal transaction helper; caller verifies upstream shutdown."""
        self._hold(connection, permit)
        job_id = permit.status.job_id
        status = self.store._status(
            self.scheduler._job(connection, job_id)
        )

        lease = connection.execute(
            """
            SELECT worker_id, attempt, lease_expires_at
            FROM service_job_leases
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

        if (
            status.state is not JobState.RUNNING
            or lease is None
            or lease["worker_id"] != permit.worker_id
            or lease["attempt"] != permit.attempt
            or now >= _parse_timestamp(lease["lease_expires_at"])
            or now >= status.expires_at
        ):
            raise ServiceError(ErrorCode.CONFLICT)

        if status.cancel_requested:
            target = JobState.CANCELLED
        elif failed:
            target = JobState.FAILED
        else:
            target = JobState.FINALIZING

        changed = self.scheduler._change(
            connection,
            job_id,
            target,
            now,
            inference_stopped=True,
        )
        connection.execute(
            "DELETE FROM service_inference_holds WHERE job_id = ?",
            (job_id,),
        )

        if target is not JobState.FINALIZING:
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (job_id,),
            )

        return changed

    def complete(self, claim: WorkerLease) -> JobStatus:
        """Complete metadata after successful finalization.

        Future worker wiring must call this only after required result and
        conversation persistence succeeds. This method stores no answer.
        """
        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self.scheduler._checked(
                connection,
                claim,
                now,
                allowed_states=frozenset({JobState.FINALIZING}),
            )

            existing = connection.execute(
                """
                SELECT 1
                FROM service_inference_holds
                WHERE job_id = ?
                """,
                (claim.status.job_id,),
            ).fetchone()
            if existing is not None:
                raise ServiceError(ErrorCode.CONFLICT)

            changed = self.scheduler._change(
                connection,
                claim.status.job_id,
                JobState.COMPLETED,
                now,
            )
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (claim.status.job_id,),
            )
            return changed

    def held_permits(self) -> tuple[InferencePermit, ...]:
        """Inspect reservations for trusted internal reconciliation.

        Includes interrupted reservations. This exposes internal credentials
        and must never be used directly as a public HTTP response.
        """
        with self.store._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT holds.job_id
                FROM service_inference_holds AS holds
                JOIN service_jobs AS jobs ON jobs.job_id = holds.job_id
                ORDER BY jobs.sequence
                """
            ).fetchall()
            return tuple(
                self._permit(connection, row["job_id"])
                for row in rows
            )

    def interrupt(self, permit: InferencePermit) -> JobStatus:
        """Record uncertainty immediately, retaining the hold and worker slot.

        Only the holder can interrupt this attempt. This also works after
        lease expiry; it grants no permission to dispatch or retry inference.
        """
        with self.store._transaction(write=True) as connection:
            self._hold(connection, permit)
            status = self.store._status(
                self.scheduler._job(connection, permit.status.job_id)
            )
            if status.state is JobState.INTERRUPTED:
                return status
            if status.state is not JobState.RUNNING:
                raise ServiceError(ErrorCode.CONFLICT)
            return self.scheduler._change(
                connection, status.job_id, JobState.INTERRUPTED,
                self.store._now(),
            )

    def reconcile_stopped(
        self,
        permit: InferencePermit,
        *,
        inference_stopped: bool,
    ) -> JobStatus:
        """Release an interrupted reservation after verified shutdown.

        The job remains interrupted and is never automatically retried.

        This is an internal control operation, not a client cancellation
        shortcut. Absence from a process list or an HTTP timeout is not
        sufficient evidence that a queued upstream request cannot run later.
        """
        _require_stop_confirmation(inference_stopped)

        with self.store._transaction(write=True) as connection:
            self._hold(connection, permit)
            job_id = permit.status.job_id
            status = self.store._status(
                self.scheduler._job(connection, job_id)
            )
            if status.state is not JobState.INTERRUPTED:
                raise ServiceError(ErrorCode.CONFLICT)

            connection.execute(
                "DELETE FROM service_inference_holds WHERE job_id = ?",
                (job_id,),
            )
            connection.execute(
                "DELETE FROM service_job_leases WHERE job_id = ?",
                (job_id,),
            )
            return status
