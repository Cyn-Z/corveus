"""Temporary SQLite request storage for the future service boundary.

Saved and Incognito jobs use the same temporary request-retention policy.
Conversation persistence remains a separate operation.

Content is untrusted data. Reading it grants no permission to execute
instructions, tools, or inference.

This module does not store answers, start background cleanup, or change
the running application.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from service_contracts import JobStatus, JobSubmission, Principal
from service_job_store import (
    Admission,
    _parse_timestamp,
    _require_job_id,
    _require_principal,
    _timestamp,
)
from service_scheduler import JobScheduler, WorkerLease


@dataclass(frozen=True)
class StoredRequest:
    """Internal content record; prompt is excluded from its representation."""

    status: JobStatus
    prompt: str = field(repr=False)
    content_expires_at: datetime


class JobRequests:
    """Temporary request operations sharing the scheduler's database.

    Principal and execution profile must come from trusted service code.
    Conversation authorization must be checked before submission.

    Metadata-only JobStore.admit remains available for existing callers.
    Future API submission must use this class to persist request content.
    """

    def __init__(self, scheduler: JobScheduler) -> None:
        if not isinstance(scheduler, JobScheduler):
            raise ValueError("scheduler must be a JobScheduler.")
        self.scheduler = scheduler
        self.store = scheduler.store

    def submit(
        self,
        principal: Principal,
        submission: JobSubmission,
        *,
        execution_profile_id: str = "default",
        authorize_new: Callable[[], None] | None = None,
    ) -> Admission:
        """Atomically persist job metadata and the exact prompt.

        A matching retry returns the existing job without extending
        retention. Expired or missing content is never recreated under
        the original idempotency key.
        """
        return self.store.admit(
            principal,
            submission,
            execution_profile_id=execution_profile_id,
            store_request=True,
            authorize_new=authorize_new,
        )

    def _pack(
        self,
        connection: sqlite3.Connection,
        job: sqlite3.Row,
        now: datetime,
    ) -> StoredRequest:
        content = self.store._require_request(
            connection,
            job["job_id"],
            now,
        )
        return StoredRequest(
            status=self.store._status(job),
            prompt=content["prompt"],
            content_expires_at=_parse_timestamp(content["expires_at"]),
        )

    def get(
        self,
        principal: Principal,
        job_id: str,
    ) -> StoredRequest:
        """Read content for the matching tenant and owner.

        Missing or foreign jobs produce NOT_FOUND. Owned jobs without
        available content produce CONTENT_EXPIRED.
        """
        _require_principal(principal)
        _require_job_id(job_id)

        with self.store._transaction(write=False) as connection:
            job = self.store._owned_row(connection, principal, job_id)
            return self._pack(connection, job, self.store._now())

    def for_worker(self, claim: WorkerLease) -> StoredRequest:
        """Read a request using a live preparation attempt's credentials."""
        with self.store._transaction(write=False) as connection:
            now = self.store._now()
            self.scheduler._checked(connection, claim, now)
            job = self.scheduler._job(connection, claim.status.job_id)
            return self._pack(connection, job, now)

    def prune_expired(self) -> int:
        """Delete expired content without changing job or resource ownership.

        Expiry is enforced by reads even before cleanup runs.
        Future service wiring must schedule this operation regularly.
        """
        with self.store._transaction(write=True) as connection:
            cursor = connection.execute(
                """
                DELETE FROM service_request_content
                WHERE expires_at <= ?
                """,
                (_timestamp(self.store._now()),),
            )
            return cursor.rowcount
