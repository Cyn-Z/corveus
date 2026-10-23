"""Durable preparation and skill review.

Only trusted worker code may publish snapshots. Client-facing review accepts
skill names to remove, never replacement instructions or token counts.

Routing and token counting run outside SQLite write transactions.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from analyst_preparation import PreparationSnapshot, PromptPreparer
from service_contracts import ErrorCode, JobState, JobStatus, Principal, ServiceError
from service_job_store import (
    _parse_timestamp,
    _require_job_id,
    _require_principal,
    _require_version,
    _timestamp,
)
from service_requests import JobRequests
from service_scheduler import JobScheduler, WorkerLease


@dataclass(frozen=True)
class ReviewSkill:
    name: str
    score: float
    characters: int


@dataclass(frozen=True)
class ReviewView:
    status: JobStatus
    input_tokens: int
    input_budget: int
    skills: tuple[ReviewSkill, ...]
    expires_at: datetime


class JobPreparation:
    def __init__(self, scheduler: JobScheduler, preparer: PromptPreparer):
        if not isinstance(scheduler, JobScheduler):
            raise ValueError("scheduler must be a JobScheduler.")
        if not isinstance(preparer, PromptPreparer):
            raise ValueError("preparer must be a PromptPreparer.")
        self.scheduler = scheduler
        self.store = scheduler.store
        self.preparer = preparer
        self.requests = JobRequests(scheduler)

    def _decode(self, row):
        try:
            digest = hashlib.sha256(
                row["payload_json"].encode("utf-8")
            ).hexdigest()
            if digest != row["payload_digest"]:
                raise ValueError
            snapshot = PreparationSnapshot.from_json(row["payload_json"])
        except (ValueError, TypeError, UnicodeError):
            raise ServiceError(ErrorCode.INTERNAL_ERROR) from None

        if snapshot.profile != self.preparer.profile:
            raise ServiceError(ErrorCode.CONFLICT)
        return snapshot

    def for_worker(self, claim: WorkerLease) -> PreparationSnapshot | None:
        with self.store._transaction(write=False) as connection:
            now = self.store._now()
            self.scheduler._checked(connection, claim, now)
            content = self.store._require_request(
                connection, claim.status.job_id, now
            )
            row = connection.execute(
                "SELECT * FROM service_preparations WHERE job_id = ?",
                (claim.status.job_id,),
            ).fetchone()
            if row is None:
                return None

            snapshot = self._decode(row)
            if snapshot.prompt != content["prompt"]:
                raise ServiceError(ErrorCode.INTERNAL_ERROR)
            return snapshot

    def prepare(self, claim: WorkerLease) -> JobStatus:
        """Reuse durable preparation or compute it once outside the write lock."""
        snapshot = self.for_worker(claim)
        if snapshot is None:
            request = self.requests.for_worker(claim)
            snapshot = self.preparer.prepare(request.prompt)
        return self.publish(claim, snapshot)

    def publish(
        self,
        claim: WorkerLease,
        snapshot: PreparationSnapshot,
    ) -> JobStatus:
        """Publish a trusted worker snapshot; never accept this from clients."""
        if not isinstance(snapshot, PreparationSnapshot):
            raise ValueError("snapshot must be a PreparationSnapshot.")
        if snapshot.profile != self.preparer.profile:
            raise ServiceError(ErrorCode.CONFLICT)

        payload = snapshot.to_json()
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            self.scheduler._checked(
                connection,
                claim,
                now,
                allowed_states=frozenset({JobState.PREPARING}),
            )
            self.store._expire(connection, now)

            job = self.store._status(
                self.scheduler._job(connection, claim.status.job_id)
            )
            content = self.store._require_request(connection, job.job_id, now)
            if (
                snapshot.profile.profile_id != job.execution_profile_id
                or snapshot.prompt != content["prompt"]
            ):
                raise ServiceError(ErrorCode.CONFLICT)

            existing = connection.execute(
                "SELECT * FROM service_preparations WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()

            # A resumed job must not reroute or replace reviewed instructions.
            if existing is not None and existing["payload_digest"] != digest:
                raise ServiceError(ErrorCode.CONFLICT)

            review_expires = None
            target = JobState.WAITING_INFERENCE
            if snapshot.needs_review:
                previous_deadline = (
                    None if existing is None
                    else existing["review_expires_at"]
                )
                review_expires = min(
                    job.expires_at,
                    _parse_timestamp(content["expires_at"]),
                    (
                        _parse_timestamp(previous_deadline)
                        if previous_deadline is not None
                        else now + timedelta(
                            seconds=self.store.settings.review_ttl_seconds
                        )
                    ),
                )

                if review_expires <= now:
                    status = self.scheduler._change(
                        connection, job.job_id, JobState.EXPIRED, now
                    )
                    connection.execute(
                        "DELETE FROM service_job_leases WHERE job_id = ?",
                        (job.job_id,),
                    )
                    return status

                pending = connection.execute(
                    """
                    SELECT COUNT(*) FROM service_jobs
                    WHERE state = 'awaiting_review'
                    """
                ).fetchone()[0]
                if pending >= self.store.settings.max_pending_reviews:
                    raise ServiceError(
                        ErrorCode.QUEUE_FULL, retry_after_seconds=1
                    )
                target = JobState.AWAITING_REVIEW

            connection.execute(
                """
                INSERT INTO service_preparations (
                    job_id, payload_json, payload_digest, review_expires_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    payload_digest = excluded.payload_digest,
                    review_expires_at = excluded.review_expires_at
                """,
                (
                    job.job_id,
                    payload,
                    digest,
                    None if review_expires is None else _timestamp(review_expires),
                ),
            )
            status = self.scheduler._change(
                connection, job.job_id, target, now
            )
            if target is JobState.AWAITING_REVIEW:
                connection.execute(
                    "DELETE FROM service_job_leases WHERE job_id = ?",
                    (job.job_id,),
                )
            return status

    def _review_state(self, connection, principal, job_id, now):
        status = self.store._status(
            self.store._owned_row(connection, principal, job_id)
        )
        if status.state is not JobState.AWAITING_REVIEW:
            raise ServiceError(ErrorCode.CONFLICT)

        self.store._require_request(connection, job_id, now)
        row = connection.execute(
            "SELECT * FROM service_preparations WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise ServiceError(ErrorCode.CONFLICT)
        if (
            row["review_expires_at"] is None
            or now >= _parse_timestamp(row["review_expires_at"])
            or now >= status.expires_at
        ):
            raise ServiceError(ErrorCode.DEADLINE_EXCEEDED)

        return status, row, self._decode(row)

    def inspect(self, principal: Principal, job_id: str) -> ReviewView:
        """Expose review metadata without returning prompt or instructions."""
        _require_principal(principal)
        _require_job_id(job_id)
        with self.store._transaction(write=False) as connection:
            status, row, snapshot = self._review_state(
                connection, principal, job_id, self.store._now()
            )
            return ReviewView(
                status=status,
                input_tokens=snapshot.input_tokens,
                input_budget=snapshot.profile.input_token_budget,
                skills=tuple(
                    ReviewSkill(skill.name, skill.score, len(skill.content))
                    for skill in snapshot.skills
                ),
                expires_at=_parse_timestamp(row["review_expires_at"]),
            )

    def review(
        self,
        principal: Principal,
        job_id: str,
        remove_names,
        *,
        expected_version: int,
    ) -> JobStatus:
        """Apply removals and recount without holding a database write lock."""
        _require_principal(principal)
        _require_job_id(job_id)
        _require_version(expected_version)

        with self.store._transaction(write=False) as connection:
            status, _, snapshot = self._review_state(
                connection, principal, job_id, self.store._now()
            )
            if status.version != expected_version:
                raise ServiceError(ErrorCode.CONFLICT)

        updated = self.preparer.remove(snapshot, remove_names)
        payload = updated.to_json()
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self.store._transaction(write=True) as connection:
            now = self.store._now()
            current, _, _ = self._review_state(
                connection, principal, job_id, now
            )
            if current.version != expected_version:
                raise ServiceError(ErrorCode.CONFLICT)

            connection.execute(
                """
                UPDATE service_preparations
                SET payload_json = ?, payload_digest = ?,
                    review_expires_at = CASE
                        WHEN ? THEN review_expires_at ELSE NULL
                    END
                WHERE job_id = ?
                """,
                (payload, digest, int(updated.needs_review), job_id),
            )

            if not updated.needs_review:
                return self.scheduler._change(
                    connection, job_id, JobState.QUEUED, now
                )

            # Content changed while the state remains awaiting_review.
            connection.execute(
                """
                UPDATE service_jobs
                SET version = version + 1,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (_timestamp(max(now, current.updated_at)), job_id),
            )
            return self.store._status(
                self.scheduler._job(connection, job_id)
            )