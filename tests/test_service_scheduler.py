"""Batch 2B.1 preparation leases; no inference is started by these tests."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from service_contracts import ErrorCode, JobState, JobSubmission, Principal, ServiceError
from service_job_store import APPLICATION_ID, SCHEMA_VERSION, JobStore
from service_scheduler import JobScheduler
from service_settings import ServiceSettings


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jobs.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.owner = Principal("local", "operator-a")
        self.store = JobStore(self.path, ServiceSettings(), clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)

    def submit(self, key, *, owner=None, conversation=None, profile="default"):
        submission = JobSubmission.from_payload(
            {"prompt": "Investigate", "conversation_id": conversation},
            idempotency_key=key,
        )
        return self.store.admit(owner or self.owner, submission,
                                execution_profile_id=profile).status

    def conflict(self, operation):
        with self.assertRaises(ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, ErrorCode.CONFLICT)

    def test_only_one_worker_wins_a_claim(self):
        original = self.submit("one")
        with ThreadPoolExecutor(max_workers=8) as executor:
            claims = list(executor.map(
                lambda index: self.scheduler.claim(f"worker-{index}"), range(12)
            ))
        winners = [claim for claim in claims if claim is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].status.job_id, original.job_id)
        self.assertEqual(winners[0].status.state, JobState.PREPARING)
        self.assertEqual(winners[0].attempt, 1)
        self.assertNotIn(winners[0].token, repr(winners[0]))

    def test_heartbeat_renews_lease_without_changing_job_version(self):
        self.submit("one")
        claim = self.scheduler.claim("worker-a")
        self.now += timedelta(seconds=10)
        renewed = self.scheduler.heartbeat(claim)
        self.assertGreater(renewed.lease_expires_at, claim.lease_expires_at)
        self.assertEqual(renewed.status.version, claim.status.version)
        self.assertEqual(renewed.token, claim.token)

    def test_expired_attempt_is_fenced_before_and_after_recovery(self):
        original = self.submit("one")
        first = self.scheduler.claim("worker-a")
        self.now = first.lease_expires_at
        self.conflict(lambda: self.scheduler.heartbeat(first))
        self.conflict(lambda: self.scheduler.mark_waiting(first))
        self.assertEqual(self.scheduler.recover(), 1)
        self.assertEqual(self.scheduler.recover(), 0)
        second = self.scheduler.claim("worker-b")
        self.assertEqual(second.status.job_id, original.job_id)
        self.assertEqual(second.attempt, 2)
        self.assertNotEqual(first.token, second.token)
        self.conflict(lambda: self.scheduler.release(first))

    def test_waiting_stage_is_leased_and_cannot_start_inference(self):
        self.submit("one")
        claim = self.scheduler.claim("worker-a")
        waiting = self.scheduler.mark_waiting(claim)
        self.assertEqual(waiting.status.state, JobState.WAITING_INFERENCE)
        self.conflict(lambda: self.scheduler.mark_waiting(waiting))
        self.now = waiting.lease_expires_at
        self.scheduler.recover()
        self.assertEqual(self.store.get(self.owner, claim.status.job_id).state, JobState.QUEUED)

    def test_cancellation_revokes_worker_authority_immediately(self):
        original = self.submit("one")
        claim = self.scheduler.claim("worker-a")
        cancelled = self.store.cancel(self.owner, original.job_id,
                                      expected_version=claim.status.version)
        self.assertEqual(cancelled.state, JobState.CANCELLED)
        self.conflict(lambda: self.scheduler.heartbeat(claim))
        self.conflict(lambda: self.scheduler.mark_waiting(claim))
        self.conflict(lambda: self.scheduler.release(claim))

    def test_deadline_expiry_is_terminal_not_requeued(self):
        original = self.submit("one")
        self.scheduler.claim("worker-a")
        self.now = original.expires_at
        self.scheduler.recover()
        self.assertEqual(self.store.get(self.owner, original.job_id).state, JobState.EXPIRED)
        self.assertIsNone(self.scheduler.claim("worker-b"))

    def test_release_preserves_attempt_counter_and_invalidates_token(self):
        self.submit("one")
        first = self.scheduler.claim("worker-a")
        released = self.scheduler.release(first)
        self.assertEqual(released.state, JobState.QUEUED)
        second = self.scheduler.claim("worker-a")
        self.assertEqual(second.attempt, 2)
        self.conflict(lambda: self.scheduler.heartbeat(first))

    def test_global_owner_and_worker_limits_are_shared_across_instances(self):
        settings = ServiceSettings(max_active_jobs=2, owner_max_active=1)
        self.store = JobStore(self.path, settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        other_scheduler = JobScheduler(JobStore(self.path, settings, clock=lambda: self.now))
        other = Principal("local", "operator-b")
        self.submit("a1")
        self.submit("a2")
        other_job = self.submit("b1", owner=other)
        self.scheduler.claim("worker-a")
        self.assertIsNone(other_scheduler.claim("worker-a"))
        claim = other_scheduler.claim("worker-b")
        self.assertEqual(claim.status.job_id, other_job.job_id)
        self.assertIsNone(self.scheduler.claim("worker-c"))

    def test_conversation_order_survives_capability_filtering(self):
        settings = ServiceSettings(max_active_jobs=2, owner_max_active=2)
        self.store = JobStore(self.path, settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        conversation = "a" * 32
        first = self.submit("first", conversation=conversation, profile="special")
        self.submit("second", conversation=conversation)
        independent = self.submit("independent")
        claim = self.scheduler.claim("worker-a")
        self.assertEqual(claim.status.job_id, independent.job_id)
        first_claim = self.scheduler.claim("worker-b", profile_ids=("special",))
        self.assertEqual(first_claim.status.job_id, first.job_id)

    def test_preparation_reserves_queue_capacity_for_safe_recovery(self):
        settings = ServiceSettings(max_waiting_jobs=1, owner_max_waiting=1)
        self.store = JobStore(self.path, settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        self.submit("one")
        self.scheduler.claim("worker-a")
        with self.assertRaises(ServiceError) as caught:
            self.submit("two")
        self.assertEqual(caught.exception.code, ErrorCode.QUEUE_FULL)

    def test_restart_preserves_lease_and_rejects_forged_claim(self):
        from dataclasses import replace
        self.submit("one")
        claim = self.scheduler.claim("worker-a")
        reopened = JobScheduler(JobStore(self.path, self.store.settings, clock=lambda: self.now))
        self.assertIsNone(reopened.claim("worker-b"))
        renewed = reopened.heartbeat(claim)
        self.assertEqual(renewed.attempt, 1)
        self.conflict(lambda: reopened.heartbeat(replace(claim, worker_id="worker-b")))
        self.conflict(lambda: reopened.heartbeat(replace(claim, token="f" * 64)))

    def test_version_one_migration_preserves_existing_admission(self):
        legacy = Path(self.directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy) as connection:
            connection.execute("""
                CREATE TABLE service_jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE, tenant_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL, idempotency_digest TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL, execution_profile_id TEXT NOT NULL,
                    conversation_id TEXT, state TEXT NOT NULL CHECK(state IN
                    ('queued','cancelled','expired')), version INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL, cancel_requested INTEGER NOT NULL,
                    UNIQUE(tenant_id,owner_id,idempotency_digest)
                )
            """)
            connection.execute("""
                INSERT INTO service_jobs VALUES
                (9, ?, 'local', 'operator-a', ?, ?, 'default', NULL, 'queued', 1,
                 '2026-01-01T00:00:00.000000Z','2026-01-01T00:00:00.000000Z',
                 '2026-01-01T00:10:00.000000Z',0)
            """, ("a" * 32, "b" * 64, "c" * 64))
            connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            connection.execute("PRAGMA user_version = 1")
        migrated = JobStore(legacy, ServiceSettings(), clock=lambda: self.now)
        claim = JobScheduler(migrated).claim("worker-a")
        self.assertEqual(claim.status.job_id, "a" * 32)
        self.assertEqual(claim.attempt, 1)
        with sqlite3.connect(legacy) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(connection.execute("SELECT sequence FROM service_jobs").fetchone()[0], 9)


if __name__ == "__main__":
    unittest.main()
