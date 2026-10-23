"""Batch 2B.2: transactional permits, cancellation, and uncertain shutdown."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from service_contracts import ErrorCode, JobState, JobSubmission, Principal, ServiceError
from service_inference import InferenceGate
from service_job_store import APPLICATION_ID, SCHEMA_VERSION, JobStore
from service_scheduler import JobScheduler
from service_settings import ServiceSettings


class InferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jobs.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.owner = Principal("local", "operator-a")
        self.configure()

    def configure(self, **changes):
        arguments = {"max_active_jobs": 3, "owner_max_active": 3}
        arguments.update(changes)
        self.settings = ServiceSettings(**arguments)
        self.store = JobStore(self.path, self.settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        self.gate = InferenceGate(self.scheduler)

    def prepare(self, key, conversation=None):
        submission = JobSubmission.from_payload(
            {"prompt": "Investigate", "conversation_id": conversation},
            idempotency_key=key,
        )
        self.store.admit(self.owner, submission)
        claim = self.scheduler.claim(f"worker-{key}")
        self.assertIsNotNone(claim)
        return self.scheduler.mark_waiting(claim)

    def conflict(self, operation):
        with self.assertRaises(ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, ErrorCode.CONFLICT)

    def test_shared_count_limit_and_no_duplicate_permit(self):
        first, second = self.prepare("one"), self.prepare("two")
        permit = self.gate.start(first, reserved_bytes=10)
        self.assertEqual(permit.status.state, JobState.RUNNING)
        self.assertNotIn(permit.token, repr(permit))
        self.assertIsNone(self.gate.start(second, reserved_bytes=10))
        self.conflict(lambda: self.gate.start(first, reserved_bytes=10))
        other = InferenceGate(JobScheduler(JobStore(
            self.path, self.settings, clock=lambda: self.now
        )))
        self.assertIsNone(other.start(second, reserved_bytes=10))

    def test_concurrent_starts_cannot_exceed_global_limit(self):
        claims = [self.prepare("one"), self.prepare("two"), self.prepare("three")]
        with ThreadPoolExecutor(max_workers=3) as executor:
            permits = list(executor.map(
                lambda claim: self.gate.start(claim, reserved_bytes=10), claims
            ))
        self.assertEqual(sum(permit is not None for permit in permits), 1)

    def test_memory_budget_is_aggregate_and_requires_valid_reservations(self):
        self.configure(max_inference_requests=2, inference_memory_budget_bytes=100)
        first, second = self.prepare("one"), self.prepare("two")
        permit = self.gate.start(first, reserved_bytes=70)
        self.assertIsNone(self.gate.start(second, reserved_bytes=40))
        self.assertIsNotNone(self.gate.start(second, reserved_bytes=30))
        self.assertIsNotNone(permit)
        for invalid in (True, 0, -1, 101):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.gate.start(first, reserved_bytes=invalid)

    def test_live_running_heartbeat_and_confirmed_completion(self):
        claim = self.prepare("one")
        permit = self.gate.start(claim, reserved_bytes=10)
        self.now += timedelta(seconds=10)
        renewed = self.scheduler.heartbeat(claim)
        self.assertEqual(renewed.status.state, JobState.RUNNING)
        finalizing = self.gate.finish(permit, inference_stopped=True)
        self.assertEqual(finalizing.state, JobState.FINALIZING)
        self.assertEqual(self.gate.held_permits(), ())
        complete = self.gate.complete(renewed)
        self.assertEqual(complete.state, JobState.COMPLETED)
        self.conflict(lambda: self.scheduler.heartbeat(claim))

    def test_no_release_without_stop_confirmation(self):
        claim = self.prepare("one")
        permit = self.gate.start(claim, reserved_bytes=10)
        with self.assertRaises(ValueError):
            self.gate.finish(permit, inference_stopped=False)
        self.assertEqual(len(self.gate.held_permits()), 1)
        self.assertEqual(self.store.get(self.owner, claim.status.job_id).state, JobState.RUNNING)

    def test_running_cancellation_keeps_permit_until_confirmed_stop(self):
        claim = self.prepare("one")
        permit = self.gate.start(claim, reserved_bytes=10)
        requested = self.store.cancel(self.owner, claim.status.job_id,
                                      expected_version=permit.status.version)
        self.assertEqual(requested.state, JobState.RUNNING)
        self.assertTrue(requested.cancel_requested)
        self.assertEqual(len(self.gate.held_permits()), 1)
        self.assertEqual(self.store.cancel(self.owner, claim.status.job_id,
                         expected_version=permit.status.version), requested)
        ended = self.gate.finish(permit, inference_stopped=True)
        self.assertEqual(ended.state, JobState.CANCELLED)
        self.assertEqual(self.gate.held_permits(), ())

    def test_expired_running_lease_is_interrupted_and_holds_capacity(self):
        first, second = self.prepare("one"), self.prepare("two")
        permit = self.gate.start(first, reserved_bytes=10)
        self.now = first.lease_expires_at
        self.assertEqual(self.scheduler.recover(), 2)
        self.assertEqual(self.store.get(self.owner, first.status.job_id).state, JobState.INTERRUPTED)
        self.assertEqual(len(self.gate.held_permits()), 1)
        replacement = self.scheduler.claim("worker-new")
        replacement = self.scheduler.mark_waiting(replacement)
        self.assertIsNone(self.gate.start(replacement, reserved_bytes=10))
        self.conflict(lambda: self.gate.finish(permit, inference_stopped=True))

    def test_job_deadline_does_not_release_uncertain_inference(self):
        claim = self.prepare("one")
        self.gate.start(claim, reserved_bytes=10)
        self.now = claim.status.expires_at
        self.scheduler.recover()
        self.assertEqual(self.store.get(self.owner, claim.status.job_id).state, JobState.INTERRUPTED)
        self.assertEqual(len(self.gate.held_permits()), 1)

    def test_late_finish_cannot_bypass_lease_expiry(self):
        claim = self.prepare("one")
        permit = self.gate.start(claim, reserved_bytes=10)
        self.now = claim.lease_expires_at
        self.conflict(lambda: self.gate.finish(permit, inference_stopped=True))
        self.assertEqual(len(self.gate.held_permits()), 1)

    def test_restart_retains_hold_and_explicit_reconciliation_releases_it(self):
        claim = self.prepare("one")
        self.gate.start(claim, reserved_bytes=10)
        self.now = claim.lease_expires_at
        reopened = InferenceGate(JobScheduler(JobStore(
            self.path, self.settings, clock=lambda: self.now
        )))
        reopened.scheduler.recover()
        held, = reopened.held_permits()
        with self.assertRaises(ValueError):
            reopened.reconcile_stopped(held, inference_stopped=False)
        released = reopened.reconcile_stopped(held, inference_stopped=True)
        self.assertEqual(released.state, JobState.INTERRUPTED)
        self.assertEqual(reopened.held_permits(), ())
        self.conflict(lambda: reopened.reconcile_stopped(held, inference_stopped=True))

    def test_forged_permit_cannot_finish_or_clear_hold(self):
        claim = self.prepare("one")
        permit = self.gate.start(claim, reserved_bytes=10)
        self.conflict(lambda: self.gate.finish(
            replace(permit, token="f" * 64), inference_stopped=True
        ))
        self.conflict(lambda: self.gate.finish(
            replace(permit, worker_id="other-worker"), inference_stopped=True
        ))
        self.assertEqual(len(self.gate.held_permits()), 1)

    def test_running_work_cannot_be_requeued_with_preparation_methods(self):
        claim = self.prepare("one")
        self.gate.start(claim, reserved_bytes=10)
        self.conflict(lambda: self.scheduler.release(claim))
        self.conflict(lambda: self.scheduler.mark_waiting(claim))

    def test_finalizing_rejects_cancellation_and_recovery_does_not_leak_worker_slot(self):
        claim = self.prepare("one")
        permit = self.gate.start(claim, reserved_bytes=10)
        status = self.gate.finish(permit, inference_stopped=True)
        self.conflict(lambda: self.store.cancel(
            self.owner, status.job_id, expected_version=status.version
        ))
        self.now = claim.lease_expires_at
        self.scheduler.recover()
        self.assertEqual(self.store.get(self.owner, status.job_id).state, JobState.INTERRUPTED)
        self.assertEqual(self.gate.held_permits(), ())
        self.assertIsNotNone(self.prepare("two"))

    def test_uncertain_hold_blocks_later_turn_until_reconciled(self):
        conversation = "a" * 32
        claim = self.prepare("one", conversation)
        permit = self.gate.start(claim, reserved_bytes=10)
        self.store.admit(self.owner, JobSubmission.from_payload(
            {"prompt": "Next turn", "conversation_id": conversation},
            idempotency_key="two",
        ))
        self.now = claim.lease_expires_at
        self.scheduler.recover()
        self.assertIsNone(self.scheduler.claim("worker-two"))
        self.gate.reconcile_stopped(permit, inference_stopped=True)
        self.assertIsNotNone(self.scheduler.claim("worker-two"))

    def test_confirmed_failure_releases_capacity_for_another_job(self):
        first, second = self.prepare("one"), self.prepare("two")
        permit = self.gate.start(first, reserved_bytes=10)
        ended = self.gate.finish(permit, inference_stopped=True, failed=True)
        self.assertEqual(ended.state, JobState.FAILED)
        self.assertIsNotNone(self.gate.start(second, reserved_bytes=10))

    def test_version_two_migration_preserves_active_preparation_lease(self):
        legacy = Path(self.directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy) as connection:
            connection.execute("""
                CREATE TABLE service_jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE, tenant_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL, idempotency_digest TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL, execution_profile_id TEXT NOT NULL,
                    conversation_id TEXT, state TEXT NOT NULL CHECK(state IN
                    ('queued','preparing','waiting_inference','cancelled','expired')),
                    version INTEGER NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL, cancel_requested INTEGER NOT NULL,
                    UNIQUE(tenant_id,owner_id,idempotency_digest)
                )
            """)
            connection.execute("""
                CREATE TABLE service_job_leases (
                    job_id TEXT PRIMARY KEY NOT NULL REFERENCES service_jobs(job_id),
                    worker_id TEXT NOT NULL, token TEXT NOT NULL,
                    attempt INTEGER NOT NULL, lease_expires_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                INSERT INTO service_jobs VALUES
                (9, ?, 'local', 'operator-a', ?, ?, 'default', NULL, 'preparing', 2, 1,
                 '2026-01-01T00:00:00.000000Z','2026-01-01T00:00:00.000000Z',
                 '2026-01-01T00:10:00.000000Z',0)
            """, ("a" * 32, "b" * 64, "c" * 64))
            connection.execute("""
                INSERT INTO service_job_leases VALUES
                (?, 'worker-legacy', ?, 1, '2026-01-01T00:00:30.000000Z')
            """, ("a" * 32, "d" * 64))
            connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            connection.execute("PRAGMA user_version = 2")
        migrated = JobStore(legacy, ServiceSettings(), clock=lambda: self.now)
        scheduler = JobScheduler(migrated)
        from service_scheduler import WorkerLease
        claim = WorkerLease(migrated.get(self.owner, "a" * 32), "worker-legacy", 1,
                            "d" * 64, self.now + timedelta(seconds=30))
        self.assertEqual(scheduler.heartbeat(claim).attempt, 1)
        with sqlite3.connect(legacy) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
