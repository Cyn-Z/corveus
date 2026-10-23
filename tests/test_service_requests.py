"""Batch 3A: atomic temporary request storage and configured expiry."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from service_contracts import ErrorCode, JobSubmission, Principal, ServiceError
from service_inference import InferenceGate
from service_job_store import SCHEMA_VERSION, JobStore
from service_requests import JobRequests
from service_scheduler import JobScheduler
from service_settings import ServiceSettings, load_service_settings


class RequestStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jobs.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.owner = Principal("local", "operator-a")
        self.settings = ServiceSettings()
        self.store = JobStore(self.path, self.settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        self.requests = JobRequests(self.scheduler)

    def submission(self, key="one", conversation=None):
        return JobSubmission.from_payload(
            {"prompt": "  Untrusted request content.\n", "conversation_id": conversation},
            idempotency_key=key,
        )

    def error(self, code, operation):
        with self.assertRaises(ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_restart_preserves_exact_prompt(self):
        admission = self.requests.submit(self.owner, self.submission())
        reopened = JobRequests(JobScheduler(JobStore(
            self.path, self.settings, clock=lambda: self.now
        )))
        content = reopened.get(self.owner, admission.status.job_id)
        self.assertEqual(content.prompt, self.submission().prompt)
        self.assertNotIn(content.prompt, repr(content))

    def test_retries_do_not_extend_retention(self):
        first = self.requests.submit(self.owner, self.submission())
        original = self.requests.get(self.owner, first.status.job_id)
        self.now += timedelta(seconds=10)
        retry = self.requests.submit(self.owner, self.submission())
        self.assertFalse(retry.created)
        self.assertEqual(self.requests.get(self.owner, retry.status.job_id).content_expires_at,
                         original.content_expires_at)

    def test_exact_expiry_and_retry_after_cleanup_do_not_resurrect_content(self):
        admitted = self.requests.submit(self.owner, self.submission())
        self.now = self.requests.get(self.owner, admitted.status.job_id).content_expires_at
        self.error(ErrorCode.CONTENT_EXPIRED, lambda: self.requests.get(self.owner, admitted.status.job_id))
        self.assertEqual(self.requests.prune_expired(), 1)
        self.error(ErrorCode.CONTENT_EXPIRED, lambda: self.requests.submit(self.owner, self.submission()))
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).job_id, admitted.status.job_id)

    def test_both_owner_and_tenant_are_required_for_content_access(self):
        admitted = self.requests.submit(self.owner, self.submission())
        for stranger in (Principal("local", "other"), Principal("other", "operator-a")):
            self.error(ErrorCode.NOT_FOUND, lambda: self.requests.get(stranger, admitted.status.job_id))

    def test_concurrent_retries_create_one_job_and_one_content_row(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(
                lambda _: self.requests.submit(self.owner, self.submission()), range(16)
            ))
        self.assertEqual(sum(result.created for result in results), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_jobs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_request_content").fetchone()[0], 1)

    def test_content_insert_failure_rolls_back_job_admission(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TRIGGER reject_content BEFORE INSERT ON service_request_content
                BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.requests.submit(self.owner, self.submission())
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_jobs").fetchone()[0], 0)

    def test_rejected_admission_leaves_no_orphan_content(self):
        settings = ServiceSettings(max_waiting_jobs=1, owner_max_waiting=1)
        requests = JobRequests(JobScheduler(JobStore(self.path, settings, clock=lambda: self.now)))
        requests.submit(self.owner, self.submission())
        self.error(ErrorCode.QUEUE_FULL, lambda: requests.submit(self.owner, self.submission("two")))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_request_content").fetchone()[0], 1)

    def test_expiry_never_releases_uncertain_inference_capacity(self):
        self.requests.submit(self.owner, self.submission())
        claim = self.scheduler.mark_waiting(self.scheduler.claim("worker-a"))
        gate = InferenceGate(self.scheduler)
        gate.start(claim, reserved_bytes=10)
        self.now += timedelta(seconds=self.settings.content_retention_seconds)
        self.assertEqual(self.requests.prune_expired(), 1)
        self.assertEqual(len(gate.held_permits()), 1)
        self.scheduler.recover()
        self.assertEqual(len(gate.held_permits()), 1)

    def test_worker_read_requires_current_attempt_credentials(self):
        admitted = self.requests.submit(self.owner, self.submission())
        claim = self.scheduler.claim("worker-a")
        self.assertEqual(self.requests.for_worker(claim).status.job_id, admitted.status.job_id)
        self.error(ErrorCode.CONFLICT, lambda: self.requests.for_worker(replace(claim, token="f" * 64)))
        self.now = claim.lease_expires_at
        self.error(ErrorCode.CONFLICT, lambda: self.requests.for_worker(claim))

    def test_retention_configuration_is_validated_and_environment_driven(self):
        for value in (True, 0, 600):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ServiceSettings(content_retention_seconds=value)
        settings = load_service_settings({"SERVICE_CONTENT_RETENTION_SECONDS": "7200"})
        self.assertEqual(settings.content_retention_seconds, 7200)

    def test_schema_three_migration_preserves_live_inference_hold(self):
        admitted = self.store.admit(self.owner, self.submission())
        claim = self.scheduler.mark_waiting(self.scheduler.claim("worker-a"))
        gate = InferenceGate(self.scheduler)
        permit = gate.start(claim, reserved_bytes=10)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE service_preparations")
            connection.execute("DROP TABLE service_job_results")
            connection.execute("DROP TABLE service_request_content")
            connection.execute("PRAGMA user_version = 3")
        migrated = JobStore(self.path, self.settings, clock=lambda: self.now)
        reopened = InferenceGate(JobScheduler(migrated))
        self.assertEqual(len(reopened.held_permits()), 1)
        self.assertEqual(reopened.held_permits()[0].token, permit.token)
        self.assertEqual(migrated.get(self.owner, admitted.status.job_id).job_id, admitted.status.job_id)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_metadata_only_retry_cannot_silently_recreate_missing_content(self):
        self.store.admit(self.owner, self.submission())
        self.error(ErrorCode.CONTENT_EXPIRED, lambda: self.requests.submit(self.owner, self.submission()))

    def test_saved_and_incognito_requests_use_same_temporary_retention(self):
        for index, conversation in enumerate((None, "a" * 32)):
            admitted = self.requests.submit(self.owner, self.submission(str(index), conversation))
            content = self.requests.get(self.owner, admitted.status.job_id)
            self.assertEqual(content.status.conversation_id, conversation)
            self.assertEqual(content.content_expires_at, self.now + timedelta(
                seconds=self.settings.content_retention_seconds
            ))

    def test_failed_content_migration_rolls_back_schema_version(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE service_preparations")
            connection.execute("DROP TABLE service_job_results")
            connection.execute("DROP TABLE service_request_content")
            connection.execute("PRAGMA user_version = 3")
        with patch("service_job_store._REQUEST_SCHEMA", "INVALID SQL"):
            with self.assertRaises(sqlite3.OperationalError):
                JobStore(self.path, self.settings)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
