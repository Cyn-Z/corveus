"""Batch 2A tests. Run after service_job_store.py has been applied."""

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
from service_job_store import JobStore, UnsupportedJobSchemaError
from service_settings import ServiceSettings


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jobs.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.owner = Principal("local", "operator-a")
        self.settings = ServiceSettings()
        self.store = JobStore(self.path, self.settings, clock=lambda: self.now)

    def submission(self, key="request-1", prompt="Investigate Event 4625."):
        return JobSubmission.from_payload({"prompt": prompt}, idempotency_key=key)

    def assert_error(self, code, operation):
        with self.assertRaises(ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_restart_preserves_metadata_and_idempotency(self):
        submission = self.submission()
        admitted = self.store.admit(self.owner, submission)
        reopened = JobStore(self.path, self.settings, clock=lambda: self.now)
        retry = reopened.admit(self.owner, submission)
        self.assertTrue(admitted.created)
        self.assertFalse(retry.created)
        self.assertEqual(retry.status, admitted.status)
        self.assertEqual(reopened.get(self.owner, admitted.status.job_id), admitted.status)

    def test_prompt_and_raw_retry_key_are_not_written_to_database(self):
        prompt = "DISTINCT_SECRET_PROMPT_723918"
        key = "DISTINCT_SECRET_KEY_723918"
        self.store.admit(self.owner, self.submission(key, prompt))
        for path in Path(self.directory.name).iterdir():
            data = path.read_bytes()
            self.assertNotIn(prompt.encode(), data)
            self.assertNotIn(key.encode(), data)

    def test_retry_cannot_change_payload_or_execution_profile(self):
        self.store.admit(self.owner, self.submission())
        self.assert_error(ErrorCode.CONFLICT, lambda: self.store.admit(
            self.owner, self.submission(prompt="Changed prompt")
        ))
        self.assert_error(ErrorCode.CONFLICT, lambda: self.store.admit(
            self.owner, self.submission(), execution_profile_id="different"
        ))

    def test_owner_and_tenant_isolation_including_retry_namespace(self):
        original = self.store.admit(self.owner, self.submission()).status
        for stranger in (Principal("local", "operator-b"), Principal("other", "operator-a")):
            self.assert_error(ErrorCode.NOT_FOUND, lambda: self.store.get(stranger, original.job_id))
            self.assert_error(ErrorCode.NOT_FOUND, lambda: self.store.cancel(
                stranger, original.job_id, expected_version=1
            ))
            other = self.store.admit(stranger, self.submission()).status
            self.assertNotEqual(other.job_id, original.job_id)

    def test_owner_limit_and_retry_when_queue_is_full(self):
        for index in range(4):
            self.store.admit(self.owner, self.submission(f"request-{index}"))
        self.assert_error(ErrorCode.QUEUE_FULL, lambda: self.store.admit(
            self.owner, self.submission("overflow")
        ))
        self.assertFalse(self.store.admit(self.owner, self.submission("request-0")).created)

    def test_global_limit_applies_across_owners(self):
        settings = ServiceSettings(max_waiting_jobs=2, owner_max_waiting=2)
        store = JobStore(self.path, settings, clock=lambda: self.now)
        store.admit(self.owner, self.submission("one"))
        store.admit(Principal("local", "operator-b"), self.submission("two"))
        self.assert_error(ErrorCode.QUEUE_FULL, lambda: store.admit(
            Principal("local", "operator-c"), self.submission("three")
        ))

    def test_cancel_requires_current_version_and_is_idempotent(self):
        status = self.store.admit(self.owner, self.submission()).status
        self.assert_error(ErrorCode.CONFLICT, lambda: self.store.cancel(
            self.owner, status.job_id, expected_version=2
        ))
        cancelled = self.store.cancel(self.owner, status.job_id, expected_version=1)
        self.assertEqual(cancelled.state, JobState.CANCELLED)
        self.assertEqual(cancelled.version, 2)
        self.assertEqual(self.store.cancel(
            self.owner, status.job_id, expected_version=1
        ), cancelled)

    def test_deadline_expiry_releases_queue_capacity_without_recreating_retry(self):
        first = self.store.admit(self.owner, self.submission()).status
        self.now = first.expires_at
        self.assertEqual(self.store.expire_queued(), 1)
        self.assertEqual(self.store.expire_queued(), 0)
        expired = self.store.get(self.owner, first.job_id)
        self.assertEqual(expired.state, JobState.EXPIRED)
        self.assertEqual(expired.version, 2)
        retry = self.store.admit(self.owner, self.submission())
        self.assertFalse(retry.created)
        self.assertEqual(retry.status.state, JobState.EXPIRED)
        self.assertTrue(self.store.admit(self.owner, self.submission("new-key")).created)

    def test_concurrent_retries_have_one_durable_job(self):
        def submit(_):
            store = JobStore(self.path, self.settings, clock=lambda: self.now)
            return store.admit(self.owner, self.submission())
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(submit, range(16)))
        self.assertEqual(sum(item.created for item in results), 1)
        self.assertEqual(len({item.status.job_id for item in results}), 1)

    def test_concurrent_admission_cannot_exceed_capacity(self):
        settings = ServiceSettings(max_waiting_jobs=2, owner_max_waiting=2)
        store = JobStore(self.path, settings, clock=lambda: self.now)
        def submit(index):
            try:
                return store.admit(self.owner, self.submission(f"request-{index}"))
            except ServiceError as exc:
                self.assertEqual(exc.code, ErrorCode.QUEUE_FULL)
                return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(submit, range(12)))
        self.assertEqual(sum(item is not None for item in results), 2)

    def test_database_identity_and_future_versions_are_rejected(self):
        unrelated = Path(self.directory.name) / "unrelated.sqlite3"
        with sqlite3.connect(unrelated) as connection:
            connection.execute("CREATE TABLE existing_data (value TEXT)")
        with self.assertRaises(UnsupportedJobSchemaError):
            JobStore(unrelated, self.settings)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA user_version = 999")
        with self.assertRaises(UnsupportedJobSchemaError):
            JobStore(self.path, self.settings)

    def test_invalid_inputs_and_backward_clock(self):
        with self.assertRaises(ValueError):
            JobStore(":memory:", self.settings)
        with self.assertRaises(ValueError):
            self.store.get(self.owner, "invalid")
        status = self.store.admit(self.owner, self.submission()).status
        with self.assertRaises(ValueError):
            self.store.cancel(self.owner, status.job_id, expected_version=True)
        self.now -= timedelta(seconds=30)
        cancelled = self.store.cancel(self.owner, status.job_id, expected_version=1)
        self.assertGreaterEqual(cancelled.updated_at, cancelled.created_at)


if __name__ == "__main__":
    unittest.main()
