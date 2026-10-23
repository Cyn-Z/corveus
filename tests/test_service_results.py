"""Batch 3B.2: durable capture and recoverable conversation finalization."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from memory_manager import MemoryManager
from service_contracts import ErrorCode, JobState, JobSubmission, Principal, ServiceError
from service_inference import InferenceGate
from service_job_store import JobStore, SCHEMA_VERSION
from service_requests import JobRequests
from service_results import JobResults, ResultSnapshot
from service_scheduler import JobScheduler
from service_settings import ServiceSettings
from token_usage import TokenUsage


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jobs.sqlite3"
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.owner = Principal("local", "operator-a")
        self.settings = ServiceSettings()
        self.store = JobStore(self.path, self.settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        self.gate = InferenceGate(self.scheduler)
        self.requests = JobRequests(self.scheduler)
        self.memory = MemoryManager(enabled=True, database_path=Path(self.directory.name) / "memory.sqlite3",
                                    scope="local-operator")
        self.results = JobResults(self.gate, memory_for=lambda principal: self.memory)
        self.snapshot = ResultSnapshot("Answer", TokenUsage("model", 100, 10, 5))

    def running(self, conversation=None):
        admission = self.requests.submit(self.owner, JobSubmission.from_payload(
            {"prompt": "Question", "conversation_id": conversation}, idempotency_key="one"
        ))
        claim = self.scheduler.mark_waiting(self.scheduler.claim("worker-a"))
        permit = self.gate.start(claim, reserved_bytes=10)
        return admission, claim, permit

    def capture(self, conversation=None):
        admission, claim, permit = self.running(conversation)
        self.results.capture(permit, self.snapshot, inference_stopped=True)
        return admission, claim, permit

    def error(self, code, operation):
        with self.assertRaises(ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_capture_commits_result_state_and_resource_release_together(self):
        admitted, _, _ = self.capture()
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).state, JobState.FINALIZING)
        self.assertEqual(self.gate.held_permits(), ())
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_job_results").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_job_leases").fetchone()[0], 0)
        self.error(ErrorCode.CONFLICT, lambda: self.results.get(self.owner, admitted.status.job_id))

    def test_result_insert_failure_rolls_back_running_state_and_permit_release(self):
        admitted, _, permit = self.running()
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TRIGGER reject_result BEFORE INSERT ON service_job_results
                BEGIN SELECT RAISE(ABORT, 'simulated failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.results.capture(permit, self.snapshot, inference_stopped=True)
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).state, JobState.RUNNING)
        self.assertEqual(len(self.gate.held_permits()), 1)

    def test_capture_retry_is_idempotent_and_changed_answer_conflicts(self):
        admitted, _, permit = self.capture()
        status = self.results.capture(permit, self.snapshot, inference_stopped=True)
        self.assertEqual(status.state, JobState.FINALIZING)
        self.error(ErrorCode.CONFLICT, lambda: self.results.capture(
            permit, ResultSnapshot("Different"), inference_stopped=True
        ))
        self.results.finalize(self.owner, admitted.status.job_id)
        self.assertEqual(self.results.capture(permit, self.snapshot, inference_stopped=True).state,
                         JobState.COMPLETED)

    def test_cancelled_capture_stores_no_answer(self):
        admitted, _, permit = self.running()
        self.store.cancel(self.owner, admitted.status.job_id, expected_version=permit.status.version)
        status = self.results.capture(permit, self.snapshot, inference_stopped=True)
        self.assertEqual(status.state, JobState.CANCELLED)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_job_results").fetchone()[0], 0)

    def test_expired_attempt_cannot_publish_result_or_free_hold(self):
        _, claim, permit = self.running()
        self.now = claim.lease_expires_at
        self.error(ErrorCode.CONFLICT, lambda: self.results.capture(
            permit, self.snapshot, inference_stopped=True
        ))
        self.assertEqual(len(self.gate.held_permits()), 1)

    def test_incognito_result_survives_restart_and_finishes_without_memory(self):
        admitted, _, _ = self.capture()
        reopened = JobResults(InferenceGate(JobScheduler(JobStore(
            self.path, self.settings, clock=lambda: self.now
        ))))
        self.assertEqual(reopened.recover(), 1)
        result = reopened.get(self.owner, admitted.status.job_id)
        self.assertEqual(result.snapshot, self.snapshot)
        self.assertEqual(result.persistence_state, "not_requested")
        self.assertNotIn("Answer", repr(result))

    def test_saved_result_persists_one_exchange_and_usage(self):
        conversation = self.memory.create_conversation()
        admitted, _, _ = self.capture(conversation.id)
        first = self.results.finalize(self.owner, admitted.status.job_id)
        second = self.results.finalize(self.owner, admitted.status.job_id)
        self.assertEqual(first, second)
        self.assertEqual(first.persistence_state, "saved")
        self.assertEqual(len(self.memory.get_messages(conversation.id)), 2)
        self.assertEqual(self.memory.get_conversation_usage(conversation.id)["total_tokens"], 15)

    def test_crash_after_conversation_commit_does_not_duplicate_on_retry(self):
        conversation = self.memory.create_conversation()
        admitted, _, _ = self.capture(conversation.id)
        with patch.object(self.results, "_mark_completed", side_effect=RuntimeError("simulated crash")):
            with self.assertRaises(RuntimeError):
                self.results.finalize(self.owner, admitted.status.job_id)
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).state, JobState.FINALIZING)
        self.assertEqual(len(self.memory.get_messages(conversation.id)), 2)
        reopened = JobResults(self.gate, memory_for=lambda principal: self.memory)
        reopened.recover()
        self.assertEqual(len(self.memory.get_messages(conversation.id)), 2)

    def test_transient_memory_failure_keeps_answer_pending_for_recovery(self):
        conversation = self.memory.create_conversation()
        admitted, _, _ = self.capture(conversation.id)
        with patch.object(self.memory, "record_completed_exchange", side_effect=sqlite3.OperationalError("busy")):
            with self.assertRaises(sqlite3.OperationalError):
                self.results.finalize(self.owner, admitted.status.job_id)
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).state, JobState.FINALIZING)
        self.assertEqual(self.results.recover(), 1)

    def test_deleted_conversation_returns_answer_with_unsaved_status(self):
        conversation = self.memory.create_conversation()
        admitted, _, _ = self.capture(conversation.id)
        self.memory.delete_conversation(conversation.id)
        result = self.results.finalize(self.owner, admitted.status.job_id)
        self.assertEqual(result.persistence_state, "conversation_deleted")
        self.assertEqual(result.snapshot.answer, "Answer")

    def test_disabled_memory_returns_explicit_unavailable_status(self):
        conversation = self.memory.create_conversation()
        admitted, _, _ = self.capture(conversation.id)
        result = JobResults(self.gate).finalize(self.owner, admitted.status.job_id)
        self.assertEqual(result.persistence_state, "memory_unavailable")
        self.assertEqual(self.memory.get_messages(conversation.id), [])

    def test_foreign_owner_or_tenant_cannot_finalize_or_read(self):
        admitted, _, _ = self.capture()
        for stranger in (Principal("other", "operator-a"), Principal("local", "operator-b")):
            self.error(ErrorCode.NOT_FOUND, lambda: self.results.finalize(stranger, admitted.status.job_id))
            self.error(ErrorCode.NOT_FOUND, lambda: self.results.get(stranger, admitted.status.job_id))

    def test_concurrent_finalizers_write_one_exchange(self):
        conversation = self.memory.create_conversation()
        admitted, _, _ = self.capture(conversation.id)
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(lambda _: self.results.finalize(self.owner, admitted.status.job_id), range(12)))
        self.assertTrue(all(result.persistence_state == "saved" for result in results))
        self.assertEqual(len(self.memory.get_messages(conversation.id)), 2)

    def test_expiry_cascades_to_answer_and_preserves_metadata(self):
        admitted, _, _ = self.capture()
        self.results.finalize(self.owner, admitted.status.job_id)
        self.now += timedelta(seconds=self.settings.content_retention_seconds)
        self.error(ErrorCode.CONTENT_EXPIRED, lambda: self.results.get(self.owner, admitted.status.job_id))
        self.requests.prune_expired()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_job_results").fetchone()[0], 0)
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).state, JobState.COMPLETED)

    def test_finalization_can_finish_after_generation_deadline_before_content_expiry(self):
        admitted, _, _ = self.capture()
        self.now = admitted.status.expires_at + timedelta(seconds=1)
        self.scheduler.recover()
        self.assertEqual(self.results.finalize(self.owner, admitted.status.job_id).status.state, JobState.COMPLETED)

    def test_recovery_never_calls_inference(self):
        self.capture()
        with patch.object(self.gate, "start", side_effect=AssertionError("must not infer")):
            self.assertEqual(self.results.recover(), 1)

    def test_schema_four_migration_preserves_existing_prompt_and_hold(self):
        admitted, _, permit = self.running()
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE service_preparations")
            connection.execute("DROP TABLE service_job_results")
            connection.execute("PRAGMA user_version = 4")
        reopened = JobStore(self.path, self.settings, clock=lambda: self.now)
        self.assertEqual(JobRequests(JobScheduler(reopened)).get(self.owner, admitted.status.job_id).prompt, "Question")
        self.assertEqual(InferenceGate(JobScheduler(reopened)).held_permits()[0].token, permit.token)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_snapshot_and_stop_confirmation_are_validated(self):
        for answer in ("", "   ", False, "\ud800", "x" * 4_000_001):
            with self.subTest(answer_type=type(answer).__name__):
                with self.assertRaises(ValueError):
                    ResultSnapshot(answer)
        _, _, permit = self.running()
        with self.assertRaises(ValueError):
            self.results.capture(permit, self.snapshot, inference_stopped=False)
        self.assertEqual(len(self.gate.held_permits()), 1)

    def test_expired_pending_finalization_fails_without_rerunning_inference(self):
        admitted, _, _ = self.capture()
        self.now += timedelta(seconds=self.settings.content_retention_seconds)
        self.requests.prune_expired()
        self.scheduler.recover()
        self.assertEqual(self.store.get(self.owner, admitted.status.job_id).state, JobState.FAILED)

    def test_forged_capture_retry_cannot_use_another_permit_token(self):
        from dataclasses import replace
        _, _, permit = self.capture()
        self.error(ErrorCode.CONFLICT, lambda: self.results.capture(
            replace(permit, token="f" * 64), self.snapshot, inference_stopped=True
        ))


if __name__ == "__main__":
    unittest.main()
