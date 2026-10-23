"""Batch 4B: persisted skill snapshots and owner-scoped review decisions."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

ANALYST_DIR = Path(__file__).resolve().parents[1] / "analyst"
sys.path.insert(0, str(ANALYST_DIR))

from analyst_preparation import PreparationSnapshot, PromptPreparer, PromptTooLargeError
from service_contracts import ErrorCode, ExecutionProfile, JobState, JobSubmission, Principal, ServiceError
from service_inference import InferenceGate
from service_job_store import JobStore, SCHEMA_VERSION
from service_preparation import JobPreparation
from service_requests import JobRequests
from service_results import JobResults, ResultSnapshot
from service_scheduler import JobScheduler
from service_settings import ServiceSettings


class PreparationTests(unittest.TestCase):
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
        self.router = Mock()
        self.registry = Mock()
        self.registry.load_instructions.side_effect = lambda skill: f"Saved guidance for {skill.name}"
        self.skills = [SimpleNamespace(name=name, description="Guidance") for name in ("one", "two", "three")]
        self.router.search.return_value = [(0.9, skill) for skill in self.skills[:2]]
        self.counter = Mock(side_effect=lambda messages: 10 + 60 * messages[0]["content"].count("## Skill: "))
        self.profile = ExecutionProfile("default", "model", "gpu", 130, 20, 10)
        self.preparer = PromptPreparer(self.router, self.registry, self.counter, self.profile, top_k=8)
        self.preparation = JobPreparation(self.scheduler, self.preparer)

    def submit(self, key="one", conversation=None):
        return self.requests.submit(self.owner, JobSubmission.from_payload(
            {"prompt": "Question", "conversation_id": conversation}, idempotency_key=key
        )).status

    def review_job(self, key="one", conversation=None):
        self.submit(key, conversation)
        claim = self.scheduler.claim(f"worker-{key}")
        return self.preparation.prepare(claim), claim

    def error(self, code, operation):
        with self.assertRaises(ServiceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_pure_snapshot_round_trip_preserves_messages_and_untrusted_guidance(self):
        snapshot = self.preparer.prepare("Question")
        restored = PreparationSnapshot.from_json(snapshot.to_json())
        self.assertEqual(snapshot, restored)
        self.assertEqual(snapshot.messages(), restored.messages())
        self.assertIn("untrusted content", restored.messages()[0]["content"])
        self.assertTrue(snapshot.needs_review)
        self.assertNotIn("Saved guidance", repr(snapshot))

    def test_under_budget_preparation_enters_waiting_inference(self):
        self.router.search.return_value = []
        self.submit()
        claim = self.scheduler.claim("worker-a")
        status = self.preparation.prepare(claim)
        self.assertEqual(status.state, JobState.WAITING_INFERENCE)
        self.assertEqual(self.preparation.for_worker(claim).input_tokens, 10)
        self.assertEqual(InferenceGate(self.scheduler).held_permits(), ())

    def test_review_releases_worker_slot_without_creating_inference_hold(self):
        status, old = self.review_job()
        self.assertEqual(status.state, JobState.AWAITING_REVIEW)
        self.error(ErrorCode.CONFLICT, lambda: self.scheduler.heartbeat(old))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_job_leases").fetchone()[0], 0)
        self.assertEqual(InferenceGate(self.scheduler).held_permits(), ())

    def test_inspection_shows_metadata_without_instruction_content(self):
        status, _ = self.review_job()
        view = self.preparation.inspect(self.owner, status.job_id)
        self.assertEqual(tuple(skill.name for skill in view.skills), ("one", "two"))
        self.assertEqual(view.input_tokens, 130)
        self.assertEqual(view.input_budget, 100)
        self.assertNotIn("Saved guidance", repr(view))

    def test_restart_and_review_resume_do_not_reroute_or_reload_instructions(self):
        status, _ = self.review_job()
        reopened = JobPreparation(JobScheduler(JobStore(self.path, self.settings, clock=lambda: self.now)), self.preparer)
        queued = reopened.review(self.owner, status.job_id, ["one"], expected_version=status.version)
        self.assertEqual(queued.state, JobState.QUEUED)
        self.registry.load_instructions.side_effect = AssertionError("must use stored instructions")
        self.router.search.side_effect = AssertionError("must not reroute")
        claim = reopened.scheduler.claim("worker-b")
        self.assertEqual(reopened.prepare(claim).state, JobState.WAITING_INFERENCE)
        self.assertEqual(tuple(skill.name for skill in reopened.for_worker(claim).skills), ("two",))

    def test_partial_removal_recounts_and_keeps_original_review_deadline(self):
        self.router.search.return_value = [(0.9, skill) for skill in self.skills]
        status, _ = self.review_job()
        deadline = self.preparation.inspect(self.owner, status.job_id).expires_at
        self.now += timedelta(seconds=10)
        updated = self.preparation.review(self.owner, status.job_id, ["one"], expected_version=status.version)
        self.assertEqual(updated.state, JobState.AWAITING_REVIEW)
        self.assertEqual(updated.version, status.version + 1)
        view = self.preparation.inspect(self.owner, status.job_id)
        self.assertEqual(view.expires_at, deadline)
        self.assertEqual(view.input_tokens, 130)

    def test_invalid_removals_and_stale_versions_do_not_change_review(self):
        status, _ = self.review_job()
        for names in ([], ["unknown"], ["one", "one"], [False], "one"):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    self.preparation.review(self.owner, status.job_id, names, expected_version=status.version)
        self.error(ErrorCode.CONFLICT, lambda: self.preparation.review(
            self.owner, status.job_id, ["one"], expected_version=status.version + 1
        ))
        self.assertEqual(self.store.get(self.owner, status.job_id).version, status.version)

    def test_foreign_owner_and_tenant_cannot_inspect_or_review(self):
        status, _ = self.review_job()
        for principal in (Principal("other", "operator-a"), Principal("local", "other")):
            self.error(ErrorCode.NOT_FOUND, lambda: self.preparation.inspect(principal, status.job_id))
            self.error(ErrorCode.NOT_FOUND, lambda: self.preparation.review(
                principal, status.job_id, ["one"], expected_version=status.version
            ))

    def test_review_expiry_is_enforced_at_exact_deadline(self):
        self.settings = ServiceSettings(review_ttl_seconds=20)
        self.store = JobStore(self.path, self.settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        self.requests = JobRequests(self.scheduler)
        self.preparation = JobPreparation(self.scheduler, self.preparer)
        status, _ = self.review_job()
        self.now += timedelta(seconds=20)
        self.error(ErrorCode.DEADLINE_EXCEEDED, lambda: self.preparation.inspect(self.owner, status.job_id))
        self.scheduler.recover()
        self.assertEqual(self.store.get(self.owner, status.job_id).state, JobState.EXPIRED)

    def test_cancelled_review_cannot_resume(self):
        status, _ = self.review_job()
        self.store.cancel(self.owner, status.job_id, expected_version=status.version)
        self.error(ErrorCode.CONFLICT, lambda: self.preparation.review(
            self.owner, status.job_id, ["one"], expected_version=status.version
        ))

    def test_same_conversation_waits_while_another_conversation_can_run(self):
        conversation = "a" * 32
        self.review_job(conversation=conversation)
        self.submit("later", conversation)
        independent = self.submit("independent")
        self.assertEqual(self.scheduler.claim("worker-b").status.job_id, independent.job_id)

    def test_pending_review_limit_is_enforced_without_losing_live_claim(self):
        self.settings = ServiceSettings(max_pending_reviews=1)
        self.store = JobStore(self.path, self.settings, clock=lambda: self.now)
        self.scheduler = JobScheduler(self.store)
        self.requests = JobRequests(self.scheduler)
        self.preparation = JobPreparation(self.scheduler, self.preparer)
        self.review_job()
        self.submit("two")
        claim = self.scheduler.claim("worker-two")
        self.error(ErrorCode.QUEUE_FULL, lambda: self.preparation.prepare(claim))
        self.assertEqual(self.scheduler.heartbeat(claim).status.state, JobState.PREPARING)

    def test_competing_review_updates_accept_only_one_version(self):
        status, _ = self.review_job()
        barrier = Barrier(2)
        def recount(messages):
            barrier.wait(timeout=5)
            return 70
        self.counter.side_effect = recount
        def review(name):
            try:
                self.preparation.review(self.owner, status.job_id, [name], expected_version=status.version)
                return "accepted"
            except ServiceError as exc:
                self.assertEqual(exc.code, ErrorCode.CONFLICT)
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(review, ("one", "two")))
        self.assertCountEqual(results, ["accepted", "conflict"])

    def test_preparation_write_failure_rolls_back_state_and_lease_revocation(self):
        self.submit()
        claim = self.scheduler.claim("worker-a")
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TRIGGER reject_preparation BEFORE INSERT ON service_preparations
                BEGIN SELECT RAISE(ABORT, 'simulated failure'); END
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.preparation.prepare(claim)
        self.assertEqual(self.scheduler.heartbeat(claim).status.state, JobState.PREPARING)

    def test_raw_prompt_that_cannot_fit_is_not_silently_approved(self):
        self.router.search.return_value = []
        self.counter.side_effect = None
        self.counter.return_value = 101
        with self.assertRaises(PromptTooLargeError):
            self.preparer.prepare("Question")

    def test_request_expiry_removes_preparation_snapshot(self):
        self.review_job()
        self.now += timedelta(seconds=self.settings.content_retention_seconds)
        self.requests.prune_expired()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM service_preparations").fetchone()[0], 0)

    def test_schema_five_migration_preserves_result_prompt_and_live_hold(self):
        self.router.search.return_value = []
        self.submit()
        first = self.scheduler.claim("worker-a")
        self.preparation.prepare(first)
        gate = InferenceGate(self.scheduler)
        permit = gate.start(first, reserved_bytes=10)
        results = JobResults(gate)
        results.capture(permit, ResultSnapshot("Answer"), inference_stopped=True)
        second = self.submit("two")
        claim = self.scheduler.claim("worker-b")
        self.preparation.prepare(claim)
        held = gate.start(claim, reserved_bytes=10)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE service_preparations")
            connection.execute("PRAGMA user_version = 5")
        migrated = JobStore(self.path, self.settings, clock=lambda: self.now)
        reopened_gate = InferenceGate(JobScheduler(migrated))
        self.assertEqual(reopened_gate.held_permits()[0].token, held.token)
        self.assertEqual(JobRequests(reopened_gate.scheduler).get(self.owner, second.job_id).prompt, "Question")
        self.assertEqual(JobResults(reopened_gate).recover(), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_failed_schema_six_migration_rolls_back_all_existing_data(self):
        original = self.submit()
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE service_preparations")
            connection.execute("PRAGMA user_version = 5")
        with patch("service_job_store._PREPARATION_SCHEMA", "INVALID SQL"):
            with self.assertRaises(sqlite3.OperationalError):
                JobStore(self.path, self.settings)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
            self.assertEqual(connection.execute("SELECT job_id FROM service_jobs").fetchone()[0], original.job_id)
            self.assertEqual(connection.execute("SELECT prompt FROM service_request_content").fetchone()[0], "Question")

    def test_changed_execution_profile_cannot_resume_an_old_review(self):
        status, _ = self.review_job()
        changed = ExecutionProfile("default", "changed-model", "gpu", 130, 20, 10)
        preparer = PromptPreparer(self.router, self.registry, self.counter, changed)
        service = JobPreparation(self.scheduler, preparer)
        self.error(ErrorCode.CONFLICT, lambda: service.review(
            self.owner, status.job_id, ["one"], expected_version=status.version
        ))

    def test_import_does_not_load_analyst_or_embedding_models(self):
        code = (
            "import sys; import analyst_preparation, service_preparation; "
            "assert not ({'analyst','config','orchestrator','torch','sentence_transformers'} & set(sys.modules))"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ANALYST_DIR,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
