"""Service Batch 1 contracts. Run after the application modules are applied."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ANALYST_DIR = Path(__file__).resolve().parents[1] / "analyst"
sys.path.insert(0, str(ANALYST_DIR))

from service_contracts import (
    CancelAction,
    ErrorCode,
    ExecutionProfile,
    HardwareTarget,
    InvalidTransitionError,
    JobState,
    JobStatus,
    JobSubmission,
    Principal,
    ServiceError,
    cancellation_action,
    validate_transition,
)
from service_settings import ServiceSettings, load_service_settings


class JobContractTests(unittest.TestCase):
    def status(self, **changes):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        arguments = {
            "job_id": "a" * 32,
            "principal": Principal("tenant-a", "operator-a"),
            "state": JobState.QUEUED,
            "version": 1,
            "created_at": created,
            "updated_at": created,
            "expires_at": created + timedelta(minutes=10),
        }
        arguments.update(changes)
        return JobStatus(**arguments)

    def test_submission_preserves_prompt_and_normalizes_absent_conversation(self):
        prompt = "  Investigate Event 4625.\n"
        submission = JobSubmission.from_payload(
            {"prompt": prompt}, idempotency_key="request-1"
        )
        self.assertEqual(submission.prompt, prompt)
        self.assertIsNone(submission.conversation_id)
        self.assertNotIn(prompt, repr(submission))

    def test_client_cannot_supply_owner_scope_or_execution_profile(self):
        for field in ("owner_id", "tenant_id", "scope", "execution_profile_id"):
            with self.subTest(field=field):
                with self.assertRaises(ServiceError) as caught:
                    JobSubmission.from_payload(
                        {"prompt": "Investigate", field: "other"},
                        idempotency_key="request-1",
                    )
                self.assertEqual(caught.exception.code, ErrorCode.INVALID_REQUEST)

    def test_invalid_submissions_are_rejected_before_admission(self):
        for payload in (
            [], {}, {"prompt": "   "}, {"prompt": False},
            {"prompt": "x", "conversation_id": "bad"},
            {"prompt": "x", "conversation_id": False},
            {"prompt": "\ud800"},
        ):
            with self.subTest(payload=repr(payload)):
                with self.assertRaises(ServiceError):
                    JobSubmission.from_payload(payload, idempotency_key="request-1")
        with self.assertRaises(ServiceError):
            JobSubmission.from_payload(
                {"prompt": "abcd"}, idempotency_key="request-1",
                max_prompt_characters=3,
            )
        for key in ("", "has spaces", "x" * 129, False):
            with self.subTest(key=key):
                with self.assertRaises(ServiceError):
                    JobSubmission.from_payload({"prompt": "x"}, idempotency_key=key)

    def test_request_fingerprint_is_about_payload_not_retry_key(self):
        first = JobSubmission.from_payload({"prompt": "x"}, idempotency_key="one")
        retry = JobSubmission.from_payload(
            {"conversation_id": None, "prompt": "x"}, idempotency_key="two"
        )
        changed = JobSubmission.from_payload({"prompt": "y"}, idempotency_key="one")
        self.assertEqual(first.fingerprint(), retry.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())
        self.assertEqual(len(first.fingerprint()), 64)

    def test_owner_and_tenant_are_both_required_for_visibility(self):
        status = self.status()
        public = status.public_status(Principal("tenant-a", "operator-a"))
        self.assertEqual(public["job_id"], "a" * 32)
        self.assertNotIn("principal", public)
        for stranger in (
            Principal("tenant-a", "operator-b"),
            Principal("tenant-b", "operator-a"),
        ):
            with self.assertRaises(ServiceError) as caught:
                status.public_status(stranger)
            self.assertEqual(caught.exception.code, ErrorCode.NOT_FOUND)
            self.assertEqual(caught.exception.status_code, 404)

    def test_utc_serialization_and_timestamp_ordering(self):
        local = datetime(2026, 1, 1, 5, tzinfo=timezone(timedelta(hours=5)))
        status = self.status(
            created_at=local, updated_at=local,
            expires_at=local + timedelta(minutes=10),
        )
        public = status.public_status(status.principal)
        self.assertEqual(public["created_at"], "2026-01-01T00:00:00.000000Z")
        json.dumps(public, allow_nan=False)
        for changes in (
            {"created_at": datetime(2026, 1, 1)},
            {"updated_at": local - timedelta(seconds=1)},
            {"expires_at": local},
            {"version": True},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.status(**changes)

    def test_normal_flow_and_review_resume_are_allowed(self):
        pairs = (
            (JobState.QUEUED, JobState.PREPARING),
            (JobState.PREPARING, JobState.AWAITING_REVIEW),
            (JobState.AWAITING_REVIEW, JobState.QUEUED),
            (JobState.PREPARING, JobState.WAITING_INFERENCE),
            (JobState.WAITING_INFERENCE, JobState.RUNNING),
            (JobState.FINALIZING, JobState.COMPLETED),
        )
        for source, target in pairs:
            validate_transition(source, target)

    def test_running_work_cannot_be_declared_stopped_without_confirmation(self):
        for target in (JobState.FINALIZING, JobState.CANCELLED, JobState.FAILED):
            with self.subTest(target=target):
                with self.assertRaises(InvalidTransitionError):
                    validate_transition(JobState.RUNNING, target)
                validate_transition(JobState.RUNNING, target, inference_stopped=True)
        validate_transition(JobState.RUNNING, JobState.INTERRUPTED)
        with self.assertRaises(InvalidTransitionError):
            validate_transition(JobState.RUNNING, JobState.QUEUED)

    def test_terminal_states_cannot_be_requeued_or_overwritten(self):
        for state in (
            JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED,
            JobState.EXPIRED, JobState.INTERRUPTED,
        ):
            with self.subTest(state=state):
                with self.assertRaises(InvalidTransitionError):
                    validate_transition(state, JobState.QUEUED)
                with self.assertRaises(InvalidTransitionError):
                    validate_transition(state, state)

    def test_cancellation_request_is_not_a_claim_that_inference_stopped(self):
        self.assertEqual(cancellation_action(JobState.QUEUED), CancelAction.CANCEL_NOW)
        self.assertEqual(cancellation_action(JobState.RUNNING), CancelAction.REQUEST_STOP)
        self.assertEqual(cancellation_action(JobState.FINALIZING), CancelAction.TOO_LATE)
        self.assertEqual(
            cancellation_action(JobState.COMPLETED), CancelAction.ALREADY_TERMINAL
        )

    def test_execution_profiles_validate_context_without_hardware_guesses(self):
        profile = ExecutionProfile(
            "gb10-default", "configured-model", "primary-gpu",
            context_size=13312, max_output_tokens=1024, safety_margin=512,
            hardware_target=HardwareTarget.GB10,
        )
        self.assertEqual(profile.input_token_budget, 11776)
        with self.assertRaises(ValueError):
            ExecutionProfile("bad", "model", "pool", 100, 80, 20)
        with self.assertRaises(ValueError):
            ExecutionProfile("bad", "model", "pool", True, 1)

    def test_error_payloads_are_stable_and_do_not_accept_raw_exception_text(self):
        error = ServiceError(ErrorCode.QUEUE_FULL, retry_after_seconds=3)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.as_payload()["error"]["code"], "queue_full")
        self.assertEqual(error.as_payload()["error"]["retry_after_seconds"], 3)
        with self.assertRaises(ValueError):
            ServiceError(ErrorCode.INTERNAL_ERROR, retry_after_seconds=True)


class ServiceSettingsTests(unittest.TestCase):
    def test_safe_defaults_and_gb10_label_do_not_raise_capacity(self):
        current = load_service_settings({})
        gb10 = load_service_settings({"SERVICE_HARDWARE_TARGET": "gb10"})
        for settings in (current, gb10):
            self.assertEqual(settings.max_active_jobs, 1)
            self.assertEqual(settings.max_inference_requests, 1)
            self.assertIsNone(settings.inference_memory_budget_bytes)
        self.assertEqual(gb10.hardware_target, HardwareTarget.GB10)

    def test_future_capacity_requires_explicit_settings(self):
        settings = load_service_settings({
            "SERVICE_MAX_ACTIVE_JOBS": "4",
            "SERVICE_WORKER_CONCURRENCY": "2",
            "SERVICE_MAX_INFERENCE_REQUESTS": "2",
            "SERVICE_OWNER_MAX_ACTIVE": "2",
            "SERVICE_INFERENCE_MEMORY_BUDGET_BYTES": "1000000000",
        })
        self.assertEqual(settings.max_active_jobs, 4)
        self.assertEqual(settings.max_inference_requests, 2)
        self.assertEqual(settings.inference_memory_budget_bytes, 1000000000)

    def test_invalid_capacity_and_lease_settings_are_rejected(self):
        for changes in (
            {"max_active_jobs": True}, {"max_inference_requests": 2},
            {"owner_max_waiting": 17}, {"worker_lease_seconds": 10},
            {"job_deadline_seconds": 5}, {"inference_memory_budget_bytes": -1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    ServiceSettings(**changes)
        for environment in (
            {"SERVICE_MAX_ACTIVE_JOBS": "NaN"},
            {"SERVICE_HARDWARE_TARGET": "unknown"},
            {"SERVICE_LOCAL_OWNER_ID": ""},
        ):
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    load_service_settings(environment)

    def test_contract_imports_do_not_load_analyst_models_or_current_config(self):
        code = (
            "import sys; import service_contracts, service_settings; "
            "assert not ({'analyst','orchestrator','torch',"
            "'sentence_transformers','config'} & set(sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=ANALYST_DIR,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
