"""Local operator job API, independent of HTTP and model initialization.

This is a single configured principal, not multiuser authentication. Transport
headers and payloads never select an owner, tenant, or execution profile.
"""

from dataclasses import asdict

from memory_manager import MemoryManager
from memory_store import ConversationNotFoundError
from service_contracts import ErrorCode, JobSubmission, ServiceError
from service_requests import JobRequests
from service_results import JobResults
from service_scheduler import _identifier
from service_preparation import JobPreparation
from service_health import WorkerHealth

class LocalJobAPI:
    def __init__(
        self,
        results: JobResults,
        *,
        profile_id: str,
        memory=None,
        preparation: JobPreparation | None = None,
    ):
        if not isinstance(results, JobResults):
            raise ValueError("results must be JobResults.")
        _identifier(profile_id, "profile_id")
        if memory is not None and not isinstance(memory, MemoryManager):
            raise ValueError("memory must be MemoryManager or None.")
        if preparation is not None:
            if not isinstance(preparation, JobPreparation):
                raise ValueError("preparation must be JobPreparation or None.")
            if preparation.scheduler is not results.scheduler:
                raise ValueError("Review and results must share the scheduler.")
            if preparation.preparer.profile.profile_id != profile_id:
                raise ValueError("Review and API profiles must match.")

        self.results = results
        self.store = results.store
        self.requests = JobRequests(results.scheduler)
        self.principal = self.store.settings.local_principal
        self.profile_id = profile_id
        self.memory = memory
        self.preparation = preparation

    def submit(self, payload, *, idempotency_key):
        submission = JobSubmission.from_payload(
            payload,
            idempotency_key=idempotency_key,
            max_prompt_characters=self.store.settings.max_prompt_characters,
        )

        def authorize_new():
            if submission.conversation_id is None:
                return
            if self.memory is None or not self.memory.enabled:
                raise ServiceError(ErrorCode.SERVICE_UNAVAILABLE)
            try:
                conversation = self.memory.get_conversation(submission.conversation_id)
            except ConversationNotFoundError:
                raise ServiceError(ErrorCode.NOT_FOUND) from None
            if conversation is None:
                raise ServiceError(ErrorCode.NOT_FOUND)

        admission = self.requests.submit(
            self.principal, submission, execution_profile_id=self.profile_id,
            authorize_new=authorize_new,
        )
        return {
            "job": admission.status.public_status(self.principal),
            "created": admission.created,
        }

    def status(self, job_id):
        return {"job": self.store.get(self.principal, job_id).public_status(self.principal)}

    def router_available(self):
        return WorkerHealth(self.store.path).router_ready()

    def cancel(self, job_id, payload):
        if (
            type(payload) is not dict or set(payload) != {"expected_version"}
            or type(payload["expected_version"]) is not int
            or payload["expected_version"] < 1
        ):
            raise ServiceError(ErrorCode.INVALID_REQUEST)
        status = self.store.cancel(
            self.principal, job_id, expected_version=payload["expected_version"],
        )
        return {"job": status.public_status(self.principal)}

    def result(self, job_id):
        completed = self.results.get(self.principal, job_id)
        return {
            "job": completed.status.public_status(self.principal),
            "answer": completed.snapshot.answer,
            "usage": (
                None if completed.snapshot.usage is None
                else asdict(completed.snapshot.usage)
            ),
            "memory_persistence": completed.persistence_state,
        }
    def _review_preparation(self):
        if self.preparation is None:
            raise ServiceError(ErrorCode.SERVICE_UNAVAILABLE)
        return self.preparation

    def inspect_review(self, job_id):
        view = self._review_preparation().inspect(self.principal, job_id)
        return {
            "job": view.status.public_status(self.principal),
            "review": {
                "input_tokens": view.input_tokens,
                "input_budget": view.input_budget,
                "skills": [asdict(skill) for skill in view.skills],
                "expires_at": view.expires_at.isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z"),
            },
        }

    def apply_review(self, job_id, payload):
        if (
            type(payload) is not dict
            or set(payload) != {"expected_version", "remove_skill_names"}
        ):
            raise ServiceError(ErrorCode.INVALID_REQUEST)

        version = payload["expected_version"]
        names = payload["remove_skill_names"]
        if (
            type(version) is not int
            or version < 1
            or type(names) is not list
            or not 1 <= len(names) <= 32
            or any(
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 256
                or not name.isprintable()
                for name in names
            )
            or len(set(names)) != len(names)
        ):
            raise ServiceError(ErrorCode.INVALID_REQUEST)

        status = self._review_preparation().review(
            self.principal,
            job_id,
            names,
            expected_version=version,
        )
        return {"job": status.public_status(self.principal)}
