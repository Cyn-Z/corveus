"""Model-independent contracts for the future Web/API and worker services.

These contracts validate values and legal state transitions. They do not
implement authentication, scheduling, leases, persistence, or resource release.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}")
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_OBJECT_ID = re.compile(r"[0-9a-f]{32}")

MAX_PROMPT_CHARACTERS = 1_000_000
MAX_PROMPT_BYTES = 4_000_000


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a valid service identifier.")


def _object_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must contain 32 lowercase hexadecimal characters.")


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a timezone-aware datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime.")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


class HardwareTarget(str, Enum):
    GENERIC = "generic"
    GB10 = "gb10"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    QUEUE_FULL = "queue_full"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    EXECUTION_INTERRUPTED = "execution_interrupted"
    INTERNAL_ERROR = "internal_error"
    CONTENT_EXPIRED = "content_expired"


_ERROR_DETAILS = {
    ErrorCode.INVALID_REQUEST: (
        400, "The request is invalid.", False
    ),
    ErrorCode.NOT_FOUND: (
        404, "The requested resource was not found.", False
    ),
    ErrorCode.CONFLICT: (
        409, "The request conflicts with the current resource state.", False
    ),
    ErrorCode.QUEUE_FULL: (
        429, "The service queue is full. Try again later.", True
    ),
    ErrorCode.SERVICE_UNAVAILABLE: (
        503, "The service cannot accept this request right now.", True
    ),
    ErrorCode.DEADLINE_EXCEEDED: (
        408, "The job deadline was exceeded.", False
    ),
    ErrorCode.EXECUTION_INTERRUPTED: (
        503, "Execution was interrupted. Check the job before resubmitting.", False
    ),
    ErrorCode.CONTENT_EXPIRED: (
        410, "The job content is no longer available.", False
    ),
    ErrorCode.INTERNAL_ERROR: (
        500, "The service could not complete the operation.", False
    ),
}


class ServiceError(Exception):
    """A stable public error that cannot contain arbitrary exception text."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        if not isinstance(code, ErrorCode):
            raise ValueError("code must be an ErrorCode.")

        status_code, message, retryable = _ERROR_DETAILS[code]

        if retry_after_seconds is not None:
            _integer(retry_after_seconds, "retry_after_seconds")
            if not retryable:
                raise ValueError("This error does not support retry advice.")

        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)

    def as_payload(self) -> dict:
        error = {
            "code": self.code.value,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.retry_after_seconds is not None:
            error["retry_after_seconds"] = self.retry_after_seconds
        return {"error": error}


@dataclass(frozen=True)
class Principal:
    """Trusted server-established identity, not an authentication mechanism."""

    tenant_id: str
    owner_id: str

    def __post_init__(self) -> None:
        _identifier(self.tenant_id, "tenant_id")
        _identifier(self.owner_id, "owner_id")


@dataclass(frozen=True)
class JobSubmission:
    prompt: str = field(repr=False)
    idempotency_key: str = field(repr=False)
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string.")
        if len(self.prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError("prompt exceeds the absolute character limit.")

        try:
            encoded = self.prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("prompt must contain valid Unicode.") from exc

        if len(encoded) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the absolute byte limit.")

        if (
            not isinstance(self.idempotency_key, str)
            or _IDEMPOTENCY_KEY.fullmatch(self.idempotency_key) is None
        ):
            raise ValueError("idempotency_key is invalid.")

        if self.conversation_id is not None:
            _object_id(self.conversation_id, "conversation_id")

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        idempotency_key: str,
        max_prompt_characters: int = 200_000,
    ) -> JobSubmission:
        """Parse client input without accepting client-supplied identity.

        Configuration errors raise ValueError. Invalid client input raises
        ServiceError(INVALID_REQUEST).
        """
        _integer(max_prompt_characters, "max_prompt_characters")
        if max_prompt_characters > MAX_PROMPT_CHARACTERS:
            raise ValueError("max_prompt_characters exceeds the absolute limit.")

        if type(payload) is not dict:
            raise ServiceError(ErrorCode.INVALID_REQUEST)

        if "prompt" not in payload or set(payload) - {"prompt", "conversation_id"}:
            raise ServiceError(ErrorCode.INVALID_REQUEST)

        prompt = payload["prompt"]
        if not isinstance(prompt, str) or len(prompt) > max_prompt_characters:
            raise ServiceError(ErrorCode.INVALID_REQUEST)

        try:
            return cls(
                prompt=prompt,
                idempotency_key=idempotency_key,
                conversation_id=payload.get("conversation_id"),
            )
        except ValueError:
            raise ServiceError(ErrorCode.INVALID_REQUEST) from None

    def fingerprint(self) -> str:
        """Canonical payload digest; storage must scope keys by Principal."""
        canonical = json.dumps(
            {
                "conversation_id": self.conversation_id,
                "prompt": self.prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JobState(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    AWAITING_REVIEW = "awaiting_review"
    WAITING_INFERENCE = "waiting_inference"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL_STATES


_TERMINAL_STATES = frozenset({
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.CANCELLED,
    JobState.EXPIRED,
    JobState.INTERRUPTED,
})

_TRANSITIONS = {
    JobState.QUEUED: frozenset({
        JobState.PREPARING,
        JobState.CANCELLED,
        JobState.EXPIRED,
    }),
    JobState.PREPARING: frozenset({
        JobState.AWAITING_REVIEW,
        JobState.WAITING_INFERENCE,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.EXPIRED,
        JobState.QUEUED,
    }),
    JobState.AWAITING_REVIEW: frozenset({
        JobState.QUEUED,
        JobState.CANCELLED,
        JobState.EXPIRED,
    }),
    JobState.WAITING_INFERENCE: frozenset({
        JobState.RUNNING,
        JobState.CANCELLED,
        JobState.EXPIRED,
        JobState.FAILED,
        JobState.QUEUED,
    }),
    JobState.RUNNING: frozenset({
        JobState.FINALIZING,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.INTERRUPTED,
    }),
    JobState.FINALIZING: frozenset({
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.INTERRUPTED,
    }),
}


class InvalidTransitionError(ServiceError):
    def __init__(self, source: JobState, target: JobState) -> None:
        self.source = source
        self.target = target
        super().__init__(ErrorCode.CONFLICT)


def validate_transition(
    source: JobState,
    target: JobState,
    *,
    inference_stopped: bool = False,
) -> None:
    """Validate state legality, not authority or resource ownership.

    Running work requires confirmation that inference stopped before being
    marked finalizing, failed, or cancelled.

    INTERRUPTED records uncertainty. It never authorizes releasing inference
    capacity or automatically retrying the request.

    Pre-inference requeue transitions require transactional lease/attempt
    fencing in the future scheduler.
    """
    if not isinstance(source, JobState) or not isinstance(target, JobState):
        raise ValueError("source and target must be JobState values.")
    if type(inference_stopped) is not bool:
        raise ValueError("inference_stopped must be a boolean.")

    if target not in _TRANSITIONS.get(source, frozenset()):
        raise InvalidTransitionError(source, target)

    if (
        source is JobState.RUNNING
        and target is not JobState.INTERRUPTED
        and not inference_stopped
    ):
        raise InvalidTransitionError(source, target)


class CancelAction(str, Enum):
    CANCEL_NOW = "cancel_now"
    REQUEST_STOP = "request_stop"
    TOO_LATE = "too_late"
    ALREADY_TERMINAL = "already_terminal"


def cancellation_action(state: JobState) -> CancelAction:
    """Return a cancellation decision; perform no mutation."""
    if not isinstance(state, JobState):
        raise ValueError("state must be a JobState.")

    if state.terminal:
        return CancelAction.ALREADY_TERMINAL
    if state is JobState.RUNNING:
        return CancelAction.REQUEST_STOP
    if state is JobState.FINALIZING:
        return CancelAction.TOO_LATE
    return CancelAction.CANCEL_NOW


@dataclass(frozen=True)
class ExecutionProfile:
    """Server-selected model limits and inference resource association."""

    profile_id: str
    model_id: str
    inference_pool_id: str
    context_size: int
    max_output_tokens: int
    safety_margin: int = 0
    hardware_target: HardwareTarget = HardwareTarget.GENERIC

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "profile_id")
        _identifier(self.inference_pool_id, "inference_pool_id")

        if (
            not isinstance(self.model_id, str)
            or not self.model_id.strip()
            or len(self.model_id) > 512
            or any(not character.isprintable() for character in self.model_id)
        ):
            raise ValueError("model_id must be a nonempty printable identifier.")

        _integer(self.context_size, "context_size")
        _integer(self.max_output_tokens, "max_output_tokens")
        _integer(self.safety_margin, "safety_margin", minimum=0)

        if not isinstance(self.hardware_target, HardwareTarget):
            raise ValueError("hardware_target must be a HardwareTarget.")
        if self.input_token_budget <= 0:
            raise ValueError("The execution profile must leave input capacity.")

    @property
    def input_token_budget(self) -> int:
        return self.context_size - self.max_output_tokens - self.safety_margin


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    principal: Principal = field(repr=False)
    state: JobState
    version: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    conversation_id: str | None = None
    cancel_requested: bool = False
    execution_profile_id: str = "default"

    def __post_init__(self) -> None:
        _object_id(self.job_id, "job_id")
        if not isinstance(self.principal, Principal):
            raise ValueError("principal must be a Principal.")
        if not isinstance(self.state, JobState):
            raise ValueError("state must be a JobState.")
        _integer(self.version, "version")

        if self.conversation_id is not None:
            _object_id(self.conversation_id, "conversation_id")
        if type(self.cancel_requested) is not bool:
            raise ValueError("cancel_requested must be a boolean.")
        _identifier(self.execution_profile_id, "execution_profile_id")

        created = _utc(self.created_at, "created_at")
        updated = _utc(self.updated_at, "updated_at")
        expires = _utc(self.expires_at, "expires_at")

        if updated < created:
            raise ValueError("updated_at cannot precede created_at.")
        if expires <= created:
            raise ValueError("expires_at must follow created_at.")

        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "expires_at", expires)

    def public_status(self, requester: Principal) -> dict:
        """Expose metadata only to the matching server-established principal."""
        if not isinstance(requester, Principal) or requester != self.principal:
            raise ServiceError(ErrorCode.NOT_FOUND)

        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "version": self.version,
            "created_at": _timestamp(self.created_at),
            "updated_at": _timestamp(self.updated_at),
            "expires_at": _timestamp(self.expires_at),
            "conversation_id": self.conversation_id,
            "cancel_requested": self.cancel_requested,
            "execution_profile_id": self.execution_profile_id,
            "terminal": self.state.terminal,
        }