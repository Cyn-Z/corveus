"""Settings for the future service boundary.

Importing this module does not load application configuration or models.
Settings take effect only when explicitly consumed by future service code.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields

from service_contracts import (
    HardwareTarget,
    MAX_PROMPT_CHARACTERS,
    Principal,
)


_ENV_INTEGER = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class ServiceSettings:
    hardware_target: HardwareTarget = HardwareTarget.GENERIC
    local_tenant_id: str = "local"
    local_owner_id: str = "local-operator"

    # Global limits must eventually be enforced by the central scheduler.
    max_active_jobs: int = 1
    max_inference_requests: int = 1
    max_waiting_jobs: int = 16

    # Per-process worker limit; never overrides global admission.
    worker_concurrency: int = 1

    owner_max_active: int = 1
    owner_max_waiting: int = 4
    max_pending_reviews: int = 4

    heartbeat_seconds: int = 5
    worker_lease_seconds: int = 30
    job_deadline_seconds: int = 600
    review_ttl_seconds: int = 600
    max_prompt_characters: int = 200_000
    content_retention_seconds: int = 3600

    # None means no configured memory budget, not unlimited safe capacity.
    inference_memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hardware_target, HardwareTarget):
            raise ValueError("hardware_target must be a HardwareTarget.")

        Principal(self.local_tenant_id, self.local_owner_id)

        for name in _INTEGER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        budget = self.inference_memory_budget_bytes
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError(
                "inference_memory_budget_bytes must be a positive integer or None."
            )

        if self.max_inference_requests > self.max_active_jobs:
            raise ValueError(
                "max_inference_requests cannot exceed max_active_jobs."
            )
        if self.worker_concurrency > self.max_active_jobs:
            raise ValueError(
                "worker_concurrency cannot exceed max_active_jobs."
            )
        if self.owner_max_active > self.max_active_jobs:
            raise ValueError(
                "owner_max_active cannot exceed max_active_jobs."
            )
        if self.owner_max_waiting > self.max_waiting_jobs:
            raise ValueError(
                "owner_max_waiting cannot exceed max_waiting_jobs."
            )
        if self.worker_lease_seconds < 3 * self.heartbeat_seconds:
            raise ValueError(
                "worker_lease_seconds must cover at least three heartbeats."
            )
        if self.job_deadline_seconds <= self.worker_lease_seconds:
            raise ValueError(
                "job_deadline_seconds must exceed worker_lease_seconds."
            )
        if self.max_prompt_characters > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                "max_prompt_characters exceeds the absolute character limit."
            )
        if self.content_retention_seconds <= self.job_deadline_seconds:
            raise ValueError(
                "content_retention_seconds must exceed job_deadline_seconds."
            )

    @property
    def local_principal(self) -> Principal:
        return Principal(self.local_tenant_id, self.local_owner_id)


_INTEGER_FIELDS = frozenset({
    "max_active_jobs",
    "max_inference_requests",
    "max_waiting_jobs",
    "worker_concurrency",
    "owner_max_active",
    "owner_max_waiting",
    "max_pending_reviews",
    "heartbeat_seconds",
    "worker_lease_seconds",
    "job_deadline_seconds",
    "review_ttl_seconds",
    "max_prompt_characters",
    "content_retention_seconds",
})


def _read_integer(value: str, environment_name: str) -> int:
    if not isinstance(value, str) or _ENV_INTEGER.fullmatch(value) is None:
        raise ValueError(
            f"{environment_name} must contain an unsigned decimal integer."
        )
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{environment_name} contains an invalid integer.") from None


def load_service_settings(
    environment: Mapping[str, str] | None = None,
) -> ServiceSettings:
    """Read explicit SERVICE_* settings without changing the environment."""
    source = os.environ if environment is None else environment
    arguments = {}

    for definition in fields(ServiceSettings):
        name = definition.name
        environment_name = f"SERVICE_{name.upper()}"

        if environment_name not in source:
            continue

        raw = source[environment_name]
        if not isinstance(raw, str):
            raise ValueError(f"{environment_name} must be a string.")

        if name in _INTEGER_FIELDS:
            arguments[name] = _read_integer(raw, environment_name)
        elif name == "hardware_target":
            try:
                arguments[name] = HardwareTarget(raw)
            except ValueError:
                raise ValueError(
                    "SERVICE_HARDWARE_TARGET must be 'generic' or 'gb10'."
                ) from None
        elif name == "inference_memory_budget_bytes":
            arguments[name] = (
                None if raw == "" else _read_integer(raw, environment_name)
            )
        else:
            arguments[name] = raw

    return ServiceSettings(**arguments)