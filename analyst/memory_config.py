"""Configuration for optional local conversation persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = False
    database_path: Path = Path("/app/data/memory.sqlite3")
    scope: str = "local-operator"
    retention_days: int | None = None
    busy_timeout_ms: int = 5000

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("MEMORY_ENABLED must be a boolean.")

        if not isinstance(self.database_path, Path):
            raise ValueError("MEMORY_DATABASE_PATH must be a Path.")

        if not self.database_path.is_absolute():
            raise ValueError("MEMORY_DATABASE_PATH must be absolute.")

        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("MEMORY_SCOPE must be non-empty.")

        if self.retention_days is not None:
            self._require_positive_integer(self.retention_days, "MEMORY_RETENTION_DAYS")

        self._require_positive_integer(self.busy_timeout_ms, "MEMORY_BUSY_TIMEOUT_MS")

    @staticmethod
    def _require_positive_integer(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def load_memory_settings(
    environment: Mapping[str, str] | None = None,
) -> MemorySettings:
    """Read configuration without creating directories or opening SQLite.

    A missing variable receives its default. Explicit empty values are
    rejected except MEMORY_RETENTION_DAYS, where empty means no retention
    limit.
    """
    values = os.environ if environment is None else environment

    enabled_text = values.get("MEMORY_ENABLED", "false").strip().lower()
    if enabled_text not in {"true", "false"}:
        raise ValueError("MEMORY_ENABLED must be 'true' or 'false'.")

    path_text = values.get("MEMORY_DATABASE_PATH", "/app/data/memory.sqlite3").strip()
    if not path_text or path_text == ":memory:":
        raise ValueError(
            "MEMORY_DATABASE_PATH must be an absolute persistent file path."
        )

    scope = values.get("MEMORY_SCOPE", "local-operator").strip()
    retention_text = values.get("MEMORY_RETENTION_DAYS", "").strip()
    timeout_text = values.get("MEMORY_BUSY_TIMEOUT_MS", "5000").strip()

    try:
        retention_days = int(retention_text) if retention_text else None
    except ValueError as exc:
        raise ValueError(
            "MEMORY_RETENTION_DAYS must be empty or a positive integer."
        ) from exc

    try:
        busy_timeout_ms = int(timeout_text)
    except ValueError as exc:
        raise ValueError("MEMORY_BUSY_TIMEOUT_MS must be a positive integer.") from exc

    return MemorySettings(
        enabled=enabled_text == "true",
        database_path=Path(path_text),
        scope=scope,
        retention_days=retention_days,
        busy_timeout_ms=busy_timeout_ms,
    )
