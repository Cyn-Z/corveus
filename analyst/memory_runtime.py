"""Application startup construction for optional conversation memory."""

from __future__ import annotations

from memory_config import MemorySettings
from memory_manager import MemoryManager


def build_memory_manager(settings: MemorySettings) -> MemoryManager:
    """Construct memory storage using trusted application configuration.

    Enabled storage failures propagate to startup. Memory is never silently
    disabled after initialization fails.

    This function does not create conversations, record exchanges, prune
    records, or assemble prompts.
    """
    return MemoryManager(
        enabled=settings.enabled,
        database_path=settings.database_path,
        scope=settings.scope,
        retention_days=settings.retention_days,
        busy_timeout_ms=settings.busy_timeout_ms,
    )
