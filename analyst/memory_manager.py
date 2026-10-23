"""Optional conversation-memory operations.

Stored messages are untrusted data. This module does not assemble prompts,
execute stored content, or perform semantic retrieval.
"""

from __future__ import annotations

from pathlib import Path

from token_usage import TokenUsage

from memory_store import (
    DEFAULT_CONVERSATION_TITLE,
    Conversation,
    MemoryStore,
    Message,
)


class MemoryManager:
    """Optional façade over a store bound to one configured scope.

    Disabled operations return empty results and never access storage.
    Enabled operations propagate validation and storage errors so callers
    cannot mistake a failed write for successful persistence.

    The scope must come from trusted application configuration. It must not
    be taken from an unauthenticated request's owner or workspace field.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        database_path: str | Path | None = None,
        scope: str | None = None,
        retention_days: int | None = None,
        busy_timeout_ms: int = 5000,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")

        self._store: MemoryStore | None = None

        if not enabled:
            return

        if database_path is None:
            raise ValueError("database_path is required when memory is enabled.")

        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("A configured scope is required when memory is enabled.")

        self._store = MemoryStore(
            database_path,
            scope=scope,
            retention_days=retention_days,
            busy_timeout_ms=busy_timeout_ms,
        )

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def create_conversation(
        self,
        *,
        title: str = DEFAULT_CONVERSATION_TITLE,
    ) -> Conversation | None:
        if self._store is None:
            return None

        return self._store.create_conversation(title=title)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        if self._store is None:
            return None

        return self._store.get_conversation(conversation_id)

    def rename_conversation(
        self,
        conversation_id: str,
        *,
        title: str,
    ) -> Conversation | None:
        if self._store is None:
            return None

        return self._store.rename_conversation(
            conversation_id,
            title=title,
        )

    def list_conversations(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[Conversation]:
        if self._store is None:
            return []

        return self._store.list_conversations(
            limit=limit,
            offset=offset,
        )

    def get_messages(
        self,
        conversation_id: str,
        *,
        recent_exchanges: int | None = None,
    ) -> list[Message]:
        if self._store is None:
            return []

        return self._store.get_messages(
            conversation_id,
            recent_exchanges=recent_exchanges,
        )

    def record_completed_exchange(
        self,
        conversation_id: str,
        *,
        user_content: str,
        assistant_content: str,
        usage: TokenUsage | None = None,
        job_id: str | None = None,
    ) -> tuple[Message, Message] | None:
        """Persist a completed exchange after successful inference.

        A supplied job ID makes identical retries return the original pair.
        Without a job ID, existing append behavior is preserved.

        Disabled memory performs no storage operation.
        """
        if self._store is None:
            return None

        arguments = {}
        if usage is not None:
            arguments["usage"] = usage
        if job_id is not None:
            arguments["job_id"] = job_id

        return self._store.append_exchange(
            conversation_id,
            user_content=user_content,
            assistant_content=assistant_content,
            **arguments,
        )

    def get_conversation_usage(self, conversation_id: str) -> dict | None:
        if self._store is None:
            return None
        return self._store.get_conversation_usage(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Return True after deletion, or False when memory is disabled.

        With memory enabled, a missing or inaccessible conversation raises
        ConversationNotFoundError, preserving the store's isolation behavior.
        """
        if self._store is None:
            return False

        self._store.delete_conversation(conversation_id)
        return True

    def prune_expired_conversations(self) -> int:
        if self._store is None:
            return 0

        return self._store.prune_expired_conversations()
