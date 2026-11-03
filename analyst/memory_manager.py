"""Optional conversation-memory operations.

Stored messages are untrusted data. Retrieval delegates to an independent
ranker; this module does not assemble prompts or execute stored content.
"""

from __future__ import annotations

from pathlib import Path

from token_usage import TokenUsage
from memory_retrieval_contracts import MemoryRetrievalPolicy, MemoryRetrievalResult

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
        retrieval_enabled: bool = False,
        retrieval_embeddings=None,
        indexing_enabled: bool | None = None,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")

        if type(retrieval_enabled) is not bool:
            raise ValueError("retrieval_enabled must be a boolean.")
        self._retrieval_enabled = retrieval_enabled
        self._retrieval_embeddings = retrieval_embeddings
        if indexing_enabled is not None and type(indexing_enabled) is not bool:
            raise ValueError("indexing_enabled must be boolean or None.")
        self._indexing_enabled = retrieval_enabled if indexing_enabled is None else indexing_enabled
        self._indexer = None
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

    def retrieve(
        self,
        query: str,
        *,
        current_conversation_id: str | None,
        policy: MemoryRetrievalPolicy,
    ) -> MemoryRetrievalResult:
        """Scoped retrieval; disabled/incognito calls perform no storage operation.

        Embeddings are a trusted constructor dependency, never request content.
        No scope, owner, or replacement backend is accepted from a caller.
        """
        if self._store is None or not self._retrieval_enabled:
            return MemoryRetrievalResult(status="disabled")
        if current_conversation_id is None:
            return MemoryRetrievalResult(status="incognito")
        if not isinstance(policy, MemoryRetrievalPolicy):
            raise ValueError("policy must be a MemoryRetrievalPolicy.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text.")
        from memory_retrieval_store import MemoryRetrievalStore
        from memory_retrieval import MemoryRetriever
        return MemoryRetriever(
            MemoryRetrievalStore(self._store), embeddings=self._retrieval_embeddings,
        ).retrieve(query, current_conversation_id=current_conversation_id, policy=policy)

    def bind_retrieval_embeddings(self, embeddings):
        """Trusted startup injection only; never enables retrieval or builds an index."""
        from retrieval_embeddings import QueryEmbeddings
        if embeddings is not None and not isinstance(embeddings, QueryEmbeddings):
            raise ValueError("embeddings must be a QueryEmbeddings adapter or None.")
        self._retrieval_embeddings = embeddings

    def retrieval_status(self):
        """Scoped counts only; no message text, tokenization, or model calls."""
        if self._store is None or not self._retrieval_enabled:
            return {"enabled": False, "state": "disabled"}
        if not self._indexing_enabled:
            return {"enabled": False, "state": "paused"}
        from memory_retrieval_store import MemoryRetrievalStore
        from memory_indexer import work_fingerprint
        repository = MemoryRetrievalStore(self._store)
        with self._store._transaction() as connection:
            cutoff = repository._cutoff()
            state = connection.execute('SELECT * FROM memory_index_state WHERE scope = ?',
                                       (self._store.scope,)).fetchone()
            if state is None:
                return {"enabled": True, "state": "waiting"}
            fingerprint = work_fingerprint(state)
            total = connection.execute('''SELECT count(*) FROM messages m
                JOIN conversations c ON c.id = m.conversation_id WHERE c.scope = ? AND c.updated_at >= ?''',
                (self._store.scope, cutoff)).fetchone()[0]
            indexed = connection.execute('''SELECT count(*) FROM memory_index_receipts r
                JOIN messages m ON m.id = r.message_id JOIN conversations c ON c.id = m.conversation_id
                WHERE c.scope = ? AND c.updated_at >= ? AND r.fingerprint = ?
                  AND r.chunk_count = (SELECT count(*) FROM memory_chunks ch
                     JOIN memory_embeddings e ON e.chunk_id = ch.id AND e.fingerprint = ?
                     WHERE ch.message_id = m.id AND ch.chunker_version = ? AND ch.lexical_contract = ?
                       AND ch.source_hash = r.source_hash)''',
                (self._store.scope, cutoff, fingerprint, state['embedding_fingerprint'],
                 state['chunker_version'], state['lexical_contract'])).fetchone()[0]
            work = dict(connection.execute('''SELECT w.state, count(*) FROM memory_index_work w
                JOIN messages m ON m.id = w.message_id JOIN conversations c ON c.id = m.conversation_id
                WHERE c.scope = ? AND c.updated_at >= ? AND w.fingerprint = ? GROUP BY w.state''',
                (self._store.scope, cutoff, fingerprint)))
            status = 'needs_attention' if work.get('failed', 0) else ('ready' if indexed == total else 'indexing')
            return {"enabled": True, "state": status, "indexed_messages": indexed, "total_messages": total,
                    "pending": work.get('pending', 0), "processing": work.get('leased', 0),
                    "failed": work.get('failed', 0)}

    def start_indexing(self, embeddings):
        """Explicit trusted worker startup; never invoked by a request or incognito."""
        if self._store is None or not self._retrieval_enabled or not self._indexing_enabled:
            return
        from memory_indexer import MemoryIndexer
        from memory_retrieval_store import MemoryRetrievalStore
        self._indexer = MemoryIndexer(MemoryRetrievalStore(self._store), embeddings)

    def indexing_step(self):
        """Bounded background maintenance; disabled paths make no store/model calls."""
        if (self._store is None or not self._retrieval_enabled or not self._indexing_enabled
                or self._indexer is None):
            return "disabled"
        return self._indexer.step()

    def retrieved_evidence_sources(self, *, current_conversation_id, evidence):
        if self._store is None or not self._retrieval_enabled or current_conversation_id is None:
            return None
        from memory_retrieval_store import MemoryRetrievalStore
        return MemoryRetrievalStore(self._store).evidence_sources(
            current_conversation_id=current_conversation_id, evidence=evidence)

    def validate_retrieved_evidence(self, *, current_conversation_id, evidence):
        """Validate saved evidence; disabled memory makes no storage call."""
        if not evidence:
            return True
        if self._store is None or not self._retrieval_enabled or current_conversation_id is None:
            return False
        from memory_retrieval_store import MemoryRetrievalStore
        return MemoryRetrievalStore(self._store).validate_evidence(
            current_conversation_id=current_conversation_id, evidence=evidence,
        )

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
        if self._retrieval_enabled and self._indexing_enabled:
            arguments["enqueue_index"] = True
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
