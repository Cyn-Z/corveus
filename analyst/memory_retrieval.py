"""Independent, scoped conversation-memory ranking. Text is untrusted evidence.

No skill routing, model loading, backfill, prompt construction or job transitions.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
import struct

from memory_retrieval_contracts import MemoryEvidence, MemoryRetrievalPolicy, MemoryRetrievalResult, content_hash
from memory_retrieval_store import MemoryRetrievalStore
from retrieval_embeddings import QueryEmbeddings, normalized_vector
from retrieval_lexical import MEMORY_LEXICAL_CONTRACT, bm25_scores, memory_tokenize


def _tie(chunk, policy):
    recent = -datetime.fromisoformat(chunk.created_at).timestamp() if policy.recency_tiebreak else 0
    return (recent, chunk.conversation_id, chunk.sequence, chunk.start_char,
            chunk.end_char, chunk.source_hash, chunk.chunk_hash)


def rank_candidates(chunks, lexical, semantic, policy):
    """Gate each complete chunk record before RRF; recency only breaks ties.

    Exact same-role text and already covered spans are redundant. Partially
    overlapping spans with distinct context remain eligible to avoid hiding facts.
    """
    if len(lexical) != len(chunks) or len(semantic) != len(chunks):
        raise ValueError('Scores must align with complete candidate records.')
    for value in lexical:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid lexical score.')
    for value in semantic:
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or not -1 <= value <= 1):
            raise ValueError('Invalid semantic score.')
    channels = []
    if policy.mode != 'semantic':
        channels.append(sorted((i for i, value in enumerate(lexical) if value >= policy.lexical_floor),
                               key=lambda i: (-lexical[i], _tie(chunks[i], policy))))
    if policy.mode != 'lexical':
        channels.append(sorted((i for i, value in enumerate(semantic)
                                if value is not None and value >= policy.semantic_floor),
                               key=lambda i: (-semantic[i], _tie(chunks[i], policy))))
    if policy.mode == 'hybrid':
        scores = Counter()
        for channel in channels:
            for position, index in enumerate(channel, 1):
                scores[index] += 1 / (policy.rrf_constant + position)
        order = sorted(scores, key=lambda i: (-scores[i], _tie(chunks[i], policy)))
    else:
        order = channels[0]
    selected, texts, covered = [], set(), {}
    for index in order:
        chunk = chunks[index]
        text_key = (chunk.role, chunk.chunk_hash)
        spans = covered.get(chunk.message_id, [])
        if text_key in texts or any(left <= chunk.start_char and chunk.end_char <= right
                                    for left, right in spans):
            continue
        selected.append(index)
        texts.add(text_key)
        covered.setdefault(chunk.message_id, []).append((chunk.start_char, chunk.end_char))
        if len(selected) == policy.max_results:
            break
    return tuple(selected)


class MemoryRetriever:
    def __init__(self, store: MemoryRetrievalStore, *, embeddings: QueryEmbeddings | None = None):
        if not isinstance(store, MemoryRetrievalStore):
            raise ValueError('store must be a scoped MemoryRetrievalStore.')
        if embeddings is not None and not isinstance(embeddings, QueryEmbeddings):
            raise ValueError('embeddings must be a trusted QueryEmbeddings adapter.')
        self.store = store
        self.embeddings = embeddings

    def retrieve(self, query, *, current_conversation_id, policy):
        if not isinstance(policy, MemoryRetrievalPolicy):
            raise ValueError('policy must be a MemoryRetrievalPolicy.')
        if not isinstance(query, str) or not query.strip() or len(query) > policy.max_query_chars:
            raise ValueError('Query must be nonempty and within its character limit.')
        provider = self.embeddings if policy.mode != 'lexical' else None
        snapshot = self.store.read_snapshot(current_conversation_id=current_conversation_id,
                                            embedding_fingerprint=provider.fingerprint if provider else None)
        unavailable = MemoryRetrievalResult(status='not_ready', policy_fingerprint=policy.fingerprint)
        if snapshot is None or snapshot.lexical_contract != MEMORY_LEXICAL_CONTRACT:
            return unavailable
        chunks = snapshot.chunks
        lexical = bm25_scores(memory_tokenize(query), [dict(chunk.terms) for chunk in chunks])
        semantic = [None] * len(chunks)
        semantic_available = False
        if provider is not None and snapshot.embedding_fingerprint == provider.fingerprint:
            vectors = {}
            for chunk_id, dimension, raw in snapshot.vectors:
                if dimension != provider.dimension or type(raw) is not bytes or len(raw) != dimension * 4:
                    continue
                try:
                    vectors[chunk_id] = normalized_vector(struct.unpack(f'<{dimension}f', raw), dimension)
                except (ValueError, struct.error):
                    continue
            if vectors:
                query_vector = provider.encode(query)  # Outside any SQLite transaction.
                if query_vector is not None:
                    semantic_available = True
                    query_norm = math.sqrt(sum(v*v for v in query_vector))
                    for i, chunk in enumerate(chunks):
                        vector = vectors.get(chunk.id)
                        if vector is not None:
                            cosine = sum(a*b for a, b in zip(query_vector, vector))
                            cosine /= query_norm * math.sqrt(sum(v*v for v in vector))
                            semantic[i] = max(-1.0, min(1.0, cosine))
        if policy.mode == 'semantic' and not semantic_available and (chunks or not snapshot.complete):
            return unavailable
        indices = rank_candidates(chunks, lexical, semantic, policy)
        evidence = []
        for rank, index in enumerate(indices, 1):
            chunk = chunks[index]
            identity = json.dumps([chunk.message_id, chunk.source_hash, chunk.start_char,
                                   chunk.end_char, chunk.chunker_version], separators=(',', ':'))
            evidence.append(MemoryEvidence(
                content_hash(identity), chunk.conversation_id, chunk.message_id, chunk.sequence,
                chunk.role, chunk.created_at, chunk.start_char, chunk.end_char, chunk.text,
                chunk.source_hash, chunk.chunk_hash, chunk.chunker_version,
                lexical[index], semantic[index], rank))
        if not self.store.snapshot_current(snapshot, current_conversation_id=current_conversation_id):
            return unavailable  # Do not silently retrieve different evidence.
        semantic_complete = (not chunks or (semantic_available and all(v is not None for v in semantic)))
        return MemoryRetrievalResult(
            status='ready', evidence=tuple(evidence), policy_fingerprint=policy.fingerprint,
            embedding_fingerprint=provider.fingerprint if semantic_available else None,
            index_generation=snapshot.generation, retrieved_at=snapshot.retrieved_at,
            index_complete=snapshot.complete and (policy.mode == 'lexical' or semantic_complete),
            lexical_contract=snapshot.lexical_contract, chunker_version=snapshot.chunker_version,
            semantic_available=semantic_available)
