"""Immutable memory evidence contracts. Historical text is untrusted data.

These types contain no router, model, prompt, or job dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
import re


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text.')
    return value


def require_hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a SHA-256 hex digest.')
    return value


def content_hash(text):
    return hashlib.sha256(text.encode('utf-8', errors='strict')).hexdigest()


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    """Untuned Phase 1A floors; eligibility precedes fusion and result limits."""
    max_results: int = 4
    version: str = 'phase1-v2'
    mode: str = 'hybrid'
    lexical_floor: float = 1.0
    semantic_floor: float = 0.45
    rrf_constant: int = 60
    recency_tiebreak: bool = False
    max_query_chars: int = 16000

    def __post_init__(self):
        if type(self.max_results) is not int or not 1 <= self.max_results <= 32:
            raise ValueError('max_results must be between 1 and 32.')
        require_text(self.version, 'version')
        if self.mode not in ('lexical', 'semantic', 'hybrid'):
            raise ValueError('Invalid retrieval mode.')
        for name in ('lexical_floor', 'semantic_floor'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Relevance floors must be finite.')
        if self.lexical_floor <= 0 or not 0 < self.semantic_floor <= 1:
            raise ValueError('Relevance floors must be positive; cosine cannot exceed one.')
        if type(self.rrf_constant) is not int or not 1 <= self.rrf_constant <= 1000:
            raise ValueError('Invalid RRF constant.')
        if type(self.recency_tiebreak) is not bool:
            raise ValueError('recency_tiebreak must be boolean.')
        if type(self.max_query_chars) is not int or not 1 <= self.max_query_chars <= 1000000:
            raise ValueError('Invalid query character bound.')

    @property
    def fingerprint(self):
        return content_hash(json.dumps(asdict(self), sort_keys=True, separators=(',', ':')))


@dataclass(frozen=True)
class MemoryEvidence:
    evidence_id: str
    conversation_id: str
    message_id: str
    sequence: int
    role: str
    created_at: str
    start_char: int
    end_char: int
    excerpt: str = field(repr=False)
    source_hash: str
    chunk_hash: str
    chunker_version: str
    lexical_score: float
    semantic_score: float | None
    rank: int

    def __post_init__(self):
        for name in ('evidence_id', 'conversation_id', 'message_id', 'chunker_version'):
            require_text(getattr(self, name), name)
        if self.role not in ('user', 'assistant'):
            raise ValueError('Invalid historical role.')
        try:
            stamp = datetime.fromisoformat(self.created_at)
            if stamp.utcoffset() is None:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('created_at must be timezone-aware.') from None
        for name in ('sequence', 'rank'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive.')
        if (type(self.start_char) is not int or type(self.end_char) is not int
                or not 0 <= self.start_char < self.end_char):
            raise ValueError('Invalid excerpt offsets.')
        require_text(self.excerpt, 'excerpt')
        if len(self.excerpt) != self.end_char - self.start_char:
            raise ValueError('Excerpt does not match offset length.')
        require_hash(self.source_hash, 'source_hash')
        require_hash(self.chunk_hash, 'chunk_hash')
        if content_hash(self.excerpt) != self.chunk_hash:
            raise ValueError('Excerpt hash mismatch.')
        for name in ('lexical_score', 'semantic_score'):
            score = getattr(self, name)
            if name == 'semantic_score' and score is None:
                continue
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('Scores must be finite numbers.')
        if self.lexical_score < 0 or (self.semantic_score is not None
                                     and not -1 <= self.semantic_score <= 1):
            raise ValueError('Score outside its channel range.')


@dataclass(frozen=True)
class MemoryRetrievalResult:
    status: str
    evidence: tuple[MemoryEvidence, ...] = field(default=(), repr=False)
    policy_fingerprint: str | None = None
    embedding_fingerprint: str | None = None
    index_generation: int | None = None
    retrieved_at: str | None = None
    index_complete: bool = False
    lexical_contract: str | None = None
    chunker_version: str | None = None
    semantic_available: bool = False

    def __post_init__(self):
        if self.status not in ('disabled', 'incognito', 'not_ready', 'ready'):
            raise ValueError('Invalid retrieval status.')
        if (type(self.evidence) is not tuple or len(self.evidence) > 32
                or not all(isinstance(item, MemoryEvidence) for item in self.evidence)):
            raise ValueError('Evidence must be an immutable bounded tuple.')
        if self.status != 'ready' and self.evidence:
            raise ValueError('Unavailable retrieval cannot contain evidence.')
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError('Duplicate evidence identifiers.')
        for name in ('policy_fingerprint', 'embedding_fingerprint'):
            value = getattr(self, name)
            if value is not None:
                require_hash(value, name)
        if self.index_generation is not None and (type(self.index_generation) is not int
                                                  or self.index_generation < 0):
            raise ValueError('Invalid index generation.')
        if type(self.semantic_available) is not bool:
            raise ValueError('semantic_available must be boolean.')
        for name in ('lexical_contract', 'chunker_version'):
            if getattr(self, name) is not None:
                require_text(getattr(self, name), name)
        if type(self.index_complete) is not bool:
            raise ValueError('index_complete must be boolean.')
        if self.status == 'ready':
            if any(item.semantic_score is not None for item in self.evidence) and self.embedding_fingerprint is None:
                raise ValueError('Semantic evidence requires an embedding fingerprint.')
            if self.policy_fingerprint is None or self.index_generation is None:
                raise ValueError('Ready results require policy and index provenance.')
            try:
                stamp = datetime.fromisoformat(self.retrieved_at)
                if stamp.utcoffset() is None:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError('Ready results require an aware retrieval timestamp.') from None
            if [item.rank for item in self.evidence] != list(range(1, len(self.evidence) + 1)):
                raise ValueError('Evidence ranks must be contiguous and ordered.')
