"""Independent memory query embeddings; never reads/writes the skill cache.

Model construction is explicitly caller-owned. Adapters may share an existing
local CPU model; constructing a MemoryManager never loads a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from threading import Lock
from typing import Callable

from memory_retrieval_contracts import require_hash


def normalized_vector(values, dimension):
    try:
        vector = tuple(float(v) for v in values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid embedding vector.') from None
    if (len(vector) != dimension or not all(math.isfinite(v) for v in vector)
            or abs(sum(v * v for v in vector) - 1) > .001):
        raise ValueError('Embedding must have the expected dimension and unit norm.')
    return vector


@dataclass(frozen=True)
class QueryEmbeddings:
    fingerprint: str
    dimension: int
    encode_query: Callable = field(repr=False)
    count_query_tokens: Callable = field(repr=False)
    max_query_tokens: int

    def __post_init__(self):
        require_hash(self.fingerprint, 'fingerprint')
        if type(self.dimension) is not int or not 1 <= self.dimension <= 65536:
            raise ValueError('Invalid embedding dimension.')
        if type(self.max_query_tokens) is not int or self.max_query_tokens < 1:
            raise ValueError('Invalid embedding query limit.')
        if not callable(self.encode_query) or not callable(self.count_query_tokens):
            raise ValueError('Embedding callbacks must be callable.')

    def encode(self, query):
        count = self.count_query_tokens(query)
        if type(count) is not int or count < 0:
            raise ValueError('Invalid embedding token count.')
        if count > self.max_query_tokens:
            return None  # Never silently encode a truncated query.
        return normalized_vector(self.encode_query(query), self.dimension)


def local_query_embeddings(model, model_path, *, lock=None):
    """Bind an already loaded local CPU model and fingerprint its effective contract.

    The caller must keep the model/tokenizer immutable for the adapter lifetime,
    and pass the same lock to all concurrent users of a shared model.
    """
    if str(model.device) != 'cpu':
        raise ValueError('Memory embeddings require a local CPU model.')
    root = Path(model_path).resolve()
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    if not root.is_dir() or not paths:
        raise ValueError('Local embedding artifacts are required.')
    artifact_hash = hashlib.sha256()
    for path in paths:
        artifact_hash.update(path.relative_to(root).as_posix().encode() + b'\0')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                artifact_hash.update(block)
        artifact_hash.update(b'\0')
    tokenizer = model.tokenizer
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    dimension = model.get_sentence_embedding_dimension()
    packages = {}
    for name in ('sentence-transformers', 'transformers', 'tokenizers', 'torch', 'numpy'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    modules = []
    for name, module in getattr(model, '_modules', {}).items():
        get_config = getattr(module, 'get_config_dict', None)
        modules.append((name, type(module).__qualname__, get_config() if callable(get_config) else None))
    contract = dict(version='memory-query-float32-v1', artifacts=artifact_hash.hexdigest(),
                    model_class=f'{type(model).__module__}.{type(model).__qualname__}',
                    dimension=dimension, max_seq_length=model.max_seq_length,
                    device=str(model.device), dtype=str(getattr(model, 'dtype', None)),
                    truncate_dim=getattr(model, 'truncate_dim', None), modules=modules,
                    prompts=getattr(model, 'prompts', {}),
                    default_prompt_name=getattr(model, 'default_prompt_name', None),
                    tokenizer_class=type(tokenizer).__qualname__,
                    tokenizer_limit=tokenizer.model_max_length,
                    tokenizer_backend=backend.to_str() if backend is not None else None,
                    special_tokens_map=getattr(tokenizer, 'special_tokens_map', {}),
                    normalize_embeddings=True, encoding='float32-le', packages=packages)
    # Prompt prefixes need separate token accounting; reject rather than truncate.
    if contract['default_prompt_name'] is not None:
        raise ValueError('Prompt-prefixed embedding models are not supported in Phase 1C.')
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False, default=str).encode()).hexdigest()
    limits = [model.max_seq_length]
    if type(tokenizer.model_max_length) is int and tokenizer.model_max_length > 0:
        limits.append(tokenizer.model_max_length)
    if any(type(n) is not int or n < 1 for n in limits):
        raise ValueError('Invalid model token limit.')
    encode_lock = lock if lock is not None else Lock()

    def count(query):
        with encode_lock:
            return len(tokenizer.encode(query, add_special_tokens=True, truncation=False))

    def encode(query):
        with encode_lock:
            return model.encode(query, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)

    return QueryEmbeddings(fingerprint, dimension, encode, count, min(limits))
