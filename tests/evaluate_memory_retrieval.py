"""Offline Phase 1C check through actual scoped storage and ranking.

Uses only the checked-in synthetic fixture and a temporary database. Phase 1A's
report, labels, thresholds, and evaluator are not modified.
"""
from dataclasses import asdict, replace
from datetime import datetime
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import tempfile
import time

from memory_manager import MemoryManager
from memory_retrieval_contracts import MemoryRetrievalPolicy, content_hash
from memory_retrieval_evaluation import ROOT, digest, manifest, metrics
from memory_retrieval_store import ChunkWrite, MemoryRetrievalStore
from memory_store import _timestamp
from orchestrator import SkillRouter
from retrieval_embeddings import local_query_embeddings
from retrieval_lexical import MEMORY_LEXICAL_CONTRACT, memory_tokenize


def run(model_path):
    fixture_bytes = (ROOT / 'tests/fixtures/memory_retrieval_baseline.json').read_bytes()
    corpus = json.loads(fixture_bytes)
    start = time.perf_counter()
    model = SkillRouter._load_embedding_model(model_path)
    provider = local_query_embeddings(model, model_path)
    initialization_ms = (time.perf_counter() - start) * 1000
    encode_calls = 0

    def encode(query):
        nonlocal encode_calls
        encode_calls += 1
        return provider.encode_query(query)

    tracked = replace(provider, encode_query=encode)
    rows = []
    with tempfile.TemporaryDirectory(prefix='corveus-phase1c-') as directory:
        database = Path(directory) / 'synthetic.sqlite3'
        managers, currents = {}, {}
        for scope in sorted({m['scope'] for m in corpus['messages']}):
            manager = MemoryManager(enabled=True, database_path=database, scope=scope,
                                     retrieval_enabled=True, retrieval_embeddings=tracked)
            manager._store._clock = lambda: datetime.fromisoformat(corpus['clock'])
            managers[scope] = manager
            currents[scope] = manager.create_conversation().id
            MemoryRetrievalStore(manager._store).configure(
                lexical_contract=MEMORY_LEXICAL_CONTRACT, chunker_version='baseline-word96-overlap16-v1',
                embedding_fingerprint=provider.fingerprint)
        sequences = Counter()
        active = [m for m in corpus['messages'] if not m['deleted'] and not m['expired']]
        with next(iter(managers.values()))._store._transaction(write=True) as connection:
            for message in active:
                sequences[message['conversation_id']] += 1
                stamp = _timestamp(datetime.fromisoformat(message['created_at']))
                connection.execute('INSERT OR IGNORE INTO conversations VALUES (?, ?, ?, ?, ?)',
                                   (message['conversation_id'], message['scope'], stamp, stamp, 'Synthetic baseline'))
                connection.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)',
                                   (message['id'], message['conversation_id'], sequences[message['conversation_id']],
                                    message['role'], message['text'], stamp))
        start = time.perf_counter()
        for message in active:
            repository = MemoryRetrievalStore(managers[message['scope']]._store)
            words = list(re.finditer(r'\S+', message['text']))
            writes = []
            excerpts = []
            for ordinal, start_word in enumerate(range(0, len(words), 80)):
                left = words[start_word].start()
                right = words[min(start_word + 96, len(words)) - 1].end()
                excerpt = message['text'][left:right]
                length = provider.count_query_tokens(excerpt)
                if length > provider.max_query_tokens:
                    raise ValueError('Synthetic chunk exceeds local model token limit.')
                writes.append(ChunkWrite(ordinal, left, right, length,
                                         tuple(sorted(Counter(memory_tokenize(excerpt)).items()))))
                excerpts.append(excerpt)
            assert repository.replace_chunks(message['id'], source_hash=content_hash(message['text']),
                                              lexical_contract=MEMORY_LEXICAL_CONTRACT,
                                              chunker_version='baseline-word96-overlap16-v1', chunks=tuple(writes))
            # Explicit synthetic indexing only; not a production backfill worker.
            vectors = model.encode(excerpts, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
            stored = [c for c in repository.candidates(current_conversation_id=currents[message['scope']])
                      if c.message_id == message['id']]
            for chunk, vector in zip(stored, vectors):
                assert repository.put_embedding(chunk.id, source_hash=chunk.source_hash,
                                                fingerprint=provider.fingerprint, dimension=provider.dimension,
                                                vector=struct.pack(f'<{provider.dimension}f', *vector))
        indexing_ms = (time.perf_counter() - start) * 1000
        for case in corpus['cases']:
            manager = managers[case['scope']]
            for mode in ('disabled', 'lexical', 'semantic', 'hybrid'):
                policy = MemoryRetrievalPolicy(mode='hybrid' if mode == 'disabled' else mode)
                before = encode_calls
                manager._retrieval_enabled = mode != 'disabled'
                start = time.perf_counter()
                result = manager.retrieve(case['query'], policy=policy,
                                          current_conversation_id=None if case.get('incognito') else currents[case['scope']])
                selected = [e.message_id for e in result.evidence]
                rows.append(dict(case=case['id'], split=case['split'], mode=mode,
                                 selected=selected, relevant=case['relevant'], **metrics(selected, case['relevant']),
                                 retrieval_ms=(time.perf_counter() - start) * 1000,
                                 query_encode_calls=encode_calls - before, status=result.status,
                                 index_complete=result.index_complete, semantic_available=result.semantic_available))
        assert all(row['query_encode_calls'] == 0 for row in rows if row['case'] == 'incognito' or row['mode'] in ('disabled', 'lexical'))
    return dict(format_version=1, corpus_sha256=digest(fixture_bytes), clock=corpus['clock'],
                manifest=manifest(model_path), embedding_fingerprint=provider.fingerprint,
                policy=asdict(MemoryRetrievalPolicy()), lexical_contract=MEMORY_LEXICAL_CONTRACT,
                initialization_ms=initialization_ms, synthetic_indexing_ms=indexing_ms, rows=rows,
                limitations=['Observed held-out labels; no tuning or independent generalization claim.',
                             'Identifier-aware tokenizer and conservative role-aware dedup differ from frozen Phase 1A.',
                             'Raw message recall counts exact duplicates separately.',
                             'Synthetic word chunks; no automatic indexing or worker/prompt integration.',
                             'Timings are single-run observations, not a performance benchmark.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.model_path.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Recorded {len(report['rows'])} synthetic case/mode results.")
