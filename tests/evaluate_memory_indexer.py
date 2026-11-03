"""Phase 1E local-model check: frozen synthetic fixture, disposable SQLite only."""
from dataclasses import asdict, replace
from datetime import datetime
from collections import Counter
import argparse
import json
from pathlib import Path
import tempfile
import time

from memory_manager import MemoryManager
from memory_retrieval_contracts import MemoryRetrievalPolicy
from memory_retrieval_evaluation import ROOT, digest, manifest, metrics
from memory_retrieval_store import MemoryRetrievalStore
from memory_store import _timestamp
from orchestrator import SkillRouter
from retrieval_embeddings import local_query_embeddings


def run(model_path):
    fixture = (ROOT / 'tests/fixtures/memory_retrieval_baseline.json').read_bytes()
    corpus = json.loads(fixture)
    start = time.perf_counter()
    model = SkillRouter._load_embedding_model(model_path)
    provider = local_query_embeddings(model, model_path)
    initialization_ms = (time.perf_counter() - start) * 1000
    calls = 0
    def encode(text):
        nonlocal calls
        calls += 1
        return provider.encode_query(text)
    tracked = replace(provider, encode_query=encode)
    rows, completed, chunker_versions = [], 0, {}
    with tempfile.TemporaryDirectory(prefix='corveus-phase1e-') as directory:
        path = Path(directory) / 'synthetic.sqlite3'
        managers, current = {}, {}
        for scope in sorted({m['scope'] for m in corpus['messages']}):
            manager = MemoryManager(enabled=True, database_path=path, scope=scope,
                                     retrieval_enabled=True, retrieval_embeddings=tracked)
            manager._store._clock = lambda: datetime.fromisoformat(corpus['clock'])
            managers[scope] = manager
            current[scope] = manager.create_conversation().id
        active = [m for m in corpus['messages'] if not m['deleted'] and not m['expired']]
        sequences = Counter()
        # Existing history predates index startup; discovery must find it.
        with managers['alpha']._store._transaction(write=True) as connection:
            for message in active:
                stamp = _timestamp(datetime.fromisoformat(message['created_at']))
                sequences[message['conversation_id']] += 1
                connection.execute('INSERT OR IGNORE INTO conversations VALUES (?, ?, ?, ?, ?)',
                    (message['conversation_id'], message['scope'], stamp, stamp, 'Synthetic fixture'))
                connection.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)',
                    (message['id'], message['conversation_id'], sequences[message['conversation_id']],
                     message['role'], message['text'], stamp))
        start = time.perf_counter()
        for scope, manager in managers.items():
            manager.start_indexing(tracked)
            chunker_versions[scope] = MemoryRetrievalStore(manager._store).state()['chunker_version']
            for _ in range(len(active) + 2):
                status = manager.indexing_step()
                if status == 'idle':
                    break
                if status != 'completed':
                    raise AssertionError('Synthetic history indexing did not complete.')
                completed += 1
            snapshot = MemoryRetrievalStore(manager._store).read_snapshot(
                current_conversation_id=current[scope], embedding_fingerprint=provider.fingerprint)
            assert snapshot.complete
            for chunk in snapshot.chunks:
                assert chunk.token_length == provider.count_query_tokens(chunk.text) <= provider.max_query_tokens
        assert completed == len(active)
        indexing_ms = (time.perf_counter() - start) * 1000
        indexing_calls = calls
        for case in corpus['cases']:
            manager = managers[case['scope']]
            for mode in ('disabled', 'lexical', 'semantic', 'hybrid'):
                manager._retrieval_enabled = mode != 'disabled'
                policy = MemoryRetrievalPolicy(mode='hybrid' if mode == 'disabled' else mode)
                before = calls
                result = manager.retrieve(case['query'], policy=policy,
                    current_conversation_id=None if case.get('incognito') else current[case['scope']])
                selected = list(dict.fromkeys(e.message_id for e in result.evidence))
                rows.append(dict(case=case['id'], split=case['split'], mode=mode,
                                 selected=selected, relevant=case['relevant'], **metrics(selected, case['relevant']),
                                 status=result.status, index_complete=result.index_complete,
                                 selected_excerpts=len(result.evidence), encode_calls=calls-before))
        assert all(row['encode_calls'] == 0 for row in rows if row['case'] == 'incognito' or row['mode'] in ('disabled','lexical'))
        # Exercise new-exchange enqueue and idempotent finalization, separately
        # from fixed-corpus metrics so lexical corpus statistics remain comparable.
        manager = managers['alpha']
        manager._retrieval_enabled = True
        arguments = dict(user_content='The canary project codename is ORBIT-4827.',
                         assistant_content='The saved codename was acknowledged.', job_id='e' * 32)
        pair = manager.record_completed_exchange(current['alpha'], **arguments)
        assert pair == manager.record_completed_exchange(current['alpha'], **arguments)
        for _ in range(3):
            manager.indexing_step()
        found = manager.retrieve('ORBIT-4827 canary project codename',
                                  current_conversation_id=current['alpha'], policy=MemoryRetrievalPolicy(mode='lexical'))
        assert pair[0].id in {e.message_id for e in found.evidence}
    return dict(format_version=1, corpus_sha256=digest(fixture), clock=corpus['clock'],
                manifest=manifest(model_path), policy=asdict(MemoryRetrievalPolicy()),
                embedding_fingerprint=provider.fingerprint, chunker_versions=chunker_versions,
                initialization_ms=initialization_ms,
                indexing_ms=indexing_ms, indexed_messages=completed, indexing_encode_calls=indexing_calls,
                automatic_enqueue_retry_and_recall=True, rows=rows,
                limitations=['Synthetic observed labels; no tuning or production answer-quality claim.',
                             'Token-bounded chunks differ from Phase 1A/C word chunks.',
                             'Single-run CPU observations, not a performance benchmark.',
                             'No application databases or GPU generation used.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.model_path.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f"Indexed {report['indexed_messages']} synthetic saved messages; recorded {len(report['rows'])} retrieval checks.")
