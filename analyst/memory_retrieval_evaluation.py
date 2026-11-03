"""Synthetic-only, offline Phase 1A evaluator. Never opens application databases.

Policy constants below are frozen experimental starting points, not production
thresholds. No skill routing or persistent embedding cache is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from orchestrator import tokenize

ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(k=4, bm25_k1=1.5, bm25_b=0.75, lexical_floor=1.0,
              semantic_floor=0.45, rrf_constant=60, chunk_words=96,
              overlap_words=16, recency=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def chunks(message):
    # Deliberately simple baseline; model-tokenizer chunking belongs to later work.
    words = message['text'].split()
    step = POLICY['chunk_words'] - POLICY['overlap_words']
    return [dict(message, chunk_id=f"{message['id']}:{start}",
                 text=' '.join(words[start:start + POLICY['chunk_words']]))
            for start in range(0, len(words), step)]


def eligible(corpus, case):
    if case.get('incognito'):
        return []
    return [chunk for message in corpus['messages']
            if message['scope'] == case['scope']
            and not message['deleted'] and not message['expired']
            for chunk in chunks(message)]


def lexical_scores(query, documents):
    terms = set(tokenize(query))
    frequencies = [Counter(tokenize(doc['text'])) for doc in documents]
    n = len(frequencies)
    average = sum(sum(freq.values()) for freq in frequencies) / (n or 1)
    df = Counter(term for freq in frequencies for term in freq)
    scores = []
    for freq in frequencies:
        length_factor = 1.5 * (0.25 + 0.75 * sum(freq.values()) / (average or 1))
        scores.append(sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5))
                          * freq[t] * 2.5 / (freq[t] + length_factor)
                          for t in terms if freq[t]))
    return scores


def rank(documents, lexical, semantic, mode):
    channels = []
    for values, floor in ((lexical, POLICY['lexical_floor']),
                          (semantic, POLICY['semantic_floor'])):
        channels.append(sorted((i for i, value in enumerate(values) if value >= floor),
                               key=lambda i: (-values[i], documents[i]['chunk_id'])))
    if mode == 'lexical':
        order = channels[0]
    elif mode == 'semantic':
        order = channels[1]
    else:
        fused = Counter()
        for channel in channels:
            for position, index in enumerate(channel, 1):
                fused[index] += 1 / (POLICY['rrf_constant'] + position)
        order = sorted(fused, key=lambda i: (-fused[i], documents[i]['chunk_id']))
    result, seen_messages, seen_text = [], set(), set()
    for index in order:
        doc = documents[index]
        text_hash = digest(doc['text'].encode())
        if doc['id'] in seen_messages or text_hash in seen_text:
            continue
        seen_messages.add(doc['id'])
        seen_text.add(text_hash)
        result.append(doc['id'])
        if len(result) == POLICY['k']:
            break
    return result


def metrics(selected, relevant):
    relevant = set(relevant)
    hits = [i for i, value in enumerate(selected, 1) if value in relevant]
    return dict(precision=len(hits) / len(selected) if selected else None,
                recall=len(hits) / len(relevant) if relevant else None,
                reciprocal_rank=1 / hits[0] if hits else 0,
                false_recall=bool(selected) if not relevant else None,
                abstained=not selected)


def manifest(model_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])
    files = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0')
    source = {}
    for name in files:
        if not name:
            continue
        path = ROOT / os.fsdecode(name)
        if path.is_file() and (name.startswith((b'analyst/', b'tests/')) or name in
                               (b'Dockerfile', b'docker-compose.yml', b'.env.example', b'.gitignore', b'README.md')):
            source[os.fsdecode(name)] = digest(path.read_bytes())
    model_files = {str(p.relative_to(model_path)): digest(p.read_bytes())
                   for p in sorted(model_path.rglob('*')) if p.is_file()} if model_path else {}
    versions = {}
    for name in ('numpy', 'sentence-transformers', 'transformers', 'tokenizers', 'torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(commit=git('rev-parse', 'HEAD').decode().strip(),
                working_tree=git('status', '--short').decode(), source_sha256=source,
                model_sha256=model_files, packages=versions,
                platform=platform.platform(), cpu=platform.processor(),
                cpu_count=os.cpu_count(), python=platform.python_version(),
                threads={k: os.environ.get(k) for k in
                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def evaluate(corpus, model=None):
    rows = []
    for case in corpus['cases']:
        start = time.perf_counter()
        documents = eligible(corpus, case)
        lexical = lexical_scores(case['query'], documents)
        semantic = []
        encode_calls = 0
        corpus_encode_ms = query_encode_ms = 0.0
        if model is not None and documents:
            # In-memory synthetic corpus only; scope filtering precedes encoding.
            encode_start = time.perf_counter()
            vectors = model.encode([doc['text'] for doc in documents],
                                   normalize_embeddings=True, show_progress_bar=False)
            corpus_encode_ms = (time.perf_counter() - encode_start) * 1000
            encode_start = time.perf_counter()
            query = model.encode(case['query'], normalize_embeddings=True,
                                 show_progress_bar=False)
            query_encode_ms = (time.perf_counter() - encode_start) * 1000
            semantic = [float(vector @ query) for vector in vectors]
            encode_calls = 2
        scoring_ms = (time.perf_counter() - start) * 1000
        modes = ('disabled', 'lexical', 'semantic', 'hybrid') if model is not None else ('disabled', 'lexical')
        for mode in modes:
            selected = [] if mode == 'disabled' else rank(documents, lexical, semantic, mode)
            rows.append(dict(case=case['id'], split=case['split'], mode=mode,
                             selected=selected, relevant=case['relevant'],
                             **metrics(selected, case['relevant']),
                             eligible_chunks=len(documents),
                             corpus_encode_ms=corpus_encode_ms if mode in ('semantic', 'hybrid') else 0,
                             query_encode_ms=query_encode_ms if mode in ('semantic', 'hybrid') else 0,
                             shared_case_setup_ms=scoring_ms if mode != 'disabled' else 0,
                             embedding_calls=encode_calls if mode in ('semantic', 'hybrid') else 0))
    summary = []
    for split in ('development', 'held_out'):
        for mode in sorted({row['mode'] for row in rows}):
            group = [row for row in rows if row['split'] == split and row['mode'] == mode]
            values = {key: [row[key] for row in group if row[key] is not None]
                      for key in ('precision', 'recall', 'reciprocal_rank', 'false_recall', 'abstained')}
            summary.append(dict(split=split, mode=mode, cases=len(group),
                                **{key: dict(mean=statistics.mean(v) if v else None, n=len(v))
                                   for key, v in values.items()}))
    return dict(rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'tests/fixtures/memory_retrieval_baseline.json')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    model = None
    start = time.perf_counter()
    model_path = args.model_path.resolve() if args.model_path else None
    if model_path:
        if not model_path.is_dir():
            parser.error('Local model directory does not exist')
        # Existing local-only CPU loader, without constructing a skill router.
        from orchestrator import SkillRouter
        model = SkillRouter._load_embedding_model(model_path)
    initialization_ms = (time.perf_counter() - start) * 1000
    report = dict(format_version=1, corpus_sha256=digest(corpus_bytes),
                  clock=corpus['clock'], policy=POLICY,
                  manifest=manifest(model_path), initialization_ms=initialization_ms,
                  embedding_contract=None if model is None else dict(
                      max_seq_length=model.max_seq_length,
                      dimension=model.get_sentence_embedding_dimension(),
                      tokenizer_class=type(model.tokenizer).__name__,
                      normalize_embeddings=True, device='cpu', local_files_only=True),
                  limitations=['Synthetic fully judged message-level corpus; no answer-quality claim.',
                               'Word chunks, not tokenizer-aware production chunks.',
                               'Per-case corpus and query encode timing; no warm index latency benchmark.',
                               'No token-budget, database, deletion-race, or durable-job integration tested.',
                               'Untuned thresholds; held-out labels must not guide tuning.',
                               'Raw message recall counts exact duplicates separately despite deduplication.',
                               'Relevance means useful historical evidence, not factual truth or permission.'],
                  **evaluate(corpus, model))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
