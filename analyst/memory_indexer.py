"""Scoped, resumable CPU indexing; no job database or inference permits.

Only explicit worker maintenance calls run this code. Callbacks are synchronous;
source text never appears in exception messages, diagnostics, or representations.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
import json
import struct
import uuid

from memory_retrieval_contracts import content_hash
from memory_retrieval_store import ChunkWrite, MemoryRetrievalStore
from memory_store import Message, _timestamp
from retrieval_embeddings import QueryEmbeddings
from retrieval_lexical import MEMORY_LEXICAL_CONTRACT, memory_tokenize


def migrate_5_to_6(connection):
    # Nullable columns preserve unused legacy queue records. Discovery replaces
    # active legacy records; obsolete records are removed in bounded batches.
    connection.execute('ALTER TABLE memory_index_work ADD COLUMN source_hash TEXT')
    connection.execute('ALTER TABLE memory_index_work ADD COLUMN lexical_contract TEXT')
    connection.execute("ALTER TABLE memory_index_state ADD COLUMN scan_after TEXT NOT NULL DEFAULT ''")
    connection.execute('ALTER TABLE memory_index_state ADD COLUMN indexing_enabled INTEGER NOT NULL DEFAULT 0 CHECK(indexing_enabled IN (0, 1))')
    connection.execute('''CREATE TABLE memory_index_receipts (
        message_id TEXT PRIMARY KEY NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
        source_hash TEXT NOT NULL CHECK(length(source_hash) = 64),
        chunk_count INTEGER NOT NULL CHECK(chunk_count > 0)
    )''')
    connection.execute('''CREATE TRIGGER memory_receipt_source_update AFTER UPDATE ON messages BEGIN
        DELETE FROM memory_index_receipts WHERE message_id = OLD.id;
    END''')


def work_fingerprint(state):
    return content_hash(json.dumps([state['lexical_contract'], state['chunker_version'],
                                    state['embedding_fingerprint']], separators=(',', ':')))


def enqueue_saved_messages(connection, scope, messages):
    """Part of the caller's exchange transaction; never tokenize or embed here."""
    state = connection.execute('SELECT * FROM memory_index_state WHERE scope = ? AND indexing_enabled = 1',
                               (scope,)).fetchone()
    if state is None:
        return
    fingerprint = work_fingerprint(state)
    for message in messages:
        connection.execute('''INSERT OR IGNORE INTO memory_index_work
            (message_id, fingerprint, chunker_version, source_hash, lexical_contract)
            VALUES (?, ?, ?, ?, ?)''',
            (message.id, fingerprint, state['chunker_version'], content_hash(message.content), state['lexical_contract']))


@dataclass(frozen=True)
class IndexLease:
    message: Message = field(repr=False)
    fingerprint: str
    source_hash: str
    chunker_version: str
    lexical_contract: str
    token: str = field(repr=False)
    attempts: int


class IndexQueue:
    def __init__(self, repository):
        self.repository = repository
        self.store = repository.store

    def discover(self, *, limit=16, expected_fingerprint=None):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Invalid discovery limit.')
        with self.store._transaction(write=True) as connection:
            state = connection.execute('SELECT * FROM memory_index_state WHERE scope = ? AND indexing_enabled = 1',
                                       (self.store.scope,)).fetchone()
            if state is None:
                return 0
            fingerprint = work_fingerprint(state)
            if expected_fingerprint is not None and fingerprint != expected_fingerprint:
                return 0
            # Remove bounded obsolete work, scoped through authoritative sources.
            connection.execute('''DELETE FROM memory_index_work WHERE rowid IN (
                SELECT w.rowid FROM memory_index_work w JOIN messages m ON m.id = w.message_id
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.scope = ? AND (w.fingerprint != ? OR c.updated_at < ?) LIMIT ?)''',
                (self.store.scope, fingerprint, self.repository._cutoff(), limit))
            rows = connection.execute('''SELECT m.* FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.scope = ? AND c.updated_at >= ? AND m.id > ? ORDER BY m.id LIMIT ?''',
                (self.store.scope, self.repository._cutoff(), state['scan_after'], limit)).fetchall()
            count = 0
            for row in rows:
                source_hash = content_hash(row['content'])
                receipt = connection.execute('''SELECT * FROM memory_index_receipts
                    WHERE message_id = ? AND fingerprint = ? AND source_hash = ?''',
                    (row['id'], fingerprint, source_hash)).fetchone()
                if receipt is not None:
                    present = connection.execute('''SELECT count(ch.id) AS chunks, count(e.chunk_id) AS vectors
                        FROM memory_chunks ch LEFT JOIN memory_embeddings e
                        ON e.chunk_id = ch.id AND e.fingerprint = ?
                        WHERE ch.message_id = ? AND ch.chunker_version = ? AND ch.lexical_contract = ?
                          AND ch.source_hash = ?''',
                        (state['embedding_fingerprint'], row['id'], state['chunker_version'], state['lexical_contract'], source_hash)).fetchone()
                    if present['chunks'] == receipt['chunk_count'] == present['vectors']:
                        continue
                previous = connection.execute('''SELECT source_hash, lexical_contract FROM memory_index_work
                    WHERE message_id = ? AND fingerprint = ? AND chunker_version = ?''',
                    (row['id'], fingerprint, state['chunker_version'])).fetchone()
                if previous is not None and previous['source_hash'] == source_hash and previous['lexical_contract'] == state['lexical_contract']:
                    continue  # Preserve live leases and terminal failure limits.
                connection.execute('''INSERT INTO memory_index_work
                    (message_id, fingerprint, chunker_version, source_hash, lexical_contract)
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT(message_id, fingerprint, chunker_version)
                    DO UPDATE SET source_hash = excluded.source_hash, lexical_contract = excluded.lexical_contract,
                        state = 'pending', lease_id = NULL, lease_until = NULL, attempts = 0, error_code = NULL''',
                    (row['id'], fingerprint, state['chunker_version'], source_hash, state['lexical_contract']))
                count += 1
            cursor = rows[-1]['id'] if rows else ''
            connection.execute('UPDATE memory_index_state SET scan_after = ? WHERE scope = ?', (cursor, self.store.scope))
            return count

    def claim(self, *, lease_seconds=120, max_attempts=3, expected_fingerprint=None):
        if (type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600
                or type(max_attempts) is not int or not 1 <= max_attempts <= 10):
            raise ValueError('Invalid indexing lease/attempt bounds.')
        with self.store._transaction(write=True) as connection:
            now = _timestamp(self.store._clock())
            state = connection.execute('SELECT * FROM memory_index_state WHERE scope = ? AND indexing_enabled = 1',
                                       (self.store.scope,)).fetchone()
            if state is None:
                return None
            fingerprint = work_fingerprint(state)
            if expected_fingerprint is not None and fingerprint != expected_fingerprint:
                return None
            # Exhausted expired leases become terminal; discovery cannot hot-loop them.
            connection.execute('''UPDATE memory_index_work SET state = 'failed', lease_id = NULL,
                lease_until = NULL, error_code = 'ATTEMPTS_EXHAUSTED'
                WHERE rowid IN (SELECT w.rowid FROM memory_index_work w
                  JOIN messages m ON m.id = w.message_id JOIN conversations c ON c.id = m.conversation_id
                  WHERE w.fingerprint = ? AND w.state = 'leased' AND w.lease_until <= ?
                    AND w.attempts >= ? AND c.scope = ? LIMIT 16)''',
                (fingerprint, now, max_attempts, self.store.scope))
            row = connection.execute('''SELECT w.*, m.conversation_id, m.sequence, m.role, m.content, m.created_at
                FROM memory_index_work w JOIN messages m ON m.id = w.message_id
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.scope = ? AND c.updated_at >= ? AND w.fingerprint = ?
                  AND w.chunker_version = ? AND w.lexical_contract = ? AND w.source_hash IS NOT NULL
                  AND w.attempts < ? AND (w.state = 'pending' OR (w.state = 'leased' AND w.lease_until <= ?))
                ORDER BY w.attempts, m.id LIMIT 1''',
                (self.store.scope, self.repository._cutoff(), fingerprint, state['chunker_version'],
                 state['lexical_contract'], max_attempts, now)).fetchone()
            if row is None:
                return None
            if content_hash(row['content']) != row['source_hash']:
                connection.execute('DELETE FROM memory_index_work WHERE message_id = ? AND fingerprint = ?',
                                   (row['message_id'], fingerprint))
                return None
            token = uuid.uuid4().hex
            until = _timestamp(self.store._clock() + timedelta(seconds=lease_seconds))
            connection.execute('''UPDATE memory_index_work SET state = 'leased', lease_id = ?, lease_until = ?,
                attempts = attempts + 1, error_code = NULL WHERE message_id = ? AND fingerprint = ? AND chunker_version = ?''',
                (token, until, row['message_id'], fingerprint, state['chunker_version']))
            message = Message(row['message_id'], row['conversation_id'], row['sequence'], row['role'], row['content'], row['created_at'])
            return IndexLease(message, fingerprint, row['source_hash'], state['chunker_version'],
                              state['lexical_contract'], token, row['attempts'] + 1)

    def _checked(self, connection, lease):
        row = connection.execute('''SELECT w.*, m.content, s.embedding_fingerprint FROM memory_index_work w
            JOIN messages m ON m.id = w.message_id JOIN conversations c ON c.id = m.conversation_id
            JOIN memory_index_state s ON s.scope = c.scope
            WHERE w.message_id = ? AND w.fingerprint = ? AND w.chunker_version = ?
              AND c.scope = ? AND c.updated_at >= ? AND s.indexing_enabled = 1
              AND s.chunker_version = w.chunker_version AND s.lexical_contract = w.lexical_contract
              AND w.state = 'leased' AND w.lease_id = ? AND w.lease_until > ?''',
            (lease.message.id, lease.fingerprint, lease.chunker_version, self.store.scope,
             self.repository._cutoff(), lease.token, _timestamp(self.store._clock()))).fetchone()
        if row is None or row['source_hash'] != lease.source_hash or content_hash(row['content']) != lease.source_hash:
            return None
        active = dict(lexical_contract=row['lexical_contract'], chunker_version=row['chunker_version'],
                      embedding_fingerprint=row['embedding_fingerprint'])
        return row if work_fingerprint(active) == lease.fingerprint else None

    def fail(self, lease, *, code='ENCODING_FAILED', max_attempts=3):
        if code not in ('ENCODING_FAILED', 'SOURCE_TOO_LARGE', 'UNENCODABLE_SOURCE'):
            raise ValueError('Invalid index error code.')
        with self.store._transaction(write=True) as connection:
            if self._checked(connection, lease) is None:
                return False
            state = 'failed' if lease.attempts >= max_attempts or code != 'ENCODING_FAILED' else 'pending'
            connection.execute('''UPDATE memory_index_work SET state = ?, lease_id = NULL,
                lease_until = NULL, error_code = ? WHERE message_id = ? AND fingerprint = ? AND chunker_version = ?''',
                (state, code, lease.message.id, lease.fingerprint, lease.chunker_version))
        return True

    def complete(self, lease, chunks, vectors, provider):
        # Validation/computation outside the write lock. No partial generation is
        # published: chunks, terms, vectors, receipt and work deletion commit together.
        from retrieval_embeddings import normalized_vector
        if not chunks or len(chunks) != len(vectors):
            raise ValueError('Invalid index result shape.')
        raw = [struct.pack(f'<{provider.dimension}f', *normalized_vector(v, provider.dimension)) for v in vectors]
        with self.store._transaction(write=True) as connection:
            row = self._checked(connection, lease)
            if row is None or row['embedding_fingerprint'] != provider.fingerprint:
                return False
            text = row['content']
            if any(c.ordinal != i or c.end_char > len(text) or not text[c.start_char:c.end_char].strip()
                   for i, c in enumerate(chunks)):
                raise ValueError('Invalid index chunk bounds.')
            connection.execute('DELETE FROM memory_chunks WHERE message_id = ?', (lease.message.id,))
            for chunk, vector in zip(chunks, raw):
                chunk_id = uuid.uuid4().hex
                connection.execute('''INSERT INTO memory_chunks
                    (id, message_id, ordinal, start_char, end_char, source_hash, chunk_hash,
                     chunker_version, token_length, lexical_contract) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (chunk_id, lease.message.id, chunk.ordinal, chunk.start_char, chunk.end_char,
                     lease.source_hash, content_hash(text[chunk.start_char:chunk.end_char]),
                     lease.chunker_version, chunk.token_length, lease.lexical_contract))
                connection.executemany('INSERT INTO memory_chunk_terms VALUES (?, ?, ?)',
                                       [(chunk_id, term, count) for term, count in chunk.terms])
                connection.execute("INSERT INTO memory_embeddings VALUES (?, ?, ?, 'float32-le', ?)",
                                   (chunk_id, provider.fingerprint, provider.dimension, vector))
            connection.execute('''INSERT INTO memory_index_receipts VALUES (?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET fingerprint = excluded.fingerprint,
                    source_hash = excluded.source_hash, chunk_count = excluded.chunk_count''',
                (lease.message.id, lease.fingerprint, lease.source_hash, len(chunks)))
            connection.execute('DELETE FROM memory_index_work WHERE message_id = ? AND fingerprint = ? AND chunker_version = ?',
                               (lease.message.id, lease.fingerprint, lease.chunker_version))
            connection.execute('UPDATE memory_index_state SET generation = generation + 1 WHERE scope = ?', (self.store.scope,))
        return True


class SourceTooLargeError(ValueError):
    pass


def chunk_message(text, provider, *, target_tokens=192, overlap_chars=64, max_chars=200000, max_chunks=1024):
    """Original character offsets, verified encoder-token fit, deterministic overlap.

    Prefix search need not find the mathematically longest BPE prefix: every
    chosen prefix is checked independently, so nonmonotonic token counts cannot
    authorize oversized chunks. No decoding/round-tripping or silent truncation.
    """
    if len(text) > max_chars:
        raise SourceTooLargeError('Index source exceeds its configured bound.')
    if (any(type(v) is not int for v in (target_tokens, overlap_chars, max_chars, max_chunks))
            or target_tokens < 1 or overlap_chars < 0 or max_chars < 1 or max_chunks < 1):
        raise ValueError('Invalid chunking bounds.')
    limit = min(target_tokens, provider.max_query_tokens)
    chunks, start = [], 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break
        low, high, best = 1, min(len(text) - start, 4096), 0
        while low <= high:
            middle = (low + high) // 2
            count = provider.count_query_tokens(text[start:start + middle])
            if type(count) is not int or count < 0:
                raise ValueError('Invalid encoder token count.')
            if count <= limit:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError('No safe source chunk can be encoded.')
        end = start + best
        if end < len(text):
            # Prefer word boundaries when a checked shorter prefix is safe.
            for boundary in range(end - 1, start + best // 2, -1):
                if text[boundary].isspace():
                    candidate_count = provider.count_query_tokens(text[start:boundary])
                    if type(candidate_count) is int and 0 < candidate_count <= limit:
                        end = boundary
                    break
        excerpt = text[start:end]
        count = provider.count_query_tokens(excerpt)
        if not 0 < count <= limit:
            raise ValueError('Unsafe source chunk token count.')
        chunks.append(ChunkWrite(len(chunks), start, end, count,
                                 tuple(sorted(Counter(memory_tokenize(excerpt)).items()))))
        if len(chunks) > max_chunks:
            raise SourceTooLargeError('Index source exceeds its configured chunk bound.')
        if end == len(text):
            break
        start = end - min(overlap_chars, (end - start) // 4)
    if not chunks:
        raise ValueError('No nonempty source chunks.')
    return tuple(chunks)


class MemoryIndexer:
    def __init__(self, repository, embeddings):
        if not isinstance(repository, MemoryRetrievalStore) or not isinstance(embeddings, QueryEmbeddings):
            raise ValueError('Indexing requires a scoped store and trusted CPU embedding adapter.')
        self.repository, self.embeddings = repository, embeddings
        self.queue = IndexQueue(repository)
        self.chunker_version = 'memory-token-chunks-v1:' + content_hash(json.dumps(
            [embeddings.fingerprint, embeddings.max_query_tokens, 192, 64, 200000, 1024]))
        with repository.store._transaction(write=True) as connection:
            old = connection.execute('SELECT * FROM memory_index_state WHERE scope = ?', (repository.store.scope,)).fetchone()
            same = (old is not None and old['lexical_contract'] == MEMORY_LEXICAL_CONTRACT
                    and old['chunker_version'] == self.chunker_version
                    and old['embedding_fingerprint'] == embeddings.fingerprint)
            if same:
                connection.execute('UPDATE memory_index_state SET indexing_enabled = 1 WHERE scope = ?', (repository.store.scope,))
            else:
                connection.execute('''INSERT INTO memory_index_state
                    (scope, lexical_contract, chunker_version, embedding_fingerprint, indexing_enabled)
                    VALUES (?, ?, ?, ?, 1) ON CONFLICT(scope) DO UPDATE SET
                    lexical_contract = excluded.lexical_contract, chunker_version = excluded.chunker_version,
                    embedding_fingerprint = excluded.embedding_fingerprint, scan_after = '', indexing_enabled = 1,
                    generation = memory_index_state.generation + 1''',
                    (repository.store.scope, MEMORY_LEXICAL_CONTRACT, self.chunker_version, embeddings.fingerprint))

    def step(self):
        expected = work_fingerprint(dict(lexical_contract=MEMORY_LEXICAL_CONTRACT,
                                        chunker_version=self.chunker_version,
                                        embedding_fingerprint=self.embeddings.fingerprint))
        self.queue.discover(expected_fingerprint=expected)
        lease = self.queue.claim(expected_fingerprint=expected)
        if lease is None:
            return 'idle'
        try:
            chunks = chunk_message(lease.message.content, self.embeddings)
            vectors = [self.embeddings.encode(lease.message.content[c.start_char:c.end_char]) for c in chunks]
            if any(v is None for v in vectors):
                raise ValueError('Chunk no longer fits the encoder.')
        except SourceTooLargeError:
            self.queue.fail(lease, code='SOURCE_TOO_LARGE')
            return 'failed'
        except Exception:
            self.queue.fail(lease)
            return 'failed'
        return 'completed' if self.queue.complete(lease, chunks, vectors, self.embeddings) else 'stale'
