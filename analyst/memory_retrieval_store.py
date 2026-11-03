"""Scoped derived-data storage in the conversation database.

No ranking, tokenization, model loading, backfill loop, or job database access.
All excerpt and vector work must be computed before calling write methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta, timezone
import math
import struct
import uuid

from memory_retrieval_contracts import content_hash, require_hash, require_text
from memory_store import ConversationNotFoundError, MemoryStore, _timestamp


def migrate_4_to_5(connection):
    """Add empty derived tables atomically; never index existing content here."""
    statements = (
        """CREATE TABLE memory_chunks (
            id TEXT PRIMARY KEY NOT NULL,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
            start_char INTEGER NOT NULL CHECK(start_char >= 0),
            end_char INTEGER NOT NULL CHECK(end_char > start_char),
            source_hash TEXT NOT NULL CHECK(length(source_hash) = 64),
            chunk_hash TEXT NOT NULL CHECK(length(chunk_hash) = 64),
            chunker_version TEXT NOT NULL CHECK(length(trim(chunker_version)) > 0),
            token_length INTEGER NOT NULL CHECK(token_length > 0),
            lexical_contract TEXT NOT NULL CHECK(length(trim(lexical_contract)) > 0),
            UNIQUE(message_id, chunker_version, ordinal)
        )""",
        """CREATE TABLE memory_chunk_terms (
            chunk_id TEXT NOT NULL REFERENCES memory_chunks(id) ON DELETE CASCADE,
            term TEXT NOT NULL CHECK(length(term) > 0),
            frequency INTEGER NOT NULL CHECK(frequency > 0),
            PRIMARY KEY(chunk_id, term)
        )""",
        'CREATE INDEX memory_terms_lookup ON memory_chunk_terms(term, chunk_id)',
        """CREATE TABLE memory_embeddings (
            chunk_id TEXT NOT NULL REFERENCES memory_chunks(id) ON DELETE CASCADE,
            fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
            dimension INTEGER NOT NULL CHECK(dimension > 0),
            encoding TEXT NOT NULL CHECK(encoding = 'float32-le'),
            vector BLOB NOT NULL CHECK(typeof(vector) = 'blob' AND length(vector) = dimension * 4),
            PRIMARY KEY(chunk_id, fingerprint)
        )""",
        """CREATE TABLE memory_index_work (
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
            chunker_version TEXT NOT NULL CHECK(length(trim(chunker_version)) > 0),
            state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending', 'leased', 'failed')),
            lease_id TEXT,
            lease_until TEXT,
            attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
            error_code TEXT CHECK(length(error_code) BETWEEN 1 AND 64),
            PRIMARY KEY(message_id, fingerprint, chunker_version),
            CHECK((state = 'leased' AND lease_id IS NOT NULL AND lease_until IS NOT NULL)
               OR (state != 'leased' AND lease_id IS NULL AND lease_until IS NULL))
        )""",
        'CREATE INDEX memory_work_pending ON memory_index_work(state, lease_until, message_id)',
        """CREATE TABLE memory_index_state (
            scope TEXT PRIMARY KEY NOT NULL CHECK(length(trim(scope)) > 0),
            generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
            lexical_contract TEXT NOT NULL,
            chunker_version TEXT NOT NULL,
            embedding_fingerprint TEXT CHECK(length(embedding_fingerprint) = 64)
        )""",
        # State rows exist only after explicit contract configuration. These
        # triggers invalidate caches without creating indexes for disabled use.
        """CREATE TRIGGER memory_source_insert AFTER INSERT ON messages BEGIN
            UPDATE memory_index_state SET generation = generation + 1
            WHERE scope = (SELECT scope FROM conversations WHERE id = NEW.conversation_id);
        END""",
        """CREATE TRIGGER memory_source_update AFTER UPDATE ON messages BEGIN
            DELETE FROM memory_chunks WHERE message_id = OLD.id;
            DELETE FROM memory_index_work WHERE message_id = OLD.id;
            UPDATE memory_index_state SET generation = generation + 1
            WHERE scope IN (SELECT scope FROM conversations
                            WHERE id IN (OLD.conversation_id, NEW.conversation_id));
        END""",
        """CREATE TRIGGER memory_source_delete BEFORE DELETE ON messages BEGIN
            UPDATE memory_index_state SET generation = generation + 1
            WHERE scope = (SELECT scope FROM conversations WHERE id = OLD.conversation_id);
        END""",
        """CREATE TRIGGER memory_conversation_delete BEFORE DELETE ON conversations BEGIN
            UPDATE memory_index_state SET generation = generation + 1 WHERE scope = OLD.scope;
        END""",
        """CREATE TRIGGER memory_conversation_update AFTER UPDATE OF scope, updated_at ON conversations BEGIN
            UPDATE memory_index_state SET generation = generation + 1 WHERE scope IN (OLD.scope, NEW.scope);
        END""",
    )
    for statement in statements:
        connection.execute(statement)


@dataclass(frozen=True)
class ChunkWrite:
    ordinal: int
    start_char: int
    end_char: int
    token_length: int
    terms: tuple[tuple[str, int], ...] = field(default=(), repr=False)

    def __post_init__(self):
        for name in ('ordinal', 'start_char', 'end_char', 'token_length'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer.')
        if self.ordinal < 0 or not 0 <= self.start_char < self.end_char or self.token_length <= 0:
            raise ValueError('Invalid chunk bounds.')
        if type(self.terms) is not tuple:
            raise ValueError('terms must be immutable.')
        names = set()
        for item in self.terms:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError('Invalid term entry.')
            term, frequency = item
            require_text(term, 'term')
            if term in names or type(frequency) is not int or frequency <= 0:
                raise ValueError('Invalid or duplicate term frequency.')
            names.add(term)


@dataclass(frozen=True)
class StoredChunk:
    id: str
    message_id: str
    conversation_id: str
    sequence: int
    role: str
    created_at: str
    start_char: int
    end_char: int
    text: str = field(repr=False)
    source_hash: str
    chunk_hash: str
    token_length: int
    chunker_version: str
    terms: tuple[tuple[str, int], ...] = field(repr=False)


@dataclass(frozen=True)
class MemoryIndexSnapshot:
    generation: int
    lexical_contract: str
    chunker_version: str
    embedding_fingerprint: str | None
    retrieved_at: str
    oldest_updated_at: str | None
    complete: bool
    chunks: tuple[StoredChunk, ...] = field(repr=False)
    vectors: tuple[tuple[str, int, bytes], ...] = field(repr=False)


class MemoryRetrievalStore:
    """Internal repository bound to a trusted MemoryStore, never client scope."""
    def __init__(self, store: MemoryStore):
        if not isinstance(store, MemoryStore):
            raise ValueError('store must be a configured MemoryStore.')
        self.store = store

    def _cutoff(self, now=None):
        if self.store.retention_days is None:
            return ''
        now = self.store._clock() if now is None else now
        _timestamp(now)  # Require an aware clock, matching persistence semantics.
        return _timestamp(now.astimezone(timezone.utc) - timedelta(days=self.store.retention_days))

    def _authorize(self, connection, conversation_id, cutoff):
        row = self.store._require_conversation(connection, conversation_id)
        if row['updated_at'] < cutoff:
            raise ConversationNotFoundError('Conversation was not found in the configured scope.')

    def authorize(self, conversation_id):
        with self.store._transaction() as connection:
            self._authorize(connection, conversation_id, self._cutoff())

    def configure(self, *, lexical_contract, chunker_version, embedding_fingerprint=None):
        """Trusted explicit contract selection; does not schedule or perform work."""
        require_text(lexical_contract, 'lexical_contract')
        require_text(chunker_version, 'chunker_version')
        if embedding_fingerprint is not None:
            require_hash(embedding_fingerprint, 'embedding_fingerprint')
        with self.store._transaction(write=True) as connection:
            connection.execute('''
                INSERT INTO memory_index_state(scope, lexical_contract, chunker_version, embedding_fingerprint)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    lexical_contract = excluded.lexical_contract,
                    chunker_version = excluded.chunker_version,
                    embedding_fingerprint = excluded.embedding_fingerprint,
                    generation = memory_index_state.generation + 1
            ''', (self.store.scope, lexical_contract, chunker_version, embedding_fingerprint))

    def state(self):
        with self.store._transaction() as connection:
            row = connection.execute('SELECT * FROM memory_index_state WHERE scope = ?',
                                     (self.store.scope,)).fetchone()
            return dict(row) if row is not None else None

    def replace_chunks(self, message_id, *, source_hash, lexical_contract, chunker_version, chunks):
        """Commit precomputed chunks only if the scoped source still matches.

        False means stale/deleted/expired/foreign source or superseded contract.
        No private content is included in errors. All writes are atomic.
        """
        require_hash(source_hash, 'source_hash')
        require_text(lexical_contract, 'lexical_contract')
        require_text(chunker_version, 'chunker_version')
        if (type(chunks) is not tuple or not chunks
                or not all(isinstance(item, ChunkWrite) for item in chunks)
                or [c.ordinal for c in chunks] != list(range(len(chunks)))):
            raise ValueError('Supply a nonempty immutable sequence of contiguous chunks.')
        with self.store._transaction(write=True) as connection:
            row = connection.execute('''
                SELECT m.content FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                JOIN memory_index_state s ON s.scope = c.scope
                WHERE m.id = ? AND c.scope = ? AND c.updated_at >= ? AND s.chunker_version = ?
                  AND s.lexical_contract = ?
            ''', (message_id, self.store.scope, self._cutoff(), chunker_version, lexical_contract)).fetchone()
            if row is None or content_hash(row['content']) != source_hash:
                return False
            source = row['content']
            for chunk in chunks:
                if chunk.end_char > len(source) or not source[chunk.start_char:chunk.end_char].strip():
                    raise ValueError('Chunk lies outside its source or is empty.')
            connection.execute('DELETE FROM memory_index_receipts WHERE message_id = ?', (message_id,))
            connection.execute('DELETE FROM memory_chunks WHERE message_id = ?', (message_id,))
            for chunk in chunks:
                chunk_id = uuid.uuid4().hex
                excerpt = source[chunk.start_char:chunk.end_char]
                connection.execute('''INSERT INTO memory_chunks
                    (id, message_id, ordinal, start_char, end_char, source_hash, chunk_hash, chunker_version, token_length, lexical_contract)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (chunk_id, message_id, chunk.ordinal, chunk.start_char, chunk.end_char,
                     source_hash, content_hash(excerpt), chunker_version, chunk.token_length, lexical_contract))
                connection.executemany('INSERT INTO memory_chunk_terms VALUES (?, ?, ?)',
                                       [(chunk_id, term, count) for term, count in chunk.terms])
            connection.execute('UPDATE memory_index_state SET generation = generation + 1 WHERE scope = ?',
                               (self.store.scope,))
        return True

    def candidates(self, *, current_conversation_id):
        snapshot = self.read_snapshot(current_conversation_id=current_conversation_id)
        return () if snapshot is None else snapshot.chunks

    def read_snapshot(self, *, current_conversation_id, embedding_fingerprint=None):
        """Read scope, contracts, candidates, vectors and coverage in one transaction.

        Only matching, authorized vectors are materialized; models run after the
        transaction closes. Coverage is conservative and requires complete source
        spans, not merely one indexed chunk per message.
        """
        now = self.store._clock()
        cutoff = self._cutoff(now)
        with self.store._transaction() as connection:
            self._authorize(connection, current_conversation_id, cutoff)
            state = connection.execute('SELECT * FROM memory_index_state WHERE scope = ?',
                                       (self.store.scope,)).fetchone()
            if state is None:
                return None
            coverage = connection.execute('''SELECT COUNT(*) AS messages, MIN(c.updated_at) AS oldest
                FROM messages m JOIN conversations c ON c.id = m.conversation_id
                WHERE c.scope = ? AND c.updated_at >= ?''', (self.store.scope, cutoff)).fetchone()
            rows = connection.execute('''
                SELECT ch.*, m.content, m.conversation_id, m.sequence, m.role, m.created_at
                FROM memory_chunks ch JOIN messages m ON m.id = ch.message_id
                JOIN conversations c ON c.id = m.conversation_id
                JOIN memory_index_state s ON s.scope = c.scope
                WHERE c.scope = ? AND c.updated_at >= ? AND ch.chunker_version = s.chunker_version
                  AND ch.lexical_contract = s.lexical_contract
                ORDER BY m.conversation_id, m.sequence, ch.ordinal, ch.id
            ''', (self.store.scope, cutoff)).fetchall()
            result, vectors, spans, lengths, sources = [], [], {}, {}, {}
            for row in rows:
                source = row['content']
                excerpt = source[row['start_char']:row['end_char']]
                if (row['end_char'] > len(source) or content_hash(source) != row['source_hash']
                        or content_hash(excerpt) != row['chunk_hash']):
                    continue
                terms = tuple(tuple(term) for term in connection.execute(
                    'SELECT term, frequency FROM memory_chunk_terms WHERE chunk_id = ? ORDER BY term',
                    (row['id'],)))
                result.append(StoredChunk(
                    row['id'], row['message_id'], row['conversation_id'], row['sequence'],
                    row['role'], row['created_at'], row['start_char'], row['end_char'], excerpt,
                    row['source_hash'], row['chunk_hash'], row['token_length'], row['chunker_version'], terms))
                spans.setdefault(row['message_id'], []).append((row['start_char'], row['end_char']))
                lengths[row['message_id']] = len(source)
                sources[row['message_id']] = source
                if embedding_fingerprint is not None and embedding_fingerprint == state['embedding_fingerprint']:
                    vector = connection.execute('''SELECT dimension, vector FROM memory_embeddings
                        WHERE chunk_id = ? AND fingerprint = ? AND encoding = 'float32-le'
                        ''',
                        (row['id'], embedding_fingerprint)).fetchone()
                    if vector is not None:
                        vectors.append((row['id'], vector['dimension'], vector['vector']))
            complete = len(spans) == coverage['messages']
            for message_id, windows in spans.items():
                end = 0
                for left, right in sorted(windows):
                    if left > end and sources[message_id][end:left].strip():
                        complete = False
                    end = max(end, right)
                if end != lengths[message_id] and sources[message_id][end:].strip():
                    complete = False
            return MemoryIndexSnapshot(state['generation'], state['lexical_contract'], state['chunker_version'],
                                       state['embedding_fingerprint'], _timestamp(now), coverage['oldest'],
                                       complete, tuple(result), tuple(vectors))

    def snapshot_current(self, snapshot, *, current_conversation_id):
        """Fail closed if sources/contracts changed or retention advanced past a source."""
        with self.store._transaction() as connection:
            cutoff = self._cutoff()
            self._authorize(connection, current_conversation_id, cutoff)
            state = connection.execute('SELECT * FROM memory_index_state WHERE scope = ?',
                                       (self.store.scope,)).fetchone()
            return (state is not None and state['generation'] == snapshot.generation
                    and state['lexical_contract'] == snapshot.lexical_contract
                    and state['chunker_version'] == snapshot.chunker_version
                    and state['embedding_fingerprint'] == snapshot.embedding_fingerprint
                    and (snapshot.oldest_updated_at is None or snapshot.oldest_updated_at >= cutoff))

    def evidence_sources(self, *, current_conversation_id, evidence):
        """Revalidate exact saved sources without reranking or using index state."""
        cutoff = self._cutoff()
        sources = []
        with self.store._transaction() as connection:
            try:
                self._authorize(connection, current_conversation_id, cutoff)
            except ConversationNotFoundError:
                return None
            for item in evidence:
                row = connection.execute('''SELECT m.*, c.title FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.id = ? AND c.id = ? AND c.scope = ? AND c.updated_at >= ?''',
                    (item.message_id, item.conversation_id, self.store.scope, cutoff)).fetchone()
                if row is None:
                    return None
                text = row['content']
                if (row['role'] != item.role or row['sequence'] != item.sequence
                        or row['created_at'] != item.created_at or item.end_char > len(text)
                        or content_hash(text) != item.source_hash
                        or text[item.start_char:item.end_char] != item.excerpt):
                    return None
                sources.append({"evidence_id": item.evidence_id, "conversation_id": item.conversation_id,
                                "title": row['title'], "role": item.role, "created_at": item.created_at,
                                "excerpt": item.excerpt[:600], "truncated": len(item.excerpt) > 600})
        return sources

    def validate_evidence(self, *, current_conversation_id, evidence):
        return self.evidence_sources(current_conversation_id=current_conversation_id, evidence=evidence) is not None

    def put_embedding(self, chunk_id, *, source_hash, fingerprint, dimension, vector):
        """Store a precomputed normalized vector after revalidating source/contract."""
        require_hash(source_hash, 'source_hash')
        require_hash(fingerprint, 'fingerprint')
        if (type(dimension) is not int or not 1 <= dimension <= 65536
                or type(vector) is not bytes or len(vector) != dimension * 4):
            raise ValueError('Invalid float32-le vector shape.')
        values = struct.unpack(f'<{dimension}f', vector)
        if not all(math.isfinite(v) for v in values) or abs(sum(v*v for v in values) - 1) > .001:
            raise ValueError('Vector must be finite and normalized.')
        with self.store._transaction(write=True) as connection:
            row = connection.execute('''
                SELECT ch.source_hash, m.content FROM memory_chunks ch
                JOIN messages m ON m.id = ch.message_id
                JOIN conversations c ON c.id = m.conversation_id
                JOIN memory_index_state s ON s.scope = c.scope
                WHERE ch.id = ? AND c.scope = ? AND c.updated_at >= ?
                  AND ch.chunker_version = s.chunker_version AND ch.lexical_contract = s.lexical_contract
                  AND s.embedding_fingerprint = ?
            ''', (chunk_id, self.store.scope, self._cutoff(), fingerprint)).fetchone()
            if row is None or row['source_hash'] != source_hash or content_hash(row['content']) != source_hash:
                return False
            connection.execute('''INSERT INTO memory_embeddings VALUES (?, ?, ?, 'float32-le', ?)
                ON CONFLICT(chunk_id, fingerprint) DO UPDATE SET
                dimension = excluded.dimension, vector = excluded.vector''',
                (chunk_id, fingerprint, dimension, vector))
            connection.execute('UPDATE memory_index_state SET generation = generation + 1 WHERE scope = ?',
                               (self.store.scope,))
        return True
