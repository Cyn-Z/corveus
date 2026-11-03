"""Phase 1B checks using only temporary databases and synthetic content."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

from memory_config import MemorySettings, load_memory_settings
from memory_manager import MemoryManager
from memory_runtime import build_memory_manager
from memory_store import MemoryStore, ConversationNotFoundError, SCHEMA_VERSION
from memory_retrieval_contracts import (
    MemoryEvidence, MemoryRetrievalPolicy, MemoryRetrievalResult, content_hash,
)
from memory_retrieval_store import ChunkWrite, MemoryRetrievalStore, migrate_4_to_5


class RetrievalStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'memory.sqlite3'
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.store = MemoryStore(self.path, scope='a', retention_days=30, clock=lambda: self.now)
        self.derived = MemoryRetrievalStore(self.store)
        self.conversation = self.store.create_conversation()
        self.user, self.assistant = self.store.append_exchange(
            self.conversation.id, user_content='port 9443', assistant_content='untrusted answer')
        self.fingerprint = content_hash('synthetic-model-contract')
        self.derived.configure(lexical_contract='lex-v1', chunker_version='chunk-v1',
                               embedding_fingerprint=self.fingerprint)

    def put(self, message=None, derived=None, **overrides):
        message = message or self.user
        args = dict(source_hash=content_hash(message.content), lexical_contract='lex-v1',
                    chunker_version='chunk-v1', chunks=(ChunkWrite(
                        0, 0, len(message.content), 3, (('port', 1), ('9443', 1))),))
        args.update(overrides)
        return (derived or self.derived).replace_chunks(message.id, **args)

    def candidates(self):
        return self.derived.candidates(current_conversation_id=self.conversation.id)

    def test_scoped_candidates_and_authorization(self):
        self.assertTrue(self.put())
        other = MemoryStore(self.path, scope='b', clock=lambda: self.now)
        conversation = other.create_conversation()
        user, _ = other.append_exchange(conversation.id, user_content='secret', assistant_content='secret')
        derived = MemoryRetrievalStore(other)
        derived.configure(lexical_contract='lex-v1', chunker_version='chunk-v1')
        self.assertTrue(self.put(user, derived))
        self.assertFalse(self.put(user))
        self.assertEqual([c.message_id for c in self.candidates()], [self.user.id])
        with self.assertRaises(ConversationNotFoundError):
            self.derived.candidates(current_conversation_id=conversation.id)
        with self.assertRaises(ConversationNotFoundError):
            self.derived.authorize('missing')

    def test_stale_commit_and_contract_change(self):
        self.assertFalse(self.put(source_hash=content_hash('stale')))
        self.assertTrue(self.put())
        self.derived.configure(lexical_contract='lex-v2', chunker_version='chunk-v1')
        self.assertEqual(self.candidates(), ())
        self.assertFalse(self.put())
        self.assertTrue(self.put(lexical_contract='lex-v2'))
        self.derived.configure(lexical_contract='lex-v2', chunker_version='chunk-v2')
        self.assertEqual(self.candidates(), ())
        self.assertFalse(self.put(lexical_contract='lex-v2'))

    def test_invalid_chunks_rollback_without_losing_existing_data(self):
        self.put()
        before = self.candidates()
        with self.assertRaises(ValueError):
            self.put(chunks=(ChunkWrite(0, 0, 900, 1),))
        self.assertEqual(before, self.candidates())
        for terms in ((('port', 1), ('port', 2)), (('port', True),)):
            with self.assertRaises(ValueError):
                ChunkWrite(0, 0, 1, 1, terms)

    def test_partial_chunk_write_failure_preserves_previous_snapshot(self):
        self.put()
        before = self.candidates()
        generation = self.derived.state()['generation']
        with self.store._transaction(write=True) as connection:
            connection.execute("""CREATE TRIGGER injected_term_failure BEFORE INSERT ON memory_chunk_terms
                                  BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.put()
        self.assertEqual(self.candidates(), before)
        self.assertEqual(self.derived.state()['generation'], generation)

    def test_precomputed_vector_revalidates_source_and_contract(self):
        self.put()
        chunk = self.candidates()[0]
        args = dict(source_hash=chunk.source_hash, fingerprint=self.fingerprint,
                    dimension=2, vector=struct.pack('<2f', 1, 0))
        self.assertTrue(self.derived.put_embedding(chunk.id, **args))
        self.assertFalse(self.derived.put_embedding(chunk.id, **dict(args, source_hash=content_hash('old'))))
        for values in ((float('nan'), 0), (0, 0), (2, 0)):
            with self.assertRaises(ValueError):
                self.derived.put_embedding(chunk.id, **dict(args, vector=struct.pack('<2f', *values)))
        self.derived.configure(lexical_contract='lex-v1', chunker_version='chunk-v1',
                               embedding_fingerprint=content_hash('new-model'))
        self.assertFalse(self.derived.put_embedding(chunk.id, **args))
        self.store.delete_conversation(self.conversation.id)
        self.assertFalse(self.derived.put_embedding(chunk.id, **args))

    def test_deletion_cascades_all_derived_records_and_invalidates_generation(self):
        self.put()
        chunk = self.candidates()[0]
        self.derived.put_embedding(chunk.id, source_hash=chunk.source_hash,
                                   fingerprint=self.fingerprint, dimension=1, vector=struct.pack('<f', 1))
        with self.store._transaction(write=True) as connection:
            connection.execute('INSERT INTO memory_index_work(message_id, fingerprint, chunker_version) VALUES (?, ?, ?)',
                               (self.user.id, self.fingerprint, 'chunk-v1'))
        before = self.derived.state()['generation']
        self.store.delete_conversation(self.conversation.id)
        self.assertGreater(self.derived.state()['generation'], before)
        with self.store._transaction() as connection:
            for table in ('memory_chunks', 'memory_chunk_terms', 'memory_embeddings', 'memory_index_work'):
                self.assertEqual(connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0], 0)
        self.assertFalse(self.put())

    def test_retention_boundary_and_cleanup(self):
        self.put()
        self.now += timedelta(days=30)
        self.assertEqual(len(self.candidates()), 1)  # Exact cutoff remains eligible.
        self.now += timedelta(microseconds=1)
        with self.assertRaises(ConversationNotFoundError):
            self.candidates()
        self.assertFalse(self.put())
        self.assertEqual(self.store.prune_expired_conversations(), 1)
        with self.store._transaction() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM memory_chunks').fetchone()[0], 0)

    def test_expired_history_filtered_when_current_conversation_is_active(self):
        self.put()
        self.now += timedelta(days=31)
        current = self.store.create_conversation()
        self.assertEqual(self.derived.candidates(current_conversation_id=current.id), ())

    def test_source_update_invalidates_derived_data_and_generation(self):
        self.put()
        generation = self.derived.state()['generation']
        with self.store._transaction(write=True) as connection:
            connection.execute('UPDATE messages SET content = ? WHERE id = ?', ('new fact', self.user.id))
        self.assertEqual(self.candidates(), ())
        self.assertGreater(self.derived.state()['generation'], generation)
        self.assertFalse(self.put())

    def test_corrupt_excerpt_is_not_returned(self):
        self.put()
        with self.store._transaction(write=True) as connection:
            connection.execute('UPDATE memory_chunks SET chunk_hash = ?', (content_hash('wrong'),))
        self.assertEqual(self.candidates(), ())

    def test_receipt_retry_does_not_duplicate_or_reindex_chunks(self):
        job = 'a' * 32
        args = dict(user_content='question', assistant_content='answer', job_id=job)
        pair = self.store.append_exchange(self.conversation.id, **args)
        self.put(pair[0])
        generation = self.derived.state()['generation']
        self.assertEqual(self.store.append_exchange(self.conversation.id, **args), pair)
        self.assertEqual(self.derived.state()['generation'], generation)
        self.assertEqual(len(self.candidates()), 1)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'legacy.sqlite3'
        with sqlite3.connect(self.path) as c:
            for migration in (MemoryStore._migrate_0_to_1, MemoryStore._migrate_1_to_2,
                              MemoryStore._migrate_2_to_3, MemoryStore._migrate_3_to_4):
                migration(c)
            c.execute('PRAGMA user_version = 4')
            c.execute("INSERT INTO conversations VALUES ('c', 'a', '2026-10-29', '2026-10-29', 'legacy')")
            c.execute("INSERT INTO messages VALUES ('u', 'c', 1, 'user', 'question', '2026-10-29')")
            c.execute("INSERT INTO messages VALUES ('a', 'c', 2, 'assistant', 'answer', '2026-10-29')")

    def test_migration_preserves_history_and_starts_empty(self):
        store = MemoryStore(self.path, scope='a')
        self.assertEqual([m.content for m in store.get_messages('c')], ['question', 'answer'])
        MemoryStore(self.path, scope='a')
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])
            for table in ('memory_chunks', 'memory_chunk_terms', 'memory_embeddings', 'memory_index_work', 'memory_index_state'):
                self.assertEqual(c.execute(f'SELECT count(*) FROM {table}').fetchone()[0], 0)

    def test_concurrent_migrators_serialize(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            stores = list(executor.map(lambda _: MemoryStore(self.path, scope='a'), range(2)))
        self.assertEqual(stores[0].get_messages('c'), stores[1].get_messages('c'))
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_failed_migration_rolls_back_all_new_objects(self):
        def fail(connection):
            migrate_4_to_5(connection)
            raise RuntimeError('injected failure')
        with patch('memory_retrieval_store.migrate_4_to_5', side_effect=fail):
            with self.assertRaises(RuntimeError):
                MemoryStore(self.path, scope='a')
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], 4)
            self.assertEqual(c.execute("SELECT name FROM sqlite_master WHERE name LIKE 'memory_%'").fetchall(), [])
            self.assertEqual(c.execute('SELECT count(*) FROM messages').fetchone()[0], 2)


class RetrievalBoundaryTests(unittest.TestCase):
    def test_disabled_and_incognito_do_not_access_storage(self):
        policy = MemoryRetrievalPolicy()
        manager = MemoryManager(retrieval_enabled=True)
        with patch('memory_manager.MemoryStore') as store:
            self.assertEqual(manager.retrieve('q', current_conversation_id='missing', policy=policy).status, 'disabled')
            store.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(enabled=True, database_path=Path(tmp) / 'm.db', scope='a', retrieval_enabled=True)
            with patch.object(manager._store, '_transaction', side_effect=AssertionError('storage accessed')):
                self.assertEqual(manager.retrieve('q', current_conversation_id=None, policy=policy).status, 'incognito')
            conversation = manager.create_conversation()
            result = manager.retrieve('q', current_conversation_id=conversation.id, policy=policy)
            self.assertEqual(result.status, 'not_ready')
            self.assertEqual(result.evidence, ())
            self.assertEqual(result.policy_fingerprint, policy.fingerprint)
            with self.assertRaises(ConversationNotFoundError):
                manager.retrieve('q', current_conversation_id='foreign', policy=policy)
            manager._retrieval_enabled = False
            with patch.object(manager._store, '_transaction', side_effect=AssertionError('storage accessed')):
                self.assertEqual(manager.retrieve('q', current_conversation_id=conversation.id, policy=policy).status, 'disabled')

    def test_configuration_and_runtime_forward_opt_in_only(self):
        self.assertFalse(load_memory_settings({}).retrieval_enabled)
        self.assertTrue(load_memory_settings({'MEMORY_RETRIEVAL_ENABLED': ' true '}).retrieval_enabled)
        for value in ('', 'yes', '1'):
            with self.assertRaises(ValueError):
                load_memory_settings({'MEMORY_RETRIEVAL_ENABLED': value})
        with self.assertRaises(ValueError):
            MemorySettings(retrieval_enabled=1)
        with patch('memory_runtime.MemoryManager') as manager:
            build_memory_manager(MemorySettings(retrieval_enabled=True))
            self.assertTrue(manager.call_args.kwargs['retrieval_enabled'])

    def test_contracts_are_immutable_and_validate_provenance(self):
        evidence = MemoryEvidence('e', 'c', 'm', 1, 'user', '2026-10-29T00:00:00+00:00',
                                  0, 4, 'fact', content_hash('full fact'), content_hash('fact'),
                                  'v1', 1, .5, 1)
        with self.assertRaises(FrozenInstanceError):
            evidence.excerpt = 'changed'
        with self.assertRaises(ValueError):
            replace(evidence, excerpt='oops')
        with self.assertRaises(ValueError):
            replace(evidence, semantic_score=float('nan'))
        with self.assertRaises(ValueError):
            MemoryRetrievalResult('disabled', (evidence,))
        with self.assertRaises(ValueError):
            MemoryRetrievalResult('ready', (evidence,))
        ready = MemoryRetrievalResult('ready', (evidence,),
                                      policy_fingerprint=MemoryRetrievalPolicy().fingerprint,
                                      embedding_fingerprint=content_hash('model'), index_generation=1,
                                      retrieved_at='2026-10-29T00:00:00+00:00')
        self.assertFalse(ready.index_complete)
        with self.assertRaises(ValueError):
            replace(ready, embedding_fingerprint=None)
        self.assertNotIn('full fact', repr(evidence))
        self.assertNotIn("excerpt='fact'", repr(evidence))


if __name__ == '__main__':
    unittest.main()
