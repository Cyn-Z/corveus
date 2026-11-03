"""Phase 1E lifecycle checks with synthetic text and disposable SQLite files."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from token_usage import TokenUsage

from memory_indexer import MemoryIndexer, IndexQueue, chunk_message, work_fingerprint, migrate_5_to_6, SourceTooLargeError
from memory_manager import MemoryManager
from memory_retrieval_contracts import MemoryRetrievalPolicy, content_hash
from memory_retrieval_store import MemoryRetrievalStore, migrate_4_to_5
from memory_store import MemoryStore, SCHEMA_VERSION
from retrieval_embeddings import QueryEmbeddings


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'memory.sqlite3'
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.encode = Mock(return_value=(1., 0.))
        # Character-level synthetic tokenizer also exercises Unicode offsets.
        self.provider = QueryEmbeddings(content_hash('synthetic local model'), 2, self.encode,
                                          lambda text: len(text) + 2, 32)
        self.memory = MemoryManager(enabled=True, database_path=self.path, scope='a',
                                    retrieval_enabled=True, retrieval_embeddings=self.provider, retention_days=30)
        self.memory._store._clock = lambda: self.now
        self.conversation = self.memory.create_conversation()
        self.repository = MemoryRetrievalStore(self.memory._store)
        self.indexer = MemoryIndexer(self.repository, self.provider)
        self.queue = self.indexer.queue

    def save(self, **kwargs):
        return self.memory.record_completed_exchange(self.conversation.id,
                    user_content='The gateway port is 9443.', assistant_content='The backup retention is thirty days.', **kwargs)

    def rows(self, table):
        with self.memory._store._transaction() as connection:
            return [dict(r) for r in connection.execute(f'SELECT * FROM {table}')]

    def finish(self, lease):
        chunks = chunk_message(lease.message.content, self.provider)
        vectors = [self.provider.encode(lease.message.content[c.start_char:c.end_char]) for c in chunks]
        return self.queue.complete(lease, chunks, vectors, self.provider)

    def test_atomic_exchange_enqueue_and_receipt_retry(self):
        first = self.save(job_id='a' * 32)
        self.assertEqual(len(self.rows('memory_index_work')), 2)
        self.assertEqual(first, self.save(job_id='a' * 32))
        self.assertEqual(len(self.rows('memory_index_work')), 2)
        self.encode.assert_not_called()

    def test_enqueue_failure_rolls_back_messages_usage_and_job_receipt(self):
        with self.memory._store._transaction(write=True) as connection:
            connection.execute("""CREATE TRIGGER reject_index_work BEFORE INSERT ON memory_index_work
                                  BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.save(job_id='a' * 32, usage=TokenUsage('model', 130, 10, 5))
        self.assertEqual(self.memory.get_messages(self.conversation.id), [])
        self.assertEqual(self.rows('job_exchange_receipts'), [])
        self.assertEqual(self.rows('exchange_usage'), [])
        self.assertEqual(self.rows('memory_index_work'), [])

    def test_completion_publishes_chunks_vectors_and_receipts_atomically(self):
        self.save()
        self.assertEqual(self.indexer.step(), 'completed')
        self.assertEqual(self.indexer.step(), 'completed')
        self.assertEqual(self.indexer.step(), 'idle')
        self.assertEqual(len(self.rows('memory_index_receipts')), 2)
        self.assertEqual(self.rows('memory_index_work'), [])
        result = self.memory.retrieve('gateway port 9443', current_conversation_id=self.conversation.id,
                                      policy=MemoryRetrievalPolicy(mode='semantic'))
        self.assertTrue(result.evidence)
        self.assertTrue(result.index_complete)

    def test_discovery_resumes_cursor_and_does_not_reencode_completed_messages(self):
        self.memory._indexing_enabled = False
        pair = self.save()
        self.assertEqual(self.rows('memory_index_work'), [])
        self.assertEqual(self.queue.discover(limit=1), 1)
        first_cursor = self.repository.state()['scan_after']
        reopened = MemoryIndexer(self.repository, self.provider)
        self.assertEqual(self.repository.state()['scan_after'], first_cursor)
        self.assertEqual(reopened.queue.discover(limit=1), 1)
        for _ in pair:
            self.assertTrue(self.finish(self.queue.claim()))
        calls = self.encode.call_count
        for _ in range(5):
            reopened.step()
        self.assertEqual(self.encode.call_count, calls)

    def test_competing_claims_cannot_own_the_same_work(self):
        self.save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _: self.queue.claim(), range(2)))
        self.assertEqual(len({c.message.id for c in claims}), 2)
        self.assertEqual(len({c.token for c in claims}), 2)
        self.assertIsNone(self.queue.claim())

    def test_discovery_claim_and_obsolete_cleanup_are_scope_local(self):
        own_pair = self.save()
        foreign = MemoryManager(enabled=True, database_path=self.path, scope='b', retrieval_enabled=True)
        foreign._store._clock = lambda: self.now
        foreign.start_indexing(self.provider)
        conversation = foreign.create_conversation()
        foreign_pair = foreign.record_completed_exchange(conversation.id, user_content='foreign fact', assistant_content='foreign answer')
        own_claims = (self.queue.claim(), self.queue.claim())
        self.assertEqual({c.message.id for c in own_claims}, {m.id for m in own_pair})
        self.assertIsNone(self.queue.claim())
        new = QueryEmbeddings(content_hash('replacement model'), 2, self.encode, lambda t: len(t)+2, 32)
        MemoryIndexer(self.repository, new).step()
        remaining = {r['message_id'] for r in self.rows('memory_index_work')}
        self.assertTrue({m.id for m in foreign_pair}.issubset(remaining))

    def test_repeated_worker_crashes_exhaust_expired_leases(self):
        self.save()
        first = self.queue.claim(lease_seconds=1)
        other = self.queue.claim()
        for attempt in (2, 3):
            self.now += timedelta(seconds=1)
            recovered = self.queue.claim(lease_seconds=1)
            self.assertEqual(recovered.message.id, first.message.id)
            self.assertEqual(recovered.attempts, attempt)
        self.now += timedelta(seconds=1)
        self.assertIsNone(self.queue.claim())
        row = next(r for r in self.rows('memory_index_work') if r['message_id'] == first.message.id)
        self.assertEqual(row['state'], 'failed')
        self.assertEqual(row['error_code'], 'ATTEMPTS_EXHAUSTED')
        self.encode.assert_not_called()

    def test_oversized_source_is_terminal_without_silent_truncation(self):
        pair = self.memory.record_completed_exchange(self.conversation.id,
                    user_content='x' * 200001, assistant_content='acknowledged')
        for _ in range(2):
            self.indexer.step()
        row = next(r for r in self.rows('memory_index_work') if r['message_id'] == pair[0].id)
        self.assertEqual((row['state'], row['attempts'], row['error_code']), ('failed', 1, 'SOURCE_TOO_LARGE'))
        self.assertTrue(all(c['message_id'] != pair[0].id for c in self.rows('memory_chunks')))

    def test_expired_lease_recovery_rejects_stale_completion(self):
        self.save()
        first = self.queue.claim(lease_seconds=1)
        self.now += timedelta(seconds=1)
        # Claim remaining pending item, then reclaim expired first lease.
        second = self.queue.claim()
        third = self.queue.claim()
        reclaimed = next(c for c in (second, third) if c.message.id == first.message.id)
        self.assertNotEqual(reclaimed.token, first.token)
        self.assertFalse(self.finish(first))
        self.assertTrue(self.finish(reclaimed))

    def test_deleted_or_changed_source_cannot_commit_computed_vectors(self):
        self.save()
        lease = self.queue.claim()
        with self.memory._store._transaction(write=True) as connection:
            connection.execute('UPDATE messages SET content = ? WHERE id = ?', ('changed fact', lease.message.id))
        self.assertFalse(self.finish(lease))
        self.queue.discover()
        lease = self.queue.claim()
        self.memory.delete_conversation(self.conversation.id)
        self.assertFalse(self.finish(lease))
        for table in ('memory_chunks','memory_embeddings','memory_index_work','memory_index_receipts'):
            self.assertEqual(self.rows(table), [])

    def test_scope_and_retention_are_rechecked_at_commit(self):
        self.save()
        lease = self.queue.claim()
        self.now += timedelta(days=31)
        self.assertFalse(self.finish(lease))
        self.assertIsNone(self.queue.claim())
        self.now -= timedelta(days=31)
        with self.memory._store._transaction(write=True) as connection:
            connection.execute('UPDATE conversations SET scope = ? WHERE id = ?', ('foreign', self.conversation.id))
        self.assertFalse(self.finish(lease))
        self.assertIsNone(self.queue.claim())

    def test_contract_change_rejects_old_work_and_rebuilds(self):
        self.save()
        lease = self.queue.claim()
        new_provider = QueryEmbeddings(content_hash('new model'), 2, self.encode, lambda s: len(s)+2, 32)
        new_indexer = MemoryIndexer(self.repository, new_provider)
        self.assertFalse(self.finish(lease))
        self.encode.reset_mock()
        self.assertEqual(self.indexer.step(), 'idle')
        self.encode.assert_not_called()
        self.assertEqual(new_indexer.step(), 'completed')
        self.assertEqual(new_indexer.step(), 'completed')
        self.assertEqual(len(self.rows('memory_index_receipts')), 2)
        self.assertTrue(all(row['fingerprint'] == new_provider.fingerprint for row in self.rows('memory_embeddings')))

    def test_transient_failure_is_bounded_and_diagnostics_omit_private_text(self):
        pair = self.save()
        self.encode.side_effect = RuntimeError('private text should never be stored')
        for _ in range(6):
            self.indexer.step()
        work = self.rows('memory_index_work')
        self.assertTrue(all(row['state'] == 'failed' and row['attempts'] == 3 for row in work))
        self.assertTrue(all(row['error_code'] == 'ENCODING_FAILED' for row in work))
        for _ in range(3):
            self.assertEqual(self.indexer.step(), 'idle')
        self.assertNotIn('private text', repr(work))

    def test_mid_commit_failure_leaves_work_and_prior_generation_intact(self):
        self.save()
        lease = self.queue.claim()
        generation = self.repository.state()['generation']
        with self.memory._store._transaction(write=True) as connection:
            connection.execute("""CREATE TRIGGER reject_vector BEFORE INSERT ON memory_embeddings
                                  BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.finish(lease)
        self.assertEqual(self.rows('memory_chunks'), [])
        self.assertEqual(self.rows('memory_index_receipts'), [])
        self.assertEqual(self.repository.state()['generation'], generation)
        self.assertEqual(next(r for r in self.rows('memory_index_work') if r['message_id'] == lease.message.id)['state'], 'leased')

    def test_disabled_indexing_does_not_read_store_or_embed(self):
        self.memory._indexing_enabled = False
        with patch.object(self.memory._store, '_transaction', side_effect=AssertionError('disabled store access')):
            self.memory.start_indexing(self.provider)
            self.assertEqual(self.memory.indexing_step(), 'disabled')
        disabled = MemoryManager(indexing_enabled=True, retrieval_enabled=True)
        disabled.start_indexing(self.provider)
        self.assertEqual(disabled.indexing_step(), 'disabled')
        self.encode.assert_not_called()

    def test_missing_derived_rows_are_repaired_by_discovery(self):
        self.save()
        self.indexer.step()
        self.indexer.step()
        with self.memory._store._transaction(write=True) as connection:
            connection.execute('DELETE FROM memory_chunks')
        for _ in range(4):
            self.indexer.step()
        self.assertTrue(self.rows('memory_chunks'))


class WorkerLoopTests(unittest.TestCase):
    def test_indexing_runs_only_between_idle_passes(self):
        from service_entrypoint import serve_worker
        class Stop:
            passes = 0
            def is_set(self):
                return self.passes >= 2
            def wait(self, seconds):
                self.passes += 1
        memory = Mock()
        runtime = SimpleNamespace(worker=Mock())
        runtime.worker.run_once.side_effect = [object(), None]
        with patch('service_entrypoint.time.monotonic', return_value=0):
            serve_worker(runtime, memory, Stop())
        memory.indexing_step.assert_called_once_with()
        self.assertEqual(runtime.worker.run_once.call_count, 2)
        runtime.worker.gate.start.assert_not_called()


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'legacy.sqlite3'
        with sqlite3.connect(self.path) as connection:
            for migration in (MemoryStore._migrate_0_to_1, MemoryStore._migrate_1_to_2,
                              MemoryStore._migrate_2_to_3, MemoryStore._migrate_3_to_4, migrate_4_to_5):
                migration(connection)
            connection.execute('PRAGMA user_version = 5')
            connection.execute("INSERT INTO conversations VALUES ('c', 'a', '2026-10-29', '2026-10-29', 'legacy')")
            connection.execute("INSERT INTO messages VALUES ('m', 'c', 1, 'user', 'legacy fact', '2026-10-29')")
            connection.execute("INSERT INTO memory_index_work(message_id, fingerprint, chunker_version) VALUES ('m', ?, 'legacy')",
                               (content_hash('old model'),))

    def test_schema_five_migration_preserves_history_and_legacy_work(self):
        store = MemoryStore(self.path, scope='a')
        self.assertEqual(store.get_messages('c')[0].content, 'legacy fact')
        with store._transaction() as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)
            row = connection.execute('SELECT * FROM memory_index_work').fetchone()
            self.assertEqual(row['state'], 'pending')
            self.assertIsNone(row['source_hash'])
            self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])

    def test_migration_failure_rolls_back_added_columns_and_receipts(self):
        def fail(connection):
            migrate_5_to_6(connection)
            raise RuntimeError('injected failure')
        with patch('memory_indexer.migrate_5_to_6', side_effect=fail):
            with self.assertRaises(RuntimeError):
                MemoryStore(self.path, scope='a')
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 5)
            columns = {r[1] for r in connection.execute('PRAGMA table_info(memory_index_work)')}
            self.assertNotIn('source_hash', columns)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'memory_index_receipts'").fetchone())
            self.assertEqual(connection.execute('SELECT count(*) FROM messages').fetchone()[0], 1)


class ChunkerTests(unittest.TestCase):
    def test_unicode_offsets_limits_and_tail_coverage(self):
        provider = QueryEmbeddings(content_hash('tokenizer'), 2, lambda t: (1, 0), lambda t: len(t)+2, 16)
        text = '  α😀 not-enabled ' * 10 + 'VAULT-5921  '
        chunks = chunk_message(text, provider)
        self.assertTrue(all(c.token_length <= provider.max_query_tokens for c in chunks))
        self.assertTrue(any('5921' in text[c.start_char:c.end_char] for c in chunks))
        covered = set(i for c in chunks for i in range(c.start_char,c.end_char))
        self.assertTrue(all(char.isspace() or i in covered for i,char in enumerate(text)))
        self.assertEqual(chunks, chunk_message(text, provider))
        with self.assertRaises(SourceTooLargeError):
            chunk_message(text, provider, max_chars=5)
        with self.assertRaises(SourceTooLargeError):
            chunk_message(text, provider, max_chunks=1)


if __name__ == '__main__':
    unittest.main()
