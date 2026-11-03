"""Read-only UI metadata must preserve ownership, deletion and Incognito gates."""
import json
import sqlite3
import unittest
from unittest.mock import Mock, patch

import test_memory_retrieval_preparation as fixtures
from memory_manager import MemoryManager
from memory_retrieval_contracts import content_hash
from retrieval_embeddings import QueryEmbeddings
from service_contracts import Principal, ServiceError, ErrorCode


class MemoryUITests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.WorkerMemoryTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def complete(self, incognito=False):
        job = self.f.submit(incognito=incognito)
        self.f.runtime.worker.run_once()
        return job.job_id

    def test_status_is_scope_local_and_contains_only_counts(self):
        other = MemoryManager(enabled=True, database_path=self.f.memory._store.database_path, scope='other')
        c = other.create_conversation()
        other.record_completed_exchange(c.id, user_content='foreign private fact', assistant_content='foreign')
        status = self.f.runtime.api.memory_status()
        self.assertEqual(status['total_messages'], 2)
        self.assertEqual(status['indexed_messages'], 0)
        self.assertEqual(status['state'], 'indexing')
        text = json.dumps(status)
        self.assertNotIn(self.f.pair[0].content, text)
        self.assertNotIn('foreign', text)

    def test_status_tracks_completed_index_receipts(self):
        adapter = QueryEmbeddings(content_hash('test model'), 2, Mock(return_value=(1.,0.)), lambda q: len(q)+2, 256)
        self.f.memory.start_indexing(adapter)
        self.f.memory.indexing_step()
        self.f.memory.indexing_step()
        status = self.f.runtime.api.memory_status()
        self.assertEqual(status['state'], 'ready')
        self.assertEqual(status['indexed_messages'], 2)

    def test_disabled_status_makes_no_store_calls(self):
        self.f.memory._retrieval_enabled = False
        with patch.object(self.f.memory._store, '_transaction', side_effect=AssertionError('disabled store call')):
            self.assertEqual(self.f.runtime.api.memory_status()['state'], 'disabled')

    def test_result_sources_are_revalidated_and_bounded(self):
        self.f.memory.rename_conversation(self.f.source.id, title='<img src=x onerror=alert(1)>')
        job = self.complete()
        result = self.f.runtime.api.result(job)
        self.assertEqual(result['memory']['state'], 'used')
        self.assertTrue(result['memory']['sources'])
        for source in result['memory']['sources']:
            self.assertEqual(source['conversation_id'], self.f.source.id)
            self.assertEqual(source['title'], '<img src=x onerror=alert(1)>')
            self.assertLessEqual(len(source['excerpt']), 600)
        self.f.memory.delete_conversation(self.f.source.id)
        result = self.f.runtime.api.result(job)
        self.assertEqual(result['answer'], 'synthetic answer')
        self.assertEqual(result['memory'], {'state':'unavailable','sources':[]})

    def test_foreign_owner_cannot_read_memory_provenance(self):
        job = self.complete()
        with self.assertRaises(ServiceError) as caught:
            self.f.preparation.memory_view(Principal('other', 'other'), job)
        self.assertEqual(caught.exception.code, ErrorCode.NOT_FOUND)

    def test_incognito_result_does_not_resolve_or_read_memory(self):
        job = self.complete(incognito=True)
        self.f.preparation.memory_for = Mock(side_effect=AssertionError('incognito resolver'))
        with patch.object(self.f.memory._store, '_transaction', side_effect=AssertionError('incognito store')):
            result = self.f.runtime.api.result(job)
        self.assertEqual(result['memory'], {'state':'incognito','sources':[]})

    def test_optional_metadata_failure_does_not_hide_completed_answer(self):
        job = self.complete()
        with patch.object(self.f.preparation, 'memory_view', side_effect=sqlite3.OperationalError('private details')):
            result = self.f.runtime.api.result(job)
        self.assertEqual(result['answer'], 'synthetic answer')
        self.assertEqual(result['memory']['state'], 'unavailable')
        self.assertNotIn('private details', json.dumps(result))
