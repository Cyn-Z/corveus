"""Phase 1C ranking checks, synthetic content and temporary SQLite only."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from memory_manager import MemoryManager
from memory_retrieval import MemoryRetriever, rank_candidates
from memory_retrieval_contracts import MemoryRetrievalPolicy, content_hash
from memory_retrieval_store import ChunkWrite, MemoryRetrievalStore, StoredChunk
from memory_store import ConversationNotFoundError
from retrieval_embeddings import QueryEmbeddings, local_query_embeddings, normalized_vector
from retrieval_lexical import MEMORY_LEXICAL_CONTRACT, bm25_scores, memory_tokenize, tokenize


def candidate(name, text=None, **changes):
    text = text or name
    value = StoredChunk(name, name, 'c', ord(name[0]) - 96, 'user', '2026-01-01T00:00:00+00:00',
                        0, len(text), text, content_hash(text), content_hash(text),
                        3, 'chunk-v1', tuple(Counter(memory_tokenize(text)).items()))
    return replace(value, **changes)


class RankingTests(unittest.TestCase):
    def test_floors_before_fusion_and_limit(self):
        chunks = (candidate('a'), candidate('b'), candidate('c'))
        policy = MemoryRetrievalPolicy(max_results=1)
        self.assertEqual(rank_candidates(chunks, (.99, 0, 0), (.449, None, -.1), policy), ())
        self.assertEqual(rank_candidates(chunks, (1, 0, 0), (None, .45, None), policy), (0,))
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                rank_candidates(chunks, (value, 0, 0), (None, None, None), policy)

    def test_partial_channels_do_not_combine_different_chunks(self):
        chunks = (candidate('a', 'one', message_id='same'),
                  candidate('b', 'two', message_id='same', start_char=4, end_char=7))
        self.assertEqual(rank_candidates(chunks, (1, 0), (None, .9), MemoryRetrievalPolicy()), (0, 1))

    def test_exact_duplicates_keep_roles_and_conflicts(self):
        chunks = (candidate('a', 'port 8443'), candidate('b', 'port 9443'),
                  candidate('c', 'port 8443'), candidate('d', 'port 8443', role='assistant'))
        self.assertEqual(rank_candidates(chunks, (2, 2, 2, 2), (None,) * 4,
                                         MemoryRetrievalPolicy(mode='lexical')), (0, 1, 3))

    def test_covered_windows_dedup_but_distinct_overlap_survives(self):
        chunks = (candidate('a', 'abcdef', message_id='same'),
                  candidate('b', 'bcd', message_id='same', start_char=1, end_char=4),
                  candidate('c', 'defghi', message_id='same', start_char=3, end_char=9))
        self.assertEqual(rank_candidates(chunks, (3, 2, 1), (None,) * 3,
                                         MemoryRetrievalPolicy(mode='lexical')), (0, 2))

    def test_recency_cannot_promote_irrelevant_or_outrank_stronger_lexical(self):
        old, new = candidate('a'), candidate('b', created_at='2026-10-01T00:00:00+00:00')
        policy = MemoryRetrievalPolicy(mode='lexical', recency_tiebreak=True)
        self.assertEqual(rank_candidates((old, new), (2, .99), (None, None), policy), (0,))
        self.assertEqual(rank_candidates((old, new), (3, 2), (None, None), policy), (0, 1))
        self.assertEqual(rank_candidates((old, new), (2, 2), (None, None), policy), (1, 0))
        self.assertEqual(rank_candidates((old, new), (2, 2), (None, None), replace(policy, recency_tiebreak=False)), (0, 1))

    def test_lexical_parity_and_identifiers_negation(self):
        from memory_retrieval_evaluation import lexical_scores
        texts = ['port 9443 enabled', 'port 8443 not enabled', 'backup archives']
        docs = [{'text': text} for text in texts]
        actual = bm25_scores(tokenize('what port is enabled?'), [Counter(tokenize(text)) for text in texts])
        self.assertEqual(list(actual), lexical_scores('what port is enabled?', docs))
        self.assertTrue({'not', 'ir-7284', '10.0.0.1'}.issubset(memory_tokenize('not IR-7284 at 10.0.0.1')))

    def test_policy_rejects_invalid_floors_and_fingerprints_all_controls(self):
        policy = MemoryRetrievalPolicy()
        for args in ({'lexical_floor': 0}, {'semantic_floor': float('nan')},
                     {'semantic_floor': 1.1}, {'rrf_constant': True}, {'mode': 'router'}):
            with self.assertRaises(ValueError):
                MemoryRetrievalPolicy(**args)
        self.assertNotEqual(policy.fingerprint, replace(policy, recency_tiebreak=True).fingerprint)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.encode = Mock(return_value=(1., 0.))
        self.provider = QueryEmbeddings(content_hash('model-v1'), 2, self.encode,
                                         lambda q: len(q.split()), 8)
        self.manager = MemoryManager(enabled=True, database_path=Path(self.tmp.name) / 'memory.sqlite3',
                                     scope='a', retention_days=30, retrieval_enabled=True,
                                     retrieval_embeddings=self.provider)
        self.manager._store._clock = lambda: self.now
        self.derived = MemoryRetrievalStore(self.manager._store)
        self.derived.configure(lexical_contract=MEMORY_LEXICAL_CONTRACT, chunker_version='chunk-v1',
                               embedding_fingerprint=self.provider.fingerprint)
        self.current = self.manager.create_conversation()
        self.source = self.manager.create_conversation()
        self.user, self.assistant = self.manager.record_completed_exchange(
            self.source.id, user_content='The gateway port is 9443.',
            assistant_content='Archive backup retention is thirty days.')

    def index(self, message, vector=None, span=None):
        start, end = span or (0, len(message.content))
        terms = tuple(sorted(Counter(memory_tokenize(message.content[start:end])).items()))
        self.assertTrue(self.derived.replace_chunks(message.id, source_hash=content_hash(message.content),
                        lexical_contract=MEMORY_LEXICAL_CONTRACT, chunker_version='chunk-v1',
                        chunks=(ChunkWrite(0, start, end, 8, terms),)))
        chunks = self.derived.candidates(current_conversation_id=self.current.id)
        chunk = next(c for c in chunks if c.message_id == message.id)
        if vector is not None:
            self.assertTrue(self.derived.put_embedding(chunk.id, source_hash=chunk.source_hash,
                            fingerprint=self.provider.fingerprint, dimension=2, vector=struct.pack('<2f', *vector)))
        return chunk

    def retrieve(self, query='gateway port 9443', **policy):
        return self.manager.retrieve(query, current_conversation_id=self.current.id,
                                     policy=MemoryRetrievalPolicy(**policy))

    def test_lexical_recall_abstention_and_complete_index(self):
        self.index(self.user)
        self.index(self.assistant)
        result = self.retrieve(mode='lexical')
        self.assertEqual(result.status, 'ready')
        self.assertEqual([e.message_id for e in result.evidence], [self.user.id])
        self.assertTrue(result.index_complete)
        self.assertEqual(result.lexical_contract, MEMORY_LEXICAL_CONTRACT)
        self.assertEqual(self.retrieve('cafeteria lunch menu', mode='lexical').evidence, ())
        self.encode.assert_not_called()

    def test_semantic_paraphrase_single_query_call(self):
        self.index(self.user, (0, 1))
        self.index(self.assistant, (1, 0))
        result = self.retrieve('How long should archives survive?', mode='semantic')
        self.assertEqual([e.message_id for e in result.evidence], [self.assistant.id])
        self.assertTrue(result.semantic_available)
        self.assertTrue(result.index_complete)
        self.assertEqual(result.embedding_fingerprint, self.provider.fingerprint)
        self.encode.assert_called_once()

    def test_disabled_incognito_and_foreign_zero_encode(self):
        self.index(self.user, (1, 0))
        policy = MemoryRetrievalPolicy()
        with patch.object(self.manager._store, '_transaction', side_effect=AssertionError('read')):
            self.assertEqual(self.manager.retrieve('q', current_conversation_id=None, policy=policy).status, 'incognito')
            self.manager._retrieval_enabled = False
            self.assertEqual(self.retrieve().status, 'disabled')
        self.manager._retrieval_enabled = True
        with self.assertRaises(ConversationNotFoundError):
            self.manager.retrieve('q', current_conversation_id='foreign', policy=policy)
        self.encode.assert_not_called()

    def test_other_scope_cannot_affect_scores_or_vector_reads(self):
        self.index(self.user, (1, 0))
        before = self.retrieve()
        foreign = MemoryManager(enabled=True, database_path=self.manager._store.database_path, scope='b')
        conversation = foreign.create_conversation()
        foreign.record_completed_exchange(conversation.id, user_content='gateway port 9443', assistant_content='foreign')
        after = self.retrieve()
        self.assertEqual(before.evidence, after.evidence)
        self.assertEqual(before.index_generation, after.index_generation)

    def test_missing_or_mismatched_vectors_degrade_hybrid_explicitly(self):
        self.index(self.user)
        self.index(self.assistant)
        result = self.retrieve()
        self.assertTrue(result.evidence)
        self.assertFalse(result.index_complete)
        self.assertFalse(result.semantic_available)
        self.assertEqual(self.retrieve(mode='semantic').status, 'not_ready')
        self.derived.configure(lexical_contract=MEMORY_LEXICAL_CONTRACT, chunker_version='chunk-v1',
                               embedding_fingerprint=content_hash('other-model'))
        self.assertFalse(self.retrieve().semantic_available)
        self.encode.assert_not_called()

    def test_overlong_query_does_not_silently_truncate(self):
        self.index(self.user, (1, 0))
        query = 'gateway port ' + 'extra ' * 10
        self.assertFalse(self.retrieve(query).semantic_available)
        self.assertEqual(self.retrieve(query, mode='semantic').status, 'not_ready')
        self.encode.assert_not_called()
        with self.assertRaises(ValueError):
            self.retrieve('too long', max_query_chars=3)

    def test_model_failure_is_visible_and_invalid_vectors_rejected(self):
        self.index(self.user, (1, 0))
        self.encode.side_effect = RuntimeError('synthetic model failure')
        with self.assertRaises(RuntimeError):
            self.retrieve()
        for vector in ((0, 0), (float('nan'), 0), (1, 0, 0)):
            with self.assertRaises(ValueError):
                normalized_vector(vector, 2)

    def test_deleted_during_encoding_fails_closed_without_reretrieval(self):
        self.index(self.user, (1, 0))
        def encode(query):
            self.manager.delete_conversation(self.source.id)
            return (1, 0)
        self.encode.side_effect = encode
        result = self.retrieve()
        self.assertEqual(result.status, 'not_ready')
        self.assertEqual(result.evidence, ())
        self.encode.assert_called_once()

    def test_retention_advancing_during_encoding_is_revalidated(self):
        self.index(self.user, (1, 0))
        self.now += timedelta(days=30)
        current = self.manager.create_conversation()
        self.current = current
        def encode(query):
            self.now += timedelta(microseconds=1)
            return (1, 0)
        self.encode.side_effect = encode
        self.assertEqual(self.retrieve().status, 'not_ready')

    def test_corrupt_vector_is_not_encoded_or_used(self):
        chunk = self.index(self.user, (1, 0))
        self.index(self.assistant)
        with self.manager._store._transaction(write=True) as connection:
            connection.execute('UPDATE memory_embeddings SET vector = ? WHERE chunk_id = ?',
                               (struct.pack('<2f', float('nan'), 0), chunk.id))
        result = self.retrieve()
        self.assertFalse(result.semantic_available)
        self.assertTrue(result.evidence)
        self.encode.assert_not_called()

    def test_dimension_mismatch_is_not_encoded(self):
        chunk = self.index(self.user, (1, 0))
        self.index(self.assistant)
        with self.manager._store._transaction(write=True) as connection:
            connection.execute('UPDATE memory_embeddings SET dimension = 1, vector = ? WHERE chunk_id = ?',
                               (struct.pack('<f', 1), chunk.id))
        self.assertFalse(self.retrieve().semantic_available)
        self.encode.assert_not_called()

    def test_index_contract_change_during_model_callback_discards_evidence(self):
        self.index(self.user, (1, 0))
        def encode(query):
            self.derived.configure(lexical_contract=MEMORY_LEXICAL_CONTRACT, chunker_version='next-chunker')
            return (1, 0)
        self.encode.side_effect = encode
        self.assertEqual(self.retrieve().status, 'not_ready')
        self.encode.assert_called_once()

    def test_partial_spans_never_claim_complete(self):
        self.index(self.user, span=(0, 10))
        self.index(self.assistant)
        self.assertFalse(self.retrieve('gateway', mode='lexical').index_complete)

    def test_stable_evidence_id_across_reindex(self):
        first_chunk = self.index(self.user)
        first = self.retrieve(mode='lexical').evidence
        second_chunk = self.index(self.user)
        self.assertNotEqual(first_chunk.id, second_chunk.id)
        self.assertEqual(first, self.retrieve(mode='lexical').evidence)

    def test_unknown_lexical_contract_and_empty_index_are_not_ready(self):
        self.assertFalse(self.retrieve(mode='lexical').index_complete)
        self.derived.configure(lexical_contract='unrelated-tokenizer', chunker_version='chunk-v1')
        self.assertEqual(self.retrieve().status, 'not_ready')
        self.encode.assert_not_called()


class EmbeddingAdapterTests(unittest.TestCase):
    def test_local_adapter_fingerprint_token_bound_and_cpu_contract(self):
        class Tokenizer:
            model_max_length = 4
            def encode(self, text, **kwargs):
                return [0] + text.split() + [1]
        class Model:
            device = 'cpu'
            max_seq_length = 8
            tokenizer = Tokenizer()
            def get_sentence_embedding_dimension(self):
                return 2
            encode = Mock(return_value=(1., 0.))
        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'weights.bin'
            artifact.write_bytes(b'synthetic local model')
            adapter = local_query_embeddings(model, tmp)
            self.assertIsNone(adapter.encode('one two three'))
            model.encode.assert_not_called()
            self.assertEqual(adapter.encode('one two'), (1., 0.))
            self.assertEqual(adapter.fingerprint, local_query_embeddings(model, tmp).fingerprint)
            artifact.write_bytes(b'new local model')
            self.assertNotEqual(adapter.fingerprint, local_query_embeddings(model, tmp).fingerprint)
            model.prompts = {'query': 'unused named prefix'}
            local_query_embeddings(model, tmp)  # Named definitions alone do not prefix encode().
            model.default_prompt_name = 'query'
            with self.assertRaises(ValueError):
                local_query_embeddings(model, tmp)
            model.default_prompt_name = None
            model.device = 'cuda'
            with self.assertRaises(ValueError):
                local_query_embeddings(model, tmp)


if __name__ == '__main__':
    unittest.main()
