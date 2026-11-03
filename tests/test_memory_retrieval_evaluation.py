"""Phase 1A evaluator checks; no application storage or inference."""
import json
from pathlib import Path
import unittest

from memory_retrieval_evaluation import eligible, evaluate, lexical_scores, metrics, rank


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.corpus = json.loads((Path(__file__).parent / 'fixtures/memory_retrieval_baseline.json').read_text())

    def test_scope_deletion_expiry_and_incognito(self):
        case = self.corpus['cases'][0]
        ids = {d['id'] for d in eligible(self.corpus, case)}
        self.assertTrue({'foreign', 'deleted', 'expired'}.isdisjoint(ids))
        self.assertEqual(eligible(self.corpus, dict(case, incognito=True)), [])

    def test_scope_local_bm25(self):
        case = self.corpus['cases'][0]
        before = eligible(self.corpus, case)
        self.corpus['messages'].append(dict(self.corpus['messages'][0], id='foreign2', scope='other'))
        after = eligible(self.corpus, case)
        self.assertEqual(lexical_scores(case['query'], before), lexical_scores(case['query'], after))

    def test_abstention_and_deduplication(self):
        docs = eligible(self.corpus, self.corpus['cases'][0])
        self.assertEqual(rank(docs, [0] * len(docs), [], 'lexical'), [])
        chosen = rank(docs, [2] * len(docs), [], 'lexical')
        self.assertFalse({'backup', 'duplicate'}.issubset(chosen))
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_conflicting_claims_remain_separate(self):
        docs = [d for d in eligible(self.corpus, self.corpus['cases'][0])
                if d['id'] in ('port-old', 'port-new', 'guess')]
        self.assertEqual(set(rank(docs, [3] * len(docs), [], 'lexical')),
                         {'port-old', 'port-new', 'guess'})

    def test_metrics_have_correct_denominators(self):
        self.assertEqual(metrics(['a', 'b'], ['a'])['precision'], .5)
        self.assertEqual(metrics(['b', 'a'], ['a'])['reciprocal_rank'], .5)
        self.assertIsNone(metrics([], [])['recall'])
        self.assertFalse(metrics([], [])['false_recall'])

    def test_hybrid_gates_before_fusion_and_breaks_ties_by_id(self):
        docs = [dict(id=x, chunk_id=x, text=x) for x in ('b', 'a', 'c')]
        self.assertEqual(rank(docs, [0, 0, .99], [.44, .44, .44], 'hybrid'), [])
        self.assertEqual(rank(docs, [1, 1, 0], [.45, .45, 0], 'hybrid'), ['a', 'b'])

    def test_fixture_judgments_reference_eligible_messages(self):
        self.assertEqual(len({c['id'] for c in self.corpus['cases']}), len(self.corpus['cases']))
        for case in self.corpus['cases']:
            ids = {d['id'] for d in eligible(self.corpus, case)}
            self.assertTrue(set(case['relevant']).issubset(ids), case['id'])

    def test_incognito_never_encodes(self):
        class ForbiddenModel:
            def encode(self, *args, **kwargs):
                raise AssertionError('incognito encoded')
        self.corpus['cases'] = [c for c in self.corpus['cases'] if c.get('incognito')]
        report = evaluate(self.corpus, ForbiddenModel())
        self.assertTrue(all(not r['selected'] and r['embedding_calls'] == 0 for r in report['rows']))


if __name__ == '__main__':
    unittest.main()
