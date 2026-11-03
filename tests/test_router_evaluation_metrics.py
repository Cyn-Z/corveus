"""Evaluation-only regressions; fake routers never load an embedding model."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analyst'))
from orchestrator import Skill
from router_evaluation import (RoutingCase, SPLITS, evaluate_router, latency_summary,
                               benchmark_router, benchmark_startup, fingerprint_model, startup_probe_arguments)


class EvaluationMetricsTests(unittest.TestCase):
    def test_comparison_subprocesses_use_a_fixed_hash_seed(self):
        from benchmark_router_embedding_cache import subprocess_environment
        with patch.dict(os.environ, {}, clear=True):
            environment = subprocess_environment()
            self.assertEqual(environment['PYTHONHASHSEED'], '0')
            self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
            self.assertEqual(environment['TRANSFORMERS_OFFLINE'], '1')
        with patch.dict(os.environ, {'PYTHONHASHSEED': '73'}, clear=True):
            self.assertEqual(subprocess_environment()['PYTHONHASHSEED'], '73')
        with patch.dict(os.environ, {'PYTHONHASHSEED': 'random'}, clear=True):
            self.assertEqual(subprocess_environment()['PYTHONHASHSEED'], '0')

    def setUp(self):
        self.skills = [Skill(n, n, 'security', n, ()) for n in ('a', 'b', 'c', 'd')]
        self.router = SimpleNamespace(
            registry=SimpleNamespace(skills=self.skills, load_instructions=Mock(return_value='abc')),
            model=SimpleNamespace(tokenizer=SimpleNamespace(encode=Mock(return_value=[1, 2]))),
            semantic_weight=.6, name_boost=3.,
            route=Mock(return_value=([(1., s) for s in self.skills], {'candidates': [], 'reason': 'test'})))
        self.case = RoutingCase('task', 'task', 'task', True, (('a', 'b'),), ('c',), (('a', 'b'),))

    def test_judgments_exclude_unjudged_and_report_coverage(self):
        report = evaluate_router(self.router, [self.case])
        self.assertEqual(report['summary']['judged_selection_precision'], 2 / 3)
        self.assertEqual(report['summary']['judgment_coverage'], 3 / 4)
        self.assertEqual(report['summary']['redundant_selected'], 1)
        self.assertEqual(report['cases'][0]['selection_judgments']['d'], 'unjudged')
        self.router.registry.load_instructions.assert_not_called()
        json.dumps(report, allow_nan=False)

    def test_negative_cases_judge_all_selections_irrelevant(self):
        report = evaluate_router(self.router, [RoutingCase('n', 'general', 'hello', False)])
        self.assertEqual(report['summary']['judged_selection_precision'], 0.)
        self.assertEqual(report['summary']['judgment_coverage'], 1.)

    def test_no_selection_has_unknown_precision(self):
        self.router.route.return_value = ([], {'candidates': [], 'reason': 'empty'})
        report = evaluate_router(self.router, [self.case])
        self.assertIsNone(report['summary']['judged_selection_precision'])
        self.assertIsNone(report['summary']['judgment_coverage'])

    def test_conflicting_or_missing_judgments_rejected(self):
        for case in (RoutingCase('x', 'task', 'x', True, (('a',),), ('a',)),
                     RoutingCase('x', 'task', 'x', True, (('a',),), ('missing',)),
                     RoutingCase('x', 'task', 'x', True, (('a',),), (), (('missing', 'a'),))):
            with self.assertRaises(ValueError):
                evaluate_router(self.router, [case])

    def test_instruction_cost_is_opt_in_and_explicitly_encoder_tokens(self):
        row = evaluate_router(self.router, [self.case], measure_instructions=True)['cases'][0]
        self.assertEqual(row['selected_instruction_characters'], 12)
        self.assertEqual(row['selected_instruction_embedding_tokens'], 8)
        self.router.model.tokenizer.encode.assert_called_with('abc', add_special_tokens=False, truncation=False)

    def test_timing_summary_uses_nearest_rank_p95(self):
        values = [n / 1000 for n in range(1, 21)]
        result = latency_summary(values)
        self.assertEqual(result['samples'], 20)
        self.assertAlmostEqual(result['median_ms'], 10.5)
        self.assertAlmostEqual(result['p95_ms'], 19.)
        self.assertIsNone(latency_summary([])['p95_ms'])

    def test_warmup_is_excluded_and_no_instruction_loading(self):
        with patch('router_evaluation.time.perf_counter', side_effect=[0., .01, 1., 1.03]):
            report = benchmark_router(self.router, [self.case], repeats=2, warmup=3)
        self.assertEqual(self.router.route.call_count, 5)
        self.assertAlmostEqual(report['cases'][0]['median_ms'], 20.)
        self.router.registry.load_instructions.assert_not_called()
        for kw in ({'repeats': 0}, {'warmup': -1}, {'limit': True}):
            with self.assertRaises(ValueError):
                benchmark_router(self.router, [self.case], **kw)

    def test_cold_measurements_use_fresh_offline_subprocesses(self):
        with patch('router_evaluation.subprocess.run', return_value=SimpleNamespace(stdout='{"startup_seconds": 1.5}')) as run:
            result = benchmark_startup('repo', 'model', runs=2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(result['median_ms'], 1500.)
        self.assertIn('--startup-probe', run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')

    def test_direct_search_benchmark_uses_requested_interface(self):
        self.router.search = Mock(return_value=[])
        result = benchmark_router(self.router, [self.case], mode='search', repeats=2, warmup=1)
        self.assertEqual(result['interface'], 'search')
        self.assertEqual(self.router.search.call_count, 3)
        self.router.route.assert_not_called()

    def test_startup_probes_forward_effective_settings_and_cache_mode(self):
        arguments = SimpleNamespace(
            top=3, semantic_weight=.4, name_boost=2., redundancy_similarity=.8,
            relative_score_floor=.7, novelty_weight=.1, min_semantic_score=.3,
            min_combined_score=.4, semantic_only_score=.7, query_chunk_tokens=128,
            query_overlap_tokens=16, query_max_views=4, query_max_directive_views=1,
            query_max_prompt_characters=10000, embedding_cache_dir='/tmp/cache-test',
            legacy_routing=True, legacy_selection=True, disable_long_requests=True,
            disable_embedding_cache=True,
        )
        forwarded = startup_probe_arguments(arguments)
        self.assertIn('--disable-long-requests', forwarded)
        self.assertIn('--disable-embedding-cache', forwarded)
        self.assertEqual(forwarded[forwarded.index('--query-max-views') + 1], '4')
        with patch('router_evaluation.subprocess.run', return_value=SimpleNamespace(
                stdout='{"startup_seconds": 1.0, "embedding_calls": {"catalog": 0, "query": 0}}')) as run:
            result = benchmark_startup('repo', 'model', runs=1, probe_arguments=forwarded)
        command = run.call_args.args[0]
        self.assertEqual(command[-len(forwarded):], forwarded)
        self.assertEqual(result['sample_details'][0]['embedding_calls']['catalog'], 0)

    def test_actual_query_calls_reported_per_case(self):
        self.router.embedding_calls = {'catalog': 1, 'query': 0}
        original_route = self.router.route

        def route(*args, **kwargs):
            self.router.embedding_calls['query'] += 3
            return original_route(*args, **kwargs)

        self.router.route = route
        result = evaluate_router(self.router, [self.case])
        self.assertEqual(result['cases'][0]['embedding_calls'], {'catalog': 0, 'query': 3})

    def test_fingerprint_changes_for_model_bytes_and_tokenizer_settings(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / 'model.bin'
            tokenizer = Path(root) / 'tokenizer.json'
            model.write_bytes(b'one')
            tokenizer.write_text('{}')
            first = fingerprint_model(root)
            self.assertEqual(first, fingerprint_model(root))
            model.write_bytes(b'two')
            second = fingerprint_model(root)
            self.assertNotEqual(first, second)
            tokenizer.write_text('{"truncation_side":"left"}')
            self.assertNotEqual(second, fingerprint_model(root))

    def test_splits_are_disjoint_and_catalog_labels_exist(self):
        names = {s['name'] for s in json.loads((Path(__file__).resolve().parents[1] / 'analyst/skills/index.json').read_text())['skills']}
        self.assertEqual(len(SPLITS['baseline']), 24)
        self.assertFalse({c.id for c in SPLITS['development']} & {c.id for c in SPLITS['validation']})
        for cases in SPLITS.values():
            self.assertEqual(len(cases), len({c.id for c in cases}))
            for case in cases:
                self.assertTrue(set(case.irrelevant_skills) <= names)
                for group in case.required_groups + case.redundancy_groups:
                    self.assertTrue(set(group) <= names)


if __name__ == '__main__':
    unittest.main()
