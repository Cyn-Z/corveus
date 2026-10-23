from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from orchestrator import Skill
from router_evaluation import CASES, RoutingCase, evaluate_router


class RouterEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.alpha = Skill("alpha", "Alpha guidance", "cybersecurity", "alpha", ())
        self.beta = Skill("beta", "Beta guidance", "cybersecurity", "beta", ())
        self.router = SimpleNamespace(
            registry=SimpleNamespace(skills=[self.alpha, self.beta]),
            semantic_weight=0.6,
            name_boost=3.0,
            score_candidates=lambda prompt, limit: [],
            search=lambda prompt, limit: {
                "negative": [(0.1, self.alpha)],
                "positive": [],
                "multi": [(0.8, self.alpha)],
            }[prompt],
        )
        self.cases = (
            RoutingCase("negative", "conversation", "negative", False),
            RoutingCase("positive", "specific", "positive", True, (("beta",),)),
            RoutingCase("multi", "multi-topic", "multi", True, (("alpha",), ("beta",))),
        )

    def test_activation_and_expected_group_coverage_are_separate(self):
        report = evaluate_router(self.router, self.cases)
        summary = report["summary"]
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["false_activations"], 1)
        self.assertEqual(summary["false_activation_rate"], 1.0)
        self.assertEqual(summary["missed_activations"], 1)
        self.assertEqual(summary["missed_activation_rate"], 0.5)
        self.assertEqual(summary["expected_group_coverage"], 1 / 3)
        self.assertEqual(summary["positive_cases_with_all_groups"], 0)

    def test_report_text_is_opt_in_and_json_serializable(self):
        report = evaluate_router(self.router, self.cases)
        self.assertTrue(all("prompt" not in row for row in report["cases"]))
        json.dumps(report, allow_nan=False)
        explicit = evaluate_router(self.router, self.cases, include_text=True)
        self.assertEqual(explicit["cases"][0]["prompt"], "negative")

    def test_missing_catalog_group_is_rejected(self):
        case = RoutingCase("missing", "specific", "positive", True, (("missing",),))
        with self.assertRaises(ValueError):
            evaluate_router(self.router, [case])

    def test_invalid_cases_and_limits_are_rejected(self):
        for cases in ((), (self.cases[0], self.cases[0])):
            with self.assertRaises(ValueError):
                evaluate_router(self.router, cases)
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                evaluate_router(self.router, self.cases, limit=limit)
        with self.assertRaises(ValueError):
            evaluate_router(self.router, [RoutingCase("bad", "specific", "positive", True)])

    def test_single_class_metrics_do_not_divide_by_zero(self):
        negative = evaluate_router(self.router, self.cases[:1])["summary"]
        self.assertIsNone(negative["missed_activation_rate"])
        self.assertIsNone(negative["expected_group_coverage"])
        positive = evaluate_router(self.router, self.cases[1:])["summary"]
        self.assertIsNone(positive["false_activation_rate"])

    def test_catalog_fingerprint_is_order_independent(self):
        before = evaluate_router(self.router, self.cases)["catalog_sha256"]
        self.router.registry.skills.reverse()
        after = evaluate_router(self.router, self.cases)["catalog_sha256"]
        self.assertEqual(before, after)

    def test_benchmark_cases_have_valid_current_catalog_names(self):
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / "analyst/skills/index.json").read_text())
        names = {skill["name"] for skill in catalog["skills"]}
        self.assertEqual(len({case.id for case in CASES}), len(CASES))
        for case in CASES:
            self.assertEqual(case.expect_skills, bool(case.required_groups))
            for group in case.required_groups:
                self.assertTrue(set(group).issubset(names), case.id)


if __name__ == "__main__":
    unittest.main()
