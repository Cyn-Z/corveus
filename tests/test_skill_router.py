"""Offline scoring regression tests; no embedding packages or models needed."""

from __future__ import annotations

import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from router_query_plan import QueryPlanSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from orchestrator import Skill, SkillRouter, tokenize
from router_policy import RouterPolicy, RouterSelectionPolicy


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, text, **options):
        self.calls.append((text, options))
        if isinstance(text, list):
            return [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8)]
        return {
            "negative-vector": (-1.0, -1.0),
            "semantic-only": (0.0, 1.0),
        }.get(text, (1.0, 0.0))


def legacy_search(router, prompt, limit):
    """Frozen pre-diagnostics scoring policy, used only for regression."""
    vector = router.model.encode(prompt)
    terms = set(tokenize(prompt))
    lexical_scores = []
    for skill, document in zip(router.registry.skills, router.documents):
        score = 0.0
        for term in terms:
            tf = document.count(term)
            if tf:
                df = router.document_frequency[term]
                idf = math.log(1 + (len(router.documents) - df + 0.5) / (df + 0.5))
                denominator = tf + 1.5 * (0.25 + 0.75 * len(document) / (router.avg_length or 1))
                score += idf * tf * 2.5 / denominator
        name_terms = set(tokenize(skill.name.replace("-", " ")))
        score += router.name_boost * len(terms & name_terms)
        lexical_scores.append(score)
    maximum = max(lexical_scores, default=0.0)
    results = []
    for skill, embedding, raw in zip(
        router.registry.skills, router.skill_embeddings, lexical_scores
    ):
        semantic = max(0.0, min(1.0, sum(a * b for a, b in zip(vector, embedding))))
        lexical = raw / maximum if maximum > 0 else 0.0
        combined = router.semantic_weight * semantic + router.lexical_weight * lexical
        if combined > 0:
            results.append((combined, skill))
    return sorted(results, key=lambda item: (-item[0], item[1].name))[:limit]


class SkillRouterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.encoder = FakeEncoder()
        skills = []
        for name, description in (
            ("network-test", "Investigate network traffic"),
            ("email-headers", "Analyze phishing email headers"),
            ("windows-events", "Investigate Windows events"),
        ):
            skills.append(Skill(
                name=name, description=description, domain="cybersecurity",
                path=name,
                tokens=tuple(tokenize(f"{name} {description} cybersecurity")),
            ))
        self.registry = SimpleNamespace(
            skills=skills, load_instructions=Mock()
        )
        self.router = self.make_router()
        self.encoder.calls.clear()

    def make_router(self, weight=0.6, boost=3.0):
        with patch.object(SkillRouter, "_load_embedding_model", return_value=self.encoder):
            with redirect_stdout(io.StringIO()):
                return SkillRouter(
                    self.registry, model_path=self.directory.name,
                    semantic_weight=weight, name_boost=boost,
                    activation_policy=RouterPolicy(enabled=False),
                    selection_policy=RouterSelectionPolicy(enabled=False),
                    query_plan_settings=QueryPlanSettings(enabled=False),
                )

    def test_catalog_work_is_not_repeated_during_search(self):
        import orchestrator
        original = orchestrator.tokenize
        with patch.object(orchestrator, "tokenize", wraps=original) as tokenize_call:
            with patch.object(orchestrator.math, "log", side_effect=AssertionError("IDF recalculated")):
                self.router.search("Analyze email headers")
        self.assertEqual(tokenize_call.call_count, 1)
        self.assertEqual(self.router.document_lengths, [len(d) for d in self.router.documents])

    def test_search_skips_diagnostic_objects_but_route_keeps_them(self):
        with patch("orchestrator.SkillScore.diagnostic", side_effect=AssertionError("diagnostics built")):
            selected = self.router.search("Analyze email headers")
        detailed, explanation = self.router.route("Analyze email headers")
        self.assertEqual(selected, detailed)
        self.assertTrue(explanation["candidates"])
        self.assertNotIn("matched_terms", explanation["candidates"][0])

    def test_matrix_scores_match_python_dot_products_at_real_dimensions(self):
        import numpy as np
        from dataclasses import replace
        query = np.arange(1, 385, dtype=np.float64)
        query /= np.linalg.norm(query)
        orthogonal = np.roll(query, 1)
        orthogonal -= np.dot(query, orthogonal) * query
        orthogonal /= np.linalg.norm(orthogonal)
        for boundary in (.36, .65):
            vectors = [cosine * query + math.sqrt(1 - cosine ** 2) * orthogonal
                       for cosine in (boundary - 1e-14, boundary, boundary + 1e-14)]
            self.router.skill_embeddings = [tuple(vector) for vector in vectors]
            self.router.embedding_matrix = np.asarray(vectors, dtype=np.float64)
            self.router.model.encode = Mock(return_value=query)
            for score in self.router.score_candidates("unknown-word", limit=3):
                index = self.registry.skills.index(score.skill)
                reference = max(0., min(1., sum(a * b for a, b in zip(query, vectors[index]))))
                self.assertAlmostEqual(score.semantic_score, reference, places=14)
                policy = RouterPolicy()
                self.assertEqual(policy.accepts(score), policy.accepts(replace(score,
                    semantic_score=reference, combined_score=.6 * reference)))

    def test_embedding_shape_mismatch_fails_explicitly(self):
        self.encoder.encode = Mock(return_value=[1., 0., 0.])
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.router.score_candidates("email")

    def test_selection_mode_discounts_only_generic_name_boosts(self):
        self.router.selection_policy = RouterSelectionPolicy()
        generic = next(x for x in self.router.score_candidates('This is a test') if x.skill.name == 'network-test')
        self.assertEqual(generic.name_boost_score, 0.)
        specific = next(x for x in self.router.score_candidates('email headers') if x.skill.name == 'email-headers')
        self.assertEqual(specific.name_boost_score, 6.)

    def test_scores_include_corpus_size_for_activation_evidence(self):
        scores = self.router.score_candidates("network", limit=3)
        self.assertTrue(scores)
        self.assertTrue(all(score.catalog_size == 3 for score in scores))

    def test_search_matches_legacy_policy_across_weights_and_queries(self):
        for weight in (0.0, 0.6, 1.0):
            for boost in (0.0, 3.0):
                router = self.make_router(weight, boost)
                for prompt in (
                    "This is a test", "Analyze email headers", "Windows events",
                    "unknown-word", "semantic-only", "negative-vector",
                ):
                    for limit in (1, 3, 10):
                        with self.subTest(weight=weight, boost=boost, prompt=prompt, limit=limit):
                            actual = router.search(prompt, limit)
                            expected = legacy_search(router, prompt, limit)
                            self.assertEqual(
                                [skill.name for _, skill in actual],
                                [skill.name for _, skill in expected],
                            )
                            for (score, _), (old_score, _) in zip(actual, expected):
                                self.assertAlmostEqual(score, old_score, places=14)

    def test_legacy_generic_test_activates_and_diagnostics_explain_why(self):
        evidence = self.router.score_candidates("This is a test")[0]
        self.assertEqual(evidence.skill.name, "network-test")
        self.assertGreater(evidence.bm25_score, 0)
        self.assertEqual(evidence.name_boost_score, 3.0)
        self.assertEqual(evidence.normalized_lexical_score, 1.0)
        self.assertIn("test", evidence.matched_terms)
        self.assertEqual(evidence.matched_name_terms, ("test",))
        self.assertEqual(dict(evidence.matched_term_document_frequencies)["test"], 1)
        self.assertTrue(self.router.search("This is a test"))

    def test_score_components_reconcile(self):
        for item in self.router.score_candidates("Investigate network test"):
            self.assertAlmostEqual(
                item.raw_lexical_score,
                item.bm25_score + item.name_boost_score,
            )
            self.assertAlmostEqual(
                item.combined_score,
                0.6 * item.semantic_score + 0.4 * item.normalized_lexical_score,
            )

    def test_semantic_match_without_keyword_overlap_is_retained(self):
        evidence = self.router.score_candidates("semantic-only")[0]
        self.assertEqual(evidence.skill.name, "email-headers")
        self.assertEqual(evidence.raw_lexical_score, 0.0)
        self.assertEqual(evidence.semantic_score, 1.0)
        self.assertEqual(evidence.combined_score, 0.6)

    def test_zero_scores_visible_in_diagnostics_but_excluded_from_search(self):
        candidates = self.router.score_candidates("negative-vector")
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(item.combined_score == 0 for item in candidates))
        self.assertFalse(candidates[0].diagnostic()["passes_current_score_floor"])
        self.assertEqual(self.router.search("negative-vector"), [])

    def test_ties_use_skill_names(self):
        router = self.make_router(weight=0.0, boost=0.0)
        candidates = router.score_candidates("unknown-word")
        self.assertEqual(
            [item.skill.name for item in candidates],
            sorted(skill.name for skill in self.registry.skills),
        )

    def test_empty_inputs_skip_encoding(self):
        for prompt in (None, "", "   "):
            self.assertEqual(self.router.search(prompt), [])
            self.assertEqual(self.router.score_candidates(prompt), [])
        self.assertEqual(self.router.search("test", limit=0), [])
        self.assertEqual(self.router.score_candidates("test", limit=-1), [])
        self.assertEqual(self.encoder.calls, [])
        self.registry.skills = []
        self.assertEqual(self.router.search("test"), [])
        self.assertEqual(self.encoder.calls, [])

    def test_scoring_encodes_query_once_and_never_loads_instructions(self):
        self.router.score_candidates("test")
        self.assertEqual(len(self.encoder.calls), 1)
        self.registry.load_instructions.assert_not_called()

    def test_search_does_not_print_diagnostics(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.router.search("test")
        self.assertEqual(output.getvalue(), "")

    def test_embedding_loader_remains_cpu_and_local_only(self):
        constructor = Mock()
        with patch.dict(sys.modules, {
            "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)
        }):
            result = SkillRouter._load_embedding_model(Path(self.directory.name))
        constructor.assert_called_once_with(
            self.directory.name, device="cpu", local_files_only=True
        )
        self.assertIs(result, constructor.return_value)

    def test_query_terms_are_explicitly_opt_in(self):
        item = self.router.score_candidates("test")[0]
        diagnostic = item.diagnostic()
        self.assertNotIn("matched_terms", diagnostic)
        self.assertNotIn("matched_name_terms", diagnostic)
        self.assertNotIn("matched_term_document_frequencies", diagnostic)
        self.assertNotIn("content", diagnostic)
        self.assertEqual(item.diagnostic(include_terms=True)["matched_terms"], ["test"])


if __name__ == "__main__":
    unittest.main()
