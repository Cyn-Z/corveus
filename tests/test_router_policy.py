from __future__ import annotations

import sys
import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from orchestrator import Skill, SkillRouter, SkillScore
from router_policy import RouterPolicy, load_router_policy


def candidate(name="event-logs", **changes):
    score = SkillScore(
        skill=Skill(name, "Example", "cybersecurity", name, ("4625",)),
        bm25_score=3.0,
        name_boost_score=0.0,
        raw_lexical_score=3.0,
        normalized_lexical_score=1.0,
        semantic_score=0.4,
        combined_score=0.6,
        matched_terms=("4625",),
        matched_name_terms=(),
        matched_term_document_frequencies=(("4625", 2),),
    )
    return replace(score, **changes)


class RouterPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = RouterPolicy()

    def test_complete_conversational_messages_bypass(self):
        for prompt in (
            "This is a test", "  THIS IS A TEST!!!  ", "Just testing.",
            "Hello!", "Thanks for your help.", "Are you there?",
            "The test worked, thank you.", "Continue with that.",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    self.policy.bypass_reason(prompt), "conversational_message"
                )

    def test_task_appended_to_conversation_never_bypasses(self):
        for prompt in (
            "This is a test; investigate Event 4625.",
            "Hello! Analyze these phishing email headers.",
            "Thanks. Now audit Kubernetes RBAC.",
            "Hello\nInvestigate suspicious logins.",
            "This is a test\n4625 failed logons",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(self.policy.bypass_reason(prompt))

    def test_simple_definitions_bypass_but_procedures_do_not(self):
        for prompt in (
            "What does SIEM stand for?",
            "What does phishing mean? Answer in one sentence.",
            "Define endpoint detection.",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.policy.bypass_reason(prompt), "simple_definition")
        for prompt in (
            "What is phishing? Then analyze this email.",
            "What is causing failed logins?",
            "What is the best investigation workflow?",
            "Define phishing and investigate this email.",
            "What is phishing?\nReceived: from suspicious.invalid",
            "How do I investigate phishing?",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(self.policy.bypass_reason(prompt))


    def test_ambiguous_questions_reach_retrieval(self):
        for prompt in (
            "What is phishing?", "What is preventing SSH access?",
            "What is blocking legitimate logins?", "What are failing logins telling us?",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(self.policy.bypass_reason(prompt))

    def test_language_tasks_bypass_only_as_complete_nonoperational_requests(self):
        for prompt in (
            "Translate the words network security into French.",
            "Suggest a title for my poem about network security.",
        ):
            self.assertEqual(self.policy.bypass_reason(prompt), "non_operational_language_task")
            self.assertIsNone(self.policy.bypass_reason(prompt + " Investigate Event 4625."))
        for prompt in (
            "Translate the phrase investigate suspicious logins into French.",
            "Suggest a title for my story about security; audit RBAC.",
            "Translate the words network security into French then explain incident response.",
            "Translate the words network security into French\nEvent 4625",
            "Translate the words network security into French\nReceived: suspicious.invalid",
        ):
            self.assertIsNone(self.policy.bypass_reason(prompt))

    def test_catalog_frequency_and_short_identifiers(self):
        common = candidate(catalog_size=100, matched_terms=("endpoint",),
                           matched_term_document_frequencies=(("endpoint", 21),))
        self.assertFalse(self.policy.accepts(common))
        self.assertTrue(self.policy.accepts(replace(common,
            matched_term_document_frequencies=(("endpoint", 20),))))
        short = candidate(catalog_size=100, matched_terms=("ad",),
                          matched_term_document_frequencies=(("ad", 4),))
        self.assertTrue(self.policy.accepts(short))
        self.assertFalse(self.policy.accepts(replace(short, matched_terms=("xx",))))
        self.assertTrue(self.policy.accepts(replace(common, semantic_score=.7)))
        self.assertTrue(RouterPolicy(enabled=False).accepts(common))

    def test_short_technical_task_is_not_a_bypass(self):
        self.assertIsNone(self.policy.bypass_reason("Investigate 4625"))
        self.assertTrue(self.policy.accepts(candidate()))

    def test_normalized_lexical_score_cannot_override_weak_semantics(self):
        score = candidate(semantic_score=0.1, combined_score=0.9)
        self.assertFalse(self.policy.accepts(score))
        self.assertEqual(self.policy.candidate_reason(score), "weak_semantic_match")

    def test_generic_keywords_do_not_establish_applicability(self):
        score = candidate(matched_terms=("test", "network", "this"))
        self.assertFalse(self.policy.accepts(score))
        self.assertEqual(self.policy.candidate_reason(score), "no_informative_keyword_match")

    def test_strong_semantic_only_match_is_allowed(self):
        score = candidate(
            matched_terms=(), semantic_score=0.7, combined_score=0.42,
            bm25_score=0.0, raw_lexical_score=0.0, normalized_lexical_score=0.0,
        )
        self.assertTrue(self.policy.accepts(score))
        self.assertEqual(self.policy.candidate_reason(score), "strong_semantic_match")

    def test_threshold_boundaries_are_inclusive(self):
        self.assertTrue(self.policy.accepts(candidate(
            semantic_score=0.36, combined_score=0.45
        )))
        self.assertFalse(self.policy.accepts(candidate(
            semantic_score=0.36, combined_score=0.449
        )))
        self.assertFalse(self.policy.accepts(candidate(
            semantic_score=0.359, combined_score=0.8
        )))
        self.assertTrue(self.policy.accepts(candidate(
            matched_terms=(), semantic_score=0.65, combined_score=0.39
        )))

    def test_legacy_mode_restores_positive_score_policy(self):
        policy = RouterPolicy(enabled=False)
        self.assertIsNone(policy.bypass_reason("This is a test"))
        self.assertTrue(policy.accepts(candidate(
            semantic_score=0.0, combined_score=0.001, matched_terms=()
        )))
        self.assertFalse(policy.accepts(candidate(combined_score=0.0)))

    def test_configuration_defaults_and_overrides(self):
        self.assertEqual(load_router_policy({}), RouterPolicy())
        self.assertEqual(load_router_policy({
            "ROUTER_ACTIVATION_ENABLED": " FALSE ",
            "ROUTER_MIN_SEMANTIC_SCORE": "0.3",
            "ROUTER_MIN_COMBINED_SCORE": "0.4",
            "ROUTER_SEMANTIC_ONLY_SCORE": "0.7",
        }), RouterPolicy(False, 0.3, 0.4, 0.7))

    def test_invalid_configuration_is_rejected(self):
        for key, value in (
            ("ROUTER_ACTIVATION_ENABLED", "yes"),
            ("ROUTER_MIN_SEMANTIC_SCORE", "nan"),
            ("ROUTER_MIN_SEMANTIC_SCORE", "-0.1"),
            ("ROUTER_MIN_COMBINED_SCORE", "inf"),
            ("ROUTER_MIN_COMBINED_SCORE", "text"),
            ("ROUTER_SEMANTIC_ONLY_SCORE", "1.1"),
            ("ROUTER_SEMANTIC_ONLY_SCORE", "0.1"),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    load_router_policy({key: value})
        with self.assertRaises(ValueError):
            RouterPolicy(enabled="true")
        with self.assertRaises(ValueError):
            RouterPolicy(min_semantic_score=True)

    def test_application_settings_load_policy_without_loading_a_model(self):
        root = Path(__file__).resolve().parents[1]
        environment = {
            "LLM_BASE_URL": "http://unused/v1/chat/completions",
            "MODEL_NAME": "test-model",
            "LLM_CONTEXT_SIZE": "1000",
            "LLM_MAX_OUTPUT_TOKENS": "100",
            "LLM_CONTEXT_SAFETY_MARGIN": "50",
            "SKILLS_DIR": str(root / "analyst/skills"),
            # Configuration validates directory existence, not model loading.
            "MINILM_MODEL_PATH": str(root),
            "ROUTER_TOP_K": "5",
            "ROUTER_SEMANTIC_WEIGHT": "0.6",
            "ROUTER_NAME_BOOST": "3",
            "LLM_TIMEOUT_SECONDS": "120",
            "TOKEN_COUNT_TIMEOUT_SECONDS": "30",
            "MAX_PENDING_REQUESTS": "32",
            "PENDING_REQUEST_TTL_SECONDS": "600",
            "ROUTER_ACTIVATION_ENABLED": "false",
            "ROUTER_MIN_SEMANTIC_SCORE": "0.3",
        }
        name = "_router_settings_test"
        spec = importlib.util.spec_from_file_location(name, root / "analyst/config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, environment, clear=True):
            with patch.dict(sys.modules, {name: module}):
                spec.loader.exec_module(module)
        self.assertFalse(module.settings.router_policy.enabled)
        self.assertEqual(module.settings.router_policy.min_semantic_score, 0.3)
        self.assertEqual(module.settings.input_token_budget, 850)


class RouterActivationTests(unittest.TestCase):
    def make_router(self, candidates):
        router = SkillRouter.__new__(SkillRouter)
        router.registry = SimpleNamespace(skills=[item.skill for item in candidates])
        router.activation_policy = RouterPolicy()
        router.score_candidates = Mock(return_value=candidates)
        return router

    def test_bypass_does_not_encode_or_score(self):
        router = self.make_router([candidate()])
        result, details = router.route("This is a test")
        self.assertEqual(result, [])
        self.assertEqual(details["reason"], "conversational_message")
        router.score_candidates.assert_not_called()

    def test_troubleshooting_questions_score_and_language_tasks_do_not(self):
        router = self.make_router([candidate()])
        result, _ = router.route("What is blocking legitimate logins?")
        self.assertTrue(result)
        router.score_candidates.assert_called_once()
        router.score_candidates.reset_mock()
        result, details = router.route("Translate the words network security into French.")
        self.assertEqual(result, [])
        self.assertEqual(details["reason"], "non_operational_language_task")
        router.score_candidates.assert_not_called()

    def test_filtering_occurs_before_top_k(self):
        irrelevant = candidate("irrelevant", semantic_score=0.1, combined_score=0.8)
        relevant = candidate("relevant", combined_score=0.6)
        router = self.make_router([irrelevant, relevant])
        result, details = router.route("Investigate Event 4625", limit=1)
        self.assertEqual(result, [(0.6, relevant.skill)])
        router.score_candidates.assert_called_once_with("Investigate Event 4625", 2)
        self.assertEqual(len(details["candidates"]), 2)
        self.assertFalse(details["candidates"][0]["selected"])
        self.assertTrue(details["candidates"][1]["selected"])

    def test_no_eligible_candidates_is_a_successful_empty_result(self):
        router = self.make_router([candidate(semantic_score=0.1)])
        result, details = router.route("Some unrelated question")
        self.assertEqual(result, [])
        self.assertEqual(details["reason"], "no_relevant_skills")

    def test_selected_results_keep_ranking_and_apply_limit(self):
        first = candidate("first", combined_score=0.8)
        second = candidate("second", combined_score=0.6)
        router = self.make_router([first, second])
        self.assertEqual(router.search("Investigate 4625", 1), [(0.8, first.skill)])

    def test_decision_terms_remain_opt_in(self):
        router = self.make_router([candidate()])
        _, default = router.route("Investigate 4625")
        _, explicit = router.route("Investigate 4625", include_terms=True)
        self.assertNotIn("matched_terms", default["candidates"][0])
        self.assertEqual(explicit["candidates"][0]["matched_terms"], ["4625"])


if __name__ == "__main__":
    unittest.main()
