from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "analyst"),
)

from orchestrator import Skill, SkillRouter, SkillScore, tokenize
from router_long_requests import route_long_request
from router_policy import RouterPolicy, RouterSelectionPolicy
from router_query_plan import (
    QueryPlan,
    QueryPlanSettings,
    QueryView,
    add_query_plan_arguments,
    load_query_plan_settings,
    query_plan_settings_from_arguments,
)
from test_router_query_plan import WordTokenizer


class Encoder:
    """Simulate an encoder that really truncates at its model limit."""

    max_seq_length = 32

    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.tokenizer.model_max_length = self.max_seq_length
        self.calls = []

    def vector(self, text):
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        visible = self.tokenizer.decode(ids[:30]).casefold()
        if "4625" in visible or "windows" in visible:
            return (1.0, 0.0)
        if "phishing" in visible:
            return (0.0, 1.0)
        return (-0.7071067811865476, -0.7071067811865476)

    def encode(self, text, **options):
        if isinstance(text, list):
            return [self.vector(item) for item in text]
        self.calls.append(text)
        return self.vector(text)


def score(combined, semantic, *, bm25=1.0):
    skill = Skill(
        "windows-events",
        "Investigate Windows Event 4625",
        "security",
        "windows-events",
        ("windows", "4625"),
    )
    return SkillScore(
        skill=skill,
        bm25_score=bm25,
        name_boost_score=0.0,
        raw_lexical_score=bm25,
        normalized_lexical_score=0.5,
        semantic_score=semantic,
        combined_score=combined,
        matched_terms=("4625",),
        matched_name_terms=(),
        matched_term_document_frequencies=(("4625", 1),),
    )


def manual_plan(*texts):
    views = tuple(
        QueryView(
            text=text,
            sources=("window",),
            token_ranges=((index, index + 1),),
            source_lines=(),
            content_tokens=1,
            encoded_tokens=3,
        )
        for index, text in enumerate(texts)
    )
    return QueryPlan(
        mode="chunked",
        views=views,
        prompt_tokens=len(texts),
        effective_token_limit=3,
        payload_token_limit=1,
        original_encoded_tokens=len(texts) + 2,
        sampled_window_tokens=len(texts),
        directive_candidates=0,
        directives_selected=0,
    )


class RouterLongRequestTests(unittest.TestCase):
    def make_router(self, *, enabled=True, selection=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        encoder = Encoder()

        skills = [
            Skill(
                "windows-events",
                "Investigate Windows Event 4625",
                "security",
                "windows-events",
                tuple(tokenize("Windows Event 4625")),
            ),
            Skill(
                "phishing-headers",
                "Analyze phishing email headers",
                "security",
                "phishing-headers",
                tuple(tokenize("phishing email headers")),
            ),
        ]
        registry = SimpleNamespace(
            skills=skills,
            load_instructions=Mock(),
        )

        with patch.object(
            SkillRouter,
            "_load_embedding_model",
            return_value=encoder,
        ):
            with redirect_stdout(io.StringIO()):
                router = SkillRouter(
                    registry,
                    model_path=directory.name,
                    semantic_weight=0.6,
                    name_boost=3.0,
                    selection_policy=RouterSelectionPolicy(
                        enabled=selection
                    ),
                    query_plan_settings=QueryPlanSettings(
                        enabled=enabled,
                        chunk_tokens=24,
                        overlap_tokens=4,
                        max_views=4,
                        max_directive_views=1,
                    ),
                )

        encoder.calls.clear()
        return router, encoder

    def pooling_router(self, results):
        return SimpleNamespace(
            registry=SimpleNamespace(skills=[score(0.6, 0.7).skill]),
            activation_policy=RouterPolicy(),
            selection_policy=RouterSelectionPolicy(enabled=False),
            score_candidates=Mock(side_effect=results),
        )

    def test_short_requests_preserve_selected_names_and_scores(self):
        enabled, _ = self.make_router()
        disabled, _ = self.make_router(enabled=False)
        prompt = "Investigate Event 4625."

        current = enabled.search(prompt)
        previous = disabled.search(prompt)

        self.assertEqual(
            [(value, skill.name) for value, skill in current],
            [(value, skill.name) for value, skill in previous],
        )

    def test_task_at_head_middle_and_tail_reaches_retrieval(self):
        router, _ = self.make_router()
        task = "Investigate Windows Event 4625."
        noise = "\n".join(["heartbeat status=ok"] * 100)

        for prompt in (
            task + "\n" + noise,
            noise + "\n" + task,
            noise + "\n" + task + "\n" + noise,
        ):
            with self.subTest(position=prompt.find(task)):
                selected = router.search(prompt)
                self.assertIn(
                    "windows-events",
                    [skill.name for _, skill in selected],
                )

    def test_selection_reuses_views_without_extra_encoding(self):
        router, encoder = self.make_router(selection=True)
        prompt = (
            "Investigate Event 4625.\n"
            + "\n".join(["heartbeat status=ok"] * 100)
        )
        _, details = router.route(prompt)

        self.assertEqual(
            len(encoder.calls),
            details["query_plan"]["view_count"],
        )
        self.assertEqual(
            len(encoder.calls),
            details["query_plan"]["embedding_queries"],
        )
        self.assertLessEqual(len(encoder.calls), 4)
        router.registry.load_instructions.assert_not_called()

    def test_ineligible_high_score_cannot_replace_eligible_evidence(self):
        router = self.pooling_router(
            [
                [score(0.9, 0.1)],
                [score(0.42, 0.7)],
            ]
        )
        selected, _ = route_long_request(
            router,
            manual_plan("weak", "strong"),
            limit=5,
        )
        self.assertEqual(selected[0][0], 0.42)

    def test_repeated_matches_do_not_accumulate_votes(self):
        router = self.pooling_router(
            [[score(0.6, 0.7)]] * 3
        )
        selected, details = route_long_request(
            router,
            manual_plan("first", "second", "third"),
            limit=5,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0], 0.6)
        self.assertEqual(
            details["query_plan"]["embedding_queries"],
            3,
        )

    def test_winning_score_keeps_its_own_evidence(self):
        router = self.pooling_router(
            [
                [score(0.42, 0.7, bm25=9.0)],
                [score(0.6, 0.45, bm25=1.0)],
            ]
        )
        _, details = route_long_request(
            router,
            manual_plan("first", "second"),
            limit=5,
        )
        item = details["candidates"][0]

        self.assertEqual(item["combined_score"], 0.6)
        self.assertEqual(item["semantic_score"], 0.45)
        self.assertEqual(item["bm25_score"], 1.0)
        self.assertEqual(item["winning_view"], 1)

    def test_no_eligible_results_is_a_normal_empty_result(self):
        router = self.pooling_router(
            [[score(0.9, 0.1)]]
        )
        selected, details = route_long_request(
            router,
            manual_plan("noise"),
            limit=5,
        )
        self.assertEqual(selected, [])
        self.assertEqual(details["reason"], "no_relevant_skills")

    def test_default_diagnostics_do_not_expose_view_text(self):
        marker = "PRIVATE_MARKER_4625"
        router = self.pooling_router(
            [[score(0.6, 0.7)]]
        )
        _, details = route_long_request(
            router,
            manual_plan(marker),
            limit=5,
        )
        self.assertNotIn(marker, json.dumps(details))

    def test_disabled_planning_restores_truncated_long_request_behavior(self):
        router, encoder = self.make_router(enabled=False)
        prompt = (
            "\n".join(["heartbeat status=ok"] * 100)
            + "\nInvestigate Event 4625."
        )
        selected = router.search(prompt)

        self.assertEqual(selected, [])
        self.assertEqual(len(encoder.calls), 1)

    def test_environment_and_cli_configuration_are_validated(self):
        self.assertEqual(
            load_query_plan_settings({}),
            QueryPlanSettings(),
        )

        parser = argparse.ArgumentParser()
        add_query_plan_arguments(parser)
        arguments = parser.parse_args(
            [
                "--disable-long-requests",
                "--query-chunk-tokens", "16",
                "--query-overlap-tokens", "4",
            ]
        )
        settings = query_plan_settings_from_arguments(arguments)

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.chunk_tokens, 16)
        self.assertEqual(settings.overlap_tokens, 4)

        for environment in (
            {"ROUTER_LONG_REQUESTS_ENABLED": "yes"},
            {"ROUTER_QUERY_MAX_VIEWS": "0"},
            {"ROUTER_QUERY_CHUNK_TOKENS": "not-an-integer"},
            {
                "ROUTER_QUERY_CHUNK_TOKENS": "8",
                "ROUTER_QUERY_OVERLAP_TOKENS": "8",
            },
        ):
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    load_query_plan_settings(environment)


if __name__ == "__main__":
    unittest.main()