from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "analyst"),
)

from router_query_plan import QueryPlanner, QueryPlanSettings


class WordTokenizer:
    """Small deterministic tokenizer with two special tokens."""

    model_max_length = 256

    def __init__(self):
        self.word_ids = {}
        self.id_words = {}
        self.calls = []

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, *, add_special_tokens=True, truncation=False):
        self.calls.append(
            {
                "text": text,
                "special": add_special_tokens,
                "truncation": truncation,
            }
        )
        ids = []
        for word in text.split():
            if word not in self.word_ids:
                identifier = len(self.word_ids) + 10
                self.word_ids[word] = identifier
                self.id_words[identifier] = word
            ids.append(self.word_ids[word])

        return [1, *ids, 2] if add_special_tokens else ids

    def decode(
        self,
        ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(self.id_words[identifier] for identifier in ids)


class ExpandingTokenizer(WordTokenizer):
    """Simulate a decode/re-encode round trip that grows by one token."""

    def decode(self, ids, **options):
        return super().decode(ids, **options) + " expanded"


class QueryPlannerTests(unittest.TestCase):
    def planner(self, tokenizer=None, model_limit=256, **settings):
        tokenizer = tokenizer or WordTokenizer()
        return QueryPlanner(
            tokenizer,
            model_max_seq_length=model_limit,
            settings=QueryPlanSettings(**settings),
        )

    def test_short_prompt_is_preserved_exactly(self):
        prompt = "  Investigate   Event 4625.\n"
        plan = self.planner().plan(prompt)

        self.assertEqual(plan.mode, "short")
        self.assertEqual(plan.views[0].text, prompt)
        self.assertFalse(plan.would_truncate)
        self.assertTrue(plan.window_coverage_complete)

    def test_effective_limit_uses_model_and_tokenizer(self):
        tokenizer = WordTokenizer()
        tokenizer.model_max_length = 64
        planner = self.planner(tokenizer, model_limit=128)

        self.assertEqual(planner.effective_token_limit, 64)
        self.assertEqual(planner.payload_token_limit, 62)

    def test_special_tokens_are_included_in_fit_checks(self):
        plan = self.planner(
            model_limit=10,
            chunk_tokens=20,
            overlap_tokens=2,
        ).plan(" ".join(f"word{i}" for i in range(25)))

        self.assertEqual(plan.payload_token_limit, 8)
        for view in plan.views:
            self.assertLessEqual(view.encoded_tokens, 10)
            self.assertLessEqual(view.content_tokens, 8)

    def test_long_prompt_has_bounded_views_and_honest_coverage(self):
        plan = self.planner(
            chunk_tokens=16,
            overlap_tokens=4,
            max_views=4,
            max_directive_views=0,
        ).plan(" ".join(f"word{i}" for i in range(1000)))

        self.assertEqual(plan.mode, "chunked")
        self.assertTrue(plan.would_truncate)
        self.assertLessEqual(len(plan.views), 4)
        self.assertFalse(plan.window_coverage_complete)
        self.assertGreater(
            plan.diagnostic()["omitted_window_tokens"],
            0,
        )

    def test_sampling_contains_head_and_tail_windows(self):
        words = [f"word{i}" for i in range(600)]
        plan = self.planner(
            chunk_tokens=32,
            overlap_tokens=8,
            max_views=4,
            max_directive_views=0,
        ).plan(" ".join(words))

        ranges = [
            interval
            for view in plan.views
            for interval in view.token_ranges
        ]
        self.assertTrue(any(start == 0 for start, _ in ranges))
        self.assertTrue(any(end == 600 for _, end in ranges))

    def test_overlap_is_not_double_counted(self):
        plan = self.planner(
            model_limit=10,
            chunk_tokens=8,
            overlap_tokens=4,
            max_views=8,
            max_directive_views=0,
        ).plan(" ".join(f"word{i}" for i in range(30)))

        self.assertEqual(plan.sampled_window_tokens, 30)
        self.assertEqual(plan.diagnostic()["omitted_window_tokens"], 0)

    def test_directives_at_beginning_middle_and_end_are_reserved(self):
        task = "Investigate Windows Event 4625 using Splunk."
        evidence = "\n".join(f"status{i}=ok" for i in range(500))

        for prompt in (
            task + "\n" + evidence,
            evidence + "\n" + task,
            evidence + "\n" + task + "\n" + evidence,
        ):
            with self.subTest(position=prompt.find(task)):
                plan = self.planner().plan(prompt)
                directive_views = [
                    view
                    for view in plan.views
                    if "directive" in view.sources
                ]
                self.assertTrue(directive_views)
                self.assertTrue(
                    any(task in view.text for view in directive_views)
                )

    def test_code_fences_and_quotes_do_not_create_directive_reservations(self):
        prompt = (
            "Investigate Windows Event 4625.\n"
            "```text\n"
            "Audit Kubernetes RBAC.\n"
            "```\n"
            "> Analyze this packet capture.\n"
            + "\n".join(f"status{i}=ok" for i in range(500))
        )
        plan = self.planner().plan(prompt)

        self.assertEqual(plan.directive_candidates, 1)
        directive_views = [
            view
            for view in plan.views
            if "directive" in view.sources
        ]
        self.assertEqual(len(directive_views), 1)
        self.assertIn("4625", directive_views[0].text)

    def test_duplicate_directives_and_windows_share_one_view(self):
        prompt = "\n".join(
            ["Investigate phishing."] * 300
        )
        plan = self.planner().plan(prompt)

        self.assertEqual(plan.directive_candidates, 1)
        texts = [view.text for view in plan.views]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertLessEqual(len(plan.views), 8)

    def test_no_silent_tokenizer_truncation_is_requested(self):
        tokenizer = WordTokenizer()
        self.planner(tokenizer).plan(
            " ".join(f"word{i}" for i in range(500))
        )

        self.assertTrue(tokenizer.calls)
        self.assertTrue(
            all(call["truncation"] is False for call in tokenizer.calls)
        )

    def test_decode_round_trip_expansion_still_fits(self):
        plan = self.planner(
            ExpandingTokenizer(),
            model_limit=10,
            chunk_tokens=8,
            overlap_tokens=2,
        ).plan(" ".join(f"word{i}" for i in range(100)))

        for view in plan.views:
            self.assertLessEqual(view.encoded_tokens, 10)
            self.assertLessEqual(view.content_tokens, 8)

    def test_disabled_mode_preserves_original_long_input(self):
        prompt = " ".join(f"word{i}" for i in range(500))
        plan = self.planner(enabled=False).plan(prompt)

        self.assertEqual(plan.mode, "disabled")
        self.assertEqual(len(plan.views), 1)
        self.assertEqual(plan.views[0].text, prompt)
        self.assertTrue(plan.would_truncate)
        self.assertFalse(plan.window_coverage_complete)

    def test_diagnostics_do_not_include_prompt_or_decoded_text(self):
        marker = "PRIVATE_MARKER_4625"
        plan = self.planner().plan(
            marker + "\n" + " ".join(f"word{i}" for i in range(500))
        )
        diagnostic = plan.diagnostic()
        serialized = json.dumps(diagnostic, allow_nan=False)

        self.assertNotIn(marker, serialized)
        self.assertTrue(
            all("text" not in view for view in diagnostic["views"])
        )

    def test_invalid_settings_are_rejected(self):
        for arguments in (
            {"enabled": "true"},
            {"chunk_tokens": True},
            {"chunk_tokens": 0},
            {"chunk_tokens": 4097},
            {"overlap_tokens": -1},
            {"chunk_tokens": 8, "overlap_tokens": 8},
            {"max_views": 1},
            {"max_views": 33},
            {"max_directive_views": -1},
            {"max_prompt_characters": 0},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    QueryPlanSettings(**arguments)

    def test_invalid_input_and_impossible_model_limits_are_rejected(self):
        planner = self.planner(max_prompt_characters=10)

        for prompt in (None, "", "   ", "x" * 11):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    planner.plan(prompt)

        for limit in (0, -1, True, 2):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.planner(model_limit=limit)


if __name__ == "__main__":
    unittest.main()