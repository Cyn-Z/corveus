"""Batch 4A: independent generation, per-request metrics, and safe boundaries."""

from __future__ import annotations

import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

ANALYST_DIR = Path(__file__).resolve().parents[1] / "analyst"
sys.path.insert(0, str(ANALYST_DIR))

from analyst_generation import (
    InferenceCallUncertainError,
    InvalidInferenceResponseError,
    generate_answer,
)
from token_usage import MAX_SQLITE_INTEGER


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "system", "content": "Analyst guidance"},
                         {"role": "user", "content": "Question"}]
        self.counter = Mock(return_value=10)
        self.client = Mock(return_value={
            "choices": [{"message": {"content": "Answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        })

    def run_generation(self, **changes):
        arguments = {"count_input_tokens": self.counter, "call_llm": self.client,
                     "model_name": "local-model", "context_size": 130, "input_budget": 100}
        arguments.update(changes)
        return generate_answer(self.messages, **arguments)

    def test_returns_answer_and_request_local_usage_in_legacy_shape(self):
        result = self.run_generation()
        self.assertEqual(result.content, "Answer")
        self.assertEqual(result.usage.total_tokens, 13)
        payload = result.as_legacy_response()
        self.assertFalse(payload["selection_required"])
        self.assertEqual(payload["stats"]["tokens_used"], 13)
        self.assertEqual(payload["stats"]["ctx_size"], 130)
        self.client.assert_called_once_with(self.messages)

    def test_budget_is_verified_immediately_before_inference(self):
        self.counter.return_value = 101
        with self.assertRaises(ValueError):
            self.run_generation()
        self.client.assert_not_called()

    def test_invalid_token_counts_block_dispatch(self):
        for value in (True, -1, None, "10"):
            with self.subTest(value=value):
                self.counter.return_value = value
                with self.assertRaises(ValueError):
                    self.run_generation()
        self.client.assert_not_called()

    def test_missing_usage_retains_verified_input_and_unknown_output(self):
        self.client.return_value.pop("usage")
        result = self.run_generation()
        self.assertEqual(result.usage.input_tokens, 10)
        self.assertIsNone(result.usage.output_tokens)
        self.assertIsNone(result.usage.total_tokens)

    def test_reported_input_is_preserved_and_total_is_reconciled(self):
        self.client.return_value["usage"] = {
            "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 999,
        }
        result = self.run_generation()
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.total_tokens, 15)

    def test_invalid_backend_counts_remain_unknown(self):
        self.client.return_value["usage"] = {
            "prompt_tokens": True, "completion_tokens": -1, "total_tokens": 1,
        }
        usage = self.run_generation().usage
        self.assertEqual(usage.input_tokens, 10)
        self.assertIsNone(usage.output_tokens)
        self.assertIsNone(usage.total_tokens)

    def test_overflow_is_not_persisted_as_an_invalid_token_total(self):
        self.client.return_value["usage"] = {
            "prompt_tokens": MAX_SQLITE_INTEGER, "completion_tokens": 1,
        }
        usage = self.run_generation().usage
        self.assertEqual(usage.input_tokens, MAX_SQLITE_INTEGER)
        self.assertIsNone(usage.output_tokens)
        self.assertIsNone(usage.total_tokens)

    def test_client_error_is_uncertain_and_does_not_expose_exception_text(self):
        self.client.side_effect = TimeoutError("secret prompt and internal URL")
        with self.assertRaises(InferenceCallUncertainError) as caught:
            self.run_generation()
        self.assertFalse(caught.exception.inference_stopped)
        self.assertNotIn("secret", str(caught.exception))
        self.client.assert_called_once()

    def test_malformed_completed_response_is_distinguished_from_transport_uncertainty(self):
        for payload in (None, {}, {"choices": []}, {"choices": [{}]},
                        {"choices": [{"message": {"content": ["invalid"]}}]}):
            with self.subTest(payload=payload):
                self.client.return_value = payload
                with self.assertRaises(InvalidInferenceResponseError) as caught:
                    self.run_generation()
                self.assertTrue(caught.exception.inference_stopped)

    def test_pre_dispatch_counter_failure_does_not_call_model(self):
        self.counter.side_effect = ValueError("tokenizer failed")
        with self.assertRaises(ValueError):
            self.run_generation()
        self.client.assert_not_called()

    def test_callback_mutation_cannot_change_caller_messages_or_model_input(self):
        original = deepcopy(self.messages)
        def count(messages):
            messages[-1]["content"] = "Changed by counter"
            return 10
        self.counter.side_effect = count
        self.run_generation()
        self.assertEqual(self.messages, original)
        self.client.assert_called_once_with(original)

    def test_request_outputs_and_returned_dictionaries_are_independent(self):
        first = self.run_generation()
        self.client.return_value["usage"]["completion_tokens"] = 7
        second = self.run_generation()
        first.as_legacy_response()["stats"]["input_tokens"] = 999
        self.assertEqual(first.usage.total_tokens, 13)
        self.assertEqual(second.usage.total_tokens, 17)
        self.assertEqual(first.as_legacy_response()["stats"]["input_tokens"], 10)

    def test_missing_content_preserves_existing_empty_response_behavior(self):
        self.client.return_value["choices"][0]["message"]["content"] = None
        self.assertEqual(self.run_generation().content, "")

    def test_invalid_settings_are_rejected_before_dispatch(self):
        for changes in ({"input_budget": True}, {"input_budget": 131},
                        {"context_size": 0}, {"model_name": ""}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.run_generation(**changes)
        self.client.assert_not_called()

    def test_import_does_not_load_models_application_or_storage(self):
        code = (
            "import sys; import analyst_generation; "
            "assert not ({'analyst','config','orchestrator','torch',"
            "'sentence_transformers','memory_manager','service_results'} & set(sys.modules))"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ANALYST_DIR,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
