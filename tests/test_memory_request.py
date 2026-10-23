from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from router_query_plan import QueryPlanSettings

ANALYST_DIR = Path(__file__).resolve().parents[1] / "analyst"
sys.path.insert(0, str(ANALYST_DIR))

from memory_manager import MemoryManager
from memory_store import ConversationNotFoundError
from orchestrator import Skill, SkillRouter as RealSkillRouter
from router_policy import RouterPolicy, RouterSelectionPolicy


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ANALYST_DIR / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load test module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.database_path = (
            Path(self.temporary_directory.name) / "memory.sqlite3"
        )
        self.manager = MemoryManager(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-a",
        )

        settings = types.SimpleNamespace(
            memory=object(),
            model_name="test-model",
            max_pending_requests=32,
            pending_request_ttl_seconds=600,
            llm_url="http://unused/v1/chat/completions",
            context_size=130,
            max_output_tokens=20,
            context_safety_margin=10,
            input_token_budget=100,
            skills_dir=ANALYST_DIR / "skills",
            minilm_model_path=Path("/unused/model"),
            router_semantic_weight=0.6,
            router_name_boost=3.0,
            router_top_k=5,
            router_policy=None,
            router_selection=RouterSelectionPolicy(),
            router_query_plan=QueryPlanSettings(enabled=False),
            token_count_timeout_seconds=30,
        )

        self.router = types.SimpleNamespace(
            search=Mock(return_value=[])
        )
        registry = types.SimpleNamespace(
            skills=[],
            load_instructions=Mock(return_value="Skill guidance"),
        )

        config_module = types.ModuleType("config")
        config_module.settings = settings

        orchestrator_module = types.ModuleType("orchestrator")
        orchestrator_module.SkillRegistry = Mock(
            return_value=registry
        )
        orchestrator_module.SkillRouter = Mock(
            return_value=self.router
        )

        runtime_module = types.ModuleType("memory_runtime")
        runtime_module.build_memory_manager = Mock(
            return_value=self.manager
        )

        self.orchestrator_module = orchestrator_module

        with patch.dict(
            sys.modules,
            {
                "config": config_module,
                "orchestrator": orchestrator_module,
                "memory_runtime": runtime_module,
            },
        ):
            self.app = load_module(
                "_memory_test_analyst", "analyst.py"
            )

        self.real_count_input_tokens = self.app.count_input_tokens
        self.app.count_input_tokens = Mock(return_value=10)
        self.app.call_llm = Mock(
            return_value={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Generated answer",
                    }
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            }
        )

        with patch.dict(
            sys.modules,
            {
                "analyst": self.app,
                "orchestrator": orchestrator_module,
            },
        ):
            self.http = load_module(
                "_memory_test_http", "http_server.py"
            )

    def create_conversation(self):
        conversation = self.manager.create_conversation()
        self.assertIsNotNone(conversation)
        return conversation

    def start_pending(self, conversation_id=None):
        skill = types.SimpleNamespace(
            name="example-skill",
            description="Example",
        )
        self.router.search.return_value = [(0.9, skill)]
        self.app.count_input_tokens.side_effect = [110, 10, 10]

        return self.app.start_prompt(
            "Question",
            conversation_id=conversation_id,
        )

    def post(self, path, body):
        handler = object.__new__(
            self.http.OrchestratorHandler
        )
        encoded = json.dumps(body).encode("utf-8")
        handler.path = path
        handler.headers = {
            "Content-Length": str(len(encoded))
        }
        handler.rfile = io.BytesIO(encoded)
        handler.send_json = Mock()
        handler.send_error = Mock()

        handler.do_POST()

        handler.send_json.assert_called_once()
        status, payload = handler.send_json.call_args.args
        return status, payload

    def test_independent_request_does_not_save_or_change_shape(self):
        conversation = self.create_conversation()

        result = self.app.start_prompt("Question")

        self.assertNotIn("conversation_id", result)
        self.assertNotIn("memory_persistence", result)
        self.assertEqual(
            self.manager.get_messages(conversation.id), []
        )
        self.assertEqual(
            len(self.manager.list_conversations()), 1
        )

    def test_conversational_bypass_reaches_general_answer_without_skill_loading(self):
        router = RealSkillRouter.__new__(RealSkillRouter)
        router.registry = types.SimpleNamespace(skills=[Skill(
            "example", "Example guidance", "cybersecurity", "example", ()
        )])
        router.activation_policy = RouterPolicy()
        router.score_candidates = Mock()
        self.app.SKILL_ROUTER = router

        result = self.app.start_prompt("This is a test")

        self.assertFalse(result["selection_required"])
        self.assertEqual(result["content"], "Generated answer")
        router.score_candidates.assert_not_called()
        self.app.SKILL_REGISTRY.load_instructions.assert_not_called()
        self.assertEqual(self.app.count_input_tokens.call_count, 2)
        messages = self.app.call_llm.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertIn("No specialist skill was selected", messages[0]["content"])

    def test_conversational_bypass_still_persists_an_opted_in_exchange(self):
        router = RealSkillRouter.__new__(RealSkillRouter)
        router.registry = types.SimpleNamespace(skills=[Skill(
            "example", "Example guidance", "cybersecurity", "example", ()
        )])
        router.activation_policy = RouterPolicy()
        router.score_candidates = Mock()
        self.app.SKILL_ROUTER = router
        conversation = self.create_conversation()

        result = self.app.start_prompt(
            "Hello", conversation_id=conversation.id
        )

        self.assertEqual(result["memory_persistence"], {"saved": True})
        self.assertEqual(
            [message.content for message in self.manager.get_messages(conversation.id)],
            ["Hello", "Generated answer"],
        )
        router.score_candidates.assert_not_called()

    def test_disabled_memory_preserves_independent_requests(self):
        self.app.MEMORY_MANAGER = MemoryManager()

        result = self.app.start_prompt("Question")

        self.assertEqual(result["content"], "Generated answer")
        self.assertNotIn("memory_persistence", result)

        with self.assertRaises(ValueError):
            self.app.start_prompt(
                "Question", conversation_id="some-id"
            )

    def test_completed_exchange_is_saved_without_history_in_prompt(self):
        conversation = self.create_conversation()
        self.manager.record_completed_exchange(
            conversation.id,
            user_content="Old question",
            assistant_content="Old answer",
        )

        result = self.app.start_prompt(
            "New question",
            conversation_id=conversation.id,
        )

        self.assertEqual(
            result["memory_persistence"], {"saved": True}
        )
        self.assertEqual(
            result["conversation_id"], conversation.id
        )
        messages = self.manager.get_messages(conversation.id)
        self.assertEqual(
            [message.content for message in messages],
            [
                "Old question",
                "Old answer",
                "New question",
                "Generated answer",
            ],
        )

        prompt = self.app.call_llm.call_args.args[0]
        self.assertEqual(len(prompt), 2)
        self.assertEqual(
            prompt[-1],
            {"role": "user", "content": "New question"},
        )

    def test_wrong_scope_is_rejected_before_routing(self):
        other = MemoryManager(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-b",
        )
        conversation = other.create_conversation()

        with self.assertRaises(ConversationNotFoundError):
            self.app.start_prompt(
                "Question", conversation_id=conversation.id
            )

        self.router.search.assert_not_called()
        self.app.call_llm.assert_not_called()

    def test_pending_selection_saves_only_after_completion(self):
        conversation = self.create_conversation()

        pending = self.start_pending(conversation.id)

        self.assertTrue(pending["selection_required"])
        self.assertEqual(
            pending["conversation_id"], conversation.id
        )
        self.assertEqual(
            self.manager.get_messages(conversation.id), []
        )

        result = self.app.resolve_selection(
            pending["request_id"], ["example-skill"]
        )

        self.assertEqual(
            result["memory_persistence"], {"saved": True}
        )
        self.assertEqual(
            len(self.manager.get_messages(conversation.id)), 2
        )

        with self.assertRaises(KeyError):
            self.app.resolve_selection(
                pending["request_id"], ["example-skill"]
            )

    def test_deleted_pending_conversation_prevents_inference(self):
        conversation = self.create_conversation()
        pending = self.start_pending(conversation.id)
        self.manager.delete_conversation(conversation.id)

        with self.assertRaises(ConversationNotFoundError):
            self.app.resolve_selection(
                pending["request_id"], ["example-skill"]
            )

        self.app.call_llm.assert_not_called()
        self.assertNotIn(
            pending["request_id"], self.app.PENDING_REQUESTS
        )

    def test_inference_failure_saves_nothing(self):
        conversation = self.create_conversation()
        self.app.call_llm.side_effect = RuntimeError(
            "Simulated inference failure"
        )

        with self.assertRaises(RuntimeError):
            self.app.start_prompt(
                "Question", conversation_id=conversation.id
            )

        self.assertEqual(
            self.manager.get_messages(conversation.id), []
        )

    def test_save_failure_returns_answer_and_explicit_status(self):
        conversation = self.create_conversation()

        with patch.object(
            self.manager,
            "record_completed_exchange",
            side_effect=RuntimeError("Sensitive database detail"),
        ):
            result = self.app.start_prompt(
                "Question", conversation_id=conversation.id
            )

        self.assertEqual(result["content"], "Generated answer")
        self.assertFalse(
            result["memory_persistence"]["saved"]
        )
        self.assertNotIn(
            "Sensitive database detail", json.dumps(result)
        )
        self.assertEqual(
            self.manager.get_messages(conversation.id), []
        )

    def test_final_budget_check_still_blocks_inference(self):
        conversation = self.create_conversation()
        self.app.count_input_tokens.side_effect = [10, 101]

        with self.assertRaises(ValueError):
            self.app.start_prompt(
                "Question", conversation_id=conversation.id
            )

        self.app.call_llm.assert_not_called()
        self.assertEqual(
            self.manager.get_messages(conversation.id), []
        )

    def test_deletion_during_inference_is_reported_as_unsaved(self):
        conversation = self.create_conversation()
        response = self.app.call_llm.return_value

        def infer_and_delete(messages):
            self.manager.delete_conversation(conversation.id)
            return response

        self.app.call_llm.side_effect = infer_and_delete

        result = self.app.start_prompt(
            "Question", conversation_id=conversation.id
        )

        self.assertEqual(result["content"], "Generated answer")
        self.assertFalse(result["memory_persistence"]["saved"])

    def test_http_conversation_creation(self):
        status, payload = self.post("/v1/conversations", {})

        self.assertEqual(status, 201)
        conversation_id = payload["conversation"]["id"]
        self.assertEqual(
            self.manager.get_conversation(conversation_id).scope,
            "workspace-a",
        )

    def test_http_disabled_creation_is_rejected(self):
        self.app.MEMORY_MANAGER = MemoryManager()

        status, payload = self.post("/v1/conversations", {})

        self.assertEqual(status, 409)
        self.assertIn("disabled", payload["error"])

    def test_http_completion_includes_persistence_status(self):
        conversation = self.create_conversation()

        status, payload = self.post(
            "/v1/chat/completions",
            {
                "conversation_id": conversation.id,
                "messages": [{
                    "role": "user",
                    "content": "Question",
                }],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["conversation_id"], conversation.id
        )
        self.assertEqual(
            payload["memory_persistence"], {"saved": True}
        )

    def test_http_unknown_conversation_returns_not_found(self):
        status, payload = self.post(
            "/v1/chat/completions",
            {
                "conversation_id": "missing-id",
                "messages": [{
                    "role": "user",
                    "content": "Question",
                }],
            },
        )

        self.assertEqual(status, 404)
        self.app.call_llm.assert_not_called()

    def test_http_selection_cannot_override_conversation(self):
        conversation = self.create_conversation()
        pending = self.start_pending(conversation.id)

        status, payload = self.post(
            "/v1/chat/selection",
            {
                "request_id": pending["request_id"],
                "remove_skill_names": ["example-skill"],
                "conversation_id": "different-id",
            },
        )

        self.assertEqual(status, 400)
        self.app.call_llm.assert_not_called()
        self.assertEqual(
            self.manager.get_messages(conversation.id), []
        )

    def test_http_creation_rejects_client_scope(self):
        status, payload = self.post(
            "/v1/conversations",
            {"scope": "workspace-b"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(
            self.manager.list_conversations(), []
        )

    def memory_api(self, method, path):
        handler = object.__new__(
        self.http.OrchestratorHandler
            )
        handler.path = path
        handler.headers = {}
        handler.send_json = Mock()
        handler.send_error = Mock()

        if method == "GET":
            handler.do_GET()
        elif method == "DELETE":
            handler.do_DELETE()
        else:
            raise ValueError("Unsupported test method.")

        handler.send_json.assert_called_once()
        return handler.send_json.call_args.args

    def test_inspection_lists_only_configured_scope(self):
        own = self.create_conversation()
        other = MemoryManager(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-b",
        )
        other.create_conversation()

        status, payload = self.memory_api(
            "GET", "/v1/conversations"
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            [item["id"] for item in payload["conversations"]],
            [own.id],
        )
        self.assertEqual(payload["limit"], 100)
        self.assertEqual(payload["offset"], 0)

    def test_inspection_list_pagination(self):
        for _ in range(3):
            self.create_conversation()

        expected = self.manager.list_conversations(
            limit=1, offset=1
        )

        status, payload = self.memory_api(
            "GET", "/v1/conversations?limit=1&offset=1"
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            [item["id"] for item in payload["conversations"]],
            [item.id for item in expected],
        )
        self.assertEqual(payload["limit"], 1)
        self.assertEqual(payload["offset"], 1)

    def test_inspection_returns_metadata_without_messages(self):
        conversation = self.create_conversation()

        status, payload = self.memory_api(
            "GET", f"/v1/conversations/{conversation.id}"
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["conversation"]["id"], conversation.id
        )
        self.assertEqual(
            payload["conversation"]["scope"], "workspace-a"
        )
        self.assertNotIn("messages", payload)

    def test_inspection_returns_recent_complete_exchange(self):
        conversation = self.create_conversation()

        for number in range(3):
            self.manager.record_completed_exchange(
                conversation.id,
                user_content=f"Question {number}",
                assistant_content=f"Answer {number}",
            )

        status, payload = self.memory_api(
            "GET",
            f"/v1/conversations/{conversation.id}/messages"
            "?recent_exchanges=1",
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            [item["content"] for item in payload["messages"]],
            ["Question 2", "Answer 2"],
        )
        self.assertEqual(
            [item["sequence"] for item in payload["messages"]],
            [5, 6],
        )

    def test_inspection_default_history_is_bounded(self):
        conversation = self.create_conversation()

        for number in range(51):
            self.manager.record_completed_exchange(
                conversation.id,
                user_content=f"Question {number}",
                assistant_content=f"Answer {number}",
            )

        status, payload = self.memory_api(
            "GET",
            f"/v1/conversations/{conversation.id}/messages",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["recent_exchanges"], 50)
        self.assertEqual(len(payload["messages"]), 100)
        self.assertEqual(
            payload["messages"][0]["sequence"], 3
        )

    def test_http_deletion_survives_restart(self):
        conversation = self.create_conversation()
        self.manager.record_completed_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )

        status, payload = self.memory_api(
            "DELETE",
            f"/v1/conversations/{conversation.id}",
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "conversation_id": conversation.id,
                "deleted": True,
            },
        )

        restarted = MemoryManager(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-a",
        )

        with self.assertRaises(ConversationNotFoundError):
            restarted.get_messages(conversation.id)

        status, payload = self.memory_api(
            "DELETE",
            f"/v1/conversations/{conversation.id}",
        )
        self.assertEqual(status, 404)

    def test_inspection_and_deletion_hide_other_scopes(self):
        other = MemoryManager(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-b",
        )
        conversation = other.create_conversation()
        missing_id = "0" * 32

        for method, suffix in (
            ("GET", ""),
            ("GET", "/messages"),
            ("DELETE", ""),
        ):
            with self.subTest(method=method, suffix=suffix):
                inaccessible = self.memory_api(
                    method,
                    f"/v1/conversations/{conversation.id}{suffix}",
                )
                missing = self.memory_api(
                    method,
                    f"/v1/conversations/{missing_id}{suffix}",
                )

                self.assertEqual(inaccessible[0], 404)
                self.assertEqual(inaccessible, missing)

        self.assertEqual(
            other.get_conversation(conversation.id).id,
            conversation.id,
        )

    def test_inspection_and_deletion_reject_disabled_memory(self):
        conversation = self.create_conversation()
        self.app.MEMORY_MANAGER = MemoryManager()

        routes = [
            ("GET", "/v1/conversations"),
            (
                "GET",
                f"/v1/conversations/{conversation.id}",
            ),
            (
                "GET",
                f"/v1/conversations/{conversation.id}/messages",
            ),
            (
                "DELETE",
                f"/v1/conversations/{conversation.id}",
            ),
        ]

        for method, path in routes:
            with self.subTest(method=method, path=path):
                status, payload = self.memory_api(method, path)
                self.assertEqual(status, 409)

    def test_inspection_rejects_invalid_query_parameters(self):
        conversation = self.create_conversation()

        paths = [
            "/v1/conversations?limit=0",
            "/v1/conversations?limit=1001",
            "/v1/conversations?offset=-1",
            "/v1/conversations?limit=1&limit=2",
            "/v1/conversations?scope=workspace-b",
            "/v1/conversations?limit=",
            "/v1/conversations?limit",
            f"/v1/conversations/{conversation.id}?scope=workspace-b",
            (
                f"/v1/conversations/{conversation.id}/messages"
                "?recent_exchanges=0"
            ),
            (
                f"/v1/conversations/{conversation.id}/messages"
                "?recent_exchanges=1001"
            ),
        ]

        for path in paths:
            with self.subTest(path=path):
                status, payload = self.memory_api("GET", path)
                self.assertEqual(status, 400)

        status, payload = self.memory_api(
            "DELETE",
            f"/v1/conversations/{conversation.id}?scope=workspace-b",
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            self.manager.get_conversation(conversation.id).id,
            conversation.id,
        )

    def test_http_deletion_prevents_pending_generation(self):
        conversation = self.create_conversation()
        pending = self.start_pending(conversation.id)

        status, payload = self.memory_api(
            "DELETE",
            f"/v1/conversations/{conversation.id}",
        )

        self.assertEqual(status, 200)

        with self.assertRaises(ConversationNotFoundError):
            self.app.resolve_selection(
                pending["request_id"], ["example-skill"]
            )

        self.app.call_llm.assert_not_called()

    def test_inspection_errors_do_not_expose_storage_details(self):
        with patch.object(
            self.manager,
            "list_conversations",
            side_effect=RuntimeError(
                "Sensitive database details"
            ),
        ):
            status, payload = self.memory_api(
                "GET", "/v1/conversations"
            )

        self.assertEqual(status, 500)
        self.assertNotIn(
            "Sensitive database details",
            json.dumps(payload),
        )

    def test_memory_routes_preserve_static_files_and_reject_bulk_delete(self):
        handler = object.__new__(
            self.http.OrchestratorHandler
        )
        handler.path = "/"
        handler.serve_file = Mock()
        handler.send_json = Mock()

        handler.do_GET()

        handler.serve_file.assert_called_once_with(
            self.http.WEB_DIR / "index.html",
            "text/html; charset=utf-8",
        )
        handler.send_json.assert_not_called()

        conversation = self.create_conversation()

        status, payload = self.memory_api(
            "DELETE", "/v1/conversations"
        )
        self.assertEqual(status, 405)

        status, payload = self.memory_api(
            "GET",
            f"/v1/conversations/{conversation.id}/unknown",
        )
        self.assertEqual(status, 404)

        self.assertEqual(
            self.manager.get_conversation(conversation.id).id,
            conversation.id,
        )

    def test_completed_request_persists_usage_and_returns_full_totals(self):
        conversation = self.create_conversation()
        for _ in range(2):
            status, payload = self.post("/v1/chat/completions", {
                "conversation_id": conversation.id,
                "messages": [{"role": "user", "content": "Question"}],
            })
            self.assertEqual(status, 200)
        self.assertEqual(payload["conversation_usage"]["total_tokens"], 26)
        self.assertEqual(payload["conversation_usage"]["exchange_count"], 2)
        self.assertEqual(payload["conversation_usage"]["latest_usage"]["context_size"], 130)
        self.assertEqual(payload["conversation_usage"]["latest_usage"]["model_name"], "test-model")
        status, inspection = self.memory_api("GET", f"/v1/conversations/{conversation.id}/usage")
        self.assertEqual(status, 200)
        self.assertEqual(inspection["usage"], payload["conversation_usage"])

    def test_missing_generation_counts_remain_unknown(self):
        conversation = self.create_conversation()
        self.app.call_llm.return_value["usage"] = {}
        result = self.app.start_prompt("Question", conversation_id=conversation.id)
        self.assertEqual(result["stats"]["input_tokens"], 10)
        self.assertIsNone(result["stats"]["output_tokens"])
        self.assertIsNone(result["stats"]["tokens_used"])
        self.assertIsNone(result["conversation_usage"]["total_tokens"])
        self.assertTrue(result["conversation_usage"]["is_partial"])

    def test_token_counter_rejects_boolean_and_negative_counts(self):
        for value in (True, -1, 1.5, "10"):
            with self.subTest(value=value):
                body = io.BytesIO(json.dumps({"input_tokens": value}).encode())
                with patch.object(self.app.urllib.request, "urlopen", return_value=body):
                    with self.assertRaises(ValueError):
                        self.real_count_input_tokens([{"role": "user", "content": "Question"}])

    def test_invalid_counts_are_ignored_and_inconsistent_total_is_reconciled(self):
        self.app.call_llm.return_value["usage"] = {
            "prompt_tokens": True, "completion_tokens": -1, "total_tokens": -1
        }
        result = self.app.start_prompt("Question")
        self.assertEqual(result["stats"]["input_tokens"], 10)
        self.assertIsNone(result["stats"]["total_tokens"])
        self.app.call_llm.return_value["usage"] = {
            "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 999
        }
        self.assertEqual(self.app.start_prompt("Question")["stats"]["total_tokens"], 13)

    def test_metrics_read_failure_does_not_misreport_a_committed_exchange(self):
        conversation = self.create_conversation()
        with patch.object(self.manager, "get_conversation_usage", side_effect=RuntimeError("Read failed")):
            result = self.app.start_prompt("Question", conversation_id=conversation.id)
        self.assertEqual(result["memory_persistence"], {"saved": True})
        self.assertIsNone(result["conversation_usage"])
        self.assertEqual(len(self.manager.get_messages(conversation.id)), 2)
        self.assertEqual(self.manager.get_conversation_usage(conversation.id)["total_tokens"], 13)

    def test_usage_endpoint_enforces_scope_and_rejects_scope_parameters(self):
        other = MemoryManager(enabled=True, database_path=self.database_path, scope="workspace-b")
        conversation = other.create_conversation()
        inaccessible = self.memory_api("GET", f"/v1/conversations/{conversation.id}/usage")
        missing = self.memory_api("GET", f"/v1/conversations/{'0' * 32}/usage")
        self.assertEqual(inaccessible[0], 404)
        self.assertEqual(inaccessible, missing)
        own = self.create_conversation()
        self.assertEqual(self.memory_api("GET", f"/v1/conversations/{own.id}/usage?scope=workspace-b")[0], 400)
        self.assertEqual(self.memory_api("DELETE", f"/v1/conversations/{own.id}/usage")[0], 405)

    def test_usage_endpoint_rejects_disabled_memory(self):
        conversation = self.create_conversation()
        self.app.MEMORY_MANAGER = MemoryManager()
        self.assertEqual(self.memory_api("GET", f"/v1/conversations/{conversation.id}/usage")[0], 409)

    def test_pending_selections_do_not_record_usage_until_completion(self):
        conversation = self.create_conversation()
        pending = self.start_pending(conversation.id)
        self.assertEqual(self.manager.get_conversation_usage(conversation.id)["exchange_count"], 0)
        result = self.app.resolve_selection(pending["request_id"], ["example-skill"])
        self.assertEqual(result["conversation_usage"]["exchange_count"], 1)
        self.assertEqual(result["conversation_usage"]["total_tokens"], 13)

    def test_http_named_creation_returns_persisted_title(self):
        status, payload = self.post(
            "/v1/conversations",
            {"title": "  Blue Lantern  "},
        )

        self.assertEqual(status, 201)
        conversation = payload["conversation"]
        self.assertEqual(conversation["title"], "Blue Lantern")
        self.assertEqual(
            self.manager.get_conversation(conversation["id"]).title,
            "Blue Lantern",
        )

        status, listing = self.memory_api(
            "GET", "/v1/conversations"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            listing["conversations"][0]["title"], "Blue Lantern"
        )

    def test_http_rename_preserves_messages_and_id(self):
        conversation = self.create_conversation()
        exchange = self.manager.record_completed_exchange(
            conversation.id,
            user_content="Question",
            assistant_content="Answer",
        )

        status, payload = self.post(
            f"/v1/conversations/{conversation.id}/title",
            {"title": "SSH investigation"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            payload["conversation"]["id"], conversation.id
        )
        self.assertEqual(
            payload["conversation"]["title"],
            "SSH investigation",
        )
        self.assertEqual(
            self.manager.get_messages(conversation.id),
            list(exchange),
        )

    def test_http_titles_reject_invalid_values(self):
        conversation = self.create_conversation()

        for title in ("", " ", "x" * 121, "Two\nlines", None):
            with self.subTest(title=repr(title)):
                status, payload = self.post(
                    "/v1/conversations", {"title": title}
                )
                self.assertEqual(status, 400)

                status, payload = self.post(
                    f"/v1/conversations/{conversation.id}/title",
                    {"title": title},
                )
                self.assertEqual(status, 400)

        self.assertEqual(
            len(self.manager.list_conversations()), 1
        )
        self.assertEqual(
            self.manager.get_conversation(conversation.id).title,
            conversation.title,
        )

    def test_http_rename_rejects_scope_and_missing_title(self):
        conversation = self.create_conversation()

        for body in (
            {},
            {"title": "New name", "scope": "workspace-b"},
        ):
            with self.subTest(body=body):
                status, payload = self.post(
                    f"/v1/conversations/{conversation.id}/title",
                    body,
                )
                self.assertEqual(status, 400)

    def test_http_rename_hides_other_scopes(self):
        other = MemoryManager(
            enabled=True,
            database_path=self.database_path,
            scope="workspace-b",
        )
        conversation = other.create_conversation(
            title="Other workspace"
        )

        inaccessible = self.post(
            f"/v1/conversations/{conversation.id}/title",
            {"title": "Unauthorized"},
        )
        missing = self.post(
            f"/v1/conversations/{'0' * 32}/title",
            {"title": "Missing"},
        )

        self.assertEqual(inaccessible[0], 404)
        self.assertEqual(inaccessible, missing)
        self.assertEqual(
            other.get_conversation(conversation.id).title,
            "Other workspace",
        )

    def test_http_rename_rejects_disabled_memory(self):
        conversation = self.create_conversation()
        self.app.MEMORY_MANAGER = MemoryManager()

        status, payload = self.post(
            f"/v1/conversations/{conversation.id}/title",
            {"title": "New name"},
        )

        self.assertEqual(status, 409)

if __name__ == "__main__":
    unittest.main()
