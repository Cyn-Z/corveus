"""End-to-end loopback HTTP coverage for analyst-directed DFIR jobs."""
from __future__ import annotations

import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from analyst_preparation import PromptPreparer
from memory_manager import MemoryManager
from service_contracts import ExecutionProfile
from service_inference import InferenceGate
from service_job_store import JobStore
from service_preparation import JobPreparation
from service_results import JobResults
from service_scheduler import JobScheduler
from service_settings import ServiceSettings
from service_worker import JobWorker

ANALYST_DIR = Path(__file__).resolve().parents[1] / "analyst"


def load_http_server():
    fake_analyst = types.SimpleNamespace(
        MEMORY_MANAGER=types.SimpleNamespace(enabled=False),
        SKILL_ROUTER=None,
        start_prompt=Mock(), resolve_selection=Mock(),
        get_llama_stats=Mock(return_value={}),
    )
    spec = importlib.util.spec_from_file_location(
        "dfir_http_test_http", ANALYST_DIR / "http_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"analyst": fake_analyst, "orchestrator": types.SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class DfirHttpApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        settings = ServiceSettings()
        self.store = JobStore(Path(self.directory.name) / "jobs.sqlite3", settings)
        scheduler = JobScheduler(self.store)
        profile = ExecutionProfile("default", "model", "gpu", 100, 20, 10)
        preparer = PromptPreparer(Mock(search=Mock(return_value=[])), Mock(), lambda messages: 1, profile)
        self.preparation = JobPreparation(scheduler, preparer)
        self.results = JobResults(InferenceGate(scheduler))
        self.worker = JobWorker(
            "http-dfir-worker", self.preparation, self.results,
            call_llm=Mock(side_effect=AssertionError("DFIR HTTP job must not call the LLM")),
            reserved_bytes=1,
        )
        self.http = load_http_server()
        self.server = self.http.HTTPServer(("127.0.0.1", 0), self.http.OrchestratorHandler)
        self.shared_memory = MemoryManager()
        self.api = self.http.LocalJobAPI(
            self.results, profile_id="default", memory=self.shared_memory,
            preparation=self.preparation,
        )
        self.http.analyst.MEMORY_MANAGER = self.shared_memory
        self.http.attach_local_job_api(self.server, self.api)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path, *, body=None, key=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        headers = {}
        encoded = None
        if body is not None:
            encoded = json.dumps(body).encode("utf-8")
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(encoded))})
        if key is not None:
            headers["Idempotency-Key"] = key
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        status = response.status
        connection.close()
        return status, data
    def test_submit_worker_and_result_round_trip_over_http(self):
        status, submitted = self.request("POST", "/v1/jobs", body=self.valid_payload(), key="http-dfir-1")
        self.assertEqual(status, 202)
        job_id = submitted["job"]["job_id"]
        self.assertEqual(self.worker.run_once().state.value, "completed")

        status, completed = self.request("GET", f"/v1/jobs/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(completed["job"]["state"], "completed")
        status, result = self.request("GET", f"/v1/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(result["dfir"]["tool_id"], "sysmon_powershell_offline_v1")
        self.assertEqual(result["dfir"]["event_count"], 4)
        self.assertTrue(result["dfir"]["provenance_valid"])
        self.assertTrue(result["dfir"]["findings"])

    def test_idempotency_and_invalid_tool_inputs_are_enforced_over_http(self):
        payload = self.valid_payload()
        first_status, first = self.request("POST", "/v1/jobs", body=payload, key="http-dfir-2")
        retry_status, retry = self.request("POST", "/v1/jobs", body=payload, key="http-dfir-2")
        self.assertEqual((first_status, retry_status), (202, 202))
        self.assertEqual(first["job"]["job_id"], retry["job"]["job_id"])
        self.assertFalse(retry["created"])

        status, invalid = self.request(
            "POST", "/v1/jobs",
            body={"prompt": "Analyze", "tool": {"tool_id": "arbitrary-shell", "input": {}}},
            key="http-dfir-bad-tool",
        )
        self.assertEqual(status, 400)
        self.assertEqual(invalid["error"]["code"], "invalid_request")

        status, invalid = self.request(
            "POST", "/v1/jobs",
            body={"prompt": "Analyze", "tool": {"tool_id": "sysmon_powershell_offline_v1", "input": {"path": "C:\\secret"}}},
            key="http-dfir-bad-path",
        )
        self.assertEqual(status, 400)
        self.assertEqual(invalid["error"]["code"], "invalid_request")

        status, conflict = self.request(
            "POST", "/v1/jobs",
            body={"prompt": "Changed", "tool": payload["tool"]},
            key="http-dfir-2",
        )
        self.assertEqual(status, 409)
        self.assertEqual(conflict["error"]["code"], "conflict")

    def valid_payload(self):
        return {
            "prompt": "Analyze the trusted Sysmon fixture.",
            "tool": {"tool_id": "sysmon_powershell_offline_v1", "input": {"fixture_id": "sysmon_powershell"}},
        }


if __name__ == "__main__":
    unittest.main()
