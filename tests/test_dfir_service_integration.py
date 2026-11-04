"""Durable execution of bounded offline DFIR tools."""
from __future__ import annotations
import sys, tempfile, unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))
from analyst_preparation import PromptPreparer
from dfir_artifacts import ArtifactStore
from dfir_tools import ToolRegistry
from service_api import LocalJobAPI
from service_contracts import ExecutionProfile, JobState, JobSubmission, Principal
from service_inference import InferenceGate
from service_job_store import JobStore
from service_preparation import JobPreparation
from service_requests import JobRequests
from service_results import JobResults
from service_scheduler import JobScheduler
from service_settings import ServiceSettings
from service_worker import JobWorker

class DfirWorkerIntegrationTests(unittest.TestCase):
    def build(self, path, registry=None):
        settings = ServiceSettings()
       
        store = JobStore(path, settings)
        scheduler = JobScheduler(store)
        profile = ExecutionProfile("default", "model", "gpu", 100, 20, 10)
        preparer = PromptPreparer(Mock(search=Mock(return_value=[])), Mock(), lambda messages: 1, profile)
        preparation = JobPreparation(scheduler, preparer)
        results = JobResults(InferenceGate(scheduler))
        worker = JobWorker("worker-dfir", preparation, results, call_llm=Mock(side_effect=AssertionError("must not call LLM")), reserved_bytes=1, tool_registry=registry)
        return store, scheduler, preparation, results, worker

    def test_worker_uses_injected_registry_for_registered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
           
            root = Path(directory)
            source = root / "events.json"
            source.write_bytes((Path(__file__).parent / "fixtures" / "dfir" / "sysmon_powershell.json").read_bytes())
            artifacts = ArtifactStore(root)
            artifact = artifacts.register(source)
            registry = ToolRegistry(artifacts)
            store, scheduler, preparation, results, worker = self.build(root / "jobs.sqlite3", registry)
            owner = Principal("local", "local-operator")
            admission = JobRequests(scheduler).submit(owner, JobSubmission.from_payload({
                "prompt": "Analyze registered artifact",
                "tool": {"tool_id": "sysmon_powershell_offline_v1", "input": {"artifact_id": artifact.artifact_id}},
            }, idempotency_key="registered-artifact"))
            status = worker.run_once()
            self.assertEqual(status.state, JobState.COMPLETED)
            result = results.get(owner, admission.status.job_id)
            self.assertIn("sysmon_powershell_offline_v1", result.snapshot.answer)


if __name__ == "__main__":
    unittest.main()
