"""Typed deterministic DFIR tool contracts."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from dfir_artifacts import ArtifactStore
from dfir_tools import ToolRegistry, ToolRequest


class ToolRegistryTests(unittest.TestCase):
    def test_registered_sysmon_tool_is_read_only_and_deterministic(self):
        registry = ToolRegistry()
        spec = registry.get("sysmon_powershell_offline_v1")
        self.assertTrue(spec.deterministic)
        self.assertTrue(spec.read_only)
        self.assertFalse(spec.network_required)
        self.assertFalse(spec.executes_processes)
        self.assertFalse(spec.modifies_artifacts)

    def test_unknown_tools_and_arbitrary_paths_are_rejected(self):
        registry = ToolRegistry()
        with self.assertRaises(ValueError):
            registry.get("arbitrary-shell")
        with self.assertRaises(ValueError):
            ToolRequest.from_payload({"tool_id": "sysmon_powershell_offline_v1", "input": {"path": "C:\\secret"}})

    def test_tool_request_round_trip_is_canonical(self):
        request = ToolRequest.from_payload({
            "tool_id": "sysmon_powershell_offline_v1",
            "input": {"fixture_id": "sysmon_powershell"},
        })
        self.assertEqual(ToolRequest.from_json(request.to_json()), request)

    def test_registered_tool_executes_fixture_without_external_processes(self):
        request = ToolRequest.from_payload({
            "tool_id": "sysmon_powershell_offline_v1",
            "input": {"fixture_id": "sysmon_powershell"},
        })
        result = ToolRegistry().execute(request)
        self.assertEqual(result.event_count, 4)
        self.assertTrue(result.provenance_valid)
        self.assertTrue(any(item.category == "encoded_powershell" for item in result.findings))

    def test_tool_result_round_trip_provides_structured_public_payload(self):
        request = ToolRequest.from_payload({
            "tool_id": "sysmon_powershell_offline_v1",
            "input": {"fixture_id": "sysmon_powershell"},
        })
        result = ToolRegistry().execute(request)
        restored = result.__class__.from_json(result.to_json())
        payload = restored.as_payload()
        self.assertEqual(payload["tool_id"], "sysmon_powershell_offline_v1")
        self.assertEqual(payload["event_count"], 4)
        self.assertTrue(payload["findings"])
        self.assertIn("evidence_refs", payload["findings"][0])

    def test_artifact_registry_assigns_id_and_resolves_only_registered_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.json"
            source.write_text((Path(__file__).parent / "fixtures" / "dfir" / "sysmon_powershell.json").read_text(), encoding="utf-8")
            registry = ArtifactStore(root)
            artifact = registry.register(source)
            self.assertEqual(registry.get(artifact.artifact_id), artifact)
            self.assertEqual(registry.open(artifact.artifact_id).read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError):
                registry.get("f" * 32)

    def test_artifact_id_tool_request_executes_registered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.json"
            source.write_bytes((Path(__file__).parent / "fixtures" / "dfir" / "sysmon_powershell.json").read_bytes())
            artifacts = ArtifactStore(root)
            artifact = artifacts.register(source)
            request = ToolRequest.from_payload({
                "tool_id": "sysmon_powershell_offline_v1",
                "input": {"artifact_id": artifact.artifact_id},
            })
            result = ToolRegistry(artifacts).execute(request)
            self.assertEqual(result.event_count, 4)



if __name__ == "__main__":
    unittest.main()
