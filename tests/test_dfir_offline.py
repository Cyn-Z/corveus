"""Deterministic offline PowerShell/Sysmon DFIR vertical slice."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))

from dfir_artifacts import ArtifactStore
from dfir_contracts import EvidenceRef, Finding
from dfir_correlation import correlate_network, reconstruct_ancestry
from dfir_findings import analyze_events, validate_findings
from dfir_powershell import decode_encoded_command, inspect_powershell
from dfir_sysmon import load_events


FIXTURE = Path(__file__).parent / "fixtures" / "dfir" / "sysmon_powershell.json"


class ArtifactTests(unittest.TestCase):
    def test_artifact_is_hashed_and_root_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.json"
            source.write_bytes(FIXTURE.read_bytes())
            store = ArtifactStore(root)
            artifact = store.register(source)
            self.assertEqual(len(artifact.sha256), 64)
            self.assertEqual(artifact.size_bytes, source.stat().st_size)
            self.assertEqual(Path(artifact.storage_path), source.resolve())
            with self.assertRaises(ValueError):
                store.register(root.parent / source.name)


class PowerShellTests(unittest.TestCase):
    def test_encoded_command_decodes_as_text_only(self):
        decoded = decode_encoded_command("UwB0AGEAcgB0AC0AUAByAG8AYwBlAHMAcwA=")
        self.assertTrue(decoded.success)
        self.assertEqual(decoded.encoding, "utf-16le")
        self.assertEqual(decoded.text, "Start-Process")

    def test_malformed_encoded_command_fails_closed(self):
        decoded = decode_encoded_command("not-base64")
        self.assertFalse(decoded.success)
        self.assertIsNone(decoded.text)

    def test_powershell_inspection_reports_encoded_command(self):
        events = load_events(FIXTURE)
        records = inspect_powershell(events)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].encoded.success)
        self.assertEqual(records[0].event.event_id, 1)


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.events = load_events(FIXTURE)

    def test_ancestry_preserves_missing_parent_as_unknown(self):
        ancestry = reconstruct_ancestry(self.events)
        child = next(item for item in ancestry if item.process_id == 1300)
        self.assertEqual(child.parent_process_id, 9999)
        self.assertFalse(child.parent_observed)
        self.assertTrue(child.evidence)

    def test_network_correlation_requires_process_and_time_window(self):
        matches = correlate_network(self.events, window_seconds=10)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].process_id, 1200)
        self.assertEqual(matches[0].network_event.event_id, 3)


class FindingTests(unittest.TestCase):
    def test_analysis_is_deterministic_and_evidence_backed(self):
        events = load_events(FIXTURE)
        first = analyze_events(events)
        second = analyze_events(events)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(isinstance(item, Finding) for item in first))
        self.assertTrue(all(item.evidence_refs for item in first))
        self.assertTrue(any(item.category == "encoded_powershell" for item in first))
        self.assertTrue(any(item.category == "process_network_correlation" for item in first))

    def test_evidence_reference_rejects_invalid_hash(self):
        with self.assertRaises(ValueError):
            EvidenceRef("event-1", "0" * 63, 0, 1)

    def test_findings_validate_against_original_events(self):
        events = load_events(FIXTURE)
        findings = analyze_events(events)
        self.assertTrue(validate_findings(findings, events))
        altered = list(events)
        altered[0] = altered[0].__class__(
            event_id=altered[0].event_id,
            provider=altered[0].provider,
            timestamp=altered[0].timestamp,
            host=altered[0].host,
            process_id=altered[0].process_id,
            parent_process_id=altered[0].parent_process_id,
            image=altered[0].image,
            command_line="powershell.exe -NoProfile -EncodedCommand altered",
            user=altered[0].user,
            raw_index=altered[0].raw_index,
        )
        self.assertFalse(validate_findings(findings, tuple(altered)))

    def test_findings_reject_unknown_event_reference(self):
        events = load_events(FIXTURE)
        finding = analyze_events(events)[0]
        unknown = finding.__class__(
            finding_id=finding.finding_id,
            category=finding.category,
            severity=finding.severity,
            title=finding.title,
            summary=finding.summary,
            confidence=finding.confidence,
            evidence_refs=(EvidenceRef("event-999", "0" * 64),),
            detector_id=finding.detector_id,
            detector_version=finding.detector_version,
        )
        self.assertFalse(validate_findings((unknown,), events))


if __name__ == "__main__":
    unittest.main()
