"""Evidence-backed deterministic findings for the first offline slice."""
from __future__ import annotations

import hashlib
import json

from dfir_contracts import EvidenceRef, Finding
from dfir_correlation import correlate_network, reconstruct_ancestry
from dfir_powershell import inspect_powershell


def _source_hash(event):
    payload = json.dumps(event.__dict__, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _ref(event):
    return EvidenceRef(f"event-{event.raw_index}", _source_hash(event))


def validate_findings(findings, events):
    """Verify every finding reference still matches the supplied events."""
    by_ref = {f"event-{event.raw_index}": _source_hash(event) for event in events}
    for finding in findings:
        for reference in finding.evidence_refs:
            if by_ref.get(reference.event_ref) != reference.source_hash:
                return False
    return True


def analyze_events(events):
    findings = []
    for record in inspect_powershell(events):
        if record.encoded.success:
            text = record.encoded.text or ""
            findings.append(Finding(
                f"encoded-powershell-{record.event.raw_index}", "encoded_powershell", "medium",
                "Encoded PowerShell command observed",
                "A PowerShell process included an encoded command that was decoded as text.",
                "high", (_ref(record.event),), "powershell-encoded-command", "v1",
                (("decoded_sha256", record.encoded.text_hash or ""),),
                ("The decoded command was not executed.",),
            ))
    for match in correlate_network(events):
        findings.append(Finding(
            f"network-correlation-{match.process_event.raw_index}-{match.network_event.raw_index}",
            "process_network_correlation", "low",
            "Observed network event for process",
            "A network event was observed for the same process identifier within the configured time window.",
            "medium", (_ref(match.process_event), _ref(match.network_event)),
            "process-network-correlation", "v1", (("process_id", str(match.process_id)),),
            ("Temporal/process-identifier correlation does not establish causation.",),
        ))
    return tuple(findings)
