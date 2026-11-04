"""Typed, bounded deterministic DFIR tools.

This module intentionally exposes no shell, subprocess, PowerShell, or binary
execution capability. Tool inputs are fixture identifiers resolved by trusted
server configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from dfir_artifacts import ArtifactStore
from dfir_contracts import Finding
from dfir_findings import analyze_events, validate_findings
from dfir_sysmon import load_events

TOOL_ID = "sysmon_powershell_offline_v1"
_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dfir"


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    version: str
    description: str
    deterministic: bool
    read_only: bool
    network_required: bool
    executes_processes: bool
    modifies_artifacts: bool


@dataclass(frozen=True)
class ToolRequest:
    tool_id: str
    fixture_id: str | None = None
    artifact_id: str | None = None

    @classmethod
    def from_payload(cls, payload):
        if type(payload) is not dict or set(payload) != {"tool_id", "input"}:
            raise ValueError("Tool request must contain tool_id and input.")
        tool_id = payload["tool_id"]
        input_value = payload["input"]
        if tool_id != TOOL_ID or type(input_value) is not dict:
            raise ValueError("Unsupported tool request.")
        if set(input_value) == {"fixture_id"}:
            fixture_id = input_value["fixture_id"]
            if fixture_id != "sysmon_powershell":
                raise ValueError("Unsupported fixture identifier.")
            return cls(tool_id, fixture_id=fixture_id)
        if set(input_value) == {"artifact_id"}:
            artifact_id = input_value["artifact_id"]
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("Invalid artifact identifier.")
            return cls(tool_id, artifact_id=artifact_id)
        raise ValueError("Unsupported tool request.")
    def to_json(self):
        input_value = {"fixture_id": self.fixture_id} if self.fixture_id is not None else {"artifact_id": self.artifact_id}
        return json.dumps({"tool_id": self.tool_id, "input": input_value}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload):
        return cls.from_payload(json.loads(payload))


@dataclass(frozen=True)
class ToolResult:
    tool_id: str
    tool_version: str
    findings: tuple[Finding, ...]
    event_count: int
    provenance_valid: bool
    warnings: tuple[str, ...] = ()

    @staticmethod
    def _finding_payload(item):
        return {
            "finding_id": item.finding_id,
            "category": item.category,
            "severity": item.severity,
            "title": item.title,
            "summary": item.summary,
            "confidence": item.confidence,
            "detector_id": item.detector_id,
            "detector_version": item.detector_version,
            "evidence_refs": [ref.__dict__ for ref in item.evidence_refs],
            "observed_values": list(item.observed_values),
            "limitations": list(item.limitations),
        }

    def as_payload(self):
        return {
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "event_count": self.event_count,
            "provenance_valid": self.provenance_valid,
            "warnings": list(self.warnings),
            "findings": [self._finding_payload(item) for item in self.findings],
        }

    @classmethod
    def from_json(cls, payload):
        from dfir_contracts import EvidenceRef
        data = json.loads(payload)
        if type(data) is not dict or set(data) != {"tool_id", "tool_version", "event_count", "provenance_valid", "warnings", "findings"}:
            raise ValueError("Invalid tool result.")
        findings = []
        for item in data["findings"]:
            if type(item) is not dict:
                raise ValueError("Invalid finding.")
            findings.append(Finding(
                item["finding_id"], item["category"], item["severity"], item["title"],
                item["summary"], item["confidence"],
                tuple(EvidenceRef(**ref) for ref in item["evidence_refs"]),
                item["detector_id"], item["detector_version"],
                tuple(tuple(value) for value in item["observed_values"]),
                tuple(item["limitations"]),
            ))
        result = cls(
            data["tool_id"], data["tool_version"], tuple(findings),
            data["event_count"], data["provenance_valid"], tuple(data["warnings"]),
        )
        if type(result.event_count) is not int or result.event_count < 0 or type(result.provenance_valid) is not bool:
            raise ValueError("Invalid tool result metadata.")
        return result

    def to_json(self):
        return json.dumps({
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "event_count": self.event_count,
            "provenance_valid": self.provenance_valid,
            "warnings": list(self.warnings),
            "findings": [self._finding_payload(item) for item in self.findings],
        }, sort_keys=True, separators=(",", ":"))


class ToolRegistry:
    def __init__(self, artifacts: ArtifactStore | None = None):
        if artifacts is not None and not isinstance(artifacts, ArtifactStore):
            raise ValueError("artifacts must be an ArtifactStore or None.")
        self.artifacts = artifacts
        self._specs = {
            TOOL_ID: ToolSpec(
                TOOL_ID, "v1", "Analyze a trusted offline Sysmon JSON fixture.",
                True, True, False, False, False,
            )
        }

    def get(self, tool_id):
        try:
            return self._specs[tool_id]
        except (KeyError, TypeError):
            raise ValueError("Unknown DFIR tool.") from None

    def validate(self, request: ToolRequest):
        if not isinstance(request, ToolRequest):
            raise ValueError("request must be a ToolRequest.")
        self.get(request.tool_id)
        return request

    def execute(self, request: ToolRequest):
        self.validate(request)
        if request.artifact_id is not None:
            if self.artifacts is None:
                raise ValueError("No trusted artifact registry is configured.")
            fixture = self.artifacts.open(request.artifact_id)
        else:
            fixture = _FIXTURE_ROOT / f"{request.fixture_id}.json"
            if not fixture.is_file() or fixture.resolve().parent != _FIXTURE_ROOT.resolve():
                raise ValueError("Fixture is unavailable.")
        events = load_events(fixture)
        findings = analyze_events(events)
        valid = validate_findings(findings, events)
        if not valid:
            raise ValueError("Tool produced invalid evidence references.")
        return ToolResult(TOOL_ID, self.get(TOOL_ID).version, findings, len(events), valid)
