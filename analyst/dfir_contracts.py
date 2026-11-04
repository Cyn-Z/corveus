"""Validated immutable contracts for offline DFIR analysis."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import re

_HASH = re.compile(r"[0-9a-f]{64}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or not value.isprintable():
        raise ValueError(f"{name} must be printable nonempty text.")
    return value


def _hash(value, name):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest.")
    return value


def _timestamp(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware.")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    size_bytes: int
    media_type: str
    original_name: str
    storage_path: str

    def __post_init__(self):
        _text(self.artifact_id, "artifact_id")
        _hash(self.sha256, "sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer.")
        for name in ("media_type", "original_name", "storage_path"):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class NormalizedEvent:
    event_id: int
    provider: str
    timestamp: datetime
    host: str
    process_id: int | None = None
    parent_process_id: int | None = None
    image: str | None = None
    command_line: str | None = None
    user: str | None = None
    source_ip: str | None = None
    source_port: int | None = None
    destination_ip: str | None = None
    destination_port: int | None = None
    protocol: str | None = None
    raw_index: int = 0

    def __post_init__(self):
        if type(self.event_id) is not int or self.event_id < 0:
            raise ValueError("event_id must be a nonnegative integer.")
        _text(self.provider, "provider")
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        _text(self.host, "host")
        for name in ("process_id", "parent_process_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} is invalid.")
        for name in ("source_port", "destination_port"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 0 <= value <= 65535):
                raise ValueError(f"{name} is invalid.")
        if self.raw_index < 0:
            raise ValueError("raw_index must be nonnegative.")


@dataclass(frozen=True)
class DecodedCommand:
    encoded_value: str
    text: str | None
    encoding: str | None
    success: bool
    text_hash: str | None = None


@dataclass(frozen=True)
class EvidenceRef:
    event_ref: str
    source_hash: str
    start_offset: int | None = None
    end_offset: int | None = None

    def __post_init__(self):
        _text(self.event_ref, "event_ref")
        _hash(self.source_hash, "source_hash")
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Evidence offsets must be supplied together.")
        if self.start_offset is not None and (type(self.start_offset) is not int or type(self.end_offset) is not int or self.start_offset < 0 or self.end_offset <= self.start_offset):
            raise ValueError("Evidence offsets are invalid.")


@dataclass(frozen=True)
class Finding:
    finding_id: str
    category: str
    severity: str
    title: str
    summary: str
    confidence: str
    evidence_refs: tuple[EvidenceRef, ...]
    detector_id: str
    detector_version: str
    observed_values: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("finding_id", "category", "severity", "title", "summary", "confidence", "detector_id", "detector_version"):
            _text(getattr(self, name), name)
        if type(self.evidence_refs) is not tuple or not self.evidence_refs or not all(isinstance(item, EvidenceRef) for item in self.evidence_refs):
            raise ValueError("Findings require immutable evidence references.")
        if not all(isinstance(item, tuple) and len(item) == 2 for item in self.observed_values):
            raise ValueError("observed_values must contain key/value pairs.")
