"""Normalize trusted offline Sysmon-shaped JSON fixtures."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dfir_contracts import NormalizedEvent


def _optional_int(value, name):
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def load_events(source) -> tuple[NormalizedEvent, ...]:
    path = Path(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(data) is not dict or type(data.get("events")) is not list:
        raise ValueError("Fixture must contain an events list.")
    events = []
    for index, item in enumerate(data["events"]):
        if type(item) is not dict:
            raise ValueError("Each event must be an object.")
        events.append(NormalizedEvent(
            event_id=item["event_id"],
            provider=item["provider"],
            timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
            host=item["host"],
            process_id=_optional_int(item.get("process_id"), "process_id"),
            parent_process_id=_optional_int(item.get("parent_process_id"), "parent_process_id"),
            image=item.get("image"), command_line=item.get("command_line"), user=item.get("user"),
            source_ip=item.get("source_ip"), source_port=_optional_int(item.get("source_port"), "source_port"),
            destination_ip=item.get("destination_ip"), destination_port=_optional_int(item.get("destination_port"), "destination_port"),
            protocol=item.get("protocol"), raw_index=index,
        ))
    return tuple(events)
