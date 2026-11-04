"""Deterministic process ancestry and network correlation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from dfir_contracts import NormalizedEvent


@dataclass(frozen=True)
class AncestryLink:
    process_id: int
    parent_process_id: int | None
    parent_observed: bool
    evidence: tuple[NormalizedEvent, ...]


def reconstruct_ancestry(events):
    processes = {event.process_id: event for event in events if event.event_id == 1 and event.process_id is not None}
    result = []
    for event in sorted(processes.values(), key=lambda item: (item.timestamp, item.raw_index)):
        parent = processes.get(event.parent_process_id)
        result.append(AncestryLink(event.process_id, event.parent_process_id, parent is not None, (event,) + ((parent,) if parent else ())))
    return tuple(result)


@dataclass(frozen=True)
class NetworkCorrelation:
    process_id: int
    process_event: NormalizedEvent
    network_event: NormalizedEvent


def correlate_network(events, *, window_seconds=30):
    if type(window_seconds) not in (int, float) or window_seconds < 0:
        raise ValueError("window_seconds must be nonnegative.")
    processes = [item for item in events if item.event_id == 1 and item.process_id is not None]
    network = [item for item in events if item.event_id == 3 and item.process_id is not None]
    matches = []
    for process in processes:
        for item in network:
            if item.process_id == process.process_id and abs(item.timestamp - process.timestamp) <= timedelta(seconds=window_seconds):
                matches.append(NetworkCorrelation(process.process_id, process, item))
    return tuple(matches)
