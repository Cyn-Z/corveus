"""Model-independent snapshots of completed-request token usage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

MAX_SQLITE_INTEGER = 2**63 - 1


def token_count(value) -> int | None:
    """Unknown or invalid backend counts remain unknown, never zero."""
    if type(value) is int and 0 <= value <= MAX_SQLITE_INTEGER:
        return value
    return None


@dataclass(frozen=True)
class TokenUsage:
    model_name: str
    context_size: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if token_count(self.context_size) is None or self.context_size == 0:
            raise ValueError("context_size must be a positive SQLite integer")
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field)
            if value is not None and token_count(value) is None:
                raise ValueError(
                    f"{field} must be a non-negative SQLite integer or None"
                )
        if self.input_tokens is not None and self.output_tokens is not None:
            total = self.input_tokens + self.output_tokens
            if total > MAX_SQLITE_INTEGER:
                raise ValueError("Combined token count exceeds SQLite integer capacity")
            if self.total_tokens is None:
                object.__setattr__(self, "total_tokens", total)
            elif self.total_tokens != total:
                raise ValueError(
                    "total_tokens must equal input_tokens plus output_tokens"
                )
        if self.total_tokens is not None:
            for part in (self.input_tokens, self.output_tokens):
                if part is not None and part > self.total_tokens:
                    raise ValueError("Token components cannot exceed total_tokens")


def summarize_usage(usages: Sequence[TokenUsage | None]) -> dict:
    """Aggregate known counts and retain explicit measurement coverage.

    The sequence contains one entry per completed exchange, oldest first.
    None represents a historical exchange without any recorded snapshot.
    """
    result = {
        "exchange_count": len(usages),
        "recorded_exchanges": sum(usage is not None for usage in usages),
        "latest_usage": (
            asdict(usages[-1]) if usages and usages[-1] is not None else None
        ),
    }
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [
            getattr(usage, field)
            for usage in usages
            if usage is not None and getattr(usage, field) is not None
        ]
        result[field] = sum(values) if values or not usages else None
        result[field + "_measured_exchanges"] = len(values)
    result["is_partial"] = any(
        result[field + "_measured_exchanges"] < len(usages)
        for field in ("input_tokens", "output_tokens", "total_tokens")
    )
    return result
