"""Token-aware, bounded retrieval planning.

This module does not load models, embed text, execute skills, or modify the
original inference prompt. Router integration is a separate implementation
batch.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, replace
from typing import Any
from typing import Mapping


@dataclass(frozen=True)
class QueryPlanSettings:
    enabled: bool = True
    chunk_tokens: int = 192
    overlap_tokens: int = 32
    max_views: int = 8
    max_directive_views: int = 3
    max_prompt_characters: int = 200_000

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")

        ranges = (
            ("chunk_tokens", self.chunk_tokens, 1, 4096),
            ("overlap_tokens", self.overlap_tokens, 0, 4095),
            ("max_views", self.max_views, 2, 32),
            ("max_directive_views", self.max_directive_views, 0, 8),
            (
                "max_prompt_characters",
                self.max_prompt_characters,
                1,
                2_000_000,
            ),
        )
        for name, value, minimum, maximum in ranges:
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be an integer between " f"{minimum} and {maximum}"
                )

        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be below chunk_tokens")


@dataclass(frozen=True)
class QueryView:
    text: str
    sources: tuple[str, ...]
    token_ranges: tuple[tuple[int, int], ...]
    source_lines: tuple[int, ...]
    content_tokens: int
    encoded_tokens: int


@dataclass(frozen=True)
class QueryPlan:
    mode: str
    views: tuple[QueryView, ...]
    prompt_tokens: int
    effective_token_limit: int
    payload_token_limit: int
    original_encoded_tokens: int
    sampled_window_tokens: int
    directive_candidates: int
    directives_selected: int

    @property
    def would_truncate(self) -> bool:
        return self.original_encoded_tokens > self.effective_token_limit

    @property
    def window_coverage_complete(self) -> bool:
        return self.sampled_window_tokens == self.prompt_tokens

    def diagnostic(self) -> dict[str, Any]:
        """Return numerical provenance without prompt or decoded view text."""
        omitted = self.prompt_tokens - self.sampled_window_tokens
        return {
            "mode": self.mode,
            "view_count": len(self.views),
            "prompt_tokens": self.prompt_tokens,
            "original_encoded_tokens": self.original_encoded_tokens,
            "effective_token_limit": self.effective_token_limit,
            "payload_token_limit": self.payload_token_limit,
            "would_truncate": self.would_truncate,
            "sampled_window_tokens": self.sampled_window_tokens,
            "omitted_window_tokens": omitted,
            "window_coverage_complete": self.window_coverage_complete,
            "window_coverage_ratio": (
                self.sampled_window_tokens / self.prompt_tokens
                if self.prompt_tokens
                else 1.0
            ),
            "directive_candidates": self.directive_candidates,
            "directives_selected": self.directives_selected,
            "views": [
                {
                    "sources": list(view.sources),
                    "token_ranges": [
                        list(token_range) for token_range in view.token_ranges
                    ],
                    "source_lines": list(view.source_lines),
                    "content_tokens": view.content_tokens,
                    "encoded_tokens": view.encoded_tokens,
                }
                for view in self.views
            ],
        }


_DIRECTIVE = re.compile(
    r"^(?:please\s+)?(?:"
    r"investigate|analy[sz]e|audit|check|inspect|examine|triage|"
    r"hunt|detect|write|configure|troubleshoot|"
    r"help\s+me|what|why|how"
    r")\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _sample_indices(count: int, limit: int) -> tuple[int, ...]:
    """Choose deterministic, evenly spaced positions including both ends."""
    if count <= 0 or limit <= 0:
        return ()
    if count <= limit:
        return tuple(range(count))
    if limit == 1:
        return (count // 2,)

    return tuple(index * (count - 1) // (limit - 1) for index in range(limit))


def _covered_tokens(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
        return 0

    ordered = sorted(ranges)
    start, end = ordered[0]
    covered = 0

    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            covered += end - start
            start, end = next_start, next_end

    return covered + end - start


def _directive_lines(prompt: str) -> list[tuple[int, str]]:
    """Find plain task lines; code fences and quoted lines are evidence."""
    results: list[tuple[int, str]] = []
    seen: set[str] = set()
    fence_character: str | None = None
    fence_length = 0

    for number, raw_line in enumerate(prompt.splitlines(), start=1):
        fence = _FENCE.match(raw_line)
        if fence:
            marker = fence.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not raw_line[fence.end() :].strip()
            ):
                fence_character = None
                fence_length = 0
            continue

        if fence_character is not None:
            continue

        line = raw_line.strip()
        if not line or line.startswith(">"):
            continue
        if not _DIRECTIVE.match(line):
            continue

        # Exact text deduplication preserves case-sensitive model behavior.
        if line in seen:
            continue
        seen.add(line)
        results.append((number, line))

    return results


class QueryPlanner:
    """Plan retrieval inputs against a supplied local encoder/tokenizer.

    The tokenizer must provide encode(), decode(), and
    num_special_tokens_to_add(). No model-specific imports are required.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_max_seq_length: int,
        settings: QueryPlanSettings | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.settings = settings or QueryPlanSettings()

        if type(model_max_seq_length) is not int or model_max_seq_length <= 0:
            raise ValueError("model_max_seq_length must be positive")

        limits = [model_max_seq_length]
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if type(tokenizer_limit) is int and tokenizer_limit > 0:
            limits.append(tokenizer_limit)

        self.effective_token_limit = min(limits)

        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        if type(special_tokens) is not int or special_tokens < 0:
            raise ValueError("Tokenizer returned an invalid special-token count")

        available = self.effective_token_limit - special_tokens
        if available <= 0:
            raise ValueError("Encoder limit leaves no room for content tokens")

        self.payload_token_limit = min(
            self.settings.chunk_tokens,
            available,
        )
        self.overlap_tokens = min(
            self.settings.overlap_tokens,
            self.payload_token_limit - 1,
        )

    def _encode(self, text: str, *, special: bool) -> list[int]:
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=special,
                truncation=False,
            )
        )

    def _view(
        self,
        ids: list[int],
        *,
        source: str,
        token_start: int | None = None,
        source_line: int | None = None,
    ) -> QueryView | None:
        """Decode and verify fit instead of assuming perfect round trips."""
        prefix = ids[: self.payload_token_limit]

        while prefix:
            text = self.tokenizer.decode(
                prefix,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(text, str):
                raise ValueError("Tokenizer decode() must return text")

            content_count = len(self._encode(text, special=False))
            encoded_count = len(self._encode(text, special=True))

            if (
                text.strip()
                and content_count <= self.payload_token_limit
                and encoded_count <= self.effective_token_limit
            ):
                ranges = (
                    ((token_start, token_start + len(prefix)),)
                    if token_start is not None
                    else ()
                )
                return QueryView(
                    text=text,
                    sources=(source,),
                    token_ranges=ranges,
                    source_lines=((source_line,) if source_line is not None else ()),
                    content_tokens=content_count,
                    encoded_tokens=encoded_count,
                )

            prefix = prefix[:-1]

        return None

    def plan(self, prompt: str) -> QueryPlan:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if len(prompt) > self.settings.max_prompt_characters:
            raise ValueError(
                "Prompt exceeds the configured retrieval-planning " "character limit"
            )

        ids = self._encode(prompt, special=False)
        original_encoded_tokens = len(self._encode(prompt, special=True))
        if not ids:
            raise ValueError("Tokenizer produced no content tokens")

        if not self.settings.enabled or (
            original_encoded_tokens <= self.effective_token_limit
        ):
            fits = original_encoded_tokens <= self.effective_token_limit
            view = QueryView(
                text=prompt,
                sources=("original",),
                token_ranges=((0, len(ids)),),
                source_lines=(),
                content_tokens=len(ids),
                encoded_tokens=original_encoded_tokens,
            )
            return QueryPlan(
                mode="short" if self.settings.enabled else "disabled",
                views=(view,),
                prompt_tokens=len(ids),
                effective_token_limit=self.effective_token_limit,
                payload_token_limit=self.payload_token_limit,
                original_encoded_tokens=original_encoded_tokens,
                sampled_window_tokens=len(ids) if fits else 0,
                directive_candidates=0,
                directives_selected=0,
            )

        views: list[QueryView] = []
        positions: dict[str, int] = {}

        def add(view: QueryView | None) -> bool:
            if view is None:
                return False

            previous = positions.get(view.text)
            if previous is not None:
                existing = views[previous]
                views[previous] = replace(
                    existing,
                    sources=tuple(dict.fromkeys(existing.sources + view.sources)),
                    token_ranges=tuple(
                        dict.fromkeys(existing.token_ranges + view.token_ranges)
                    ),
                    source_lines=tuple(
                        dict.fromkeys(existing.source_lines + view.source_lines)
                    ),
                )
                return True

            if len(views) >= self.settings.max_views:
                return False

            positions[view.text] = len(views)
            views.append(view)
            return True

        directives = _directive_lines(prompt)
        directive_budget = min(
            self.settings.max_directive_views,
            self.settings.max_views - 2,
        )
        directives_selected = 0

        for index in _sample_indices(len(directives), directive_budget):
            line_number, text = directives[index]
            view = self._view(
                self._encode(text, special=False),
                source="directive",
                source_line=line_number,
            )
            if add(view):
                directives_selected += 1

        # Reserve evidence sampling even when many directive lines exist.
        window_budget = self.settings.max_views - len(views)
        payload = self.payload_token_limit
        step = payload - self.overlap_tokens
        final_start = max(0, len(ids) - payload)

        starts = list(range(0, final_start + 1, step))
        if starts[-1] != final_start:
            starts.append(final_start)

        for index in _sample_indices(len(starts), window_budget):
            start = starts[index]
            add(
                self._view(
                    ids[start : start + payload],
                    source="window",
                    token_start=start,
                )
            )

        if not views:
            raise ValueError("Tokenizer produced no usable retrieval views")

        ranges = [token_range for view in views for token_range in view.token_ranges]

        return QueryPlan(
            mode="chunked",
            views=tuple(views),
            prompt_tokens=len(ids),
            effective_token_limit=self.effective_token_limit,
            payload_token_limit=self.payload_token_limit,
            original_encoded_tokens=original_encoded_tokens,
            sampled_window_tokens=_covered_tokens(ranges),
            directive_candidates=len(directives),
            directives_selected=directives_selected,
        )


_QUERY_SETTINGS = (
    ("chunk_tokens", "ROUTER_QUERY_CHUNK_TOKENS"),
    ("overlap_tokens", "ROUTER_QUERY_OVERLAP_TOKENS"),
    ("max_views", "ROUTER_QUERY_MAX_VIEWS"),
    ("max_directive_views", "ROUTER_QUERY_MAX_DIRECTIVE_VIEWS"),
    ("max_prompt_characters", "ROUTER_QUERY_MAX_PROMPT_CHARACTERS"),
)


def load_query_plan_settings(
    environment: Mapping[str, str] | None = None,
) -> QueryPlanSettings:
    values = os.environ if environment is None else environment
    enabled = values.get("ROUTER_LONG_REQUESTS_ENABLED", "true").strip().casefold()

    if enabled not in {"true", "false"}:
        raise ValueError("ROUTER_LONG_REQUESTS_ENABLED must be 'true' or 'false'")

    defaults = QueryPlanSettings()
    arguments = {}

    for field, key in _QUERY_SETTINGS:
        try:
            arguments[field] = int(values.get(key, str(getattr(defaults, field))))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc

    return QueryPlanSettings(
        enabled=enabled == "true",
        **arguments,
    )


def add_query_plan_arguments(parser: Any) -> None:
    defaults = QueryPlanSettings()
    parser.add_argument(
        "--disable-long-requests",
        action="store_true",
        help="Use Batch 4 retrieval without long-request planning",
    )

    for field, _ in _QUERY_SETTINGS:
        parser.add_argument(
            "--query-" + field.replace("_", "-"),
            dest="query_" + field,
            type=int,
            default=getattr(defaults, field),
        )


def query_plan_settings_from_arguments(arguments: Any) -> QueryPlanSettings:
    return QueryPlanSettings(
        enabled=not arguments.disable_long_requests,
        **{field: getattr(arguments, "query_" + field) for field, _ in _QUERY_SETTINGS},
    )
