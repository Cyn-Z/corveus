"""Local, model-independent applicability checks for skill retrieval."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from orchestrator import SkillScore


# These can contribute to ranking but cannot establish specialist applicability
# on their own. A sufficiently strong semantic-only match is still allowed.
GENERIC_ACTIVATION_TERMS = frozenset(
    {
        "this",
        "that",
        "test",
        "tests",
        "testing",
        "example",
        "examples",
        "help",
        "hello",
        "thanks",
        "network",
        "security",
        "cybersecurity",
        "analyze",
        "analysis",
        "investigate",
        "investigation",
        "system",
        "systems",
        "server",
        "servers",
        "configure",
        "configuration",
        "environment",
        "project",
    }
)


@dataclass(frozen=True)
class RouterPolicy:
    enabled: bool = True
    min_semantic_score: float = 0.36
    min_combined_score: float = 0.45
    semantic_only_score: float = 0.65

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("ROUTER_ACTIVATION_ENABLED must be a boolean")
        for name, value in (
            ("ROUTER_MIN_SEMANTIC_SCORE", self.min_semantic_score),
            ("ROUTER_MIN_COMBINED_SCORE", self.min_combined_score),
            ("ROUTER_SEMANTIC_ONLY_SCORE", self.semantic_only_score),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if self.semantic_only_score < self.min_semantic_score:
            raise ValueError(
                "ROUTER_SEMANTIC_ONLY_SCORE cannot be below ROUTER_MIN_SEMANTIC_SCORE"
            )

    def bypass_reason(self, prompt: str) -> str | None:
        if not self.enabled:
            return None
        # Full-message matches only. An appended task, log, or second question
        # invalidates these patterns instead of being silently discarded.
        text = re.sub(r"\s+", " ", prompt.casefold()).strip()
        text = re.sub(r"[.!?]+$", "", text).strip()
        if re.fullmatch(
            r"(?:hello|hi|hey|good morning|good afternoon|good evening|"
            r"thanks|thank you)(?: for (?:your |the )?help)?|"
            r"(?:this is (?:just )?a test(?: message)?|just testing|"
            r"are you there|the test worked,? (?:thank you|thanks)|"
            r"continue(?: with that)?|go on|ok|okay|yes|no)",
            text,
        ):
            return "conversational_message"

        # Only explicit definition frames bypass. "What is/are ..." can
        # describe a fault ("What is blocking logins?") and must be retrieved.
        # Restrict non-operational language tasks to one complete sentence;
        # appended tasks, logs, punctuation, or operational verbs fall through.
        operational = re.search(
            r"\b(?:analy[sz]e|investigate|audit|detect|triage|hunt|configure|"
            r"troubleshoot|execute|run|check|inspect|review|then|also|and)\b",
            text,
        )
        if (
            operational is None
            and "\n" not in prompt
            and "\r" not in prompt
            and re.fullmatch(
                r"(?:translate (?:the words|the phrase) [\w -]+ into [\w -]+|"
                r"(?:suggest|write) a title for (?:my |a |the )?(?:poem|story|novel) about [\w -]+)",
                text,
            )
        ):
            return "non_operational_language_task"

        # Explicit elementary-definition frames, not arbitrary questions.
        definition = re.fullmatch(
            r"(?:what does (?P<meaning>[\w+#./ -]+) mean|"
            r"what does (?P<acronym>[\w+#./ -]+) stand for|"
            r"define (?P<definition>[\w+#./ -]+))"
            r"(?:[?.] ?answer in one sentence)?",
            text,
        )
        if definition is not None:
            topic = next(value for value in definition.groups() if value is not None)
            words = re.findall(r"\w+", topic)
            # Exclude common troubleshooting/procedural frames. This is a
            # conservative shortcut, not a general intent classifier.
            task_words = {
                "causing",
                "wrong",
                "happening",
                "best",
                "steps",
                "procedure",
                "workflow",
                "investigate",
                "detect",
                "configure",
                "recommended",
                "difference",
                "differences",
                "and",
                "then",
                "using",
                "with",
            }
            if 1 <= len(words) <= 4 and not task_words.intersection(words):
                return "simple_definition"
        return None

    @staticmethod
    def informative_terms(score: SkillScore) -> tuple[str, ...]:
        frequencies = dict(score.matched_term_document_frequencies)
        return tuple(
            term
            for term in score.matched_terms
            if (len(term) >= 3 or term in {"ad"})
            and term not in GENERIC_ACTIVATION_TERMS
            # Common catalog vocabulary is weak evidence even when not in
            # the generic-word list. Unknown corpus size preserves callers
            # constructing historical SkillScore records without this field.
            and (
                not score.catalog_size
                or frequencies.get(term, score.catalog_size)
                <= score.catalog_size * 0.20
            )
        )

    def candidate_reason(self, score: SkillScore) -> str:
        if score.combined_score <= 0:
            return "nonpositive_score"
        if not self.enabled:
            return "legacy_positive_score"
        if score.semantic_score >= self.semantic_only_score:
            return "strong_semantic_match"
        if score.semantic_score < self.min_semantic_score:
            return "weak_semantic_match"
        if not self.informative_terms(score):
            return "no_informative_keyword_match"
        if score.combined_score < self.min_combined_score:
            return "weak_combined_match"
        return "hybrid_match"

    def accepts(self, score: SkillScore) -> bool:
        return self.candidate_reason(score) in {
            "legacy_positive_score",
            "strong_semantic_match",
            "hybrid_match",
        }


def load_router_policy(environment: Mapping[str, str] | None = None) -> RouterPolicy:
    values = os.environ if environment is None else environment
    enabled = values.get("ROUTER_ACTIVATION_ENABLED", "true").strip().casefold()
    if enabled not in {"true", "false"}:
        raise ValueError("ROUTER_ACTIVATION_ENABLED must be 'true' or 'false'")
    arguments = {}
    for field, name, default in (
        ("min_semantic_score", "ROUTER_MIN_SEMANTIC_SCORE", "0.36"),
        ("min_combined_score", "ROUTER_MIN_COMBINED_SCORE", "0.45"),
        ("semantic_only_score", "ROUTER_SEMANTIC_ONLY_SCORE", "0.65"),
    ):
        try:
            arguments[field] = float(values.get(name, default))
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
    return RouterPolicy(enabled=enabled == "true", **arguments)


@dataclass(frozen=True)
class RouterSelectionPolicy:
    """Heuristic selection controls, independent of activation and token budgets."""

    enabled: bool = True
    redundancy_similarity: float = 0.70
    relative_score_floor: float = 0.80
    novelty_weight: float = 0.25

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("ROUTER_SELECTION_ENABLED must be a boolean")
        for name in ("redundancy_similarity", "relative_score_floor", "novelty_weight"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Router selection {name} must be finite and between 0 and 1"
                )


def load_selection_policy(
    environment: Mapping[str, str] | None = None,
) -> RouterSelectionPolicy:
    values = os.environ if environment is None else environment
    enabled = values.get("ROUTER_SELECTION_ENABLED", "true").strip().casefold()
    if enabled not in {"true", "false"}:
        raise ValueError("ROUTER_SELECTION_ENABLED must be 'true' or 'false'")
    arguments = {}
    for field, key, default in (
        ("redundancy_similarity", "ROUTER_REDUNDANCY_SIMILARITY", "0.70"),
        ("relative_score_floor", "ROUTER_RELATIVE_SCORE_FLOOR", "0.80"),
        ("novelty_weight", "ROUTER_NOVELTY_WEIGHT", "0.25"),
    ):
        try:
            arguments[field] = float(values.get(key, default))
        except ValueError as exc:
            raise ValueError(f"{key} must be a number") from exc
    return RouterSelectionPolicy(enabled=enabled == "true", **arguments)
