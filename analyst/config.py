import os
import math
from pathlib import Path
from dataclasses import dataclass, field
from memory_config import MemorySettings, load_memory_settings
from router_policy import (
    RouterPolicy,
    load_router_policy,
    RouterSelectionPolicy,
    load_selection_policy,
)
from router_query_plan import QueryPlanSettings, load_query_plan_settings

# This is where we define the configuration settings for the application, including required environment variables and their validation.


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()

    if not value:
        raise ValueError(f"Missing required environment variable: {name}")

    return value


def integer_env(name: str) -> int:
    value = required_env(name)

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc


def float_env(name: str) -> float:
    value = required_env(name)

    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {value!r}") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


@dataclass(frozen=True)  # frozen prevents accidental mutation of settings
class Settings:
    llm_url: str
    model_name: str
    context_size: int
    max_output_tokens: int
    context_safety_margin: int
    skills_dir: Path
    minilm_model_path: Path
    router_top_k: int
    router_semantic_weight: float
    router_name_boost: float
    llm_timeout_seconds: int
    token_count_timeout_seconds: int
    max_pending_requests: int
    pending_request_ttl_seconds: int
    memory: MemorySettings = field(default_factory=load_memory_settings)
    router_policy: RouterPolicy = field(default_factory=load_router_policy)
    router_selection: RouterSelectionPolicy = field(
        default_factory=load_selection_policy
    )
    router_query_plan: QueryPlanSettings = field(
        default_factory=load_query_plan_settings
    )

    @property
    def router_lexical_weight(self) -> float:
        return 1.0 - self.router_semantic_weight

    @property
    def input_token_budget(self) -> int:
        return self.context_size - self.max_output_tokens - self.context_safety_margin

    def __post_init__(self):
        if self.context_size <= 0:
            raise ValueError("LLM_CONTEXT_SIZE must be positive")

        if self.max_output_tokens <= 0:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be positive")

        if self.context_safety_margin < 0:
            raise ValueError("LLM_CONTEXT_SAFETY_MARGIN cannot be negative")

        if self.input_token_budget <= 0:
            raise ValueError(
                "Context size must exceed the output reserve and safety margin"
            )

        if not (self.skills_dir / "index.json").is_file():
            raise ValueError(
                f"Skills index not found: {self.skills_dir / 'index.json'}"
            )

        if not self.minilm_model_path.is_dir():
            raise ValueError(
                f"Embedding model directory not found: {self.minilm_model_path}"
            )

        if not 0 <= self.router_semantic_weight <= 1:
            raise ValueError("ROUTER_SEMANTIC_WEIGHT must be between 0 and 1")

        if self.router_name_boost < 0:
            raise ValueError("ROUTER_NAME_BOOST cannot be negative")

        positive_settings = {
            "ROUTER_TOP_K": self.router_top_k,
            "LLM_TIMEOUT_SECONDS": self.llm_timeout_seconds,
            "TOKEN_COUNT_TIMEOUT_SECONDS": self.token_count_timeout_seconds,
            "MAX_PENDING_REQUESTS": self.max_pending_requests,
            "PENDING_REQUEST_TTL_SECONDS": self.pending_request_ttl_seconds,
        }

        for name, value in positive_settings.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


settings = Settings(
    llm_url=required_env("LLM_BASE_URL"),
    model_name=required_env("MODEL_NAME"),
    context_size=integer_env("LLM_CONTEXT_SIZE"),
    max_output_tokens=integer_env("LLM_MAX_OUTPUT_TOKENS"),
    context_safety_margin=integer_env("LLM_CONTEXT_SAFETY_MARGIN"),
    skills_dir=Path(required_env("SKILLS_DIR")).resolve(),
    minilm_model_path=Path(required_env("MINILM_MODEL_PATH")).resolve(),
    router_top_k=integer_env("ROUTER_TOP_K"),
    router_semantic_weight=float_env("ROUTER_SEMANTIC_WEIGHT"),
    router_name_boost=float_env("ROUTER_NAME_BOOST"),
    llm_timeout_seconds=integer_env("LLM_TIMEOUT_SECONDS"),
    token_count_timeout_seconds=integer_env("TOKEN_COUNT_TIMEOUT_SECONDS"),
    max_pending_requests=integer_env("MAX_PENDING_REQUESTS"),
    pending_request_ttl_seconds=integer_env("PENDING_REQUEST_TTL_SECONDS"),
)
