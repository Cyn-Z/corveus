"""Request-local generation shared by legacy requests and future workers.

The supplied call_llm callback must be synchronous and non-streaming.
Normal return must mean the complete upstream response has been received.

This module does not load models, route skills, save conversations,
grant inference permits, or release inference capacity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

from token_usage import MAX_SQLITE_INTEGER, TokenUsage, token_count


class InferenceCallUncertainError(RuntimeError):
    """The model call failed without proof that upstream execution stopped."""

    inference_stopped = False

    def __init__(self) -> None:
        super().__init__(
            "The inference call failed; upstream execution may still be active."
        )


class InvalidInferenceResponseError(RuntimeError):
    """A completed callback returned an unusable response.

    This distinction relies on the callback's non-streaming contract:
    normal return means the upstream response has completed.
    """

    inference_stopped = True

    def __init__(self) -> None:
        super().__init__(
            "The completed inference response could not be processed."
        )


@dataclass(frozen=True)
class GenerationOutput:
    content: str = field(repr=False)
    usage: TokenUsage

    def as_legacy_response(self) -> dict:
        """Return a fresh dictionary matching the existing web response."""
        return {
            "selection_required": False,
            "content": self.content,
            "stats": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "tokens_used": self.usage.total_tokens,
                "ctx_size": self.usage.context_size,
                "model_name": self.usage.model_name,
            },
        }


def generate_answer(
    messages: list[dict],
    *,
    count_input_tokens: Callable,
    call_llm: Callable,
    model_name: str,
    context_size: int,
    input_budget: int,
) -> GenerationOutput:
    """Verify the current budget, generate once, and normalize usage.

    Errors before call_llm dispatch propagate without calling the model.
    Errors raised by call_llm are classified as uncertain; callers must
    retain any inference permit until upstream shutdown is confirmed.

    Future workers must obtain an inference permit before invoking this
    function. This function never acquires or releases one itself.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list.")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("Each message must be a dictionary.")
    if not callable(count_input_tokens) or not callable(call_llm):
        raise ValueError("Token counter and model client must be callable.")

    # Validate server-selected model settings before dispatch.
    TokenUsage(model_name=model_name, context_size=context_size)
    if (
        type(input_budget) is not int
        or not 1 <= input_budget <= context_size
    ):
        raise ValueError(
            "input_budget must be a positive integer within context_size."
        )

    # Keep caller state and callback mutations separate.
    snapshot = deepcopy(messages)
    verified_tokens = count_input_tokens(deepcopy(snapshot))

    if token_count(verified_tokens) is None:
        raise ValueError("The token counter returned an invalid count.")
    if verified_tokens > input_budget:
        raise ValueError(
            f"Input requires {verified_tokens} tokens; "
            f"budget is {input_budget}."
        )

    try:
        response_json = call_llm(deepcopy(snapshot))
    except Exception:
        # Do not expose exception text containing prompts or internal URLs.
        # A timeout does not prove that inference stopped.
        raise InferenceCallUncertainError() from None

    try:
        if not isinstance(response_json, dict):
            raise ValueError("Expected a response object.")

        choices = response_json["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("Expected at least one response choice.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Expected a response choice object.")

        message = choice["message"]
        if not isinstance(message, dict):
            raise ValueError("Expected a response message object.")

        # Preserve the existing behavior for missing or null content.
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Response content must be a string.")

        reported_usage = response_json.get("usage")
        if not isinstance(reported_usage, dict):
            reported_usage = {}

        reported_input = token_count(reported_usage.get("prompt_tokens"))
        input_tokens = (
            verified_tokens if reported_input is None else reported_input
        )
        output_tokens = token_count(
            reported_usage.get("completion_tokens")
        )

        if (
            output_tokens is not None
            and input_tokens + output_tokens > MAX_SQLITE_INTEGER
        ):
            output_tokens = None

        total_tokens = token_count(reported_usage.get("total_tokens"))
        if output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        elif total_tokens is not None and total_tokens < input_tokens:
            total_tokens = None

        usage = TokenUsage(
            model_name=model_name,
            context_size=context_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return GenerationOutput(content=content, usage=usage)

    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        raise InvalidInferenceResponseError() from None