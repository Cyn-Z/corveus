import os
import json
import re
import time
import uuid
import threading
import urllib.request
from config import settings
from memory_runtime import build_memory_manager
from memory_store import ConversationNotFoundError
from token_usage import MAX_SQLITE_INTEGER, TokenUsage, token_count
from http.server import HTTPServer, BaseHTTPRequestHandler
from orchestrator import SkillRegistry, SkillRouter
from analyst_generation import generate_answer
from analyst_preparation import build_messages as build_prepared_messages

# Configuration
# Optional conversation persistence.
# Initialization errors propagate rather than silently disabling persistence.
MEMORY_MANAGER = build_memory_manager(settings.memory)
# Configuration
PENDING_REQUESTS = {}
PENDING_REQUESTS_LOCK = threading.Lock()

MAX_PENDING_REQUESTS = settings.max_pending_requests
PENDING_REQUEST_TTL = settings.pending_request_ttl_seconds

# Keep your existing LLM container endpoint.
LLM_URL = settings.llm_url
CONTEXT_SIZE = settings.context_size
MAX_OUTPUT_TOKENS = settings.max_output_tokens
CONTEXT_SAFETY_MARGIN = settings.context_safety_margin
INPUT_TOKEN_BUDGET = settings.input_token_budget

if INPUT_TOKEN_BUDGET <= 0:
    raise ValueError("Invalid LLM context budget")

TOKEN_COUNT_URL = (
    LLM_URL.rsplit("/chat/completions", 1)[0] + "/chat/completions/input_tokens"
)

print(
    f"LLM context={CONTEXT_SIZE}, "
    f"output_limit={MAX_OUTPUT_TOKENS}, "
    f"input_budget={INPUT_TOKEN_BUDGET}",
    flush=True,
)

# This must point to the skills directory INSIDE the
# HTTP server container, where index.json is located.
SKILLS_DIR = settings.skills_dir


# Initialize the skill orchestrator once at startup


# Read skill metadata and build the local search index.
# Full SKILL.md instructions are loaded only when needed.
if os.environ.get("SERVICE_PROCESS_ROLE") == "api":
    SKILL_REGISTRY = None
    SKILL_ROUTER = None
else:
    SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)
    SKILL_ROUTER = SkillRouter(
        SKILL_REGISTRY,
        model_path=settings.minilm_model_path,
        semantic_weight=settings.router_semantic_weight,
        name_boost=settings.router_name_boost,
        activation_policy=settings.router_policy,
        selection_policy=settings.router_selection,
        query_plan_settings=settings.router_query_plan,
    )
    print(
        f"Loaded {len(SKILL_REGISTRY.skills)} skill definitions.",
        flush=True,
    )


# Collects metrics from LLama to display in UI
_last_llama_stats = {
    "tokens_used": 0,
    "ctx_size": CONTEXT_SIZE,
    "input_tokens": 0,
    "output_tokens": None,
    "total_tokens": None,
}


def get_llama_stats():
    # Return metrics recorded for the most recent chat request.
    return _last_llama_stats.copy()


def count_input_tokens(messages):
    payload = {
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_COUNT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(
        req, timeout=settings.token_count_timeout_seconds
    ) as response:
        result = json.loads(response.read().decode("utf-8"))

    count = result.get("input_tokens")

    if token_count(count) is None:
        raise ValueError(f"Unexpected token-count response: {result}")

    return count


# LLM client
def call_llm(messages, tools=None):
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        LLM_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


# Skill tool definitions


def make_tool_name(skill_name):
    """
    Convert a repository skill name into a function-tool name.

    Tool names should use a predictable format. Keep the mapping
    consistent so the returned function name can be resolved back
    to the selected Skill object.
    """
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", skill_name)
    name = name.replace("-", "_")

    if not name or not name[0].isalpha():
        name = "skill_" + name

    return name


def _build_messages(user_prompt, skill_sections):
    """Share prompt construction with durable worker preparation."""
    return build_prepared_messages(user_prompt, skill_sections)


def _validate_conversation(conversation_id):
    """Validate an optional ID against the server-configured memory scope."""
    if conversation_id is None:
        return

    if not MEMORY_MANAGER.enabled:
        raise ValueError("Conversation persistence is disabled.")

    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string.")

    MEMORY_MANAGER.get_conversation(conversation_id)


def _record_exchange_result(result, user_prompt, conversation_id, *, usage=None):
    """Preserve a successful answer even if its subsequent save fails."""
    if conversation_id is None:
        return result

    result["conversation_id"] = conversation_id

    try:
        exchange = MEMORY_MANAGER.record_completed_exchange(
            conversation_id,
            user_content=user_prompt,
            assistant_content=result["content"],
            usage=usage,
        )

        if exchange is None:
            raise RuntimeError("Conversation persistence is unavailable.")

    except Exception:
        # Do not log message contents or database exception details.
        # A concurrent deletion also reaches this path.
        result["memory_persistence"] = {
            "saved": False,
            "message": (
                "The response was generated, but the exchange could not "
                "be saved. Resubmitting the prompt may generate another "
                "response."
            ),
        }
    else:
        result["memory_persistence"] = {"saved": True}
        try:
            result["conversation_usage"] = MEMORY_MANAGER.get_conversation_usage(
                conversation_id
            )
        except Exception:
            # A metrics read failure must not misreport a committed exchange
            # as unsaved or prompt the caller to retry generation.
            result["conversation_usage"] = None

    return result


def _selection_response(request_id, state, input_tokens):
    """Return the data needed to render the browser's skill-review panel."""
    result = {
        "selection_required": True,
        "request_id": request_id,
        "input_tokens": input_tokens,
        "input_budget": INPUT_TOKEN_BUDGET,
        "context_size": CONTEXT_SIZE,
        "skills": [
            {
                "name": item["name"],
                "score": item["score"],
                "characters": len(item["content"]),
            }
            for item in state["skills"]
        ],
        "message": (
            "The selected skills exceed the input-token budget. "
            "Choose one or more skills to remove."
        ),
    }
    if state.get("conversation_id") is not None:
        result["conversation_id"] = state["conversation_id"]

    return result


def _prune_expired_requests():
    """Discard abandoned selections."""
    now = time.monotonic()

    expired = [
        key
        for key, value in PENDING_REQUESTS.items()
        if now - value["created_at"] > PENDING_REQUEST_TTL
    ]

    for key in expired:
        PENDING_REQUESTS.pop(key, None)


def start_prompt(user_prompt, *, conversation_id=None):
    # Route a prompt and either request user approval or run the model.
    _validate_conversation(conversation_id)

    candidates = SKILL_ROUTER.search(user_prompt, limit=settings.router_top_k)

    skill_sections = []

    for score, skill in candidates:
        instructions = SKILL_REGISTRY.load_instructions(skill)

        skill_sections.append(
            {
                "name": skill.name,
                "score": float(score),
                "content": (
                    f"## Skill: {skill.name}\n"
                    f"Description: {skill.description}\n\n"
                    f"{instructions}"
                ),
            }
        )

    messages = _build_messages(user_prompt, skill_sections)
    input_tokens = count_input_tokens(messages)

    print(
        f"[Context] Initial input={input_tokens}, "
        f"budget={INPUT_TOKEN_BUDGET}, skills={len(skill_sections)}",
        flush=True,
    )

    if input_tokens <= INPUT_TOKEN_BUDGET:
        return _run_accepted_prompt(
            messages,
            input_tokens,
            conversation_id=conversation_id,
        )

    request_id = uuid.uuid4().hex

    with PENDING_REQUESTS_LOCK:
        _prune_expired_requests()

        if len(PENDING_REQUESTS) >= MAX_PENDING_REQUESTS:
            raise RuntimeError("Too many pending skill selections. Try again shortly.")

        PENDING_REQUESTS[request_id] = {
            "created_at": time.monotonic(),
            "user_prompt": user_prompt,
            "skills": skill_sections,
            "conversation_id": conversation_id,
        }

    return _selection_response(
        request_id,
        PENDING_REQUESTS[request_id],
        input_tokens,
    )


def resolve_selection(request_id, remove_skill_names):
    """Apply user-selected removals, recount, and repeat if necessary."""
    if not isinstance(request_id, str) or not isinstance(remove_skill_names, list):
        raise ValueError("Invalid selection request.")

    if not remove_skill_names or not all(
        isinstance(name, str) for name in remove_skill_names
    ):
        raise ValueError("Select at least one skill to remove.")

    with PENDING_REQUESTS_LOCK:
        _prune_expired_requests()
        state = PENDING_REQUESTS.get(request_id)

        if state is None:
            raise KeyError(
                "Selection expired or was not found. Submit the prompt again."
            )

        conversation_id = state.get("conversation_id")

        try:
            _validate_conversation(conversation_id)
        except ConversationNotFoundError:
            PENDING_REQUESTS.pop(request_id, None)
            raise

        available_names = {item["name"] for item in state["skills"]}

        if len(set(remove_skill_names)) != len(remove_skill_names):
            raise ValueError("Duplicate skill names were supplied.")

        if not set(remove_skill_names).issubset(available_names):
            raise ValueError("Selection contains an unknown skill.")

        remaining = [
            item
            for item in state["skills"]
            if item["name"] not in set(remove_skill_names)
        ]

        # Save the updated selection before recounting.
        state["skills"] = remaining

        messages = _build_messages(state["user_prompt"], remaining)
        input_tokens = count_input_tokens(messages)

        print(
            f"[Context] User removed {remove_skill_names}; "
            f"input={input_tokens}, budget={INPUT_TOKEN_BUDGET}, "
            f"skills={len(remaining)}",
            flush=True,
        )

        if input_tokens > INPUT_TOKEN_BUDGET:
            if not remaining:
                PENDING_REQUESTS.pop(request_id, None)
                raise ValueError(
                    "The prompt exceeds the input budget even without "
                    "skills. Shorten the prompt or adjust the verified "
                    "context configuration."
                )

            return _selection_response(request_id, state, input_tokens)
        # The prompt fits. Consume the pending request so it cannot be
        # submitted repeatedly.
        PENDING_REQUESTS.pop(request_id, None)

    return _run_accepted_prompt(messages, input_tokens, conversation_id=conversation_id)


def _run_accepted_prompt(messages, input_tokens, *, conversation_id=None):
    """Recheck scope and delegate generation before optional persistence."""
    _validate_conversation(conversation_id)

    output = generate_answer(
        messages,
        count_input_tokens=count_input_tokens,
        call_llm=call_llm,
        model_name=settings.model_name,
        context_size=CONTEXT_SIZE,
        input_budget=INPUT_TOKEN_BUDGET,
    )

    return _record_exchange_result(
        output.as_legacy_response(),
        messages[-1]["content"],
        conversation_id,
        usage=output.usage,
    )