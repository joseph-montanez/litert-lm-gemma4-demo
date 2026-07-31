import json
import sys
from typing import Any

from .. import config as _cfg
from ..config import (
    MAX_TOOL_RESPONSE_TOKENS,
    ANTI_REPEAT_INSTRUCTION,
    CONTEXT_SAFETY_MARGIN_TOKENS,
    _console_lock,
)

# ---------------------------------------------------------------------------
# Low-level content utilities (used by both token estimation and message
# canonicalization – placed here to avoid circular imports.)
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    except Exception:
        return str(value)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            if item_type in {"text", "input_text", "output_text"}:
                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

            elif item_type == "tool_response":
                response = item.get("response")

                if isinstance(response, str):
                    parts.append(response)
                elif response is not None:
                    parts.append(json.dumps(response, ensure_ascii=False))

        return "\n".join(parts)

    if isinstance(content, dict):
        for key in ("text", "response", "content"):
            value = content.get(key)

            if isinstance(value, str):
                return value

        return json.dumps(content, ensure_ascii=False, sort_keys=True)

    if content is None:
        return ""

    return str(content)


def message_to_text(message: dict[str, Any]) -> str:
    return content_to_text(message.get("content")).strip()


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


# ---------------------------------------------------------------------------
# Token counting & estimation
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    if not text:
        return 0

    try:
        return len(_cfg.engine.tokenize(text))
    except Exception:
        return max(1, round(len(text) / 4))


def truncate_text_to_token_limit(
    text: str,
    token_limit: int,
) -> tuple[str, int, int]:
    original_tokens = count_tokens(text)

    if original_tokens <= token_limit:
        return text, original_tokens, original_tokens

    marker = (
        "\n\n[Tool output truncated by LiteRT proxy: "
        f"{original_tokens:,} tokens exceeded the "
        f"{token_limit:,}-token per-result limit.]\n\n"
    )
    marker_tokens = count_tokens(marker)
    content_budget = max(64, token_limit - marker_tokens)
    head_budget = max(32, int(content_budget * 0.8))
    tail_budget = max(16, content_budget - head_budget)

    def fit_prefix(value: str, budget: int) -> str:
        low = 0
        high = len(value)

        while low < high:
            mid = (low + high + 1) // 2

            if count_tokens(value[:mid]) <= budget:
                low = mid
            else:
                high = mid - 1

        return value[:low]

    def fit_suffix(value: str, budget: int) -> str:
        low = 0
        high = len(value)

        while low < high:
            mid = (low + high + 1) // 2
            candidate = value[len(value) - mid :]

            if count_tokens(candidate) <= budget:
                low = mid
            else:
                high = mid - 1

        return value[len(value) - low :] if low else ""

    head = fit_prefix(text, head_budget)
    tail = fit_suffix(text, tail_budget)
    truncated = head + marker + tail

    while count_tokens(truncated) > token_limit and len(head) > 32:
        head = head[: int(len(head) * 0.95)]
        truncated = head + marker + tail

    return truncated, original_tokens, count_tokens(truncated)


def truncate_tool_content(content: Any) -> Any:
    if MAX_TOOL_RESPONSE_TOKENS <= 0 or content is None:
        return content

    if isinstance(content, str):
        serialized = content
    else:
        serialized = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    truncated, original_tokens, final_tokens = truncate_text_to_token_limit(
        serialized,
        MAX_TOOL_RESPONSE_TOKENS,
    )

    if original_tokens != final_tokens:
        with _console_lock:
            print(
                "\n[tool-response-truncated] "
                f"{original_tokens:,} -> {final_tokens:,} tokens",
                file=sys.stderr,
                flush=True,
            )

    return truncated


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    parts = []

    for message in messages:
        text = message_to_text(message)

        if text:
            parts.append(f"<{message['role']}>\n{text}")

    parts.append(f"<system>\n{ANTI_REPEAT_INSTRUCTION}")
    return count_tokens("\n\n".join(parts))


def estimate_input_tokens(input_message: Any) -> int:
    if isinstance(input_message, str):
        return count_tokens(input_message)

    if isinstance(input_message, dict):
        role = input_message.get("role", "user")
        text = content_to_text(input_message.get("content"))

        if not text:
            text = json.dumps(
                input_message,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

        return count_tokens(f"<{role}>\n{text}")

    return count_tokens(str(input_message))


def requested_output_tokens(
    request: "ChatCompletionRequest",
) -> int:
    from ..config import DEFAULT_MAX_OUTPUT_TOKENS

    value = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )

    if value is None:
        value = DEFAULT_MAX_OUTPUT_TOKENS

    return max(1, int(value))


def estimate_tool_schema_tokens(
    request: "ChatCompletionRequest",
) -> int:
    from .tools import normalize_tool_definitions
    from ..workspace_tools import (
        resolve_workspace_root,
        workspace_tool_definitions,
    )

    definitions = normalize_tool_definitions(request)
    if request.workspace_tools:
        root = resolve_workspace_root(request.workspace_path or "")
        definitions.extend(
            workspace_tool_definitions(
                root,
                read_only=request.workspace_read_only,
            )
        )

    if not definitions:
        return 0

    serialized = json.dumps(
        definitions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return count_tokens(serialized)


def estimate_context_budget(
    request: "ChatCompletionRequest",
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    message_tokens = estimate_messages_tokens(messages)
    tool_schema_tokens = estimate_tool_schema_tokens(request)
    output_tokens = requested_output_tokens(request)
    projected_tokens = (
        message_tokens
        + tool_schema_tokens
        + output_tokens
        + CONTEXT_SAFETY_MARGIN_TOKENS
    )

    return {
        "message_tokens": message_tokens,
        "tool_schema_tokens": tool_schema_tokens,
        "output_tokens": output_tokens,
        "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
        "projected_tokens": projected_tokens,
    }
