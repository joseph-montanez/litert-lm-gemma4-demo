import json
import sys
from typing import Any, List, Optional

from ..config import (
    ANTI_REPEAT_INSTRUCTION,
    _console_lock,
)
from ..models import ChatMessage
from .token import (
    content_to_text,
    json_safe,
    message_to_text,
    parse_tool_arguments,
    truncate_tool_content,
)


def canonical_message(message: ChatMessage) -> dict[str, Any]:
    role = message.role.strip().lower()

    if role == "model":
        role = "assistant"

    content = json_safe(message.content)

    if role == "tool":
        content = truncate_tool_content(content)

    result: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    if message.name is not None:
        result["name"] = message.name

    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id

    if message.tool_calls is not None:
        result["tool_calls"] = json_safe(message.tool_calls)

    if message.reasoning_content is not None:
        result["reasoning_content"] = message.reasoning_content

    return result


def canonical_messages(messages: List[ChatMessage]) -> list[dict[str, Any]]:
    return [canonical_message(message) for message in messages]


def build_name_by_tool_call_id_map(
    messages: list[dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}

    for message in messages:
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            tool_call_id = tool_call.get("id")
            function = tool_call.get("function")

            if not isinstance(function, dict):
                continue

            name = function.get("name")

            if tool_call_id and isinstance(name, str) and name:
                result[str(tool_call_id)] = name

    return result


def translate_openai_message(
    message: dict[str, Any],
    name_by_tool_call_id: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    role = message.get("role", "user")
    content = message.get("content")

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        name = message.get("name")

        if not name and tool_call_id and name_by_tool_call_id:
            name = name_by_tool_call_id.get(str(tool_call_id))

        if not name:
            raise ValueError(
                f"No matching tool name for tool_call_id={tool_call_id!r}."
            )

        return {
            "role": "tool",
            "content": [
                {
                    "type": "tool_response",
                    "name": str(name),
                    "response": content,
                }
            ],
        }

    if role == "assistant" and isinstance(message.get("tool_calls"), list):
        tool_calls = []

        for tool_call in message.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue

            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue

            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": parse_tool_arguments(
                            function.get("arguments", {})
                        ),
                    },
                }
            )

        translated: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }

        if content:
            translated["content"] = content

        return translated

    return {
        "role": role,
        "content": content if content is not None else "",
    }


def message_to_litert(
    message: dict[str, Any],
    name_by_tool_call_id: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return translate_openai_message(
        message,
        name_by_tool_call_id,
    )


def tool_response_content(
    message: dict[str, Any],
    name_by_tool_call_id: dict[str, str],
) -> dict[str, Any]:
    translated = translate_openai_message(
        message,
        name_by_tool_call_id,
    )
    content = translated.get("content", [])

    if not isinstance(content, list) or not content:
        raise ValueError("Tool response content is empty.")

    return content[0]


def build_initial_messages(
    messages: list[dict[str, Any]],
) -> list[Any]:
    system_parts = []
    history = []
    name_map = build_name_by_tool_call_id_map(messages)

    for message in messages:
        role = message.get("role", "user")

        if role == "system":
            text = content_to_text(message.get("content")).strip()
            if text:
                system_parts.append(text)
            continue

        history.append(
            message_to_litert(
                message,
                name_map,
            )
        )

    system_parts.append(ANTI_REPEAT_INSTRUCTION)

    initial_messages = [
        {
            "role": "system",
            "content": "\n\n".join(system_parts),
        }
    ]
    initial_messages.extend(history)

    return initial_messages


def last_input_index(
    messages: list[dict[str, Any]],
) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") in {"user", "tool"}:
            return index

    return -1


def starts_with_messages(
    messages: list[dict[str, Any]],
    prefix: list[dict[str, Any]],
) -> bool:
    return (
        len(messages) >= len(prefix)
        and messages[: len(prefix)] == prefix
    )


def assistant_response_matches(
    message: dict[str, Any],
    expected: Optional[dict[str, Any]],
) -> bool:
    if expected is None or message.get("role") != "assistant":
        return False

    # Reasoning channels are optional OpenAI extensions and many clients do
    # not echo them back in the next request. They must not invalidate an
    # otherwise reusable KV-cache prefix.
    comparable_keys = ("role", "content", "tool_calls")
    actual = {
        key: message.get(key)
        for key in comparable_keys
        if key in message or key in expected
    }
    wanted = {
        key: expected.get(key)
        for key in comparable_keys
        if key in message or key in expected
    }

    return json_safe(actual) == json_safe(wanted)


def combine_delta_messages(
    messages: list[dict[str, Any]],
    all_messages: list[dict[str, Any]],
    last_response_message: Optional[dict[str, Any]],
) -> Any:
    if not messages:
        raise ValueError("No appended input message was found.")

    name_source = list(all_messages)
    if last_response_message is not None:
        name_source.append(last_response_message)

    name_map = build_name_by_tool_call_id_map(name_source)

    if all(message.get("role") == "tool" for message in messages):
        contents = [
            tool_response_content(message, name_map)
            for message in messages
        ]
        with _console_lock:
            print(
                "\n[tool-response] "
                + json.dumps(contents, ensure_ascii=False, default=str),
                file=sys.stderr,
                flush=True,
            )
        return {
            "role": "tool",
            "content": contents,
        }

    if len(messages) == 1:
        return message_to_litert(
            messages[0],
            name_map,
        )

    parts = []

    for message in messages:
        role = message.get("role", "user")
        text = content_to_text(message.get("content")).strip()

        if text:
            parts.append(f"[{role}]\n{text}")

    return {
        "role": "user",
        "content": "\n\n".join(parts),
    }
