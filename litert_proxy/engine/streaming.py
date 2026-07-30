import json
import time
from typing import Any, Optional


def make_stream_chunk(
    model_name: str,
    text: str,
    finish_reason: Any = None,
    usage: Optional[dict[str, Any]] = None,
) -> str:
    chunk = {
        "id": "chatcmpl-litert",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": (
                    {"content": text}
                    if text
                    else {}
                ),
                "finish_reason": finish_reason,
            }
        ],
    }

    if usage is not None:
        chunk["usage"] = usage

    return f"data: {json.dumps(chunk)}\n\n"


def make_stream_reasoning_chunk(
    model_name: str,
    text: str,
    channel: str = "thinking",
) -> str:
    chunk = {
        "id": "chatcmpl-litert",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning_content": text,
                    "channel": channel,
                },
                "finish_reason": None,
            }
        ],
    }

    return f"data: {json.dumps(chunk)}\n\n"


def make_stream_tool_calls_chunk(
    model_name: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    delta_tool_calls = []

    for index, tool_call in enumerate(tool_calls):
        delta_tool_calls.append(
            {
                "index": index,
                "id": tool_call.get("id"),
                "type": "function",
                "function": {
                    "name": tool_call.get("function", {}).get("name"),
                    "arguments": tool_call.get("function", {}).get(
                        "arguments",
                        "{}",
                    ),
                },
            }
        )

    chunk = {
        "id": "chatcmpl-litert",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": delta_tool_calls},
                "finish_reason": None,
            }
        ],
    }

    return f"data: {json.dumps(chunk)}\n\n"


def make_initial_stream_chunk(model_name: str) -> str:
    chunk = {
        "id": "chatcmpl-litert",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }

    return f"data: {json.dumps(chunk)}\n\n"


def make_error_stream_chunk(exc: Exception) -> str:
    error_chunk = {
        "error": {
            "message": str(exc),
            "type": "server_error",
            "code": None,
        }
    }

    return f"data: {json.dumps(error_chunk)}\n\n"
