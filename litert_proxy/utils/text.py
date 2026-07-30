import json
import re
from typing import Any, Optional


class RepetitionGuard:
    def __init__(self, minimum_tokens: int = 3, maximum_tokens: int = 96):
        self.text = ""
        self.minimum_tokens = minimum_tokens
        self.maximum_tokens = maximum_tokens
        self.stopped = False

    def add(self, fragment: str) -> tuple[str, bool]:
        if self.stopped or not fragment:
            return "", self.stopped

        old_length = len(self.text)
        candidate = self.text + fragment
        cut = self._find_repetition_cut(candidate)

        if cut is None:
            self.text = candidate
            return fragment, False

        self.text = candidate[:cut].rstrip()
        self.stopped = True
        allowed = max(0, cut - old_length)
        return fragment[:allowed], True

    def _find_repetition_cut(self, text: str) -> Optional[int]:
        matches = list(re.finditer(r"\S+\s*", text))
        if len(matches) < 9:
            return None

        tokens = [match.group(0).strip().casefold() for match in matches]
        max_size = min(self.maximum_tokens, len(tokens) // 3)

        for size in range(max_size, self.minimum_tokens - 1, -1):
            first = tokens[-3 * size : -2 * size]
            second = tokens[-2 * size : -size]
            third = tokens[-size:]

            if first == second == third:
                return matches[-3 * size].start()

        for repeats in (8, 7, 6):
            if len(tokens) >= repeats and len(set(tokens[-repeats:])) == 1:
                return matches[-repeats].start()

        return None


def normalized_stops(stop: Any) -> list[str]:
    if isinstance(stop, str) and stop:
        return [stop]

    if isinstance(stop, list):
        return [
            value
            for value in stop
            if isinstance(value, str) and value
        ]

    return []


def apply_stop_sequences(
    accumulated: str,
    fragment: str,
    stops: list[str],
) -> tuple[str, bool]:
    if not stops:
        return fragment, False

    candidate = accumulated + fragment
    positions = [candidate.find(stop) for stop in stops]
    positions = [position for position in positions if position >= 0]

    if not positions:
        return fragment, False

    cut = min(positions)
    allowed = max(0, cut - len(accumulated))

    return fragment[:allowed], True


def extract_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk

    if not isinstance(chunk, dict):
        return ""

    direct_text = chunk.get("text")
    if isinstance(direct_text, str):
        return direct_text

    content = chunk.get("content", [])

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        content = [content]

    if not isinstance(content, list):
        return ""

    parts = []

    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).lower()
        channel = str(item.get("channel", "")).lower()

        if (
            item_type in {"text", "output_text", "assistant_text"}
            and channel not in {
                "analysis",
                "thinking",
                "thought",
                "reasoning",
            }
        ):
            text = item.get("text")

            if isinstance(text, str):
                parts.append(text)

    return "".join(parts)


def extract_reasoning_channels(
    chunk: Any,
) -> dict[str, str]:
    if not isinstance(chunk, dict):
        return {}

    result: dict[str, str] = {}

    channels = chunk.get("channels")
    if isinstance(channels, dict):
        for name, value in channels.items():
            if isinstance(value, str) and value:
                result[str(name)] = value

    for key in (
        "reasoning_content",
        "reasoning",
        "thinking",
        "analysis",
        "thought",
    ):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            result.setdefault(key, value)

    content = chunk.get("content", [])
    if isinstance(content, dict):
        content = [content]

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "")).lower()
            channel = str(item.get("channel", "")).lower()
            name = channel or item_type

            if (
                item_type
                not in {
                    "reasoning",
                    "reasoning_text",
                    "thinking",
                    "thought",
                    "analysis",
                }
                and channel
                not in {
                    "reasoning",
                    "thinking",
                    "thought",
                    "analysis",
                }
            ):
                continue

            text = item.get("text")
            if isinstance(text, str) and text:
                result[name or "thinking"] = text

    return result


def stream_value_delta(
    previous: str,
    current: str,
) -> tuple[str, str]:
    """Handle both fragment streams and cumulative snapshot streams."""
    if not current:
        return "", previous

    if not previous:
        return current, current

    if current.startswith(previous):
        return current[len(previous) :], current

    if previous.startswith(current):
        # A cumulative snapshot was rewritten backwards. Do not duplicate it.
        return "", current

    # Most LiteRT async callbacks are fragments. Treat non-prefix values as
    # the next fragment and retain the accumulated logical stream.
    return current, previous + current


def extract_litert_tool_calls(chunk: Any) -> list[dict[str, Any]]:
    if not isinstance(chunk, dict):
        return []

    raw_tool_calls = chunk.get("tool_calls", [])

    if not raw_tool_calls:
        content = chunk.get("content", [])
        if isinstance(content, dict):
            content = [content]

        if isinstance(content, list):
            raw_tool_calls = [
                item
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "tool_call"
            ]

    if not isinstance(raw_tool_calls, list):
        return []

    normalized = []

    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get("function")

        if not isinstance(function, dict):
            function = tool_call

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                arguments = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                arguments = {}

        if not isinstance(arguments, dict):
            arguments = {}

        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )

    return normalized


def filter_kwargs_for_callable(
    callable_object: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    import inspect

    parameters = set(inspect.signature(callable_object).parameters)
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }
