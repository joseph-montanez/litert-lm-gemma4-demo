import argparse
import ast
import asyncio
import inspect
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional

import litert_lm
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

MODEL_PATH = os.environ.get(
    "LITERT_MODEL_PATH",
    "gemma-4-E4B-it.litertlm",
)
MAX_NUM_TOKENS = int(os.environ.get("LITERT_MAX_NUM_TOKENS", "32768"))
DEFAULT_MAX_OUTPUT_TOKENS = int(
    os.environ.get("LITERT_MAX_OUTPUT_TOKENS", "4096")
)
MAX_TOOL_RESPONSE_TOKENS = int(
    os.environ.get("LITERT_MAX_TOOL_RESPONSE_TOKENS", "4096")
)
CONTEXT_SAFETY_MARGIN_TOKENS = int(
    os.environ.get("LITERT_CONTEXT_SAFETY_MARGIN", "1024")
)
INFERENCE_TIMEOUT_SECONDS = float(
    os.environ.get("LITERT_INFERENCE_TIMEOUT", "180")
)
MALFORMED_TOOL_CALL_RETRIES = int(
    os.environ.get("LITERT_MALFORMED_TOOL_RETRIES", "1")
)
MAX_TOOL_ARGUMENT_STRING_LENGTH = int(
    os.environ.get("LITERT_MAX_TOOL_ARGUMENT_LENGTH", "16384")
)
CACHE_DIR = os.environ.get(
    "LITERT_CACHE_DIR",
    str(Path(MODEL_PATH).resolve().parent),
)

DEFAULT_TEMPERATURE = float(
    os.environ.get("LITERT_TEMPERATURE", "1.0")
)
DEFAULT_REASONING_EFFORT = os.environ.get(
    "LITERT_REASONING_EFFORT",
    "high",
).strip().lower()
THINKING_TOKEN_BUDGETS = {
    "none": 0,
    "minimal": 256,
    "low": 512,
    "medium": 1024,
    "high": 2048,
    "xhigh": 4096,
}
DEFAULT_TOOL_TEMPERATURE = float(
    os.environ.get("LITERT_TOOL_TEMPERATURE", "0.6")
)
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_REPETITION_PENALTY = 1.12
DEFAULT_NO_REPEAT_NGRAM_SIZE = 6
DEFAULT_REPETITION_WINDOW = 256

PROGRESS_INTERVAL_SECONDS = 1.0
SSE_HEARTBEAT_SECONDS = 5.0
ENABLE_CONSTRAINED_DECODING = os.environ.get(
    "LITERT_CONSTRAINED_DECODING",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}

WEB_UI_ENABLED = os.environ.get(
    "LITERT_WEB_UI",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
WEB_UI_TITLE = os.environ.get(
    "LITERT_WEB_UI_TITLE",
    "LiteRT Chat",
)

TOOL_CONTEXT_MODES = {"merged", "separate"}
TOOL_CONTEXT_MODE = os.environ.get(
    "LITERT_TOOL_CONTEXT_MODE",
    "separate",
).strip().lower()

if TOOL_CONTEXT_MODE not in TOOL_CONTEXT_MODES:
    raise RuntimeError(
        "LITERT_TOOL_CONTEXT_MODE must be 'merged' or 'separate'."
    )

ANTI_REPEAT_INSTRUCTION = (
    "Do not repeat sentences, paragraphs, lists, or sections. "
    "Once the answer is complete, stop generating."
)

TOOL_USE_INSTRUCTION = ""

engine = None
conversation_worker = None

_console_lock = threading.Lock()
_request_lock = threading.Lock()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Any = None
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model: Optional[str] = MODEL_PATH
    messages: List[ChatMessage]
    stream: bool = False

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Any = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    repetition_window: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None
    include_reasoning: Optional[bool] = None

    tools: Any = None
    tool_choice: Any = None
    parallel_tool_calls: Any = None


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


class ConsoleProgress:
    def __init__(
        self,
        total_prompt_tokens: int,
        prefill_tokens: int,
        cache_mode: str,
    ):
        self.request_id = uuid.uuid4().hex[:8]
        self.total_prompt_tokens = total_prompt_tokens
        self.prefill_tokens = prefill_tokens
        self.cached_tokens = max(0, total_prompt_tokens - prefill_tokens)
        self.cache_mode = cache_mode

        self.started_at = time.perf_counter()
        self.first_token_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.blocking_generation_started_at: Optional[float] = None
        self.blocking_generation_finished_at: Optional[float] = None

        self.pieces = 0
        self.stream_chunks = 0
        self.characters = 0
        self.output_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.generated_parts: list[str] = []

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._write(
            f"[{self.request_id}] cache={self.cache_mode}"
            f" | total≈{self.total_prompt_tokens:,}"
            f" | prefill≈{self.prefill_tokens:,}"
            f" | cached≈{self.cached_tokens:,}"
        )
        self._thread.start()

    def set_blocking_generation_timing(
        self,
        started_at: float,
        finished_at: float,
    ):
        # Compatibility with earlier server revisions. The active generation
        # path now uses send_message_async() for all requests.
        with self._state_lock:
            self.blocking_generation_started_at = started_at
            self.blocking_generation_finished_at = finished_at

    def observe_stream_chunk(self):
        now = time.perf_counter()

        with self._state_lock:
            if self.first_token_at is None:
                self.first_token_at = now

            self.stream_chunks += 1

    def add_fragment(self, fragment: str):
        if not fragment:
            return

        now = time.perf_counter()

        with self._state_lock:
            if self.first_token_at is None:
                self.first_token_at = now

            self.pieces += 1
            self.characters += len(fragment)
            self.output_parts.append(fragment)
            self.generated_parts.append(fragment)

    def add_reasoning(self, fragment: str):
        if not fragment:
            return

        now = time.perf_counter()

        with self._state_lock:
            if self.first_token_at is None:
                self.first_token_at = now

            self.pieces += 1
            self.characters += len(fragment)
            self.reasoning_parts.append(fragment)
            self.generated_parts.append(fragment)

    def add_tool_calls(self, tool_calls: list[dict[str, Any]]):
        serialized = json.dumps(
            tool_calls,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.add_fragment(serialized)

    def output_text(self) -> str:
        with self._state_lock:
            return "".join(self.output_parts)

    def reasoning_text(self) -> str:
        with self._state_lock:
            return "".join(self.reasoning_parts)

    def generated_text(self) -> str:
        with self._state_lock:
            return "".join(self.generated_parts)

    def finish(self, status: str = "done") -> dict[str, Any]:
        self.finished_at = time.perf_counter()
        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(timeout=0.2)

        output_text = self.output_text()
        reasoning_text = self.reasoning_text()
        generated_text = self.generated_text()
        output_tokens = count_tokens(output_text)
        reasoning_tokens = count_tokens(reasoning_text)
        generated_tokens = count_tokens(generated_text)
        total_seconds = max(self.finished_at - self.started_at, 0.000001)

        ttft = None
        prefill_tps = None
        decode_tps = None
        visible_tps = None
        generation_seconds = None
        generation_tps = None
        blocking_response = False

        with self._state_lock:
            blocking_started_at = self.blocking_generation_started_at
            blocking_finished_at = self.blocking_generation_finished_at

        if (
            blocking_started_at is not None
            and blocking_finished_at is not None
        ):
            blocking_response = True
            generation_seconds = max(
                blocking_finished_at - blocking_started_at,
                0.000001,
            )
            generation_tps = output_tokens / generation_seconds
        elif self.first_token_at is not None:
            ttft = max(self.first_token_at - self.started_at, 0.000001)
            prefill_tps = (
                self.prefill_tokens / ttft if self.prefill_tokens else None
            )
            decode_seconds = max(
                self.finished_at - self.first_token_at,
                0.000001,
            )
            decode_tps = generated_tokens / decode_seconds
            visible_tps = output_tokens / decode_seconds

        parts = [
            f"[{self.request_id}] {status}",
            f"cache={self.cache_mode}",
            f"total≈{self.total_prompt_tokens:,}",
            f"prefill≈{self.prefill_tokens:,}",
            f"cached≈{self.cached_tokens:,}",
            f"output={output_tokens:,}",
            f"thinking={reasoning_tokens:,}",
            f"generated={generated_tokens:,}",
            f"time={total_seconds:.2f}s",
        ]

        if blocking_response:
            parts.append(f"response={generation_seconds:.2f}s")
            parts.append(f"e2e={generation_tps:,.1f} tok/s")
            parts.append("decode=n/a (blocking tool response)")
        else:
            if ttft is not None:
                parts.append(f"TTFT={ttft:.2f}s")

            if prefill_tps is not None:
                parts.append(f"PPS≈{prefill_tps:,.1f} tok/s")

            if decode_tps is not None:
                parts.append(f"TPS≈{decode_tps:,.1f} generated tok/s")

            if visible_tps is not None:
                parts.append(f"visible≈{visible_tps:,.1f} tok/s")

        self._write(" | ".join(parts), final=True)

        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": self.total_prompt_tokens + generated_tokens,
            "prefill_tokens": self.prefill_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_mode": self.cache_mode,
            "time_to_first_token": ttft,
            "prefill_tokens_per_second": prefill_tps,
            "decode_tokens_per_second": decode_tps,
            "visible_tokens_per_second": visible_tps,
            "blocking_response": blocking_response,
            "generation_seconds": generation_seconds,
            "generation_tokens_per_second": generation_tps,
            "total_seconds": total_seconds,
        }

    def _run(self):
        while not self._stop_event.wait(PROGRESS_INTERVAL_SECONDS):
            now = time.perf_counter()

            with self._state_lock:
                first_token_at = self.first_token_at
                pieces = self.pieces
                stream_chunks = self.stream_chunks
                characters = self.characters

            if first_token_at is None:
                elapsed = now - self.started_at
                line = (
                    f"[{self.request_id}] cache={self.cache_mode}"
                    f" | prefilling {elapsed:.1f}s"
                    f" | new≈{self.prefill_tokens:,}"
                    f" | cached≈{self.cached_tokens:,}"
                )
            else:
                decode_elapsed = max(now - first_token_at, 0.000001)
                chunk_rate = stream_chunks / decode_elapsed
                line = (
                    f"[{self.request_id}] decoding {decode_elapsed:.1f}s"
                    f" | chunks={stream_chunks:,}"
                    f" | emitted={pieces:,}"
                    f" | stream≈{chunk_rate:,.1f} chunks/s"
                    f" | chars={characters:,}"
                )

            self._write(line)

    def _write(self, text: str, final: bool = False):
        with _console_lock:
            if final:
                sys.stderr.write("\r" + (" " * 200) + "\r")
                sys.stderr.write(text + "\n")
            else:
                sys.stderr.write("\r" + text[:199].ljust(200))

            sys.stderr.flush()


class MalformedToolCallError(RuntimeError):
    pass


@dataclass
class GenerationState:
    cancelled: bool = False
    repetition_stopped: bool = False
    stop_sequence_hit: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InferenceJob:
    request: ChatCompletionRequest
    messages: list[dict[str, Any]]
    result_queue: queue.Queue
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class ConversationPlan:
    mode: str
    input_message: Any = None
    initial_messages: Optional[list[Any]] = None
    total_prompt_tokens: int = 0
    prefill_tokens: int = 0


@dataclass
class ProxyTool(litert_lm.Tool):
    definition: dict[str, Any]

    def get_tool_description(self) -> dict[str, Any]:
        return self.definition

    def execute(self, param: Any) -> Any:
        raise NotImplementedError(
            "Proxy tools are executed by the OpenAI-compatible client."
        )


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


def message_to_text(message: dict[str, Any]) -> str:
    return content_to_text(message.get("content")).strip()


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


def enrich_tool_definition(
    tool: dict[str, Any],
) -> dict[str, Any]:
    return json_safe(tool)


def normalize_tool_definitions(
    request: ChatCompletionRequest,
) -> list[dict[str, Any]]:
    if request.tool_choice == "none":
        return []

    tools = request.tools if isinstance(request.tools, list) else []
    definitions = []

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        definitions.append(enrich_tool_definition(tool))

    if isinstance(request.tool_choice, dict):
        function = request.tool_choice.get("function")
        required_name = (
            function.get("name") if isinstance(function, dict) else None
        )

        if required_name:
            definitions = [
                definition
                for definition in definitions
                if definition.get("function", {}).get("name") == required_name
            ]

    return definitions


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

def count_tokens(text: str) -> int:
    if not text:
        return 0

    try:
        return len(engine.tokenize(text))
    except Exception:
        return max(1, round(len(text) / 4))


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
    request: ChatCompletionRequest,
) -> int:
    value = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )

    if value is None:
        value = DEFAULT_MAX_OUTPUT_TOKENS

    return max(1, int(value))


def estimate_tool_schema_tokens(
    request: ChatCompletionRequest,
) -> int:
    definitions = normalize_tool_definitions(request)

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
    request: ChatCompletionRequest,
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

def build_sampler_config(request: ChatCompletionRequest):
    sampler_type = getattr(litert_lm, "SamplerConfig", None)

    if sampler_type is None:
        return None

    requested_temperature = (
        request.temperature
        if request.temperature is not None
        else DEFAULT_TEMPERATURE
    )
    effective_temperature = requested_temperature

    kwargs = {
        "top_k": (
            request.top_k
            if request.top_k is not None
            else DEFAULT_TOP_K
        ),
        "top_p": (
            request.top_p
            if request.top_p is not None
            else DEFAULT_TOP_P
        ),
        "temperature": effective_temperature,
    }

    sampler_parameters = set(inspect.signature(sampler_type).parameters)

    if request.seed is not None and "seed" in sampler_parameters:
        kwargs["seed"] = request.seed

    return sampler_type(**kwargs)



def normalize_reasoning_effort(
    request: ChatCompletionRequest,
) -> str:
    value = (
        request.reasoning_effort
        if request.reasoning_effort is not None
        else DEFAULT_REASONING_EFFORT
    )

    value = str(value).strip().lower()
    aliases = {
        "off": "none",
        "disabled": "none",
        "false": "none",
        "0": "none",
        "on": "medium",
        "enabled": "medium",
        "true": "medium",
        "1": "medium",
        "extra_high": "xhigh",
        "very_high": "xhigh",
    }
    value = aliases.get(value, value)

    if value not in THINKING_TOKEN_BUDGETS:
        raise HTTPException(
            status_code=400,
            detail=(
                "reasoning_effort must be one of: "
                "none, minimal, low, medium, high, xhigh."
            ),
        )

    return value


def thinking_enabled(
    request: ChatCompletionRequest,
) -> bool:
    return normalize_reasoning_effort(request) != "none"


def thinking_token_budget(
    request: ChatCompletionRequest,
) -> int:
    return THINKING_TOKEN_BUDGETS[
        normalize_reasoning_effort(request)
    ]


def build_native_thinking_config(
    request: ChatCompletionRequest,
) -> Any:
    thinking_type = getattr(
        litert_lm,
        "ThinkingConfig",
        None,
    )

    if thinking_type is None:
        interfaces_module = getattr(
            litert_lm,
            "interfaces",
            None,
        )
        thinking_type = getattr(
            interfaces_module,
            "ThinkingConfig",
            None,
        )

    if thinking_type is None:
        return None

    parameters = set(
        inspect.signature(thinking_type).parameters
    )
    kwargs: dict[str, Any] = {}

    if "enable_thinking" in parameters:
        kwargs["enable_thinking"] = thinking_enabled(request)

    budget = thinking_token_budget(request)

    if "thinking_token_budget" in parameters:
        kwargs["thinking_token_budget"] = budget
    elif "thinking_budget" in parameters:
        kwargs["thinking_budget"] = budget

    return thinking_type(**kwargs)


def thinking_runtime_mode(
    request: ChatCompletionRequest,
) -> str:
    if not thinking_enabled(request):
        return "disabled"

    if build_native_thinking_config(request) is not None:
        return "native-thinking-config"

    return "template-extra-context"

def conversation_config_signature(
    request: ChatCompletionRequest,
) -> str:
    signature = {
        "model": request.model or MODEL_PATH,
        "temperature": (
            request.temperature
            if request.temperature is not None
            else DEFAULT_TEMPERATURE
        ),
        "top_p": (
            request.top_p
            if request.top_p is not None
            else DEFAULT_TOP_P
        ),
        "top_k": (
            request.top_k
            if request.top_k is not None
            else DEFAULT_TOP_K
        ),
        "seed": request.seed,
        "reasoning_effort": normalize_reasoning_effort(request),
        "tools": json_safe(request.tools),
        "tool_choice": json_safe(request.tool_choice),
        "parallel_tool_calls": json_safe(request.parallel_tool_calls),
        "max_tokens": request.max_tokens,
        "max_completion_tokens": request.max_completion_tokens,
        "stop": json_safe(request.stop),
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "repetition_window": request.repetition_window,
        "no_repeat_ngram_size": request.no_repeat_ngram_size,
    }

    return json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_conversation_kwargs(
    request: ChatCompletionRequest,
    initial_messages: list[Any],
) -> dict[str, Any]:
    parameters = set(inspect.signature(engine.create_conversation).parameters)
    kwargs: dict[str, Any] = {
        "messages": initial_messages,
    }

    if "sampler_config" in parameters:
        sampler_config = build_sampler_config(request)

        if sampler_config is not None:
            kwargs["sampler_config"] = sampler_config

    tool_definitions = normalize_tool_definitions(request)

    if tool_definitions:
        with _console_lock:
            print(
                "\n[tools] "
                + ", ".join(
                    definition.get("function", {}).get(
                        "name",
                        "<unnamed>",
                    )
                    for definition in tool_definitions
                ),
                file=sys.stderr,
                flush=True,
            )

    if "tools" in parameters:
        kwargs["tools"] = (
            [ProxyTool(definition) for definition in tool_definitions]
            if tool_definitions
            else None
        )

    if "automatic_tool_calling" in parameters:
        kwargs["automatic_tool_calling"] = False

    if (
        tool_definitions
        and ENABLE_CONSTRAINED_DECODING
        and "enable_constrained_decoding" in parameters
    ):
        kwargs["enable_constrained_decoding"] = True

    enabled = thinking_enabled(request)

    if "extra_context" in parameters:
        kwargs["extra_context"] = {
            "enable_thinking": enabled,
        }

    native_thinking_config = build_native_thinking_config(
        request
    )

    if (
        native_thinking_config is not None
        and "thinking_config" in parameters
    ):
        kwargs["thinking_config"] = native_thinking_config

    if (
        not tool_definitions
        and enabled
        and "filter_channel_content_from_kv_cache" in parameters
    ):
        kwargs["filter_channel_content_from_kv_cache"] = True

    with _console_lock:
        print(
            "\n[thinking] "
            f"effort={normalize_reasoning_effort(request)} "
            f"mode={thinking_runtime_mode(request)} "
            f"budget={thinking_token_budget(request):,}",
            file=sys.stderr,
            flush=True,
        )

    return kwargs

def build_send_kwargs(
    conversation: Any,
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    parameters = set(
        inspect.signature(conversation.send_message_async).parameters
    )
    kwargs: dict[str, Any] = {}

    output_limit = requested_output_tokens(request)

    if "max_output_tokens" in parameters:
        kwargs["max_output_tokens"] = output_limit

    repetition_type = getattr(
        litert_lm,
        "RepetitionPenaltyConfig",
        None,
    )

    if (
        repetition_type is not None
        and "repetition_penalty_config" in parameters
    ):
        repetition_parameters = set(
            inspect.signature(repetition_type).parameters
        )
        repetition_kwargs: dict[str, Any] = {}

        if "repetition_penalty" in repetition_parameters:
            repetition_kwargs["repetition_penalty"] = (
                request.repetition_penalty
                if request.repetition_penalty is not None
                else DEFAULT_REPETITION_PENALTY
            )

        if (
            request.presence_penalty is not None
            and "presence_penalty" in repetition_parameters
        ):
            repetition_kwargs["presence_penalty"] = (
                request.presence_penalty
            )

        if (
            request.frequency_penalty is not None
            and "frequency_penalty" in repetition_parameters
        ):
            repetition_kwargs["frequency_penalty"] = (
                request.frequency_penalty
            )

        if "window_size" in repetition_parameters:
            repetition_kwargs["window_size"] = (
                request.repetition_window
                if request.repetition_window is not None
                else DEFAULT_REPETITION_WINDOW
            )

        kwargs["repetition_penalty_config"] = repetition_type(
            **repetition_kwargs
        )

    ngram_type = getattr(litert_lm, "NoRepeatNgramConfig", None)

    if ngram_type is not None and "no_repeat_ngram_config" in parameters:
        ngram_parameters = set(inspect.signature(ngram_type).parameters)
        ngram_kwargs: dict[str, Any] = {}

        if "no_repeat_ngram_size" in ngram_parameters:
            ngram_kwargs["no_repeat_ngram_size"] = (
                request.no_repeat_ngram_size
                if request.no_repeat_ngram_size is not None
                else DEFAULT_NO_REPEAT_NGRAM_SIZE
            )

        if "window_size" in ngram_parameters:
            ngram_kwargs["window_size"] = (
                request.repetition_window
                if request.repetition_window is not None
                else DEFAULT_REPETITION_WINDOW
            )

        kwargs["no_repeat_ngram_config"] = ngram_type(**ngram_kwargs)

    native_thinking_config = build_native_thinking_config(
        request
    )

    if (
        native_thinking_config is not None
        and "thinking_config" in parameters
    ):
        kwargs["thinking_config"] = native_thinking_config

    return kwargs


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



_PROTOCOL_LEAK_PATTERNS = (
    re.compile(
        r"(?:^|[\r\n\]}])(?:user|assistant|model|tool)\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[(?:user|assistant|model|tool)\](?:\r?\n|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\|(?:user|assistant|model|tool|turn|channel|end)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s.}\]])call:[A-Za-z_][A-Za-z0-9_.-]*\{",
        re.IGNORECASE,
    ),
    re.compile(r"}response:", re.IGNORECASE),
)


def value_contains_protocol_leak(value: Any) -> bool:
    if isinstance(value, str):
        if len(value) > MAX_TOOL_ARGUMENT_STRING_LENGTH:
            return True

        return any(
            pattern.search(value)
            for pattern in _PROTOCOL_LEAK_PATTERNS
        )

    if isinstance(value, dict):
        return any(
            value_contains_protocol_leak(key)
            or value_contains_protocol_leak(item)
            for key, item in value.items()
        )

    if isinstance(value, list):
        return any(
            value_contains_protocol_leak(item)
            for item in value
        )

    return False


def json_schema_type_matches(
    value: Any,
    expected_type: Any,
) -> bool:
    if isinstance(expected_type, list):
        return any(
            json_schema_type_matches(value, item)
            for item in expected_type
        )

    if expected_type == "null":
        return value is None

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if expected_type == "array":
        return isinstance(value, list)

    if expected_type == "object":
        return isinstance(value, dict)

    return True


def validate_schema_value(
    value: Any,
    schema: Any,
    path: str,
) -> list[str]:
    if not isinstance(schema, dict):
        return []

    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type is not None and not json_schema_type_matches(
        value,
        expected_type,
    ):
        errors.append(
            f"{path} has the wrong type; expected {expected_type!r}."
        )
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path} is not one of the allowed values.")

    if isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(
                        f"{path}.{key} is required."
                    )

        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                errors.append(
                    f"{path} contains unknown properties: "
                    + ", ".join(unknown)
                )

        for key, item in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                errors.extend(
                    validate_schema_value(
                        item,
                        child_schema,
                        f"{path}.{key}",
                    )
                )

    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    validate_schema_value(
                        item,
                        item_schema,
                        f"{path}[{index}]",
                    )
                )

    return errors


def tool_definition_map(
    request: ChatCompletionRequest,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for definition in normalize_tool_definitions(request):
        function = definition.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if isinstance(name, str) and name:
            result[name] = function

    return result


def validate_litert_tool_calls(
    tool_calls: list[dict[str, Any]],
    request: ChatCompletionRequest,
) -> None:
    definitions = tool_definition_map(request)

    if not tool_calls:
        return

    if not definitions:
        raise MalformedToolCallError(
            "The model returned tool calls but the request supplied no tools."
        )

    errors: list[str] = []

    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function")
        if not isinstance(function, dict):
            errors.append(
                f"tool_calls[{index}] has no function object."
            )
            continue

        name = function.get("name")
        arguments = function.get("arguments")

        if not isinstance(name, str) or not name:
            errors.append(
                f"tool_calls[{index}] has no valid function name."
            )
            continue

        definition = definitions.get(name)
        if definition is None:
            errors.append(
                f"tool_calls[{index}] requested unknown tool {name!r}."
            )
            continue

        if not isinstance(arguments, dict):
            errors.append(
                f"tool_calls[{index}] arguments are not an object."
            )
            continue

        if value_contains_protocol_leak(arguments):
            errors.append(
                f"tool_calls[{index}] for {name!r} contains "
                "chat-template or tool-protocol leakage."
            )
            continue

        schema = definition.get("parameters")
        errors.extend(
            validate_schema_value(
                arguments,
                schema,
                f"tool_calls[{index}].arguments",
            )
        )

    if errors:
        raise MalformedToolCallError(" ".join(errors))


def normalize_recovered_tool_call(
    name: Any,
    arguments: Any,
) -> Optional[dict[str, Any]]:
    if not isinstance(name, str) or not name.strip():
        return None

    if isinstance(arguments, str):
        parsed_arguments: Any = None

        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            try:
                parsed_arguments = ast.literal_eval(arguments)
            except (ValueError, SyntaxError):
                parsed_arguments = None

        arguments = parsed_arguments

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return None

    return {
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


def recover_tool_calls_from_object(
    value: Any,
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        recovered: list[dict[str, Any]] = []

        for item in value:
            recovered.extend(recover_tool_calls_from_object(item))

        return recovered

    if not isinstance(value, dict):
        return []

    direct = extract_litert_tool_calls(value)
    if direct:
        return direct

    function = value.get("function")
    if isinstance(function, dict):
        recovered_call = normalize_recovered_tool_call(
            function.get("name"),
            function.get("arguments", {}),
        )
        if recovered_call is not None:
            return [recovered_call]

    if value.get("type") in {"tool_call", "function"}:
        recovered_call = normalize_recovered_tool_call(
            value.get("name"),
            value.get("arguments", {}),
        )
        if recovered_call is not None:
            return [recovered_call]

    recovered = []

    for key in (
        "message",
        "response",
        "output",
        "content",
        "tool_calls",
    ):
        if key in value:
            recovered.extend(
                recover_tool_calls_from_object(value.get(key))
            )

    return recovered


def recover_tool_calls_from_text(
    text: str,
) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        recovered = recover_tool_calls_from_object(parsed)
        if recovered:
            return recovered

    for match in re.finditer(
        r"(?:^|[\s<\[\]}>])call:"
        r"([A-Za-z_][A-Za-z0-9_.-]*)\s*",
        stripped,
        flags=re.IGNORECASE,
    ):
        name = match.group(1)
        brace_index = stripped.find("{", match.end())

        if brace_index < 0:
            continue

        depth = 0
        quote: Optional[str] = None
        escaped = False
        argument_text = None

        for index in range(brace_index, len(stripped)):
            character = stripped[index]

            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue

            if character in {"'", '"'}:
                quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1

                if depth == 0:
                    argument_text = stripped[brace_index : index + 1]
                    break

        if argument_text is None:
            continue

        try:
            arguments = json.loads(argument_text)
        except json.JSONDecodeError:
            try:
                arguments = ast.literal_eval(argument_text)
            except (ValueError, SyntaxError):
                continue

        recovered_call = normalize_recovered_tool_call(
            name,
            arguments,
        )
        if recovered_call is not None:
            return [recovered_call]

    return []


def looks_like_tool_protocol(text: str) -> bool:
    stripped = text.lstrip()
    lowered = stripped.casefold()

    return (
        lowered.startswith("call:")
        or lowered.startswith("<|tool")
        or lowered.startswith("<start_function_call>")
        or '"tool_calls"' in lowered
        or '"tool_call"' in lowered
    )

def litert_tool_calls_to_openai(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name")

        if not isinstance(name, str) or not name:
            continue

        arguments = function.get("arguments", {})

        result.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )

    return result


def tool_call_key(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function", {})
    return json.dumps(
        {
            "name": function.get("name"),
            "arguments": function.get("arguments", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def message_to_litert(
    message: dict[str, Any],
    name_by_tool_call_id: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return translate_openai_message(
        message,
        name_by_tool_call_id,
    )


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


def filter_kwargs_for_callable(
    callable_object: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    parameters = set(inspect.signature(callable_object).parameters)
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }


def process_final_response(
    response: Any,
    request: ChatCompletionRequest,
    progress: ConsoleProgress,
    state: GenerationState,
) -> Iterator[tuple[str, Any]]:
    litert_tool_calls = extract_litert_tool_calls(response)

    if litert_tool_calls:
        validate_litert_tool_calls(
            litert_tool_calls,
            request,
        )
        openai_tool_calls = litert_tool_calls_to_openai(
            litert_tool_calls
        )
        state.tool_calls.extend(openai_tool_calls)
        progress.add_tool_calls(openai_tool_calls)

        with _console_lock:
            for debug_call in litert_tool_calls:
                debug_function = debug_call.get("function", {})
                print(
                    "\n[tool-call-final] "
                    f"name={debug_function.get('name')!r} "
                    f"arguments={debug_function.get('arguments')!r}",
                    file=sys.stderr,
                    flush=True,
                )

        yield "tool_calls", openai_tool_calls
        return

    fragment = extract_text(response)
    if not fragment:
        return

    fragment, stop_hit = apply_stop_sequences(
        "",
        fragment,
        normalized_stops(request.stop),
    )

    if stop_hit:
        state.stop_sequence_hit = True

    guard = RepetitionGuard()
    guarded_fragment, repeated = guard.add(fragment)

    if guarded_fragment:
        progress.add_fragment(guarded_fragment)
        yield "text", guarded_fragment

    if repeated:
        state.repetition_stopped = True


def generation_events(
    conversation: Any,
    input_message: Any,
    request: ChatCompletionRequest,
    progress: ConsoleProgress,
    cancel_event: threading.Event,
    state: GenerationState,
) -> Iterator[tuple[str, Any]]:
    send_kwargs = build_send_kwargs(conversation, request)
    has_tools = bool(normalize_tool_definitions(request))
    include_reasoning = (
        request.include_reasoning
        if request.include_reasoning is not None
        else thinking_enabled(request)
    )

    guard = RepetitionGuard()
    stops = normalized_stops(request.stop)
    accumulated = ""
    visible_stream_state = ""
    channel_stream_state: dict[str, str] = {}

    latest_litert_tool_calls: list[dict[str, Any]] = []
    tool_protocol_buffer = ""
    emitted_text = False
    emitted_reasoning = False
    saw_generated_chunk = False

    timed_out = threading.Event()

    def cancel_for_timeout():
        timed_out.set()
        cancel_event.set()

        try:
            conversation.cancel_process()
        except Exception:
            pass

    timeout_timer: Optional[threading.Timer] = None

    if INFERENCE_TIMEOUT_SECONDS > 0:
        timeout_timer = threading.Timer(
            INFERENCE_TIMEOUT_SECONDS,
            cancel_for_timeout,
        )
        timeout_timer.daemon = True
        timeout_timer.start()

    def emit_visible(fragment: str) -> Iterator[tuple[str, Any]]:
        nonlocal accumulated
        nonlocal emitted_text

        fragment, stop_hit = apply_stop_sequences(
            accumulated,
            fragment,
            stops,
        )

        if fragment:
            guarded_fragment, repeated = guard.add(fragment)

            if guarded_fragment:
                accumulated += guarded_fragment
                emitted_text = True
                progress.add_fragment(guarded_fragment)
                yield "text", guarded_fragment

            if repeated:
                state.cancelled = True
                state.repetition_stopped = True

                try:
                    conversation.cancel_process()
                except Exception:
                    pass

        if stop_hit:
            state.cancelled = True
            state.stop_sequence_hit = True

            try:
                conversation.cancel_process()
            except Exception:
                pass

    stream = conversation.send_message_async(
        input_message,
        **send_kwargs,
    )

    try:
        for chunk in stream:
            if timed_out.is_set():
                state.cancelled = True
                raise TimeoutError(
                    "LiteRT inference exceeded "
                    f"{INFERENCE_TIMEOUT_SECONDS:.0f} seconds "
                    "and was cancelled."
                )

            if cancel_event.is_set():
                state.cancelled = True

                try:
                    conversation.cancel_process()
                except Exception:
                    pass

                break

            fragment = extract_text(chunk)
            reasoning_channels = extract_reasoning_channels(chunk)
            chunk_tool_calls = extract_litert_tool_calls(chunk)

            if fragment or reasoning_channels or chunk_tool_calls:
                saw_generated_chunk = True
                progress.observe_stream_chunk()

            for channel_name, channel_value in reasoning_channels.items():
                previous = channel_stream_state.get(channel_name, "")
                reasoning_delta, updated = stream_value_delta(
                    previous,
                    channel_value,
                )
                channel_stream_state[channel_name] = updated

                if not reasoning_delta:
                    continue

                emitted_reasoning = True
                progress.add_reasoning(reasoning_delta)

                if include_reasoning:
                    yield "reasoning", {
                        "channel": channel_name,
                        "text": reasoning_delta,
                    }

            if chunk_tool_calls:
                latest_litert_tool_calls = chunk_tool_calls
                continue

            if not fragment:
                continue

            visible_delta, visible_stream_state = stream_value_delta(
                visible_stream_state,
                fragment,
            )

            if not visible_delta:
                continue

            if has_tools and (
                tool_protocol_buffer
                or looks_like_tool_protocol(visible_delta)
            ):
                tool_protocol_buffer += visible_delta
                recovered = recover_tool_calls_from_text(
                    tool_protocol_buffer
                )

                if recovered:
                    latest_litert_tool_calls = recovered
                    continue

                # Do not hold ordinary prose indefinitely. Only keep buffering
                # while the content still resembles an actual tool protocol.
                if (
                    len(tool_protocol_buffer) < 512
                    and looks_like_tool_protocol(tool_protocol_buffer)
                ):
                    continue

                buffered = tool_protocol_buffer
                tool_protocol_buffer = ""
                yield from emit_visible(buffered)

                if state.cancelled:
                    break

                continue

            yield from emit_visible(visible_delta)

            if state.cancelled:
                break

        if timed_out.is_set():
            state.cancelled = True
            raise TimeoutError(
                "LiteRT inference exceeded "
                f"{INFERENCE_TIMEOUT_SECONDS:.0f} seconds "
                "and was cancelled."
            )

        if state.cancelled:
            return

        if not latest_litert_tool_calls and tool_protocol_buffer:
            latest_litert_tool_calls = recover_tool_calls_from_text(
                tool_protocol_buffer
            )

        if latest_litert_tool_calls:
            final_tool_response = {
                "tool_calls": latest_litert_tool_calls,
            }
            yield from process_final_response(
                final_tool_response,
                request,
                progress,
                state,
            )
            return

        if tool_protocol_buffer:
            yield from emit_visible(tool_protocol_buffer)

        if not emitted_text and not emitted_reasoning and not state.tool_calls:
            with _console_lock:
                print(
                    "\n[empty-generation] "
                    f"chunks_seen={saw_generated_chunk}",
                    file=sys.stderr,
                    flush=True,
                )

            raise RuntimeError(
                "LiteRT async generation completed without assistant text, "
                "thinking-channel content, or a recoverable tool call."
            )
    finally:
        if timeout_timer is not None:
            timeout_timer.cancel()

        close = getattr(stream, "close", None)

        if callable(close):
            close()


class PersistentConversationWorker:
    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="litert-conversation-worker",
            daemon=True,
        )
        self._stop_event = threading.Event()
        self._conversation = None
        self._active_conversation = None
        self._current_job: Optional[InferenceJob] = None

        self._last_request_messages: list[dict[str, Any]] = []
        self._last_response_text = ""
        self._last_reasoning_text = ""
        self._last_tool_calls: list[dict[str, Any]] = []
        self._last_response_message: Optional[dict[str, Any]] = None
        self._last_finish_reason = "stop"
        self._last_usage: dict[str, Any] = {}
        self._config_signature: Optional[str] = None

    def start(self):
        self._thread.start()

    def submit(self, job: InferenceJob):
        self._jobs.put(job)

    def cancel_current(self):
        current_job = self._current_job

        if current_job is not None:
            current_job.cancel_event.set()

        conversations = [
            self._active_conversation,
            self._conversation,
        ]

        for conversation in conversations:
            if conversation is None:
                continue

            try:
                conversation.cancel_process()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        self.cancel_current()
        self._jobs.put(None)
        self._thread.join(timeout=10)

    def status(self) -> dict[str, Any]:
        return {
            "conversation_cached": self._conversation is not None,
            "active_conversation": self._active_conversation is not None,
            "cached_request_messages": len(self._last_request_messages),
            "last_response_characters": len(self._last_response_text),
            "last_reasoning_characters": len(self._last_reasoning_text),
            "last_tool_calls": len(self._last_tool_calls),
            "tool_context_mode": TOOL_CONTEXT_MODE,
            "tool_chain_persistent": TOOL_CONTEXT_MODE == "merged",
            "busy": self._current_job is not None,
        }

    def reset(self):
        self.cancel_current()
        self._reset_conversation()

    def _run(self):
        while not self._stop_event.is_set():
            job = self._jobs.get()

            if job is None:
                break

            self._current_job = job

            try:
                self._process(job)
            except Exception as exc:
                separate_tool_request = (
                    TOOL_CONTEXT_MODE == "separate"
                    and bool(normalize_tool_definitions(job.request))
                )

                if not separate_tool_request:
                    self._reset_conversation()

                job.result_queue.put(("error", exc))
            finally:
                self._current_job = None

        self._reset_conversation()

    def _process(self, job: InferenceJob):
        has_tools = bool(normalize_tool_definitions(job.request))
        separate_tools = (
            has_tools and TOOL_CONTEXT_MODE == "separate"
        )
        attempts = (
            MALFORMED_TOOL_CALL_RETRIES + 1
            if has_tools
            else 1
        )

        processor = (
            self._process_separate_tool_request
            if separate_tools
            else self._process_persistent_request
        )

        for attempt in range(attempts):
            try:
                processor(job)
                return
            except MalformedToolCallError as exc:
                if not separate_tools:
                    self._reset_conversation()

                if attempt + 1 >= attempts:
                    raise

                with _console_lock:
                    print(
                        "\n[tool-call-retry] "
                        f"Rejected malformed call: {exc} "
                        "Rebuilding the tool context and retrying once.",
                        file=sys.stderr,
                        flush=True,
                    )

    def _process_separate_tool_request(
        self,
        job: InferenceJob,
    ):
        total_tokens = estimate_messages_tokens(job.messages)
        active_index = last_input_index(job.messages)

        if active_index < 0:
            raise HTTPException(
                status_code=400,
                detail="No user or tool message was found.",
            )

        name_map = build_name_by_tool_call_id_map(job.messages)
        initial_messages = build_initial_messages(
            job.messages[:active_index]
        )
        input_message = message_to_litert(
            job.messages[active_index],
            name_map,
        )
        conversation_kwargs = build_conversation_kwargs(
            job.request,
            initial_messages,
        )

        progress = ConsoleProgress(
            total_prompt_tokens=total_tokens,
            prefill_tokens=total_tokens,
            cache_mode="tool-separate",
        )
        progress.start()

        state = GenerationState()
        response_parts: list[str] = []
        reasoning_parts: list[str] = []

        with _console_lock:
            print(
                "\n[tool-context] separate | fresh full-history conversation",
                file=sys.stderr,
                flush=True,
            )

        try:
            with engine.create_conversation(
                **conversation_kwargs
            ) as conversation:
                self._active_conversation = conversation

                for event, payload in generation_events(
                    conversation,
                    input_message,
                    job.request,
                    progress,
                    job.cancel_event,
                    state,
                ):
                    if event == "text":
                        response_parts.append(payload)
                    elif event == "reasoning":
                        reasoning_parts.append(payload.get("text", ""))

                    job.result_queue.put((event, payload))

            response_text = "".join(response_parts)
            reasoning_text = "".join(reasoning_parts)
            finish_reason = (
                "tool_calls"
                if state.tool_calls
                else "stop"
            )
            usage = progress.finish(
                "cancelled"
                if state.cancelled
                else "done"
            )
            usage["cache_mode"] = "tool-separate"
            usage["cached_tokens"] = 0
            usage["prefill_tokens"] = total_tokens

            job.result_queue.put(
                (
                    "done",
                    {
                        "usage": usage,
                        "response_text": response_text,
                        "reasoning_text": reasoning_text,
                        "tool_calls": state.tool_calls,
                        "finish_reason": finish_reason,
                        "cache_mode": "tool-separate",
                    },
                )
            )
        except Exception:
            progress.finish("error")
            raise
        finally:
            self._active_conversation = None

    def _process_persistent_request(
        self,
        job: InferenceJob,
    ):
        config_signature = conversation_config_signature(job.request)
        plan = self._build_plan(
            job.messages,
            config_signature,
        )

        if plan.mode == "replay":
            if self._last_reasoning_text:
                job.result_queue.put(
                    (
                        "reasoning",
                        {
                            "channel": "thinking",
                            "text": self._last_reasoning_text,
                        },
                    )
                )

            if self._last_response_text:
                job.result_queue.put(("text", self._last_response_text))

            if self._last_tool_calls:
                job.result_queue.put(("tool_calls", self._last_tool_calls))

            usage = dict(self._last_usage)
            usage["cache_mode"] = "replay"
            usage["prefill_tokens"] = 0
            usage["cached_tokens"] = plan.total_prompt_tokens

            job.result_queue.put(
                (
                    "done",
                    {
                        "usage": usage,
                        "response_text": self._last_response_text,
                        "reasoning_text": self._last_reasoning_text,
                        "tool_calls": self._last_tool_calls,
                        "finish_reason": self._last_finish_reason,
                        "cache_mode": "replay",
                    },
                )
            )
            return

        if plan.mode == "reset":
            self._reset_conversation()

            conversation_kwargs = build_conversation_kwargs(
                job.request,
                plan.initial_messages or [],
            )
            self._conversation = engine.create_conversation(
                **conversation_kwargs
            )
            self._config_signature = config_signature

        if self._conversation is None:
            raise RuntimeError(
                "Persistent LiteRT conversation was not created."
            )

        has_tools = bool(normalize_tool_definitions(job.request))
        display_cache_mode = (
            f"tool-{plan.mode}"
            if has_tools
            else plan.mode
        )

        if has_tools:
            with _console_lock:
                print(
                    "\n[tool-context] merged | "
                    f"persistent KV | plan={plan.mode} | "
                    f"thinking={normalize_reasoning_effort(job.request)}",
                    file=sys.stderr,
                    flush=True,
                )

        progress = ConsoleProgress(
            total_prompt_tokens=plan.total_prompt_tokens,
            prefill_tokens=plan.prefill_tokens,
            cache_mode=display_cache_mode,
        )
        progress.start()

        state = GenerationState()
        response_parts = []
        reasoning_parts = []

        try:
            for event, payload in generation_events(
                self._conversation,
                plan.input_message,
                job.request,
                progress,
                job.cancel_event,
                state,
            ):
                if event == "text":
                    response_parts.append(payload)
                elif event == "reasoning":
                    reasoning_parts.append(payload.get("text", ""))
                job.result_queue.put((event, payload))

            response_text = "".join(response_parts)
            reasoning_text = "".join(reasoning_parts)
            finish_reason = "tool_calls" if state.tool_calls else "stop"
            usage = progress.finish(
                "cancelled" if state.cancelled else "done"
            )

            response_message: dict[str, Any] = {
                "role": "assistant",
                "content": response_text if response_text else None,
            }

            if reasoning_text:
                response_message["reasoning_content"] = reasoning_text

            if state.tool_calls:
                response_message["tool_calls"] = state.tool_calls

            if state.cancelled:
                self._reset_conversation()
            else:
                self._last_request_messages = job.messages
                self._last_response_text = response_text
                self._last_reasoning_text = reasoning_text
                self._last_tool_calls = state.tool_calls
                self._last_response_message = response_message
                self._last_finish_reason = finish_reason
                self._last_usage = usage
                self._config_signature = config_signature

            job.result_queue.put(
                (
                    "done",
                    {
                        "usage": usage,
                        "response_text": response_text,
                        "reasoning_text": reasoning_text,
                        "tool_calls": state.tool_calls,
                        "finish_reason": finish_reason,
                        "cache_mode": display_cache_mode,
                    },
                )
            )
        except Exception:
            progress.finish("error")
            self._reset_conversation()
            raise

    def _build_plan(
        self,
        messages: list[dict[str, Any]],
        config_signature: str,
    ) -> ConversationPlan:
        total_tokens = estimate_messages_tokens(messages)

        if (
            self._conversation is not None
            and self._config_signature == config_signature
            and messages == self._last_request_messages
        ):
            return ConversationPlan(
                mode="replay",
                total_prompt_tokens=total_tokens,
                prefill_tokens=0,
            )

        if (
            self._conversation is not None
            and self._config_signature == config_signature
            and starts_with_messages(
                messages,
                self._last_request_messages,
            )
        ):
            delta = messages[len(self._last_request_messages) :]

            if delta and delta[0].get("role") == "assistant":
                if not assistant_response_matches(
                    delta[0],
                    self._last_response_message,
                ):
                    delta = []
                else:
                    delta = delta[1:]

            reusable_delta = False

            if delta:
                if all(
                    message.get("role") == "tool"
                    for message in delta
                ):
                    reusable_delta = True
                elif (
                    len(delta) == 1
                    and delta[0].get("role") == "user"
                ):
                    reusable_delta = True

            if reusable_delta:
                input_message = combine_delta_messages(
                    delta,
                    messages,
                    self._last_response_message,
                )

                return ConversationPlan(
                    mode="reuse",
                    input_message=input_message,
                    total_prompt_tokens=total_tokens,
                    prefill_tokens=estimate_input_tokens(input_message),
                )

        active_index = last_input_index(messages)

        if active_index < 0:
            raise HTTPException(
                status_code=400,
                detail="No user or tool message was found.",
            )

        initial_messages = build_initial_messages(
            messages[:active_index]
        )
        name_map = build_name_by_tool_call_id_map(messages)
        input_message = message_to_litert(
            messages[active_index],
            name_map,
        )

        return ConversationPlan(
            mode="reset",
            input_message=input_message,
            initial_messages=initial_messages,
            total_prompt_tokens=total_tokens,
            prefill_tokens=total_tokens,
        )

    def _reset_conversation(self):
        conversation = self._conversation
        self._conversation = None

        if self._active_conversation is conversation:
            self._active_conversation = None

        if conversation is not None:
            try:
                conversation.close()
            except Exception:
                try:
                    conversation.__exit__(None, None, None)
                except Exception:
                    pass

        self._last_request_messages = []
        self._last_response_text = ""
        self._last_reasoning_text = ""
        self._last_tool_calls = []
        self._last_response_message = None
        self._last_finish_reason = "stop"
        self._last_usage = {}
        self._config_signature = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global conversation_worker

    litert_lm.set_min_log_severity(
        litert_lm.LogSeverity.ERROR
    )

    print("Loading LiteRT-LM model with the GPU backend...")
    print(f"Model: {Path(MODEL_PATH).resolve()}")
    print(f"Disk cache: {Path(CACHE_DIR).resolve()}")
    print(f"KV capacity: {MAX_NUM_TOKENS:,} tokens")
    print(f"Default output limit: {DEFAULT_MAX_OUTPUT_TOKENS:,} tokens")
    print(
        "Tool response limit: "
        f"{MAX_TOOL_RESPONSE_TOKENS:,} tokens per result"
    )
    print(
        "Context safety margin: "
        f"{CONTEXT_SAFETY_MARGIN_TOKENS:,} tokens"
    )
    print(
        "Inference watchdog: "
        f"{INFERENCE_TIMEOUT_SECONDS:.0f} seconds"
    )
    print(
        "Malformed tool-call retries: "
        f"{MALFORMED_TOOL_CALL_RETRIES}"
    )
    print(
        "Tool constrained decoding: "
        f"{'enabled' if ENABLE_CONSTRAINED_DECODING else 'disabled'}"
    )
    print(
        "Default reasoning effort: "
        f"{DEFAULT_REASONING_EFFORT}"
    )
    print(
        "Default sampling: "
        f"temperature={DEFAULT_TEMPERATURE}, "
        f"top_p={DEFAULT_TOP_P}, top_k={DEFAULT_TOP_K}"
    )
    print(f"Tool-call temperature: request/default ({DEFAULT_TEMPERATURE})")

    engine_kwargs: dict[str, Any] = {
        "backend": litert_lm.Backend.GPU(),
        "enable_speculative_decoding": False,
        "max_num_tokens": MAX_NUM_TOKENS,
    }

    engine_parameters = set(
        inspect.signature(litert_lm.Engine).parameters
    )

    if "cache_dir" in engine_parameters:
        engine_kwargs["cache_dir"] = CACHE_DIR

    engine = litert_lm.Engine(
        MODEL_PATH,
        **engine_kwargs,
    )
    engine.__enter__()

    conversation_worker = PersistentConversationWorker()
    conversation_worker.start()

    print("Model loaded. Persistent KV cache is enabled for text.")
    print(f"Tool context mode: {TOOL_CONTEXT_MODE}")

    if TOOL_CONTEXT_MODE == "merged":
        print(
            "Tool calls share the live conversation, KV cache, "
            "and thinking state."
        )
    else:
        print(
            "Each tool turn uses a fresh full-history conversation; "
            "tool KV state is discarded afterward."
        )

    print(
        "Malformed tool calls reset only the affected context "
        "before retry."
    )
    print("Generation streaming: asynchronous for every request.")
    print(
        "Thinking channels stream as OpenAI delta.reasoning_content."
    )
    print(
        f"Built-in web chat: "
        f"{'http://localhost:8000/' if WEB_UI_ENABLED else 'disabled'}"
    )

    yield

    print("Shutting down...")

    if conversation_worker is not None:
        conversation_worker.stop()
        conversation_worker = None

    if engine is not None:
        engine.__exit__(None, None, None)
        engine = None


app = FastAPI(
    title="LiteRT-LM Hybrid OpenAI-Compatible Proxy",
    lifespan=lifespan,
)


WEB_CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {
  color-scheme: dark;
  --bg:#0b0d12; --panel:#131722; --panel2:#191f2d;
  --text:#eef2ff; --muted:#9aa6bd; --line:#293146;
  --accent:#7aa2ff; --user:#20365e; --assistant:#171d29;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
main { height:100vh; display:grid; grid-template-rows:auto 1fr auto; }
header { padding:12px 18px; border-bottom:1px solid var(--line);
  background:var(--panel); display:flex; gap:12px; align-items:center;
  flex-wrap:wrap; }
header strong { margin-right:auto; }
label { color:var(--muted); font-size:13px; }
select,input,button,textarea { font:inherit; }
select,input { background:var(--panel2); color:var(--text);
  border:1px solid var(--line); border-radius:7px; padding:6px 8px; }
button { background:var(--accent); color:#071226; border:0;
  border-radius:8px; padding:8px 12px; font-weight:650; cursor:pointer; }
button.secondary { background:var(--panel2); color:var(--text);
  border:1px solid var(--line); }
button:disabled { opacity:.5; cursor:not-allowed; }
#messages { overflow:auto; padding:24px max(18px,calc((100vw - 900px)/2));
  display:flex; flex-direction:column; gap:16px; }
.message { border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; white-space:pre-wrap; overflow-wrap:anywhere; }
.user { background:var(--user); align-self:flex-end; max-width:82%; }
.assistant { background:var(--assistant); align-self:stretch; }
.role { color:var(--muted); font-size:12px; text-transform:uppercase;
  letter-spacing:.08em; margin-bottom:6px; }
.reasoning { margin:0 0 12px; border:1px solid var(--line);
  border-radius:8px; background:#10141d; }
.reasoning summary { cursor:pointer; color:var(--muted); padding:8px 10px; }
.reasoning pre { margin:0; padding:0 10px 10px; white-space:pre-wrap;
  color:#bdc8dc; font:13px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }
.metrics { color:var(--muted); font-size:12px; margin-top:10px; }
form { border-top:1px solid var(--line); background:var(--panel);
  padding:12px max(18px,calc((100vw - 900px)/2)); }
textarea { width:100%; resize:vertical; min-height:74px; max-height:240px;
  background:var(--panel2); color:var(--text); border:1px solid var(--line);
  border-radius:10px; padding:11px; }
.actions { display:flex; justify-content:space-between; margin-top:8px; }
#status { color:var(--muted); font-size:12px; }
.empty { color:var(--muted); text-align:center; margin:auto; }
</style>
</head>
<body>
<main>
<header>
  <strong>{title}</strong>
  <label>Thinking
    <select id="effort">
      <option>none</option><option>low</option><option>medium</option>
      <option selected>high</option><option>xhigh</option>
    </select>
  </label>
  <label>Temperature <input id="temperature" type="number" min="0" max="2"
    step="0.05" value="1.0" style="width:72px"></label>
  <label>Max output <input id="maxTokens" type="number" min="1"
    value="4096" style="width:90px"></label>
  <button id="newChat" class="secondary" type="button">New chat</button>
</header>

<section id="messages">
  <div class="empty" id="empty">No tools, MCP, skills, or AGENTS.md are loaded here.</div>
</section>

<form id="composer">
  <textarea id="prompt" placeholder="Message the local model…" required></textarea>
  <div class="actions">
    <span id="status">Ready</span>
    <div>
      <button id="stop" class="secondary" type="button" disabled>Stop</button>
      <button id="send" type="submit">Send</button>
    </div>
  </div>
</form>
</main>

<script>
const messages = [];
const list = document.getElementById("messages");
const empty = document.getElementById("empty");
const form = document.getElementById("composer");
const promptBox = document.getElementById("prompt");
const status = document.getElementById("status");
const sendButton = document.getElementById("send");
const stopButton = document.getElementById("stop");
let controller = null;

function addMessage(role, text="") {
  empty.style.display = "none";
  const box = document.createElement("article");
  box.className = `message ${role}`;
  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role;
  const reasoning = document.createElement("details");
  reasoning.className = "reasoning";
  reasoning.open = true;
  reasoning.style.display = "none";
  const summary = document.createElement("summary");
  summary.textContent = "Thinking";
  const reasoningText = document.createElement("pre");
  reasoning.append(summary, reasoningText);
  const content = document.createElement("div");
  content.textContent = text;
  const metrics = document.createElement("div");
  metrics.className = "metrics";
  box.append(roleEl, reasoning, content, metrics);
  list.appendChild(box);
  list.scrollTop = list.scrollHeight;
  return {box, content, reasoning, reasoningText, metrics};
}

function parseEvent(block) {
  const lines = block.split(/\r?\n/);
  const data = lines.filter(x => x.startsWith("data:"))
    .map(x => x.slice(5).trim()).join("\n");
  return data;
}

async function sendMessage(text) {
  const user = {role:"user", content:text};
  messages.push(user);
  addMessage("user", text);
  const assistant = addMessage("assistant");
  controller = new AbortController();
  sendButton.disabled = true;
  stopButton.disabled = false;
  status.textContent = "Generating…";

  const body = {
    model: "local-litert",
    stream: true,
    include_reasoning: true,
    reasoning_effort: document.getElementById("effort").value,
    temperature: Number(document.getElementById("temperature").value),
    max_completion_tokens: Number(document.getElementById("maxTokens").value),
    messages
  };

  let answer = "";
  let thinking = "";

  try {
    const response = await fetch("/v1/chat/completions", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body),
      signal:controller.signal
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = parseEvent(block);
        if (!data || data === "[DONE]") continue;

        const event = JSON.parse(data);
        if (event.error) throw new Error(event.error.message);

        const delta = event.choices?.[0]?.delta || {};
        if (delta.reasoning_content) {
          thinking += delta.reasoning_content;
          assistant.reasoning.style.display = "";
          assistant.reasoningText.textContent = thinking;
        }
        if (delta.content) {
          answer += delta.content;
          assistant.content.textContent = answer;
        }
        if (event.usage) {
          const u = event.usage;
          const fields = [];
          if (u.time_to_first_token != null)
            fields.push(`TTFT ${u.time_to_first_token.toFixed(2)}s`);
          if (u.prefill_tokens_per_second != null)
            fields.push(`PPS ${u.prefill_tokens_per_second.toFixed(1)}`);
          if (u.decode_tokens_per_second != null)
            fields.push(`TPS ${u.decode_tokens_per_second.toFixed(1)}`);
          if (u.reasoning_tokens != null)
            fields.push(`thinking ${u.reasoning_tokens} tok`);
          assistant.metrics.textContent = fields.join(" · ");
        }
        list.scrollTop = list.scrollHeight;
      }
    }

    if (!answer && !thinking) throw new Error("The model returned no content.");
    messages.push({role:"assistant", content:answer || ""});
    status.textContent = "Ready";
  } catch (error) {
    if (error.name === "AbortError") {
      status.textContent = "Stopped";
    } else {
      assistant.content.textContent = `Error: ${error.message}`;
      status.textContent = "Error";
    }
  } finally {
    controller = null;
    sendButton.disabled = false;
    stopButton.disabled = true;
    promptBox.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const text = promptBox.value.trim();
  if (!text || controller) return;
  promptBox.value = "";
  sendMessage(text);
});

promptBox.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

stopButton.addEventListener("click", () => controller?.abort());

document.getElementById("newChat").addEventListener("click", async () => {
  controller?.abort();
  await fetch("/v1/conversation/reset", {method:"POST"});
  messages.length = 0;
  list.replaceChildren(empty);
  empty.style.display = "";
  status.textContent = "New conversation";
  promptBox.focus();
});
</script>
</body>
</html>"""

WEB_CHAT_HTML = WEB_CHAT_HTML.replace("{title}", WEB_UI_TITLE)


@app.get("/", response_class=HTMLResponse)
async def web_chat():
    if not WEB_UI_ENABLED:
        raise HTTPException(
            status_code=404,
            detail="The built-in web UI is disabled.",
        )

    return HTMLResponse(WEB_CHAT_HTML)


@app.get("/chat", response_class=HTMLResponse)
async def web_chat_alias():
    return await web_chat()


@app.post("/v1/conversation/reset")
async def reset_conversation():
    if conversation_worker is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Cannot reset while generation is active.",
        )

    try:
        conversation_worker.reset()
        return {"status": "ok"}
    finally:
        _request_lock.release()


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine is not None else "loading",
        "backend": "gpu",
        "model": MODEL_PATH,
        "max_num_tokens": MAX_NUM_TOKENS,
        "default_max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "max_tool_response_tokens": MAX_TOOL_RESPONSE_TOKENS,
        "context_safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
        "cache_dir": CACHE_DIR,
        "constrained_decoding": ENABLE_CONSTRAINED_DECODING,
        "tool_temperature": DEFAULT_TEMPERATURE,
        "tool_context_mode": TOOL_CONTEXT_MODE,
        "tool_request_mode": (
            "persistent-kv-chain"
            if TOOL_CONTEXT_MODE == "merged"
            else "separate-full-history"
        ),
        "text_request_mode": "persistent-kv",
        "generation_streaming": "async",
        "reasoning_stream_field": "choices[0].delta.reasoning_content",
        "web_ui_enabled": WEB_UI_ENABLED,
        "web_ui_path": "/" if WEB_UI_ENABLED else None,
        "malformed_tool_call_retries": MALFORMED_TOOL_CALL_RETRIES,
        "max_tool_argument_string_length": (
            MAX_TOOL_ARGUMENT_STRING_LENGTH
        ),
        "conversation": (
            conversation_worker.status()
            if conversation_worker is not None
            else None
        ),
    }


@app.get("/v1/models")
async def list_models():
    model_id = Path(MODEL_PATH).stem
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-litert",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
):
    if engine is None or conversation_worker is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty.",
        )

    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="The model is already processing another request.",
        )

    try:
        normalized_messages = canonical_messages(
            request.messages
        )
        budget = estimate_context_budget(
            request,
            normalized_messages,
        )

        with _console_lock:
            print(
                "\n[context-budget] "
                f"messages≈{budget['message_tokens']:,} "
                f"tools≈{budget['tool_schema_tokens']:,} "
                f"output={budget['output_tokens']:,} "
                f"margin={budget['safety_margin_tokens']:,} "
                f"projected≈{budget['projected_tokens']:,}/"
                f"{MAX_NUM_TOKENS:,}",
                file=sys.stderr,
                flush=True,
            )

        if budget["projected_tokens"] > MAX_NUM_TOKENS:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Context window would be exceeded: "
                    f"messages≈{budget['message_tokens']:,}, "
                    f"tool schemas≈{budget['tool_schema_tokens']:,}, "
                    f"output reserve={budget['output_tokens']:,}, "
                    f"safety margin={budget['safety_margin_tokens']:,}, "
                    f"projected≈{budget['projected_tokens']:,}, "
                    f"limit={MAX_NUM_TOKENS:,}. "
                    "Compact or start a new Pi session, reduce tool output, "
                    "or increase LITERT_MAX_NUM_TOKENS."
                ),
            )

        model_name = request.model or MODEL_PATH
        result_queue: queue.Queue = queue.Queue()
        job = InferenceJob(
            request=request,
            messages=normalized_messages,
            result_queue=result_queue,
        )
        conversation_worker.submit(job)
    except Exception:
        _request_lock.release()
        raise

    if request.stream:
        def stream_generator():
            completed = False

            try:
                yield make_initial_stream_chunk(model_name)

                while True:
                    try:
                        event, payload = result_queue.get(
                            timeout=SSE_HEARTBEAT_SECONDS
                        )
                    except queue.Empty:
                        yield (
                            f": keep-alive {int(time.time())}\n\n"
                        )
                        continue

                    if event == "text":
                        yield make_stream_chunk(
                            model_name,
                            payload,
                        )
                        continue

                    if event == "reasoning":
                        yield make_stream_reasoning_chunk(
                            model_name,
                            payload.get("text", ""),
                            payload.get("channel", "thinking"),
                        )
                        continue

                    if event == "tool_calls":
                        yield make_stream_tool_calls_chunk(
                            model_name,
                            payload,
                        )
                        continue

                    if event == "done":
                        completed = True
                        finish_reason = (
                            payload.get("finish_reason", "stop")
                            if isinstance(payload, dict)
                            else "stop"
                        )
                        usage = (
                            payload.get("usage")
                            if isinstance(payload, dict)
                            else None
                        )
                        yield make_stream_chunk(
                            model_name,
                            "",
                            finish_reason,
                            usage=usage,
                        )
                        yield "data: [DONE]\n\n"
                        break

                    if event == "error":
                        completed = True
                        yield make_error_stream_chunk(payload)
                        yield "data: [DONE]\n\n"
                        break
            finally:
                if not completed:
                    job.cancel_event.set()
                    conversation_worker.cancel_current()

                _request_lock.release()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response_parts = []
        reasoning_parts = []
        response_tool_calls: list[dict[str, Any]] = []
        result_payload = None

        while True:
            event, payload = await asyncio.to_thread(
                result_queue.get
            )

            if event == "text":
                response_parts.append(payload)
                continue

            if event == "reasoning":
                reasoning_parts.append(payload.get("text", ""))
                continue

            if event == "tool_calls":
                response_tool_calls.extend(payload)
                continue

            if event == "done":
                result_payload = payload
                break

            if event == "error":
                raise payload

        response_text = "".join(response_parts)
        reasoning_text = "".join(reasoning_parts)
        usage = (
            result_payload.get("usage", {})
            if isinstance(result_payload, dict)
            else {}
        )
        finish_reason = (
            result_payload.get("finish_reason", "stop")
            if isinstance(result_payload, dict)
            else "stop"
        )

        response_message: dict[str, Any] = {
            "role": "assistant",
            "content": response_text if response_text else None,
        }

        if reasoning_text:
            response_message["reasoning_content"] = reasoning_text

        if response_tool_calls:
            response_message["tool_calls"] = response_tool_calls

        return {
            "id": "chatcmpl-litert",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get(
                    "prompt_tokens",
                    estimate_messages_tokens(
                        normalized_messages
                    ),
                ),
                "completion_tokens": usage.get(
                    "completion_tokens",
                    count_tokens(response_text),
                ),
                "total_tokens": usage.get(
                    "total_tokens",
                    estimate_messages_tokens(
                        normalized_messages
                    )
                    + count_tokens(response_text),
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
    finally:
        _request_lock.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI-compatible LiteRT-LM server with selectable "
            "tool-context handling."
        )
    )
    parser.add_argument(
        "--tool-context-mode",
        choices=sorted(TOOL_CONTEXT_MODES),
        default=TOOL_CONTEXT_MODE,
        help=(
            "merged keeps one live KV cache across tool calls; "
            "separate rebuilds a fresh full-history context for "
            "each tool turn. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--no-web-ui",
        action="store_true",
        help="Disable the built-in browser chat interface.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TOOL_CONTEXT_MODE = args.tool_context_mode
    WEB_UI_ENABLED = WEB_UI_ENABLED and not args.no_web_ui

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
