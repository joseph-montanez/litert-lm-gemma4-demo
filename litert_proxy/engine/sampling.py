import inspect
import json
import sys
from typing import Any, Optional

import litert_lm
from fastapi import HTTPException

from ..config import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_REPETITION_WINDOW,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ENABLE_CONSTRAINED_DECODING,
    THINKING_TOKEN_BUDGETS,
    _console_lock,
)
from ..models import ChatCompletionRequest, ProxyTool
from ..utils.token import json_safe, requested_output_tokens
from ..utils.tools import normalize_tool_definitions
from ..workspace_tools import (
    build_workspace_tools,
    resolve_workspace_root,
    WorkspaceToolEventHandler,
)


# ---------------------------------------------------------------------------
# Sampler config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reasoning / thinking config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Conversation config signature (used for KV-cache reuse decisions)
# ---------------------------------------------------------------------------


def conversation_config_signature(
    request: ChatCompletionRequest,
) -> str:
    from ..config import MODEL_PATH

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
        "workspace_tools": request.workspace_tools,
        "workspace_path": request.workspace_path,
        "workspace_read_only": request.workspace_read_only,
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


# ---------------------------------------------------------------------------
# Conversation & send_message_async kwargs builders
# ---------------------------------------------------------------------------


def build_conversation_kwargs(
    request: ChatCompletionRequest,
    initial_messages: list[Any],
) -> dict[str, Any]:
    from .. import config as _sampling_cfg

    parameters = set(inspect.signature(_sampling_cfg.engine.create_conversation).parameters)
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

    native_workspace_tools = []
    if request.workspace_tools:
        root = resolve_workspace_root(request.workspace_path or "")
        native_workspace_tools = build_workspace_tools(
            root,
            read_only=request.workspace_read_only,
        )
        with _console_lock:
            print(
                "\n[workspace-tools] "
                f"root={root} "
                f"mode={'read-only' if request.workspace_read_only else 'read-write'} "
                f"tools={','.join(tool.name for tool in native_workspace_tools)}",
                file=sys.stderr,
                flush=True,
            )

    if "tools" in parameters:
        if native_workspace_tools:
            kwargs["tools"] = native_workspace_tools
        elif tool_definitions:
            kwargs["tools"] = [
                ProxyTool(definition)
                for definition in tool_definitions
            ]
        else:
            kwargs["tools"] = None

    if "automatic_tool_calling" in parameters:
        kwargs["automatic_tool_calling"] = bool(native_workspace_tools)

    if native_workspace_tools and "tool_event_handler" in parameters:
        kwargs["tool_event_handler"] = WorkspaceToolEventHandler()

    if (
        (tool_definitions or native_workspace_tools)
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

    if "filter_channel_content_from_kv_cache" in parameters:
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
