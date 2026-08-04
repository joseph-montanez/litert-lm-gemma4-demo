import asyncio
import inspect
import os
import queue
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import litert_lm
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from . import config
from .config import (
    MODEL_PATH,
    MODEL_REGISTRY,
    SSE_HEARTBEAT_SECONDS,
    WEB_UI_TITLE,
    _console_lock,
    _request_lock,
    engine,
    conversation_worker,
    ensure_model,
    resolve_backend,
)
from .models import ChatCompletionRequest, InferenceJob


tool_engine = config.tool_engine
tool_conversation_worker = config.tool_conversation_worker


def _primary_is_tool_model() -> bool:
    return bool(
        config.CURRENT_MODEL_KEY
        and config.CURRENT_MODEL_KEY == config.TOOL_MODEL_KEY
    )


def _workspace_tooling_enabled_for_primary() -> bool:
    return bool(
        config.CURRENT_MODEL_KEY != "gemma-4-12B-it"
        or config.ENABLE_12B_TOOLS
    )


def _apply_model_engine_options(
    engine_kwargs: dict[str, Any],
    engine_parameters: set[str],
    model_key: str | None,
    backend: Any,
) -> None:
    """Apply model-specific LiteRT engine compatibility options."""
    model_profile = MODEL_REGISTRY.get(model_key, {}) if model_key else {}
    if (
        model_profile.get("enable_benchmark")
        and type(backend).__name__.upper() == "GPU"
        and "enable_benchmark" in engine_parameters
    ):
        # LiteRT's benchmark prefill explicitly waits for GPU completion. That
        # selects its retrying readback path instead of aborting when a large
        # uncached prefill exceeds WebGPU's first 20-second wait interval.
        engine_kwargs["enable_benchmark"] = True


class AdminReconfigureRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_num_tokens: Optional[int] = None
    default_max_output_tokens: Optional[int] = None
    max_tool_response_tokens: Optional[int] = None
    context_safety_margin_tokens: Optional[int] = None
    inference_timeout_seconds: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    reasoning_effort: Optional[str] = None
    repetition_penalty: Optional[float] = None
    constrained_decoding: Optional[bool] = None
    speculative_decoding: Optional[bool] = None
    backend: Optional[str] = None
    malformed_tool_call_retries: Optional[int] = None
    max_tool_argument_string_length: Optional[int] = None
    max_tool_calls_per_generation: Optional[int] = None


class ShellApprovalDecision(BaseModel):
    approved: bool


from .utils.message import canonical_messages
from .utils.tools import normalize_tool_definitions
from .utils.token import (
    count_tokens,
    estimate_context_budget,
    estimate_messages_tokens,
)
from .workspace_tools import (
    resolve_workspace_root,
    shell_approval_broker,
)
from .engine.streaming import (
    make_error_stream_chunk,
    make_initial_stream_chunk,
    make_stream_chunk,
    make_stream_reasoning_chunk,
    make_stream_tool_activity_chunk,
    make_stream_tool_calls_chunk,
)


# ---------------------------------------------------------------------------
# Web chat HTML (loaded once at import time)
# ---------------------------------------------------------------------------

_web_chat_html: str = ""
_highlight_js_path = (
    Path(__file__).resolve().parent
    / "vendor"
    / "highlightjs"
    / "highlight.min.js"
)

def _load_web_chat_html() -> str:
    global _web_chat_html
    if _web_chat_html:
        return _web_chat_html

    html_path = Path(__file__).resolve().parent / "templates" / "web_chat.html"
    try:
        _web_chat_html = html_path.read_text(encoding="utf-8")
    except Exception:
        _web_chat_html = "<!doctype html><title>Error</title><p>web_chat.html not found.</p>"

    _web_chat_html = _web_chat_html.replace("{title}", WEB_UI_TITLE)
    return _web_chat_html


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def _initialize_model_runtime():
    global engine
    global conversation_worker
    global tool_engine
    global tool_conversation_worker

    litert_lm.set_min_log_severity(
        litert_lm.LogSeverity.ERROR
    )

    # Determine model key from MODEL_PATH for registry matching.
    resolved_model_path = Path(MODEL_PATH).resolve()
    model_key: str | None = None
    explicit_path = os.environ.get("LITERT_MODEL_PATH", "").strip()
    for key, entry in MODEL_REGISTRY.items():
        if resolved_model_path.name == entry["filename"] or resolved_model_path.name == f"{key}.litertlm":
            model_key = key
            break

    if explicit_path and model_key:
        # User set explicit path that matches a registry filename.
        # Use user's path but still track the model key for display.
        resolved_model_path = ensure_model(None)
        config.MODEL_PATH = resolved_model_path
        config.CURRENT_MODEL_KEY = model_key

    else:
        # No explicit path or no registry match — use registry if key found.
        resolved_model_path = ensure_model(model_key)
        config.MODEL_PATH = resolved_model_path
        config.CURRENT_MODEL_KEY = model_key

    config.apply_model_runtime_defaults(model_key)

    cache_dir = os.environ.get(
        "LITERT_CACHE_DIR",
        str(Path(resolved_model_path).parent),
    )

    backend = resolve_backend(model_key)
    backend_name = type(backend).__name__

    print(f"Loading LiteRT-LM model with the {backend_name} backend...")
    print(f"Model: {Path(resolved_model_path).resolve()}")
    if model_key:
        print(f"Model key: {model_key}")
    print(f"Disk cache: {Path(cache_dir).resolve()}")
    effective_max_num_tokens = _effective_max_num_tokens(model_key)
    print(f"KV capacity: {effective_max_num_tokens:,} tokens")
    print(
        "Default output limit: "
        f"{config.DEFAULT_MAX_OUTPUT_TOKENS:,} tokens"
    )
    print(
        "Tool response limit: "
        f"{config.MAX_TOOL_RESPONSE_TOKENS:,} tokens per result"
    )
    print(
        "Context safety margin: "
        f"{config.CONTEXT_SAFETY_MARGIN_TOKENS:,} tokens"
    )
    print(
        "Inference watchdog: "
        f"{config.INFERENCE_TIMEOUT_SECONDS:.0f} seconds"
    )
    print(
        "Malformed tool-call retries: "
        f"{config.MALFORMED_TOOL_CALL_RETRIES}"
    )
    print(
        "Max workspace tool calls per generation: "
        f"{config.MAX_TOOL_CALLS_PER_GENERATION}"
    )
    print(
        "Tool constrained decoding: "
        f"{'enabled' if config.ENABLE_CONSTRAINED_DECODING else 'disabled'}"
    )
    print(
        "Default reasoning effort: "
        f"{config.DEFAULT_REASONING_EFFORT}"
    )
    print(
        "Default sampling: "
        f"temperature={config.DEFAULT_TEMPERATURE}, "
        f"top_p={config.DEFAULT_TOP_P}, top_k={config.DEFAULT_TOP_K}"
    )
    print(
        "Tool-call temperature: request/default "
        f"({config.DEFAULT_TEMPERATURE})"
    )

    engine_kwargs: dict[str, Any] = {
        "backend": backend,
        "enable_speculative_decoding": config.ENABLE_SPECULATIVE_DECODING,
        "max_num_tokens": effective_max_num_tokens,
    }

    engine_parameters = set(
        inspect.signature(litert_lm.Engine).parameters
    )

    if "cache_dir" in engine_parameters:
        engine_kwargs["cache_dir"] = cache_dir

    _apply_model_engine_options(
        engine_kwargs,
        engine_parameters,
        model_key,
        backend,
    )

    config.engine = litert_lm.Engine(
        resolved_model_path,
        **engine_kwargs,
    )
    config.engine.__enter__()
    engine = config.engine

    from .engine.worker import PersistentConversationWorker

    conversation_worker = PersistentConversationWorker(
        config.engine,
        name="primary",
    )
    conversation_worker.start()
    config.conversation_worker = conversation_worker

    print("Model loaded. Persistent KV cache is enabled for text.")
    print(f"Tool context mode: {config.TOOL_CONTEXT_MODE}")

    if config.TOOL_CONTEXT_MODE == "merged":
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
    print(
        "Generation streaming: asynchronous for plain chat; "
        "native workspace tools use blocking automatic execution."
    )
    print(
        "Thinking channels stream as OpenAI delta.reasoning_content."
    )

    if not _workspace_tooling_enabled_for_primary():
        tool_engine = None
        tool_conversation_worker = None
        config.tool_engine = None
        config.tool_conversation_worker = None
        config.TOOL_MODEL_LOAD_ERROR = None
        print(
            "Native workspace tooling: disabled for Gemma 4 12B. Set "
            "LITERT_ENABLE_12B_TOOLS=1 to override."
        )
        print(
            "Client-supplied OpenAI function tools remain available on "
            "the primary 12B model."
        )
    elif config.TOOL_ROUTING_ENABLED and _primary_is_tool_model():
        tool_engine = None
        tool_conversation_worker = None
        config.tool_engine = None
        config.tool_conversation_worker = None
        config.TOOL_MODEL_LOAD_ERROR = None
        print(
            "Tool-model routing reuses the primary model; "
            "no second engine was loaded."
        )
    elif config.TOOL_ROUTING_ENABLED:
        try:
            tool_engine, tool_conversation_worker = _create_tool_runtime()
            config.tool_engine = tool_engine
            config.tool_conversation_worker = tool_conversation_worker
            config.TOOL_MODEL_LOAD_ERROR = None
        except Exception as exc:
            config.TOOL_MODEL_LOAD_ERROR = str(exc)
            tool_engine = None
            tool_conversation_worker = None
            config.tool_engine = None
            config.tool_conversation_worker = None
            print(
                "Tool-model routing is unavailable: "
                f"{config.TOOL_MODEL_LOAD_ERROR}",
                file=sys.stderr,
                flush=True,
            )
    else:
        print("Tool-model routing: disabled")

    print(
        f"Built-in web chat: "
        f"{'http://localhost:8000/' if config.WEB_UI_ENABLED else 'disabled'}"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global conversation_worker
    global tool_engine
    global tool_conversation_worker

    config.MODEL_LOADING = True
    config.MODEL_LOAD_ERROR = None
    _request_lock.acquire()

    async def load_model():
        try:
            await asyncio.to_thread(_initialize_model_runtime)
        except Exception as exc:
            config.MODEL_LOAD_ERROR = str(exc)
            with _console_lock:
                print(
                    f"\n[model-load-error] {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            config.MODEL_LOADING = False
            _request_lock.release()

    load_task = asyncio.create_task(load_model())

    yield

    print("Shutting down...")
    await load_task

    if conversation_worker is not None:
        conversation_worker.stop()
        conversation_worker = None
        config.conversation_worker = None

    if tool_conversation_worker is not None:
        tool_conversation_worker.stop()
        tool_conversation_worker = None
        config.tool_conversation_worker = None

    if engine is not None:
        config.engine.__exit__(None, None, None)
        config.engine = None
        engine = None

    if tool_engine is not None:
        tool_engine.__exit__(None, None, None)
        tool_engine = None
        config.tool_engine = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LiteRT-LM Hybrid OpenAI-Compatible Proxy",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def web_chat():
    if not config.WEB_UI_ENABLED:
        raise HTTPException(
            status_code=404,
            detail="The built-in web UI is disabled.",
        )

    return HTMLResponse(_load_web_chat_html())


@app.get("/chat", response_class=HTMLResponse)
async def web_chat_alias():
    return await web_chat()


@app.get(
    "/assets/highlight.min.js",
    response_class=FileResponse,
    include_in_schema=False,
)
async def highlight_javascript():
    if not config.WEB_UI_ENABLED or not _highlight_js_path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")

    return FileResponse(
        _highlight_js_path,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.post("/v1/conversation/reset")
async def reset_conversation():
    workers = [
        worker
        for worker in (conversation_worker, tool_conversation_worker)
        if worker is not None
    ]
    if not workers:
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
        for worker in workers:
            worker.reset()
        return {"status": "ok"}
    finally:
        _request_lock.release()


@app.post("/v1/conversation/cancel")
async def cancel_conversation():
    workers = [
        worker
        for worker in (conversation_worker, tool_conversation_worker)
        if worker is not None
    ]
    if not workers:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    cancelled = False
    for worker in workers:
        cancelled = worker.cancel_current() or cancelled
    return {
        "status": "cancelling" if cancelled else "idle",
    }


@app.get("/v1/workspace/shell-approvals/{approval_session_id}")
async def get_shell_approval(approval_session_id: str):
    """Return the command currently waiting on this approval session."""
    return {
        "pending": shell_approval_broker.get_pending(
            approval_session_id
        )
    }


@app.post(
    "/v1/workspace/shell-approvals/"
    "{approval_session_id}/{call_id}"
)
async def decide_shell_approval(
    approval_session_id: str,
    call_id: str,
    body: ShellApprovalDecision,
):
    resolved = shell_approval_broker.resolve(
        approval_session_id,
        call_id,
        body.approved,
    )
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="This shell approval is no longer pending.",
        )
    return {"status": "approved" if body.approved else "denied"}


@app.get("/health")
async def health():
    model_key = config.CURRENT_MODEL_KEY
    model_display = None
    if model_key and model_key in MODEL_REGISTRY:
        model_display = MODEL_REGISTRY[model_key]["display"]

    workspace_tooling_enabled = _workspace_tooling_enabled_for_primary()
    tool_uses_primary = bool(
        workspace_tooling_enabled
        and config.TOOL_ROUTING_ENABLED
        and _primary_is_tool_model()
    )
    routed_tool_worker = (
        conversation_worker
        if tool_uses_primary
        else tool_conversation_worker
    )

    return {
        "status": (
            "error"
            if config.MODEL_LOAD_ERROR
            else "ok"
            if engine is not None
            else "loading"
        ),
        "model_loading": config.MODEL_LOADING,
        "model_load_error": config.MODEL_LOAD_ERROR,
        "backend": config.DEFAULT_BACKEND,
        "model": config.MODEL_PATH,
        "model_key": model_key,
        "model_display": model_display,
        # Kept for older web clients; this means native workspace tooling.
        "tooling_enabled": workspace_tooling_enabled,
        "workspace_tooling_enabled": workspace_tooling_enabled,
        "client_tooling_enabled": True,
        "tooling_disabled_reason": (
            None
            if workspace_tooling_enabled
            else "Native workspace tooling is disabled for Gemma 4 12B."
        ),
        "max_num_tokens": _effective_max_num_tokens(model_key),
        "default_max_output_tokens": _effective_default_max_output_tokens(model_key),
        "max_tool_response_tokens": config.MAX_TOOL_RESPONSE_TOKENS,
        "context_safety_margin_tokens": config.CONTEXT_SAFETY_MARGIN_TOKENS,
        "inference_timeout_seconds": config.INFERENCE_TIMEOUT_SECONDS,
        "cache_dir": os.environ.get(
            "LITERT_CACHE_DIR",
            str(Path(config.MODEL_PATH).parent),
        ),
        "constrained_decoding": config.ENABLE_CONSTRAINED_DECODING,
        "tool_temperature": config.DEFAULT_TEMPERATURE,
        "tool_context_mode": config.TOOL_CONTEXT_MODE,
        "tool_request_mode": (
            "persistent-kv-chain"
            if config.TOOL_CONTEXT_MODE == "merged"
            else "separate-full-history"
        ),
        "text_request_mode": "persistent-kv",
        "generation_streaming": "async",
        "reasoning_stream_field": "choices[0].delta.reasoning_content",
        "web_ui_enabled": config.WEB_UI_ENABLED,
        "web_ui_path": "/" if config.WEB_UI_ENABLED else None,
        "malformed_tool_call_retries": config.MALFORMED_TOOL_CALL_RETRIES,
        "max_tool_argument_string_length": (
            config.MAX_TOOL_ARGUMENT_STRING_LENGTH
        ),
        "conversation": (
            conversation_worker.status()
            if conversation_worker is not None
            else None
        ),
        "tool_routing": {
            "enabled": bool(
                workspace_tooling_enabled and config.TOOL_ROUTING_ENABLED
            ),
            "model_key": config.TOOL_MODEL_KEY,
            "model": (
                config.MODEL_PATH
                if tool_uses_primary
                else config.TOOL_MODEL_PATH
            ),
            "backend": (
                config.DEFAULT_BACKEND
                if tool_uses_primary
                else config.TOOL_BACKEND
            ),
            "max_num_tokens": (
                _effective_max_num_tokens(config.CURRENT_MODEL_KEY)
                if tool_uses_primary
                else _effective_tool_max_num_tokens()
            ),
            "reasoning_effort": config.TOOL_REASONING_EFFORT,
            "loaded": (
                bool(
                    workspace_tooling_enabled
                    and (
                        engine is not None
                        if tool_uses_primary
                        else tool_engine is not None
                    )
                )
            ),
            "uses_primary": tool_uses_primary,
            "load_error": config.TOOL_MODEL_LOAD_ERROR,
            "routes": (
                []
                if not workspace_tooling_enabled
                else ["workspace_tools", "openai_tools"]
                if config.TOOL_ROUTE_OPENAI_TOOLS
                else ["workspace_tools"]
            ),
            "workspace_generation": "blocking-automatic-tools",
            "conversation": (
                routed_tool_worker.status()
                if workspace_tooling_enabled and routed_tool_worker is not None
                else None
            ),
        },
    }


@app.get("/v1/models")
async def list_models():
    model_id = Path(config.MODEL_PATH).stem
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
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty.",
        )

    has_openai_tools = bool(normalize_tool_definitions(request))
    if (
        request.workspace_tools
        and not _workspace_tooling_enabled_for_primary()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Native server-side workspace tooling is disabled by "
                "default for Gemma 4 12B. Client-supplied OpenAI function "
                "tools remain available. "
                "Set LITERT_ENABLE_12B_TOOLS=1 and restart to override."
            ),
        )

    if request.workspace_tools:
        if normalize_tool_definitions(request):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Native workspace tools cannot be combined with "
                    "client-supplied OpenAI tools in the same request."
                ),
            )
        try:
            workspace_root = resolve_workspace_root(
                request.workspace_path or ""
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        request.workspace_path = str(workspace_root)
        if request.workspace_shell_approval_id is not None:
            approval_id = request.workspace_shell_approval_id.strip()
            if (
                not approval_id
                or len(approval_id) > 128
                or not all(
                    character.isalnum() or character in "-_"
                    for character in approval_id
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid workspace_shell_approval_id.",
                )
            request.workspace_shell_approval_id = approval_id

    route_to_tool_model = bool(
        config.TOOL_ROUTING_ENABLED
        and _workspace_tooling_enabled_for_primary()
        and (
            request.workspace_tools
            or (
                has_openai_tools
                and config.TOOL_ROUTE_OPENAI_TOOLS
            )
        )
    )
    tool_uses_primary = route_to_tool_model and _primary_is_tool_model()
    selected_engine = (
        engine
        if tool_uses_primary or not route_to_tool_model
        else tool_engine
    )
    selected_worker = (
        conversation_worker
        if tool_uses_primary or not route_to_tool_model
        else tool_conversation_worker
    )
    selected_model_key = (
        config.TOOL_MODEL_KEY
        if route_to_tool_model
        else config.CURRENT_MODEL_KEY
    )

    if route_to_tool_model:
        # The E4B CPU bundle is reliable at automatic native tool execution
        # with thinking disabled. At high reasoning it can exhaust a turn in
        # the thought channel or stop after only the <|tool_call> marker.
        request = request.model_copy(
            update={
                "reasoning_effort": config.TOOL_REASONING_EFFORT,
            }
        )

    if selected_engine is None or selected_worker is None:
        detail = (
            "Tool model is not loaded: "
            f"{config.TOOL_MODEL_LOAD_ERROR or 'unknown error'}"
            if route_to_tool_model
            else "Primary model is not loaded."
        )
        raise HTTPException(status_code=503, detail=detail)

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

        effective_max_num_tokens = (
            _effective_tool_max_num_tokens()
            if route_to_tool_model and not tool_uses_primary
            else _effective_max_num_tokens(selected_model_key)
        )
        with _console_lock:
            print(
                "\n[model-route] "
                f"{'tools' if route_to_tool_model else 'plain'} -> "
                f"{selected_model_key or config.MODEL_PATH} "
                f"({config.TOOL_BACKEND if route_to_tool_model and not tool_uses_primary else config.DEFAULT_BACKEND})"
                f"{' reasoning=' + config.TOOL_REASONING_EFFORT if route_to_tool_model else ''}\n"
                "\n[context-budget] "
                f"messages≈{budget['message_tokens']:,} "
                f"tools≈{budget['tool_schema_tokens']:,} "
                f"output={budget['output_tokens']:,} "
                f"margin={budget['safety_margin_tokens']:,} "
                f"projected≈{budget['projected_tokens']:,}/"
                f"{effective_max_num_tokens:,}",
                file=sys.stderr,
                flush=True,
            )

        if budget["projected_tokens"] > effective_max_num_tokens:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Context window would be exceeded: "
                    f"messages≈{budget['message_tokens']:,}, "
                    f"tool schemas≈{budget['tool_schema_tokens']:,}, "
                    f"output reserve={budget['output_tokens']:,}, "
                    f"safety margin={budget['safety_margin_tokens']:,}, "
                    f"projected≈{budget['projected_tokens']:,}, "
                    f"limit={effective_max_num_tokens:,}. "
                    "Compact or start a new Pi session, reduce tool output, "
                    "or increase LITERT_MAX_NUM_TOKENS."
                ),
            )

        model_name = (
            selected_model_key
            or request.model
            or config.MODEL_PATH
        )
        result_queue: queue.Queue = queue.Queue()
        job = InferenceJob(
            request=request,
            messages=normalized_messages,
            result_queue=result_queue,
        )
        selected_worker.submit(job)
    except Exception:
        _request_lock.release()
        raise

    if request.stream:
        async def stream_generator():
            completed = False

            try:
                yield make_initial_stream_chunk(model_name)

                while True:
                    try:
                        event, payload = await asyncio.to_thread(
                            result_queue.get,
                            True,
                            SSE_HEARTBEAT_SECONDS,
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

                    if event == "tool_activity":
                        yield make_stream_tool_activity_chunk(
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
                if request.workspace_shell_approval_id:
                    shell_approval_broker.deny_pending(
                        request.workspace_shell_approval_id
                    )
                if not completed:
                    job.cancel_event.set()
                    selected_worker.cancel_current()

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
        if request.workspace_shell_approval_id:
            shell_approval_broker.deny_pending(
                request.workspace_shell_approval_id
            )
        _request_lock.release()


# ---------------------------------------------------------------------------
# Engine lifecycle helper
# ---------------------------------------------------------------------------


def _effective_max_num_tokens(model_key: str | None = None) -> int:
    """Return the configured context size capped by the model bundle."""
    key = model_key if model_key is not None else config.CURRENT_MODEL_KEY
    entry = MODEL_REGISTRY.get(key, {}) if key else {}
    model_limit = entry.get("max_num_tokens")
    if model_limit is None:
        return config.MAX_NUM_TOKENS
    return min(config.MAX_NUM_TOKENS, int(model_limit))


def _effective_default_max_output_tokens(model_key: str | None = None) -> int:
    """Return the configured output limit capped by the model default."""
    key = model_key if model_key is not None else config.CURRENT_MODEL_KEY
    entry = MODEL_REGISTRY.get(key, {}) if key else {}
    model_limit = entry.get("default_max_output_tokens")
    if model_limit is None:
        return config.DEFAULT_MAX_OUTPUT_TOKENS
    return min(config.DEFAULT_MAX_OUTPUT_TOKENS, int(model_limit))


def _effective_tool_max_num_tokens() -> int:
    return min(
        config.TOOL_MAX_NUM_TOKENS,
        _effective_max_num_tokens(config.TOOL_MODEL_KEY),
    )


def _create_engine(
    backend=None,
    *,
    model_path: str | None = None,
    model_key: str | None = None,
    cache_dir_env: str = "LITERT_CACHE_DIR",
    enable_speculative_decoding: bool | None = None,
    max_num_tokens: int | None = None,
):
    active_model_key = (
        config.CURRENT_MODEL_KEY
        if model_key is None
        else model_key
    )
    if backend is None:
        backend = resolve_backend(active_model_key)

    if model_path is None:
        model_path = config.MODEL_PATH

    if enable_speculative_decoding is None:
        enable_speculative_decoding = config.ENABLE_SPECULATIVE_DECODING
    if max_num_tokens is None:
        max_num_tokens = _effective_max_num_tokens(active_model_key)

    engine_kwargs: dict[str, Any] = {
        "backend": backend,
        "enable_speculative_decoding": enable_speculative_decoding,
        "max_num_tokens": max_num_tokens,
    }

    engine_parameters = set(
        inspect.signature(litert_lm.Engine).parameters
    )

    if "cache_dir" in engine_parameters:
        engine_kwargs["cache_dir"] = os.environ.get(
            cache_dir_env,
            str(Path(model_path).parent),
        )

    _apply_model_engine_options(
        engine_kwargs,
        engine_parameters,
        active_model_key,
        backend,
    )

    engine = litert_lm.Engine(
        model_path,
        **engine_kwargs,
    )
    engine.__enter__()
    return engine


def _create_tool_runtime():
    """Load the dedicated tool model and start its conversation worker."""
    from .engine.worker import PersistentConversationWorker

    tool_key = config.TOOL_MODEL_KEY
    if tool_key not in MODEL_REGISTRY:
        raise RuntimeError(
            f"Unknown LITERT_TOOL_MODEL_KEY: {tool_key!r}. "
            f"Available: {', '.join(MODEL_REGISTRY)}"
        )

    configured_path = Path(config.TOOL_MODEL_PATH).expanduser()
    registry_path = Path(config._MODELS_DIR) / f"{tool_key}.litertlm"
    if configured_path == registry_path or not configured_path.is_file():
        resolved_path = ensure_model(tool_key)
    else:
        resolved_path = str(configured_path.resolve())

    config.TOOL_MODEL_PATH = resolved_path
    backend = resolve_backend(
        tool_key,
        request_backend=config.TOOL_BACKEND,
    )
    backend_name = type(backend).__name__
    print(
        "Loading dedicated tool model with the "
        f"{backend_name} backend..."
    )
    print(f"Tool model: {Path(resolved_path).resolve()}")
    print(f"Tool model key: {tool_key}")
    print(f"Tool reasoning effort: {config.TOOL_REASONING_EFFORT}")

    loaded_engine = None
    try:
        loaded_engine = _create_engine(
            backend=backend,
            model_path=resolved_path,
            model_key=tool_key,
            cache_dir_env="LITERT_TOOL_CACHE_DIR",
            # The tool model favors predictable latency over a second draft model.
            enable_speculative_decoding=False,
            max_num_tokens=_effective_tool_max_num_tokens(),
        )
        worker = PersistentConversationWorker(
            loaded_engine,
            name="tools",
        )
        worker.start()
    except Exception:
        if loaded_engine is not None:
            loaded_engine.__exit__(None, None, None)
        raise
    print(
        "Tool-model routing ready: native workspace tools -> "
        f"{tool_key} ({config.TOOL_BACKEND}); plain chat -> "
        f"{config.CURRENT_MODEL_KEY or config.MODEL_PATH}."
    )
    if config.TOOL_ROUTE_OPENAI_TOOLS:
        print("Client-supplied OpenAI tools also use the tool model.")
    else:
        print("Client-supplied OpenAI tools use the primary model.")
    return loaded_engine, worker


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------


@app.get("/v1/admin/config")
async def admin_get_config():
    return {
        "max_num_tokens": _effective_max_num_tokens(),
        "default_max_output_tokens": _effective_default_max_output_tokens(),
        "max_tool_response_tokens": config.MAX_TOOL_RESPONSE_TOKENS,
        "context_safety_margin_tokens": config.CONTEXT_SAFETY_MARGIN_TOKENS,
        "inference_timeout_seconds": config.INFERENCE_TIMEOUT_SECONDS,
        "temperature": config.DEFAULT_TEMPERATURE,
        "top_p": config.DEFAULT_TOP_P,
        "top_k": config.DEFAULT_TOP_K,
        "reasoning_effort": config.DEFAULT_REASONING_EFFORT,
        "repetition_penalty": config.DEFAULT_REPETITION_PENALTY,
        "constrained_decoding": config.ENABLE_CONSTRAINED_DECODING,
        "speculative_decoding": config.ENABLE_SPECULATIVE_DECODING,
        "backend": config.DEFAULT_BACKEND,
        "malformed_tool_call_retries": config.MALFORMED_TOOL_CALL_RETRIES,
        "max_tool_argument_string_length": config.MAX_TOOL_ARGUMENT_STRING_LENGTH,
        "max_tool_calls_per_generation": config.MAX_TOOL_CALLS_PER_GENERATION,
    }


@app.post("/v1/admin/reconfigure")
async def admin_reconfigure(body: AdminReconfigureRequest):
    global engine, conversation_worker
    global tool_engine, tool_conversation_worker

    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Cannot reconfigure while generation is active.",
        )

    try:
        changed = []

        if body.max_num_tokens is not None:
            config.MAX_NUM_TOKENS = body.max_num_tokens
            changed.append(f"max_num_tokens={body.max_num_tokens}")
        if body.default_max_output_tokens is not None:
            config.DEFAULT_MAX_OUTPUT_TOKENS = body.default_max_output_tokens
            changed.append(f"default_max_output_tokens={body.default_max_output_tokens}")
        if body.max_tool_response_tokens is not None:
            config.MAX_TOOL_RESPONSE_TOKENS = body.max_tool_response_tokens
            changed.append(f"max_tool_response_tokens={body.max_tool_response_tokens}")
        if body.context_safety_margin_tokens is not None:
            config.CONTEXT_SAFETY_MARGIN_TOKENS = body.context_safety_margin_tokens
            changed.append(f"context_safety_margin_tokens={body.context_safety_margin_tokens}")
        if body.inference_timeout_seconds is not None:
            config.INFERENCE_TIMEOUT_SECONDS = body.inference_timeout_seconds
            changed.append(f"inference_timeout_seconds={body.inference_timeout_seconds}")
        if body.temperature is not None:
            config.DEFAULT_TEMPERATURE = body.temperature
            changed.append(f"temperature={body.temperature}")
        if body.top_p is not None:
            config.DEFAULT_TOP_P = body.top_p
            changed.append(f"top_p={body.top_p}")
        if body.top_k is not None:
            config.DEFAULT_TOP_K = body.top_k
            changed.append(f"top_k={body.top_k}")
        if body.reasoning_effort is not None:
            config.DEFAULT_REASONING_EFFORT = body.reasoning_effort
            changed.append(f"reasoning_effort={body.reasoning_effort}")
        if body.repetition_penalty is not None:
            config.DEFAULT_REPETITION_PENALTY = body.repetition_penalty
            changed.append(f"repetition_penalty={body.repetition_penalty}")
        if body.constrained_decoding is not None:
            config.ENABLE_CONSTRAINED_DECODING = body.constrained_decoding
            changed.append(f"constrained_decoding={body.constrained_decoding}")
        if body.speculative_decoding is not None:
            config.ENABLE_SPECULATIVE_DECODING = body.speculative_decoding
            changed.append(f"speculative_decoding={body.speculative_decoding}")
        if body.backend is not None:
            config.DEFAULT_BACKEND = body.backend
            changed.append(f"backend={body.backend}")
        if body.malformed_tool_call_retries is not None:
            config.MALFORMED_TOOL_CALL_RETRIES = body.malformed_tool_call_retries
            changed.append(f"malformed_tool_call_retries={body.malformed_tool_call_retries}")
        if body.max_tool_argument_string_length is not None:
            config.MAX_TOOL_ARGUMENT_STRING_LENGTH = body.max_tool_argument_string_length
            changed.append(f"max_tool_argument_string_length={body.max_tool_argument_string_length}")
        if body.max_tool_calls_per_generation is not None:
            config.MAX_TOOL_CALLS_PER_GENERATION = body.max_tool_calls_per_generation
            changed.append(f"max_tool_calls_per_generation={body.max_tool_calls_per_generation}")

        if not changed:
            return {"status": "unchanged", "changes": []}

        print(f"\n[admin] Reconfiguring: {', '.join(changed)}")

        # Stop and tear down both routed runtimes.
        if tool_conversation_worker is not None:
            tool_conversation_worker.stop()
            tool_conversation_worker = None
            config.tool_conversation_worker = None

        if tool_engine is not None:
            tool_engine.__exit__(None, None, None)
            tool_engine = None
            config.tool_engine = None

        if conversation_worker is not None:
            conversation_worker.stop()
            conversation_worker = None
            config.conversation_worker = None

        if engine is not None:
            config.engine.__exit__(None, None, None)
            config.engine = None
            engine = None

        # Recreate
        engine = _create_engine(
            backend=resolve_backend(
                config.CURRENT_MODEL_KEY,
                request_backend=config.DEFAULT_BACKEND,
            )
        )
        config.engine = engine

        from .engine.worker import PersistentConversationWorker

        conversation_worker = PersistentConversationWorker(
            engine,
            name="primary",
        )
        conversation_worker.start()
        config.conversation_worker = conversation_worker

        tool_warning = None
        if not _workspace_tooling_enabled_for_primary():
            config.TOOL_MODEL_LOAD_ERROR = None
            print(
                "[admin] Native workspace tooling remains disabled for "
                "Gemma 4 12B; client tools use the primary model."
            )
        elif config.TOOL_ROUTING_ENABLED and _primary_is_tool_model():
            config.TOOL_MODEL_LOAD_ERROR = None
            print(
                "[admin] Tool routing reuses the primary model; "
                "no second engine was loaded."
            )
        elif config.TOOL_ROUTING_ENABLED:
            try:
                tool_engine, tool_conversation_worker = _create_tool_runtime()
                config.tool_engine = tool_engine
                config.tool_conversation_worker = tool_conversation_worker
                config.TOOL_MODEL_LOAD_ERROR = None
            except Exception as exc:
                tool_warning = str(exc)
                config.TOOL_MODEL_LOAD_ERROR = tool_warning
                tool_engine = None
                tool_conversation_worker = None
                config.tool_engine = None
                config.tool_conversation_worker = None
                print(
                    f"[admin] Tool-model routing is unavailable: {tool_warning}",
                    file=sys.stderr,
                    flush=True,
                )

        print("[admin] Routed engines restarted with new configuration.")

        return {
            "status": "restarted",
            "changes": changed,
            "tool_model_load_error": tool_warning,
        }
    except Exception:
        _request_lock.release()
        raise
    finally:
        if _request_lock.locked():
            _request_lock.release()


# ---------------------------------------------------------------------------
# Model selection routes
# ---------------------------------------------------------------------------


@app.get("/v1/admin/models")
async def admin_list_models():
    """Return available models and the currently loaded model."""
    models = []
    for key, entry in MODEL_REGISTRY.items():
        dest = Path(config._MODELS_DIR) / f"{key}.litertlm"
        models.append({
            "key": key,
            "display": entry["display"],
            "repo": entry["repo"],
            "filename": entry["filename"],
            "backend": entry.get("backend", config.DEFAULT_BACKEND),
            "downloaded": dest.is_file(),
            "path": str(dest) if dest.is_file() else None,
        })

    return {
        "models": models,
        "current": config.CURRENT_MODEL_KEY,
        "current_path": config.MODEL_PATH,
        "current_backend": config.DEFAULT_BACKEND,
    }


class SwitchModelRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    key: str


@app.post("/v1/admin/switch-model")
async def admin_switch_model(body: SwitchModelRequest):
    """Switch to a different model from the registry."""
    global engine, conversation_worker
    global tool_engine, tool_conversation_worker

    if body.key not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model key: {body.key}. "
                    f"Available: {', '.join(MODEL_REGISTRY.keys())}",
        )

    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Cannot switch models while generation is active.",
        )

    try:
        entry = MODEL_REGISTRY[body.key]
        print(f"\n[admin] Switching model to: {body.key} ({entry['display']})")

        # Download model if needed. This may take a while for large models.
        resolved_path = ensure_model(body.key)
        config.MODEL_PATH = resolved_path
        config.CURRENT_MODEL_KEY = body.key
        config.apply_model_runtime_defaults(body.key)

        print(f"[admin] Model path: {resolved_path}")

        # Stop and tear down current routed runtimes. The tool runtime is
        # recreated only when the new primary is not already the tool model.
        if tool_conversation_worker is not None:
            tool_conversation_worker.stop()
            tool_conversation_worker = None
            config.tool_conversation_worker = None

        if tool_engine is not None:
            tool_engine.__exit__(None, None, None)
            tool_engine = None
            config.tool_engine = None

        if conversation_worker is not None:
            conversation_worker.stop()
            conversation_worker = None
            config.conversation_worker = None

        if engine is not None:
            config.engine.__exit__(None, None, None)
            config.engine = None
            engine = None

        # Create new engine with the new model
        engine = _create_engine(
            backend=resolve_backend(body.key)
        )
        config.engine = engine

        from .engine.worker import PersistentConversationWorker

        conversation_worker = PersistentConversationWorker(
            engine,
            name="primary",
        )
        conversation_worker.start()
        config.conversation_worker = conversation_worker

        tool_warning = None
        if not _workspace_tooling_enabled_for_primary():
            config.TOOL_MODEL_LOAD_ERROR = None
            print(
                "[admin] Native workspace tooling is disabled for the new "
                "12B primary; client tools use the primary model."
            )
        elif config.TOOL_ROUTING_ENABLED and _primary_is_tool_model():
            config.TOOL_MODEL_LOAD_ERROR = None
            print(
                "[admin] Tool routing reuses the new primary model; "
                "no second engine was loaded."
            )
        elif config.TOOL_ROUTING_ENABLED:
            try:
                tool_engine, tool_conversation_worker = _create_tool_runtime()
                config.tool_engine = tool_engine
                config.tool_conversation_worker = tool_conversation_worker
                config.TOOL_MODEL_LOAD_ERROR = None
            except Exception as exc:
                tool_warning = str(exc)
                config.TOOL_MODEL_LOAD_ERROR = tool_warning
                tool_engine = None
                tool_conversation_worker = None
                config.tool_engine = None
                config.tool_conversation_worker = None
                print(
                    f"[admin] Tool-model routing is unavailable: {tool_warning}",
                    file=sys.stderr,
                    flush=True,
                )

        print(f"[admin] Successfully switched to: {body.key}")

        return {
            "status": "switched",
            "key": body.key,
            "display": entry["display"],
            "path": resolved_path,
            "tool_model_load_error": tool_warning,
        }
    except Exception:
        _request_lock.release()
        raise
    finally:
        if _request_lock.locked():
            _request_lock.release()
