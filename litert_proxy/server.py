import asyncio
import inspect
import queue
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import litert_lm
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from . import config
from .config import (
    CACHE_DIR,
    CONTEXT_SAFETY_MARGIN_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ENABLE_CONSTRAINED_DECODING,
    INFERENCE_TIMEOUT_SECONDS,
    MALFORMED_TOOL_CALL_RETRIES,
    MAX_NUM_TOKENS,
    MAX_TOOL_ARGUMENT_STRING_LENGTH,
    MAX_TOOL_RESPONSE_TOKENS,
    MODEL_PATH,
    SSE_HEARTBEAT_SECONDS,
    WEB_UI_TITLE,
    _console_lock,
    _request_lock,
    engine,
    conversation_worker,
)
from .models import ChatCompletionRequest, InferenceJob
from .utils.message import canonical_messages
from .utils.token import (
    count_tokens,
    estimate_context_budget,
    estimate_messages_tokens,
)
from .engine.streaming import (
    make_error_stream_chunk,
    make_initial_stream_chunk,
    make_stream_chunk,
    make_stream_reasoning_chunk,
    make_stream_tool_calls_chunk,
)


# ---------------------------------------------------------------------------
# Web chat HTML (loaded once at import time)
# ---------------------------------------------------------------------------

_web_chat_html: str = ""

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

    config.engine = litert_lm.Engine(
        MODEL_PATH,
        **engine_kwargs,
    )
    config.engine.__enter__()
    engine = config.engine

    from .engine.worker import PersistentConversationWorker

    conversation_worker = PersistentConversationWorker()
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
    print("Generation streaming: asynchronous for every request.")
    print(
        "Thinking channels stream as OpenAI delta.reasoning_content."
    )
    print(
        f"Built-in web chat: "
        f"{'http://localhost:8000/' if config.WEB_UI_ENABLED else 'disabled'}"
    )

    yield

    print("Shutting down...")

    if conversation_worker is not None:
        conversation_worker.stop()
        conversation_worker = None
        config.conversation_worker = None

    if engine is not None:
        config.engine.__exit__(None, None, None)
        config.engine = None
        engine = None


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
