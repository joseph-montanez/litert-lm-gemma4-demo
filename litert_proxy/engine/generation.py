import sys
import threading
from typing import Any, Iterator, Optional

from ..config import (
    INFERENCE_TIMEOUT_SECONDS,
    _console_lock,
)
from ..models import ChatCompletionRequest, GenerationState
from .progress import ConsoleProgress
from .sampling import (
    build_send_kwargs,
    thinking_enabled,
)
from ..utils.text import (
    RepetitionGuard,
    apply_stop_sequences,
    extract_litert_tool_calls,
    extract_reasoning_channels,
    extract_text,
    normalized_stops,
    stream_value_delta,
)
from ..utils.tools import (
    litert_tool_calls_to_openai,
    looks_like_tool_protocol,
    normalize_tool_definitions,
    recover_tool_calls_from_text,
    validate_litert_tool_calls,
)


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
    if cancel_event.is_set():
        state.cancelled = True
        return

    tool_event_handler = getattr(
        conversation,
        "tool_event_handler",
        None,
    )
    set_shell_approval_id = getattr(
        tool_event_handler,
        "set_shell_approval_id",
        None,
    )
    if callable(set_shell_approval_id):
        set_shell_approval_id(request.workspace_shell_approval_id)
    reset_tool_budget = getattr(tool_event_handler, "reset", None)
    if callable(reset_tool_budget):
        reset_tool_budget()

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
