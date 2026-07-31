import queue
import sys
import threading
from typing import Any, Optional

from fastapi import HTTPException

from .. import config
from ..config import (
    MALFORMED_TOOL_CALL_RETRIES,
    _console_lock,
)
from ..models import (
    ChatCompletionRequest,
    ConversationPlan,
    GenerationState,
    InferenceJob,
    MalformedToolCallError,
)
from .generation import generation_events
from ..utils.message import (
    assistant_response_matches,
    build_initial_messages,
    build_name_by_tool_call_id_map,
    combine_delta_messages,
    last_input_index,
    message_to_litert,
    starts_with_messages,
)
from .progress import ConsoleProgress
from .sampling import (
    build_conversation_kwargs,
    conversation_config_signature,
    normalize_reasoning_effort,
)
from ..utils.token import (
    estimate_input_tokens,
    estimate_messages_tokens,
)
from ..utils.tools import normalize_tool_definitions


class PersistentConversationWorker:
    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="litert-conversation-worker",
            daemon=True,
        )
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._conversation = None
        self._active_conversation = None
        self._pending_job: Optional[InferenceJob] = None
        self._current_job: Optional[InferenceJob] = None

        self._last_request_messages: list[dict[str, Any]] = []
        self._last_response_text = ""
        self._last_reasoning_text = ""
        self._last_tool_calls: list[dict[str, Any]] = []
        self._last_response_message: Optional[dict[str, Any]] = None
        self._last_finish_reason = "stop"
        self._last_usage: dict[str, Any] = {}
        self._context_usage: dict[str, Any] = {}
        self._config_signature: Optional[str] = None

    def start(self):
        self._thread.start()

    def submit(self, job: InferenceJob):
        with self._state_lock:
            self._pending_job = job
        self._jobs.put(job)

    def cancel_current(self) -> bool:
        with self._state_lock:
            pending_job = self._pending_job
            current_job = self._current_job
            conversations = [
                self._active_conversation,
                self._conversation,
            ]

        cancelled = pending_job is not None or current_job is not None
        for job in (pending_job, current_job):
            if job is None:
                continue
            job.cancel_event.set()

        if current_job is None:
            return cancelled

        for conversation in conversations:
            if conversation is None:
                continue

            try:
                conversation.cancel_process()
            except Exception:
                pass

        return cancelled

    def stop(self):
        self._stop_event.set()
        self.cancel_current()
        self._jobs.put(None)
        self._thread.join(timeout=10)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            pending = self._pending_job is not None
            busy = self._current_job is not None
            context_usage = dict(self._context_usage)

        return {
            "conversation_cached": self._conversation is not None,
            "active_conversation": self._active_conversation is not None,
            "cached_request_messages": len(self._last_request_messages),
            "last_response_characters": len(self._last_response_text),
            "last_reasoning_characters": len(self._last_reasoning_text),
            "last_tool_calls": len(self._last_tool_calls),
            "tool_context_mode": config.TOOL_CONTEXT_MODE,
            "tool_chain_persistent": config.TOOL_CONTEXT_MODE == "merged",
            "busy": pending or busy,
            "context_tokens": int(
                context_usage.get(
                    "context_tokens",
                    context_usage.get("total_tokens", 0),
                )
            ),
            "context_prompt_tokens": int(
                context_usage.get("prompt_tokens", 0)
            ),
            "context_generated_tokens": int(
                context_usage.get("generated_tokens", 0)
            ),
            "context_reasoning_tokens": int(
                context_usage.get("reasoning_tokens", 0)
            ),
            "context_estimated": True,
        }

    def reset(self):
        self.cancel_current()
        self._reset_conversation()

    def _run(self):
        while not self._stop_event.is_set():
            job = self._jobs.get()

            if job is None:
                break

            with self._state_lock:
                if self._pending_job is job:
                    self._pending_job = None
                self._current_job = job

            try:
                self._process(job)
            except Exception as exc:
                separate_tool_request = (
                    config.TOOL_CONTEXT_MODE == "separate"
                    and bool(normalize_tool_definitions(job.request))
                )

                if not separate_tool_request:
                    self._reset_conversation()

                job.result_queue.put(("error", exc))
            finally:
                with self._state_lock:
                    if self._current_job is job:
                        self._current_job = None

        self._reset_conversation()

    def _process(self, job: InferenceJob):
        has_tools = bool(normalize_tool_definitions(job.request))
        separate_tools = (
            has_tools and config.TOOL_CONTEXT_MODE == "separate"
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
            if job.cancel_event.is_set():
                job.result_queue.put(
                    (
                        "done",
                        {
                            "usage": {},
                            "response_text": "",
                            "reasoning_text": "",
                            "tool_calls": [],
                            "finish_reason": "stop",
                            "cache_mode": "cancelled",
                        },
                    )
                )
                return

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
            with config.engine.create_conversation(
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
            usage["context_tokens"] = (
                0 if state.cancelled else usage["total_tokens"]
            )

            if state.cancelled:
                self._record_context_usage()
            else:
                self._record_context_usage(usage)

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
        previous_context_tokens = self._context_token_count()

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
            self._conversation = config.engine.create_conversation(
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
            if state.cancelled:
                usage["context_tokens"] = 0
            elif plan.mode == "reuse":
                usage["context_tokens"] = (
                    previous_context_tokens
                    + plan.prefill_tokens
                    + usage.get("generated_tokens", 0)
                )
            else:
                usage["context_tokens"] = usage["total_tokens"]

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
                self._record_context_usage(usage)
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
        self._record_context_usage()
        self._config_signature = None

    def _record_context_usage(
        self,
        usage: Optional[dict[str, Any]] = None,
    ):
        with self._state_lock:
            self._context_usage = dict(usage or {})

    def _context_token_count(self) -> int:
        with self._state_lock:
            return int(
                self._context_usage.get(
                    "context_tokens",
                    self._context_usage.get("total_tokens", 0),
                )
            )
