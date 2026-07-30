import json
import sys
import threading
import time
import uuid
from typing import Any, Optional

from ..config import (
    PROGRESS_INTERVAL_SECONDS,
    _console_lock,
)
from ..utils.token import count_tokens


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
