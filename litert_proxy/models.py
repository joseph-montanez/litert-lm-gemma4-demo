import queue
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

import litert_lm
from pydantic import BaseModel, ConfigDict

from .config import MODEL_PATH


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
