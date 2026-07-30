import os
import threading
from pathlib import Path

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

ENABLE_SPECULATIVE_DECODING = os.environ.get(
    "LITERT_SPECULATIVE_DECODING",
    "0",
).strip().lower() not in {"0", "false", "no", "off"}

engine = None
conversation_worker = None

_console_lock = threading.Lock()
_request_lock = threading.Lock()
