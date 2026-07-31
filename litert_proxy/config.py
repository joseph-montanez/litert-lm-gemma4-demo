import os
import threading
from pathlib import Path

MODEL_PATH = os.environ.get(
    "LITERT_MODEL_PATH",
    "gemma-4-E4B-it.litertlm",
)

# HuggingFace repo for auto-download when model file is missing.
# Set LITERT_HF_REPO to override; set LITERT_HF_FILENAME to override filename.
_HF_REPO = os.environ.get(
    "LITERT_HF_REPO",
    "litert-community/gemma-4-E4B-it-litert-lm",
)
_HF_FILENAME = os.environ.get(
    "LITERT_HF_FILENAME",
    "gemma-4-E4B-it.litertlm",
)

# Model registry: key -> {repo, filename, display}
# Used by web UI model selector and switch-model API.
MODEL_REGISTRY = {
    "gemma-4-E2B-it": {
        "repo": "litert-community/gemma-4-E2B-it-litert-lm",
        "filename": "gemma-4-E2B-it.litertlm",
        "display": "Gemma 4 E2B (2B)",
    },
    "gemma-4-E4B-it": {
        "repo": "litert-community/gemma-4-E4B-it-litert-lm",
        "filename": "gemma-4-E4B-it.litertlm",
        "display": "Gemma 4 E4B (4B)",
    },
    "gemma-4-12B-it": {
        "repo": "litert-community/gemma-4-12B-it-litert-lm",
        "filename": "gemma-4-12B-it.litertlm",
        "display": "Gemma 4 12B",
    },
}

# Currently loaded model key (None = using raw MODEL_PATH / env vars).
CURRENT_MODEL_KEY: str | None = None

# Directory where registry models are stored.
_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".litert-lm", "models")
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
MAX_TOOL_CALLS_PER_GENERATION = int(
    os.environ.get("LITERT_MAX_TOOL_CALLS_PER_GENERATION", "6")
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
MODEL_LOADING = False
MODEL_LOAD_ERROR: str | None = None

_console_lock = threading.Lock()
_request_lock = threading.Lock()


def _download_model(repo_id: str, filename: str, dest: Path) -> str:
    """Download a model file from HuggingFace to dest path.

    Returns the resolved path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "Model file not found and huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest.parent,
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded).resolve()
    if downloaded_path != dest:
        import shutil
        shutil.move(str(downloaded_path), str(dest))

    print(f"Model downloaded to: {dest}")
    return str(dest)


def ensure_model(model_key: str | None = None) -> str:
    """Ensure model file exists on disk, downloading if needed.

    If model_key is provided, looks up MODEL_REGISTRY and downloads
    to ~/.litert-lm/models/<key>.litertlm. Otherwise uses MODEL_PATH
    and _HF_REPO/_HF_FILENAME env vars.

    Returns resolved absolute model path.
    """
    if model_key is not None and model_key in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_key]
        dest = Path(_MODELS_DIR) / f"{model_key}.litertlm"
        if dest.is_file():
            return str(dest)
        return _download_model(entry["repo"], entry["filename"], dest)

    model_path = Path(MODEL_PATH).resolve()
    if model_path.is_file():
        return str(model_path)

    print(f"Model not found at: {model_path}")
    return _download_model(_HF_REPO, _HF_FILENAME, model_path)
