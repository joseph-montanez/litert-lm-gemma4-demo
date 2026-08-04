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
        "backend": "gpu",
    },
    "gemma-4-E4B-it": {
        "repo": "litert-community/gemma-4-E4B-it-litert-lm",
        "filename": "gemma-4-E4B-it.litertlm",
        "display": "Gemma 4 E4B (4B)",
        "backend": "gpu",
    },
    "gemma-4-12B-it": {
        "repo": "litert-community/gemma-4-12B-it-litert-lm",
        "filename": "gemma-4-12B-it.litertlm",
        "display": "Gemma 4 12B",
        # The published desktop bundle declares a GPU main-backend
        # constraint; no CPU variant is currently available.
        "backend": "gpu",
        # LiteRT-LM's normal WebGPU prefill path has a 20-second tensor
        # readback deadline. Benchmark mode makes prefill wait and retry until
        # the GPU finishes, which is required for uncached 12B prompts that
        # take longer than that deadline.
        "enable_benchmark": True,
        "max_num_tokens": 32768,
        "default_max_output_tokens": 4096,
        "runtime_defaults": {
            "DEFAULT_MAX_OUTPUT_TOKENS": 4096,
            "MAX_TOOL_RESPONSE_TOKENS": 4096,
            "DEFAULT_TEMPERATURE": 0.8,
            "DEFAULT_TOP_P": 0.9,
            "DEFAULT_TOP_K": 40,
            "DEFAULT_REASONING_EFFORT": "none",
            "DEFAULT_REPETITION_PENALTY": 1.1,
            "MALFORMED_TOOL_CALL_RETRIES": 2,
            "MAX_TOOL_ARGUMENT_STRING_LENGTH": 4096,
            "MAX_TOOL_CALLS_PER_GENERATION": 12,
            "ENABLE_CONSTRAINED_DECODING": True,
            "ENABLE_SPECULATIVE_DECODING": False,
        },
    },
}

# Currently loaded model key (None = using raw MODEL_PATH / env vars).
CURRENT_MODEL_KEY: str | None = None

# Directory where registry models are stored.
_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".litert-lm", "models")

# Optional second runtime used for native workspace tools. Automatic native
# tool execution is disabled for the 12B primary by default; client-supplied
# OpenAI tools still run through 12B for execution by the external client. The
# experimental override keeps the E4B fallback on CPU so it does not share the
# WebGPU device with the larger primary model on macOS.
TOOL_ROUTING_ENABLED = os.environ.get(
    "LITERT_TOOL_ROUTING_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
TOOL_ROUTE_OPENAI_TOOLS = os.environ.get(
    "LITERT_TOOL_ROUTE_OPENAI_TOOLS",
    "0",
).strip().lower() not in {"0", "false", "no", "off"}
ENABLE_12B_TOOLS = os.environ.get(
    "LITERT_ENABLE_12B_TOOLS",
    "0",
).strip().lower() not in {"0", "false", "no", "off"}
TOOL_MODEL_KEY = os.environ.get(
    "LITERT_TOOL_MODEL_KEY",
    "gemma-4-E4B-it",
).strip()
TOOL_MODEL_PATH = os.environ.get(
    "LITERT_TOOL_MODEL_PATH",
    os.path.join(_MODELS_DIR, f"{TOOL_MODEL_KEY}.litertlm"),
)
TOOL_BACKEND = os.environ.get(
    "LITERT_TOOL_BACKEND",
    "cpu",
).strip().lower()
TOOL_MAX_NUM_TOKENS = int(
    os.environ.get("LITERT_TOOL_MAX_NUM_TOKENS", "8192")
)
TOOL_REASONING_EFFORT = os.environ.get(
    "LITERT_TOOL_REASONING_EFFORT",
    "none",
).strip().lower()
TOOL_MODEL_LOAD_ERROR: str | None = None
MAX_NUM_TOKENS = int(os.environ.get("LITERT_MAX_NUM_TOKENS", "131072"))
DEFAULT_MAX_OUTPUT_TOKENS = int(
    os.environ.get("LITERT_MAX_OUTPUT_TOKENS", "32768")
)
MAX_TOOL_RESPONSE_TOKENS = int(
    os.environ.get("LITERT_MAX_TOOL_RESPONSE_TOKENS", "32768")
)
CONTEXT_SAFETY_MARGIN_TOKENS = int(
    os.environ.get("LITERT_CONTEXT_SAFETY_MARGIN", "1024")
)
INFERENCE_TIMEOUT_SECONDS = float(
    os.environ.get("LITERT_INFERENCE_TIMEOUT", "600")
)
MALFORMED_TOOL_CALL_RETRIES = int(
    os.environ.get("LITERT_MALFORMED_TOOL_RETRIES", "6")
)
MAX_TOOL_ARGUMENT_STRING_LENGTH = int(
    os.environ.get("LITERT_MAX_TOOL_ARGUMENT_LENGTH", "16384")
)
MAX_TOOL_CALLS_PER_GENERATION = int(
    os.environ.get("LITERT_MAX_TOOL_CALLS_PER_GENERATION", "100")
)
CACHE_DIR = os.environ.get(
    "LITERT_CACHE_DIR",
    str(Path(MODEL_PATH).resolve().parent),
)

DEFAULT_TEMPERATURE = float(
    os.environ.get("LITERT_TEMPERATURE", "1.0")
)
DEFAULT_BACKEND = os.environ.get(
    "LITERT_BACKEND",
    "gpu",
).strip().lower()
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
    "max": 32768,
}
DEFAULT_TOOL_TEMPERATURE = float(
    os.environ.get("LITERT_TOOL_TEMPERATURE", "0.6")
)
DEFAULT_TOP_P = float(os.environ.get("LITERT_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.environ.get("LITERT_TOP_K", "64"))
DEFAULT_REPETITION_PENALTY = float(
    os.environ.get("LITERT_REPETITION_PENALTY", "1.12")
)
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

# Values restored before applying a model-specific runtime profile. Explicit
# environment variables always win over the built-in profile.
_RUNTIME_DEFAULT_BASELINES = {
    "DEFAULT_MAX_OUTPUT_TOKENS": DEFAULT_MAX_OUTPUT_TOKENS,
    "MAX_TOOL_RESPONSE_TOKENS": MAX_TOOL_RESPONSE_TOKENS,
    "DEFAULT_TEMPERATURE": DEFAULT_TEMPERATURE,
    "DEFAULT_TOP_P": DEFAULT_TOP_P,
    "DEFAULT_TOP_K": DEFAULT_TOP_K,
    "DEFAULT_REASONING_EFFORT": DEFAULT_REASONING_EFFORT,
    "DEFAULT_REPETITION_PENALTY": DEFAULT_REPETITION_PENALTY,
    "MALFORMED_TOOL_CALL_RETRIES": MALFORMED_TOOL_CALL_RETRIES,
    "MAX_TOOL_ARGUMENT_STRING_LENGTH": MAX_TOOL_ARGUMENT_STRING_LENGTH,
    "MAX_TOOL_CALLS_PER_GENERATION": MAX_TOOL_CALLS_PER_GENERATION,
    "ENABLE_CONSTRAINED_DECODING": ENABLE_CONSTRAINED_DECODING,
    "ENABLE_SPECULATIVE_DECODING": ENABLE_SPECULATIVE_DECODING,
}
_RUNTIME_DEFAULT_ENV_VARS = {
    "DEFAULT_MAX_OUTPUT_TOKENS": "LITERT_MAX_OUTPUT_TOKENS",
    "MAX_TOOL_RESPONSE_TOKENS": "LITERT_MAX_TOOL_RESPONSE_TOKENS",
    "DEFAULT_TEMPERATURE": "LITERT_TEMPERATURE",
    "DEFAULT_TOP_P": "LITERT_TOP_P",
    "DEFAULT_TOP_K": "LITERT_TOP_K",
    "DEFAULT_REASONING_EFFORT": "LITERT_REASONING_EFFORT",
    "DEFAULT_REPETITION_PENALTY": "LITERT_REPETITION_PENALTY",
    "MALFORMED_TOOL_CALL_RETRIES": "LITERT_MALFORMED_TOOL_RETRIES",
    "MAX_TOOL_ARGUMENT_STRING_LENGTH": "LITERT_MAX_TOOL_ARGUMENT_LENGTH",
    "MAX_TOOL_CALLS_PER_GENERATION": "LITERT_MAX_TOOL_CALLS_PER_GENERATION",
    "ENABLE_CONSTRAINED_DECODING": "LITERT_CONSTRAINED_DECODING",
    "ENABLE_SPECULATIVE_DECODING": "LITERT_SPECULATIVE_DECODING",
}


def apply_model_runtime_defaults(model_key: str | None) -> None:
    """Restore baseline defaults, then apply the selected model's profile."""
    for name, value in _RUNTIME_DEFAULT_BASELINES.items():
        globals()[name] = value

    entry = MODEL_REGISTRY.get(model_key, {}) if model_key else {}
    profile = entry.get("runtime_defaults", {})
    for name, value in profile.items():
        environment_name = _RUNTIME_DEFAULT_ENV_VARS.get(name)
        if environment_name and environment_name in os.environ:
            continue
        globals()[name] = value

engine = None
conversation_worker = None
tool_engine = None
tool_conversation_worker = None
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


BACKEND_MAP = {
    "gpu": "GPU",
    "cpu": "CPU",
    "npu": "NPU",
}


def resolve_backend(
    model_key: str | None = None,
    request_backend: str | None = None,
):
    """Resolve litert_lm Backend from model registry, request, or default.

    Priority: request > model registry entry > DEFAULT_BACKEND env.
    Returns a litert_lm.Backend instance.
    """
    import litert_lm

    backend_str = None

    if request_backend is not None:
        backend_str = request_backend.strip().lower()
    elif model_key is not None and model_key in MODEL_REGISTRY:
        backend_str = MODEL_REGISTRY[model_key].get(
            "backend", DEFAULT_BACKEND
        )
    else:
        backend_str = DEFAULT_BACKEND

    if backend_str not in BACKEND_MAP:
        raise ValueError(
            f"Unknown backend: {backend_str!r}. "
            f"Must be one of: {', '.join(BACKEND_MAP.keys())}"
        )

    backend_class = getattr(litert_lm.Backend, BACKEND_MAP[backend_str])
    return backend_class()


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
