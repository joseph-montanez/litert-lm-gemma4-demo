$env:LITERT_MODEL_PATH = "C:\Users\Joseph Montanez\.litert-lm\models\gemma4-e4b\model.litertlm"
$env:LITERT_CACHE_DIR = "C:\Users\Joseph Montanez\.litert-lm\models\gemma4-e4b"

$env:LITERT_MAX_NUM_TOKENS = "131072"
$env:LITERT_MAX_OUTPUT_TOKENS = "16384"
$env:LITERT_MAX_TOOL_RESPONSE_TOKENS = "4096"
$env:LITERT_CONTEXT_SAFETY_MARGIN = "1024"

$env:LITERT_INFERENCE_TIMEOUT = "180"
$env:LITERT_MALFORMED_TOOL_RETRIES = "1"
$env:LITERT_MAX_TOOL_ARGUMENT_LENGTH = "16384"

$env:LITERT_TEMPERATURE = "1.0"
$env:LITERT_REASONING_EFFORT = "high"
$env:LITERT_CONSTRAINED_DECODING = "1"
$env:LITERT_TOOL_CONTEXT_MODE = "merged"

python .\main.py `
  --host 0.0.0.0 `
  --port 8000
