# LiteRT-LM Selectable Tool-Context Server

OpenAI-compatible FastAPI proxy for LiteRT-LM with:

- GPU inference
- persistent KV cache for normal text conversations
- selectable tool-call context behavior
- Gemma thinking/reasoning support
- OpenAI-style streaming and tool-call responses
- native, folder-scoped `list`, `glob`, `grep`, `read`, `line_count`, and
  `write` tools
- repetition protection
- context-window preflight checks
- tool-output truncation
- malformed tool-call validation and retry
- console TTFT, prefill, and decode metrics

Server file:

```text
main.py
```

## Requirements

### Python

Python 3.10 or newer.

The server uses these third-party packages:

| Package | Purpose |
|---|---|
| `litert-lm` | Model loading and inference |
| `fastapi` | HTTP API |
| `uvicorn` | ASGI web server |
| `pydantic` | Request validation |

Everything else imported by the server is part of the Python standard library.

### Install on Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Direct installation without `requirements.txt`:

```powershell
python -m pip install "litert-lm>=0.14.0,<0.15" "fastapi>=0.116,<1" "uvicorn>=0.35,<1" "pydantic>=2,<3"
```

The server was developed around LiteRT-LM 0.13.1. Its API checks use runtime introspection so newer compatible LiteRT-LM builds can expose optional features such as native thinking configuration.

Check installed versions:

```powershell
python -c "import importlib.metadata as m; print('litert-lm', m.version('litert-lm')); print('fastapi', m.version('fastapi')); print('uvicorn', m.version('uvicorn')); print('pydantic', m.version('pydantic'))"
```

## Basic startup

```powershell
$env:LITERT_MODEL_PATH = "C:\Users\Joseph Montanez\.litert-lm\models\gemma4-e4b\model.litertlm"

python .\main.py
```

Default address:

```text
http://0.0.0.0:8000
```

Useful URLs:

```text
GET  http://localhost:8000/health
GET  http://localhost:8000/v1/models
POST http://localhost:8000/v1/chat/completions
GET  http://localhost:8000/docs
```

The server has no authentication or TLS. Do not expose it directly to the public Internet.

## Command-line parameters

```text
usage: main.py
       [--tool-context-mode {merged,separate}]
       [--host HOST]
       [--port PORT]
```

### `--tool-context-mode`

Values:

| Value | Behavior |
|---|---|
| `separate` | Creates a fresh full-history LiteRT conversation for every tool turn. Safer for tool formatting, but it must prefill the history again. This is the default. |
| `merged` | Keeps one live LiteRT conversation and KV cache across user, tool-call, tool-result, and final-answer turns. Faster, but LiteRT/Gemma tool-protocol state may become corrupted. |

The CLI value overrides `LITERT_TOOL_CONTEXT_MODE`.

Examples:

```powershell
python .\main.py --tool-context-mode separate
```

```powershell
python .\main.py --tool-context-mode merged
```

### `--host`

Interface on which Uvicorn listens.

Default:

```text
0.0.0.0
```

Local-machine only:

```powershell
python .\main.py --host 127.0.0.1
```

### `--port`

Listening port.

Default:

```text
8000
```

Example:

```powershell
python .\main.py --port 8080
```

## Environment variables

Environment variables are read when Python imports the server. Restart the process after changing them.

### Model and cache

#### `LITERT_MODEL_PATH`

Path to the `.litertlm` model bundle.

Default:

```text
gemma-4-E4B-it.litertlm
```

Example:

```powershell
$env:LITERT_MODEL_PATH = "C:\Users\Joseph Montanez\.litert-lm\models\gemma4-e4b\model.litertlm"
```

#### `LITERT_CACHE_DIR`

Directory used by LiteRT for compiled model/backend cache files.

Default:

```text
parent directory of LITERT_MODEL_PATH
```

Example:

```powershell
$env:LITERT_CACHE_DIR = "C:\Users\Joseph Montanez\.litert-lm\cache"
```

This is not the conversational KV cache. Conversation KV state remains in RAM/GPU memory and cannot currently be restored from this directory.

### Context and output limits

#### `LITERT_MAX_NUM_TOKENS`

Maximum token capacity configured on the LiteRT engine.

Default:

```text
32768
```

The effective budget includes:

```text
messages
+ tool schemas
+ reserved completion tokens
+ safety margin
```

Example:

```powershell
$env:LITERT_MAX_NUM_TOKENS = "32768"
```

Higher values can substantially increase memory consumption.

#### `LITERT_MAX_OUTPUT_TOKENS`

Default maximum generated tokens when the request does not provide `max_completion_tokens` or `max_tokens`.

Default:

```text
4096
```

Precedence:

```text
request.max_completion_tokens
request.max_tokens
LITERT_MAX_OUTPUT_TOKENS
```

#### `LITERT_MAX_TOOL_RESPONSE_TOKENS`

Maximum tokens retained from each incoming tool result.

Default:

```text
4096
```

Oversized tool results are truncated before they are inserted into the LiteRT conversation.

Set to `0` or a negative value to disable server-side tool-result truncation.

#### `LITERT_CONTEXT_SAFETY_MARGIN`

Tokens reserved to account for model chat formatting, control tokens, estimation error, and runtime overhead.

Default:

```text
1024
```

Before inference, the server rejects a projected overflow with HTTP `413`.

### Inference protection

#### `LITERT_INFERENCE_TIMEOUT`

Maximum duration, in seconds, for a synchronous tool-capable inference operation.

Default:

```text
180
```

When exceeded, the server calls `conversation.cancel_process()` and returns an error.

#### `LITERT_MALFORMED_TOOL_RETRIES`

Number of clean-context retries after the server rejects a malformed tool call.

Default:

```text
1
```

A value of `1` means one initial attempt plus one retry.

#### `LITERT_MAX_TOOL_ARGUMENT_LENGTH`

Maximum permitted length of any individual string inside generated tool arguments.

Default:

```text
16384
```

This limit is measured in Python characters, not model tokens.

The server also rejects tool arguments containing detected chat-template or tool-protocol leakage.

### Sampling and repetition

#### `LITERT_TEMPERATURE`

Default generation temperature when the request omits `temperature`.

Default:

```text
1.0
```

The client request overrides this value.

#### `LITERT_TOOL_TEMPERATURE`

Default declared value:

```text
0.6
```

Current status:

```text
reserved / currently unused
```

The current server uses `LITERT_TEMPERATURE` or the request-level `temperature` for both text and tool-capable requests. Changing `LITERT_TOOL_TEMPERATURE` currently has no effect.

Fixed sampling defaults that do not currently have environment variables:

| Setting | Default |
|---|---:|
| `top_p` | `0.95` |
| `top_k` | `64` |
| repetition penalty | `1.12` |
| repetition window | `256` |
| no-repeat n-gram size | `6` |

Request-level values override these defaults where supported by the installed LiteRT-LM version.

### Thinking

#### `LITERT_REASONING_EFFORT`

Default thinking level.

Default:

```text
high
```

Accepted canonical values:

| Value | Internal budget when native LiteRT thinking budgets are available |
|---|---:|
| `none` | `0` |
| `minimal` | `256` |
| `low` | `512` |
| `medium` | `1024` |
| `high` | `2048` |
| `xhigh` | `4096` |

Accepted aliases:

```text
off, disabled, false, 0  -> none
on, enabled, true, 1     -> medium
extra_high, very_high    -> xhigh
```

On LiteRT-LM versions without native `ThinkingConfig`, the server enables thinking through template `extra_context`; the exact budget may not be enforceable.

The request-level `reasoning_effort` field overrides this environment variable.

### Tool handling

#### `LITERT_CONSTRAINED_DECODING`

Enables LiteRT constrained decoding when tools are supplied and the installed LiteRT-LM API supports it.

Default:

```text
1
```

Values treated as false:

```text
0
false
no
off
```

All other values are treated as true.

#### `LITERT_TOOL_CONTEXT_MODE`

Default tool context mode.

Default:

```text
separate
```

Allowed values:

```text
merged
separate
```

The `--tool-context-mode` CLI argument overrides this environment variable.

## Full PowerShell environment example

```powershell
$env:LITERT_MODEL_PATH = "C:\Users\Joseph Montanez\.litert-lm\models\gemma4-e4b\model.litertlm"
$env:LITERT_CACHE_DIR = "C:\Users\Joseph Montanez\.litert-lm\models\gemma4-e4b"

$env:LITERT_MAX_NUM_TOKENS = "32768"
$env:LITERT_MAX_OUTPUT_TOKENS = "4096"
$env:LITERT_MAX_TOOL_RESPONSE_TOKENS = "4096"
$env:LITERT_CONTEXT_SAFETY_MARGIN = "1024"

$env:LITERT_INFERENCE_TIMEOUT = "180"
$env:LITERT_MALFORMED_TOOL_RETRIES = "1"
$env:LITERT_MAX_TOOL_ARGUMENT_LENGTH = "16384"

$env:LITERT_TEMPERATURE = "1.0"
$env:LITERT_REASONING_EFFORT = "high"
$env:LITERT_CONSTRAINED_DECODING = "1"
$env:LITERT_TOOL_CONTEXT_MODE = "separate"

python .\main.py `
  --host 0.0.0.0 `
  --port 8000
```

## OpenAI-compatible request parameters

Endpoint:

```text
POST /v1/chat/completions
```

Unknown JSON fields are ignored.

### Required fields

#### `messages`

Non-empty array of chat messages.

Supported message fields:

| Field | Type | Notes |
|---|---|---|
| `role` | string | Required. Common values are `system`, `user`, `assistant`, and `tool`. `model` is normalized to `assistant`. |
| `content` | string, array, object, or null | Message content. |
| `name` | string or null | Used especially for tool responses. |
| `tool_call_id` | string or null | Associates a `tool` response with an assistant tool call. |
| `tool_calls` | array or null | Assistant OpenAI-style function calls. |

### Optional top-level fields

| Field | Type | Default/behavior |
|---|---|---|
| `model` | string | Defaults to `LITERT_MODEL_PATH`; used primarily in response metadata. The loaded engine model does not change per request. |
| `stream` | boolean | Default `false`. When true, uses OpenAI-style server-sent events. |
| `temperature` | number | Overrides `LITERT_TEMPERATURE`. |
| `max_completion_tokens` | integer | Preferred output limit. |
| `max_tokens` | integer | Used only when `max_completion_tokens` is absent. |
| `top_p` | number | Default `0.95`. |
| `top_k` | integer | Default `64`. |
| `stop` | string or array of strings | Stops emitted text when a sequence is detected. |
| `presence_penalty` | number or null | Forwarded only when the installed LiteRT-LM repetition API supports it. |
| `frequency_penalty` | number or null | Forwarded only when supported. |
| `repetition_penalty` | number or null | Default `1.12` when supported. |
| `repetition_window` | integer or null | Default `256` when supported. |
| `no_repeat_ngram_size` | integer or null | Default `6` when supported. |
| `seed` | integer or null | Forwarded when `SamplerConfig` supports it. |
| `reasoning_effort` | string or null | Overrides `LITERT_REASONING_EFFORT`. |
| `tools` | array or null | OpenAI function-tool schemas. Only entries with `type: "function"` and a valid function name are retained. |
| `tool_choice` | string or object | `"none"` removes all tools. A named OpenAI function choice filters the available tools to that name. Other values behave like automatic selection. |
| `parallel_tool_calls` | any | Accepted and included in conversation-cache identity, but the server does not independently implement parallel tool execution. The client executes returned calls. |
| `workspace_tools` | boolean | Default `false`. Registers native LiteRT-LM filesystem tools that the model executes automatically. |
| `workspace_path` | string or null | Folder that bounds path-based tools and becomes the shell working directory. It is not a shell sandbox. Required when `workspace_tools` is true. |
| `workspace_read_only` | boolean | Default `true`. When true, `write`, `edit`, and `shell` are not registered. |
| `workspace_shell_approval_id` | string or null | Opaque client-generated ID used to retrieve and decide per-command shell approvals. Shell calls without one are denied. |

### Request example

```json
{
  "model": "gemma4-e4b",
  "stream": true,
  "reasoning_effort": "high",
  "temperature": 1.0,
  "max_completion_tokens": 2048,
  "messages": [
    {
      "role": "user",
      "content": "Summarize this code."
    }
  ]
}
```

### Tool request example

```json
{
  "model": "gemma4-e4b",
  "stream": true,
  "reasoning_effort": "high",
  "messages": [
    {
      "role": "user",
      "content": "Find every use of credit_txn_id."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "grep",
        "description": "Search file contents.",
        "parameters": {
          "type": "object",
          "properties": {
            "pattern": {
              "type": "string"
            },
            "path": {
              "type": "string"
            }
          },
          "required": [
            "pattern",
            "path"
          ],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

### Native workspace tools

The built-in web chat has a **Workspace tools** control above the prompt. Enter
an absolute folder path and leave **Read only** enabled to give the model these
automatically executed LiteRT-LM tools:

- `find`: find files and directories by path or glob pattern
- `grep`: search text files with a regular expression
- `read`: read bounded line ranges from UTF-8 text files
- `line_count`: count the lines in a UTF-8 text file
- `size`, `sort`, `head`, `tail`, and `platform`: bounded inspection helpers

Disabling **Read only** also registers `write`, `edit`, and `shell`. The web chat
requires a separate approval for every shell command and displays the exact
command and working directory before execution. Closing or stopping the request
denies a pending command; unanswered approvals expire after five minutes. A
working directory is not a process sandbox, so an approved shell command can
still access parent or absolute paths with the server user's permissions.

API clients must generate a hard-to-guess `workspace_shell_approval_id`, poll
`GET /v1/workspace/shell-approvals/{approval_id}`, and decide a pending call with
`POST /v1/workspace/shell-approvals/{approval_id}/{call_id}` using
`{"approved": true}` or `{"approved": false}`. Without an approval ID, shell
calls are denied rather than executed.

Paths handled by the filesystem tools are resolved against the selected folder.
Parent traversal and symlinks that escape that folder are rejected. Reads,
searches, result counts, and writes are bounded to protect the model context and
server process. Approved shell commands are the exception: the selected folder
is only their initial working directory.

API example:

```json
{
  "model": "gemma4-e4b",
  "stream": true,
  "workspace_tools": true,
  "workspace_path": "/absolute/path/to/project",
  "workspace_read_only": true,
  "messages": [
    {
      "role": "user",
      "content": "Find where the HTTP server is initialized."
    }
  ]
}
```

Native workspace tools cannot be combined with client-supplied OpenAI `tools`
in one request. The former execute inside LiteRT-LM; the latter are returned to
the API client for execution.

## Endpoints

### `GET /health`

Returns:

- model path
- GPU backend
- token limits
- disk cache directory
- constrained-decoding state
- selected tool-context mode
- current conversation-cache status
- malformed-call safeguards

### `GET /v1/models`

Returns one OpenAI-style model entry derived from the loaded model filename.

### `POST /v1/chat/completions`

OpenAI-style chat-completion endpoint supporting normal responses, SSE streaming, function tools, and tool-result continuation.

### `GET /docs`

FastAPI-generated interactive OpenAPI documentation.

## Runtime behavior

### Concurrency

The server allows only one active inference request.

A concurrent request receives:

```text
HTTP 429
```

### Context overflow

The server estimates:

```text
messages + tool schemas + output reserve + safety margin
```

A projected overflow receives:

```text
HTTP 413
```

### Streaming keep-alive

During long prefill operations, the SSE response emits a comment approximately every five seconds:

```text
: keep-alive TIMESTAMP
```

This helps prevent clients and proxies from treating the stream as idle.

### Console metrics

The server prints:

- total estimated prompt tokens
- newly prefilling tokens
- cached tokens
- time to first token
- estimated prefill tokens per second
- decode tokens per second
- tool calls and tool responses
- context-budget calculation

Example:

```text
[abc12345] done | cache=reuse | total≈8,950 | prefill≈7 | cached≈8,943 | output=59 | TTFT=2.87s
```

For synchronous tool responses, the displayed decode rate can be artificially high because the complete response arrives before it is measured.

## Tool-context mode details

### Separate mode

```powershell
python .\main.py --tool-context-mode separate
```

Behavior:

1. Receive the complete OpenAI message history.
2. Create a fresh LiteRT conversation.
3. Prefill the complete history.
4. Generate one tool call or final answer.
5. Destroy that tool conversation.

Advantages:

- isolates each tool turn
- avoids carrying malformed tool-protocol state forward
- safer default

Disadvantages:

- repeats full-history prefill on every tool turn
- does not preserve tool-chain thinking/KV state
- slower for large histories

### Merged mode

```powershell
python .\main.py --tool-context-mode merged
```

Behavior:

1. Create or reuse one live conversation.
2. Keep the model KV cache in RAM/GPU memory.
3. Append user messages and tool responses.
4. Continue the same thinking/tool chain.

Advantages:

- fast incremental prefill
- preserves thinking state
- preserves KV cache across tool turns

Disadvantages:

- LiteRT/Gemma tool-protocol state may become corrupted
- malformed tool calls can force conversation reset
- KV cache cannot currently be parked on disk and restored

## Recommended configurations

### Reliable text generation

```powershell
$env:LITERT_TOOL_CONTEXT_MODE = "separate"
$env:LITERT_REASONING_EFFORT = "high"

python .\main.py
```

Use the client without tools when possible.

### Faster experimental tool chains

```powershell
$env:LITERT_TOOL_CONTEXT_MODE = "merged"
$env:LITERT_REASONING_EFFORT = "high"

python .\main.py
```

Use a small, focused tool set and reset the session if arguments begin containing protocol fragments.

### Lower memory

```powershell
$env:LITERT_MAX_NUM_TOKENS = "16384"
$env:LITERT_MAX_OUTPUT_TOKENS = "2048"
```

The complete request budget must fit inside that capacity.

## Troubleshooting

### `No files matched` when searching source content

A filename glob tool searches paths, not file contents. Provide a `grep`/content-search tool or a shell tool that can execute `rg`/`grep`.

### XNNPACK allocation errors

The server always selects:

```python
litert_lm.Backend.GPU()
```

If logs mention XNNPACK, confirm that the expected server file is running and terminate stale Python processes:

```powershell
Get-Process python,py -ErrorAction SilentlyContinue | Stop-Process -Force
```

### GPU stops working during a large request

Check the printed context budget. Large tool results can push the effective prompt near the engine limit even when the visible chat history appears smaller.

### Every tool call prefills the entire history

This is expected in:

```text
--tool-context-mode separate
```

Use `merged` to preserve the live KV cache, with the tool-protocol reliability tradeoff described above.

### Tool arguments contain `user]`, `tool]`, or `call:...`

The server rejects known protocol-leak patterns and can retry according to `LITERT_MALFORMED_TOOL_RETRIES`. In merged mode, a malformed call may reset the persistent conversation.

### Check active configuration

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 10
```
