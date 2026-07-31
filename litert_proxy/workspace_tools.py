import json
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litert_lm

from . import config as _config
from .utils.token import truncate_tool_content


MAX_LIST_RESULTS = 200
MAX_SHELL_OUTPUT_BYTES = 50_000
DEFAULT_SHELL_TIMEOUT = 30
MAX_GREP_RESULTS = 100
MAX_READ_LINES = 1000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOOL_RESULT_CHARS = 50_000
MAX_IDENTICAL_TOOL_CALLS = 2
SHELL_APPROVAL_TIMEOUT_SECONDS = 300


@dataclass
class _PendingShellApproval:
    call_id: str
    session_id: str
    command: str
    workspace: str
    created_at: float
    event: threading.Event
    approved: bool | None = None


class ShellApprovalBroker:
    """Coordinates blocking LiteRT tool hooks with HTTP approval clients."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingShellApproval] = {}

    def request(
        self,
        session_id: str,
        command: str,
        workspace: Path,
        *,
        timeout: float = SHELL_APPROVAL_TIMEOUT_SECONDS,
    ) -> bool:
        pending = _PendingShellApproval(
            call_id=uuid.uuid4().hex,
            session_id=session_id,
            command=command,
            workspace=str(workspace),
            created_at=time.time(),
            event=threading.Event(),
        )

        with self._lock:
            if session_id in self._pending:
                raise RuntimeError(
                    "A shell command is already awaiting approval for this request."
                )
            self._pending[session_id] = pending

        pending.event.wait(timeout=timeout)

        with self._lock:
            if self._pending.get(session_id) is pending:
                self._pending.pop(session_id, None)

        return pending.approved is True

    def get_pending(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is None:
                return None
            return {
                "call_id": pending.call_id,
                "command": pending.command,
                "workspace": pending.workspace,
                "created_at": pending.created_at,
                "expires_at": (
                    pending.created_at + SHELL_APPROVAL_TIMEOUT_SECONDS
                ),
            }

    def resolve(
        self,
        session_id: str,
        call_id: str,
        approved: bool,
    ) -> bool:
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is None or pending.call_id != call_id:
                return False
            self._pending.pop(session_id, None)
            pending.approved = bool(approved)
            pending.event.set()
            return True

    def deny_pending(self, session_id: str) -> bool:
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is None:
                return False
            self._pending.pop(session_id, None)
            pending.approved = False
            pending.event.set()
            return True


shell_approval_broker = ShellApprovalBroker()


def resolve_workspace_root(value: str) -> Path:
    if not value or not value.strip():
        raise ValueError("A workspace folder path is required.")

    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Workspace path is not a directory: {root}")
    return root


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _result(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + "\n... [tool result truncated]"


class Workspace:
    def __init__(self, root: Path):
        self.root = root

    def readable_path(self, value: str, *, directory: bool | None = None) -> Path:
        try:
            candidate = (self.root / (value or ".")).resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(f"Path does not exist: {value}") from exc
        if not _is_within(self.root, candidate):
            raise ValueError("Path escapes the selected workspace.")
        if directory is True and not candidate.is_dir():
            raise ValueError(f"Not a directory: {value}")
        if directory is False and not candidate.is_file():
            raise ValueError(f"Not a file: {value}")
        return candidate

    def writable_path(self, value: str) -> Path:
        if not value or value in {".", ".."}:
            raise ValueError("A file path is required.")

        candidate = self.root / value
        parent = candidate.parent.resolve(strict=True)
        if not _is_within(self.root, parent):
            raise ValueError("Path escapes the selected workspace.")

        if candidate.exists():
            resolved = candidate.resolve(strict=True)
            if not _is_within(self.root, resolved):
                raise ValueError("Path escapes the selected workspace.")
            if not resolved.is_file():
                raise ValueError(f"Not a file: {value}")
            return resolved

        return parent / candidate.name

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


class WorkspaceTool(litert_lm.Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: Callable[[dict[str, Any]], Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._execute = execute

    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, param: Any) -> Any:
        arguments = dict(param) if isinstance(param, Mapping) else {}
        return truncate_tool_content(self._execute(arguments))


class WorkspaceToolEventHandler(litert_lm.ToolEventHandler):
    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        shell_approval_id: str | None = None,
    ):
        self._workspace_root = workspace_root
        self._shell_approval_id = shell_approval_id
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._call_count = 0
            self._signature_counts: dict[str, int] = {}

    def set_shell_approval_id(self, shell_approval_id: str | None):
        """Update the per-request approval channel on reused conversations."""
        self._shell_approval_id = shell_approval_id

    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        function = tool_call.get("function", {})
        name = str(function.get("name", "<unknown>"))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
        else:
            parsed_arguments = arguments
        signature = json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        with self._lock:
            self._call_count += 1
            signature_count = self._signature_counts.get(signature, 0) + 1
            self._signature_counts[signature] = signature_count
            call_count = self._call_count

        print(
            f"\n[workspace-tool-call] {call_count}/"
            f"{_config.MAX_TOOL_CALLS_PER_GENERATION} name={name}",
            file=sys.stderr,
            flush=True,
        )

        if signature_count > MAX_IDENTICAL_TOOL_CALLS:
            raise RuntimeError(
                f"Workspace tool '{name}' repeated the same call more than "
                f"{MAX_IDENTICAL_TOOL_CALLS} times."
            )
        if call_count > _config.MAX_TOOL_CALLS_PER_GENERATION:
            raise RuntimeError(
                "Workspace tool-call limit exceeded for one generation."
            )

        if name == "shell":
            command = (
                str(parsed_arguments.get("command", "")).strip()
                if isinstance(parsed_arguments, Mapping)
                else ""
            )
            if not command:
                print(
                    "[workspace-tool-call] denied shell command: "
                    "command arguments were missing or invalid",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            if not self._shell_approval_id or self._workspace_root is None:
                print(
                    "[workspace-tool-call] denied shell command: "
                    "no approval session was supplied",
                    file=sys.stderr,
                    flush=True,
                )
                return False

            approved = shell_approval_broker.request(
                self._shell_approval_id,
                command,
                self._workspace_root,
            )
            print(
                "[workspace-tool-call] "
                f"{'approved' if approved else 'denied'} shell command",
                file=sys.stderr,
                flush=True,
            )
            return approved

        return True

    def process_tool_response(self, tool_response: Any) -> Any:
        return tool_response


def build_workspace_tools(
    root: Path,
    *,
    read_only: bool,
) -> list[WorkspaceTool]:
    root = root.resolve(strict=True)
    workspace = Workspace(root)

    def find_files(args: dict[str, Any]) -> str:
        path = workspace.readable_path(
            str(args.get("path", ".")),
            directory=True,
        )
        pattern = str(args.get("pattern", "")).strip() or "*"
        if Path(pattern).is_absolute():
            raise ValueError("Glob patterns must be relative to the workspace.")
        recursive = bool(args.get("recursive", False))

        glob_str = f"**/{pattern}" if recursive else pattern
        entries = []

        for entry in path.glob(glob_str):
            try:
                resolved = entry.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not _is_within(root, resolved):
                continue

            entries.append({
                "path": workspace.relative(entry),
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
            if len(entries) >= MAX_LIST_RESULTS:
                break

        return _result({
            "entries": entries,
            "truncated": len(entries) >= MAX_LIST_RESULTS,
        })

    def grep_files(args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            raise ValueError("A search pattern is required.")
        if len(pattern) > 500:
            raise ValueError("Search patterns are limited to 500 characters.")

        flags = re.IGNORECASE if bool(args.get("ignore_case", False)) else 0
        try:
            expression = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc

        search_path = workspace.readable_path(str(args.get("path", ".")))
        file_glob = str(args.get("file_glob", "*")) or "*"
        candidates = (
            [search_path]
            if search_path.is_file()
            else search_path.rglob(file_glob)
        )
        matches = []

        for file_path in candidates:
            if not file_path.is_file():
                continue
            try:
                resolved = file_path.resolve(strict=True)
                if not _is_within(root, resolved):
                    continue
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
                raw = file_path.read_bytes()
                if b"\0" in raw:
                    continue
                text = raw.decode("utf-8", errors="replace")
            except (OSError, UnicodeError):
                continue

            for line_number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    matches.append({
                        "path": workspace.relative(file_path),
                        "line": line_number,
                        "text": line[:1000],
                    })
                    if len(matches) >= MAX_GREP_RESULTS:
                        return _result({
                            "matches": matches,
                            "truncated": True,
                        })

        return _result({"matches": matches, "truncated": False})

    def read_file(args: dict[str, Any]) -> str:
        file_path = workspace.readable_path(
            str(args.get("path", "")),
            directory=False,
        )
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES}-byte read limit."
            )

        start_line = max(1, int(args.get("start_line", 1)))
        max_lines = min(
            MAX_READ_LINES,
            max(1, int(args.get("max_lines", 400))),
        )
        raw = file_path.read_bytes()
        if b"\0" in raw:
            raise ValueError("Binary files cannot be read.")

        lines = raw.decode("utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1:start_line - 1 + max_lines]
        numbered = [
            f"{number}: {line}"
            for number, line in enumerate(selected, start_line)
        ]
        return _result({
            "path": workspace.relative(file_path),
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "total_lines": len(lines),
            "content": "\n".join(numbered),
            "truncated": start_line - 1 + len(selected) < len(lines),
        })

    def line_count(args: dict[str, Any]) -> str:
        file_path = workspace.readable_path(
            str(args.get("path", "")),
            directory=False,
        )
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES}-byte read limit."
            )

        raw = file_path.read_bytes()
        if b"\0" in raw:
            raise ValueError("Binary files cannot be counted.")

        return _result({
            "path": workspace.relative(file_path),
            "lines": len(raw.decode("utf-8", errors="replace").splitlines()),
            "bytes": file_size,
        })

    def size_file(args: dict[str, Any]) -> str:
        file_path = workspace.readable_path(
            str(args.get("path", "")),
            directory=False,
        )
        file_size = file_path.stat().st_size
        return _result({
            "path": workspace.relative(file_path),
            "bytes": file_size,
        })

    def sort_file(args: dict[str, Any]) -> str:
        file_path = workspace.readable_path(
            str(args.get("path", "")),
            directory=False,
        )
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES}-byte read limit."
            )

        raw = file_path.read_bytes()
        if b"\0" in raw:
            raise ValueError("Binary files cannot be sorted.")

        lines = raw.decode("utf-8", errors="replace").splitlines()
        reverse = bool(args.get("reverse", False))
        unique = bool(args.get("unique", False))

        if unique:
            lines = list(dict.fromkeys(lines))
        lines.sort(reverse=reverse)

        return _result({
            "path": workspace.relative(file_path),
            "lines": len(lines),
            "content": "\n".join(lines),
        })

    def head_file(args: dict[str, Any]) -> str:
        file_path = workspace.readable_path(
            str(args.get("path", "")),
            directory=False,
        )
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES}-byte read limit."
            )

        raw = file_path.read_bytes()
        if b"\0" in raw:
            raise ValueError("Binary files cannot be read.")

        n = max(1, int(args.get("lines", 10)))
        lines = raw.decode("utf-8", errors="replace").splitlines()
        head = lines[:n]

        return _result({
            "path": workspace.relative(file_path),
            "lines": len(head),
            "content": "\n".join(head),
        })

    def tail_file(args: dict[str, Any]) -> str:
        file_path = workspace.readable_path(
            str(args.get("path", "")),
            directory=False,
        )
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES}-byte read limit."
            )

        raw = file_path.read_bytes()
        if b"\0" in raw:
            raise ValueError("Binary files cannot be read.")

        n = max(1, int(args.get("lines", 10)))
        lines = raw.decode("utf-8", errors="replace").splitlines()
        tail = lines[-n:] if n < len(lines) else lines

        return _result({
            "path": workspace.relative(file_path),
            "lines": len(tail),
            "content": "\n".join(tail),
        })

    def os_info(args: dict[str, Any]) -> str:
        return _result({
            "os": sys.platform,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        })

    def run_shell(args: dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("A shell command is required.")

        timeout = float(args.get("timeout", DEFAULT_SHELL_TIMEOUT))
        if timeout <= 0 or timeout > 300:
            raise ValueError("Timeout must be between 1 and 300 seconds.")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                cwd=str(root),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _result({
                "exit_code": None,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s.",
                "timed_out": True,
            })

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        if len(stdout.encode("utf-8")) > MAX_SHELL_OUTPUT_BYTES:
            stdout = stdout[:MAX_SHELL_OUTPUT_BYTES] + "\n... [stdout truncated]"
        if len(stderr.encode("utf-8")) > MAX_SHELL_OUTPUT_BYTES:
            stderr = stderr[:MAX_SHELL_OUTPUT_BYTES] + "\n... [stderr truncated]"

        return _result({
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
        })

    def write_file(args: dict[str, Any]) -> str:
        file_path = workspace.writable_path(str(args.get("path", "")))
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string.")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(
                f"Content exceeds the {MAX_FILE_BYTES}-byte write limit."
            )

        mode = str(args.get("mode", "overwrite")).lower()
        if mode not in {"overwrite", "append"}:
            raise ValueError("mode must be 'overwrite' or 'append'.")

        existed = file_path.exists()
        with file_path.open("a" if mode == "append" else "w", encoding="utf-8") as handle:
            handle.write(content)

        return _result({
            "path": workspace.relative(file_path),
            "operation": (
                "appended" if mode == "append"
                else "overwritten" if existed
                else "created"
            ),
            "bytes_written": len(content.encode("utf-8")),
        })

    def edit_file(args: dict[str, Any]) -> str:
        file_path = workspace.writable_path(str(args.get("path", "")))
        old_text = args.get("oldText")
        new_text = args.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValueError("oldText and newText must be strings.")
        if not old_text:
            raise ValueError("oldText must not be empty.")

        raw = file_path.read_bytes()
        if b"\0" in raw:
            raise ValueError("Binary files cannot be edited.")
        text = raw.decode("utf-8", errors="replace")

        count = text.count(old_text)
        if count == 0:
            raise ValueError("oldText not found in file.")
        if count > 1:
            raise ValueError(
                f"oldText matches {count} locations in file; must be unique."
            )

        replaced = text.replace(old_text, new_text, 1)
        if len(replaced.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(
                f"Resulting file exceeds the {MAX_FILE_BYTES}-byte limit."
            )

        with file_path.open("w", encoding="utf-8") as handle:
            handle.write(replaced)

        return _result({
            "path": workspace.relative(file_path),
            "operation": "edited",
            "bytes_written": len(replaced.encode("utf-8")),
        })


    string_property = {"type": "string"}
    tools = [
        WorkspaceTool(
            "find",
            "Find files and directories inside the workspace. Supports glob patterns and recursive search.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative directory path. Defaults to '.'.",
                    },
                    "pattern": {
                        **string_property,
                        "description": "Glob pattern such as '*.py' or '**/*.md'. Defaults to '*'.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursively search descendants. When true, pattern is matched at any depth.",
                    },
                },
            },
            find_files,
        ),
        WorkspaceTool(
            "grep",
            "Search text files in the workspace using a regular expression.",
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        **string_property,
                        "description": "Python-compatible regular expression.",
                    },
                    "path": {
                        **string_property,
                        "description": "Relative file or directory. Defaults to '.'.",
                    },
                    "file_glob": {
                        **string_property,
                        "description": "File glob such as '*.py'. Defaults to '*'.",
                    },
                    "ignore_case": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
            grep_files,
        ),
        WorkspaceTool(
            "read",
            "Read a UTF-8 text file inside the workspace with line numbers.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative file path.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_READ_LINES,
                    },
                },
                "required": ["path"],
            },
            read_file,
        ),
        WorkspaceTool(
            "line_count",
            "Count the number of lines in a UTF-8 text file in the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative file path.",
                    },
                },
                "required": ["path"],
            },
            line_count,
        ),
        WorkspaceTool(
            "size",
            "Get the size of a file in bytes.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative file path.",
                    },
                },
                "required": ["path"],
            },
            size_file,
        ),
        WorkspaceTool(
            "sort",
            "Sort lines of a UTF-8 text file. Supports reversing order and deduplication.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative file path.",
                    },
                    "reverse": {
                        "type": "boolean",
                        "description": "Sort in descending order.",
                    },
                    "unique": {
                        "type": "boolean",
                        "description": "Remove duplicate lines before sorting.",
                    },
                },
                "required": ["path"],
            },
            sort_file,
        ),
        WorkspaceTool(
            "head",
            "Return the first N lines of a UTF-8 text file.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative file path.",
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of lines to return. Defaults to 10.",
                    },
                },
                "required": ["path"],
            },
            head_file,
        ),
        WorkspaceTool(
            "tail",
            "Return the last N lines of a UTF-8 text file.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative file path.",
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of lines to return. Defaults to 10.",
                    },
                },
                "required": ["path"],
            },
            tail_file,
        ),
        WorkspaceTool(
            "platform",
            "Return the operating system and platform information the server is running on.",
            {
                "type": "object",
                "properties": {},
            },
            os_info,
        ),
    ]

    if not read_only:
        tools.append(
            WorkspaceTool(
                "write",
                "Create, overwrite, or append to a UTF-8 file in the workspace.",
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            **string_property,
                            "description": (
                                "Relative file path. Its parent directory "
                                "must already exist."
                            ),
                        },
                        "content": {
                            **string_property,
                            "description": "Complete text to write.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["overwrite", "append"],
                        },
                    },
                    "required": ["path", "content"],
                },
                write_file,
            )
        )
        tools.append(
            WorkspaceTool(
                "edit",
                "Edit a UTF-8 text file by exact string replacement. Finds the first unique occurrence of oldText and replaces it with newText.",
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            **string_property,
                            "description": "Relative file path to edit.",
                        },
                        "oldText": {
                            **string_property,
                            "description": "Exact text to find and replace. Must match exactly one location in the file.",
                        },
                        "newText": {
                            **string_property,
                            "description": "Replacement text.",
                        },
                    },
                    "required": ["path", "oldText", "newText"],
                },
                edit_file,
            )
        )
        tools.append(
            WorkspaceTool(
                "shell",
                "Execute a shell command inside the workspace directory. Uses the system shell (/bin/sh on Unix, cmd.exe on Windows). Returns exit code, stdout, and stderr. Commands run with a default 30-second timeout.",
                {
                    "type": "object",
                    "properties": {
                        "command": {
                            **string_property,
                            "description": "Shell command to execute.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (1-300). Defaults to 30.",
                        },
                    },
                    "required": ["command"],
                },
                run_shell,
            )
        )

    return tools


def workspace_tool_definitions(
    root: Path,
    *,
    read_only: bool,
) -> list[dict[str, Any]]:
    return [
        tool.get_tool_description()
        for tool in build_workspace_tools(root, read_only=read_only)
    ]
