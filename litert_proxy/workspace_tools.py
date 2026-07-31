import json
import re
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import litert_lm

from .utils.token import truncate_tool_content


MAX_LIST_RESULTS = 200
MAX_GLOB_RESULTS = 200
MAX_GREP_RESULTS = 100
MAX_READ_LINES = 1000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOOL_RESULT_CHARS = 50_000
MAX_TOOL_CALLS_PER_GENERATION = 6
MAX_IDENTICAL_TOOL_CALLS = 2


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
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._call_count = 0
            self._signature_counts: dict[str, int] = {}

    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        function = tool_call.get("function", {})
        name = str(function.get("name", "<unknown>"))
        arguments = function.get("arguments", {})
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
            f"{MAX_TOOL_CALLS_PER_GENERATION} name={name}",
            file=sys.stderr,
            flush=True,
        )

        if signature_count > MAX_IDENTICAL_TOOL_CALLS:
            raise RuntimeError(
                f"Workspace tool '{name}' repeated the same call more than "
                f"{MAX_IDENTICAL_TOOL_CALLS} times."
            )
        if call_count > MAX_TOOL_CALLS_PER_GENERATION:
            raise RuntimeError(
                "Workspace tool-call limit exceeded for one generation."
            )
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

    def list_files(args: dict[str, Any]) -> str:
        path = workspace.readable_path(
            str(args.get("path", ".")),
            directory=True,
        )
        recursive = bool(args.get("recursive", False))
        iterator = path.rglob("*") if recursive else path.iterdir()
        entries = []

        for entry in iterator:
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

    def glob_files(args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("A glob pattern is required.")
        if Path(pattern).is_absolute():
            raise ValueError("Glob patterns must be relative to the workspace.")

        matches = []
        for entry in root.glob(pattern):
            try:
                resolved = entry.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not _is_within(root, resolved):
                continue
            matches.append(workspace.relative(entry))
            if len(matches) >= MAX_GLOB_RESULTS:
                break

        return _result({
            "matches": matches,
            "truncated": len(matches) >= MAX_GLOB_RESULTS,
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

    string_property = {"type": "string"}
    tools = [
        WorkspaceTool(
            "list",
            "List files and directories inside the selected workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        **string_property,
                        "description": "Relative directory path. Defaults to '.'.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursively list descendants.",
                    },
                },
            },
            list_files,
        ),
        WorkspaceTool(
            "glob",
            "Find workspace files and directories matching a glob pattern.",
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        **string_property,
                        "description": "Relative glob such as '**/*.py'.",
                    },
                },
                "required": ["pattern"],
            },
            glob_files,
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
