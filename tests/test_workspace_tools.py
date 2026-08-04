import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from litert_proxy.engine.streaming import make_stream_tool_activity_chunk
from litert_proxy.workspace_tools import (
    build_workspace_tools,
    resolve_workspace_root,
    ShellApprovalBroker,
    WorkspaceToolEventHandler,
)


class WorkspaceToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "first line\nneedle here\nlast line\n",
            encoding="utf-8",
        )
        (self.root / "README.md").write_text("hello\n", encoding="utf-8")
        (self.root / "data.bin").write_bytes(b"\x00\x01\x02\x03")

    def tearDown(self):
        self.temp_dir.cleanup()

    def tools(self, *, read_only=True):
        return {
            tool.name: tool
            for tool in build_workspace_tools(
                self.root,
                read_only=read_only,
            )
        }

    def execute(self, tool, arguments):
        return json.loads(tool.execute(arguments))

    def test_resolves_directory(self):
        self.assertEqual(
            resolve_workspace_root(str(self.root)),
            self.root.resolve(),
        )
        with self.assertRaises(ValueError):
            resolve_workspace_root(str(self.root / "README.md"))

    def test_read_only_toolset_does_not_register_write(self):
        self.assertEqual(
            set(self.tools()),
            {"find", "grep", "read", "line_count", "size", "sort", "head", "tail", "platform"},
        )

    def test_find_grep_and_read(self):
        tools = self.tools()

        found = self.execute(
            tools["find"],
            {"path": "src", "recursive": True},
        )
        self.assertEqual(found["entries"][0]["path"], "src/app.py")

        found_glob = self.execute(
            tools["find"],
            {"pattern": "**/*.py", "recursive": True},
        )
        self.assertEqual(
            [e["path"] for e in found_glob["entries"]],
            ["src/app.py"],
        )

        grepped = self.execute(
            tools["grep"],
            {"pattern": "needle", "file_glob": "*.py"},
        )
        self.assertEqual(grepped["matches"][0]["line"], 2)

        read = self.execute(
            tools["read"],
            {"path": "src/app.py", "start_line": 2, "max_lines": 1},
        )
        self.assertEqual(read["content"], "2: needle here")

        counted = self.execute(
            tools["line_count"],
            {"path": "src/app.py"},
        )
        self.assertEqual(counted["lines"], 3)

    def test_find_excludes_common_generated_directories_by_default(self):
        for directory in (".venv", ".git", "__pycache__"):
            ignored = self.root / directory
            ignored.mkdir()
            (ignored / "ignored.py").write_text(
                "ignored = True\n",
                encoding="utf-8",
            )

        result = self.execute(
            self.tools()["find"],
            {"pattern": "**/*.py", "recursive": True},
        )
        paths = {entry["path"] for entry in result["entries"]}

        self.assertIn("src/app.py", paths)
        self.assertFalse(any("ignored.py" in path for path in paths))
        self.assertEqual(
            result["excluded_dirs"],
            [".venv", ".git", "__pycache__"],
        )

    def test_find_allows_custom_exclusions_and_empty_override(self):
        vendor = self.root / "vendor"
        vendor.mkdir()
        (vendor / "vendored.py").write_text("value = 1\n", encoding="utf-8")
        virtualenv = self.root / ".venv"
        virtualenv.mkdir()
        (virtualenv / "dependency.py").write_text(
            "value = 2\n",
            encoding="utf-8",
        )

        excluded = self.execute(
            self.tools()["find"],
            {
                "pattern": "**/*.py",
                "recursive": True,
                "exclude_dirs": ["vendor", ".venv"],
            },
        )
        included = self.execute(
            self.tools()["find"],
            {
                "pattern": "**/*.py",
                "recursive": True,
                "exclude_dirs": [],
            },
        )

        self.assertNotIn(
            "vendor/vendored.py",
            {entry["path"] for entry in excluded["entries"]},
        )
        included_paths = {entry["path"] for entry in included["entries"]}
        self.assertIn("vendor/vendored.py", included_paths)
        self.assertIn(".venv/dependency.py", included_paths)

    def test_grep_prunes_excluded_directories(self):
        generated = self.root / "src" / "generated"
        generated.mkdir()
        (generated / "generated.py").write_text(
            "needle generated\n",
            encoding="utf-8",
        )
        cache = self.root / "src" / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text(
            "needle cached\n",
            encoding="utf-8",
        )

        result = self.execute(
            self.tools()["grep"],
            {
                "pattern": "needle",
                "file_glob": "*.py",
                "exclude_dirs": ["generated", "__pycache__"],
            },
        )

        self.assertEqual(
            [match["path"] for match in result["matches"]],
            ["src/app.py"],
        )

    def test_find_and_grep_schemas_document_exclude_dirs(self):
        definitions = {
            tool.name: tool.get_tool_description()["function"]
            for tool in self.tools().values()
        }

        for name in ("find", "grep"):
            exclude_schema = definitions[name]["parameters"]["properties"][
                "exclude_dirs"
            ]
            self.assertEqual(exclude_schema["type"], "array")
            self.assertEqual(
                exclude_schema["default"],
                [".venv", ".git", "__pycache__"],
            )

    def test_sort_tool(self):
        tools = self.tools()
        (self.root / "items.txt").write_text(
            "c\na\nb\na\n",
            encoding="utf-8",
        )

        result = self.execute(tools["sort"], {"path": "items.txt"})
        self.assertEqual(result["content"], "a\na\nb\nc")

        result = self.execute(
            tools["sort"],
            {"path": "items.txt", "reverse": True, "unique": True},
        )
        self.assertEqual(result["content"], "c\nb\na")

    def test_head_tool(self):
        tools = self.tools()
        (self.root / "numbers.txt").write_text(
            "1\n2\n3\n4\n5\n",
            encoding="utf-8",
        )

        result = self.execute(tools["head"], {"path": "numbers.txt", "lines": 3})
        self.assertEqual(result["content"], "1\n2\n3")
        self.assertEqual(result["lines"], 3)

    def test_tail_tool(self):
        tools = self.tools()
        (self.root / "numbers.txt").write_text(
            "1\n2\n3\n4\n5\n",
            encoding="utf-8",
        )

        result = self.execute(tools["tail"], {"path": "numbers.txt", "lines": 2})
        self.assertEqual(result["content"], "4\n5")
        self.assertEqual(result["lines"], 2)

    def test_platform_tool(self):
        tools = self.tools()
        result = self.execute(tools["platform"], {})
        self.assertIn("os", result)
        self.assertIn("system", result)

    def test_size_tool_returns_bytes(self):
        tools = self.tools()
        result = self.execute(tools["size"], {"path": "data.bin"})
        self.assertEqual(result["bytes"], 4)

        result = self.execute(tools["size"], {"path": "src/app.py"})
        self.assertEqual(result["bytes"], 33)

    def test_line_count_handles_empty_and_unterminated_files(self):
        tools = self.tools()
        (self.root / "empty.txt").write_bytes(b"")
        (self.root / "unterminated.txt").write_text(
            "first\nsecond",
            encoding="utf-8",
        )

        empty = self.execute(
            tools["line_count"],
            {"path": "empty.txt"},
        )
        unterminated = self.execute(
            tools["line_count"],
            {"path": "unterminated.txt"},
        )

        self.assertEqual(empty["lines"], 0)
        self.assertEqual(unterminated["lines"], 2)

    def test_edit_tool_exact_replacement(self):
        tools = self.tools(read_only=False)
        edit = tools["edit"]
        (self.root / "config.ini").write_text(
            "[server]\nport = 8080\nhost = localhost\n",
            encoding="utf-8",
        )

        result = self.execute(
            edit,
            {
                "path": "config.ini",
                "oldText": "port = 8080",
                "newText": "port = 9090",
            },
        )
        self.assertEqual(result["operation"], "edited")
        self.assertEqual(
            (self.root / "config.ini").read_text(encoding="utf-8"),
            "[server]\nport = 9090\nhost = localhost\n",
        )

    def test_edit_tool_rejects_duplicate_oldtext(self):
        tools = self.tools(read_only=False)
        edit = tools["edit"]
        (self.root / "dupes.txt").write_text(
            "dup\nmiddle\ndup\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "2 locations"):
            self.execute(
                edit,
                {"path": "dupes.txt", "oldText": "dup", "newText": "x"},
            )

    def test_edit_tool_rejects_missing_oldtext(self):
        tools = self.tools(read_only=False)
        edit = tools["edit"]
        (self.root / "data.txt").write_text("hello", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not found"):
            self.execute(
                edit,
                {"path": "data.txt", "oldText": "gone", "newText": "x"},
            )

    def test_shell_tool_executes_command(self):
        tools = self.tools(read_only=False)
        shell = tools["shell"]

        result = self.execute(
            shell,
            {"command": "echo hello"},
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello", result["stdout"])

    def test_shell_tool_reports_failure(self):
        tools = self.tools(read_only=False)
        shell = tools["shell"]

        result = self.execute(
            shell,
            {"command": "exit 1"},
        )
        self.assertEqual(result["exit_code"], 1)

    def test_shell_call_is_denied_without_approval_session(self):
        handler = WorkspaceToolEventHandler(workspace_root=self.root)

        self.assertFalse(handler.approve_tool_call({
            "function": {
                "name": "shell",
                "arguments": {"command": "echo should-not-run"},
            }
        }))

    def test_shell_approval_broker_exposes_and_resolves_exact_command(self):
        broker = ShellApprovalBroker()
        result = []

        waiter = threading.Thread(
            target=lambda: result.append(
                broker.request(
                    "browser-session",
                    "echo hello && pwd",
                    self.root,
                    timeout=1,
                )
            )
        )
        waiter.start()

        deadline = time.monotonic() + 1
        pending = None
        while pending is None and time.monotonic() < deadline:
            pending = broker.get_pending("browser-session")
            if pending is None:
                time.sleep(0.01)

        self.assertIsNotNone(pending)
        self.assertEqual(pending["command"], "echo hello && pwd")
        self.assertEqual(pending["workspace"], str(self.root))
        self.assertTrue(
            broker.resolve(
                "browser-session",
                pending["call_id"],
                True,
            )
        )
        waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [True])

    def test_write_tool_creates_and_appends(self):
        write = self.tools(read_only=False)["write"]

        created = self.execute(
            write,
            {"path": "notes.txt", "content": "one", "mode": "overwrite"},
        )
        self.assertEqual(created["operation"], "created")

        appended = self.execute(
            write,
            {"path": "notes.txt", "content": "\ntwo", "mode": "append"},
        )
        self.assertEqual(appended["operation"], "appended")
        self.assertEqual(
            (self.root / "notes.txt").read_text(encoding="utf-8"),
            "one\ntwo",
        )

    def test_read_and_write_reject_parent_escape(self):
        tools = self.tools(read_only=False)

        with self.assertRaises(ValueError):
            tools["read"].execute({"path": "../outside.txt"})
        with self.assertRaises(ValueError):
            tools["write"].execute({
                "path": "../outside.txt",
                "content": "no",
            })

    def test_symlink_escape_is_rejected(self):
        outside_dir = Path(self.temp_dir.name).parent
        outside_file = outside_dir / "litert-workspace-tools-outside.txt"
        outside_file.write_text("secret", encoding="utf-8")
        link = self.root / "outside-link"

        try:
            link.symlink_to(outside_file)
            tools = self.tools(read_only=False)
            with self.assertRaises(ValueError):
                tools["read"].execute({"path": "outside-link"})
            with self.assertRaises(ValueError):
                tools["write"].execute({
                    "path": "outside-link",
                    "content": "no",
                })
        finally:
            link.unlink(missing_ok=True)
            outside_file.unlink(missing_ok=True)

    def test_repeated_native_tool_calls_are_stopped(self):
        handler = WorkspaceToolEventHandler()
        find_call = {
            "function": {
                "name": "find",
                "arguments": {"path": ".", "recursive": True},
            }
        }

        self.assertTrue(handler.approve_tool_call(find_call))
        self.assertTrue(handler.approve_tool_call(find_call))
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            handler.approve_tool_call(find_call)

    def test_tool_lifecycle_events_include_arguments_and_returned_response(self):
        handler = WorkspaceToolEventHandler()
        events = []
        handler.set_event_callback(events.append)
        tool_call = {
            "id": "call-1",
            "function": {
                "name": "find",
                "arguments": {"path": ".", "pattern": "*.py"},
            },
        }

        self.assertTrue(handler.approve_tool_call(tool_call))
        response = '{"entries":["app.py"]}'
        self.assertIs(handler.process_tool_response(response), response)

        self.assertEqual(events[0]["phase"], "call")
        self.assertEqual(events[0]["name"], "find")
        self.assertEqual(
            events[0]["arguments"],
            {"path": ".", "pattern": "*.py"},
        )
        self.assertEqual(events[1]["phase"], "result")
        self.assertEqual(events[1]["result"], response)
        self.assertFalse(events[1]["truncated"])

    def test_tool_activity_stream_chunk_uses_delta_extension(self):
        chunk = make_stream_tool_activity_chunk(
            "gemma-4-12B-it",
            {"id": "call-1", "phase": "call", "name": "find"},
        )
        payload = json.loads(chunk.removeprefix("data: ").strip())

        activity = payload["choices"][0]["delta"]["tool_activity"]
        self.assertEqual(activity["name"], "find")
        self.assertEqual(activity["phase"], "call")


if __name__ == "__main__":
    unittest.main()
