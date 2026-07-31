import json
import tempfile
import unittest
from pathlib import Path

from litert_proxy.workspace_tools import (
    build_workspace_tools,
    resolve_workspace_root,
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
            {"list", "glob", "grep", "read", "line_count"},
        )

    def test_list_glob_grep_and_read(self):
        tools = self.tools()

        listed = self.execute(
            tools["list"],
            {"path": "src", "recursive": True},
        )
        self.assertEqual(listed["entries"][0]["path"], "src/app.py")

        globbed = self.execute(tools["glob"], {"pattern": "**/*.py"})
        self.assertEqual(globbed["matches"], ["src/app.py"])

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
        call = {
            "function": {
                "name": "list",
                "arguments": {"path": ".", "recursive": True},
            }
        }

        self.assertTrue(handler.approve_tool_call(call))
        self.assertTrue(handler.approve_tool_call(call))
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            handler.approve_tool_call(call)


if __name__ == "__main__":
    unittest.main()
