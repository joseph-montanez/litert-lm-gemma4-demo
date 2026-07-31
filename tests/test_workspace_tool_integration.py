import tempfile
import unittest
from unittest.mock import patch

from litert_proxy.engine.sampling import (
    build_conversation_kwargs,
    conversation_config_signature,
)
from litert_proxy.models import ChatCompletionRequest


class FakeEngine:
    def create_conversation(
        self,
        *,
        messages=None,
        tools=None,
        automatic_tool_calling=False,
        extra_context=None,
        filter_channel_content_from_kv_cache=False,
        sampler_config=None,
        enable_constrained_decoding=False,
        tool_event_handler=None,
    ):
        raise AssertionError("This signature is inspected, not called.")


class WorkspaceToolIntegrationTest(unittest.TestCase):
    def request(self, root, *, read_only=True):
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "Inspect the project."}],
            workspace_tools=True,
            workspace_path=root,
            workspace_read_only=read_only,
            workspace_shell_approval_id="test-session",
        )

    def test_native_tools_are_registered_for_automatic_execution(self):
        with tempfile.TemporaryDirectory() as root:
            request = self.request(root)
            with patch("litert_proxy.config.engine", FakeEngine()):
                kwargs = build_conversation_kwargs(request, [])

        self.assertEqual(
            [tool.name for tool in kwargs["tools"]],
            ["find", "grep", "read", "line_count", "size", "sort", "head", "tail", "platform"],
        )
        self.assertTrue(kwargs["automatic_tool_calling"])
        self.assertIn("tool_event_handler", kwargs)

    def test_shell_approval_session_reaches_tool_event_handler(self):
        with tempfile.TemporaryDirectory() as root:
            request = self.request(root, read_only=False)
            with patch("litert_proxy.config.engine", FakeEngine()):
                kwargs = build_conversation_kwargs(request, [])

        handler = kwargs["tool_event_handler"]
        self.assertEqual(handler._shell_approval_id, "test-session")

    def test_write_mode_changes_conversation_identity(self):
        with tempfile.TemporaryDirectory() as root:
            read_only = conversation_config_signature(
                self.request(root, read_only=True)
            )
            read_write = conversation_config_signature(
                self.request(root, read_only=False)
            )

        self.assertNotEqual(read_only, read_write)

    def test_approval_session_does_not_invalidate_conversation_cache(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.request(root, read_only=False)
            second = self.request(root, read_only=False)
            second.workspace_shell_approval_id = "next-request-session"

            self.assertEqual(
                conversation_config_signature(first),
                conversation_config_signature(second),
            )


if __name__ == "__main__":
    unittest.main()
