import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from litert_proxy import config
from litert_proxy import server
from litert_proxy.models import ChatCompletionRequest
from litert_proxy.workspace_tools import ShellApprovalBroker


class ServerStartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_endpoint_cancels_active_generation(self):
        worker = Mock()
        worker.cancel_current.return_value = True

        with patch.object(server, "conversation_worker", worker):
            response = await server.cancel_conversation()

        self.assertEqual(response, {"status": "cancelling"})
        worker.cancel_current.assert_called_once_with()

    async def test_closing_stream_cancels_backend_job(self):
        worker = Mock()
        request = ChatCompletionRequest(
            stream=True,
            messages=[{"role": "user", "content": "keep going"}],
        )

        with (
            patch.object(server, "engine", object()),
            patch.object(server, "conversation_worker", worker),
        ):
            response = await server.chat_completions(request)
            stream = response.body_iterator
            await anext(stream)
            await stream.aclose()

        job = worker.submit.call_args.args[0]
        self.assertTrue(job.cancel_event.is_set())
        worker.cancel_current.assert_called_once_with()

        self.assertTrue(server._request_lock.acquire(blocking=False))
        server._request_lock.release()

    async def test_shell_approval_endpoints_resolve_waiting_command(self):
        broker = ShellApprovalBroker()
        result = []

        waiter = threading.Thread(
            target=lambda: result.append(
                broker.request(
                    "web-session",
                    "echo exact command",
                    Path("/tmp"),
                    timeout=1,
                )
            )
        )

        with patch.object(server, "shell_approval_broker", broker):
            waiter.start()
            for _ in range(100):
                response = await server.get_shell_approval("web-session")
                if response["pending"] is not None:
                    break
                await asyncio.sleep(0.01)

            pending = response["pending"]
            self.assertIsNotNone(pending)
            self.assertEqual(pending["command"], "echo exact command")
            decision = await server.decide_shell_approval(
                "web-session",
                pending["call_id"],
                server.ShellApprovalDecision(approved=True),
            )

        waiter.join(timeout=1)
        self.assertEqual(decision["status"], "approved")
        self.assertEqual(result, [True])

    async def test_highlight_javascript_is_served_from_vendor_bundle(self):
        with patch.object(config, "WEB_UI_ENABLED", True):
            response = await server.highlight_javascript()

        self.assertEqual(Path(response.path), server._highlight_js_path)
        self.assertEqual(
            response.media_type,
            "application/javascript",
        )
        self.assertIn("immutable", response.headers["cache-control"])

    async def test_web_chat_exposes_fresh_chat_compaction(self):
        server._web_chat_html = ""

        with patch.object(config, "WEB_UI_ENABLED", True):
            response = await server.web_chat()

        html = response.body.decode("utf-8")
        self.assertIn('id="compactChat"', html)
        self.assertIn('id="compactGoal"', html)
        self.assertIn('id="contextMeter"', html)
        self.assertIn('id="contextProgress"', html)
        self.assertIn("Create a compact handoff for a fresh conversation", html)
        self.assertIn("Continue from the compact handoff", html)
        self.assertIn("isNearMessageBottom", html)
        self.assertIn("BOTTOM_FOLLOW_THRESHOLD", html)
        self.assertIn('fetch("/v1/conversation/reset"', html)

    async def test_lifespan_serves_while_model_loads(self):
        started = threading.Event()
        finish = threading.Event()

        def slow_initialize():
            started.set()
            finish.wait(timeout=2)

        with patch.object(
            server,
            "_initialize_model_runtime",
            slow_initialize,
        ):
            manager = server.lifespan(server.app)
            await manager.__aenter__()
            try:
                loaded = await asyncio.to_thread(started.wait, 1)
                self.assertTrue(loaded)
                self.assertTrue(config.MODEL_LOADING)
                self.assertEqual(
                    (await server.health())["status"],
                    "loading",
                )
            finally:
                finish.set()
                await manager.__aexit__(None, None, None)

        self.assertFalse(config.MODEL_LOADING)


if __name__ == "__main__":
    unittest.main()
