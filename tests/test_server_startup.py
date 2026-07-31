import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from litert_proxy import config
from litert_proxy import server
from litert_proxy.workspace_tools import ShellApprovalBroker


class ServerStartupTest(unittest.IsolatedAsyncioTestCase):
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
