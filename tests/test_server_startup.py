import asyncio
import threading
import unittest
from unittest.mock import patch

from litert_proxy import config
from litert_proxy import server


class ServerStartupTest(unittest.IsolatedAsyncioTestCase):
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
