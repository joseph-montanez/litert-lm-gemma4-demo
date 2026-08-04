import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

from litert_proxy import config
from litert_proxy import server
from litert_proxy.models import ChatCompletionRequest
from litert_proxy.workspace_tools import ShellApprovalBroker


class ServerStartupTest(unittest.IsolatedAsyncioTestCase):
    def test_12b_gpu_engine_waits_for_long_prefill_completion(self):
        class GPU:
            pass

        kwargs = {"backend": GPU()}
        server._apply_model_engine_options(
            kwargs,
            {"backend", "enable_benchmark"},
            "gemma-4-12B-it",
            kwargs["backend"],
        )

        self.assertTrue(kwargs["enable_benchmark"])

    def test_prefill_wait_is_not_enabled_for_other_models_or_cpu(self):
        class GPU:
            pass

        class CPU:
            pass

        other_model_kwargs = {"backend": GPU()}
        server._apply_model_engine_options(
            other_model_kwargs,
            {"backend", "enable_benchmark"},
            "gemma-4-E4B-it",
            other_model_kwargs["backend"],
        )
        cpu_kwargs = {"backend": CPU()}
        server._apply_model_engine_options(
            cpu_kwargs,
            {"backend", "enable_benchmark"},
            "gemma-4-12B-it",
            cpu_kwargs["backend"],
        )

        self.assertNotIn("enable_benchmark", other_model_kwargs)
        self.assertNotIn("enable_benchmark", cpu_kwargs)

    async def test_cancel_endpoint_cancels_active_generation(self):
        primary_worker = Mock()
        primary_worker.cancel_current.return_value = True
        tool_worker = Mock()
        tool_worker.cancel_current.return_value = False

        with (
            patch.object(server, "conversation_worker", primary_worker),
            patch.object(server, "tool_conversation_worker", tool_worker),
        ):
            response = await server.cancel_conversation()

        self.assertEqual(response, {"status": "cancelling"})
        primary_worker.cancel_current.assert_called_once_with()
        tool_worker.cancel_current.assert_called_once_with()

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

    async def test_workspace_tools_route_to_dedicated_tool_worker(self):
        primary_worker = Mock()
        tool_worker = Mock()
        request = ChatCompletionRequest(
            stream=True,
            messages=[{"role": "user", "content": "list the files"}],
            workspace_tools=True,
            workspace_path=str(Path(__file__).parents[1]),
            reasoning_effort="high",
        )

        with (
            patch.object(config, "TOOL_ROUTING_ENABLED", True),
            patch.object(config, "TOOL_ROUTE_OPENAI_TOOLS", False),
            patch.object(config, "ENABLE_12B_TOOLS", True),
            patch.object(config, "CURRENT_MODEL_KEY", "gemma-4-12B-it"),
            patch.object(config, "TOOL_MODEL_KEY", "gemma-4-E4B-it"),
            patch.object(config, "TOOL_MAX_NUM_TOKENS", 8192),
            patch.object(config, "TOOL_REASONING_EFFORT", "none"),
            patch.object(config, "DEFAULT_MAX_OUTPUT_TOKENS", 3072),
            patch.object(server, "engine", object()),
            patch.object(server, "conversation_worker", primary_worker),
            patch.object(server, "tool_engine", object()),
            patch.object(server, "tool_conversation_worker", tool_worker),
        ):
            response = await server.chat_completions(request)
            stream = response.body_iterator
            first_chunk = await anext(stream)
            await stream.aclose()

        self.assertIn("gemma-4-E4B-it", first_chunk)
        primary_worker.submit.assert_not_called()
        self.assertEqual(tool_worker.submit.call_count, 1)
        job = tool_worker.submit.call_args.args[0]
        self.assertEqual(job.request.reasoning_effort, "none")
        self.assertTrue(job.cancel_event.is_set())
        tool_worker.cancel_current.assert_called_once_with()

        self.assertTrue(server._request_lock.acquire(blocking=False))
        server._request_lock.release()

    async def test_health_reports_dedicated_tool_runtime(self):
        tool_worker = Mock()
        tool_worker.status.return_value = {"runtime": "tools"}

        with (
            patch.object(config, "TOOL_ROUTING_ENABLED", True),
            patch.object(config, "TOOL_ROUTE_OPENAI_TOOLS", False),
            patch.object(config, "ENABLE_12B_TOOLS", True),
            patch.object(config, "CURRENT_MODEL_KEY", "gemma-4-12B-it"),
            patch.object(config, "TOOL_MODEL_KEY", "gemma-4-E4B-it"),
            patch.object(config, "TOOL_BACKEND", "cpu"),
            patch.object(config, "TOOL_MAX_NUM_TOKENS", 8192),
            patch.object(config, "TOOL_REASONING_EFFORT", "none"),
            patch.object(config, "MAX_TOOL_RESPONSE_TOKENS", 512),
            patch.object(server, "tool_engine", object()),
            patch.object(server, "tool_conversation_worker", tool_worker),
        ):
            response = await server.health()

        self.assertEqual(response["max_tool_response_tokens"], 512)
        self.assertTrue(response["tooling_enabled"])
        self.assertEqual(
            response["tool_routing"],
            {
                "enabled": True,
                "model_key": "gemma-4-E4B-it",
                "model": config.TOOL_MODEL_PATH,
                "backend": "cpu",
                "max_num_tokens": 8192,
                "reasoning_effort": "none",
                "loaded": True,
                "uses_primary": False,
                "load_error": config.TOOL_MODEL_LOAD_ERROR,
                "routes": ["workspace_tools"],
                "workspace_generation": "blocking-automatic-tools",
                "conversation": {"runtime": "tools"},
            },
        )

    async def test_12b_disables_and_rejects_tools_by_default(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "list files"}],
            workspace_tools=True,
            workspace_path=str(Path(__file__).parents[1]),
        )

        with (
            patch.object(config, "CURRENT_MODEL_KEY", "gemma-4-12B-it"),
            patch.object(config, "ENABLE_12B_TOOLS", False),
            patch.object(config, "TOOL_ROUTING_ENABLED", True),
            patch.object(server, "engine", object()),
            patch.object(server, "conversation_worker", Mock()),
            patch.object(server, "tool_engine", object()),
            patch.object(server, "tool_conversation_worker", Mock()),
        ):
            health = await server.health()
            with self.assertRaises(HTTPException) as raised:
                await server.chat_completions(request)

        self.assertFalse(health["tooling_enabled"])
        self.assertFalse(health["workspace_tooling_enabled"])
        self.assertTrue(health["client_tooling_enabled"])
        self.assertFalse(health["tool_routing"]["enabled"])
        self.assertFalse(health["tool_routing"]["loaded"])
        self.assertEqual(health["tool_routing"]["routes"], [])
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("disabled by default", raised.exception.detail)

    async def test_12b_allows_client_openai_tools_on_primary_by_default(self):
        primary_worker = Mock()
        request = ChatCompletionRequest(
            stream=True,
            messages=[{"role": "user", "content": "check weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        with (
            patch.object(config, "CURRENT_MODEL_KEY", "gemma-4-12B-it"),
            patch.object(config, "ENABLE_12B_TOOLS", False),
            patch.object(config, "TOOL_ROUTE_OPENAI_TOOLS", True),
            patch.object(server, "engine", object()),
            patch.object(server, "conversation_worker", primary_worker),
            patch.object(server, "tool_engine", None),
            patch.object(server, "tool_conversation_worker", None),
        ):
            response = await server.chat_completions(request)
            stream = response.body_iterator
            first_chunk = await anext(stream)
            await stream.aclose()

        self.assertIn("gemma-4-12B-it", first_chunk)
        self.assertEqual(primary_worker.submit.call_count, 1)
        job = primary_worker.submit.call_args.args[0]
        self.assertIsNotNone(job.request.tools)

    async def test_openai_tools_stay_on_primary_worker_by_default(self):
        primary_worker = Mock()
        tool_worker = Mock()
        request = ChatCompletionRequest(
            stream=True,
            messages=[{"role": "user", "content": "check the weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get the weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            ],
            max_tokens=256,
        )

        with (
            patch.object(config, "TOOL_ROUTING_ENABLED", True),
            patch.object(config, "TOOL_ROUTE_OPENAI_TOOLS", False),
            patch.object(config, "CURRENT_MODEL_KEY", "gemma-4-E4B-it"),
            patch.object(server, "engine", object()),
            patch.object(server, "conversation_worker", primary_worker),
            patch.object(server, "tool_engine", object()),
            patch.object(server, "tool_conversation_worker", tool_worker),
        ):
            response = await server.chat_completions(request)
            stream = response.body_iterator
            first_chunk = await anext(stream)
            await stream.aclose()

        self.assertIn("gemma-4-E4B-it", first_chunk)
        self.assertEqual(primary_worker.submit.call_count, 1)
        tool_worker.submit.assert_not_called()

    async def test_workspace_tools_reuse_matching_primary_model(self):
        primary_worker = Mock()
        tool_worker = Mock()
        request = ChatCompletionRequest(
            stream=True,
            messages=[{"role": "user", "content": "list the files"}],
            workspace_tools=True,
            workspace_path=str(Path(__file__).parents[1]),
            reasoning_effort="high",
            max_tokens=256,
        )

        with (
            patch.object(config, "TOOL_ROUTING_ENABLED", True),
            patch.object(config, "TOOL_ROUTE_OPENAI_TOOLS", False),
            patch.object(config, "CURRENT_MODEL_KEY", "gemma-4-E4B-it"),
            patch.object(config, "TOOL_MODEL_KEY", "gemma-4-E4B-it"),
            patch.object(config, "TOOL_REASONING_EFFORT", "none"),
            patch.object(server, "engine", object()),
            patch.object(server, "conversation_worker", primary_worker),
            patch.object(server, "tool_engine", None),
            patch.object(server, "tool_conversation_worker", tool_worker),
        ):
            response = await server.chat_completions(request)
            stream = response.body_iterator
            first_chunk = await anext(stream)
            await stream.aclose()

        self.assertIn("gemma-4-E4B-it", first_chunk)
        self.assertEqual(primary_worker.submit.call_count, 1)
        tool_worker.submit.assert_not_called()
        job = primary_worker.submit.call_args.args[0]
        self.assertEqual(job.request.reasoning_effort, "none")

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
        self.assertIn("applyToolingAvailability", html)
        self.assertIn("end_to_end_tokens_per_second", html)
        self.assertIn("delta.tool_activity", html)
        self.assertIn("Response passed back to the model", html)
        self.assertIn("PPS/TPS n/a (blocking tool run)", html)
        self.assertIn("assistant-pending-dot", html)
        self.assertIn("Assistant is processing", html)
        self.assertIn("hideAssistantPending(assistant)", html)
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
