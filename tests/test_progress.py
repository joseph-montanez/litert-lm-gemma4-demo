import unittest
from unittest.mock import patch

from litert_proxy.engine.progress import ConsoleProgress


class ConsoleProgressTest(unittest.TestCase):
    def test_buffered_callbacks_report_end_to_end_rate(self):
        progress = ConsoleProgress(20, 20, "reset")
        progress.started_at = 10.0
        progress.first_token_at = 19.999
        progress.output_parts = ["finished response"]
        progress.generated_parts = ["finished response"]

        with (
            patch(
                "litert_proxy.engine.progress.count_tokens",
                side_effect=[100, 0, 100],
            ),
            patch(
                "litert_proxy.engine.progress.time.perf_counter",
                return_value=20.0,
            ),
            patch.object(progress, "_write"),
        ):
            usage = progress.finish()

        self.assertTrue(usage["buffered_response"])
        self.assertIsNone(usage["time_to_first_token"])
        self.assertIsNone(usage["prefill_tokens_per_second"])
        self.assertIsNone(usage["decode_tokens_per_second"])
        self.assertAlmostEqual(
            usage["end_to_end_tokens_per_second"],
            10.0,
        )

    def test_genuine_streaming_keeps_decode_rate(self):
        progress = ConsoleProgress(20, 20, "reset")
        progress.started_at = 5.0
        progress.first_token_at = 10.0
        progress.output_parts = ["streamed response"]
        progress.generated_parts = ["streamed response"]

        with (
            patch(
                "litert_proxy.engine.progress.count_tokens",
                side_effect=[100, 0, 100],
            ),
            patch(
                "litert_proxy.engine.progress.time.perf_counter",
                return_value=20.0,
            ),
            patch.object(progress, "_write"),
        ):
            usage = progress.finish()

        self.assertFalse(usage["buffered_response"])
        self.assertEqual(usage["time_to_first_token"], 5.0)
        self.assertEqual(usage["decode_tokens_per_second"], 10.0)

    def test_blocking_tool_run_reports_only_end_to_end_rate(self):
        progress = ConsoleProgress(20, 20, "tool-separate")
        progress.started_at = 5.0
        progress.output_parts = ["final tool answer"]
        progress.generated_parts = ["final tool answer"]
        progress.set_blocking_generation_timing(6.0, 16.0)

        with (
            patch(
                "litert_proxy.engine.progress.count_tokens",
                side_effect=[100, 0, 100],
            ),
            patch(
                "litert_proxy.engine.progress.time.perf_counter",
                return_value=20.0,
            ),
            patch.object(progress, "_write"),
        ):
            usage = progress.finish()

        self.assertTrue(usage["blocking_response"])
        self.assertEqual(usage["generation_seconds"], 10.0)
        self.assertEqual(usage["generation_tokens_per_second"], 10.0)
        self.assertIsNone(usage["prefill_tokens_per_second"])
        self.assertIsNone(usage["decode_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
