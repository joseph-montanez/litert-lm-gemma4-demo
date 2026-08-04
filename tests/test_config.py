import os
import unittest
from unittest.mock import patch

from litert_proxy import config


class ModelRuntimeDefaultsTest(unittest.TestCase):
    def test_12b_profile_keeps_32k_context_and_sane_chat_defaults(self):
        profile = config.MODEL_REGISTRY["gemma-4-12B-it"][
            "runtime_defaults"
        ]
        original = {
            name: getattr(config, name)
            for name in profile
        }

        try:
            with patch.dict(os.environ, {}, clear=True):
                config.apply_model_runtime_defaults("gemma-4-12B-it")

            self.assertEqual(
                config.MODEL_REGISTRY["gemma-4-12B-it"]["max_num_tokens"],
                32768,
            )
            self.assertEqual(config.DEFAULT_MAX_OUTPUT_TOKENS, 4096)
            self.assertEqual(config.DEFAULT_REASONING_EFFORT, "none")
            self.assertEqual(config.DEFAULT_TEMPERATURE, 0.8)
            self.assertEqual(config.DEFAULT_TOP_P, 0.9)
            self.assertEqual(config.DEFAULT_TOP_K, 40)
            self.assertEqual(config.MAX_TOOL_RESPONSE_TOKENS, 4096)
            self.assertTrue(config.ENABLE_CONSTRAINED_DECODING)
            self.assertFalse(config.ENABLE_SPECULATIVE_DECODING)
        finally:
            for name, value in original.items():
                setattr(config, name, value)

    def test_non_profile_model_restores_baseline_defaults(self):
        profile = config.MODEL_REGISTRY["gemma-4-12B-it"][
            "runtime_defaults"
        ]
        original = {
            name: getattr(config, name)
            for name in profile
        }

        try:
            with patch.dict(os.environ, {}, clear=True):
                config.apply_model_runtime_defaults("gemma-4-12B-it")
                config.apply_model_runtime_defaults("gemma-4-E4B-it")

            self.assertEqual(
                config.DEFAULT_TEMPERATURE,
                config._RUNTIME_DEFAULT_BASELINES["DEFAULT_TEMPERATURE"],
            )
            self.assertEqual(
                config.DEFAULT_REASONING_EFFORT,
                config._RUNTIME_DEFAULT_BASELINES[
                    "DEFAULT_REASONING_EFFORT"
                ],
            )
        finally:
            for name, value in original.items():
                setattr(config, name, value)


if __name__ == "__main__":
    unittest.main()
