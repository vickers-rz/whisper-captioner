import json
import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock PySide6 to avoid ImportError during tests
sys.modules['PySide6'] = MagicMock()
sys.modules['PySide6.QtCore'] = MagicMock()

from whisper_captioner.llm_handler import _build_llm_call, test_llm_connection
from whisper_captioner.models import LLM_PROVIDERS
from whisper_captioner.qwen_chat_service import QwenChatServiceManager


class Gemma4ProviderTest(unittest.TestCase):
    def test_nuc_gemma4_uses_explicit_context_and_output_cap(self):
        provider = next(item for item in LLM_PROVIDERS if item.key == "nuc_ollama_gemma4")

        url, body, headers = _build_llm_call(
            provider,
            "",
            "测试正文",
            max_tokens=24_000,
        )

        self.assertEqual(provider.model_id, "gemma4:latest")
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual(headers, {"Content-Type": "application/json"})
        self.assertFalse(body["think"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["keep_alive"], "10m")
        self.assertEqual(body["options"]["num_ctx"], 16_384)
        self.assertEqual(body["options"]["num_predict"], 8_192)
        self.assertEqual(body["options"]["temperature"], 0.1)

    def test_nuc_gemma4_respects_smaller_output_cap(self):
        provider = next(item for item in LLM_PROVIDERS if item.key == "nuc_ollama_gemma4")

        url, body, headers = _build_llm_call(
            provider,
            "",
            "测试正文",
            max_tokens=1000,
        )

        self.assertEqual(body["options"]["num_predict"], 1000)

    def test_other_ollama_providers_are_not_affected(self):
        provider = next(item for item in LLM_PROVIDERS if item.key == "nuc_ollama_14b")

        url, body, headers = _build_llm_call(
            provider,
            "",
            "测试正文",
            max_tokens=24_000,
        )

        self.assertNotIn("num_ctx", body.get("options", {}))
        self.assertNotIn("keep_alive", body)
        self.assertEqual(body["options"]["num_predict"], 24_000)

    @patch("whisper_captioner.llm_handler._llm_request")
    def test_connection_allows_for_cold_model_load(self, request):
        request.return_value = json.dumps({"message": {"content": "Hello"}})
        provider = next(item for item in LLM_PROVIDERS if item.key == "nuc_ollama_gemma4")

        ok, _ = test_llm_connection(provider, "")

        self.assertTrue(ok)
        self.assertEqual(request.call_args.kwargs["timeout"], 120)

    def test_long_context_warning_for_gemma4(self):
        manager = QwenChatServiceManager(None)
        
        # Test safe limit
        safe_warning = manager._long_context_warning(char_count=15000, segment_count=300, provider_key="nuc_ollama_gemma4")
        self.assertEqual(safe_warning, "")
        
        # Test beyond comfortable limit
        long_warning = manager._long_context_warning(char_count=19000, segment_count=300, provider_key="nuc_ollama_gemma4")
        self.assertIn("建议拆分处理，或改用 Gemini 2.5 Pro", long_warning)

        long_segments_warning = manager._long_context_warning(char_count=10000, segment_count=600, provider_key="nuc_ollama_gemma4")
        self.assertIn("建议拆分处理，或改用 Gemini 2.5 Pro", long_segments_warning)


if __name__ == "__main__":
    unittest.main()
