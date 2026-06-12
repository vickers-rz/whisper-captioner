import unittest
from unittest.mock import patch

from whisper_captioner.qwen_chat_service import QwenChatServiceManager


class _FakeSettings:
    def __init__(self, values=None):
        self._values = values or {}

    def value(self, key, default=None):
        return self._values.get(key, default)


class QwenChatServiceTest(unittest.TestCase):
    def test_config_payload_marks_local_ollama_ready(self):
        manager = QwenChatServiceManager()
        with patch(
            "whisper_captioner.qwen_chat_service.QSettings",
            return_value=_FakeSettings(),
        ):
            payload = manager._config_payload()

        provider = next(
            item
            for item in payload["providers"]
            if item["key"] == "local_ollama_qwen35_4b"
        )
        self.assertTrue(provider["ready"])
        self.assertEqual(provider["model_id"], "qwen3.5:4b")

    def test_provider_help_text_mentions_local_ollama(self):
        manager = QwenChatServiceManager()
        text = manager._provider_help_text("local_ollama_qwen35_4b")
        self.assertIn("Ollama", text)
        self.assertIn("不需要 API Key", text)

    def test_long_context_warning_is_generic_for_local_models(self):
        manager = QwenChatServiceManager()
        text = manager._long_context_warning(
            char_count=50000,
            segment_count=1300,
            provider_key="local_ollama_qwen35_4b",
        )
        self.assertIn("本地小中型本机模型", text)
        self.assertNotIn("Qwen3-8B", text)


if __name__ == "__main__":
    unittest.main()
