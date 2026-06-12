import unittest
from unittest.mock import MagicMock, patch

from whisper_captioner.llm_handler import (
    _gemini_native_proofread,
    ensure_nuc_ollama_ready,
    llm_proofread,
    test_llm_connection,
    wake_on_lan_nuc,
)
from whisper_captioner.models import LLM_PROVIDERS, SubtitleSegment, resolved_llm_api_key


class LLMHandlerTest(unittest.TestCase):
    def test_environment_api_key_overrides_saved_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "new-key"}):
            self.assertEqual(
                resolved_llm_api_key("gemini_flash", "old-key"),
                "new-key",
            )

    def test_saved_api_key_is_used_without_environment_override(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                resolved_llm_api_key("gemini_flash", "saved-key"),
                "saved-key",
            )

    def test_wol_sends_global_and_subnet_broadcasts(self):
        sock = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = sock
        context.__exit__.return_value = False
        with patch("whisper_captioner.llm_handler.socket.socket", return_value=context):
            wake_on_lan_nuc()

        targets = [call.args[1] for call in sock.sendto.call_args_list]
        self.assertIn(("255.255.255.255", 9), targets)
        self.assertIn(("192.168.31.255", 9), targets)

    def test_ensure_nuc_ollama_wakes_then_waits_until_ready(self):
        with (
            patch(
                "whisper_captioner.llm_handler._nuc_ollama_is_ready",
                side_effect=[False, False, True],
            ),
            patch("whisper_captioner.llm_handler.wake_on_lan_nuc") as wake,
            patch("whisper_captioner.llm_handler.time.sleep"),
        ):
            ensure_nuc_ollama_ready(timeout=10)

        wake.assert_called_once_with()

    def test_nuc_proofread_waits_for_ollama_before_request(self):
        provider = next(
            item for item in LLM_PROVIDERS if item.key == "nuc_ollama_gemma4"
        )
        response = '{"message":{"content":"1: 修正文本"}}'
        with (
            patch("whisper_captioner.llm_handler.ensure_nuc_ollama_ready") as ready,
            patch("whisper_captioner.llm_handler._llm_request", return_value=response),
        ):
            result = llm_proofread(
                [SubtitleSegment(0, 1, "原始文本")],
                provider,
                "",
            )

        ready.assert_called_once_with()
        self.assertEqual(result[0].text, "修正文本")

    def test_local_ollama_provider_uses_nothink_chat_request(self):
        provider = next(
            item for item in LLM_PROVIDERS if item.key == "local_ollama_qwen35_4b"
        )
        response = '{"message":{"content":"1: 修正文本"},"eval_count":4,"eval_duration":1000000000}'
        with patch(
            "whisper_captioner.llm_handler._llm_request",
            return_value=response,
        ) as request:
            result = llm_proofread(
                [SubtitleSegment(0, 1, "原始文本")],
                provider,
                "",
                max_tokens=64,
            )

        body = request.call_args.args[1]
        self.assertFalse(body["think"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertEqual(body["options"]["num_predict"], 64)
        self.assertEqual(result[0].text, "修正文本")

    def test_local_ollama_connection_allows_cold_start(self):
        provider = next(
            item for item in LLM_PROVIDERS if item.key == "local_ollama_qwen35_4b"
        )
        with patch(
            "whisper_captioner.llm_handler._llm_request",
            return_value='{"message":{"content":"OK"}}',
        ) as request:
            ok, message = test_llm_connection(provider, "")

        self.assertTrue(ok)
        self.assertIn("successful", message)
        self.assertEqual(request.call_args.kwargs["timeout"], 60)

    def test_gemini_native_honors_timeout_and_output_limit(self):
        response = MagicMock()
        response.parsed.text = "1: 修正文本"
        response.text = ""
        client = MagicMock()
        client.models.generate_content.return_value = response

        with patch("whisper_captioner.llm_handler.genai.Client", return_value=client) as factory:
            result = _gemini_native_proofread(
                [SubtitleSegment(0, 1, "原始文本")],
                "test-key",
                "gemini-2.5-flash",
                "system prompt",
                timeout=120,
                max_tokens=60000,
            )

        http_options = factory.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, 120_000)
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.max_output_tokens, 60000)
        self.assertEqual(result, {0: "修正文本"})

    def test_gemini_native_failure_falls_back_to_openai_compatible_api(self):
        provider = next(
            item for item in LLM_PROVIDERS if item.key == "gemini_flash"
        )
        fallback_response = '{"choices":[{"message":{"content":"1: 降级修正"}}]}'
        with (
            patch(
                "whisper_captioner.llm_handler._gemini_native_proofread",
                return_value={},
            ) as native,
            patch(
                "whisper_captioner.llm_handler._llm_request",
                return_value=fallback_response,
            ) as fallback,
        ):
            result = llm_proofread(
                [SubtitleSegment(0, 1, "原始文本")],
                provider,
                "test-key",
                timeout=120,
                max_tokens=60000,
            )

        native.assert_called_once()
        fallback.assert_called_once()
        self.assertEqual(result[0].text, "降级修正")


if __name__ == "__main__":
    unittest.main()
