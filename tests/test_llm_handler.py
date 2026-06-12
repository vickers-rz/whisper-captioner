import unittest
from unittest.mock import MagicMock, patch

from whisper_captioner.llm_handler import (
    ensure_nuc_ollama_ready,
    llm_proofread,
    wake_on_lan_nuc,
)
from whisper_captioner.models import LLM_PROVIDERS, SubtitleSegment


class LLMHandlerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
