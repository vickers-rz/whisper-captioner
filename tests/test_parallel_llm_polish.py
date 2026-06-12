import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "parallel_llm_polish.py"
SPEC = importlib.util.spec_from_file_location("parallel_llm_polish_for_tests", MODULE_PATH)
assert SPEC and SPEC.loader
polish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(polish)


class ParallelLlmPolishTest(unittest.TestCase):
    def test_native_retry_exhaustion_falls_back_to_urllib(self):
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("native unavailable")
        fallback_payload = {
            "choices": [{"message": {"content": "1: 降级修正"}}],
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            fallback_payload
        ).encode("utf-8")

        with (
            patch.object(polish, "GEMINI_API_KEY", "test-key"),
            patch.object(polish.genai, "Client", return_value=client) as factory,
            patch.object(polish.urllib.request, "urlopen", return_value=response) as urlopen,
            patch.object(polish.time, "sleep"),
            patch("sys.stdout", new=io.StringIO()),
        ):
            result = polish.polish_batch(
                0,
                [{"text": "原始文本"}],
                1,
                "system prompt",
            )

        http_options = factory.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, 90_000)
        self.assertEqual(client.models.generate_content.call_count, 3)
        urlopen.assert_called_once()
        self.assertEqual(result, {0: "降级修正"})


if __name__ == "__main__":
    unittest.main()
