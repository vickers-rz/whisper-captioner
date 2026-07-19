import tempfile
import unittest
from pathlib import Path

from scripts.gemini_youtube_url_asr_smoke import (
    PROMPT,
    save_result,
    transcribe_youtube_url,
)


class FakeInteractions:
    def __init__(self):
        self.arguments = None

    def create(self, **arguments):
        self.arguments = arguments
        return {
            "status": "completed",
            "outputs": [{"type": "text", "text": "这是完整语音转写。"}],
        }


class FakeClient:
    def __init__(self):
        self.interactions = FakeInteractions()


class GeminiYoutubeUrlAsrTest(unittest.TestCase):
    def test_audio_only_contract_and_artifacts(self):
        client = FakeClient()
        result = transcribe_youtube_url(
            url="https://www.youtube.com/watch?v=test123",
            api_key="not-a-real-key",
            client=client,
        )

        self.assertEqual(result.text, "这是完整语音转写。")
        self.assertFalse(result.metadata["visual_analysis_requested"])
        self.assertFalse(result.metadata["timestamps_requested"])
        request = client.interactions.arguments
        self.assertEqual(request["input"][0]["type"], "video")
        self.assertEqual(request["input"][0]["resolution"], "low")
        self.assertEqual(request["input"][1]["text"], PROMPT)
        self.assertFalse(request["store"])

        with tempfile.TemporaryDirectory() as directory:
            outputs = save_result(result, Path(directory))
            self.assertEqual(
                Path(outputs["transcript"]).read_text(encoding="utf-8"),
                "这是完整语音转写。\n",
            )
            self.assertTrue(Path(outputs["metadata"]).is_file())


if __name__ == "__main__":
    unittest.main()
