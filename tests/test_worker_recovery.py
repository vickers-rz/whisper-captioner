import os
import time
import unittest
from unittest.mock import patch

from whisper_captioner.models import MODES, SubtitleSegment
from whisper_captioner.workers import (
    QueueRunConfig,
    QueueWorker,
    parse_silencedetect_voice_window,
)


class WorkerRecoveryTest(unittest.TestCase):
    def test_vad_trims_silent_edges_with_guards(self):
        output = """
        [silencedetect] silence_start: 0
        [silencedetect] silence_end: 1.2 | silence_duration: 1.2
        [silencedetect] silence_start: 8.0
        """
        window = parse_silencedetect_voice_window(output, 10.0)
        self.assertIsNotNone(window)
        self.assertAlmostEqual(window.start, 1.1)
        self.assertAlmostEqual(window.duration, 7.05)

    def test_vad_all_silence_is_legal_empty_window(self):
        output = "[silencedetect] silence_start: 0"
        self.assertIsNone(parse_silencedetect_voice_window(output, 30.0))

    def test_environment_config_is_clamped(self):
        with patch.dict(
            os.environ,
            {
                "WHISPER_CAPTIONER_QWEN_REPLICAS": "9",
                "WHISPER_CAPTIONER_QWEN_CHUNK_SECONDS": "5",
                "WHISPER_CAPTIONER_QWEN_PARALLEL": "1",
                "WHISPER_CAPTIONER_ADAPTIVE_SPLIT": "true",
            },
            clear=False,
        ):
            config = QueueRunConfig.from_environment()
        self.assertEqual(config.qwen_replicas, 4)
        self.assertEqual(config.qwen_chunk_seconds, 10.0)
        self.assertTrue(config.qwen_parallel_enabled)
        self.assertTrue(config.adaptive_split_enabled)

    def test_parallel_qwen_results_are_sorted_and_progress_is_structured(self):
        mode = next(mode for mode in MODES if mode.key == "qwen3_asr_06b_4bit_mlx")
        worker = QueueWorker(
            [],
            mode,
            QueueRunConfig(
                qwen_replicas=2,
                qwen_chunk_seconds=45,
                qwen_parallel_enabled=True,
            ),
        )
        worker._get_duration = lambda _path: 91.0

        def fake_task(_wav, task, _cancel_event, _holder):
            time.sleep(0.003 if task["start"] == 0 else 0.001)
            return [SubtitleSegment(task["start"], task["start"] + 1, task["label"])]

        worker._run_qwen_chunk_task = fake_task
        progress = []
        worker.chunk_progress.connect(progress.append)
        segments = worker._transcribe_local_qwen3_asr_chunked("/unused.wav")
        self.assertEqual([segment.start for segment in segments], [0, 45, 90])
        self.assertEqual(progress[-1]["done"], 3)
        self.assertEqual(progress[-1]["total"], 3)
        self.assertTrue(progress[-1]["finished"])

    @patch("whisper_captioner.workers._request_json_url")
    def test_release_busy_only_logs(self, request):
        request.return_value = {"status": "busy"}
        mode = next(mode for mode in MODES if mode.key == "nuc_asr")
        worker = QueueWorker([], mode)
        messages = []
        worker.status.connect(messages.append)
        worker._release_nuc_asr()
        self.assertIn("busy", messages[-1])

    @patch("whisper_captioner.workers._request_json_url", side_effect=TimeoutError("offline"))
    def test_release_offline_only_logs(self, _request):
        mode = next(mode for mode in MODES if mode.key == "nuc_asr")
        worker = QueueWorker([], mode)
        messages = []
        worker.status.connect(messages.append)
        worker._release_nuc_asr()
        self.assertIn("unavailable", messages[-1])


if __name__ == "__main__":
    unittest.main()
