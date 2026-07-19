import tempfile
import unittest
from pathlib import Path

from scripts.asr_entrypoints import (
    ASRResult,
    completed_gemini_job,
    completed_stage,
    multipart_body,
    qwen_segments,
    run_gemini_local_audio,
    run_faster_whisper,
    run_faster_whisper_single,
    trim_words_to_window,
    whisper_chunk_windows,
)
from whisper_captioner.models import SubtitleSegment, SubtitleWord


class AsrEntrypointsTest(unittest.TestCase):
    def test_multipart_contains_model_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"RIFF-test-audio")
            body, boundary = multipart_body([("model", "large-v3")], audio)

        self.assertIn(f"--{boundary}".encode(), body)
        self.assertIn(b'name="model"', body)
        self.assertIn(b"large-v3", body)
        self.assertIn(b"RIFF-test-audio", body)

    def test_qwen_segments_are_read_as_proxy_pseudo_timeline(self):
        segments = qwen_segments(
            {
                "segments": [
                    {"start": 0, "end": 1.5, "text": "第一句。"},
                    {"start": 1.5, "end": 3, "text": "第二句。"},
                ]
            }
        )
        self.assertEqual([segment.text for segment in segments], ["第一句。", "第二句。"])

    def test_completed_stage_requires_every_published_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for key in ("transcript", "srt", "asr_json", "raw_response"):
                path = root / key
                path.write_text("ok", encoding="utf-8")
                paths[key] = str(path)
            self.assertTrue(completed_stage({"status": "completed", **paths}))
            (root / "srt").unlink()
            self.assertFalse(completed_stage({"status": "completed", **paths}))

    def test_completed_gemini_job_requires_same_url_model_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.txt"
            metadata = root / "metadata.json"
            transcript.write_text("text", encoding="utf-8")
            metadata.write_text("{}", encoding="utf-8")
            manifest = root / "asr-manifest.json"
            manifest.write_text(
                '{"status":"completed","source":"https://youtu.be/id",'
                '"model":"gemini-test","outputs":{'
                f'"transcript":"{transcript}","metadata":"{metadata}"}}}}',
                encoding="utf-8",
            )
            self.assertIsNotNone(
                completed_gemini_job(
                    manifest,
                    source="https://youtu.be/id",
                    model="gemini-test",
                )
            )
            self.assertIsNone(
                completed_gemini_job(
                    manifest,
                    source="https://youtu.be/other",
                    model="gemini-test",
                )
            )

    def test_whisper_chunk_windows_add_context_without_duplicate_timeline(self):
        windows = whisper_chunk_windows(125.0, 60.0, 2.0)

        self.assertEqual(len(windows), 3)
        self.assertEqual((windows[0].start, windows[0].duration), (0.0, 62.0))
        self.assertEqual((windows[0].leading_trim, windows[0].trailing_trim), (0.0, 2.0))
        self.assertEqual((windows[1].start, windows[1].duration), (58.0, 64.0))
        self.assertEqual((windows[1].leading_trim, windows[1].trailing_trim), (2.0, 2.0))
        self.assertEqual((windows[2].start, windows[2].duration), (118.0, 7.0))
        self.assertEqual((windows[2].leading_trim, windows[2].trailing_trim), (2.0, 0.0))

    def test_trim_words_to_window_drops_overlap_context(self):
        words = [
            SubtitleWord(0.2, 0.8, "left"),
            SubtitleWord(2.2, 2.8, "keep"),
            SubtitleWord(62.2, 62.8, "right"),
        ]

        trimmed = trim_words_to_window(
            words,
            leading_trim=2.0,
            trailing_trim=2.0,
            chunk_duration=64.0,
        )

        self.assertEqual([word.text for word in trimmed], ["keep"])

    def test_native_batch_size_is_sent_as_upload_field(self):
        captured = {}

        def fake_submit(**kwargs):
            captured.update(kwargs)
            return (
                {
                    "text": "ok",
                    "segments": [
                        {
                            "id": 0,
                            "start": 0,
                            "end": 1,
                            "text": "ok",
                            "words": [{"start": 0, "end": 1, "word": "ok"}],
                        }
                    ],
                },
                {"id": "task-1"},
            )

        import scripts.asr_entrypoints as entrypoints

        original = entrypoints.submit_nuc_job
        try:
            entrypoints.submit_nuc_job = fake_submit
            with tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / "audio.wav"
                audio.write_bytes(b"RIFF-test-audio")
                run_faster_whisper_single(
                    audio,
                    "http://nuc.test",
                    60.0,
                    native_batch_size=8,
                )
        finally:
            entrypoints.submit_nuc_job = original

        self.assertIn(("batch_size", "8"), captured["fields"])
        self.assertIn(("vad_filter", "true"), captured["fields"])

    def test_native_batch_size_disables_chunked_upload(self):
        calls = {"single": 0, "chunked": 0}

        def fake_duration(_wav):
            return 600.0

        def fake_single(*_args, **_kwargs):
            calls["single"] += 1
            return (
                ASRResult(
                    language="zh",
                    words=[SubtitleWord(0, 1, "ok")],
                    segments=[SubtitleSegment(0, 1, "ok")],
                    diagnostics={},
                ),
                {"text": "ok"},
                {"id": "task-1"},
            )

        def fake_chunked(*_args, **_kwargs):
            calls["chunked"] += 1
            raise AssertionError("native batch mode must not use chunked upload")

        import scripts.asr_entrypoints as entrypoints

        originals = (
            entrypoints.probe_audio_duration,
            entrypoints.run_faster_whisper_single,
            entrypoints.run_faster_whisper_chunked,
            entrypoints.repair_faster_whisper_gaps,
        )
        try:
            entrypoints.probe_audio_duration = fake_duration
            entrypoints.run_faster_whisper_single = fake_single
            entrypoints.run_faster_whisper_chunked = fake_chunked
            entrypoints.repair_faster_whisper_gaps = lambda _wav, result, **_kwargs: result
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                audio = root / "audio.wav"
                audio.write_bytes(b"RIFF-test-audio")
                run_faster_whisper(
                    audio,
                    root,
                    "http://nuc.test",
                    60.0,
                    chunk_seconds=30.0,
                    adaptive_parallel=True,
                    native_batch_size=8,
                )
        finally:
            (
                entrypoints.probe_audio_duration,
                entrypoints.run_faster_whisper_single,
                entrypoints.run_faster_whisper_chunked,
                entrypoints.repair_faster_whisper_gaps,
            ) = originals

        self.assertEqual(calls, {"single": 1, "chunked": 0})

    def test_gemini_local_audio_transcodes_to_ogg_and_saves_outputs(self):
        import scripts.asr_entrypoints as entrypoints

        class FakeResult:
            status = "completed"
            text = "转写正文"
            model = "gemini-test"
            elapsed = 1.25
            diagnostics = {"transport": "file-api"}
            warning = ""

        calls = {"commands": [], "audio": None}

        def fake_stream_info(_source):
            return {
                "audio_streams": 1,
                "video_streams": 0,
                "selected_audio_stream": 0,
                "selected_audio_codec": "opus",
                "selected_audio_sample_rate": "48000",
                "selected_audio_channels": 2,
            }

        def fake_run_command(command, _label):
            calls["commands"].append(command)
            Path(command[-1]).write_bytes(b"ogg")

        def fake_gemini(audio_path, *_args, **kwargs):
            calls["audio"] = audio_path
            calls["force_file_api"] = kwargs.get("force_file_api")
            return FakeResult()

        originals = (
            entrypoints.media_stream_info,
            entrypoints.run_command,
            entrypoints.gemini_api_key,
            entrypoints.gemini_transcribe_audio,
        )
        try:
            entrypoints.media_stream_info = fake_stream_info
            entrypoints.run_command = fake_run_command
            entrypoints.gemini_api_key = lambda: "secret"
            entrypoints.gemini_transcribe_audio = fake_gemini
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.webm"
                source.write_bytes(b"webm")
                result = run_gemini_local_audio(
                    source,
                    root,
                    model="gemini-test",
                    timeout=10,
                    upload_timeout=10,
                    processing_timeout=10,
                )
                transcript_exists = Path(result["transcript"]).is_file()
                metadata_exists = Path(result["metadata"]).is_file()
        finally:
            (
                entrypoints.media_stream_info,
                entrypoints.run_command,
                entrypoints.gemini_api_key,
                entrypoints.gemini_transcribe_audio,
            ) = originals

        self.assertEqual(calls["audio"].name, "gemini-audio.ogg")
        self.assertTrue(calls["force_file_api"])
        self.assertIn("-c:a", calls["commands"][0])

    def test_gemini_local_uses_original_ogg_opus_without_transcoding(self):
        import scripts.asr_entrypoints as entrypoints

        class FakeResult:
            status = "completed"
            text = "原始 OGG 转写"
            model = "gemini-test"
            elapsed = 1.0
            diagnostics = {"transport": "file-api"}
            warning = ""

        calls = {"commands": [], "audio": None}

        def fake_stream_info(_source):
            return {
                "audio_streams": 1,
                "video_streams": 0,
                "selected_audio_stream": 0,
                "selected_audio_codec": "opus",
                "selected_audio_sample_rate": "48000",
                "selected_audio_channels": 2,
            }

        def fake_run_command(command, _label):
            calls["commands"].append(command)

        def fake_gemini(audio_path, *_args, **kwargs):
            calls["audio"] = audio_path
            calls["force_file_api"] = kwargs.get("force_file_api")
            return FakeResult()

        originals = (
            entrypoints.media_stream_info,
            entrypoints.run_command,
            entrypoints.gemini_api_key,
            entrypoints.gemini_transcribe_audio,
        )
        try:
            entrypoints.media_stream_info = fake_stream_info
            entrypoints.run_command = fake_run_command
            entrypoints.gemini_api_key = lambda: "secret"
            entrypoints.gemini_transcribe_audio = fake_gemini
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.ogg"
                source.write_bytes(b"original ogg")
                result = run_gemini_local_audio(
                    source,
                    root,
                    model="gemini-test",
                    timeout=10,
                    upload_timeout=10,
                    processing_timeout=10,
                )
                transcript_exists = Path(result["transcript"]).is_file()
                metadata_exists = Path(result["metadata"]).is_file()
        finally:
            (
                entrypoints.media_stream_info,
                entrypoints.run_command,
                entrypoints.gemini_api_key,
                entrypoints.gemini_transcribe_audio,
            ) = originals

        self.assertEqual(calls["audio"], source)
        self.assertEqual(calls["commands"], [])
        self.assertTrue(calls["force_file_api"])
        self.assertTrue(transcript_exists)
        self.assertTrue(metadata_exists)
        self.assertEqual(result["characters"], len("原始 OGG 转写"))
        self.assertTrue(transcript_exists)
        self.assertTrue(metadata_exists)


if __name__ == "__main__":
    unittest.main()
