from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from whisper_captioner.external_backends import (
    _merge_adjacent_duplicate_segments,
    _fusion_confidence,
    baseline_token_coverage,
    fuse_gemini_with_whisper,
    gemini_transcribe_audio,
    parse_parakeet_report,
    parse_whisperkit_report,
    run_omnivad_shadow,
)
from whisper_captioner.models import SubtitleSegment, SubtitleWord


class ExternalBackendTests(unittest.TestCase):
    def test_parakeet_subword_tokens_become_word_timestamps(self) -> None:
        result = parse_parakeet_report(
            {
                "sentences": [
                    {
                        "text": " going on a holiday.",
                        "start": 4.32,
                        "end": 5.2,
                        "tokens": [
                            {"text": " going", "start": 4.32, "end": 4.48, "confidence": 1},
                            {"text": " on", "start": 4.48, "end": 4.64, "confidence": 1},
                            {"text": " a", "start": 4.64, "end": 4.72, "confidence": 0.95},
                            {"text": " h", "start": 4.72, "end": 4.80, "confidence": 1},
                            {"text": "ol", "start": 4.80, "end": 4.88, "confidence": 1},
                            {"text": "id", "start": 4.88, "end": 5.04, "confidence": 1},
                            {"text": "ay", "start": 5.04, "end": 5.12, "confidence": 1},
                            {"text": ".", "start": 5.12, "end": 5.20, "confidence": 1},
                        ],
                    }
                ]
            }
        )
        self.assertEqual(
            [word.text for word in result.words],
            ["going", "on", "a", "holiday."],
        )
        self.assertEqual((result.words[-1].start, result.words[-1].end), (4.72, 5.2))

    def test_whisperkit_report_preserves_word_timestamps(self) -> None:
        result = parse_whisperkit_report(
            {
                "language": "en",
                "segments": [
                    {
                        "start": 3.56,
                        "end": 5.46,
                        "text": "<|3.56|> going on a holiday<|5.46|>",
                        "words": [
                            {
                                "start": 4.20,
                                "end": 4.46,
                                "word": " going",
                                "probability": 1.0,
                            },
                            {
                                "start": 4.46,
                                "end": 4.62,
                                "word": " on",
                                "probability": 1.0,
                            },
                            {
                                "start": 4.62,
                                "end": 4.64,
                                "word": " a",
                                "probability": 0.75,
                            },
                            {
                                "start": 4.64,
                                "end": 5.0,
                                "word": " holiday",
                                "probability": 1.0,
                            },
                        ],
                    }
                ],
            }
        )
        self.assertEqual(result.language, "en")
        self.assertEqual([word.text.strip() for word in result.words], ["going", "on", "a", "holiday"])
        self.assertEqual(result.segments[0].text, "going on a holiday")
        self.assertEqual(result.diagnostics["word_timestamp_source"], "whisperkit-report")

    def test_omnivad_missing_falls_back_without_exception(self) -> None:
        with patch.dict(
            "os.environ",
            {"WHISPER_CAPTIONER_OMNIVAD_COMMAND": "/missing/omnivad {audio} -o {output}"},
        ):
            result = run_omnivad_shadow(Path("/tmp/audio.wav"), Path("/tmp/output"))
        self.assertEqual(result.status, "unavailable")
        self.assertTrue(result.warning)

    def test_omnivad_json_tiers_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake-omnivad"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "omnivad-shadow.json").write_text(
                json.dumps(
                    {
                        "tiers": {
                            "VAD": [
                                {"start": 1.0, "end": 2.5, "label": "speech"}
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "WHISPER_CAPTIONER_OMNIVAD_COMMAND": (
                        f"{executable} {{audio}} -o {{output}}"
                    )
                },
            ):
                result = run_omnivad_shadow(root / "audio.wav", output_dir)
        self.assertEqual(result.status, "completed")
        self.assertEqual((result.regions[0].start, result.regions[0].end), (1.0, 2.5))

    def test_gemini_transcribe_missing_key_is_skipped(self) -> None:
        result = gemini_transcribe_audio(Path("/tmp/audio.wav"), "")
        self.assertEqual(result.status, "skipped")

    def test_fusion_aligns_gemini_text_with_whisper_words(self) -> None:
        words = [
            SubtitleWord(0.0, 0.5, "Hello"),
            SubtitleWord(0.6, 1.2, "world"),
            SubtitleWord(1.5, 2.0, "this"),
            SubtitleWord(2.1, 2.8, "is"),
            SubtitleWord(3.0, 3.5, "a"),
            SubtitleWord(3.6, 4.2, "test"),
        ]
        gemini_lines = ["Hello world.", "This is a test."]
        fusion = fuse_gemini_with_whisper(gemini_lines, words)
        segments = fusion.segments
        self.assertEqual(fusion.status, "completed")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hello world.")
        self.assertEqual(segments[1].text, "This is a test.")
        # Timestamps should align with word ranges
        self.assertAlmostEqual(segments[0].start, 0.0, delta=0.1)
        self.assertAlmostEqual(segments[0].end, 1.2, delta=0.2)
        self.assertAlmostEqual(segments[1].start, 1.5, delta=0.2)
        self.assertAlmostEqual(segments[1].end, 4.2, delta=0.2)

    def test_fusion_normalizes_timeline_monotonic(self) -> None:
        words = [
            SubtitleWord(0.0, 0.3, "A"),
            SubtitleWord(2.0, 2.3, "B"),
            SubtitleWord(5.0, 5.3, "C"),
        ]
        gemini_lines = ["A.", "B.", "C."]
        segments = fuse_gemini_with_whisper(gemini_lines, words).segments
        self.assertEqual(len(segments), 3)
        for i in range(1, len(segments)):
            self.assertGreaterEqual(segments[i].start, segments[i - 1].end,
                f"segment {i} start {segments[i].start} < previous end {segments[i-1].end}")
            self.assertGreater(segments[i].end, segments[i].start,
                f"segment {i} has non-positive duration")

    def test_fusion_blocks_without_words(self) -> None:
        fusion = fuse_gemini_with_whisper(["One.", "Two.", "Three."], [])
        self.assertEqual(fusion.status, "blocked")
        self.assertEqual(fusion.segments, [])

    def test_fusion_confidence_full_match(self) -> None:
        conf = _fusion_confidence("hello world", 2, 2, 11, 11)
        self.assertGreater(conf, 0.9)

    def test_fusion_confidence_partial_match(self) -> None:
        conf = _fusion_confidence("hello world extra", 1, 3, 5, 17)
        self.assertLess(conf, 0.5)

    def test_fusion_confidence_no_match(self) -> None:
        conf = _fusion_confidence("completely different", 0, 0, 0, 20)
        self.assertLess(conf, 0.3)

    def test_high_confidence_gemini_text_corrects_whisper_baseline(self) -> None:
        words = [
            SubtitleWord(0.0, 0.5, "Hello"),
            SubtitleWord(0.6, 1.2, "world"),
            SubtitleWord(1.5, 2.0, "this"),
            SubtitleWord(2.1, 2.8, "is"),
            SubtitleWord(3.0, 4.2, "test"),
        ]
        gemini_lines = ["Hello world.", "This is test."]
        whisper = [
            SubtitleSegment(0.0, 1.2, "Hello word."),
            SubtitleSegment(1.5, 4.2, "This is test."),
        ]
        result = fuse_gemini_with_whisper(
            gemini_lines, words, whisper_segments=whisper, min_confidence=0.7
        )
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].text, "Hello world.")
        self.assertEqual(result.segments[1].text, "This is test.")

    def test_high_confidence_proper_noun_correction_preserves_gemini_text(self) -> None:
        # Gemini says "Deberg" (correct), Whisper says "Danberg" (wrong)
        words = [
            SubtitleWord(95.0, 95.3, "Danberg"),
            SubtitleWord(95.4, 96.0, "the"),
            SubtitleWord(96.1, 96.4, "director"),
        ]
        gemini_lines = ["Deberg the director."]
        result = fuse_gemini_with_whisper(
            gemini_lines,
            words,
            whisper_segments=[SubtitleSegment(95.0, 96.4, "Danberg the director.")],
            min_confidence=0.7,
        )
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "Deberg the director.")

    def test_duplicate_lines_map_to_distinct_time_ranges(self) -> None:
        words = [
            SubtitleWord(0.0, 0.5, "hello"),
            SubtitleWord(0.6, 1.0, "world"),
            SubtitleWord(4.0, 4.5, "hello"),
            SubtitleWord(4.6, 5.0, "world"),
        ]
        result = fuse_gemini_with_whisper(["Hello world.", "Hello world."], words)
        self.assertEqual(result.status, "completed")
        self.assertLess(result.segments[0].end, result.segments[1].start)
        self.assertAlmostEqual(result.segments[1].start, 4.0, delta=0.2)

    def test_adjacent_duplicate_fallback_segments_are_merged(self) -> None:
        long_text = (
            "He claims his inspiration for the film is his own experiences "
            "growing up in 1950s Liverpool."
        )
        segment_type = __import__(
            "whisper_captioner.models", fromlist=["SubtitleSegment"]
        ).SubtitleSegment
        segments = [
            segment_type(109.42, 115.72, long_text),
            segment_type(118.74, 119.26, long_text),
            segment_type(119.26, 119.34, long_text),
            segment_type(119.34, 119.66, long_text),
            segment_type(119.66, 119.72, long_text),
            segment_type(119.72, 119.86, long_text),
            segment_type(119.86, 119.98, long_text),
        ]
        merged, count = _merge_adjacent_duplicate_segments(segments)
        self.assertEqual(len(merged), 1)
        self.assertEqual(count, 6)
        self.assertEqual((merged[0].start, merged[0].end), (109.42, 119.98))

    def test_unmatched_gemini_text_preserves_whisper(self) -> None:
        words = [SubtitleWord(1.0, 2.0, "original")]
        whisper = [__import__("whisper_captioner.models", fromlist=["SubtitleSegment"]).SubtitleSegment(
            1.0, 2.0, "original"
        )]
        result = fuse_gemini_with_whisper(
            ["completely unrelated hallucination"],
            words,
            whisper_segments=whisper,
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.segments[0].text, "original")

    def test_low_confidence_text_keeps_whisper_baseline(self) -> None:
        words = [
            SubtitleWord(0.0, 0.4, "hello"),
            SubtitleWord(0.5, 1.0, "world"),
            SubtitleWord(1.1, 1.5, "other"),
        ]
        whisper = [
            SubtitleSegment(0.0, 1.0, "hello world"),
            SubtitleSegment(1.1, 1.5, "other"),
        ]
        result = fuse_gemini_with_whisper(
            ["hello world", "other extra"],
            words,
            whisper_segments=whisper,
            min_confidence=0.6,
        )
        candidate = result.diagnostics["line_candidates"][1]
        self.assertFalse(candidate["accepted"])
        self.assertEqual(result.segments[-1].text, "other")

    def test_missing_gemini_section_preserves_whisper_section(self) -> None:
        words = [
            SubtitleWord(0.0, 0.5, "first"),
            SubtitleWord(0.6, 1.0, "line"),
            SubtitleWord(2.0, 2.5, "missing"),
            SubtitleWord(2.6, 3.0, "section"),
            SubtitleWord(4.0, 4.5, "last"),
            SubtitleWord(4.6, 5.0, "line"),
        ]
        whisper = [
            SubtitleSegment(0.0, 1.0, "first line"),
            SubtitleSegment(2.0, 3.0, "missing section"),
            SubtitleSegment(4.0, 5.0, "last line"),
        ]
        result = fuse_gemini_with_whisper(
            ["First line.", "Last line."],
            words,
            whisper_segments=whisper,
        )
        self.assertEqual(
            [segment.text for segment in result.segments],
            ["First line.", "missing section", "Last line."],
        )

    def test_candidate_that_drops_baseline_tokens_is_blocked(self) -> None:
        baseline = [
            SubtitleSegment(0.0, 1.0, "Now, what would you like to do about the courses?"),
            SubtitleSegment(2.0, 3.0, "Questions eleven to fifteen."),
        ]
        candidate = [SubtitleSegment(2.0, 3.0, "Questions eleven to fifteen.")]
        coverage = baseline_token_coverage(baseline, candidate)
        self.assertLess(coverage["coverage"], 0.985)
        self.assertTrue(coverage["missing_runs"])

    def test_fusion_blocks_when_high_confidence_rewrite_drops_baseline_section(self) -> None:
        words = [
            SubtitleWord(0.0, 0.2, "now"),
            SubtitleWord(0.2, 0.4, "what"),
            SubtitleWord(0.4, 0.6, "would"),
            SubtitleWord(0.6, 0.8, "you"),
            SubtitleWord(0.8, 1.0, "like"),
            SubtitleWord(1.0, 1.2, "to"),
            SubtitleWord(1.2, 1.4, "do"),
            SubtitleWord(1.4, 1.6, "about"),
            SubtitleWord(2.0, 2.2, "questions"),
            SubtitleWord(2.2, 2.4, "eleven"),
            SubtitleWord(2.4, 2.6, "to"),
            SubtitleWord(2.6, 2.8, "fifteen"),
        ]
        whisper = [
            SubtitleSegment(0.0, 1.6, "Now, what would you like to do about"),
            SubtitleSegment(2.0, 2.8, "Questions eleven to fifteen."),
        ]
        result = fuse_gemini_with_whisper(
            ["Questions eleven to fifteen."],
            words,
            whisper_segments=whisper,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(
            [segment.text for segment in result.segments],
            ["Now, what would you like to do about", "Questions eleven to fifteen."],
        )
        self.assertFalse(
            result.diagnostics["baseline_token_coverage"]["missing_runs"]
        )

    def test_real_ielts_fixture_preserves_known_whisper_only_phrases(self) -> None:
        root = Path(
            "/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/1729839784568919"
        )
        asr_path = root / "1729839784568919-20260614-194754-Whisper原始-asr.json"
        whisper_path = root / "1729839784568919-20260614-194754-Whisper原始.srt"
        gemini_path = root / "1729839784568919-20260614-194754-Gemini原文.txt"
        if not (asr_path.exists() and whisper_path.exists() and gemini_path.exists()):
            self.skipTest("local IELTS regression fixture is unavailable")
        from whisper_captioner.subtitle_io import parse_srt

        data = json.loads(asr_path.read_text(encoding="utf-8"))
        words = [
            SubtitleWord(
                float(item["start"]),
                float(item["end"]),
                str(item["text"]),
                item.get("probability"),
            )
            for item in data["words"]
        ]
        result = fuse_gemini_with_whisper(
            [
                line.strip()
                for line in gemini_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            words,
            whisper_segments=parse_srt(whisper_path),
        )
        text = " ".join(segment.text for segment in result.segments)
        self.assertIn("going on a holiday", text)
        self.assertIn("Questions six to ten", text)
        self.assertIn("Question 4", text)
        self.assertEqual(result.status, "blocked")

    def test_file_api_upload_poll_generate_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "long.wav"
            audio.write_bytes(b"x")
            uploaded = SimpleNamespace(name="files/test", state="PROCESSING")
            active = SimpleNamespace(name="files/test", state="ACTIVE")
            response = SimpleNamespace(
                text="first line\nsecond line",
                candidates=[SimpleNamespace(finish_reason="STOP")],
                usage_metadata=SimpleNamespace(candidates_token_count=2),
            )
            generation_client = MagicMock()
            file_client = MagicMock()
            file_client.files.upload.return_value = uploaded
            file_client.files.get.return_value = active
            generation_client.models.generate_content.return_value = response
            fake_types = SimpleNamespace(
                HttpOptions=lambda **kwargs: kwargs,
                UploadFileConfig=lambda **kwargs: kwargs,
                GenerateContentConfig=lambda **kwargs: kwargs,
            )
            fake_google = types.ModuleType("google")
            fake_google.genai = SimpleNamespace(
                Client=MagicMock(side_effect=[generation_client, file_client]),
                types=fake_types,
            )
            with (
                patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_google.genai}),
                patch("whisper_captioner.external_backends.GEMINI_INLINE_MAX_BYTES", 0),
                patch("whisper_captioner.external_backends.time.sleep"),
            ):
                result = gemini_transcribe_audio(audio, "secret")
        self.assertEqual(result.status, "completed")
        file_client.files.upload.assert_called_once()
        file_client.files.get.assert_called_once_with(name="files/test")
        file_client.files.delete.assert_called_once_with(name="files/test")
        self.assertEqual(result.diagnostics["cleanup"], "deleted")

    def test_token_limit_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "short.wav"
            audio.write_bytes(b"x")
            response = SimpleNamespace(
                text="partial transcript",
                candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
                usage_metadata=SimpleNamespace(candidates_token_count=8192),
            )
            client = MagicMock()
            client.models.generate_content.return_value = response
            fake_types = SimpleNamespace(
                HttpOptions=lambda **kwargs: kwargs,
                GenerateContentConfig=lambda **kwargs: kwargs,
            )
            fake_google = types.ModuleType("google")
            fake_google.genai = SimpleNamespace(
                Client=MagicMock(return_value=client),
                types=fake_types,
            )
            with patch.dict(
                sys.modules,
                {"google": fake_google, "google.genai": fake_google.genai},
            ):
                result = gemini_transcribe_audio(audio, "secret")
        self.assertEqual(result.status, "failed")
        self.assertIn("truncated", result.warning)

    def test_gemini_generation_timeout_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "short.wav"
            audio.write_bytes(b"x")

            def hang_forever(*_args, **_kwargs):
                import time

                time.sleep(5)

            client = MagicMock()
            client.models.generate_content.side_effect = hang_forever
            fake_types = SimpleNamespace(
                HttpOptions=lambda **kwargs: kwargs,
                GenerateContentConfig=lambda **kwargs: kwargs,
            )
            fake_google = types.ModuleType("google")
            fake_google.genai = SimpleNamespace(
                Client=MagicMock(return_value=client),
                types=fake_types,
            )
            progress: list[str] = []
            with patch.dict(
                sys.modules,
                {"google": fake_google, "google.genai": fake_google.genai},
            ):
                result = gemini_transcribe_audio(
                    audio,
                    "secret",
                    timeout=0.01,
                    progress_callback=progress.append,
                )
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.warning)
        self.assertIn("Gemini generation started", progress)

if __name__ == "__main__":
    unittest.main()
