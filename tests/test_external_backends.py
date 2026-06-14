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
    fuse_gemini_with_whisper,
    gemini_transcribe_audio,
    run_omnivad_shadow,
)
from whisper_captioner.models import SubtitleSegment, SubtitleWord


class ExternalBackendTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
