from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whisper_captioner.external_backends import (
    _fusion_confidence,
    fuse_gemini_with_whisper,
    fuse_gemini_with_whisper_arbitrated,
    gemini_transcribe_audio,
    refine_timing_with_qwen,
    run_omnivad_shadow,
)
from whisper_captioner.models import SubtitleWord


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
        segments = fuse_gemini_with_whisper(gemini_lines, words)
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
        segments = fuse_gemini_with_whisper(gemini_lines, words)
        self.assertEqual(len(segments), 3)
        for i in range(1, len(segments)):
            self.assertGreaterEqual(segments[i].start, segments[i - 1].end,
                f"segment {i} start {segments[i].start} < previous end {segments[i-1].end}")
            self.assertGreater(segments[i].end, segments[i].start,
                f"segment {i} has non-positive duration")

    def test_fusion_fallback_without_words(self) -> None:
        segments = fuse_gemini_with_whisper(["One.", "Two.", "Three."], [])
        self.assertEqual(len(segments), 3)
        self.assertGreater(segments[-1].end, 0)

    def test_fusion_confidence_full_match(self) -> None:
        conf = _fusion_confidence("hello world", 2, 2, 11, 11)
        self.assertGreater(conf, 0.9)

    def test_fusion_confidence_partial_match(self) -> None:
        conf = _fusion_confidence("hello world extra", 1, 3, 5, 17)
        self.assertLess(conf, 0.5)

    def test_fusion_confidence_no_match(self) -> None:
        conf = _fusion_confidence("completely different", 0, 0, 0, 20)
        self.assertLess(conf, 0.3)

    def test_arbitrated_fusion_same_as_basic_when_no_low_confidence(self) -> None:
        words = [
            SubtitleWord(0.0, 0.5, "Hello"),
            SubtitleWord(0.6, 1.2, "world"),
            SubtitleWord(1.5, 2.0, "this"),
            SubtitleWord(2.1, 2.8, "is"),
            SubtitleWord(3.0, 4.2, "test"),
        ]
        gemini_lines = ["Hello world.", "This is test."]
        # min_confidence=0 means ALL segments are low-conf → arbiter called
        # Use min_confidence=2.0 so NO arbiter is triggered
        result = fuse_gemini_with_whisper_arbitrated(gemini_lines, words, min_confidence=2.0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Hello world.")
        self.assertEqual(result[1].text, "This is test.")

    def test_arbitrated_fusion_preserves_gemini_text(self) -> None:
        # Gemini says "Deberg" (correct), Whisper says "Danberg" (wrong)
        words = [
            SubtitleWord(95.0, 95.3, "Danberg"),
            SubtitleWord(95.4, 96.0, "the"),
            SubtitleWord(96.1, 96.4, "director"),
        ]
        gemini_lines = ["Deberg the director."]
        result = fuse_gemini_with_whisper_arbitrated(gemini_lines, words, min_confidence=2.0)
        self.assertEqual(len(result), 1)
        # Text must always be Gemini's version, never Whisper's
        self.assertEqual(result[0].text, "Deberg the director.")

    def test_refine_timing_no_words_returns_none(self) -> None:
        result = refine_timing_with_qwen("test sentence", [], 0.0, 1.0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
