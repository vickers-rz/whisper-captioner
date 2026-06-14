from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whisper_captioner.models import ASRResult, SpeechRegion, SubtitleSegment, SubtitleWord
from whisper_captioner.subtitle_io import load_asr_result, save_asr_result
from whisper_captioner.subtitle_reliability import (
    LanguagePin,
    audit_asr_result,
    build_cues,
    merge_retry_regions,
    parse_verbose_asr_response,
    replace_segments_in_regions,
)
from whisper_captioner.models import RetryRegion


class SubtitleReliabilityTests(unittest.TestCase):
    def test_parses_top_level_words(self) -> None:
        result = parse_verbose_asr_response(
            {
                "language": "zh",
                "segments": [{"start": 0, "end": 1, "text": "你好"}],
                "words": [{"start": 0, "end": 0.5, "word": "你", "probability": 0.9}],
            }
        )
        self.assertEqual(result.words[0].text, "你")
        self.assertEqual(result.diagnostics["word_timestamp_source"], "top-level words")

    def test_parses_segment_words(self) -> None:
        result = parse_verbose_asr_response(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "hello",
                        "words": [{"start": 0, "end": 0.4, "word": "hello"}],
                    }
                ]
            }
        )
        self.assertEqual(len(result.words), 1)
        self.assertEqual(result.diagnostics["word_timestamp_source"], "segment.words")

    def test_missing_words_falls_back_with_warning(self) -> None:
        result = parse_verbose_asr_response(
            {"segments": [{"start": 0, "end": 1, "text": "fallback"}]}
        )
        cues, warnings = build_cues(result.words, result.segments)
        self.assertEqual(cues[0].text, "fallback")
        self.assertTrue(result.diagnostics["capability_warnings"])
        self.assertTrue(warnings)

    def test_cue_builder_splits_on_punctuation_pause_and_limits(self) -> None:
        words = [
            SubtitleWord(0.0, 0.3, "这是"),
            SubtitleWord(0.3, 0.6, "第一句。"),
            SubtitleWord(1.6, 2.0, "Second"),
            SubtitleWord(2.0, 2.4, "sentence"),
            SubtitleWord(2.4, 2.7, "!"),
        ]
        cues, warnings = build_cues(words, [])
        self.assertFalse(warnings)
        self.assertEqual(len(cues), 2)
        self.assertTrue(all(cue.text and cue.end > cue.start for cue in cues))
        self.assertTrue(all(current.start >= previous.end for previous, current in zip(cues, cues[1:])))
        self.assertTrue(all(cue.end - cue.start <= 5.0 for cue in cues))

    def test_audit_detects_covered_but_low_density_span(self) -> None:
        result = ASRResult(
            language="zh",
            words=[],
            segments=[SubtitleSegment(0, 10, "很少")],
        )
        report = audit_asr_result(result, [SpeechRegion(0, 10)], duration=10)
        self.assertEqual(report.status, "incomplete_speech_coverage")
        self.assertTrue(any("low text density" in item.reason for item in report.suspicious_regions))

    def test_retry_regions_apply_guard_and_merge(self) -> None:
        merged = merge_retry_regions(
            [
                RetryRegion(10, 11, "a"),
                RetryRegion(11.4, 12, "b"),
            ],
            guard=2,
            duration=20,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (8, 14))

    def test_detect_once_then_pin(self) -> None:
        state = LanguagePin()
        state.observe("zh", 0.4, 5)
        self.assertEqual(state.request_language, "auto")
        state.observe("zh", 0.9, 5)
        self.assertEqual(state.request_language, "zh")
        state.observe("en", 0.99, 20)
        self.assertEqual(state.request_language, "zh")

    def test_v2_round_trip_and_legacy_read(self) -> None:
        result = ASRResult(
            language="zh",
            words=[SubtitleWord(0, 0.5, "你", 0.9)],
            segments=[SubtitleSegment(0, 0.5, "你")],
            diagnostics={"source": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asr.json"
            save_asr_result(path, result)
            loaded = load_asr_result(path)
        self.assertEqual(loaded.words, result.words)
        self.assertEqual(loaded.segments, result.segments)

    def test_local_repair_preserves_reliable_segments_and_school_names(self) -> None:
        original = [
            SubtitleSegment(0, 8, "前面的可靠字幕"),
            SubtitleSegment(8, 21, "学校"),
            SubtitleSegment(21, 30, "后面的可靠字幕"),
        ]
        repaired = [
            SubtitleSegment(8.0, 10.0, "航道实业高中"),
            SubtitleSegment(10.0, 12.0, "仁川云峰工业高中"),
            SubtitleSegment(12.0, 14.0, "云山机械工高中"),
        ]
        result = replace_segments_in_regions(
            original,
            repaired,
            [RetryRegion(8, 21, "low text density")],
        )
        text = "".join(segment.text for segment in result)
        self.assertIn("前面的可靠字幕", text)
        self.assertIn("后面的可靠字幕", text)
        self.assertIn("航道实业高中", text)
        self.assertIn("仁川云峰工业高中", text)
        self.assertIn("云山机械工高中", text)

    def test_local_repair_drops_original_cue_overlapped_by_guarded_replacement(self) -> None:
        result = replace_segments_in_regions(
            [
                SubtitleSegment(0, 1.5, "旧开头"),
                SubtitleSegment(1.5, 10, "问题文本"),
            ],
            [
                SubtitleSegment(0, 3, "新开头和补录"),
                SubtitleSegment(3, 5, "恢复文本"),
            ],
            [RetryRegion(2, 10, "low text density")],
        )
        self.assertNotIn("旧开头", "".join(segment.text for segment in result))
        self.assertTrue(all(segment.end > segment.start for segment in result))
        self.assertTrue(
            all(current.start >= previous.end for previous, current in zip(result, result[1:]))
        )

    def test_local_repair_preserves_original_cue_after_region(self) -> None:
        result = replace_segments_in_regions(
            [
                SubtitleSegment(2, 10, "问题文本"),
                SubtitleSegment(10.1, 13, "窗口后的可靠文本"),
            ],
            [SubtitleSegment(2, 11, "补录文本")],
            [RetryRegion(2, 10, "low text density")],
        )
        self.assertIn("窗口后的可靠文本", "".join(segment.text for segment in result))
        self.assertTrue(
            all(current.start >= previous.end for previous, current in zip(result, result[1:]))
        )


if __name__ == "__main__":
    unittest.main()
