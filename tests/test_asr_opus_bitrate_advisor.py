from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import asr_opus_bitrate_advisor as advisor


def test_normalize_transcript_ignores_spacing_and_punctuation() -> None:
    assert advisor.normalize_transcript("你好， 世界！\n") == "你好世界"


def test_levenshtein_counts_character_edits() -> None:
    assert advisor.levenshtein("战略", "战") == 1
    assert advisor.levenshtein("abc", "abc") == 0


def test_consensus_is_anchored_to_highest_three_bitrates() -> None:
    texts = {
        48: "高码率共识",
        32: "高码率共识",
        24: "高码率共识",
        20: "低码率另一结果",
        16: "低码率另一结果",
        12: "低码率另一结果",
        8: "低码率另一结果",
    }
    assert advisor.choose_consensus(texts) == "高码率共识"


def test_consensus_tie_breaks_toward_highest_bitrate() -> None:
    texts = {48: "甲", 32: "乙", 24: "丙", 20: "乙"}
    assert advisor.choose_consensus(texts) == "甲"


def test_recommendation_adds_one_stable_tier_margin() -> None:
    floor, recommended = advisor.recommend_bitrate([48, 32, 24, 20], margin_steps=1)
    assert floor == 20
    assert recommended == 24


def test_recommendation_can_use_measured_floor_directly() -> None:
    floor, recommended = advisor.recommend_bitrate([48, 32, 24, 20], margin_steps=0)
    assert floor == 20
    assert recommended == 20


def test_sample_starts_are_evenly_distributed() -> None:
    starts = advisor.sample_starts(duration=2467.1, count=6, seconds=45)
    assert len(starts) == 6
    assert starts == sorted(starts)
    assert starts[0] >= 0
    assert starts[-1] + 45 <= 2467.1 + 1e-6
