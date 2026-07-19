#!/usr/bin/env python3
"""Use an aligned Gemini transcript to correct and complete burned-subtitle OCR.

The output keeps the observed OCR string alongside the template-derived string so
downstream consumers can distinguish visual evidence from transcript completion.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any


def normalized_chars(text: str) -> tuple[list[str], list[int]]:
    chars: list[str] = []
    original_indices: list[int] = []
    for original_index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).lower():
            if normalized.isalnum() or "\u3400" <= normalized <= "\u9fff":
                chars.append(normalized)
                original_indices.append(original_index)
    return chars, original_indices


def matched_coverage(candidate: str, reference: str) -> float:
    candidate_chars, _ = normalized_chars(candidate)
    reference_chars, _ = normalized_chars(reference)
    if not candidate_chars or not reference_chars:
        return 0.0
    matcher = difflib.SequenceMatcher(
        None, candidate_chars, reference_chars, autojunk=False
    )
    return sum(block.size for block in matcher.get_matching_blocks()) / len(candidate_chars)


def build_nuc_character_timeline(asr: dict[str, Any]) -> tuple[list[str], list[float]]:
    chars: list[str] = []
    times: list[float] = []
    for word in asr.get("words", []):
        word_chars, _ = normalized_chars(str(word.get("text", "")))
        if not word_chars:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        for index, char in enumerate(word_chars):
            chars.append(char)
            times.append(start + (end - start) * (index + 0.5) / len(word_chars))
    return chars, times


def interpolate_template_times(
    template_chars: list[str], nuc_chars: list[str], nuc_times: list[float]
) -> tuple[list[float], dict[str, Any]]:
    matcher = difflib.SequenceMatcher(
        None, nuc_chars, template_chars, autojunk=False
    )
    assigned: dict[int, float] = {}
    matched = 0
    for nuc_start, template_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            assigned[template_start + offset] = nuc_times[nuc_start + offset]
            matched += 1
    if not assigned:
        raise RuntimeError("Gemini template and NUC transcript have no character matches")

    anchors = sorted(assigned.items())
    anchor_positions = [item[0] for item in anchors]
    output: list[float] = []
    for index in range(len(template_chars)):
        if index in assigned:
            output.append(assigned[index])
            continue
        insertion = bisect.bisect_left(anchor_positions, index)
        if insertion == 0:
            output.append(anchors[0][1])
        elif insertion == len(anchors):
            output.append(anchors[-1][1])
        else:
            left_index, left_time = anchors[insertion - 1]
            right_index, right_time = anchors[insertion]
            ratio = (index - left_index) / max(1, right_index - left_index)
            output.append(left_time + (right_time - left_time) * ratio)
    return output, {
        "nuc_characters": len(nuc_chars),
        "template_characters": len(template_chars),
        "matched_characters": matched,
        "template_alignment_coverage": round(matched / len(template_chars), 6),
    }


def template_span_for_time(
    original_text: str,
    original_indices: list[int],
    times: list[float],
    start: float,
    end: float,
) -> tuple[str, int, int]:
    left = bisect.bisect_left(times, start)
    right = bisect.bisect_right(times, end)
    if left >= right:
        return "", -1, -1
    original_start = original_indices[left]
    original_end = original_indices[right - 1] + 1
    return original_text[original_start:original_end].strip(), left, right


def template_span_for_indices(
    original_text: str,
    original_indices: list[int],
    start: int,
    end: int,
) -> str:
    if start < 0 or end <= start:
        return ""
    return original_text[original_indices[start]:original_indices[end - 1] + 1].strip()


def has_unbalanced_wrappers(text: str) -> bool:
    pairs = (("《", "》"), ("〈", "〉"), ("（", "）"), ("(", ")"), ("[", "]"), ("【", "】"))
    return any(text.count(left) != text.count(right) for left, right in pairs)


def exact_template_patch(
    raw_ocr: str,
    template_chars: list[str],
    template_times: list[float],
    template_original_text: str,
    template_original_indices: list[int],
    start: float,
    end: float,
) -> tuple[str, str, float, int, int]:
    """Return an exact normalized Gemini substring near the visual cue.

    OCR defines the subtitle boundary. Gemini is allowed to restore punctuation
    inside that same boundary, but never to extend the phrase with extra words.
    """
    ocr_chars, _ = normalized_chars(raw_ocr)
    if not ocr_chars:
        return raw_ocr, "ocr_empty", 0.0, -1, -1

    estimated_left = bisect.bisect_left(template_times, max(0.0, start - 1.5))
    estimated_right = bisect.bisect_right(template_times, end + 1.5)
    left = max(0, estimated_left - 40)
    right = min(len(template_chars), estimated_right + 40)
    query = "".join(ocr_chars)
    local = "".join(template_chars[left:right])
    candidates: list[int] = []
    offset = local.find(query)
    while offset >= 0:
        candidates.append(left + offset)
        offset = local.find(query, offset + 1)
    if not candidates:
        return raw_ocr, "ocr_visual_no_exact_template_match", 0.0, -1, -1

    cue_center = (start + end) / 2
    position = min(
        candidates,
        key=lambda item: abs(template_times[min(len(template_times) - 1, item + len(ocr_chars) // 2)] - cue_center),
    )
    patched = template_span_for_indices(
        template_original_text,
        template_original_indices,
        position,
        position + len(ocr_chars),
    )
    if has_unbalanced_wrappers(patched):
        return raw_ocr, "ocr_visual_template_unbalanced_punctuation", 1.0, position, position + len(ocr_chars)
    if patched == raw_ocr:
        return raw_ocr, "ocr_visual_exact_template_match", 1.0, position, position + len(ocr_chars)
    return patched, "gemini_template_exact_normalized_match", 1.0, position, position + len(ocr_chars)


def apply_ocr_timing_anchors(
    template_times: list[float], cues: list[dict[str, Any]]
) -> tuple[list[float], dict[str, Any]]:
    """Warp the NUC timeline toward visually observed hard-subtitle boundaries.

    Only exact normalized OCR/Gemini matches contribute anchors.  Corrections are
    interpolated as offsets, retaining NUC timing shape in visual gaps.
    """
    anchors: list[tuple[int, float]] = []
    last_end = -1
    accepted = 0
    rejected = 0
    for cue in cues:
        start = int(cue.get("matched_template_character_start", -1))
        end = int(cue.get("matched_template_character_end", -1))
        if start < 0 or end <= start or start < last_end:
            if start >= 0:
                rejected += 1
            continue
        last_end = end
        anchors.extend(((start, float(cue["start"])), (end - 1, float(cue["end"]))))
        accepted += 1
    if len(anchors) < 2:
        return template_times, {"accepted_cues": accepted, "rejected_cues": rejected, "anchor_points": len(anchors)}

    grouped: dict[int, list[float]] = {}
    for index, value in anchors:
        grouped.setdefault(index, []).append(value)
    points = sorted((index, statistics.median(values)) for index, values in grouped.items())
    positions = [item[0] for item in points]
    offsets = [value - template_times[index] for index, value in points]
    warped: list[float] = []
    for index, original in enumerate(template_times):
        insertion = bisect.bisect_left(positions, index)
        if insertion == 0:
            offset = offsets[0]
        elif insertion == len(points):
            offset = offsets[-1]
        else:
            left_position, right_position = positions[insertion - 1], positions[insertion]
            ratio = (index - left_position) / max(1, right_position - left_position)
            offset = offsets[insertion - 1] + (offsets[insertion] - offsets[insertion - 1]) * ratio
        warped.append(max(warped[-1] if warped else 0.0, original + offset))
    return warped, {
        "accepted_cues": accepted,
        "rejected_cues": rejected,
        "anchor_points": len(points),
        "mean_absolute_offset_seconds": round(statistics.mean(abs(value) for value in offsets), 6),
        "max_absolute_offset_seconds": round(max(abs(value) for value in offsets), 6),
    }


def srt_timestamp(value: float) -> str:
    milliseconds = max(0, round(value * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(path: Path, cues: list[dict[str, Any]]) -> None:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n{srt_timestamp(cue['start'])} --> {srt_timestamp(cue['end'])}\n{cue['text']}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def build_full_template_timeline(
    original_text: str,
    original_indices: list[int],
    times: list[float],
    duration: float,
) -> list[dict[str, Any]]:
    """Produce a readable Gemini transcript timeline from the NUC word anchors.

    These cues are content-complete relative to Gemini but are deliberately
    separate from the visual OCR evidence because their boundaries are inferred.
    """
    last = bisect.bisect_right(times, duration)
    cues: list[dict[str, Any]] = []
    start = 0
    while start < last:
        end = min(last, start + 24)
        while end < last and times[end - 1] - times[start] < 2.8:
            end += 1
            if end - start >= 32:
                break
        segment = template_span_for_indices(original_text, original_indices, start, end)
        # Prefer a natural sentence boundary once a cue is long enough.
        if end < last and end - start >= 12:
            for index in range(end - 1, start + 9, -1):
                original = original_text[original_indices[index]]
                if original in "，。！？；：":
                    end = index + 1
                    segment = template_span_for_indices(original_text, original_indices, start, end)
                    break
        if segment:
            cue_start = max(0.0, times[start] - 0.06)
            cue_end = min(duration, times[end - 1] + 0.06)
            if cue_end <= cue_start:
                cue_end = cue_start + 0.1
            cues.append(
                {
                    "start": cue_start,
                    "end": cue_end,
                    "text": segment,
                    "source": "gemini_template_nuc_timeline_inferred",
                    "template_character_start": start,
                    "template_character_end": end,
                }
            )
        start = end
    return cues


def sequence_stats(left: str, right: str) -> dict[str, Any]:
    a, _ = normalized_chars(left)
    b, _ = normalized_chars(right)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return {
        "left_characters": len(a),
        "right_characters": len(b),
        "matched_characters": matched,
        "left_precision": round(matched / max(1, len(a)), 6),
        "right_coverage": round(matched / max(1, len(b)), 6),
        "sequence_similarity": round(matcher.ratio(), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--minimum-visual-template-coverage", type=float, default=0.60)
    args = parser.parse_args()

    ocr_data = json.loads(args.ocr.read_text(encoding="utf-8"))
    asr_data = json.loads(args.asr.read_text(encoding="utf-8"))
    gemini_text = args.gemini.read_text(encoding="utf-8")
    template_chars, template_original_indices = normalized_chars(gemini_text)
    nuc_chars, nuc_times = build_nuc_character_timeline(asr_data)
    template_times, alignment = interpolate_template_times(
        template_chars, nuc_chars, nuc_times
    )

    template_end = bisect.bisect_right(template_times, args.duration)
    template_text_first_window = gemini_text[
        : template_original_indices[max(0, template_end - 1)] + 1
    ]
    enhanced: list[dict[str, Any]] = []
    for cue in ocr_data.get("cues", []):
        start = float(cue["start"])
        end = float(cue["end"])
        raw_ocr = str(cue["text"]).strip()
        template_text, template_start, template_end_index = template_span_for_time(
            gemini_text,
            template_original_indices,
            template_times,
            max(0.0, start - 0.05),
            min(args.duration, end + 0.05),
        )
        time_window_template_text = template_span_for_indices(
            gemini_text,
            template_original_indices,
            template_start,
            template_end_index,
        )
        text, source, visual_template_coverage, matched_start, matched_end = exact_template_patch(
            raw_ocr,
            template_chars,
            template_times,
            gemini_text,
            template_original_indices,
            start,
            end,
        )
        enhanced.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "source": source,
                "ocr_text": raw_ocr,
                "gemini_time_window_text": time_window_template_text,
                "gemini_template_text": text if source.startswith("gemini_") else "",
                "visual_template_coverage": round(visual_template_coverage, 6),
                "ocr_confidence": cue.get("confidence", 0.0),
                "template_character_start": template_start,
                "template_character_end": template_end_index,
                "matched_template_character_start": matched_start,
                "matched_template_character_end": matched_end,
                "frame": cue.get("frame", ""),
            }
        )

    patched = [cue for cue in enhanced if cue["source"] == "gemini_template_exact_normalized_match"]
    visually_verified = [
        cue for cue in enhanced if cue["source"] != "ocr_visual_no_exact_template_match"
    ]
    raw_ocr_text = "".join(str(cue.get("ocr_text", "")) for cue in enhanced)
    enhanced_text = "".join(str(cue["text"]) for cue in enhanced)
    visually_verified_text = "".join(str(cue["text"]) for cue in visually_verified)
    full_template_timeline = build_full_template_timeline(
        gemini_text,
        template_original_indices,
        template_times,
        args.duration,
    )
    ocr_anchored_times, anchor_report = apply_ocr_timing_anchors(template_times, enhanced)
    ocr_anchored_full_template_timeline = build_full_template_timeline(
        gemini_text,
        template_original_indices,
        ocr_anchored_times,
        args.duration,
    )
    report = {
        "duration_seconds": args.duration,
        "gemini_nuc_alignment": alignment,
        "ocr_cues": len(enhanced),
        "gemini_punctuation_or_spelling_patched_cues": len(patched),
        "template_visually_verified_cues": len(visually_verified),
        "ocr_retained_cues": len(enhanced) - len(patched),
        "mean_visual_template_coverage": round(
            statistics.mean(float(cue["visual_template_coverage"]) for cue in enhanced), 6
        ) if enhanced else 0.0,
        "raw_ocr_vs_gemini_template": sequence_stats(raw_ocr_text, template_text_first_window),
        "enhanced_vs_gemini_template": sequence_stats(enhanced_text, template_text_first_window),
        "visually_verified_vs_gemini_template": sequence_stats(visually_verified_text, template_text_first_window),
        "full_gemini_template_timeline_vs_gemini_template": sequence_stats(
            "".join(cue["text"] for cue in full_template_timeline), template_text_first_window
        ),
        "ocr_anchored_timing": anchor_report,
        "ocr_anchored_full_gemini_template_timeline_vs_gemini_template": sequence_stats(
            "".join(cue["text"] for cue in ocr_anchored_full_template_timeline), template_text_first_window
        ),
        "template_text_first_window": template_text_first_window,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "apple-vision+gemini-template+NUC-timeline",
        "gemini_source": str(args.gemini),
        "asr_timeline_source": str(args.asr),
        "minimum_visual_template_coverage": args.minimum_visual_template_coverage,
        "cues": enhanced,
    }
    (args.output_dir / "burned-subtitle-gemini-template.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_srt(args.output_dir / "burned-subtitle-gemini-template.srt", enhanced)
    (args.output_dir / "gemini-template-full-timeline.json").write_text(
        json.dumps(
            {
                "backend": "gemini-template+NUC-word-timeline",
                "evidence": "inferred_timestamps_not_frame_ocr",
                "cues": full_template_timeline,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_srt(args.output_dir / "gemini-template-full-timeline.srt", full_template_timeline)
    (args.output_dir / "gemini-template-ocr-anchored-timeline.json").write_text(
        json.dumps(
            {
                "backend": "gemini-template+NUC-word-timeline+apple-vision-ocr-anchors",
                "evidence": "OCR anchors correct cue boundaries; gaps interpolate NUC word timing",
                "anchor_report": anchor_report,
                "cues": ocr_anchored_full_template_timeline,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_srt(
        args.output_dir / "gemini-template-ocr-anchored-timeline.srt",
        ocr_anchored_full_template_timeline,
    )
    (args.output_dir / "burned-subtitle-gemini-template-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
