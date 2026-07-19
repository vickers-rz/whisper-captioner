#!/usr/bin/env python3
"""Merge frame OCR into subtitle cues and compare it with a timed ASR baseline."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\u3400-\u9fff]+", text.lower()))


def similarity(left: str, right: str) -> float:
    a = normalize(left)
    b = normalize(right)
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(a)


def global_text_comparison(left: str, right: str) -> dict[str, Any]:
    a = normalize(left)
    b = normalize(right)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return {
        "ocr_characters": len(a),
        "asr_characters": len(b),
        "matched_characters": matched,
        "ocr_character_precision": round(matched / max(1, len(a)), 6),
        "asr_character_coverage": round(matched / max(1, len(b)), 6),
        "sequence_similarity": round(matcher.ratio(), 6),
    }


def subtitle_text(frame: dict[str, Any]) -> tuple[str, float, list[dict[str, Any]]]:
    candidates = []
    for observation in frame.get("observations", []):
        x = float(observation.get("x", 0))
        y = float(observation.get("y", 0))
        width = float(observation.get("width", 0))
        height = float(observation.get("height", 0))
        center_x = x + width / 2
        # Vision coordinates use a lower-left origin. In this video the hard
        # subtitle baseline fluctuates around y=0.09 while the fixed chapter
        # navigation is at y=0.94.
        if not (0.07 <= y <= 0.18 and 0.12 <= center_x <= 0.88):
            continue
        if (
            height < 0.018
            or float(observation.get("confidence", 0)) < 0.45
            or len(normalize(str(observation.get("text", "")))) < 2
        ):
            continue
        candidates.append(observation)
    candidates.sort(key=lambda item: (-float(item.get("y", 0)), float(item.get("x", 0))))
    text = " ".join(str(item.get("text", "")).strip() for item in candidates).strip()
    confidence = statistics.mean(
        [float(item.get("confidence", 0)) for item in candidates]
    ) if candidates else 0.0
    return text, confidence, candidates


def representative_text(samples: list[dict[str, Any]]) -> tuple[str, float, list[str]]:
    ranked = sorted(
        samples,
        key=lambda item: (
            round(float(item["confidence"]), 2),
            len(normalize(item["text"])),
        ),
        reverse=True,
    )
    representative = ranked[0]
    variants = sorted({str(item["text"]) for item in samples})
    return str(representative["text"]), float(representative["confidence"]), variants


def merge_frames(frames: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    samples = []
    for frame in frames:
        text, confidence, boxes = subtitle_text(frame)
        if text:
            samples.append(
                {
                    "timestamp": float(frame["timestamp"]),
                    "frame": frame["frame"],
                    "text": text,
                    "confidence": confidence,
                    "boxes": boxes,
                }
            )

    groups: list[list[dict[str, Any]]] = []
    for sample in samples:
        if not groups:
            groups.append([sample])
            continue
        previous = groups[-1][-1]
        gap = sample["timestamp"] - previous["timestamp"]
        same = similarity(sample["text"], previous["text"]) >= 0.72
        if gap <= max(1.0, 2.0 / fps) and same:
            groups[-1].append(sample)
        else:
            groups.append([sample])

    cues = []
    for group in groups:
        text, confidence, variants = representative_text(group)
        cues.append(
            {
                "start": group[0]["timestamp"],
                "end": group[-1]["timestamp"] + 1.0 / fps,
                "text": text,
                "confidence": round(confidence, 6),
                "sample_count": len(group),
                "frame": group[len(group) // 2]["frame"],
                "bbox": group[len(group) // 2]["boxes"],
                "ocr_variants": variants,
            }
        )
    return cues


def baseline_segments(path: Path, maximum_end: float) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        segment
        for segment in data.get("segments", [])
        if float(segment.get("start", 0)) < maximum_end
    ]


def compare_with_asr(cues: list[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
    for cue in cues:
        overlapping = [
            segment
            for segment in segments
            if float(segment.get("end", 0)) > cue["start"] - 2.0
            and float(segment.get("start", 0)) < cue["end"] + 2.0
        ]
        asr_text = "".join(str(segment.get("text", "")) for segment in overlapping)
        cue["asr_window_text"] = asr_text
        cue["asr_character_coverage"] = round(similarity(cue["text"], asr_text), 6)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=300.0)
    args = parser.parse_args()

    frames = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line]
    cues = merge_frames(frames, args.fps)
    segments = baseline_segments(args.asr, args.duration)
    compare_with_asr(cues, segments)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "backend": "apple-vision",
        "fps": args.fps,
        "duration": args.duration,
        "subtitle_roi_vision_coordinates": [0.0, 0.07, 1.0, 0.11],
        "frames_processed": len(frames),
        "frames_with_subtitle_text": sum(bool(subtitle_text(frame)[0]) for frame in frames),
        "cues": cues,
    }
    (args.output_dir / "burned-subtitle-ocr.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_srt(args.output_dir / "burned-subtitle-ocr.srt", cues)

    coverages = [float(cue["asr_character_coverage"]) for cue in cues]
    ocr_text = "".join(str(cue["text"]) for cue in cues)
    asr_text = "".join(str(segment.get("text", "")) for segment in segments)
    report = {
        "frames_processed": len(frames),
        "frames_with_subtitle_text": payload["frames_with_subtitle_text"],
        "frame_detection_rate": round(payload["frames_with_subtitle_text"] / max(1, len(frames)), 6),
        "cue_count": len(cues),
        "mean_ocr_confidence": round(
            statistics.mean(float(cue["confidence"]) for cue in cues) if cues else 0, 6
        ),
        "mean_asr_character_coverage": round(statistics.mean(coverages) if coverages else 0, 6),
        "median_asr_character_coverage": round(statistics.median(coverages) if coverages else 0, 6),
        "cues_at_least_80_percent_matched": sum(value >= 0.8 for value in coverages),
        "cues_at_least_60_percent_matched": sum(value >= 0.6 for value in coverages),
        "global_text_comparison": global_text_comparison(ocr_text, asr_text),
        "ocr_errors": [frame for frame in frames if frame.get("error")],
    }
    (args.output_dir / "burned-subtitle-ocr-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
