#!/usr/bin/env python3
"""Plan dense, targeted OCR sampling from transcript disagreements and NUC words."""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import statistics
import unicodedata
from pathlib import Path
from typing import Any


def normalized_chars(text: str) -> list[str]:
    result: list[str] = []
    for char in text:
        for normalized in unicodedata.normalize("NFKC", char).lower():
            if normalized.isalnum() or "\u3400" <= normalized <= "\u9fff":
                result.append(normalized)
    return result


def nuc_timeline(asr: dict[str, Any]) -> tuple[list[str], list[float]]:
    chars: list[str] = []
    times: list[float] = []
    for word in asr.get("words", []):
        word_chars = normalized_chars(str(word.get("text", "")))
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        for index, char in enumerate(word_chars):
            chars.append(char)
            times.append(start + (end - start) * (index + 0.5) / len(word_chars))
    return chars, times


def template_times(template: list[str], nuc_chars: list[str], nuc_times: list[float]) -> list[float]:
    matcher = difflib.SequenceMatcher(None, nuc_chars, template, autojunk=False)
    assigned: dict[int, float] = {}
    for nuc_start, template_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            assigned[template_start + offset] = nuc_times[nuc_start + offset]
    if not assigned:
        raise RuntimeError("No Gemini/NUC character alignment")
    anchors = sorted(assigned.items())
    positions = [item[0] for item in anchors]
    output: list[float] = []
    for index in range(len(template)):
        if index in assigned:
            output.append(assigned[index])
            continue
        insertion = bisect.bisect_left(positions, index)
        if insertion == 0:
            output.append(anchors[0][1])
        elif insertion == len(anchors):
            output.append(anchors[-1][1])
        else:
            left_index, left_time = anchors[insertion - 1]
            right_index, right_time = anchors[insertion]
            output.append(left_time + (right_time - left_time) * (index - left_index) / max(1, right_index - left_index))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ogg-transcript", type=Path, required=True)
    parser.add_argument("--nuc-asr", type=Path, required=True)
    parser.add_argument("--differences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--padding", type=float, default=0.8)
    parser.add_argument("--merge-gap", type=float, default=0.5)
    parser.add_argument("--long-window", type=float, default=8.0)
    parser.add_argument("--long-fps", type=float, default=2.0)
    parser.add_argument("--edge-seconds", type=float, default=1.5)
    args = parser.parse_args()
    template = normalized_chars(args.ogg_transcript.read_text(encoding="utf-8"))
    asr = json.loads(args.nuc_asr.read_text(encoding="utf-8"))
    nuc_chars, nuc_times = nuc_timeline(asr)
    times = template_times(template, nuc_chars, nuc_times)
    difference_payload = json.loads(args.differences.read_text(encoding="utf-8"))
    # New deterministic command output compares NUC text with Gemini directly.
    # Keep compatibility with the earlier OGG-vs-URL comparison report.
    if "difference_groups" in difference_payload:
        differences = difference_payload["difference_groups"]
        range_key = "gemini_range"
    else:
        differences = difference_payload["ogg_vs_url"]["differences"]
        range_key = "left_range"
    raw_windows: list[dict[str, Any]] = []
    for difference in differences:
        start, end = map(int, difference[range_key])
        if start >= len(times):
            continue
        left_time = times[max(0, start)]
        right_time = times[min(len(times) - 1, max(start, end - 1))]
        raw_windows.append(
            {
                "difference_indexes": [difference["index"]],
                "start": max(0.0, min(left_time, right_time) - args.padding),
                "end": max(left_time, right_time) + args.padding,
            }
        )
    raw_windows.sort(key=lambda item: item["start"])
    merged: list[dict[str, Any]] = []
    for window in raw_windows:
        if merged and window["start"] <= merged[-1]["end"] + args.merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], window["end"])
            merged[-1]["difference_indexes"].extend(window["difference_indexes"])
        else:
            merged.append(window)
    adaptive: list[dict[str, Any]] = []
    for window in merged:
        duration = window["end"] - window["start"]
        if duration <= args.long_window:
            adaptive.append({**window, "sampling_reason": "short_disagreement"})
            continue
        edge_end = min(window["end"], window["start"] + args.edge_seconds)
        edge_start = max(edge_end, window["end"] - args.edge_seconds)
        adaptive.extend(
            [
                {"difference_indexes": window["difference_indexes"], "start": window["start"], "end": edge_end, "fps": args.fps, "sampling_reason": "long_disagreement_start_edge"},
                {"difference_indexes": window["difference_indexes"], "start": edge_end, "end": edge_start, "fps": args.long_fps, "sampling_reason": "long_disagreement_body"},
                {"difference_indexes": window["difference_indexes"], "start": edge_start, "end": window["end"], "fps": args.fps, "sampling_reason": "long_disagreement_end_edge"},
            ]
        )
    for index, window in enumerate(adaptive, 1):
        window["id"] = index
        window.setdefault("fps", args.fps)
        window["estimated_frames"] = max(1, round((window["end"] - window["start"]) * window["fps"]))
    output = {
        "strategy": "NUC-word-timeline-targeted-OCR-v1",
        "input_difference_groups": len(differences),
        "sampling_fps": args.fps,
        "padding_seconds": args.padding,
        "long_window_seconds": args.long_window,
        "long_window_body_fps": args.long_fps,
        "windows": adaptive,
        "estimated_frames": sum(window["estimated_frames"] for window in adaptive),
        "median_window_seconds": round(statistics.median(window["end"] - window["start"] for window in adaptive), 3) if adaptive else 0.0,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
