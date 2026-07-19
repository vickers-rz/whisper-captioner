#!/usr/bin/env python3
"""Apply only OCR-supported URL corrections to an OGG-based SRT timeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.subtitle_io import parse_srt


def normalized_char_locations(segments: list[Any]) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    for segment_index, segment in enumerate(segments):
        for char_index, char in enumerate(segment.text):
            if re.fullmatch(r"[0-9A-Za-z\u3400-\u9fff]", char.lower()):
                locations.append((segment_index, char_index))
    return locations


def timestamp(value: float) -> str:
    millis = round(max(0, value) * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-srt", type=Path, required=True)
    parser.add_argument("--adjudication-json", type=Path, required=True)
    parser.add_argument("--output-srt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    segments = parse_srt(args.final_srt)
    locations = normalized_char_locations(segments)
    decisions = json.loads(args.adjudication_json.read_text(encoding="utf-8"))["decisions"]
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.get("verdict") != "ocr_supports_url":
            continue
        start, end = map(int, decision["left_range"])
        old, new = str(decision["ogg_changed"]), str(decision["url_changed"])
        if end - start != 1 or len(old) != 1 or start >= len(locations):
            skipped.append({"index": decision["index"], "reason": "non-single-character edit"})
            continue
        segment_index, char_index = locations[start]
        segment = segments[segment_index]
        if segment.text[char_index] != old:
            skipped.append({"index": decision["index"], "reason": "source character mismatch"})
            continue
        segments[segment_index] = type(segment)(
            start=segment.start,
            end=segment.end,
            text=segment.text[:char_index] + new + segment.text[char_index + 1:],
        )
        applied.append({"index": decision["index"], "old": old, "new": new, "segment": segment_index + 1, "start": segment.start, "end": segment.end})
    blocks = [f"{index}\n{timestamp(segment.start)} --> {timestamp(segment.end)}\n{segment.text}" for index, segment in enumerate(segments, 1)]
    args.output_srt.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps({"source": str(args.final_srt), "applied": applied, "skipped": skipped}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"applied": applied, "skipped": skipped}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
