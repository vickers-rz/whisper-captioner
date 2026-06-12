"""
字幕输入输出处理模块

负责多种字幕格式（如 SRT, VTT, JSON）的解析、格式化和保存。
主要职责包括：
1. 从不同格式的文本文件中提取带时间戳的字幕片段（SubtitleSegment）。
2. 将字幕片段对象序列化并保存为指定的字幕格式。
3. 提供基于时间轴的字幕重叠匹配算法，用于辅助对齐和 LLM 后处理。
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .models import SubtitleSegment


SRT_BLOCK_RE = re.compile(
    r"(?ms)^\s*\d+\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3}).*?\n"
    r"(.*?)(?=\n\s*\d+\s*\n|\Z)"
)
SENSE_VOICE_BLOCK_RE = re.compile(r"^\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.+)$", re.MULTILINE)


def segment_to_dict(segment: SubtitleSegment) -> dict:
    return {"start": segment.start, "end": segment.end, "text": segment.text}


def _coerce_segment_number(value: object, field: str, path: Path, index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid subtitle cache at {path}: segment #{index} field '{field}' must be a number"
        ) from exc
    return number


def _validate_segment_mapping(data: object, path: Path, index: int) -> dict:
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid subtitle cache at {path}: segment #{index} must be an object"
        )
    missing = [field for field in ("start", "end", "text") if field not in data]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(
            f"Invalid subtitle cache at {path}: segment #{index} missing field(s): {missing_fields}"
        )
    return data


def segment_from_dict(data: object, path: Path | None = None, index: int = -1) -> SubtitleSegment:
    source_path = path or Path("<memory>")
    segment_index = index if index >= 0 else 0
    item = _validate_segment_mapping(data, source_path, segment_index)
    start = _coerce_segment_number(item["start"], "start", source_path, segment_index)
    end = _coerce_segment_number(item["end"], "end", source_path, segment_index)
    text = item["text"]
    if text is None:
        raise ValueError(
            f"Invalid subtitle cache at {source_path}: segment #{segment_index} field 'text' must not be null"
        )
    text_value = str(text).strip()
    if end <= start:
        raise ValueError(
            f"Invalid subtitle cache at {source_path}: segment #{segment_index} end must be greater than start"
        )
    return SubtitleSegment(start, end, text_value)


def save_segments(path: Path, segments: list[SubtitleSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps([segment_to_dict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_segments(path: Path) -> list[SubtitleSegment]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid subtitle cache at {path}: malformed JSON ({exc})") from exc
    if not isinstance(data, list):
        raise ValueError(f"Invalid subtitle cache at {path}: top-level JSON must be a list")
    return [segment_from_dict(item, path=path, index=index) for index, item in enumerate(data)]


def parse_srt_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def format_srt_timestamp(seconds: float) -> str:
    millis_total = max(0, int(round(seconds * 1000)))
    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    secs = total_seconds % 60
    minutes_total = total_seconds // 60
    minutes = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(path: Path) -> list[SubtitleSegment]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    segments: list[SubtitleSegment] = []
    for match in SRT_BLOCK_RE.finditer(content):
        text = re.sub(r"<[^>]+>", "", match.group(3))
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if text:
            segments.append(
                SubtitleSegment(
                    parse_srt_timestamp(match.group(1)),
                    parse_srt_timestamp(match.group(2)),
                    text,
                )
            )
    return segments


def parse_vtt_timestamp(value: str) -> float:
    parts = value.split(".")
    main = parts[0]
    millis = int(parts[1][:3]) if len(parts) > 1 else 0
    fields = [int(part) for part in main.split(":")]
    if len(fields) == 2:
        minutes, seconds = fields
        hours = 0
    else:
        hours, minutes, seconds = fields
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_vtt(path: Path) -> list[SubtitleSegment]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    content = re.sub(r"(?m)^WEBVTT.*$", "", content)
    blocks = re.split(r"\n\s*\n", content)
    segments: list[SubtitleSegment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        timing = lines[timing_index]
        start_raw, end_raw = timing.split("-->", 1)
        start = parse_vtt_timestamp(start_raw.strip().split()[0])
        end = parse_vtt_timestamp(end_raw.strip().split()[0])
        text_lines = lines[timing_index + 1 :]
        text = " ".join(re.sub(r"<[^>]+>", "", line).strip() for line in text_lines)
        text = re.sub(r"\s+", " ", html.unescape(text)).strip()
        if text:
            segments.append(SubtitleSegment(start, end, text))
    return segments


def parse_subtitle_file(path: Path) -> list[SubtitleSegment]:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt(path)
    if suffix == ".vtt":
        return parse_vtt(path)
    if suffix == ".txt":
        return parse_plaintext(path)
    return []


def parse_plaintext(path: Path) -> list[SubtitleSegment]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    segments: list[SubtitleSegment] = []
    cursor = 0.0
    for line in lines:
        end = cursor + 1.0
        segments.append(SubtitleSegment(cursor, end, line))
        cursor = end
    return segments


def parse_sense_voice_output(text: str) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    for match in SENSE_VOICE_BLOCK_RE.finditer(text):
        raw_text = re.sub(r"<\|[^|>]+\|>", "", match.group(3)).strip()
        clean_text = re.sub(r"\s+", " ", raw_text)
        if not clean_text:
            continue
        segments.append(
            SubtitleSegment(
                float(match.group(1)),
                float(match.group(2)),
                clean_text,
            )
        )
    return segments


def save_segments_as_srt(path: Path, segments: list[SubtitleSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, segment in enumerate(segments, 1):
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n"
            f"{segment.text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def save_segments_as_txt(path: Path, segments: list[SubtitleSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(segment.text for segment in segments), encoding="utf-8")


def overlapping_segments(
    segments: list[SubtitleSegment],
    start: float,
    end: float,
    tolerance: float = 0.75,
) -> list[SubtitleSegment]:
    return [
        segment
        for segment in segments
        if segment.end >= start - tolerance and segment.start <= end + tolerance
    ]
