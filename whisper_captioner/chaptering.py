"""
视频章节处理模块

负责解析大语言模型生成的 JSON 章节数据，并将其应用到现有的字幕片段中。
主要职责包括：
1. 从 LLM 的返回结果中提取并反序列化结构化的章节对象 (VideoChapter)。
2. 提供将章节信息格式化为 Markdown 报告的功能。
3. 将章节标题作为特殊的标注块，直接插入或拼接到现有的字幕时间轴上。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from whisper_captioner.models import SubtitleSegment


@dataclass(frozen=True)
class VideoChapter:
    start_seconds: float
    title: str
    description: str = ""


def parse_chapters_response(text: str) -> list[VideoChapter]:
    payload = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", payload, re.IGNORECASE)
    if fenced:
        payload = fenced.group(1).strip()
    if not payload.startswith("["):
        start = payload.find("[")
        end = payload.rfind("]")
        if start >= 0 and end > start:
            payload = payload[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"章节结果不是有效 JSON：{exc}") from exc
    if not isinstance(data, list):
        raise ValueError("章节结果必须是 JSON 数组")

    chapters: list[VideoChapter] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"章节 #{index + 1} 必须是对象")
        try:
            start_seconds = max(0.0, float(item["start_seconds"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"章节 #{index + 1} 缺少有效的 start_seconds") from exc
        title = str(item.get("title", "")).strip()
        if not title:
            raise ValueError(f"章节 #{index + 1} 缺少标题")
        description = str(item.get("description", "")).strip()
        chapters.append(VideoChapter(start_seconds, title, description))

    chapters.sort(key=lambda chapter: chapter.start_seconds)
    deduplicated: list[VideoChapter] = []
    for chapter in chapters:
        if deduplicated and abs(chapter.start_seconds - deduplicated[-1].start_seconds) < 1:
            continue
        deduplicated.append(chapter)
    if not deduplicated:
        raise ValueError("LLM 没有生成任何章节")
    return deduplicated


def chapters_to_json(chapters: list[VideoChapter]) -> str:
    return json.dumps(
        [
            {
                "start_seconds": chapter.start_seconds,
                "title": chapter.title,
                "description": chapter.description,
            }
            for chapter in chapters
        ],
        ensure_ascii=False,
        indent=2,
    )


def chapters_to_markdown(chapters: list[VideoChapter]) -> str:
    lines = ["# 视频章节", ""]
    for chapter in chapters:
        total = int(chapter.start_seconds)
        timestamp = f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
        lines.append(f"## [{timestamp}] {chapter.title}")
        if chapter.description:
            lines.extend(["", chapter.description])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def add_chapters_to_subtitles(
    segments: list[SubtitleSegment],
    chapters: list[VideoChapter],
) -> list[SubtitleSegment]:
    updated = list(segments)
    for chapter in reversed(chapters):
        marker = f"【章节：{chapter.title}】"
        if chapter.description:
            marker += f"\n{chapter.description}"
        target_index = next(
            (
                index
                for index, segment in enumerate(updated)
                if segment.start >= chapter.start_seconds - 0.5
            ),
            None,
        )
        if target_index is None:
            updated.append(
                SubtitleSegment(
                    chapter.start_seconds,
                    chapter.start_seconds + 3.0,
                    marker,
                )
            )
            continue
        target = updated[target_index]
        updated[target_index] = SubtitleSegment(
            target.start,
            target.end,
            f"{marker}\n{target.text}",
        )
    return sorted(updated, key=lambda segment: (segment.start, segment.end))
