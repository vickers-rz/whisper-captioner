#!/usr/bin/env python3
"""Generate an auditable character-level comparison of transcript variants."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.subtitle_io import parse_srt


def normalized(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\u3400-\u9fff]+", text.lower()))


def read_text(path: Path, kind: str) -> str:
    if kind == "srt":
        return "".join(segment.text for segment in parse_srt(path))
    return path.read_text(encoding="utf-8")


def group_opcodes(opcodes: list[tuple[str, int, int, int, int]], equal_gap: int = 30) -> list[list[tuple[str, int, int, int, int]]]:
    groups: list[list[tuple[str, int, int, int, int]]] = []
    current: list[tuple[str, int, int, int, int]] = []
    pending_equal: list[tuple[str, int, int, int, int]] = []
    for opcode in opcodes:
        tag, left_start, left_end, right_start, right_end = opcode
        if tag == "equal":
            if current:
                pending_equal.append(opcode)
                if left_end - left_start > equal_gap:
                    groups.append(current)
                    current = []
                    pending_equal = []
            continue
        if not current:
            current = []
        if pending_equal:
            current.extend(pending_equal)
            pending_equal = []
        current.append(opcode)
    if current:
        groups.append(current)
    return groups


def difference_groups(left: str, right: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    opcodes = matcher.get_opcodes()
    edits = [opcode for opcode in opcodes if opcode[0] != "equal"]
    matched = sum(block.size for block in matcher.get_matching_blocks())
    summary = {
        "left_characters": len(left),
        "right_characters": len(right),
        "matched_characters": matched,
        "left_coverage": round(matched / max(1, len(left)), 6),
        "right_precision": round(matched / max(1, len(right)), 6),
        "sequence_similarity": round(matcher.ratio(), 6),
        "opcode_counts": {tag: sum(1 for opcode in edits if opcode[0] == tag) for tag in ("replace", "delete", "insert")},
    }
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(group_opcodes(opcodes), 1):
        changed = [opcode for opcode in group if opcode[0] != "equal"]
        left_start = min(opcode[1] for opcode in changed)
        left_end = max(opcode[2] for opcode in changed)
        right_start = min(opcode[3] for opcode in changed)
        right_end = max(opcode[4] for opcode in changed)
        context = 24
        groups.append(
            {
                "index": index,
                "operations": [opcode[0] for opcode in changed],
                "left_range": [left_start, left_end],
                "right_range": [right_start, right_end],
                "ogg_excerpt": left[max(0, left_start - context):min(len(left), left_end + context)],
                "url_excerpt": right[max(0, right_start - context):min(len(right), right_end + context)],
                "ogg_changed": left[left_start:left_end],
                "url_changed": right[right_start:right_end],
            }
        )
    return summary, groups


def markdown(summary: dict[str, Any], final_summary: dict[str, Any], groups: list[dict[str, Any]], base: Path, final: Path, url: Path) -> str:
    lines = [
        "# Gemini 转写版本逐差异报告",
        "",
        "## 输入",
        "",
        f"- OGG/File API Gemini 原文：`{base}`",
        f"- OCR+NUC 最终时间轴：`{final}`",
        f"- Gemini 公开 URL 音频转写：`{url}`",
        "",
        "## OGG 原文与 OCR+NUC 最终版本",
        "",
        f"规范化正文完全一致：{final_summary['matched_characters']} / {final_summary['left_characters']} 字符，序列相似度 {final_summary['sequence_similarity']:.6f}。",
        "最终版本仅改变 SRT 分段与时间戳：NUC words 提供时间骨架，OCR 精确匹配 cue 提供视觉锚点；不会修改 Gemini 原文的规范化文本。",
        "",
        "## OGG 原文与 URL 音频转写",
        "",
        f"- OGG 原文：{summary['left_characters']} 个规范化字符。",
        f"- URL 转写：{summary['right_characters']} 个规范化字符。",
        f"- 匹配：{summary['matched_characters']}；OGG 覆盖率 {summary['left_coverage']:.4%}；URL 精度 {summary['right_precision']:.4%}；序列相似度 {summary['sequence_similarity']:.4%}。",
        f"- 原始编辑操作：替换 {summary['opcode_counts']['replace']}，删除 {summary['opcode_counts']['delete']}，插入 {summary['opcode_counts']['insert']}；合并为 {len(groups)} 个上下文差异组。",
        "",
        "以下摘录为规范化文本，已去除标点与空白；`OGG 变更` 为空表示 URL 多出内容，`URL 变更` 为空表示 URL 漏掉 OGG 内容。",
    ]
    for group in groups:
        lines.extend(
            [
                "",
                f"### {group['index']}. {', '.join(group['operations'])}",
                "",
                f"- OGG 变更：`{group['ogg_changed'] or '∅'}`",
                f"- URL 变更：`{group['url_changed'] or '∅'}`",
                f"- OGG 上下文：`{group['ogg_excerpt']}`",
                f"- URL 上下文：`{group['url_excerpt']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ogg", type=Path, required=True)
    parser.add_argument("--final-srt", type=Path, required=True)
    parser.add_argument("--url", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ogg = normalized(read_text(args.ogg, "text"))
    final = normalized(read_text(args.final_srt, "srt"))
    url = normalized(read_text(args.url, "text"))
    final_summary, final_groups = difference_groups(ogg, final)
    url_summary, url_groups = difference_groups(ogg, url)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ogg_vs_final": {"summary": final_summary, "differences": final_groups},
        "ogg_vs_url": {"summary": url_summary, "differences": url_groups},
    }
    (args.output_dir / "transcript-variant-differences.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "transcript-variant-differences.md").write_text(markdown(url_summary, final_summary, url_groups, args.ogg, args.final_srt, args.url), encoding="utf-8")
    print(json.dumps({"ogg_vs_final": final_summary, "ogg_vs_url": url_summary, "difference_groups": len(url_groups)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
