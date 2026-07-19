#!/usr/bin/env python3
"""Create readable subtitle cues from an OCR-anchored Gemini transcript.

The source transcript is immutable.  A local LLM is only permitted to insert
``|`` boundary markers; every reply is rejected unless removing those markers
recovers the input chunk byte-for-byte.  NUC word timestamps remain the timing
backbone, while the OCR-anchored cue boundaries are used as monotonic visual
timing anchors.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.llm_handler import _build_llm_call, _extract_llm_reply, _llm_request
from whisper_captioner.models import LLMProvider


def normalized_chars(text: str) -> list[str]:
    chars: list[str] = []
    for char in text:
        for normalized in unicodedata.normalize("NFKC", char).lower():
            if normalized.isalnum() or "\u3400" <= normalized <= "\u9fff":
                chars.append(normalized)
    return chars


def normalized_chars_with_indices(text: str) -> tuple[list[str], list[int]]:
    chars: list[str] = []
    indices: list[int] = []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).lower():
            if normalized.isalnum() or "\u3400" <= normalized <= "\u9fff":
                chars.append(normalized)
                indices.append(index)
    return chars, indices


def build_nuc_timeline(asr: dict[str, Any]) -> tuple[list[str], list[float]]:
    chars: list[str] = []
    times: list[float] = []
    for word in asr.get("words", []):
        word_chars = normalized_chars(str(word.get("text", "")))
        if not word_chars:
            continue
        start = float(word.get("start", 0.0))
        end = max(start, float(word.get("end", start)))
        for index, char in enumerate(word_chars):
            chars.append(char)
            times.append(start + (end - start) * (index + 0.5) / len(word_chars))
    return chars, times


def interpolate_times(template: list[str], nuc: list[str], nuc_times: list[float]) -> tuple[list[float], int]:
    matcher = difflib.SequenceMatcher(None, nuc, template, autojunk=False)
    assigned: dict[int, float] = {}
    for nuc_start, template_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            assigned[template_start + offset] = nuc_times[nuc_start + offset]
    if not assigned:
        raise RuntimeError("Gemini text and NUC words have no aligned characters")
    anchors = sorted(assigned.items())
    positions = [item[0] for item in anchors]
    result: list[float] = []
    for index in range(len(template)):
        if index in assigned:
            result.append(assigned[index])
            continue
        insertion = bisect.bisect_left(positions, index)
        if insertion == 0:
            result.append(anchors[0][1])
        elif insertion == len(anchors):
            result.append(anchors[-1][1])
        else:
            left_index, left_time = anchors[insertion - 1]
            right_index, right_time = anchors[insertion]
            ratio = (index - left_index) / max(1, right_index - left_index)
            result.append(left_time + (right_time - left_time) * ratio)
    return result, len(assigned)


def apply_visual_anchors(times: list[float], cues: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    """Apply the existing OCR anchor timing without adopting its cue breaks."""
    anchors: list[tuple[int, float]] = []
    for cue in cues:
        start = int(cue.get("template_character_start", -1))
        end = int(cue.get("template_character_end", -1))
        if 0 <= start < end <= len(times):
            anchors.extend(((start, float(cue["start"])), (end - 1, float(cue["end"]))))
    grouped: dict[int, list[float]] = {}
    for index, value in anchors:
        grouped.setdefault(index, []).append(value)
    points = sorted((index, statistics.median(values)) for index, values in grouped.items())
    if len(points) < 2:
        return times, {"anchor_points": len(points), "mean_absolute_offset_seconds": 0.0}
    positions = [index for index, _ in points]
    offsets = [value - times[index] for index, value in points]
    warped: list[float] = []
    for index, value in enumerate(times):
        insertion = bisect.bisect_left(positions, index)
        if insertion == 0:
            offset = offsets[0]
        elif insertion == len(points):
            offset = offsets[-1]
        else:
            left, right = positions[insertion - 1], positions[insertion]
            ratio = (index - left) / max(1, right - left)
            offset = offsets[insertion - 1] + (offsets[insertion] - offsets[insertion - 1]) * ratio
        warped.append(max(warped[-1] if warped else 0.0, value + offset))
    return warped, {
        "anchor_points": len(points),
        "mean_absolute_offset_seconds": round(statistics.mean(abs(item) for item in offsets), 6),
    }


def expand_times_to_text(text: str, normalized_indices: list[int], times: list[float]) -> list[float]:
    """Give punctuation the interpolated time between neighboring spoken chars."""
    assigned = {original: times[index] for index, original in enumerate(normalized_indices)}
    anchors = sorted(assigned)
    output: list[float] = []
    for index in range(len(text)):
        if index in assigned:
            output.append(assigned[index])
            continue
        insertion = bisect.bisect_left(anchors, index)
        if insertion == 0:
            output.append(times[0])
        elif insertion == len(anchors):
            output.append(times[-1])
        else:
            left, right = anchors[insertion - 1], anchors[insertion]
            ratio = (index - left) / max(1, right - left)
            output.append(assigned[left] + (assigned[right] - assigned[left]) * ratio)
    return output


def chunk_ranges(text: str, target: int = 760, maximum: int = 980) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target)
        if end < len(text):
            for index in range(min(len(text) - 1, start + maximum), end - 1, -1):
                if text[index] in "。！？；":
                    end = index + 1
                    break
            else:
                for index in range(end, min(len(text), start + maximum)):
                    if text[index] in "。！？；，":
                        end = index + 1
                        break
        ranges.append((start, end))
        start = end
    return ranges


SEGMENT_PROMPT = """你是中文纪录片字幕编辑。下面的原文不可改动、不可删减、不可增补、不可修正标点。
只允许在适合字幕停顿的位置插入 ASCII 字符 | 。优先按完整句、分句、转折、因果、并列和引语边界断开；避免把定语、专名、介词结构、固定搭配和主谓宾强行拆散。
每段通常 12-26 个汉字；句意未完时可以稍长。只输出插入 | 后的原文，不要 Markdown、解释或代码块。
"""

CHOICE_PROMPT = """你是中文纪录片字幕编辑。请从候选标点边界中挑选适合字幕停顿的编号。
优先完整句、分句、转折、因果、并列和引语边界；不要在定语、专名、固定搭配或主谓宾中间断开。
只输出英文逗号分隔的编号，例如 1,4,7。不要解释、不要输出原文、不要添加其他字符。
"""


def llm_boundaries(text: str, provider: LLMProvider) -> tuple[set[int], bool, str]:
    url, body, headers = _build_llm_call(
        provider, "", text, max_tokens=1800, system_prompt=SEGMENT_PROMPT,
    )
    raw = _llm_request(url, body, headers, timeout=300)
    reply = _extract_llm_reply(json.loads(raw), provider.format).strip()
    # Models occasionally wrap the answer in a code fence.  Search each
    # paragraph for a lossless marked reconstruction rather than trusting prose.
    candidates = [reply, *reply.splitlines()]
    for candidate in candidates:
        candidate = candidate.strip().strip("`")
        if candidate.replace("|", "") == text:
            boundaries: set[int] = set()
            position = 0
            for char in candidate:
                if char == "|":
                    if 0 < position < len(text):
                        boundaries.add(position)
                else:
                    position += 1
            return boundaries, True, "accepted"
    return set(), False, "rejected_non_lossless_reply"


def llm_punctuation_choices(text: str, provider: LLMProvider) -> tuple[set[int], bool, str]:
    candidates = [index + 1 for index, char in enumerate(text) if char in "，、：；。！？"]
    if not candidates:
        return set(), True, "no_punctuation_candidates"
    lines = []
    for number, position in enumerate(candidates, 1):
        left = text[max(0, position - 18):position]
        right = text[position:min(len(text), position + 18)]
        lines.append(f"{number}: {left}[边界]{right}")
    provider_prompt = CHOICE_PROMPT + "\n候选边界：\n" + "\n".join(lines)
    url, body, headers = _build_llm_call(
        provider, "", provider_prompt, max_tokens=260,
        system_prompt="严格遵守用户输出格式。",
    )
    raw = _llm_request(url, body, headers, timeout=120)
    reply = _extract_llm_reply(json.loads(raw), provider.format)
    chosen_ids = {int(item) for item in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", reply)}
    valid_ids = {item for item in chosen_ids if 1 <= item <= len(candidates)}
    if not valid_ids:
        return set(), False, "rejected_no_valid_candidate_ids"
    return {candidates[item - 1] for item in valid_ids}, True, "accepted_punctuation_ids"


def natural_boundaries(
    text: str, ranges: list[tuple[int, int]], provider: LLMProvider
) -> tuple[set[int], list[dict[str, Any]]]:
    all_boundaries: set[int] = set()
    audits: list[dict[str, Any]] = []
    for start, end in ranges:
        boundaries, accepted, reason = llm_boundaries(text[start:end], provider)
        if not accepted:
            boundaries, accepted, reason = llm_punctuation_choices(text[start:end], provider)
        all_boundaries.update(start + item for item in boundaries)
        audits.append({"start": start, "end": end, "llm_accepted": accepted, "reason": reason, "boundaries": len(boundaries)})
    return all_boundaries, audits


def choose_cues(text: str, times: list[float], preferred: set[int]) -> list[dict[str, Any]]:
    """Constrain semantic breaks to readable cue lengths and speech durations."""
    cues: list[dict[str, Any]] = []
    start = 0
    strong = "。！？；"
    weak = "，、："
    while start < len(text):
        remaining = len(text) - start
        if remaining <= 30:
            end = len(text)
        else:
            candidates: list[tuple[float, int]] = []
            for end in range(start + 8, min(len(text), start + 34) + 1):
                duration = times[end - 1] - times[start]
                if duration < 0.75:
                    continue
                score = abs(end - start - 20) * 0.22 + abs(duration - 3.1) * 1.1
                previous = text[end - 1]
                if end in preferred:
                    score -= 5.0
                if previous in strong:
                    score -= 3.0
                elif previous in weak:
                    score -= 1.2
                # Do not leave a closing quote/bracket stranded at the start.
                if end < len(text) and text[end] in "》）】”’":
                    score += 5.0
                candidates.append((score, end))
            if not candidates:
                end = min(len(text), start + 24)
            else:
                end = min(candidates)[1]
        cue_start = max(0.0, times[start] - 0.08)
        cue_end = times[end - 1] + 0.08
        if cues:
            cue_start = max(cue_start, cues[-1]["end"])
        if cue_end <= cue_start:
            cue_end = cue_start + 0.12
        cues.append({
            "start": round(cue_start, 3), "end": round(cue_end, 3),
            "text": text[start:end], "character_start": start, "character_end": end,
            "preferred_boundary": end in preferred,
        })
        start = end
    return cues


def srt_timestamp(value: float) -> str:
    milliseconds = max(0, round(value * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(path: Path, cues: list[dict[str, Any]]) -> None:
    path.write_text("\n\n".join(
        f"{index}\n{srt_timestamp(cue['start'])} --> {srt_timestamp(cue['end'])}\n{cue['text']}"
        for index, cue in enumerate(cues, 1)
    ) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchored", type=Path, required=True)
    parser.add_argument("--nuc-asr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("local-rapidmlx-8b", "nuc-qwen3-14b"), default="local-rapidmlx-8b")
    args = parser.parse_args()
    anchored = json.loads(args.anchored.read_text(encoding="utf-8"))
    source_cues = list(anchored.get("cues", []))
    text = "".join(str(cue.get("text", "")) for cue in source_cues)
    if not text:
        raise RuntimeError("No text cues in OCR-anchored timeline")
    template, normalized_indices = normalized_chars_with_indices(text)
    nuc, nuc_times = build_nuc_timeline(json.loads(args.nuc_asr.read_text(encoding="utf-8")))
    times, aligned = interpolate_times(template, nuc, nuc_times)
    times, anchor_report = apply_visual_anchors(times, source_cues)
    text_times = expand_times_to_text(text, normalized_indices, times)
    providers = {
        "local-rapidmlx-8b": LLMProvider(
            "local_rapidmlx_8b", "Local Rapid-MLX Qwen3-8B",
            "http://127.0.0.1:8766/v1/chat/completions", "qwen3-8b-mlx", "openai", False,
        ),
        "nuc-qwen3-14b": LLMProvider(
            "nuc_ollama_14b", "NUC Ollama Qwen3-14B",
            "http://192.168.31.196:11434/api/chat", "qwen3-14b-nothink:latest", "ollama", False,
        ),
    }
    provider = providers[args.provider]
    ranges = chunk_ranges(text)
    preferred, llm_audit = natural_boundaries(text, ranges, provider)
    cues = choose_cues(text, text_times, preferred)
    if "".join(cue["text"] for cue in cues) != text:
        raise RuntimeError("Cue reconstruction changed the source transcript")
    report = {
        "source": str(args.anchored), "nuc_asr": str(args.nuc_asr), "llm_provider": args.provider,
        "source_characters": len(text), "nuc_matched_characters": aligned,
        "visual_anchor_report": anchor_report, "llm_chunks": llm_audit,
        "accepted_llm_chunks": sum(item["llm_accepted"] for item in llm_audit),
        "cue_count": len(cues),
        "mean_characters_per_cue": round(statistics.mean(len(item["text"]) for item in cues), 3),
        "mean_duration_seconds": round(statistics.mean(item["end"] - item["start"] for item in cues), 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "semantic-ocr-anchored-timeline.json").write_text(
        json.dumps({"backend": "local-qwen3-8b-semantic-breaks+NUC-words+OCR-anchors", "cues": cues}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_srt(args.output_dir / "semantic-ocr-anchored-timeline.srt", cues)
    (args.output_dir / "semantic-ocr-anchored-timeline-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
