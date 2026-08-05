#!/usr/bin/env python3
"""Command-line entry point for the deterministic subtitle-forensics workflow.

Examples:
  # Download only seven low-resolution, three-second samples and detect hard subs
  python scripts/forensic_subtitle_command.py probe-hard-subs URL --output-dir artifacts/probe

  # Gemini text backfill + deterministic re-segmentation from NUC word timings
  python scripts/forensic_subtitle_command.py finalize \
    --nuc-asr nuc-asr.json --gemini gemini.txt --output-dir artifacts/final
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.config import FFMPEG, YT_DLP
from whisper_captioner.subtitle_io import format_srt_timestamp
from scripts.chrome_cookie_snapshot import yt_dlp_cookie_session

LLMBreakProvider = Callable[[str, list[int], list[float], int, int, int, float], list[int]]


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def normalized_with_indices(text: str) -> tuple[list[str], list[int]]:
    chars: list[str] = []
    indices: list[int] = []
    for index, char in enumerate(text):
        for item in unicodedata.normalize("NFKC", char).lower():
            if item.isalnum() or "\u3400" <= item <= "\u9fff":
                chars.append(item)
                indices.append(index)
    return chars, indices


def nuc_character_times(asr: dict[str, Any]) -> tuple[list[str], list[float]]:
    chars: list[str] = []
    times: list[float] = []
    for word in asr.get("words", []):
        word_chars, _ = normalized_with_indices(str(word.get("text", "")))
        if not word_chars:
            continue
        start = float(word.get("start", 0.0))
        end = max(start, float(word.get("end", start)))
        for index, char in enumerate(word_chars):
            chars.append(char)
            times.append(start + (end - start) * (index + 0.5) / len(word_chars))
    return chars, times


def aligned_template_times(template: list[str], nuc: list[str], nuc_times: list[float]) -> tuple[list[float], int]:
    matched: dict[int, float] = {}
    for nuc_start, template_start, size in difflib.SequenceMatcher(None, nuc, template, autojunk=False).get_matching_blocks():
        for offset in range(size):
            matched[template_start + offset] = nuc_times[nuc_start + offset]
    if not matched:
        raise RuntimeError("Gemini transcript has no character alignment with NUC words")
    anchors = sorted(matched.items())
    positions = [item[0] for item in anchors]
    output: list[float] = []
    for index in range(len(template)):
        if index in matched:
            output.append(matched[index])
            continue
        insertion = bisect.bisect_left(positions, index)
        if insertion == 0:
            output.append(anchors[0][1])
        elif insertion == len(anchors):
            output.append(anchors[-1][1])
        else:
            left_i, left_t = anchors[insertion - 1]
            right_i, right_t = anchors[insertion]
            output.append(left_t + (right_t - left_t) * (index - left_i) / max(1, right_i - left_i))
    return output, len(matched)


def original_span(text: str, indices: list[int], start: int, end: int) -> str:
    if start >= end:
        return ""
    span_start = indices[start]
    leading = "《（【“‘「『("
    while span_start > 0 and text[span_start - 1] in leading:
        span_start -= 1
    span_end = indices[end - 1] + 1
    limit = indices[end] if end < len(indices) else len(text)
    trailing = "，。！？、：；,.!?;:》）】”’」』"
    while span_end < limit and text[span_end] in trailing:
        span_end += 1
    return text[span_start:span_end].strip()


def optional_jieba_boundaries(chars: list[str]) -> set[int] | None:
    try:
        import jieba  # type: ignore
    except Exception:
        return None
    for word in ("青蛙刀圣", "沙丘", "弗雷曼人", "厄崔迪", "哈克南", "贝尼杰瑟里特", "香料", "宇航工会"):
        jieba.add_word(word)
    boundaries = {0, len(chars)}
    offset = 0
    for word in jieba.cut("".join(chars), HMM=False):
        word_chars, _ = normalized_with_indices(word)
        offset += len(word_chars)
        boundaries.add(offset)
    return boundaries


def is_ascii_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def boundary_trailing_text(text: str, indices: list[int], boundary: int) -> str:
    if boundary <= 0:
        return ""
    span_end = indices[boundary - 1] + 1
    limit = indices[boundary] if boundary < len(indices) else len(text)
    return text[span_end:limit]


def boundary_has_punctuation(text: str, indices: list[int], boundary: int, punctuation: str) -> bool:
    if boundary <= 0:
        return False
    char = text[indices[boundary - 1]]
    return char in punctuation or any(item in punctuation for item in boundary_trailing_text(text, indices, boundary))


DANGLING_NEXT_CHARS = "，。！？、：；,.!?;:》）】”’」』的了着过和与及或在是有这那其为以被把对从到中上下内外后前里之而但就都也又还更能会要将让给"
DANGLING_PREVIOUS_CHARS = "其土实穆伊弗雷厄哈贝阿欧美反殖宗预救保男女公皇宇香沙主核"


def write_srt(path: Path, cues: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(
        f"{number}\n{format_srt_timestamp(cue['start'])} --> {format_srt_timestamp(cue['end'])}\n{cue['text']}\n"
        for number, cue in enumerate(cues, 1)
    ), encoding="utf-8")


def build_backfill_cues(asr: dict[str, Any], text: str, indices: list[int], times: list[float]) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for segment in asr.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        left = bisect.bisect_left(times, start)
        right = bisect.bisect_right(times, end)
        replacement = original_span(text, indices, left, right)
        cues.append({"start": start, "end": max(start + 0.001, end), "text": replacement or str(segment.get("text", ""))})
    return cues


def build_differences(nuc: list[str], template: list[str], template_times: list[float]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, (tag, left_start, left_end, right_start, right_end) in enumerate(
        difflib.SequenceMatcher(None, nuc, template, autojunk=False).get_opcodes(), 1
    ):
        if tag == "equal":
            continue
        anchors = [item for item in (right_start, right_end - 1) if 0 <= item < len(template_times)]
        time_window = [template_times[min(anchors)], template_times[max(anchors)]] if anchors else [None, None]
        groups.append({
            "index": index, "operation": tag, "nuc_range": [left_start, left_end],
            "gemini_range": [right_start, right_end], "nuc_text": "".join(nuc[left_start:left_end]),
            "gemini_text": "".join(template[right_start:right_end]), "time_window": time_window,
        })
    return groups


def semantic_units(text: str, indices: list[int]) -> list[tuple[int, int]]:
    """Return normalized-character spans that preserve Gemini line and sentence boundaries."""
    units: list[tuple[int, int]] = []
    start = 0
    strong = "。！？；"
    for position, original_index in enumerate(indices):
        next_original = indices[position + 1] if position + 1 < len(indices) else len(text)
        between = text[original_index + 1:next_original]
        if text[original_index] in strong or any(item in strong for item in between) or "\n" in between:
            end = position + 1
            if end > start:
                units.append((start, end))
            start = end
    if start < len(indices):
        units.append((start, len(indices)))
    return units


def cue_from_span(text: str, indices: list[int], times: list[float], cues: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    cue_start = max(0.0, times[start] - 0.08)
    if cues:
        cue_start = max(cue_start, cues[-1]["end"])
    cue_end = max(cue_start + 0.1, times[end - 1] + 0.08)
    return {
        "start": round(cue_start, 3), "end": round(cue_end, 3),
        "text": original_span(text, indices, start, end),
        "template_character_start": start, "template_character_end": end,
    }


def choose_unit_break(
    text: str,
    indices: list[int],
    times: list[float],
    start: int,
    end: int,
    *,
    min_chars: int,
    max_chars: int,
    max_duration: float,
    target_duration: float = 4.2,
    word_boundaries: set[int] | None = None,
) -> int:
    strong, weak = "。！？；", "，、："
    bad_split_pairs = {
        "提出", "提升", "行动", "行事", "最终", "主动", "直接", "间接", "相信", "预言", "实现", "民族", "独立", "斗争", "描写",
        "土著", "弗雷", "雷曼", "保罗", "厄崔", "哈克", "穆斯", "斯林", "伊斯", "斯兰",
        "沙丘", "香料", "皇帝", "公爵", "姐妹", "宇航", "工会", "家族", "救世", "殖民",
        "宗教", "文化", "社会", "问题", "故事", "电影", "小说", "作者", "人类", "科技",
        "复杂", "计算", "思考", "行政", "决策", "高级", "任务", "越来越", "不断",
    }
    lower = min(end, start + max(1, min_chars))
    upper = min(end, start + max(1, max_chars))
    soft_upper = min(end, upper + 4)
    if upper <= start:
        return end
    punctuation_available = any(boundary_has_punctuation(text, indices, position, strong + weak) for position in range(lower, upper + 1))
    internal_punctuation_available = any(
        boundary_has_punctuation(text, indices, position, weak)
        for position in range(lower, end)
    )
    effective_target_duration = min(target_duration, 2.8)
    candidates: list[tuple[float, int]] = []
    for candidate in range(lower, soft_upper + 1):
        char = text[indices[candidate - 1]]
        next_char = text[indices[candidate]] if candidate < len(indices) else ""
        duration = max(0.001, times[candidate - 1] - times[start])
        length = candidate - start
        remaining = end - candidate
        cps = length / duration
        score = abs(length - min(max_chars, 18)) * 0.08
        score += abs(duration - min(target_duration, 2.8)) * 1.4
        if candidate > upper:
            score += (candidate - upper) * 0.9
        trailing = boundary_trailing_text(text, indices, candidate)
        has_strong = char in strong or any(item in strong for item in trailing)
        has_weak = char in weak or any(item in weak for item in trailing)
        if has_strong:
            score -= 14.0
            if candidate == end and duration > effective_target_duration and internal_punctuation_available and end - start >= 18:
                score += 32.0
        elif has_weak:
            score -= 10.0
            later_punctuation = any(
                boundary_has_punctuation(text, indices, later, strong + weak)
                for later in range(candidate + min_chars, end + 1)
            )
            if later_punctuation:
                score += length * 0.40
        elif candidate == end:
            score -= 5.0
        elif punctuation_available:
            score += 6.0
        if char in DANGLING_PREVIOUS_CHARS or next_char in DANGLING_NEXT_CHARS:
            score += 2.5
        if char + next_char in bad_split_pairs:
            score += 20.0
        if word_boundaries is not None and candidate not in word_boundaries:
            score += 4.0
        if is_ascii_word_char(char) and is_ascii_word_char(next_char):
            score += 30.0
        if next_char in "》）】”’":
            score += 5.0
        if duration > max_duration:
            score += (duration - max_duration) * 8.0
        if 0 < remaining < min_chars:
            score += (min_chars - remaining) * 6.0
        if cps > 7.5:
            score += (cps - 7.5) * 0.8
        candidates.append((score, candidate))
    return min(candidates)[1] if candidates else upper


def validated_llm_breaks(
    breaks: list[int],
    text: str,
    indices: list[int],
    start: int,
    end: int,
    times: list[float],
    *,
    min_chars: int,
    max_chars: int,
    max_duration: float,
    target_duration: float,
    word_boundaries: set[int] | None,
) -> list[int]:
    valid: list[int] = []
    previous = start
    minimum_duration = 0.85
    punctuation = "，。！？、：；,.!?;:"
    effective_target_duration = min(target_duration, 2.8)
    for boundary in sorted(set(breaks)):
        if boundary <= previous or boundary >= end:
            continue
        length = boundary - previous
        duration = max(0.001, times[boundary - 1] - times[previous])
        char = text[indices[boundary - 1]]
        next_char = text[indices[boundary]] if boundary < len(indices) else ""
        has_boundary_punctuation = boundary_has_punctuation(text, indices, boundary, punctuation)
        if length < min_chars:
            continue
        if char in DANGLING_PREVIOUS_CHARS or next_char in DANGLING_NEXT_CHARS:
            continue
        if duration < minimum_duration:
            continue
        if duration > effective_target_duration and any(
            boundary_has_punctuation(text, indices, candidate, punctuation)
            for candidate in range(previous + min_chars, boundary)
        ):
            continue
        prior_punctuation = any(
            boundary_has_punctuation(text, indices, candidate, punctuation)
            for candidate in range(previous + min_chars, boundary)
        )
        if has_boundary_punctuation and prior_punctuation and duration > effective_target_duration:
            continue
        if not has_boundary_punctuation and prior_punctuation:
            continue
        if char not in punctuation and not has_boundary_punctuation and length < max(18, min_chars):
            continue
        if length > max_chars + 4:
            continue
        if duration > max_duration + 0.8:
            continue
        if word_boundaries is not None and boundary not in word_boundaries:
            continue
        valid.append(boundary)
        previous = boundary
    if valid and end - valid[-1] < min_chars:
        valid.pop()
    return valid


def ollama_break_provider(base_url: str, model: str, timeout: float) -> LLMBreakProvider:
    endpoint = base_url.rstrip("/") + "/api/generate"

    def provide(text: str, indices: list[int], times: list[float], start: int, end: int, max_chars: int, max_duration: float) -> list[int]:
        span_text = original_span(text, indices, start, end)
        duration = max(0.001, times[end - 1] - times[start])
        prompt = (
            "你是中文影视字幕断句器。只输出严格 JSON，不要解释。\n"
            "任务：给下面文本选择字幕断点。断点使用 normalized_char_offset，表示在第几个有效中文/字母/数字字符之后断开。\n"
            "硬规则：不要改字，不要删字；不要切开中文词、专名、书名号；避免字幕尾部留下的/了/着/其/将/最/但/和/与/在/为等悬空字。\n"
            "断句原则：按逻辑信息块断句，不要机械遵循语法完整句。来源、引导、转折、背景铺垫等不强相关信息可以单独成条，让后面的核心信息更容易理解。\n"
            "快语速旁白要更细拆；遇到多个逗号时，优先把每个相对独立意群拆开，而不是贪长合并。\n"
            f"字幕约束：每条尽量 12-{max_chars} 字，最长约 {max_duration:.1f} 秒；全文估算时长 {duration:.2f} 秒。\n"
            f"文本：{span_text}\n"
            "输出格式：{\"breaks\":[整数数组]}"
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 160},
        }
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode())
        body = str(raw.get("response", "")).strip()
        match = re.search(r"\{.*\}", body, re.S)
        if not match:
            return []
        parsed = json.loads(match.group(0))
        offsets = [int(item) for item in parsed.get("breaks", [])]
        return [start + offset for offset in offsets]

    return provide


def choose_breaks(
    text: str,
    indices: list[int],
    times: list[float],
    min_chars: int,
    max_chars: int,
    max_duration: float,
    *,
    target_duration: float = 4.2,
    llm_break_provider: LLMBreakProvider | None = None,
) -> list[dict[str, Any]]:
    """Deterministically select readable subtitle cues without changing text."""
    cues: list[dict[str, Any]] = []
    word_boundaries = optional_jieba_boundaries([text[index] for index in indices])
    effective_target_duration = min(target_duration, 2.8)
    for unit_start, unit_end in semantic_units(text, indices):
        llm_boundaries: list[int] = []
        if llm_break_provider and unit_end - unit_start > max_chars:
            try:
                llm_boundaries = validated_llm_breaks(
                    llm_break_provider(text, indices, times, unit_start, unit_end, max_chars, max_duration),
                    text,
                    indices,
                    unit_start,
                    unit_end,
                    times,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    max_duration=max_duration,
                    target_duration=target_duration,
                    word_boundaries=word_boundaries,
                )
            except Exception:
                llm_boundaries = []
        start = unit_start
        while start < unit_end:
            next_llm = next((boundary for boundary in llm_boundaries if boundary > start), None)
            if next_llm is not None:
                end = next_llm
            else:
                remaining = unit_end - start
                duration = times[unit_end - 1] - times[start]
                has_internal_punctuation = any(
                    boundary_has_punctuation(text, indices, candidate, "，、：；,.!?;")
                    for candidate in range(start + min_chars, unit_end)
                )
                if remaining <= max_chars and duration <= max_duration and (remaining < 18 or duration <= effective_target_duration or not has_internal_punctuation):
                    end = unit_end
                else:
                    end = choose_unit_break(
                        text,
                        indices,
                        times,
                        start,
                        unit_end,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        max_duration=max_duration,
                        target_duration=target_duration,
                        word_boundaries=word_boundaries,
                    )
            if end <= start:
                end = unit_end
            cues.append(cue_from_span(text, indices, times, cues, start, end))
            start = end
    return cues


def finalize(args: argparse.Namespace) -> int:
    asr = json.loads(args.nuc_asr.read_text(encoding="utf-8"))
    gemini_text = args.gemini.read_text(encoding="utf-8").strip()
    template, indices = normalized_with_indices(gemini_text)
    nuc, nuc_times = nuc_character_times(asr)
    if not template or not nuc:
        raise RuntimeError("Gemini text and NUC word timestamps are both required")
    times, matched = aligned_template_times(template, nuc, nuc_times)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backfill = build_backfill_cues(asr, gemini_text, indices, times)
    llm_provider = None
    backend = "gemini-text+NUC-word-times+deterministic-segmentation"
    if args.llm_segmentation:
        llm_provider = ollama_break_provider(args.ollama_url, args.ollama_model, args.ollama_timeout)
        backend = f"gemini-text+NUC-word-times+ollama-{args.ollama_model}+validated-segmentation"
    final = choose_breaks(
        gemini_text,
        indices,
        times,
        args.min_chars,
        args.max_chars,
        args.max_duration,
        target_duration=args.target_duration,
        llm_break_provider=llm_provider,
    )
    reconstructed, _ = normalized_with_indices("".join(cue["text"] for cue in final))
    if reconstructed != template:
        raise RuntimeError("Deterministic segmentation failed to preserve Gemini text")
    differences = build_differences(nuc, template, times)
    write_srt(args.output_dir / "gemini-backfilled-local-timeline.srt", backfill)
    write_srt(args.output_dir / "final.srt", final)
    (args.output_dir / "final-timeline.json").write_text(json.dumps({"backend": backend, "cues": final}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "transcript-differences.json").write_text(json.dumps({"difference_groups": differences}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"gemini_characters": len(template), "nuc_characters": len(nuc), "matched_characters": matched, "difference_groups": len(differences), "backfilled_cues": len(backfill), "final_cues": len(final), "llm_segmentation": bool(args.llm_segmentation), "ollama_model": args.ollama_model if args.llm_segmentation else None}
    (args.output_dir / "final-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_probe_hard_subs(args: argparse.Namespace, cookie_args: list[str]) -> int:
    from evaluate_burned_subtitle_ocr import subtitle_text

    output = args.output_dir
    clips, frames = output / "clips", output / "frames"
    output.mkdir(parents=True, exist_ok=True)
    clips.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    remote_source = is_url(args.source)
    if args.timestamps:
        timestamps = [float(item) for item in args.timestamps.split(",")]
    elif remote_source:
        duration = float(subprocess.check_output(
            [YT_DLP, "--no-playlist", "--print", "duration", *cookie_args, args.source], text=True
        ).strip())
        timestamps = [max(0.0, duration * ratio - args.window / 2) for ratio in (0.01, 0.10, 0.30, 0.50, 0.70, 0.90, 0.98)]
    else:
        local_source = Path(args.source).expanduser().resolve()
        if not local_source.is_file():
            raise RuntimeError(f"Local video not found: {local_source}")
        duration = float(subprocess.check_output(
            ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(local_source)],
            text=True,
        ).strip())
        timestamps = [max(0.0, duration * ratio - args.window / 2) for ratio in (0.01, 0.10, 0.30, 0.50, 0.70, 0.90, 0.98)]
    for number, timestamp in enumerate(timestamps, 1):
        frame_pattern = frames / f"sample-{number:02d}-%02d.jpg"
        if remote_source:
            target = clips / f"sample-{number:02d}.mp4"
            if not target.exists():
                run([YT_DLP, "--no-playlist", "-f", "bv*[height<=360]+ba/b[height<=360]", "--download-sections", f"*{timestamp:.3f}-{timestamp + args.window:.3f}", *cookie_args, "--merge-output-format", "mp4", "-o", str(target), args.source])
            run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(target), "-vf", "fps=1", "-q:v", "3", str(frame_pattern)])
        else:
            run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}", "-t", f"{args.window:.3f}", "-i", str(local_source), "-vf", "fps=1", "-q:v", "3", str(frame_pattern)])
    binary = output / "apple-vision-ocr"
    if not binary.exists():
        module_cache = output / "swift-module-cache"
        module_cache.mkdir(exist_ok=True)
        run(["swiftc", "-O", "-Xcc", f"-fmodules-cache-path={module_cache}", str(PROJECT_ROOT / "scripts/apple_vision_ocr.swift"), "-o", str(binary)])
    raw = output / "apple-vision-raw.jsonl"
    run([str(binary), "--frames", str(frames), "--output", str(raw), "--fps", "1"])
    observations = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
    errors = [item for item in observations if item.get("error")]
    if errors and len(errors) / max(1, len(observations)) > 0.20:
        raise RuntimeError(
            f"Apple Vision failed on {len(errors)}/{len(observations)} probe frames; "
            "hard subtitles were not classified"
        )
    detected = [item for item in observations if subtitle_text(item)[0]]
    decision = {"source": args.source, "sample_timestamps": timestamps, "frames_processed": len(observations), "vision_errors": len(errors), "frames_with_candidate_subtitles": len(detected), "burned_subtitles_present": len(detected) >= args.minimum_hits, "minimum_hits": args.minimum_hits}
    (output / "hard-subtitle-probe.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def probe_hard_subs(args: argparse.Namespace) -> int:
    with yt_dlp_cookie_session(
        enabled=args.cookies_from_chrome,
        chrome_profile=args.chrome_profile,
        forwarded_browser_spec=args.cookies_browser_spec,
    ) as cookie_session:
        return run_probe_hard_subs(args, cookie_session.arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    finalize_parser = subcommands.add_parser("finalize", help="Backfill NUC SRT text and deterministically re-segment it")
    finalize_parser.add_argument("--nuc-asr", type=Path, required=True)
    finalize_parser.add_argument("--gemini", type=Path, required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--min-chars", type=int, default=10)
    finalize_parser.add_argument("--max-chars", type=int, default=30)
    finalize_parser.add_argument("--max-duration", type=float, default=6.5)
    finalize_parser.add_argument("--target-duration", type=float, default=4.2)
    finalize_parser.add_argument("--llm-segmentation", action="store_true", help="Use Ollama to propose semantic subtitle breaks; local validation still owns timestamps")
    finalize_parser.add_argument("--ollama-url", default="http://192.168.31.196:11434")
    finalize_parser.add_argument("--ollama-model", default="qwen2.5:7b")
    finalize_parser.add_argument("--ollama-timeout", type=float, default=90.0)
    finalize_parser.set_defaults(handler=finalize)
    probe_parser = subcommands.add_parser("probe-hard-subs", help="Download only short low-res samples and OCR them")
    probe_parser.add_argument("source")
    probe_parser.add_argument("--output-dir", type=Path, required=True)
    probe_parser.add_argument("--timestamps", default="", help="comma-separated seconds; defaults to seven positions distributed over the duration")
    probe_parser.add_argument("--window", type=float, default=3.0)
    probe_parser.add_argument("--minimum-hits", type=int, default=3)
    probe_parser.add_argument("--cookies-from-chrome", action="store_true")
    probe_parser.add_argument("--chrome-profile", default="Default")
    probe_parser.add_argument(
        "--cookies-browser-spec",
        default=None,
        help=argparse.SUPPRESS,
    )
    probe_parser.set_defaults(handler=probe_hard_subs)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
