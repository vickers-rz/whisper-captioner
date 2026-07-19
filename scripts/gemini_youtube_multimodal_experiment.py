#!/usr/bin/env python3
"""Compare Gemini direct-YouTube understanding with an audio transcript baseline."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


STRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_summary": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "rhetorical_function": {"type": "string"},
                },
                "required": ["start_seconds", "end_seconds", "title", "summary"],
            },
        },
        "visual_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "event_type": {"type": "string"},
                    "description": {"type": "string"},
                    "ocr_text": {"type": "string"},
                    "analytical_value": {"type": "string"},
                },
                "required": ["start_seconds", "end_seconds", "event_type", "description"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "claim": {"type": "string"},
                    "claim_type": {"type": "string"},
                    "evidence_modalities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "visual_support": {"type": "string"},
                    "requires_external_verification": {"type": "boolean"},
                },
                "required": [
                    "start_seconds",
                    "end_seconds",
                    "claim",
                    "claim_type",
                    "evidence_modalities",
                    "requires_external_verification",
                ],
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "first_seen_seconds": {"type": "number"},
                    "source_modalities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "entity_type", "first_seen_seconds", "source_modalities"],
            },
        },
    },
    "required": ["video_summary", "chapters", "visual_events", "claims", "entities"],
}


TRANSCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "duration_seconds": {"type": "number"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "visually_resolved_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["start_seconds", "end_seconds", "speaker", "text"],
            },
        },
    },
    "required": ["language", "duration_seconds", "segments"],
}


VISUAL_FOCUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp_seconds": {"type": "number"},
                    "event_type": {"type": "string"},
                    "exact_on_screen_text": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "visual_description": {"type": "string"},
                    "relation_to_audio": {"type": "string"},
                    "confidence": {"type": "string"},
                },
                "required": [
                    "timestamp_seconds",
                    "event_type",
                    "exact_on_screen_text",
                    "visual_description",
                    "relation_to_audio",
                    "confidence",
                ],
            },
        },
        "visual_only_information": {
            "type": "array",
            "items": {"type": "string"},
        },
        "audio_claims_visually_supported": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["observations", "visual_only_information", "audio_claims_visually_supported"],
}


STRUCTURE_PROMPT = """
你是视频证据提取器。完整观看并聆听此公开视频。不要依赖视频标题、简介或章节列表来猜测内容；
只有实际音频或画面支持的信息才能输出。请用简体中文返回符合 JSON schema 的结果。

任务：
1. 按真实论述转折划分章节，记录秒级起止时间、摘要和修辞功能。
2. 提取有分析价值的视觉事件，包括片头标题、屏幕文字、人物标签、书名、数据、图表、新闻截图、引文和来源。
3. 提取主要事实主张、因果主张、预测和价值判断，标明证据来自 audio、visual 或 audio+visual。
4. 提取人名、组织、著作、历史事件、年份等实体，说明来自音频还是画面。
5. 不要把主持人的主张写成已核实事实；需要外部核查的项目必须标记。
6. 时间戳必须来自视频内容。看不清的 OCR 留空，不得补写。
""".strip()


TRANSCRIPT_PROMPT = """
请完整聆听视频并结合画面，输出连续、尽量逐字的简体中文语义转写。
返回符合 JSON schema 的结果，每个 segment 建议 15 至 35 秒，不能省略重复、过渡句或结尾。
start_seconds 与 end_seconds 必须单调递增并覆盖整段有人声内容；speaker 无法判断时写“旁白”。
画面明确帮助纠正了人名、书名、机构、年份或术语时，把该词写入 visually_resolved_terms；否则为空数组。
不要依据标题或简介补写没有听到的内容，不要总结，不要改写观点。
""".strip()


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _extract_text(value: Any) -> str:
    data = _dump_model(value)
    texts: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(data.get("steps", data) if isinstance(data, dict) else data)
    return "\n".join(texts).strip()


def run_interaction(
    *, api_key: str, model: str, video_url: str, prompt: str, schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    started = time.monotonic()
    response = client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": video_url, "resolution": "low"},
            {"type": "text", "text": prompt},
        ],
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 65536,
            "thinking_level": "low",
        },
        response_format={"type": "text", "mime_type": "application/json", "schema": schema},
        response_modalities=["text"],
        store=False,
        timeout=1200.0,
    )
    raw = _dump_model(response)
    text = _extract_text(raw)
    if not text:
        raise RuntimeError(f"Gemini returned no text (status={raw.get('status', 'unknown')})")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}: {text[:500]}") from exc
    raw["experiment_elapsed_seconds"] = round(time.monotonic() - started, 3)
    return parsed, raw


def run_transcript_chunk(
    *,
    api_key: str,
    model: str,
    video_url: str,
    start_seconds: float,
    end_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from google import genai
    from google.genai import types

    prompt = (
        f"{TRANSCRIPT_PROMPT}\n\n"
        f"本次只处理原视频 {start_seconds:.3f} 秒到 {end_seconds:.3f} 秒。"
        "所有 start_seconds/end_seconds 必须使用原视频的绝对秒数，不得从零重新计时。"
    )
    client = genai.Client(api_key=api_key)
    started = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part(
                file_data=types.FileData(file_uri=video_url),
                video_metadata=types.VideoMetadata(
                    start_offset=f"{start_seconds:.3f}s",
                    end_offset=f"{end_seconds:.3f}s",
                ),
                media_resolution=types.PartMediaResolution(
                    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
                ),
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=32768,
            response_mime_type="application/json",
            response_json_schema=TRANSCRIPT_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = _dump_model(response)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned no transcript text for chunk")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid chunk JSON: {exc}: {text[:500]}") from exc
    raw["experiment_elapsed_seconds"] = round(time.monotonic() - started, 3)
    raw["requested_start_seconds"] = start_seconds
    raw["requested_end_seconds"] = end_seconds
    return parsed, raw


def run_visual_focus(
    *,
    api_key: str,
    model: str,
    video_url: str,
    start_seconds: float,
    end_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from google import genai
    from google.genai import types

    prompt = f"""
只检查原视频 {start_seconds:.3f} 秒至 {end_seconds:.3f} 秒的画面，并用原视频绝对秒数定位。
逐项记录标题卡、广告、二维码、网址、优惠码、人物标签、数据、图表和引用。
合并连续重复画面，只保留最多 16 个具有分析价值的场景，不要逐秒罗列。
exact_on_screen_text 只能逐字记录真正看清的画面文字；不清楚时留空，不能用音频补齐。
relation_to_audio 只能写 visual_only、supports_audio、contradicts_audio 或 illustrative。
audio_claims_visually_supported 仅收录画面本身提供证据的主张，装饰性素材不能算证据。
返回符合 JSON schema 的简体中文结果。
""".strip()
    client = genai.Client(api_key=api_key)
    started = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part(
                file_data=types.FileData(file_uri=video_url),
                video_metadata=types.VideoMetadata(
                    start_offset=f"{start_seconds:.3f}s",
                    end_offset=f"{end_seconds:.3f}s",
                ),
                media_resolution=types.PartMediaResolution(
                    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
                ),
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_json_schema=VISUAL_FOCUS_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = _dump_model(response)
    parsed = json.loads((response.text or "").strip())
    raw["experiment_elapsed_seconds"] = round(time.monotonic() - started, 3)
    raw["requested_start_seconds"] = start_seconds
    raw["requested_end_seconds"] = end_seconds
    return parsed, raw


def chunked_transcript(
    *,
    api_key: str,
    model: str,
    video_url: str,
    duration_seconds: float,
    chunk_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    raw_chunks: list[dict[str, Any]] = []
    start = 0.0
    chunk_index = 0
    while start < duration_seconds:
        end = min(duration_seconds, start + chunk_seconds)
        chunk_index += 1
        print(
            f"  chunk {chunk_index}: {start:.1f}-{end:.1f}s",
            flush=True,
        )
        parsed, raw = run_transcript_chunk(
            api_key=api_key,
            model=model,
            video_url=video_url,
            start_seconds=start,
            end_seconds=end,
        )
        chunk_segments = parsed.get("segments", [])
        if start > 0 and chunk_segments:
            maximum_end = max(float(item.get("end_seconds", 0)) for item in chunk_segments)
            if maximum_end <= chunk_seconds + 30:
                for item in chunk_segments:
                    item["start_seconds"] = float(item.get("start_seconds", 0)) + start
                    item["end_seconds"] = float(item.get("end_seconds", 0)) + start
                raw["timestamps_shifted_from_clip_relative"] = True
        for item in chunk_segments:
            item["chunk_index"] = chunk_index
        segments.extend(chunk_segments)
        raw_chunks.append(raw)
        start = end
    segments.sort(key=lambda item: (float(item.get("start_seconds", 0)), float(item.get("end_seconds", 0))))
    return {
        "language": "zh-CN",
        "duration_seconds": duration_seconds,
        "segments": segments,
    }, raw_chunks


def normalize(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\u3400-\u9fff]+", text.lower()))


def matched_character_stats(baseline: str, candidate: str) -> dict[str, Any]:
    baseline_norm = normalize(baseline)
    candidate_norm = normalize(candidate)
    matcher = difflib.SequenceMatcher(None, baseline_norm, candidate_norm, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return {
        "baseline_normalized_characters": len(baseline_norm),
        "candidate_normalized_characters": len(candidate_norm),
        "matched_characters": matched,
        "baseline_coverage": round(matched / max(1, len(baseline_norm)), 6),
        "candidate_precision": round(matched / max(1, len(candidate_norm)), 6),
        "sequence_similarity": round(matcher.ratio(), 6),
        "length_ratio": round(len(candidate_norm) / max(1, len(baseline_norm)), 6),
    }


def validate_timeline(items: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_ranges = 0
    backwards = 0
    gaps: list[float] = []
    previous_end: float | None = None
    for item in items:
        start = float(item.get("start_seconds", -1))
        end = float(item.get("end_seconds", -1))
        if start < 0 or end <= start:
            invalid_ranges += 1
        if previous_end is not None:
            if start < previous_end:
                backwards += 1
            elif start > previous_end:
                gaps.append(start - previous_end)
        previous_end = max(previous_end or 0.0, end)
    return {
        "items": len(items),
        "invalid_ranges": invalid_ranges,
        "overlaps_or_backwards": backwards,
        "maximum_gap_seconds": round(max(gaps, default=0.0), 3),
        "gaps_over_10_seconds": sum(gap > 10 for gap in gaps),
        "last_end_seconds": round(previous_end or 0.0, 3),
    }


def timed_alignment_report(
    transcript: dict[str, Any], timed_baseline_path: Path, chunk_seconds: float
) -> dict[str, Any]:
    baseline_data = json.loads(timed_baseline_path.read_text(encoding="utf-8"))
    baseline_chars: list[str] = []
    baseline_times: list[float] = []
    for segment in baseline_data.get("segments", []):
        text = normalize(str(segment.get("text", "")))
        if not text:
            continue
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        for index, char in enumerate(text):
            baseline_chars.append(char)
            baseline_times.append(start + (end - start) * (index + 0.5) / len(text))

    candidate_chars: list[str] = []
    candidate_times: list[float] = []
    candidate_chunks: list[int] = []
    for segment in transcript.get("segments", []):
        text = normalize(str(segment.get("text", "")))
        if not text:
            continue
        start = float(segment.get("start_seconds", 0))
        end = float(segment.get("end_seconds", start))
        chunk_index = int(segment.get("chunk_index", 0))
        for index, char in enumerate(text):
            candidate_chars.append(char)
            candidate_times.append(start + (end - start) * (index + 0.5) / len(text))
            candidate_chunks.append(chunk_index)

    matcher = difflib.SequenceMatcher(
        None, "".join(baseline_chars), "".join(candidate_chars), autojunk=False
    )
    candidate_to_baseline: dict[int, int] = {}
    matched_baseline: set[int] = set()
    for baseline_start, candidate_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            baseline_index = baseline_start + offset
            candidate_index = candidate_start + offset
            candidate_to_baseline[candidate_index] = baseline_index
            matched_baseline.add(baseline_index)

    errors = [
        candidate_times[candidate_index] - baseline_times[baseline_index]
        for candidate_index, baseline_index in candidate_to_baseline.items()
    ]
    absolute_errors = [abs(error) for error in errors]
    chunk_reports: list[dict[str, Any]] = []
    chunk_count = max(candidate_chunks, default=0)
    for chunk_index in range(1, chunk_count + 1):
        candidate_indices = [
            index for index, value in enumerate(candidate_chunks) if value == chunk_index
        ]
        baseline_indices = [
            candidate_to_baseline[index]
            for index in candidate_indices
            if index in candidate_to_baseline
        ]
        chunk_errors = [
            candidate_times[index] - baseline_times[candidate_to_baseline[index]]
            for index in candidate_indices
            if index in candidate_to_baseline
        ]
        chunk_reports.append(
            {
                "chunk_index": chunk_index,
                "requested_start_seconds": (chunk_index - 1) * chunk_seconds,
                "requested_end_seconds": min(
                    float(transcript.get("duration_seconds", 0)), chunk_index * chunk_seconds
                ),
                "candidate_characters": len(candidate_indices),
                "matched_characters": len(baseline_indices),
                "matched_source_start_seconds": round(
                    min((baseline_times[index] for index in baseline_indices), default=0), 3
                ),
                "matched_source_end_seconds": round(
                    max((baseline_times[index] for index in baseline_indices), default=0), 3
                ),
                "median_signed_timestamp_error_seconds": round(
                    statistics.median(chunk_errors) if chunk_errors else 0, 3
                ),
                "median_absolute_timestamp_error_seconds": round(
                    statistics.median(map(abs, chunk_errors)) if chunk_errors else 0, 3
                ),
            }
        )

    window_reports: list[dict[str, Any]] = []
    duration = float(transcript.get("duration_seconds", 0))
    window_start = 0.0
    while window_start < duration:
        window_end = min(duration, window_start + chunk_seconds)
        indices = [
            index
            for index, timestamp in enumerate(baseline_times)
            if window_start <= timestamp < window_end
        ]
        matched = sum(index in matched_baseline for index in indices)
        window_reports.append(
            {
                "start_seconds": window_start,
                "end_seconds": window_end,
                "baseline_characters": len(indices),
                "matched_characters": matched,
                "baseline_character_coverage": round(matched / max(1, len(indices)), 6),
            }
        )
        window_start = window_end

    ordered_absolute_errors = sorted(absolute_errors)
    p90_index = max(0, int(len(ordered_absolute_errors) * 0.9) - 1)
    return {
        "baseline_path": str(timed_baseline_path),
        "matched_characters": len(candidate_to_baseline),
        "median_absolute_timestamp_error_seconds": round(
            statistics.median(absolute_errors) if absolute_errors else 0, 3
        ),
        "p90_absolute_timestamp_error_seconds": round(
            ordered_absolute_errors[p90_index] if ordered_absolute_errors else 0, 3
        ),
        "maximum_absolute_timestamp_error_seconds": round(
            max(absolute_errors, default=0), 3
        ),
        "chunk_alignment": chunk_reports,
        "source_window_coverage": window_reports,
    }


def compare_results(
    structure: dict[str, Any],
    transcript: dict[str, Any],
    baseline_text: str,
    timed_baseline_path: Path | None = None,
    chunk_seconds: float = 480.0,
) -> dict[str, Any]:
    segments = transcript.get("segments", [])
    candidate_text = "\n".join(str(item.get("text", "")) for item in segments)
    visual_events = structure.get("visual_events", [])
    resolved_terms = sorted(
        {
            str(term).strip()
            for segment in segments
            for term in segment.get("visually_resolved_terms", [])
            if str(term).strip()
        }
    )
    visual_claims = [
        claim
        for claim in structure.get("claims", [])
        if any("visual" in str(modality).lower() for modality in claim.get("evidence_modalities", []))
    ]
    report = {
        "text_comparison": matched_character_stats(baseline_text, candidate_text),
        "transcript_timeline": validate_timeline(segments),
        "chapter_timeline": validate_timeline(structure.get("chapters", [])),
        "counts": {
            "chapters": len(structure.get("chapters", [])),
            "visual_events": len(visual_events),
            "visual_events_with_ocr": sum(bool(str(event.get("ocr_text", "")).strip()) for event in visual_events),
            "claims": len(structure.get("claims", [])),
            "claims_with_visual_evidence": len(visual_claims),
            "entities": len(structure.get("entities", [])),
            "visually_resolved_terms": len(resolved_terms),
        },
        "visually_resolved_terms": resolved_terms,
        "visual_evidence_claims": visual_claims,
    }
    if timed_baseline_path is not None:
        report["timed_alignment"] = timed_alignment_report(
            transcript, timed_baseline_path, chunk_seconds
        )
    return report


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=480.0)
    parser.add_argument("--force-structure", action="store_true")
    parser.add_argument("--reuse-transcript", action="store_true")
    parser.add_argument("--timed-baseline", type=Path)
    parser.add_argument("--visual-focus", help="Optional start:end seconds for a high-resolution visual pass")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.baseline.is_file():
        print(f"Baseline transcript does not exist: {args.baseline}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "video_url": args.url,
        "model": args.model,
        "media_resolution": "low",
        "source_mode": "gemini_youtube_url",
        "captions_used": "unknown_not_relied_upon",
        "word_timestamps": False,
    }

    structure_path = args.output_dir / "gemini-youtube-structure.json"
    if structure_path.exists() and not args.force_structure:
        print("[1/2] Reusing multimodal structure", flush=True)
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
    else:
        print("[1/2] Extracting multimodal structure", flush=True)
        structure, structure_raw = run_interaction(
            api_key=api_key,
            model=args.model,
            video_url=args.url,
            prompt=STRUCTURE_PROMPT,
            schema=STRUCTURE_SCHEMA,
        )
        write_json(structure_path, {**metadata, **structure})
        write_json(args.output_dir / "gemini-youtube-structure-raw.json", structure_raw)

    transcript_path = args.output_dir / "gemini-youtube-transcript.json"
    if args.reuse_transcript and transcript_path.exists():
        print("[2/2] Reusing chunked timestamped transcript", flush=True)
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    else:
        print("[2/2] Extracting chunked timestamped transcript", flush=True)
        transcript, transcript_raw = chunked_transcript(
            api_key=api_key,
            model=args.model,
            video_url=args.url,
            duration_seconds=args.duration,
            chunk_seconds=args.chunk_seconds,
        )
        write_json(transcript_path, {**metadata, **transcript})
        write_json(args.output_dir / "gemini-youtube-transcript-raw.json", transcript_raw)
    transcript_text = "\n".join(segment.get("text", "") for segment in transcript.get("segments", []))
    (args.output_dir / "gemini-youtube-transcript.txt").write_text(transcript_text, encoding="utf-8")

    baseline_text = args.baseline.read_text(encoding="utf-8")
    comparison = {
        **metadata,
        **compare_results(
            structure,
            transcript,
            baseline_text,
            timed_baseline_path=args.timed_baseline,
            chunk_seconds=args.chunk_seconds,
        ),
    }
    write_json(args.output_dir / "gemini-youtube-ab-comparison.json", comparison)
    if args.visual_focus:
        focus_start, focus_end = (float(value) for value in args.visual_focus.split(":", 1))
        print(f"[extra] High-resolution visual focus {focus_start:.1f}-{focus_end:.1f}s", flush=True)
        visual_focus, visual_focus_raw = run_visual_focus(
            api_key=api_key,
            model=args.model,
            video_url=args.url,
            start_seconds=focus_start,
            end_seconds=focus_end,
        )
        write_json(
            args.output_dir / "gemini-youtube-visual-focus.json",
            {**metadata, "media_resolution": "high", **visual_focus},
        )
        write_json(args.output_dir / "gemini-youtube-visual-focus-raw.json", visual_focus_raw)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
