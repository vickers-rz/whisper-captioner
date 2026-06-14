"""External backend adapters (OmniVAD shadow, Gemini transcription, etc.)."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import ASRResult, SpeechRegion, SubtitleSegment, SubtitleWord

# ---------------------------------------------------------------------------
# OmniVAD shadow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OmniVADShadowResult:
    status: str
    regions: list[SpeechRegion]
    warning: str = ""


def run_omnivad_shadow(audio_path: Path, output_dir: Path) -> OmniVADShadowResult:
    command_template = os.environ.get(
        "WHISPER_CAPTIONER_OMNIVAD_COMMAND",
        f"{Path(sys.executable).with_name('omnivad')} "
        "{audio} -m vad -f json --chunk 600 --workers 2 -o {output}",
    ).strip()
    executable = shlex.split(command_template)[0]
    if not shutil.which(executable):
        return OmniVADShadowResult("unavailable", [], f"OmniVAD executable unavailable: {executable}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "omnivad-shadow.json"
    command = [
        part.format(audio=str(audio_path), output=str(result_path), output_dir=str(output_dir))
        for part in shlex.split(command_template)
    ]
    try:
        subprocess.run(command, check=True, timeout=300, capture_output=True, text=True)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        items = payload.get("tiers", {}).get("VAD", payload.get("regions", []))
        regions = [
            SpeechRegion(
                float(item["start"]),
                float(item["end"]),
                float(item["confidence"]) if item.get("confidence") is not None else None,
                "omnivad",
            )
            for item in items
            if float(item["end"]) > float(item["start"])
        ]
        return OmniVADShadowResult("completed", regions)
    except Exception as exc:
        return OmniVADShadowResult("failed", [], str(exc))


# ---------------------------------------------------------------------------
# Gemini audio transcription
# ---------------------------------------------------------------------------

GEMINI_TRANSCRIBE_PROMPT = (
    "Transcribe this audio. Output ONLY the transcription text. "
    "Break the text into natural sentences or logical segments, one per line. "
    "Do NOT add timestamps, speaker labels, or any formatting. "
    "Do NOT add any commentary before or after the transcription. "
    "Output plain text only — each sentence on its own line."
)


@dataclass
class GeminiTranscribeResult:
    status: str  # "completed" | "failed" | "skipped"
    text: str
    lines: list[str] = field(default_factory=list)
    model: str = ""
    elapsed: float = 0.0
    warning: str = ""


def gemini_transcribe_audio(
    audio_path: Path,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str = GEMINI_TRANSCRIBE_PROMPT,
    timeout: int = 180,
    max_tokens: int = 8192,
) -> GeminiTranscribeResult:
    """Send audio to Gemini for text-only transcription (no timestamps)."""
    if not api_key.strip():
        return GeminiTranscribeResult("skipped", "", warning="Gemini API key not configured")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return GeminiTranscribeResult("failed", "", warning="google-genai not installed")

    try:
        audio_bytes = audio_path.read_bytes()
        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=max(30, timeout) * 1000),
        )
        t0 = time.time()
        response = client.models.generate_content(
            model=model,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}},
                    {"text": prompt},
                ],
            }],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
            ),
        )
        elapsed = time.time() - t0
        text = (response.text or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return GeminiTranscribeResult(
            "completed", text, lines=lines, model=model, elapsed=elapsed,
        )
    except Exception as exc:
        return GeminiTranscribeResult("failed", "", warning=str(exc)[:200])


# ---------------------------------------------------------------------------
# Gemini + Whisper word-timestamp fusion
# ---------------------------------------------------------------------------


def fuse_gemini_with_whisper(
    gemini_lines: list[str],
    whisper_words: list[SubtitleWord],
) -> list[SubtitleSegment]:
    """Align Gemini's sentence-level text with Whisper's word timestamps.

    Uses difflib.SequenceMatcher to find matching character spans between
    Gemini's output and Whisper's full text, then maps each Gemini sentence
    to the corresponding Whisper word time range.

    Timing policy:
    - Gemini provides the text (accurate, well-punctuated).
    - Whisper provides the time axis (word-level precision).
    - Gemini-only text that has no Whisper match gets proportional allocation.
    """
    if not whisper_words:
        # No word timestamps — assign evenly across 0 to N seconds
        total = max(1.0, float(len(gemini_lines)))
        return [
            SubtitleSegment(i / total * 180, (i + 1) / total * 180, line)
            for i, line in enumerate(gemini_lines)
        ]

    # Build Whisper full text with word index spans
    word_spans: list[tuple[int, int, float, float, int]] = []
    pos = 0
    for i, w in enumerate(whisper_words):
        token = w.text
        word_spans.append((pos, pos + len(token), w.start, w.end, i))
        pos += len(token) + 1  # +1 for joining space

    whisper_full = " ".join(w.text for w in whisper_words)
    gemini_full = "\n".join(gemini_lines)

    # Align characters
    matcher = difflib.SequenceMatcher(
        a=whisper_full.lower(), b=gemini_full.lower()
    )
    matches = matcher.get_matching_blocks()

    gemini_char_to_word: list[int | None] = [None] * len(gemini_full)
    for a, b, size in matches:
        if size == 0:
            continue
        for offset in range(size):
            g_pos = b + offset
            w_pos = a + offset
            for ws, we, _, _, wi in word_spans:
                if ws <= w_pos < we:
                    gemini_char_to_word[g_pos] = wi
                    break

    # Build segments
    fused: list[SubtitleSegment] = []
    for line in gemini_lines:
        idx = gemini_full.index(line)
        g_start = idx
        g_end = idx + len(line)

        word_indices: set[int] = set()
        for g_pos in range(g_start, min(g_end, len(gemini_char_to_word))):
            wi = gemini_char_to_word[g_pos]
            if wi is not None:
                word_indices.add(wi)

        if word_indices:
            t_start = whisper_words[min(word_indices)].start
            t_end = whisper_words[max(word_indices)].end
        else:
            # No matching words — use nearest adjacent times
            t_start = 0.0
            for g_pos in range(g_start, -1, -1):
                if g_pos < len(gemini_char_to_word) and gemini_char_to_word[g_pos] is not None:
                    t_start = whisper_words[gemini_char_to_word[g_pos]].end
                    break
            t_end = t_start + 1.0
            for g_pos in range(g_end, len(gemini_char_to_word)):
                if gemini_char_to_word[g_pos] is not None:
                    t_end = whisper_words[gemini_char_to_word[g_pos]].start
                    break

        fused.append(SubtitleSegment(
            max(0.0, t_start),
            max(t_start + 0.4, t_end),
            line,
        ))

    # Normalize timeline (monotonic, non-overlapping, min duration)
    fused.sort(key=lambda seg: (seg.start, seg.end))
    normalized: list[SubtitleSegment] = []
    for i, seg in enumerate(fused):
        text = seg.text.strip()
        if not text:
            continue
        start = seg.start
        next_start = fused[i + 1].start if i + 1 < len(fused) else seg.end + 1.0
        end = max(seg.end, start + 0.4)
        end = min(end, next_start)
        if normalized and start < normalized[-1].end:
            start = normalized[-1].end
        if end <= start:
            end = start + 0.001
        normalized.append(SubtitleSegment(start, end, text))

    return normalized


def _fusion_confidence(
    gemini_line: str,
    matched_word_count: int,
    total_words_in_span: int,
    matched_chars: int,
    total_chars: int,
) -> float:
    """Estimate alignment confidence for a fused segment (0.0–1.0).

    High confidence = most Gemini chars matched to Whisper words.
    Low confidence = Gemini text has little overlap with Whisper text.
    """
    if total_chars == 0:
        return 1.0
    char_ratio = matched_chars / total_chars
    if total_words_in_span == 0:
        return min(char_ratio, 0.3)  # no Whisper match at all
    word_ratio = matched_word_count / max(1, total_words_in_span)
    return 0.4 * char_ratio + 0.6 * word_ratio  # word match weighted higher


def refine_timing_with_qwen(
    gemini_sentence: str,
    whisper_words_nearby: list[SubtitleWord],
    proposed_start: float,
    proposed_end: float,
    ollama_host: str = "192.168.31.196",
    ollama_port: str = "11434",
    timeout: int = 30,
) -> tuple[float, float] | None:
    """Use NUC Qwen3.5-4B to refine sentence-to-timeline alignment.

    Qwen's role (timing arbiter only):
    - Gemini text is TRUSTED and MUST NOT be modified.
    - Only output refined start_ms and end_ms based on the Whisper word timeline.
    """
    if not whisper_words_nearby:
        return None

    # Build a word timeline reference for Qwen
    timeline_lines = []
    for w in whisper_words_nearby:
        timeline_lines.append(f"  [{w.start:.2f}s-{w.end:.2f}s] {w.text}")
    timeline = "\n".join(timeline_lines)

    prompt = (
        "你是一个时间轴仲裁者。以下是 Whisper 提供的带时间戳的词级转写：\n\n"
        f"{timeline}\n\n"
        "以下是 Gemini 提供的目标句子（文本内容不可修改）：\n\n"
        f"「{gemini_sentence}」\n\n"
        "你的任务：在 Whisper 时间轴上找到最能覆盖这句 Gemini 句子的起止时间。\n"
        "严格要求：\n"
        "1. 只输出两个数字：start_ms end_ms（毫秒，整数）\n"
        "2. 不要输出任何其他文字、标点或解释\n"
        "3. Gemini 文本不可修改，你只负责定位时间轴\n"
        "4. 如果找不到匹配，输出两个 0\n\n"
        f"当前建议时间范围：{proposed_start*1000:.0f}ms-{proposed_end*1000:.0f}ms\n"
    )

    try:
        import urllib.request
        payload = json.dumps({
            "model": "qwen3.5:4b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0.0},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://{ollama_host}:{ollama_port}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "").strip()

        # Parse "start_ms end_ms"
        parts = content.replace(",", " ").split()
        if len(parts) >= 2:
            start_ms = int(parts[0])
            end_ms = int(parts[1])
            if start_ms >= 0 and end_ms > start_ms:
                return (start_ms / 1000.0, end_ms / 1000.0)
        return None
    except Exception:
        return None


@dataclass
class FusedSegment:
    start: float
    end: float
    text: str
    confidence: float = 1.0


def fuse_gemini_with_whisper_arbitrated(
    gemini_lines: list[str],
    whisper_words: list[SubtitleWord],
    *,
    ollama_host: str = "192.168.31.196",
    ollama_port: str = "11434",
    min_confidence: float = 0.4,
) -> list[SubtitleSegment]:
    """Fuse Gemini text with Whisper timestamps, with Qwen arbiter for low-confidence alignments.

    Policy:
    - Gemini text is the SOLE source of truth for content — never modified.
    - Whisper word timestamps are the time axis.
    - difflib provides initial alignment + confidence score.
    - Low-confidence segments get Qwen timing refinement.
    - Qwen only adjusts start_ms/end_ms, never text.
    """
    if not whisper_words:
        total = max(1.0, float(len(gemini_lines)))
        return [
            SubtitleSegment(i / total * 180, (i + 1) / total * 180, line)
            for i, line in enumerate(gemini_lines)
        ]

    # Build Whisper full text and word index spans
    word_spans: list[tuple[int, int, float, float, int]] = []
    pos = 0
    for i, w in enumerate(whisper_words):
        token = w.text
        word_spans.append((pos, pos + len(token), w.start, w.end, i))
        pos += len(token) + 1

    whisper_full = " ".join(w.text for w in whisper_words)
    gemini_full = "\n".join(gemini_lines)

    # difflib character alignment
    matcher = difflib.SequenceMatcher(
        a=whisper_full.lower(), b=gemini_full.lower()
    )
    matches = matcher.get_matching_blocks()

    gemini_char_to_word: list[int | None] = [None] * len(gemini_full)
    for a, b, size in matches:
        if size == 0:
            continue
        for offset in range(size):
            g_pos = b + offset
            w_pos = a + offset
            for ws, we, _, _, wi in word_spans:
                if ws <= w_pos < we:
                    gemini_char_to_word[g_pos] = wi
                    break

    # Build fused segments with confidence
    fused: list[FusedSegment] = []
    for line in gemini_lines:
        idx = gemini_full.index(line)
        g_start = idx
        g_end = idx + len(line)

        word_indices: set[int] = set()
        matched_chars = 0
        for g_pos in range(g_start, min(g_end, len(gemini_char_to_word))):
            wi = gemini_char_to_word[g_pos]
            if wi is not None:
                word_indices.add(wi)
                matched_chars += 1
        total_chars = g_end - g_start

        if word_indices:
            t_start = whisper_words[min(word_indices)].start
            t_end = whisper_words[max(word_indices)].end
            total_words = max(word_indices) - min(word_indices) + 1
        else:
            t_start = 0.0
            for g_pos in range(g_start, -1, -1):
                if g_pos < len(gemini_char_to_word) and gemini_char_to_word[g_pos] is not None:
                    t_start = whisper_words[gemini_char_to_word[g_pos]].end
                    break
            t_end = t_start + 1.0
            for g_pos in range(g_end, len(gemini_char_to_word)):
                if gemini_char_to_word[g_pos] is not None:
                    t_end = whisper_words[gemini_char_to_word[g_pos]].start
                    break
            total_words = 0

        conf = _fusion_confidence(line, len(word_indices), total_words, matched_chars, total_chars)
        fused.append(FusedSegment(
            start=max(0.0, t_start),
            end=max(t_start + 0.4, t_end),
            text=line,
            confidence=conf,
        ))

    # Refine low-confidence segments with Qwen arbiter
    low_conf = [s for s in fused if s.confidence < min_confidence]
    if low_conf and whisper_words:
        for seg in low_conf:
            mid = (seg.start + seg.end) / 2
            window = 15.0
            nearby = [
                w for w in whisper_words
                if seg.start - window <= w.start <= seg.end + window
            ]
            refined = refine_timing_with_qwen(
                seg.text, nearby, seg.start, seg.end,
                ollama_host=ollama_host, ollama_port=ollama_port,
            )
            if refined:
                seg.start, seg.end = refined
                seg.confidence = min(1.0, seg.confidence + 0.3)  # boost after refinement

    # Normalize timeline
    normalized: list[SubtitleSegment] = []
    for i, seg in enumerate(fused):
        text = seg.text.strip()
        if not text:
            continue
        start = seg.start
        next_start = fused[i + 1].start if i + 1 < len(fused) else seg.end + 1.0
        end = max(seg.end, start + 0.4)
        end = min(end, next_start)
        if normalized and start < normalized[-1].end:
            start = normalized[-1].end
        if end <= start:
            end = start + 0.001
        normalized.append(SubtitleSegment(start, end, text))

    return normalized


def gemini_fusion_cache_key(audio_path: Path, gemini_model: str, whisper_model: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(audio_path).encode())
    digest.update(gemini_model.encode())
    digest.update(whisper_model.encode())
    digest.update(b"gemini-whisper-fusion-v2")
    return digest.hexdigest()[:24]
