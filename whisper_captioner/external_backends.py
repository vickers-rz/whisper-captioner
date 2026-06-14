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


def gemini_fusion_cache_key(audio_path: Path, gemini_model: str, whisper_model: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(audio_path).encode())
    digest.update(gemini_model.encode())
    digest.update(whisper_model.encode())
    digest.update(b"gemini-whisper-fusion-v1")
    return digest.hexdigest()[:24]
