"""External backend adapters (OmniVAD shadow, Gemini transcription, etc.)."""

from __future__ import annotations

import difflib
import hashlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata
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
GEMINI_PROMPT_VERSION = "transcribe-lines-v2"
GEMINI_FUSION_VERSION = "whisper-baseline-gemini-correction-v1"
GEMINI_INLINE_MAX_BYTES = 20 * 1024 * 1024


@dataclass
class GeminiTranscribeResult:
    status: str  # "completed" | "failed" | "skipped"
    text: str
    lines: list[str] = field(default_factory=list)
    model: str = ""
    elapsed: float = 0.0
    warning: str = ""
    diagnostics: dict = field(default_factory=dict)


def gemini_transcribe_audio(
    audio_path: Path,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str = GEMINI_TRANSCRIBE_PROMPT,
    timeout: int = 180,
    max_tokens: int = 8192,
    upload_timeout: int = 300,
    processing_timeout: int = 600,
) -> GeminiTranscribeResult:
    """Send audio to Gemini for text-only transcription (no timestamps)."""
    if not api_key.strip():
        return GeminiTranscribeResult("skipped", "", warning="Gemini API key not configured")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return GeminiTranscribeResult("failed", "", warning="google-genai not installed")

    uploaded_file = None
    file_client = None
    diagnostics: dict = {"transport": "inline", "cleanup": "not-needed"}
    try:
        file_size = audio_path.stat().st_size
        use_file_api = file_size > GEMINI_INLINE_MAX_BYTES
        output_limit = max(max_tokens, 60000) if use_file_api else max_tokens
        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=max(30, timeout) * 1000),
        )
        if use_file_api:
            diagnostics["transport"] = "file-api"
            file_client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(
                    timeout=max(30, upload_timeout) * 1000
                ),
            )
            upload_started = time.monotonic()
            uploaded_file = file_client.files.upload(
                file=str(audio_path),
                config=genai_types.UploadFileConfig(
                    mime_type=mimetypes.guess_type(audio_path.name)[0] or "audio/wav",
                ),
            )
            diagnostics["file_name"] = str(getattr(uploaded_file, "name", ""))
            diagnostics["upload_elapsed"] = time.monotonic() - upload_started
            deadline = time.monotonic() + max(1, processing_timeout)
            while True:
                state = str(getattr(uploaded_file, "state", "")).upper()
                if state.endswith("ACTIVE"):
                    break
                if state.endswith("FAILED"):
                    raise RuntimeError("Gemini File API processing failed")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Gemini File API processing timed out")
                time.sleep(1.0)
                uploaded_file = file_client.files.get(name=uploaded_file.name)
            contents = [uploaded_file, prompt]
        else:
            contents = [{
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mimetypes.guess_type(audio_path.name)[0] or "audio/wav",
                            "data": audio_path.read_bytes(),
                        }
                    },
                    {"text": prompt},
                ],
            }]
        t0 = time.time()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=output_limit,
            ),
        )
        elapsed = time.time() - t0
        diagnostics["generation_elapsed"] = elapsed
        text = (response.text or "").strip()
        finish_reason = ""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "")).upper()
        usage = getattr(response, "usage_metadata", None)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        diagnostics.update(
            finish_reason=finish_reason,
            output_tokens=output_tokens,
            max_output_tokens=output_limit,
        )
        if not text:
            raise RuntimeError("Gemini returned an empty transcription")
        if "MAX_TOKENS" in finish_reason or output_tokens >= int(output_limit * 0.95):
            raise RuntimeError("Gemini transcription appears truncated at the output token limit")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Gemini returned no usable transcription lines")
        return GeminiTranscribeResult(
            "completed",
            text,
            lines=lines,
            model=model,
            elapsed=elapsed,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        return GeminiTranscribeResult(
            "failed",
            "",
            model=model,
            warning=str(exc)[:300],
            diagnostics=diagnostics,
        )
    finally:
        if uploaded_file is not None:
            try:
                (file_client or client).files.delete(name=uploaded_file.name)
                diagnostics["cleanup"] = "deleted"
            except Exception as exc:
                diagnostics["cleanup"] = f"failed: {str(exc)[:120]}"


# ---------------------------------------------------------------------------
# Gemini + Whisper word-timestamp fusion
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    status: str
    segments: list[SubtitleSegment]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _normalize_alignment_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    output: list[str] = []
    pending_space = False
    for char in normalized:
        if char.isalnum() or "\u3400" <= char <= "\u9fff":
            if pending_space and output:
                output.append(" ")
            output.append(char)
            pending_space = False
        elif char.isspace():
            pending_space = True
    return "".join(output).strip()


def _fallback_segments_for_window(
    whisper_segments: list[SubtitleSegment],
    whisper_words: list[SubtitleWord],
    first_word: int,
    last_word: int,
) -> list[SubtitleSegment]:
    start = whisper_words[first_word].start
    end = whisper_words[last_word].end
    overlapping = [
        segment
        for segment in whisper_segments
        if segment.end > start and segment.start < end
    ]
    if overlapping:
        return overlapping
    text = " ".join(word.text for word in whisper_words[first_word:last_word + 1]).strip()
    return [SubtitleSegment(start, max(end, start + 0.001), text)] if text else []


def _merge_adjacent_duplicate_segments(
    segments: list[SubtitleSegment],
    *,
    maximum_gap: float = 5.0,
    minimum_text_length: int = 20,
) -> tuple[list[SubtitleSegment], int]:
    merged: list[SubtitleSegment] = []
    merge_count = 0
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        text = " ".join(segment.text.split()).strip()
        if not text:
            continue
        if (
            merged
            and len(_normalize_alignment_text(text)) >= minimum_text_length
            and _normalize_alignment_text(merged[-1].text)
            == _normalize_alignment_text(text)
            and segment.start - merged[-1].end <= maximum_gap
        ):
            previous = merged[-1]
            merged[-1] = SubtitleSegment(
                previous.start,
                max(previous.end, segment.end),
                previous.text,
            )
            merge_count += 1
            continue
        merged.append(SubtitleSegment(segment.start, segment.end, text))
    return merged, merge_count


def fuse_gemini_with_whisper(
    gemini_lines: list[str],
    whisper_words: list[SubtitleWord],
    *,
    whisper_segments: Optional[list[SubtitleSegment]] = None,
    min_confidence: float = 0.72,
) -> FusionResult:
    """Apply only high-confidence Gemini corrections to a Whisper baseline."""
    lines = [line.strip() for line in gemini_lines if line.strip()]
    fallback = sorted(whisper_segments or [], key=lambda item: (item.start, item.end))
    if not whisper_words:
        return FusionResult(
            "blocked",
            fallback,
            0.0,
            ["Whisper word timestamps are unavailable"],
            {"algorithm_version": GEMINI_FUSION_VERSION},
        )
    if not lines:
        return FusionResult(
            "blocked",
            fallback,
            0.0,
            ["Gemini returned no usable lines"],
            {"algorithm_version": GEMINI_FUSION_VERSION},
        )

    whisper_chars: list[str] = []
    whisper_char_to_word: list[int | None] = []
    for word_index, word in enumerate(whisper_words):
        token = _normalize_alignment_text(word.text)
        if not token:
            continue
        if whisper_chars:
            whisper_chars.append(" ")
            whisper_char_to_word.append(None)
        whisper_chars.extend(token)
        whisper_char_to_word.extend([word_index] * len(token))
    whisper_full = "".join(whisper_chars)

    gemini_chars: list[str] = []
    line_spans: list[tuple[int, int, str]] = []
    for line in lines:
        normalized_line = _normalize_alignment_text(line)
        if gemini_chars:
            gemini_chars.append(" ")
        start = len(gemini_chars)
        gemini_chars.extend(normalized_line)
        line_spans.append((start, len(gemini_chars), line))
    gemini_full = "".join(gemini_chars)
    if not whisper_full or not gemini_full:
        return FusionResult(
            "blocked",
            fallback,
            0.0,
            ["Normalized transcript is empty"],
            {"algorithm_version": GEMINI_FUSION_VERSION},
        )

    char_to_word: list[int | None] = [None] * len(gemini_full)
    matcher = difflib.SequenceMatcher(a=whisper_full, b=gemini_full, autojunk=False)
    matched_chars = 0
    for whisper_pos, gemini_pos, size in matcher.get_matching_blocks():
        for offset in range(size):
            word_index = whisper_char_to_word[whisper_pos + offset]
            if word_index is not None:
                char_to_word[gemini_pos + offset] = word_index
                matched_chars += 1
    significant_chars = sum(not char.isspace() for char in gemini_full)
    global_confidence = matched_chars / max(1, significant_chars)
    diagnostics: dict = {
        "algorithm_version": GEMINI_FUSION_VERSION,
        "global_confidence": global_confidence,
        "line_candidates": [],
        "adjacent_duplicate_merges": 0,
        "accepted_corrections": 0,
        "preserved_whisper_segments": 0,
    }
    if global_confidence < min_confidence:
        diagnostics["gemini_candidates"] = lines
        return FusionResult(
            "blocked",
            fallback or [
                SubtitleSegment(
                    whisper_words[0].start,
                    whisper_words[-1].end,
                    " ".join(word.text for word in whisper_words).strip(),
                )
            ],
            global_confidence,
            ["Gemini/Whisper global alignment confidence is too low"],
            diagnostics,
        )

    corrections: list[SubtitleSegment] = []
    warnings: list[str] = []
    last_word_index = -1
    for line_index, (start, end, original_line) in enumerate(line_spans):
        indices = [
            index
            for index in (char_to_word[position] for position in range(start, end))
            if index is not None and index >= last_word_index
        ]
        unique_indices = sorted(set(indices))
        line_chars = max(1, sum(not char.isspace() for char in gemini_full[start:end]))
        line_confidence = len(indices) / line_chars
        candidate = {
            "line_index": line_index,
            "text": original_line,
            "confidence": line_confidence,
            "matched_word_indices": unique_indices,
        }
        diagnostics["line_candidates"].append(candidate)
        if not unique_indices:
            warnings.append(f"Line {line_index + 1} had no reliable Whisper match")
            continue

        first_word = unique_indices[0]
        last_word = unique_indices[-1]
        proposed_start = whisper_words[first_word].start
        proposed_end = whisper_words[last_word].end
        candidate["accepted"] = line_confidence >= min_confidence
        if line_confidence >= min_confidence:
            corrections.append(
                SubtitleSegment(proposed_start, proposed_end, original_line)
            )
            last_word_index = last_word
        else:
            warnings.append(f"Line {line_index + 1} kept Whisper text")

    if fallback:
        output: list[SubtitleSegment] = []
        fallback_index = 0
        correction_index = 0
        while fallback_index < len(fallback) and correction_index < len(corrections):
            current = fallback[fallback_index]
            correction = corrections[correction_index]
            if current.end <= correction.start:
                output.append(current)
                fallback_index += 1
            elif current.start >= correction.end:
                output.append(correction)
                correction_index += 1
            else:
                while (
                    fallback_index < len(fallback)
                    and fallback[fallback_index].start < correction.end
                ):
                    fallback_index += 1
                output.append(correction)
                correction_index += 1
        output.extend(fallback[fallback_index:])
        output.extend(corrections[correction_index:])
    else:
        output = corrections
    diagnostics["accepted_corrections"] = len(corrections)
    diagnostics["preserved_whisper_segments"] = sum(
        segment in fallback for segment in output
    )
    output, duplicate_merge_count = _merge_adjacent_duplicate_segments(output)
    diagnostics["adjacent_duplicate_merges"] = duplicate_merge_count
    normalized: list[SubtitleSegment] = []
    for segment in output:
        start = max(segment.start, normalized[-1].end if normalized else 0.0)
        end = max(start + 0.001, segment.end)
        if segment.text.strip():
            normalized.append(SubtitleSegment(start, end, segment.text.strip()))
    return FusionResult(
        "completed" if corrections else "blocked",
        normalized,
        global_confidence,
        warnings,
        diagnostics,
    )


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


def gemini_fusion_cache_key(
    audio_path: Path,
    gemini_model: str,
    whisper_model: str,
) -> str:
    digest = hashlib.sha256()
    stat = audio_path.stat()
    digest.update(str(audio_path.resolve()).encode())
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    digest.update(gemini_model.encode())
    digest.update(whisper_model.encode())
    digest.update(GEMINI_PROMPT_VERSION.encode())
    digest.update(GEMINI_FUSION_VERSION.encode())
    return digest.hexdigest()[:24]
