from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .models import (
    ASRResult,
    QualityReport,
    RetryRegion,
    SpeechRegion,
    SubtitleSegment,
    SubtitleWord,
)


HARD_BOUNDARY_RE = re.compile(r"[。！？.!?][”’\"']?$")
SOFT_BOUNDARY_RE = re.compile(r"[，、；,;:：][”’\"']?$")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


@dataclass(frozen=True)
class CueBuilderConfig:
    max_cjk_chars: int = 22
    max_latin_chars: int = 42
    max_duration: float = 5.0
    min_duration: float = 0.4
    pause_boundary: float = 0.8
    cjk_target_cps: float = 12.0
    latin_target_cps: float = 17.0
    soft_boundary_min_chars: int = 8


def parse_verbose_asr_response(data: dict, *, requested_words: bool = True) -> ASRResult:
    segments: list[SubtitleSegment] = []
    words: list[SubtitleWord] = []
    for segment_data in data.get("segments") or []:
        text = str(segment_data.get("text", "")).strip()
        if text:
            segments.append(
                SubtitleSegment(
                    float(segment_data["start"]),
                    float(segment_data["end"]),
                    text,
                )
            )
        for word_data in segment_data.get("words") or []:
            word = _word_from_upstream(word_data)
            if word is not None:
                words.append(word)
    if not words:
        for word_data in data.get("words") or []:
            word = _word_from_upstream(word_data)
            if word is not None:
                words.append(word)
    warnings: list[str] = []
    if requested_words and not words:
        warnings.append("NUC ASR did not return word timestamps; using segment fallback")
    diagnostics = {
        "capability_warnings": warnings,
        "word_timestamp_source": (
            "segment.words"
            if any(segment.get("words") for segment in data.get("segments") or [])
            else "top-level words" if words else "unavailable"
        ),
        "language_probability": data.get("language_probability"),
        "nuc_result_dir": data.get("nuc_result_dir"),
        "upstream_response": data,
    }
    return ASRResult(
        language=str(data.get("language") or ""),
        words=words,
        segments=segments,
        diagnostics=diagnostics,
    )


def _word_from_upstream(data: object) -> SubtitleWord | None:
    if not isinstance(data, dict):
        return None
    text = str(data.get("word", data.get("text", "")))
    try:
        start = float(data["start"])
        end = float(data["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not text.strip() or end <= start:
        return None
    probability = data.get("probability", data.get("prob"))
    return SubtitleWord(
        start=start,
        end=end,
        text=text,
        probability=float(probability) if probability is not None else None,
    )


def _compact_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _is_cjk(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return bool(compact) and len(CJK_RE.findall(compact)) >= max(1, len(compact) // 3)


def _join_words(words: list[SubtitleWord]) -> str:
    text = ""
    for word in words:
        token = word.text
        if not token:
            continue
        if not text:
            text = token.strip()
        elif _is_cjk(text + token) or re.match(r"^[，。！？、；：,.!?;:)]", token):
            text += token.strip()
        else:
            text += " " + token.strip()
    return re.sub(r"\s+", " ", text).strip()


def build_cues(
    words: list[SubtitleWord],
    fallback_segments: list[SubtitleSegment],
    config: CueBuilderConfig | None = None,
) -> tuple[list[SubtitleSegment], list[str]]:
    config = config or CueBuilderConfig()
    if not words:
        return normalize_timeline(fallback_segments, config.min_duration), [
            "segment timestamp fallback: word timestamps unavailable"
        ]

    groups: list[list[SubtitleWord]] = []
    current: list[SubtitleWord] = []
    for word in sorted(words, key=lambda item: (item.start, item.end)):
        if word.end <= word.start or not word.text.strip():
            continue
        if current:
            current_text = _join_words(current)
            projected_text = _join_words([*current, word])
            projected_duration = word.end - current[0].start
            limit = config.max_cjk_chars if _is_cjk(projected_text) else config.max_latin_chars
            target_cps = config.cjk_target_cps if _is_cjk(projected_text) else config.latin_target_cps
            gap = word.start - current[-1].end
            should_break = (
                gap > config.pause_boundary
                or HARD_BOUNDARY_RE.search(current_text) is not None
                or _compact_length(projected_text) > limit
                or projected_duration > config.max_duration
                or (
                    projected_duration > 0
                    and _compact_length(projected_text) / projected_duration > target_cps * 1.5
                    and _compact_length(current_text) >= config.soft_boundary_min_chars
                )
                or (
                    SOFT_BOUNDARY_RE.search(current_text) is not None
                    and _compact_length(current_text) >= config.soft_boundary_min_chars
                )
            )
            if should_break:
                groups.append(current)
                current = []
        current.append(word)
        if HARD_BOUNDARY_RE.search(_join_words(current)):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    cues = [
        SubtitleSegment(group[0].start, group[-1].end, _join_words(group))
        for group in groups
        if group and _join_words(group)
    ]
    return normalize_timeline(cues, config.min_duration), []


def normalize_timeline(
    segments: Iterable[SubtitleSegment],
    min_duration: float = 0.4,
) -> list[SubtitleSegment]:
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    normalized: list[SubtitleSegment] = []
    for index, segment in enumerate(ordered):
        text = segment.text.strip()
        if not text:
            continue
        start = max(0.0, segment.start)
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else None
        end = max(segment.end, start + min_duration)
        if next_start is not None:
            end = min(end, max(start, next_start))
        if normalized and start < normalized[-1].end:
            start = normalized[-1].end
        if end <= start:
            end = start + 0.001
        normalized.append(SubtitleSegment(start, end, text))
    return normalized


def parse_silencedetect_regions(
    output: str,
    duration: float,
    *,
    source: str = "ffmpeg",
    minimum_voice: float = 0.10,
) -> list[SpeechRegion]:
    starts = re.compile(r"silence_start:\s*([0-9.]+)")
    ends = re.compile(r"silence_end:\s*([0-9.]+)")
    silences: list[tuple[float, float]] = []
    active: float | None = None
    for line in output.splitlines():
        start_match = starts.search(line)
        if start_match:
            active = float(start_match.group(1))
        end_match = ends.search(line)
        if end_match:
            silences.append((active if active is not None else 0.0, float(end_match.group(1))))
            active = None
    if active is not None:
        silences.append((active, duration))
    regions: list[SpeechRegion] = []
    cursor = 0.0
    for start, end in sorted(silences):
        if start - cursor >= minimum_voice:
            regions.append(SpeechRegion(cursor, start, source=source))
        cursor = max(cursor, end)
    if duration - cursor >= minimum_voice:
        regions.append(SpeechRegion(cursor, duration, source=source))
    return regions


def merge_retry_regions(
    regions: Iterable[RetryRegion],
    *,
    guard: float = 2.0,
    merge_gap: float = 0.5,
    duration: float | None = None,
) -> list[RetryRegion]:
    expanded = sorted(
        (
            RetryRegion(
                max(0.0, region.start - guard),
                min(duration, region.end + guard) if duration is not None else region.end + guard,
                region.reason,
                region.attempts,
            )
            for region in regions
        ),
        key=lambda item: item.start,
    )
    merged: list[RetryRegion] = []
    for region in expanded:
        if merged and region.start - merged[-1].end < merge_gap:
            previous = merged[-1]
            previous.end = max(previous.end, region.end)
            previous.reason = ", ".join(dict.fromkeys([*previous.reason.split(", "), region.reason]))
            previous.attempts = max(previous.attempts, region.attempts)
        else:
            merged.append(region)
    return merged


def replace_segments_in_regions(
    original: list[SubtitleSegment],
    replacement: list[SubtitleSegment],
    regions: list[RetryRegion],
) -> list[SubtitleSegment]:
    def intersects(segment: SubtitleSegment, region: RetryRegion) -> bool:
        return segment.end > region.start and segment.start < region.end

    inserted = [
        segment
        for segment in replacement
        if any(intersects(segment, region) for region in regions)
    ]
    kept = [
        segment
        for segment in original
        if not any(intersects(segment, region) for region in regions)
        and not any(
            segment.end <= region.start
            and segment.end > replacement_segment.start
            and segment.start < replacement_segment.end
            for region in regions
            for replacement_segment in inserted
        )
    ]
    candidates = sorted([*kept, *inserted], key=lambda item: (item.start, -item.end))
    deduplicated: list[SubtitleSegment] = []
    seen: set[tuple[int, str]] = set()
    for segment in candidates:
        normalized_text = re.sub(r"[\W_]+", "", segment.text, flags=re.UNICODE).lower()
        key = (round(segment.start * 5), normalized_text)
        if normalized_text and key not in seen:
            deduplicated.append(segment)
            seen.add(key)
    return normalize_timeline(deduplicated, min_duration=0.001)


def _interval_overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def audit_asr_result(
    result: ASRResult,
    speech_regions: list[SpeechRegion],
    *,
    duration: float | None = None,
) -> QualityReport:
    suspicious: list[RetryRegion] = []
    uncovered: list[RetryRegion] = []
    warnings = list(result.diagnostics.get("capability_warnings", []))
    explained = [
        SubtitleWord(segment.start, segment.end, segment.text)
        for segment in result.segments
    ]
    speech_duration = sum(max(0.0, region.end - region.start) for region in speech_regions)
    covered_duration = 0.0
    for region in speech_regions:
        overlaps = [
            _interval_overlap(region.start, region.end, item.start, item.end)
            for item in explained
            if item.text.strip()
        ]
        covered = min(region.end - region.start, sum(overlaps))
        covered_duration += covered
        if region.end - region.start - covered > 0.6:
            uncovered.append(RetryRegion(region.start, region.end, "unexplained speech"))

    segments = normalize_timeline(result.segments, min_duration=0.001)
    for segment in segments:
        span = segment.end - segment.start
        density = _compact_length(segment.text) / span if span > 0 else 0.0
        if span >= 6.0 and density < 1.5:
            suspicious.append(RetryRegion(segment.start, segment.end, "low text density"))
        cps_limit = 24.0 if _is_cjk(segment.text) else 30.0
        if span > 0 and _compact_length(segment.text) / span > cps_limit:
            suspicious.append(RetryRegion(segment.start, segment.end, "abnormal cps"))
        if span > 5.25:
            suspicious.append(RetryRegion(segment.start, segment.end, "long cue"))

    for previous, current in zip(result.segments, result.segments[1:]):
        if current.start < previous.start:
            suspicious.append(RetryRegion(current.start, current.end, "reverse timestamp"))
        elif current.start < previous.end:
            suspicious.append(RetryRegion(current.start, previous.end, "overlapping timestamp"))

    compact_texts = [re.sub(r"\s+", "", segment.text) for segment in result.segments if segment.text.strip()]
    if len(compact_texts) >= 8:
        most_repeated = max((compact_texts.count(text) for text in set(compact_texts)), default=0)
        if most_repeated >= 8 and most_repeated / len(compact_texts) >= 0.3:
            suspicious.append(RetryRegion(0.0, duration or result.segments[-1].end, "repetition hallucination"))

    if result.words and result.segments:
        word_start, word_end = result.words[0].start, result.words[-1].end
        segment_start, segment_end = result.segments[0].start, result.segments[-1].end
        if abs(word_start - segment_start) > 1.0 or abs(word_end - segment_end) > 1.0:
            suspicious.append(
                RetryRegion(min(word_start, segment_start), max(word_end, segment_end), "word/segment span mismatch")
            )
        low_confidence_words = [
            word
            for word in result.words
            if word.probability is not None and word.probability < 0.35
        ]
        if low_confidence_words:
            suspicious.extend(
                RetryRegion(word.start, word.end, "low word confidence")
                for word in low_confidence_words
            )

    coverage = 1.0 if speech_duration <= 0 else min(1.0, covered_duration / speech_duration)
    merged_uncovered = merge_retry_regions(uncovered, guard=0.0, duration=duration)
    merged_suspicious = merge_retry_regions(suspicious, guard=0.0, duration=duration)
    if merged_uncovered or merged_suspicious:
        status = "incomplete_speech_coverage"
    elif warnings:
        status = "passed_with_warnings"
    else:
        status = "passed"
    return QualityReport(
        status=status,
        speech_coverage=coverage,
        uncovered_regions=merged_uncovered,
        suspicious_regions=merged_suspicious,
        warnings=warnings,
        diagnostics={
            "word_timestamps": bool(result.words),
            "speech_region_count": len(speech_regions),
            "segment_fallback": not bool(result.words),
        },
    )


def quality_report_to_dict(report: QualityReport) -> dict:
    return asdict(report)


@dataclass
class LanguagePin:
    explicit_language: str = "auto"
    threshold: float = 0.65
    maximum_detection_speech: float = 30.0
    language: str = ""
    confidence: float | None = None
    observed_speech: float = 0.0

    @property
    def request_language(self) -> str:
        if self.explicit_language.lower() not in {"", "auto"}:
            return self.explicit_language
        return self.language or "auto"

    def observe(self, language: str, confidence: float | None, speech_seconds: float) -> None:
        if self.explicit_language.lower() not in {"", "auto"} or self.language:
            return
        self.observed_speech += max(0.0, speech_seconds)
        effective_confidence = confidence
        if effective_confidence is None and speech_seconds >= 6.0:
            effective_confidence = 1.0
        if language and (
            (effective_confidence is not None and effective_confidence >= self.threshold)
            or self.observed_speech >= self.maximum_detection_speech
        ):
            self.language = language
            self.confidence = confidence
