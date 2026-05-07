"""Worker classes for async caption processing and transcription tasks.

This module provides worker classes that run in separate threads:
- RealtimeWorker: Stream audio directly to Whisper for real-time captions
- QueueWorker: Process a queue of media files (URLs or local files)
- RollingPrefetchWorker: Download, chunk, and incrementally transcribe with LLM polish
"""

from __future__ import annotations

import json
import re
import shlex
import signal
import subprocess
import tempfile
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from whisper_captioner.cache import cache_slug, canonical_media_url
from whisper_captioner.config import (
    BUFFER_PAUSE_MARGIN,
    BUFFER_RESUME_MARGIN,
    CACHE_DIR,
    DEFAULT_SUBTITLE_OFFSET,
    FFMPEG,
    FFPROBE,
    MLX_AUDIO_STT,
    MLX_WHISPER,
    MLX_TERMS_SCRIPT,
    MODELS_DIR,
    OUTPUT_DIR,
    RAPIDMLX_8B_MODEL,
    RAPIDMLX_8B_PORT,
    RAPIDMLX_8B_SERVED_MODEL,
    RAPIDMLX_BIN,
    RAPIDMLX_HOST,
    RAPIDMLX_MODEL,
    RAPIDMLX_PORT,
    RAPIDMLX_PYTHON,
    RAPIDMLX_SERVED_MODEL,
    SENSE_VOICE_CPP_MAIN,
    SENSE_VOICE_CHUNK_OVERLAP_SECONDS,
    SUBTITLE_PIPELINE_VERSION,
    WHISPER_CLI,
    WHISPER_STREAM,
    YT_DLP,
)
from whisper_captioner.llm_handler import (
    llm_fuse_with_reference,
    llm_generate_text,
    llm_proofread,
    llm_provider_ready,
)
from whisper_captioner.models import CaptionMode, LLMProvider, SubtitleSegment
from whisper_captioner.subtitle_io import (
    load_segments,
    overlapping_segments,
    parse_srt,
    parse_sense_voice_output,
    parse_subtitle_file,
    save_segments,
    save_segments_as_srt,
    save_segments_as_txt,
    segment_from_dict,
    segment_to_dict,
)


TIMESTAMP_RE = re.compile(r"^\[[^\]]+\]\s*(.*)$")
PROGRESS_LINE_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)%")


def clean_title_for_filename(title: str, fallback: str = "item") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", title).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:120].strip() or fallback


def append_label_to_basename(path: Path, label: str) -> Path:
    return path.with_name(f"{path.name}-{label}")


def source_output_dir(base_dir: Path, title: str) -> Path:
    directory = base_dir / clean_title_for_filename(title)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _terminate_process(proc: Optional[subprocess.Popen[str]], timeout: float = 3.0) -> None:
    if not proc or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_temp_paths(paths: list[Path]) -> None:
    for path in reversed(paths):
        _safe_unlink(path)
    paths.clear()


def qwen3_asr_mode(mode: CaptionMode) -> bool:
    return mode.key in {"qwen3_asr_06b_4bit_mlx", "qwen3_asr_17b_8bit_mlx"}


def qwen3_event_label(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[\(\[（【][^)\]）】]{1,24}[\)\]）】]", stripped))


def pseudo_timestamp_qwen3_text(text: str, duration: float) -> list[SubtitleSegment]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。？！!?；;……])\s*", clean) if part.strip()]
    if len(parts) == 1 or any(len(part) > 48 for part in parts):
        expanded: list[str] = []
        for part in parts:
            if len(part) > 48:
                expanded.extend(
                    sub_part.strip()
                    for sub_part in re.split(r"(?<=[，、,])\s*", part)
                    if sub_part.strip()
                )
            else:
                expanded.append(part)
        parts = expanded
    if not parts:
        return [SubtitleSegment(0.0, duration, clean)]
    char_weights = [
        max(1, len(re.sub(r"[，。！？!?；;、,（）()【】\\[\\]\\s]", "", part)))
        for part in parts
    ]
    total_weight = sum(char_weights)
    estimated_durations = [duration * (weight / total_weight) for weight in char_weights]
    refined_parts: list[str] = []
    refined_weights: list[int] = []
    for index, part in enumerate(parts):
        if not qwen3_event_label(part) and len(part) > 48 and estimated_durations[index] > 8.0:
            sub_parts = [
                sub_part.strip()
                for sub_part in re.split(r"(?<=[，、,])\s*", part)
                if sub_part.strip()
            ]
            if len(sub_parts) > 1:
                refined_parts.extend(sub_parts)
                refined_weights.extend(
                    max(1, len(re.sub(r"[，。！？!?；;、,（）()【】\\[\\]\\s]", "", sub_part)))
                    for sub_part in sub_parts
                )
                continue
        refined_parts.append(part)
        refined_weights.append(char_weights[index])
    parts = refined_parts
    weights = refined_weights
    total_weight = sum(weights)
    max_seconds_per_char = 0.5
    effective_duration = min(duration, total_weight * max_seconds_per_char)
    tail_padding = max(0.0, duration - effective_duration)
    cursor = 0.0
    segments: list[SubtitleSegment] = []
    for index, part in enumerate(parts):
        segment_duration = effective_duration * (weights[index] / total_weight)
        if qwen3_event_label(part):
            segment_duration = min(segment_duration, 1.2)
        end = effective_duration if index == len(parts) - 1 else min(effective_duration, cursor + segment_duration)
        segments.append(SubtitleSegment(cursor, end, part))
        cursor = end
    if tail_padding > 0 and segments:
        last = segments[-1]
        segments[-1] = SubtitleSegment(last.start, duration, last.text)
    return merge_short_qwen3_segments(segments)


def merge_short_qwen3_segments(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    if not segments:
        return []
    merged: list[SubtitleSegment] = []
    for segment in segments:
        duration = segment.end - segment.start
        if (
            merged
            and duration < 1.5
            and not qwen3_event_label(segment.text)
            and not qwen3_event_label(merged[-1].text)
        ):
            previous = merged[-1]
            joiner = "" if previous.text.endswith(("，", "。", "！", "？", ",", ".", "!", "?")) else " "
            merged[-1] = SubtitleSegment(
                previous.start,
                segment.end,
                f"{previous.text}{joiner}{segment.text}".strip(),
            )
            continue
        merged.append(segment)
    return merged


def infer_source_title(source: str) -> str:
    source = source.strip()
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        if "bilibili.com" in parsed.netloc.lower():
            match = re.search(r"/video/([^/?#]+)", parsed.path)
            if match:
                return match.group(1)
        if "youtube.com" in parsed.netloc.lower():
            match = re.search(r"[?&]v=([^&]+)", parsed.query)
            if match:
                return match.group(1)
        return parsed.path.rstrip("/").split("/")[-1] or parsed.netloc or "video"
    return Path(source).stem or "media"


class RealtimeWorker(QObject):
    """Stream audio directly to Whisper for real-time captions."""

    caption = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(self, mode: CaptionMode, capture_id: int = 0) -> None:
        super().__init__()
        self.mode = mode
        self.capture_id = capture_id
        self.proc: Optional[subprocess.Popen[str]] = None
        self._stop = False

    def run(self) -> None:
        if self.mode.backend != "whisper_cpp":
            self.status.emit("Realtime capture currently requires whisper.cpp/whisper-stream.")
            self.finished.emit()
            return
        if not self.mode.available:
            self.status.emit(f"Missing model: {self.mode.model}")
            self.finished.emit()
            return

        cmd = [
            WHISPER_STREAM,
            "-m",
            str(self.mode.model),
            "-t",
            "8",
            "-c",
            str(self.capture_id),
            *self.mode.args,
        ]
        self.status.emit("Starting: " + " ".join(shlex.quote(part) for part in cmd))
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.proc.stdout
            for line in self.proc.stdout:
                if self._stop:
                    break
                text = self._extract_caption(line)
                if text:
                    self.caption.emit(text)
            self.status.emit("Realtime worker stopped")
        except Exception as exc:
            self.status.emit(f"Realtime error: {exc}")
        finally:
            self.stop()
            self.finished.emit()

    def stop(self) -> None:
        self._stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    @staticmethod
    def _extract_caption(line: str) -> str:
        line = line.strip()
        match = TIMESTAMP_RE.match(line)
        if match:
            return match.group(1).strip()
        if line and not line.startswith(("whisper_", "ggml_", "main:", "system_info")):
            return line
        return ""


def emit_throttled_process_output(status_signal: Signal, line: str, state: dict) -> None:
    line = line.rstrip()
    if not line:
        return
    progress_match = PROGRESS_LINE_RE.search(line)
    now = time.monotonic()
    if progress_match:
        progress_key = progress_match.group(1)
        if state.get("last_progress_key") != progress_key or now - state.get("last_emit", 0.0) >= 0.6:
            status_signal.emit(line)
            state["last_progress_key"] = progress_key
            state["last_emit"] = now
        return
    if any(token in line for token in ("frame=", "size=", "speed=", "time=", "bitrate=")):
        if now - state.get("last_emit", 0.0) >= 1.0:
            status_signal.emit(line)
            state["last_emit"] = now
        return
    status_signal.emit(line)
    state["last_emit"] = now


class QueueWorker(QObject):
    """Process a queue of media files (URLs or local files)."""

    status = Signal(str)
    caption = Signal(str)
    output_ready = Signal(str, str)  # (source, output_base)
    finished_item = Signal(str, bool)
    finished = Signal()

    def __init__(self, items: list[str], mode: CaptionMode) -> None:
        super().__init__()
        self.items = items
        self.mode = mode
        self._stop = False
        self.proc: Optional[subprocess.Popen[str]] = None
        self._temp_paths: list[Path] = []

    def run(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for source in self.items:
            if self._stop:
                break
            ok = self._process(source)
            self.finished_item.emit(source, ok)
        self.finished.emit()

    def stop(self) -> None:
        self._stop = True
        _terminate_process(self.proc)
        self.proc = None

    def _track_temp_path(self, path: Path) -> Path:
        self._temp_paths.append(path)
        return path

    def _process(self, source: str) -> bool:
        if not self.mode.available:
            self.status.emit(f"Missing model: {self.mode.model}")
            return False

        source_title = infer_source_title(source)
        safe_name = clean_title_for_filename(source_title)
        output_dir = source_output_dir(OUTPUT_DIR, source_title)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = output_dir / f"{safe_name}-{stamp}"
        self._temp_paths = []
        wav = self._track_temp_path(Path(tempfile.gettempdir()) / f"whisper-captioner-{stamp}.wav")

        try:
            if source.startswith(("http://", "https://")):
                downloaded = Path(tempfile.gettempdir()) / f"whisper-captioner-{stamp}.%(ext)s"
                self._run(
                    [
                        YT_DLP,
                        "-x",
                        "--audio-format",
                        "wav",
                        "--cookies-from-browser",
                        "chrome",
                        "-o",
                        str(downloaded),
                        source,
                    ],
                    "Downloading audio",
                )
                candidates = sorted(Path(tempfile.gettempdir()).glob(f"whisper-captioner-{stamp}.*"))
                audio = next((p for p in candidates if p.suffix.lower() in {".wav", ".m4a", ".mp3", ".opus"}), None)
                if audio is None:
                    raise RuntimeError("yt-dlp did not produce an audio file")
                for candidate in candidates:
                    self._track_temp_path(candidate)
                wav = audio
            else:
                self._run(
                    [
                        FFMPEG,
                        "-hide_banner",
                        "-y",
                        "-i",
                        source,
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(wav),
                    ],
                    "Preparing audio",
                )

            if self.mode.backend == "mlx_audio":
                if qwen3_asr_mode(self.mode):
                    segments = self._transcribe_local_qwen3_asr_chunked(wav)
                    save_segments_as_txt(base.with_suffix(".txt"), segments)
                    save_segments_as_srt(base.with_suffix(".srt"), segments)
                else:
                    self._run(
                        [
                            MLX_AUDIO_STT,
                            "--model", str(self.mode.model),
                            "--audio", str(wav),
                            "--output-path", str(base),
                            "--format", "txt",
                            "--language", "zh",
                            "--chunk-duration", "30",
                        ],
                        "Transcribing with MLX-Audio",
                    )
            elif self.mode.backend == "mlx_whisper":
                self._run(
                    [
                        MLX_WHISPER,
                        str(wav),
                        "--model", str(self.mode.model),
                        "--language", "zh",
                        "--output-format", "txt",
                        "--output-dir", str(base.parent),
                        "--output-name", base.name,
                        "--verbose", "False",
                    ],
                    "Transcribing with MLX Whisper",
                )
            elif self.mode.backend == "sense_voice_cpp":
                segments = self._transcribe_local_sense_voice_cpp_chunked(wav)
                save_segments_as_txt(base.with_suffix(".txt"), segments)
                save_segments_as_srt(base.with_suffix(".srt"), segments)
            else:
                self._run(
                    [
                        WHISPER_CLI,
                        "-m", str(self.mode.model),
                        "-f", str(wav),
                        "-t", "8",
                        *self.mode.args,
                        "-of", str(base),
                    ],
                    "Transcribing",
                )
            txt = base.with_suffix(".txt")
            if txt.exists():
                self.caption.emit(txt.read_text(encoding="utf-8", errors="ignore")[-1200:])
            self.output_ready.emit(source, str(base))
            self.status.emit(f"Done: {base}")
            return True
        except Exception as exc:
            self.status.emit(f"Failed {source}: {exc}")
            return False
        finally:
            _cleanup_temp_paths(self._temp_paths)

    def _transcribe_local_qwen3_asr_chunked(self, wav: Path) -> list[SubtitleSegment]:
        duration = self._get_duration(wav)
        chunk_seconds = 30.0
        all_segments: list[SubtitleSegment] = []
        offset = 0.0
        chunk_index = 0
        while offset < duration and not self._stop:
            remaining = min(chunk_seconds, duration - offset)
            chunk_wav = self._track_temp_path(Path(tempfile.gettempdir()) / f"{wav.stem}-qwen3-chunk{chunk_index}.wav")
            chunk_out = Path(tempfile.gettempdir()) / f"{wav.stem}-qwen3-chunk{chunk_index}"
            self._track_temp_path(chunk_out.with_suffix(".txt"))
            self._run(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss", str(offset),
                    "-t", str(remaining),
                    "-i", str(wav),
                    "-ac", "1",
                    "-ar", "16000",
                    str(chunk_wav),
                ],
                f"Preparing Qwen3-ASR chunk {chunk_index}",
            )
            output_text = self._run_capture(
                [
                    MLX_AUDIO_STT,
                    "--model", str(self.mode.model),
                    "--audio", str(chunk_wav),
                    "--output-path", str(chunk_out),
                    "--format", "txt",
                    "--language", "zh",
                    "--chunk-duration", str(max(1, int(remaining))),
                ],
                f"Transcribing Qwen3-ASR chunk {chunk_index}",
            )
            chunk_txt = chunk_out.with_suffix(".txt")
            chunk_text = chunk_txt.read_text(encoding="utf-8", errors="ignore").strip() if chunk_txt.exists() else output_text
            all_segments.extend(
                SubtitleSegment(segment.start + offset, segment.end + offset, segment.text)
                for segment in pseudo_timestamp_qwen3_text(chunk_text, remaining)
            )
            offset += remaining
            chunk_index += 1
        return all_segments

    def _transcribe_local_sense_voice_cpp_chunked(self, wav: Path) -> list[SubtitleSegment]:
        duration = self._get_duration(wav)
        if duration <= self._sense_voice_effective_chunk_seconds():
            output_text = self._run_capture(
                [
                    SENSE_VOICE_CPP_MAIN,
                    "-m", str(self.mode.model),
                    "-f", str(wav),
                    "-t", "8",
                    "-l", "zh",
                    "-itn",
                ],
                "Transcribing with SenseVoice.cpp",
            )
            return parse_sense_voice_output(output_text)

        self.status.emit(
            f"SenseVoice.cpp local audio duration: {duration:.1f}s — processing in chunked pipeline"
        )
        segments = self._transcribe_sense_voice_cpp_chunk_series(
            wav,
            chunk_stem=wav.stem,
            chunk_duration=self._sense_voice_effective_chunk_seconds(),
        )
        self.status.emit(f"SenseVoice.cpp merged {len(segments)} segment(s)")
        return segments

    @staticmethod
    def _sense_voice_effective_chunk_seconds() -> float:
        return 30.0

    @staticmethod
    def _sense_voice_chunk_window_seconds() -> float:
        return 30.0 + SENSE_VOICE_CHUNK_OVERLAP_SECONDS

    def _transcribe_sense_voice_cpp_chunk_series(
        self,
        audio_path: Path,
        chunk_stem: str,
        chunk_duration: float,
    ) -> list[SubtitleSegment]:
        overlap = SENSE_VOICE_CHUNK_OVERLAP_SECONDS
        step = max(1.0, chunk_duration)
        total_duration = self._get_duration(audio_path)
        segments: list[SubtitleSegment] = []
        offset = 0.0
        chunk_index = 0
        while offset < total_duration and not self._stop:
            actual_duration = min(chunk_duration + overlap, total_duration - offset)
            chunk_wav = self._track_temp_path(
                Path(tempfile.gettempdir()) / f"{chunk_stem}-sensevoice-chunk{chunk_index}.wav"
            )
            self._run(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss", str(offset),
                    "-t", str(actual_duration),
                    "-i", str(audio_path),
                    "-ac", "1",
                    "-ar", "16000",
                    str(chunk_wav),
                ],
                f"Preparing SenseVoice.cpp chunk {chunk_index}",
            )
            output_text = self._run_capture(
                [
                    SENSE_VOICE_CPP_MAIN,
                    "-m", str(self.mode.model),
                    "-f", str(chunk_wav),
                    "-t", "8",
                    "-l", "zh",
                    "-itn",
                ],
                f"Transcribing SenseVoice.cpp chunk {chunk_index}",
            )
            local_segments = self._trim_sense_voice_overlap(
                parse_sense_voice_output(output_text),
                leading_trim=0.0,
                trailing_trim=0.0 if offset + actual_duration >= total_duration else overlap,
                chunk_duration=actual_duration,
            )
            segments.extend(
                SubtitleSegment(segment.start + offset, segment.end + offset, segment.text)
                for segment in local_segments
            )
            offset += step
            chunk_index += 1
        return segments

    @staticmethod
    def _trim_sense_voice_overlap(
        segments: list[SubtitleSegment],
        leading_trim: float,
        trailing_trim: float,
        chunk_duration: float,
    ) -> list[SubtitleSegment]:
        trimmed: list[SubtitleSegment] = []
        upper_bound = max(0.0, chunk_duration - trailing_trim)
        for segment in segments:
            start = max(segment.start, leading_trim)
            end = min(segment.end, upper_bound)
            text = segment.text.strip()
            if text and end > start:
                trimmed.append(SubtitleSegment(start, end, text))
        return trimmed

    @staticmethod
    def _get_duration(audio_path: Path) -> float:
        result = subprocess.run(
            [
                FFPROBE,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not determine audio duration: {result.stderr.strip()}")
        try:
            return float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(f"Could not determine audio duration: {result.stdout.strip()}") from exc

    def _run(self, cmd: list[str], label: str) -> None:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert self.proc.stdout
            output_lines: list[str] = []
            output_state: dict[str, object] = {}
            for line in self.proc.stdout:
                if self._stop:
                    _terminate_process(self.proc)
                    raise RuntimeError("Queue stopped")
                if line.strip():
                    output_lines.append(line.rstrip())
                emit_throttled_process_output(self.status, line, output_state)
            if self.proc.wait() != 0:
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
                raise RuntimeError(f"command failed: {cmd[0]}\n\nOutput:\n{error_context}")
        finally:
            self.proc = None

    def _run_capture(self, cmd: list[str], label: str) -> str:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert self.proc.stdout
            output_lines: list[str] = []
            output_state: dict[str, object] = {}
            for line in self.proc.stdout:
                if self._stop:
                    _terminate_process(self.proc)
                    raise RuntimeError("Queue stopped")
                if line.strip():
                    output_lines.append(line.rstrip())
                emit_throttled_process_output(self.status, line, output_state)
            if self.proc.wait() != 0:
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
                raise RuntimeError(f"command failed: {cmd[0]}\n\nOutput:\n{error_context}")
            return "\n".join(output_lines)
        finally:
            self.proc = None


class LLMTextWorker(QObject):
    """Run a long-form LLM post-process task without blocking the Qt UI."""

    status = Signal(str)
    result = Signal(str, str)  # (task_key, text)
    finished = Signal()

    def __init__(
        self,
        task_key: str,
        user_text: str,
        provider: LLMProvider,
        api_key: str,
        api_url: str = "",
        model_id: str = "",
        system_prompt: str = "你是一个严谨的视频内容分析助手。",
        max_tokens: int = 16000,
    ) -> None:
        super().__init__()
        self.task_key = task_key
        self.user_text = user_text
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._stop = False

    def run(self) -> None:
        try:
            if self._stop:
                return
            self.status.emit(f"Running LLM post-process: {self.task_key}")
            text = llm_generate_text(
                self.user_text,
                self.provider,
                self.api_key,
                self.api_url,
                self.model_id,
                self.system_prompt,
                timeout=240,
                max_tokens=self.max_tokens,
            )
            if not self._stop:
                self.result.emit(self.task_key, text)
        except Exception as exc:
            self.status.emit(f"LLM post-process failed: {exc}")
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop = True


class RollingPrefetchWorker(QObject):
    """Download audio, split into chunks, transcribe incrementally.

    Emits *first_segments* when the first chunk is ready (triggers playback),
    then *more_segments* for each subsequent chunk.  *all_done* fires after
    every chunk has been transcribed.
    """

    status = Signal(str)
    first_segments = Signal(list)
    more_segments = Signal(list)
    progress = Signal(int, int)  # (current_chunk, total_chunks)
    all_done = Signal()
    native_subtitles_detected = Signal(list, str)  # (segments, message)
    finished = Signal()

    def __init__(
        self, url: str, mode: CaptionMode, chunk_seconds: int = 30,
        llm_provider: Optional[LLMProvider] = None, llm_api_key: str = "",
        llm_api_url: str = "", llm_model_id: str = "",
    ) -> None:
        super().__init__()
        self.url = url
        self.mode = mode
        self.chunk_seconds = chunk_seconds
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_api_url = llm_api_url
        self.llm_model_id = llm_model_id
        self._stop = False
        self.proc: Optional[subprocess.Popen[str]] = None
        self._temp_paths: list[Path] = []

    def run(self) -> None:
        try:
            self._do_rolling_prefetch()
        except Exception as exc:
            self.status.emit(f"Rolling prefetch failed: {exc}")
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop = True
        _terminate_process(self.proc)
        self.proc = None

    def _track_temp_path(self, path: Path) -> Path:
        self._temp_paths.append(path)
        return path

    # ---- internal pipeline ----

    def _do_rolling_prefetch(self) -> None:
        self._temp_paths = []
        try:
            if not self.mode.available:
                raise RuntimeError(f"Missing model: {self.mode.model}")

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.cache_url = canonical_media_url(self.url)
            job_key = cache_slug(self.cache_url, self.mode.backend, self.mode.model_name, self.chunk_seconds)
            job_cache_dir = CACHE_DIR / job_key
            job_cache_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "url": self.url,
                "cache_url": self.cache_url,
                "model": str(self.mode.model),
                "backend": self.mode.backend,
                "chunk_seconds": self.chunk_seconds,
                "pipeline_version": SUBTITLE_PIPELINE_VERSION,
                "llm_provider": self.llm_provider.key if self.llm_provider else "raw",
                "llm_model": self.llm_model_id or (self.llm_provider.model_id if self.llm_provider else ""),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (job_cache_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stamp = time.strftime("%Y%m%d-%H%M%S")
            raw_output_base = self._raw_output_base(stamp)
            output_base = self._optimized_output_base(stamp)

            native_segments, native_kind = self._load_or_fetch_native_subtitles(job_cache_dir)
            if native_kind == "zh":
                self.native_subtitles_detected.emit(
                    native_segments,
                    "检测到视频自带中文字幕，已加载。跳过 Whisper 和 LLM 识别以节省资源。"
                )
                return

            # 1. Pause Chrome
            self.status.emit("Pausing Chrome while captions are prepared")
            from whisper_captioner.chrome_control import chrome_pause, chrome_pause_url
            if chrome_pause_url(self.cache_url) is None:
                chrome_pause()
            if self._stop:
                return

            final_cache = self._final_subtitle_cache_path(job_cache_dir)
            cached_segments = self._load_current_final_cache(final_cache)
            if cached_segments is not None:
                try:
                    self.status.emit(f"Loaded final subtitle cache: {final_cache}")
                    self._export_final_subtitles(cached_segments, output_base)
                    self.first_segments.emit(cached_segments)
                    self.all_done.emit()
                    return
                except Exception as exc:
                    self.status.emit(f"Final subtitle cache unreadable, rebuilding: {exc}")

            # 2. Download full audio
            downloaded = Path(tempfile.gettempdir()) / f"whisper-rolling-{stamp}.%(ext)s"
            self._run_cmd(
                [
                    YT_DLP,
                    "-x",
                    "--audio-format",
                    "wav",
                    "--cookies-from-browser",
                    "chrome",
                    "-o",
                    str(downloaded),
                    self.url,
                ],
                "Downloading video audio",
            )
            candidates = sorted(Path(tempfile.gettempdir()).glob(f"whisper-rolling-{stamp}.*"))
            audio = next(
                (p for p in candidates if p.suffix.lower() in {".wav", ".m4a", ".mp3", ".opus"}),
                None,
            )
            if audio is None:
                raise RuntimeError("yt-dlp did not produce an audio file")
            for candidate in candidates:
                self._track_temp_path(candidate)
            if self._stop:
                return

            # 3. Get duration
            duration = self._get_duration(audio)
            num_chunks = int(duration // self.chunk_seconds) + (1 if duration % self.chunk_seconds else 0)
            self.status.emit(
                f"Audio duration: {duration:.1f}s — processing in {num_chunks} chunk(s) of {self.chunk_seconds}s"
            )

            # 4. Chunk → transcribe → cache. LLM proofreading runs once on the full transcript.
            offset = 0.0
            chunk_index = 0
            rebuilt_raw_cache = False

            while offset < duration and not self._stop:
                remaining = min(self.chunk_seconds, duration - offset)
                chunk_wav = self._track_temp_path(
                    Path(tempfile.gettempdir()) / f"whisper-rolling-{stamp}-chunk{chunk_index}.wav"
                )
                chunk_out = Path(tempfile.gettempdir()) / f"whisper-rolling-{stamp}-chunk{chunk_index}"
                for suffix in (".srt", ".txt"):
                    self._track_temp_path(chunk_out.with_suffix(suffix))
                raw_cache = job_cache_dir / f"chunk-{chunk_index:04d}-raw.json"

                segments: list[SubtitleSegment] = []
                use_cached = False
                if raw_cache.exists():
                    cached_segments = load_segments(raw_cache)
                    if self._chunk_cache_looks_bad(cached_segments, remaining):
                        self.status.emit(
                            f"Chunk {chunk_index}: cached Whisper result looks sparse; rebuilding"
                        )
                        raw_cache.unlink(missing_ok=True)
                        rebuilt_raw_cache = True
                    else:
                        segments = cached_segments
                        use_cached = True
                        self.status.emit(f"Chunk {chunk_index}: loaded Whisper cache")

                if not use_cached:
                    self.status.emit(f"Extracting chunk {chunk_index} ({offset:.0f}s – {offset + remaining:.0f}s)")
                    self._run_cmd(
                        [
                            FFMPEG, "-hide_banner", "-y",
                            "-ss", str(offset),
                            "-t", str(remaining),
                            "-i", str(audio),
                            "-ac", "1", "-ar", "16000",
                            str(chunk_wav),
                        ],
                        f"Preparing chunk {chunk_index}",
                    )
                    if self._stop:
                        break

                    self.status.emit(f"Transcribing chunk {chunk_index}")
                    srt_path = self._transcribe_chunk(chunk_wav, chunk_out, chunk_index)
                    if self._stop:
                        break

                    if srt_path.exists():
                        raw_segments = parse_srt(srt_path)
                        raw_segments = self._clamp_chunk_segments(raw_segments, remaining)
                        if self._chunk_cache_looks_bad(
                            [SubtitleSegment(s.start, s.end, s.text) for s in raw_segments],
                            remaining,
                        ):
                            repaired_segments = self._repair_sparse_chunk_with_subchunks(
                                chunk_wav, chunk_out, chunk_index, remaining
                            )
                            if repaired_segments:
                                raw_segments = repaired_segments
                        # Shift timestamps to absolute position in the full audio.
                        segments = [
                            SubtitleSegment(s.start + offset, s.end + offset, s.text)
                            for s in raw_segments
                        ]
                        save_segments(raw_cache, segments)
                        rebuilt_raw_cache = True
                        self.status.emit(f"Chunk {chunk_index}: saved Whisper cache")
                    else:
                        segments = []
                        self.status.emit(f"Chunk {chunk_index}: no SRT produced")

                    if self._chunk_cache_looks_bad(segments, remaining):
                        self.status.emit(
                            f"Chunk {chunk_index}: transcription still looks sparse after rebuild"
                        )

                if segments:
                    self.status.emit(
                        f"Chunk {chunk_index}: {len(segments)} raw segment(s) cached"
                    )
                else:
                    self.status.emit(f"Chunk {chunk_index}: no speech detected")

                self.progress.emit(chunk_index + 1, num_chunks)
                offset += remaining  # Use actual chunk duration, not fixed chunk_seconds
                chunk_index += 1

            if not self._stop:
                if rebuilt_raw_cache:
                    self._discard_derived_subtitle_caches(job_cache_dir)
                all_raw_segments = self._load_all_raw_segments(job_cache_dir)
                if all_raw_segments:
                    self._export_subtitles(all_raw_segments, raw_output_base)
                    self.status.emit(f"Raw subtitles written: {raw_output_base.with_suffix('.srt')}")
                final_segments = self._run_full_document_polish(job_cache_dir, output_base)
                if final_segments:
                    self.first_segments.emit(final_segments)
                self.all_done.emit()
                self.status.emit("All chunks transcribed")
        finally:
            _cleanup_temp_paths(self._temp_paths)

    def _llm_cache_path(self, job_cache_dir: Path, chunk_index: int) -> Path:
        if not self.llm_provider:
            return job_cache_dir / f"chunk-{chunk_index:04d}-llm-disabled.json"
        provider_key = self.llm_provider.key
        model_key = self.llm_model_id or self.llm_provider.model_id
        proofread_key = cache_slug(provider_key, model_key, self.llm_api_url)
        return job_cache_dir / f"chunk-{chunk_index:04d}-llm-{proofread_key}.json"

    def _fusion_cache_path(self, job_cache_dir: Path, chunk_index: int) -> Path:
        if not self.llm_provider:
            return job_cache_dir / f"chunk-{chunk_index:04d}-fusion-disabled.json"
        provider_key = self.llm_provider.key
        model_key = self.llm_model_id or self.llm_provider.model_id
        fusion_key = cache_slug(provider_key, model_key, self.llm_api_url, "native-fusion-v1")
        return job_cache_dir / f"chunk-{chunk_index:04d}-fusion-{fusion_key}.json"

    def _full_polish_cache_path(self, job_cache_dir: Path) -> Path:
        provider_key = self.llm_provider.key if self.llm_provider else "disabled"
        model_key = self.llm_model_id or (self.llm_provider.model_id if self.llm_provider else "")
        polish_key = cache_slug(provider_key, model_key, self.llm_api_url, "full-document-polish-v1")
        return job_cache_dir / f"all-polished-{polish_key}.json"

    def _load_all_raw_segments(self, job_cache_dir: Path) -> list[SubtitleSegment]:
        raw_files = sorted(job_cache_dir.glob("chunk-*-raw.json"))
        all_segments: list[SubtitleSegment] = []
        for path in raw_files:
            all_segments.extend(load_segments(path))
        all_segments.sort(key=lambda item: (item.start, item.end))
        return all_segments

    def _final_subtitle_cache_path(self, job_cache_dir: Path) -> Path:
        return job_cache_dir / "final-subtitles-current.json"

    def _discard_derived_subtitle_caches(self, job_cache_dir: Path) -> None:
        for path in [self._final_subtitle_cache_path(job_cache_dir), self._full_polish_cache_path(job_cache_dir)]:
            if path.exists():
                path.unlink()
                self.status.emit(f"Discarded derived subtitle cache after raw rebuild: {path.name}")

    def _transcribe_chunk(self, chunk_wav: Path, chunk_out: Path, chunk_index: int) -> Path:
        if self.mode.backend == "mlx_audio":
            if qwen3_asr_mode(self.mode):
                output_path = chunk_out.with_suffix("")
                output_text = self._run_cmd_capture(
                    [
                        MLX_AUDIO_STT,
                        "--model", str(self.mode.model),
                        "--audio", str(chunk_wav),
                        "--output-path", str(output_path),
                        "--format", "txt",
                        "--language", "zh",
                        "--chunk-duration", str(max(1, int(self.chunk_seconds))),
                    ],
                    f"Transcribing chunk {chunk_index} with Qwen3-ASR",
                )
                txt_path = Path(str(output_path) + ".txt")
                chunk_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip() if txt_path.exists() else output_text
                segments = pseudo_timestamp_qwen3_text(chunk_text, float(self.chunk_seconds))
                srt_path = chunk_out.with_suffix(".srt")
                save_segments_as_srt(srt_path, segments)
                return srt_path
            output_path = chunk_out.with_suffix("")
            self._run_cmd(
                [
                    MLX_AUDIO_STT,
                    "--model", str(self.mode.model),
                    "--audio", str(chunk_wav),
                    "--output-path", str(output_path),
                    "--format", "srt",
                    "--language", "zh",
                    "--chunk-duration", str(max(1, int(self.chunk_seconds))),
                ],
                f"Transcribing chunk {chunk_index} with MLX-Audio",
            )
            return Path(str(output_path) + ".srt")
        if self.mode.backend == "mlx_whisper":
            self._run_cmd(
                [
                    MLX_WHISPER,
                    str(chunk_wav),
                    "--model", str(self.mode.model),
                    "--language", "zh",
                    "--output-format", "srt",
                    "--output-dir", str(chunk_out.parent),
                    "--output-name", chunk_out.name,
                    "--verbose", "False",
                ],
                f"Transcribing chunk {chunk_index} with MLX Whisper",
            )
            return chunk_out.with_suffix(".srt")
        if self.mode.backend == "sense_voice_cpp":
            output_text = self._run_cmd_capture(
                [
                    SENSE_VOICE_CPP_MAIN,
                    "-m", str(self.mode.model),
                    "-f", str(chunk_wav),
                    "-t", "8",
                    "-l", "zh",
                    "-itn",
                ],
                f"Transcribing chunk {chunk_index} with SenseVoice.cpp",
            )
            segments = self._trim_sense_voice_overlap(
                parse_sense_voice_output(output_text),
                leading_trim=0.0,
                trailing_trim=0.0,
                chunk_duration=self._sense_voice_chunk_window_seconds(),
            )
            srt_path = chunk_out.with_suffix(".srt")
            save_segments_as_srt(srt_path, segments)
            return srt_path
        self._run_cmd(
            [
                WHISPER_CLI,
                "-m", str(self.mode.model),
                "-f", str(chunk_wav),
                "-t", "8",
                "-l", "zh",
                "-osrt",
                "-of", str(chunk_out),
            ],
            f"Transcribing chunk {chunk_index}",
        )
        return chunk_out.with_suffix(".srt")

    def _repair_sparse_chunk_with_subchunks(
        self,
        chunk_wav: Path,
        chunk_out: Path,
        chunk_index: int,
        chunk_duration: float,
    ) -> list[SubtitleSegment]:
        if self.mode.backend not in {"mlx_whisper", "mlx_audio"} or chunk_duration <= 12:
            return []
        self.status.emit(
            f"Chunk {chunk_index}: retrying sparse result with smaller MLX subchunks"
        )
        half = chunk_duration / 2
        repaired: list[SubtitleSegment] = []
        for part_index, start_offset in enumerate((0.0, half)):
            part_duration = min(half, chunk_duration - start_offset)
            if part_duration <= 0:
                continue
            part_wav = self._track_temp_path(chunk_wav.with_name(f"{chunk_wav.stem}-part{part_index}.wav"))
            part_out = chunk_out.with_name(f"{chunk_out.stem}-part{part_index}")
            for suffix in (".srt", ".txt"):
                self._track_temp_path(part_out.with_suffix(suffix))
            self._run_cmd(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss",
                    str(start_offset),
                    "-t",
                    str(part_duration),
                    "-i",
                    str(chunk_wav),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(part_wav),
                ],
                f"Preparing chunk {chunk_index} subchunk {part_index}",
            )
            srt_path = self._transcribe_chunk(part_wav, part_out, int(f"{chunk_index}{part_index}"))
            if not srt_path.exists():
                continue
            sub_segments = self._clamp_chunk_segments(parse_srt(srt_path), part_duration)
            repaired.extend(
                SubtitleSegment(s.start + start_offset, s.end + start_offset, s.text)
                for s in sub_segments
            )
        return self._clamp_chunk_segments(repaired, chunk_duration)

    @staticmethod
    def _clamp_chunk_segments(segments: list[SubtitleSegment], chunk_duration: float) -> list[SubtitleSegment]:
        cleaned: list[SubtitleSegment] = []
        for segment in segments:
            start = max(0.0, min(segment.start, chunk_duration))
            end = max(0.0, min(segment.end, chunk_duration))
            text = segment.text.strip()
            if text and end > start:
                cleaned.append(SubtitleSegment(start, end, text))
        return cleaned

    @staticmethod
    def _chunk_cache_looks_bad(segments: list[SubtitleSegment], chunk_duration: float) -> bool:
        if chunk_duration < 20:
            return False
        if not segments:
            return True
        speech_span = max(segment.end for segment in segments) - min(segment.start for segment in segments)
        total_text = sum(len(segment.text.strip()) for segment in segments)
        if len(segments) <= 2 and speech_span < chunk_duration * 0.25 and total_text < 40:
            return True
        last_text = segments[-1].text.strip()
        if len(last_text) > 30:
            words = last_text.split()
            if len(words) >= 6 and len(set(words[-6:])) <= 2:
                return True
            if any(last_text.count(token * 2) for token in set(words[: min(6, len(words))])):
                return True
        return False

    def _pipeline_signature(self) -> dict:
        return {
            "pipeline_version": SUBTITLE_PIPELINE_VERSION,
            "whisper_backend": self.mode.backend,
            "whisper_model": self.mode.model_name,
            "chunk_seconds": self.chunk_seconds,
            "native_subtitles": "zh-only-skip",
            "llm_provider": self.llm_provider.key if self.llm_provider else "raw",
            "llm_model": self.llm_model_id or (self.llm_provider.model_id if self.llm_provider else ""),
            "llm_api_url": self.llm_api_url,
            "proofread_scope": "full-document",
            "ai_zh_reference": False,
            "term_extraction": False,
        }

    def _load_current_final_cache(self, cache_path: Path) -> Optional[list[SubtitleSegment]]:
        if not cache_path.exists():
            return None
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            self.status.emit("Ignoring legacy final subtitle cache without pipeline signature")
            return None
        if data.get("pipeline_signature") != self._pipeline_signature():
            self.status.emit("Final subtitle cache belongs to a different pipeline; rebuilding")
            return None
        return [segment_from_dict(item) for item in data.get("segments", [])]

    def _save_current_final_cache(self, cache_path: Path, segments: list[SubtitleSegment]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline_signature": self._pipeline_signature(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "url": self.url,
            "cache_url": self.cache_url,
            "segments": [segment_to_dict(segment) for segment in segments],
        }
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cache_path)

    def _output_base(self, stamp: str) -> Path:
        title = clean_title_for_filename(self._source_title())
        return source_output_dir(OUTPUT_DIR, title) / f"{title}-{self._output_variant_suffix()}-{stamp}"

    def _raw_output_base(self, stamp: str) -> Path:
        title = clean_title_for_filename(self._source_title())
        return source_output_dir(OUTPUT_DIR, title) / f"{title}-{self._output_variant_suffix()}-原始识别字幕"

    def _optimized_output_base(self, stamp: str) -> Path:
        title = clean_title_for_filename(self._source_title())
        return source_output_dir(OUTPUT_DIR, title) / f"{title}-{self._output_variant_suffix()}-LLM优化字幕"

    def _output_variant_suffix(self) -> str:
        parts = [clean_title_for_filename(self.mode.key, fallback="mode")]
        if self.llm_provider:
            parts.append(clean_title_for_filename(self.llm_provider.key, fallback="llm"))
            model_name = self.llm_model_id or self.llm_provider.model_id
            if model_name:
                parts.append(clean_title_for_filename(model_name, fallback="model"))
        else:
            parts.append("raw")
        return "-".join(parts)

    def _source_title(self) -> str:
        title = self._fetch_remote_title() if self.url.startswith(("http://", "https://")) else ""
        return title or infer_source_title(self.url)

    def _fetch_remote_title(self) -> str:
        try:
            result = subprocess.run(
                [
                    YT_DLP,
                    "--cookies-from-browser",
                    "chrome",
                    "--print",
                    "title",
                    self.url,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            title = result.stdout.strip().splitlines()
            return title[0].strip() if title else ""
        except Exception:
            return ""

    def _export_subtitles(self, segments: list[SubtitleSegment], output_base: Path) -> None:
        save_segments_as_srt(output_base.with_suffix(".srt"), segments)
        save_segments_as_txt(output_base.with_suffix(".txt"), segments)

    def _export_final_subtitles(self, segments: list[SubtitleSegment], output_base: Path) -> None:
        self._export_subtitles(segments, output_base)
        self.status.emit(f"Final subtitles written: {output_base.with_suffix('.srt')}")

    def _run_full_document_polish(self, job_cache_dir: Path, output_base: Path) -> list[SubtitleSegment]:
        raw_files = sorted(job_cache_dir.glob("chunk-*-raw.json"))
        if not raw_files:
            return []
        cache_path = self._full_polish_cache_path(job_cache_dir)
        all_segments: list[SubtitleSegment] = []
        for path in raw_files:
            all_segments.extend(load_segments(path))
        all_segments.sort(key=lambda item: (item.start, item.end))
        try:
            if self.llm_provider and llm_provider_ready(self.llm_provider, self.llm_api_key) and cache_path.exists():
                polished = load_segments(cache_path)
                self.status.emit("Loaded full-document LLM polish cache")
            elif self.llm_provider and llm_provider_ready(self.llm_provider, self.llm_api_key):
                self.status.emit(f"Running full-document LLM polish on {len(all_segments)} segment(s)")
                t0 = time.monotonic()
                polished = llm_proofread(
                    all_segments,
                    self.llm_provider,
                    self.llm_api_key,
                    self.llm_api_url,
                    self.llm_model_id,
                    timeout=120,
                    max_tokens=60000 if self.llm_provider.key == "gemini_flash" else 12000,
                )
                save_segments(cache_path, polished)
                self.status.emit(
                    f"Full-document LLM polish cached ({time.monotonic() - t0:.1f}s)"
                )
            else:
                polished = all_segments
                self.status.emit("No LLM provider/API key configured; exporting raw Whisper subtitles")
            if polished:
                self._save_current_final_cache(self._final_subtitle_cache_path(job_cache_dir), polished)
                self.status.emit("Updated final subtitle cache for current pipeline")
                self._export_final_subtitles(polished, output_base)
            return polished
        except Exception as exc:
            self.status.emit(f"Full-document LLM polish failed: {exc}; exporting raw Whisper subtitles")
            if all_segments:
                self._save_current_final_cache(self._final_subtitle_cache_path(job_cache_dir), all_segments)
                self._export_final_subtitles(all_segments, output_base)
            return all_segments

    def _load_or_fetch_native_subtitles(self, job_cache_dir: Path) -> tuple[list[SubtitleSegment], str]:
        preferred = [
            ("zh", "zh.*"),
        ]
        for cache_name, lang_expr in preferred:
            cache_path = job_cache_dir / f"native-subtitles-{cache_name}.json"
            if cache_path.exists():
                try:
                    segments = load_segments(cache_path)
                    return segments, cache_name
                except Exception as exc:
                    self.status.emit(f"Native subtitle cache unreadable ({cache_name}): {exc}")

            subs_dir = job_cache_dir / f"native-subs-{cache_name}"
            subs_dir.mkdir(parents=True, exist_ok=True)
            out_template = subs_dir / "native.%(ext)s"
            try:
                self._run_cmd(
                    [
                        YT_DLP,
                        "--skip-download",
                        "--write-subs",
                        "--sub-langs",
                        lang_expr,
                        "--sub-format",
                        "srt/vtt/best",
                        "--cookies-from-browser",
                        "chrome",
                        "-o",
                        str(out_template),
                        self.url,
                    ],
                    f"Fetching native subtitles ({lang_expr})",
                )
            except Exception as exc:
                self.status.emit(f"No native subtitles fetched for {lang_expr}: {exc}")
                continue

            subtitle_files = sorted(
                p for p in subs_dir.glob("native.*") if p.suffix.lower() in {".srt", ".vtt"}
            )
            segments_by_key: dict[tuple[float, float, str], SubtitleSegment] = {}
            for subtitle_file in subtitle_files:
                try:
                    for segment in parse_subtitle_file(subtitle_file):
                        key = (round(segment.start, 2), round(segment.end, 2), segment.text)
                        segments_by_key[key] = segment
                except Exception as exc:
                    self.status.emit(f"Could not parse native subtitle {subtitle_file.name}: {exc}")
            segments = sorted(segments_by_key.values(), key=lambda item: (item.start, item.end))
            if segments:
                save_segments(cache_path, segments)
                self.status.emit(
                    f"Saved native subtitle cache from {len(subtitle_files)} file(s) for {lang_expr}"
                )
                return segments, cache_name

        self.status.emit("No usable native subtitle segments found")
        return [], "none"

    def _extract_terms_with_mlx(self, job_cache_dir: Path, segments: list[SubtitleSegment]) -> list[str]:
        cache_path = job_cache_dir / "mlx-terms.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return [str(item["term"]) for item in data.get("terms", []) if item.get("term")]
            except Exception as exc:
                self.status.emit(f"MLX term cache unreadable: {exc}")
        if not segments or not Path(RAPIDMLX_PYTHON).exists() or not MLX_TERMS_SCRIPT.exists():
            return []
        sample = "\n".join(segment.text for segment in segments[:180])
        try:
            proc = subprocess.run(
                [RAPIDMLX_PYTHON, str(MLX_TERMS_SCRIPT)],
                input=json.dumps({"text": sample}, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=45,
                check=False,
            )
            if proc.returncode != 0:
                self.status.emit(f"MLX term extraction failed: {proc.stderr.strip()[-300:]}")
                return []
            raw = proc.stdout.strip()
            first = raw.find("{")
            last = raw.rfind("}")
            data = json.loads(raw[first : last + 1])
            cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            terms = [str(item["term"]) for item in data.get("terms", []) if item.get("term")]
            self.status.emit(f"MLX extracted {len(terms)} term(s)")
            return terms
        except Exception as exc:
            self.status.emit(f"MLX term extraction error: {exc}")
            return []

    def _get_duration(self, audio_path: Path) -> float:
        result = subprocess.run(
            [
                FFPROBE,
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(audio_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            raise RuntimeError(f"Could not determine audio duration: {result.stderr}")

    def _run_cmd(self, cmd: list[str], label: str) -> None:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            assert self.proc.stdout
            output_lines = []
            output_state: dict[str, object] = {}
            for line in self.proc.stdout:
                if self._stop:
                    _terminate_process(self.proc)
                    raise RuntimeError("Rolling prefetch stopped")
                if line.strip():
                    output_lines.append(line.rstrip())
                    emit_throttled_process_output(self.status, line, output_state)

            returncode = self.proc.wait()
            if returncode != 0 and not self._stop:
                cmd_name = cmd[0]
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"

                # Provide better error messages for yt-dlp failures
                if "yt-dlp" in cmd_name or "yt-dlp" in " ".join(cmd):
                    error_msg = f"yt-dlp failed to download the URL.\n\nLast output:\n{error_context}\n\nPossible reasons:\n- URL is not supported by yt-dlp (e.g., GitHub docs, Wikipedia)\n- Video is region-restricted or requires login\n- Chrome cookies are unavailable, expired, or locked by the browser\n- URL format is incorrect"
                    raise RuntimeError(error_msg)

                raise RuntimeError(f"command failed: {cmd_name}\n\nOutput:\n{error_context}")
        finally:
            self.proc = None

    def _run_cmd_capture(self, cmd: list[str], label: str) -> str:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            assert self.proc.stdout
            output_lines = []
            output_state: dict[str, object] = {}
            for line in self.proc.stdout:
                if self._stop:
                    _terminate_process(self.proc)
                    raise RuntimeError("Rolling prefetch stopped")
                if line.strip():
                    output_lines.append(line.rstrip())
                    emit_throttled_process_output(self.status, line, output_state)

            returncode = self.proc.wait()
            if returncode != 0 and not self._stop:
                cmd_name = cmd[0]
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
                raise RuntimeError(f"command failed: {cmd_name}\n\nOutput:\n{error_context}")
            return "\n".join(output_lines)
        finally:
            self.proc = None

    def _sense_voice_effective_chunk_seconds(self) -> float:
        return float(self.chunk_seconds if hasattr(self, "chunk_seconds") else 30)

    @staticmethod
    def _sense_voice_chunk_window_seconds() -> float:
        return 30.0 + SENSE_VOICE_CHUNK_OVERLAP_SECONDS

    def _transcribe_sense_voice_cpp_chunk_series(
        self,
        audio_path: Path,
        chunk_stem: str,
        chunk_duration: float,
    ) -> list[SubtitleSegment]:
        overlap = SENSE_VOICE_CHUNK_OVERLAP_SECONDS
        step = max(1.0, chunk_duration - overlap)
        total_duration = self._get_duration(audio_path)
        segments: list[SubtitleSegment] = []
        offset = 0.0
        chunk_index = 0
        while offset < total_duration and not self._stop:
            actual_duration = min(chunk_duration + overlap, total_duration - offset)
            chunk_wav = self._track_temp_path(
                Path(tempfile.gettempdir()) / f"{chunk_stem}-sensevoice-chunk{chunk_index}.wav"
            )
            self._run_cmd(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss", str(offset),
                    "-t", str(actual_duration),
                    "-i", str(audio_path),
                    "-ac", "1",
                    "-ar", "16000",
                    str(chunk_wav),
                ],
                f"Preparing SenseVoice.cpp chunk {chunk_index}",
            )
            output_text = self._run_cmd_capture(
                [
                    SENSE_VOICE_CPP_MAIN,
                    "-m", str(self.mode.model),
                    "-f", str(chunk_wav),
                    "-t", "8",
                    "-l", "zh",
                    "-itn",
                ],
                f"Transcribing SenseVoice.cpp chunk {chunk_index}",
            )
            local_segments = self._trim_sense_voice_overlap(
                parse_sense_voice_output(output_text),
                leading_trim=0.0 if chunk_index == 0 else overlap / 2,
                trailing_trim=0.0 if offset + actual_duration >= total_duration else overlap / 2,
                chunk_duration=actual_duration,
            )
            segments.extend(
                SubtitleSegment(segment.start + offset, segment.end + offset, segment.text)
                for segment in local_segments
            )
            offset += step
            chunk_index += 1
        return segments

    @staticmethod
    def _trim_sense_voice_overlap(
        segments: list[SubtitleSegment],
        leading_trim: float,
        trailing_trim: float,
        chunk_duration: float,
    ) -> list[SubtitleSegment]:
        trimmed: list[SubtitleSegment] = []
        upper_bound = max(0.0, chunk_duration - trailing_trim)
        for segment in segments:
            start = max(segment.start, leading_trim)
            end = min(segment.end, upper_bound)
            text = segment.text.strip()
            if text and end > start:
                trimmed.append(SubtitleSegment(start, end, text))
        return trimmed
