"""
异步任务和底层工作线程模块

该模块包含了 Whisper Captioner 用于处理耗时任务的各类异步 Worker (继承自 QThread 或 QObject)。
主要职责包括：
- 本地和远程媒体下载、缓存与音频预处理。
- 本地 ASR 模型 (如 whisper.cpp, MLX) 的子进程调用。
- 远程 ASR 服务节点 (NUC) 的 API 通信。
- 实时麦克风录音识别与字幕流处理。
- LLM 文本后处理 (润色、翻译、纠错) 的并行调度。

主要 Worker 类：
- RealtimeWorker / NUCRealtimeWorker: 处理麦克风实时转录。
- QueueWorker: 处理音视频文件的离线批量转录任务。
- RollingPrefetchWorker: 处理基于 URL 的流式增量下载与边下边转录任务。
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QThread, Signal

from whisper_captioner.asr_history import ASRHistoryStore, audio_cache_key_for_url
from whisper_captioner.cache import cache_slug, canonical_media_url, controlled_cache_dir_name
from whisper_captioner.config import (
    BUFFER_PAUSE_MARGIN,
    BUFFER_RESUME_MARGIN,
    CACHE_DIR,
    DEFAULT_SUBTITLE_OFFSET,
    FFMPEG,
    FFPROBE,
    GENERATED_DIR,
    LOCAL_AUDIO_CACHE_DIR,
    MLX_AUDIO_STT,
    MLX_WHISPER,
    MLX_TERMS_SCRIPT,
    MODELS_DIR,
    NUC_OLLAMA_HOST,
    NUC_REMOTE_ASR_ROOT,
    NUC_SSH_PORT,
    NUC_SSH_USER,
    OUTPUT_DIR,
    REALTIME_DIR,
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
from whisper_captioner.external_backends import (
    fuse_gemini_with_whisper,
    gemini_transcribe_audio,
    run_omnivad_shadow,
)
from whisper_captioner.models import (
    ASRResult,
    CaptionMode,
    LLMProvider,
    RetryRegion,
    SubtitleSegment,
    SubtitleWord,
)
from whisper_captioner.subtitle_reliability import (
    LanguagePin,
    audit_asr_result,
    build_cues,
    parse_silencedetect_regions,
    parse_verbose_asr_response,
    quality_report_to_dict,
    merge_retry_regions,
    replace_segments_in_regions,
)
from whisper_captioner.subtitle_io import (
    save_segments,
    save_segments_as_srt,
    save_segments_as_txt,
)
from whisper_captioner.subtitle_io import (
    load_asr_result,
    load_segments,
    overlapping_segments,
    parse_srt,
    parse_sense_voice_output,
    parse_subtitle_file,
    save_segments,
    save_asr_result,
    save_segments_as_srt,
    save_segments_as_txt,
    segment_from_dict,
    segment_to_dict,
)


TIMESTAMP_RE = re.compile(r"^\[[^\]]+\]\s*(.*)$")
PROGRESS_LINE_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)%")
SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")
ADAPTIVE_SPLIT_MULTIPLIER = 1.5
LOCAL_AUDIO_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
NUC_ASR_AUTO_LANGUAGE = "auto"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QueueRunConfig:
    prepared_wavs: dict[str, str] | None = None
    qwen_replicas: int = 2
    qwen_chunk_seconds: float = 45.0
    qwen_parallel_enabled: bool = True
    adaptive_split_enabled: bool = True
    remote_vad_enabled: bool = True
    cpp_threads: int = 6
    cpp_flash_attn: bool = True

    @classmethod
    def from_environment(cls, **overrides: Any) -> "QueueRunConfig":
        values: dict[str, Any] = {
            "qwen_replicas": max(1, min(4, int(os.environ.get("WHISPER_CAPTIONER_QWEN_REPLICAS", "2")))),
            "qwen_chunk_seconds": max(
                10.0, float(os.environ.get("WHISPER_CAPTIONER_QWEN_CHUNK_SECONDS", "45"))
            ),
            "qwen_parallel_enabled": _env_bool("WHISPER_CAPTIONER_QWEN_PARALLEL", True),
            "adaptive_split_enabled": _env_bool("WHISPER_CAPTIONER_ADAPTIVE_SPLIT", True),
            "remote_vad_enabled": _env_bool("WHISPER_CAPTIONER_REMOTE_VAD", True),
            "cpp_threads": max(
                1, min(8, int(os.environ.get("WHISPER_CAPTIONER_CPP_THREADS", "6")))
            ),
            "cpp_flash_attn": _env_bool("WHISPER_CAPTIONER_CPP_FLASH_ATTN", True),
        }
        values.update(overrides)
        return cls(**values)

    def cpp_args(self) -> list[str]:
        return [
            "-t",
            str(self.cpp_threads),
            "--flash-attn" if self.cpp_flash_attn else "--no-flash-attn",
        ]


@dataclass(frozen=True)
class VoiceWindow:
    start: float
    duration: float


def parse_silencedetect_voice_window(
    output: str,
    duration: float,
    *,
    leading_guard: float = 0.10,
    trailing_guard: float = 0.15,
) -> VoiceWindow | None:
    silences: list[tuple[float, float]] = []
    active_start: float | None = None
    for line in output.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            active_start = float(start_match.group(1))
        end_match = SILENCE_END_RE.search(line)
        if end_match:
            silences.append((active_start if active_start is not None else 0.0, float(end_match.group(1))))
            active_start = None
    if active_start is not None:
        silences.append((active_start, duration))
    voice_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in sorted(silences):
        if silence_start > cursor:
            voice_ranges.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        voice_ranges.append((cursor, duration))
    stable = [(start, end) for start, end in voice_ranges if end - start >= 0.10]
    if not stable:
        return None
    start = max(0.0, stable[0][0] - leading_guard)
    end = min(duration, stable[-1][1] + trailing_guard)
    return VoiceWindow(start, max(0.0, end - start))


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


def _load_json_url(url: str, *, timeout: float) -> Any:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _request_json_url(
    url: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    timeout: float,
    method: str = "GET",
) -> Any:
    import urllib.request

    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _nuc_ssh_base_command() -> list[str]:
    return ["ssh", "-p", str(NUC_SSH_PORT), f"{NUC_SSH_USER}@{NUC_OLLAMA_HOST}"]


def _nuc_scp_base_command() -> list[str]:
    return ["scp", "-P", str(NUC_SSH_PORT)]


def _local_audio_cache_key(source: str) -> str:
    path = Path(source).expanduser().resolve()
    stat = path.stat()
    return cache_slug(str(path), stat.st_size, int(stat.st_mtime))


def local_audio_cache_dir_for_source(source: str) -> Path:
    return LOCAL_AUDIO_CACHE_DIR / _local_audio_cache_key(source)


def url_audio_cache_dir(source: str) -> Path:
    return LOCAL_AUDIO_CACHE_DIR / audio_cache_key_for_url(source)


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        children = list(path.iterdir())
    except OSError:
        return 0
    for child in children:
        total += _path_size(child)
    return total


def prune_local_audio_cache(
    *,
    max_bytes: int = LOCAL_AUDIO_CACHE_MAX_BYTES,
    keep: Path | None = None,
) -> list[Path]:
    if not LOCAL_AUDIO_CACHE_DIR.exists():
        return []
    entries = [path for path in LOCAL_AUDIO_CACHE_DIR.iterdir() if path.is_dir()]
    sizes = {path: _path_size(path) for path in entries}
    total = sum(sizes.values())
    if total <= max_bytes:
        return []
    keep_resolved = keep.resolve() if keep is not None else None
    candidates = []
    for path in entries:
        if keep_resolved is not None and path.resolve() == keep_resolved:
            continue
        try:
            last_used = path.stat().st_mtime
        except OSError:
            last_used = 0.0
        candidates.append((last_used, path))
    removed: list[Path] = []
    for _last_used, path in sorted(candidates, key=lambda item: item[0]):
        shutil.rmtree(path, ignore_errors=True)
        total -= sizes[path]
        removed.append(path)
        if total <= max_bytes:
            break
    return removed


def prepare_url_audio_cache(
    source: str,
    *,
    run_command: Any,
    status_signal: Signal,
) -> Path:
    cache_dir = url_audio_cache_dir(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav = cache_dir / "audio-16k-mono.wav"
    meta_path = cache_dir / "metadata.json"
    if wav.exists() and wav.stat().st_size > 0:
        now = time.time()
        os.utime(cache_dir, (now, now))
        status_signal.emit(f"Reusing URL audio cache: {wav}")
        return wav

    token = uuid.uuid4().hex
    downloaded_template = cache_dir / f".download-{token}.%(ext)s"
    temp_wav = cache_dir / f".audio-{token}.wav"
    try:
        run_command(
            [
                YT_DLP,
                "-x",
                "--audio-format",
                "wav",
                "--cookies-from-browser",
                "chrome",
                "-o",
                str(downloaded_template),
                source,
            ],
            "Downloading URL audio cache",
        )
        candidates = sorted(cache_dir.glob(f".download-{token}.*"))
        audio = next(
            (path for path in candidates if path.suffix.lower() in {".wav", ".m4a", ".mp3", ".opus"}),
            None,
        )
        if audio is None:
            raise RuntimeError("yt-dlp did not produce an audio file")
        run_command(
            [
                FFMPEG,
                "-hide_banner",
                "-y",
                "-i",
                str(audio),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(temp_wav),
            ],
            "Preparing URL audio cache",
        )
        temp_wav.replace(wav)
        _write_json_local(
            meta_path,
            {
                "kind": "url",
                "source": source,
                "identity": canonical_media_url(source),
                "wav": str(wav),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        now = time.time()
        os.utime(cache_dir, (now, now))
        removed = prune_local_audio_cache(keep=cache_dir)
        if removed:
            status_signal.emit(
                f"Local audio cache exceeded 2 GiB; removed {len(removed)} oldest cache item(s)"
            )
        return wav
    finally:
        _safe_unlink(temp_wav)
        for candidate in cache_dir.glob(f".download-{token}.*"):
            _safe_unlink(candidate)


def controlled_cache_dir(
    url: str,
    backend: str,
    model_name: str,
    chunk_seconds: int | float,
) -> Path:
    canonical = canonical_media_url(url)
    readable = CACHE_DIR / controlled_cache_dir_name(
        canonical,
        backend,
        model_name,
        chunk_seconds,
    )
    legacy = CACHE_DIR / cache_slug(canonical, backend, model_name, chunk_seconds)
    if legacy.exists() and not readable.exists():
        legacy.rename(readable)
    return readable


def _write_json_local(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _segments_from_verbose_result(data: dict[str, Any]) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    for seg in data.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if text:
            segments.append(SubtitleSegment(float(seg["start"]), float(seg["end"]), text))
    return segments


def _subchunk_label(chunk_index: int, part_index: int) -> str:
    return f"{chunk_index}.part{part_index}"


def _stream_process_output(
    proc: subprocess.Popen[str],
    *,
    status_signal: Signal,
    stop_flag: callable,
    stop_message: str,
) -> list[str]:
    assert proc.stdout
    output_lines: list[str] = []
    output_state: dict[str, object] = {}
    for line in proc.stdout:
        if stop_flag():
            _terminate_process(proc)
            raise RuntimeError(stop_message)
        if line.strip():
            output_lines.append(line.rstrip())
        emit_throttled_process_output(status_signal, line, output_state)
    return output_lines


def _should_retry_yt_dlp_cookie_read(cmd: list[str], output_lines: list[str]) -> bool:
    if "--cookies-from-browser" not in cmd:
        return False
    if not cmd or "yt-dlp" not in Path(cmd[0]).name:
        return False
    output = "\n".join(output_lines).lower()
    return "sign in to confirm you" in output and "not a bot" in output


def _probe_audio_duration(audio_path: Path) -> float:
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
        cwd=str(OUTPUT_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not determine audio duration: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not determine audio duration: {result.stdout.strip()}") from exc


def qwen3_asr_mode(mode: CaptionMode) -> bool:
    return mode.key in {"qwen3_asr_06b_4bit_mlx", "qwen3_asr_1p7b_8bit_mlx"}


def nuc_qwen3_asr_1p7b_mode(mode: CaptionMode) -> bool:
    return mode.backend == "nuc_qwen3_asr_1p7b"


def _nuc_asr_model_for_mode(mode: CaptionMode) -> str:
    if mode.key == "nuc_asr_turbo":
        return "deepdml/faster-whisper-large-v3-turbo-ct2"
    return "large-v3"


def remote_asr_quality_issue(segments: list[SubtitleSegment]) -> str | None:
    texts = [" ".join(segment.text.split()) for segment in segments if segment.text.strip()]
    if len(texts) < 20:
        return None

    counts = Counter(texts)
    most_common_text, most_common_count = counts.most_common(1)[0]
    repeated_share = most_common_count / len(texts)
    unique_share = len(counts) / len(texts)
    if most_common_count >= 8 and repeated_share >= 0.30:
        return (
            f"one phrase occupies {repeated_share:.0%} of {len(texts)} segments "
            f"({most_common_text[:60]!r})"
        )
    if len(texts) >= 40 and unique_share <= 0.12:
        return (
            f"only {len(counts)} unique texts across {len(texts)} segments "
            f"({unique_share:.0%} unique)"
        )
    return None


def validate_remote_asr_segments(segments: list[SubtitleSegment]) -> None:
    issue = remote_asr_quality_issue(segments)
    if issue:
        raise RuntimeError(
            "NUC ASR returned a likely repetition hallucination; subtitle was not saved: "
            f"{issue}"
        )


def _transcribe_via_nuc_asr_result(
    audio_path: Path,
    base_url: str = "",
    model: str = "large-v3",
    language: str = NUC_ASR_AUTO_LANGUAGE,
    response_format: str = "verbose_json",
    vad_filter: bool = False,
    timeout: int = 120,
    status_signal: Signal | None = None,
    heartbeat_interval: float = 10.0,
) -> ASRResult:
    """Send audio to NUC faster-whisper and retain word-level response data."""
    import urllib.request
    import uuid

    if not base_url:
        base_url = f"http://{NUC_OLLAMA_HOST}:8000"
    url = f"{base_url}/v1/audio/transcriptions"
    busy_url = f"{base_url}/busy"

    boundary = uuid.uuid4().hex
    audio_data = audio_path.read_bytes()
    filename = audio_path.name

    body_parts = []
    for field_name, field_value in [
        ("model", model),
        ("language", language),
        ("response_format", response_format),
        ("vad_filter", "true" if vad_filter else "false"),
        ("timestamp_granularities[]", "word"),
    ]:
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
            f"{field_value}\r\n"
        )
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    )
    body = b""
    for part in body_parts:
        body += part.encode("utf-8")
    body += audio_data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    stop_heartbeat = threading.Event()

    def emit_heartbeat() -> None:
        next_emit = 0.0
        while not stop_heartbeat.wait(heartbeat_interval):
            if time.monotonic() < next_emit:
                continue
            try:
                with urllib.request.urlopen(busy_url, timeout=5) as heartbeat_resp:
                    heartbeat = json.loads(heartbeat_resp.read().decode("utf-8"))
            except Exception as exc:
                if status_signal:
                    status_signal.emit(f"NUC ASR heartbeat unavailable: {exc}")
                next_emit = time.monotonic() + heartbeat_interval
                continue
            if not status_signal:
                continue
            current = heartbeat.get("current_request") or {}
            elapsed = current.get("elapsed_seconds")
            heartbeat_filename = current.get("filename") or filename
            if isinstance(elapsed, (int, float)):
                status_signal.emit(
                    f"NUC ASR heartbeat: busy={heartbeat.get('busy')} "
                    f"active={heartbeat.get('active_requests')} elapsed={elapsed:.0f}s file={heartbeat_filename}"
                )
            else:
                status_signal.emit(
                    f"NUC ASR heartbeat: busy={heartbeat.get('busy')} "
                    f"active={heartbeat.get('active_requests')}"
                )
            next_emit = time.monotonic() + heartbeat_interval

    heartbeat_thread: threading.Thread | None = None
    if status_signal:
        heartbeat_thread = threading.Thread(target=emit_heartbeat, daemon=True)
        heartbeat_thread.start()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    finally:
        stop_heartbeat.set()
        if heartbeat_thread:
            heartbeat_thread.join(timeout=1)

    if response_format == "verbose_json":
        return parse_verbose_asr_response(data, requested_words=True)
    text = data.get("text", "").strip()
    duration = float(data.get("duration", 30.0))
    segments = [SubtitleSegment(0.0, duration, text)] if text else []
    return ASRResult(
        language=str(data.get("language") or ""),
        words=[],
        segments=segments,
        diagnostics={
            "capability_warnings": ["non-verbose ASR response has no word timestamps"],
            "upstream_response": data,
        },
    )


def _transcribe_via_nuc_asr(
    audio_path: Path,
    base_url: str = "",
    model: str = "large-v3",
    language: str = NUC_ASR_AUTO_LANGUAGE,
    response_format: str = "verbose_json",
    vad_filter: bool = False,
    timeout: int = 120,
    status_signal: Signal | None = None,
    heartbeat_interval: float = 10.0,
) -> list[SubtitleSegment]:
    return _transcribe_via_nuc_asr_result(
        audio_path,
        base_url=base_url,
        model=model,
        language=language,
        response_format=response_format,
        vad_filter=vad_filter,
        timeout=timeout,
        status_signal=status_signal,
        heartbeat_interval=heartbeat_interval,
    ).segments


def _transcribe_via_nuc_qwen3_asr_1p7b(
    audio_path: Path,
    base_url: str = "",
    language: str = "zh",
    response_format: str = "verbose_json",
    timeout: int = 900,
) -> list[SubtitleSegment]:
    """Send an audio file to the NUC Qwen3-ASR 1.7B proxy and return pseudo-timestamped segments."""
    import urllib.request
    import uuid

    if not base_url:
        base_url = f"http://{NUC_OLLAMA_HOST}:8001"
    url = f"{base_url}/v1/audio/transcriptions"

    boundary = uuid.uuid4().hex
    audio_data = audio_path.read_bytes()
    filename = audio_path.name

    body_parts = []
    for field_name, field_value in [
        ("model", "qwen3-asr-1p7b"),
        ("language", language),
        ("response_format", response_format),
    ]:
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
            f"{field_value}\r\n"
        )
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    )
    body = b""
    for part in body_parts:
        body += part.encode("utf-8")
    body += audio_data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    segments: list[SubtitleSegment] = []
    if response_format == "verbose_json":
        for seg in data.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(SubtitleSegment(seg["start"], seg["end"], text))
    else:
        text = data.get("text", "").strip()
        if text:
            duration = data.get("duration", 30.0)
            segments.append(SubtitleSegment(0.0, duration, text))
    return segments


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
                cwd=str(OUTPUT_DIR),
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


class NUCRealtimeWorker(QObject):
    """Capture Loopback audio in rolling 3s chunks and transcribe via NUC CUDA."""

    caption = Signal(str)
    status = Signal(str)
    finished = Signal()
    session_saved = Signal(str)
    segments_updated = Signal(list)

    def __init__(
        self,
        base_url: str,
        capture_id: int = 0,
        chunk_seconds: float = 3.0,
        save_audio: bool = True,
        session_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.capture_id = capture_id
        self.chunk_seconds = chunk_seconds
        self.save_audio = save_audio
        self.session_dir = session_dir
        self._stop = False
        self._recording_proc: Optional[subprocess.Popen[str]] = None
        self._all_segments: list[SubtitleSegment] = []

    def run(self) -> None:
        import threading

        if self.session_dir is None:
            session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session_dir = REALTIME_DIR / session_id

        self.session_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = self.session_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        self.status.emit(
            f"NUC realtime: capturing Loopback :{self.capture_id}, "
            f"{self.chunk_seconds}s chunks → {self.base_url}"
        )

        # Quick connectivity check
        try:
            import urllib.request
            test_req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            urllib.request.urlopen(test_req, timeout=3)
        except Exception as exc:
            self.status.emit(f"NUC ASR 不可达：{exc}")
            self.caption.emit("NUC ASR 不可达，请检查 NUC 是否在线")
            self.finished.emit()
            return

        chunk_index = 0
        transcribe_thread: Optional[threading.Thread] = None
        transcribe_result: list[str] = []

        def _transcribe_chunk(wav_path: Path, result_list: list[str], current_index: int) -> None:
            """Run in background thread to overlap with next recording."""
            try:
                segments = _transcribe_via_nuc_asr(
                    wav_path,
                    base_url=self.base_url,
                    timeout=int(self.chunk_seconds * 10),
                )
                
                # Shift timestamps and accumulate
                offset = current_index * self.chunk_seconds
                shifted_segments = []
                for seg in segments:
                    shifted_seg = SubtitleSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text
                    )
                    shifted_segments.append(shifted_seg)
                
                if shifted_segments:
                    self._all_segments.extend(shifted_segments)
                    self.segments_updated.emit(self._all_segments.copy())

                text = " ".join(seg.text for seg in segments).strip()
                result_list.append(text)
            except Exception as exc:
                result_list.append(f"[NUC ASR error: {exc}]")
            finally:
                if not self.save_audio:
                    _safe_unlink(wav_path)

        try:
            while not self._stop:
                # Record one chunk
                if self.save_audio:
                    chunk_wav = audio_dir / f"{chunk_index:03d}.wav"
                else:
                    chunk_wav = Path(tempfile.gettempdir()) / f"nuc-rt-chunk-{chunk_index}.wav"
                    
                record_ok = self._record_chunk(chunk_wav)

                if self._stop:
                    if not self.save_audio:
                        _safe_unlink(chunk_wav)
                    break

                # Collect previous transcription result
                if transcribe_thread is not None:
                    transcribe_thread.join(timeout=self.chunk_seconds * 5)
                    if transcribe_result:
                        text = transcribe_result[-1]
                        if text and not text.startswith("[NUC ASR error"):
                            self.caption.emit(text)
                        elif text:
                            self.status.emit(text)
                    transcribe_result.clear()

                if not record_ok:
                    if not self.save_audio:
                        _safe_unlink(chunk_wav)
                    self.status.emit("Recording failed, retrying...")
                    time.sleep(0.5)
                    continue

                # Start transcription in background
                transcribe_thread = threading.Thread(
                    target=_transcribe_chunk,
                    args=(chunk_wav, transcribe_result, chunk_index),
                    daemon=True,
                )
                transcribe_thread.start()
                chunk_index += 1

            # Collect final result
            if transcribe_thread is not None:
                transcribe_thread.join(timeout=10)
                if transcribe_result:
                    text = transcribe_result[-1]
                    if text and not text.startswith("[NUC ASR error"):
                        self.caption.emit(text)

        except Exception as exc:
            self.status.emit(f"NUC realtime error: {exc}")
        finally:
            self.status.emit("NUC realtime stopped")
            self._save_session_artifacts(chunk_index)
            self.finished.emit()

    def _save_session_artifacts(self, num_chunks: int) -> None:
        """Save segments, transcript, and manifest. Concat audio if saved."""
        if not self.session_dir or not self.session_dir.exists():
            return
            
        try:
            if self._all_segments:
                save_segments(self.session_dir / "raw-segments.json", self._all_segments)
                save_segments_as_txt(self.session_dir / "transcript-raw.txt", self._all_segments)
                save_segments_as_srt(self.session_dir / "transcript.srt", self._all_segments)
            
            manifest = {
                "start_time": self.session_dir.name,
                "duration_seconds": num_chunks * self.chunk_seconds,
                "num_chunks": num_chunks,
                "chunk_seconds": self.chunk_seconds,
                "model": "nuc_asr_large_v3",
                "has_audio": self.save_audio,
                "num_segments": len(self._all_segments),
            }
            with open(self.session_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                
            if self.save_audio and num_chunks > 0:
                self._concat_audio_chunks(num_chunks)
                
            self.session_saved.emit(str(self.session_dir))
        except Exception as exc:
            self.status.emit(f"Error saving session artifacts: {exc}")

    def _concat_audio_chunks(self, num_chunks: int) -> None:
        """Concatenate individual chunks into full-audio.wav."""
        audio_dir = self.session_dir / "audio"
        list_file = self.session_dir / "concat_list.txt"
        
        try:
            with open(list_file, "w", encoding="utf-8") as f:
                for i in range(num_chunks):
                    wav_path = audio_dir / f"{i:03d}.wav"
                    if wav_path.exists():
                        f.write(f"file '{wav_path.absolute()}'\n")
                        
            out_file = self.session_dir / "full-audio.wav"
            cmd = [
                FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(out_file)
            ]
            subprocess.run(cmd, check=True, cwd=str(OUTPUT_DIR))
        except Exception as exc:
            self.status.emit(f"Error concatenating audio: {exc}")
        finally:
            _safe_unlink(list_file)

    def stop(self) -> None:
        self._stop = True
        if self._recording_proc and self._recording_proc.poll() is None:
            self._recording_proc.send_signal(signal.SIGTERM)
            try:
                self._recording_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._recording_proc.kill()
        self._recording_proc = None

    def _record_chunk(self, output_path: Path) -> bool:
        """Record one chunk from Loopback via ffmpeg AVFoundation."""
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "avfoundation",
            "-i", f":{self.capture_id}",
            "-t", str(self.chunk_seconds),
            "-ac", "1",
            "-ar", "16000",
            "-y",
            str(output_path),
        ]
        try:
            self._recording_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(OUTPUT_DIR)
            )
            _, stderr = self._recording_proc.communicate(
                timeout=self.chunk_seconds + 5
            )
            ok = self._recording_proc.returncode == 0 and output_path.exists()
            if not ok and stderr:
                self.status.emit(f"ffmpeg capture error: {stderr.strip()[:200]}")
            self._recording_proc = None
            return ok
        except subprocess.TimeoutExpired:
            if self._recording_proc:
                self._recording_proc.kill()
            self._recording_proc = None
            return False
        except Exception:
            self._recording_proc = None
            return False


class RealtimePolishWorker(QObject):
    """Batch polish segments from a realtime session using LLM."""

    status = Signal(str)
    polished_segments = Signal(list)
    finished = Signal()

    def __init__(
        self,
        session_dir: Path,
        segments: list[SubtitleSegment],
        provider: LLMProvider,
        api_key: str,
        api_url_override: str = "",
        model_id_override: str = "",
        batch_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        self.session_dir = session_dir
        self.segments = segments
        self.provider = provider
        self.api_key = api_key
        self.api_url_override = api_url_override
        self.model_id_override = model_id_override
        self.batch_seconds = batch_seconds
        self._stop = False

    def run(self) -> None:
        if not self.segments:
            self.finished.emit()
            return

        self.status.emit(f"开始 LLM 规整... ({len(self.segments)} 段)")
        polished: list[SubtitleSegment] = []
        batch: list[SubtitleSegment] = []
        batch_start_time = self.segments[0].start

        try:
            for seg in self.segments:
                if self._stop:
                    break
                batch.append(seg)
                if seg.end - batch_start_time >= self.batch_seconds:
                    self._process_batch(batch, polished)
                    batch.clear()
                    if self._stop:
                        break
                    batch_start_time = seg.end

            if batch and not self._stop:
                self._process_batch(batch, polished)

            if not self._stop:
                if polished:
                    # Save results
                    save_segments(self.session_dir / "polished-segments.json", polished)
                    save_segments_as_txt(self.session_dir / "transcript-polished.txt", polished)
                self.polished_segments.emit(polished)
                self.status.emit("LLM 规整完成")

        except Exception as exc:
            self.status.emit(f"LLM 规整错误: {exc}")
        finally:
            self.finished.emit()

    def _process_batch(self, batch: list[SubtitleSegment], polished_list: list[SubtitleSegment]) -> None:
        self.status.emit(f"规整进度: {len(polished_list)} / {len(self.segments)}...")
        corrected = llm_proofread(
            batch,
            self.provider,
            self.api_key,
            self.api_url_override,
            self.model_id_override,
            timeout=30,
        )
        polished_list.extend(corrected)

    def stop(self) -> None:
        self._stop = True


class RealtimeReRecognizeWorker(QObject):
    """Re-recognize full audio from a realtime session and optionally polish."""

    status = Signal(str)
    result_segments = Signal(list)
    finished = Signal()

    def __init__(
        self,
        session_dir: Path,
        base_url: str,
        provider: Optional[LLMProvider] = None,
        api_key: str = "",
        api_url_override: str = "",
        model_id_override: str = "",
    ) -> None:
        super().__init__()
        self.session_dir = session_dir
        self.base_url = base_url
        self.provider = provider
        self.api_key = api_key
        self.api_url_override = api_url_override
        self.model_id_override = model_id_override

    def run(self) -> None:
        audio_file = self.session_dir / "full-audio.wav"
        if not audio_file.exists():
            self.status.emit("找不到完整的会话音频。")
            self.finished.emit()
            return

        try:
            self.status.emit("开始重新识别全量音频...")
            segments = _transcribe_via_nuc_asr(
                audio_file, base_url=self.base_url, timeout=600
            )
            validate_remote_asr_segments(segments)
            
            # Save new raw segments
            save_segments(self.session_dir / "raw-segments.json", segments)
            save_segments_as_txt(self.session_dir / "transcript-raw.txt", segments)
            save_segments_as_srt(self.session_dir / "transcript.srt", segments)
            
            # Update manifest segment count
            manifest_file = self.session_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest["num_segments"] = len(segments)
                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)

            if self.provider:
                self.status.emit("开始重新 LLM 规整...")
                # Polish the entire new segments
                # For long sessions, might need batching, but let's reuse RealtimePolishWorker logic manually
                polished: list[SubtitleSegment] = []
                batch: list[SubtitleSegment] = []
                batch_start_time = segments[0].start if segments else 0
                
                for seg in segments:
                    batch.append(seg)
                    if seg.end - batch_start_time >= 30.0:
                        corrected = llm_proofread(
                            batch, self.provider, self.api_key,
                            self.api_url_override, self.model_id_override, timeout=30
                        )
                        polished.extend(corrected)
                        batch.clear()
                        batch_start_time = seg.end
                        
                if batch:
                    corrected = llm_proofread(
                        batch, self.provider, self.api_key,
                        self.api_url_override, self.model_id_override, timeout=30
                    )
                    polished.extend(corrected)
                    
                save_segments(self.session_dir / "polished-segments.json", polished)
                save_segments_as_txt(self.session_dir / "transcript-polished.txt", polished)
                self.result_segments.emit(polished)
                self.status.emit("重识别 + 规整完成")
            else:
                self.result_segments.emit(segments)
                self.status.emit("重新识别完成")
                
        except Exception as exc:
            self.status.emit(f"重识别错误: {exc}")
        finally:
            self.finished.emit()


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
    chunk_progress = Signal(object)
    finished = Signal()

    def __init__(
        self,
        items: list[str],
        mode: CaptionMode,
        config: QueueRunConfig | None = None,
    ) -> None:
        super().__init__()
        self.items = items
        self.mode = mode
        self.config = config or QueueRunConfig.from_environment()
        self._stop = False
        self.proc: Optional[subprocess.Popen[str]] = None
        self._temp_paths: list[Path] = []
        self._process_lock = threading.RLock()
        self._active_processes: set[subprocess.Popen[str]] = set()
        self._used_nuc_asr = False
        self.history = ASRHistoryStore()

    def run(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for source in self.items:
                if self._stop:
                    break
                ok = self._process(source)
                self.finished_item.emit(source, ok)
        finally:
            if self._used_nuc_asr:
                self._release_nuc_asr()
            self.finished.emit()

    def stop(self) -> None:
        self._stop = True
        with self._process_lock:
            active = list(self._active_processes)
        for proc in active:
            _terminate_process(proc)
        self.proc = None

    def _register_process(self, proc: subprocess.Popen[str]) -> None:
        with self._process_lock:
            self._active_processes.add(proc)
            self.proc = proc

    def _unregister_process(self, proc: subprocess.Popen[str]) -> None:
        with self._process_lock:
            self._active_processes.discard(proc)
            if self.proc is proc:
                self.proc = next(iter(self._active_processes), None)

    def _release_nuc_asr(self) -> None:
        try:
            result = _request_json_url(
                f"{self.mode.model}/release/asr",
                timeout=10,
                method="POST",
            )
            self.status.emit(f"NUC ASR release: {result.get('status', result)}")
        except Exception as exc:
            self.status.emit(f"NUC ASR release unavailable: {exc}")

    def _track_temp_path(self, path: Path) -> Path:
        self._temp_paths.append(path)
        return path

    def _prepare_local_audio_cache(self, source: str) -> Path:
        source_path = Path(source).expanduser().resolve()
        cache_dir = local_audio_cache_dir_for_source(source)
        cache_dir.mkdir(parents=True, exist_ok=True)
        wav = cache_dir / "audio-16k-mono.wav"
        meta_path = cache_dir / "metadata.json"
        source_stat = source_path.stat()
        metadata = {
            "source": str(source_path),
            "size": source_stat.st_size,
            "mtime": int(source_stat.st_mtime),
            "wav": str(wav),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if wav.exists():
            self.status.emit(f"Reusing local audio cache: {wav}")
            _write_json_local(meta_path, metadata)
            return wav
        self._run(
            [
                FFMPEG,
                "-hide_banner",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav),
            ],
            "Preparing audio",
        )
        _write_json_local(meta_path, metadata)
        return wav

    def _prepared_wav(self, source: str) -> Path | None:
        prepared = (self.config.prepared_wavs or {}).get(source)
        if not prepared:
            return None
        path = Path(prepared).expanduser()
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Prepared WAV no longer exists: {path}")
        self.status.emit(f"Reusing prepared WAV: {path}")
        return path

    def _transcribe_local_file_via_nuc_job(
        self,
        wav: Path,
        *,
        base_url: str,
        model: str = "large-v3",
    ) -> list[SubtitleSegment]:
        import urllib.request

        self.status.emit(f"Uploading cached WAV to NUC via HTTP: {wav.name}")
        boundary = uuid.uuid4().hex
        audio_data = wav.read_bytes()
        body_parts = []
        for field_name, field_value in [
            ("model", model),
            ("language", NUC_ASR_AUTO_LANGUAGE),
            ("response_format", "verbose_json"),
        ]:
            body_parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
                f"{field_value}\r\n"
            )
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        )
        body = b"".join(part.encode("utf-8") for part in body_parts) + audio_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/jobs/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            task = json.loads(resp.read().decode("utf-8"))
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"NUC local-file upload job did not return task id: {task}")
        return self._wait_for_nuc_job(task_id, base_url=base_url, filename=wav.name, busy_endpoint="/busy")

    def _transcribe_file_via_nuc_chunks(
        self,
        wav: Path,
        *,
        base_url: str,
        model: str,
        chunk_seconds: float = 60.0,
    ) -> list[SubtitleSegment]:
        duration = _probe_audio_duration(wav)
        all_segments: list[SubtitleSegment] = []
        chunk_count = max(1, math.ceil(duration / chunk_seconds))
        with tempfile.TemporaryDirectory(prefix="whisper-captioner-nuc-asr-") as directory:
            chunk_dir = Path(directory)
            for chunk_index in range(chunk_count):
                if self._stop:
                    raise RuntimeError("Queue stopped")
                start = chunk_index * chunk_seconds
                remaining = min(chunk_seconds, duration - start)
                chunk_wav = chunk_dir / f"chunk-{chunk_index:04d}.wav"
                self.status.emit(
                    f"Preparing NUC ASR chunk {chunk_index + 1}/{chunk_count} "
                    f"({start:.0f}-{start + remaining:.0f}s)"
                )
                self._run(
                    [
                        FFMPEG,
                        "-y",
                        "-ss",
                        f"{start:.3f}",
                        "-t",
                        f"{remaining:.3f}",
                        "-i",
                        str(wav),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(chunk_wav),
                    ],
                    f"Preparing NUC ASR chunk {chunk_index + 1}/{chunk_count}",
                )
                voice_window = self._detect_voice_window(
                    chunk_wav,
                    remaining,
                    f"NUC-{chunk_index + 1}",
                )
                if voice_window is None:
                    self.status.emit(
                        f"NUC ASR chunk {chunk_index + 1}/{chunk_count}: "
                        "no stable voice window detected"
                    )
                    continue
                chunk_segments = _transcribe_via_nuc_asr(
                    chunk_wav,
                    base_url=base_url,
                    model=model,
                    timeout=max(180, int(remaining * 4)),
                    status_signal=self.status,
                )
                issue = remote_asr_quality_issue(chunk_segments)
                if issue:
                    self.status.emit(
                        f"NUC ASR chunk {chunk_index + 1}/{chunk_count} "
                        f"looked unstable ({issue}); retrying with VAD"
                    )
                    chunk_segments = _transcribe_via_nuc_asr(
                        chunk_wav,
                        base_url=base_url,
                        model=model,
                        vad_filter=True,
                        timeout=max(180, int(remaining * 4)),
                        status_signal=self.status,
                    )
                    validate_remote_asr_segments(chunk_segments)
                for segment_index, segment in enumerate(chunk_segments):
                    local_start = max(0.0, min(remaining, segment.start))
                    local_end = max(0.0, min(remaining, segment.end))
                    if (
                        segment_index == 0
                        and local_start <= 0.1
                        and voice_window.start >= 2.0
                    ):
                        local_start = min(local_end, voice_window.start)
                    if segment.text.strip() and local_end > local_start:
                        all_segments.append(
                            SubtitleSegment(
                                local_start + start,
                                local_end + start,
                                segment.text,
                            )
                        )
                self.status.emit(
                    f"NUC ASR chunk {chunk_index + 1}/{chunk_count}: "
                    f"{len(chunk_segments)} segment(s)"
                )
        validate_remote_asr_segments(all_segments)
        return all_segments

    def _transcribe_local_file_via_nuc_qwen_job(self, wav: Path, *, base_url: str) -> list[SubtitleSegment]:
        import urllib.request

        self.status.emit(f"Uploading cached WAV to NUC Qwen ASR via HTTP: {wav.name}")
        boundary = uuid.uuid4().hex
        audio_data = wav.read_bytes()
        body_parts = []
        for field_name, field_value in [
            ("model", "qwen3-asr-1p7b"),
            ("language", "zh"),
            ("response_format", "verbose_json"),
        ]:
            body_parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
                f"{field_value}\r\n"
            )
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        )
        body = b"".join(part.encode("utf-8") for part in body_parts) + audio_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/jobs/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            task = json.loads(resp.read().decode("utf-8"))
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"NUC Qwen upload job did not return task id: {task}")
        return self._wait_for_nuc_job(task_id, base_url=base_url, filename=wav.name, busy_endpoint=None)

    def _wait_for_nuc_job(
        self,
        task_id: str,
        *,
        base_url: str,
        filename: str,
        busy_endpoint: str | None,
    ) -> list[SubtitleSegment]:
        started = time.monotonic()
        last_status = ""
        while not self._stop:
            task = _load_json_url(f"{base_url}/jobs/{task_id}", timeout=20)
            status = str(task.get("status") or "unknown")
            result_dir = task.get("result_dir")
            if status != last_status:
                if result_dir:
                    self.status.emit(f"NUC ASR job {task_id}: {status} ({result_dir})")
                else:
                    self.status.emit(f"NUC ASR job {task_id}: {status}")
                last_status = status
            if status == "completed":
                result = task.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"NUC ASR job missing result payload: {task}")
                self.status.emit(
                    f"NUC ASR job completed in {time.monotonic() - started:.0f}s; result dir: {result.get('nuc_result_dir', result_dir)}"
                )
                return _segments_from_verbose_result(result)
            if status == "failed":
                raise RuntimeError(f"NUC ASR job failed: {task.get('error')}")
            time.sleep(5)
            if not busy_endpoint:
                self.status.emit(
                    f"NUC ASR job {task_id}: still {status} after {time.monotonic() - started:.0f}s"
                )
                continue
            try:
                busy = _load_json_url(f"{base_url}{busy_endpoint}", timeout=10)
            except Exception as exc:
                self.status.emit(f"NUC ASR heartbeat unavailable: {exc}")
                continue
            current = busy.get("current_request") or {}
            elapsed = current.get("elapsed_seconds")
            remote_name = current.get("filename") or filename
            if isinstance(elapsed, (int, float)):
                self.status.emit(
                    f"NUC ASR heartbeat: busy={busy.get('busy')} active={busy.get('active_requests')} elapsed={elapsed:.0f}s file={remote_name}"
                )
        raise RuntimeError("Queue stopped")

    def _process(self, source: str) -> bool:
        if not self.mode.available:
            self.status.emit(f"Missing model: {self.mode.model}")
            return False

        source_title = infer_source_title(source)
        safe_name = clean_title_for_filename(source_title)
        output_dir = source_output_dir(GENERATED_DIR, source_title)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = output_dir / f"{safe_name}-{stamp}"
        self._temp_paths = []
        wav = self._track_temp_path(Path(tempfile.gettempdir()) / f"whisper-captioner-{stamp}.wav")
        self.history.upsert(
            source,
            title=source_title,
            last_mode_key=self.mode.key,
            last_mode_label=self.mode.label,
            status="running",
        )

        try:
            prepared_wav = self._prepared_wav(source)
            if prepared_wav:
                wav = prepared_wav
            elif source.startswith(("http://", "https://")):
                wav = prepare_url_audio_cache(
                    source,
                    run_command=self._run,
                    status_signal=self.status,
                )
            else:
                wav = self._prepare_local_audio_cache(source)

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
            elif self.mode.backend == "nuc_asr":
                self._used_nuc_asr = True
                nuc_asr_model = _nuc_asr_model_for_mode(self.mode)
                self.status.emit(
                    f"Transcribing with NUC remote ASR (faster-whisper CUDA, {nuc_asr_model})..."
                )
                duration = _probe_audio_duration(wav)
                self.status.emit(
                    f"NUC remote ASR will process {duration:.1f}s audio in 60s chunks"
                )
                segments = self._transcribe_file_via_nuc_chunks(
                    wav,
                    base_url=str(self.mode.model),
                    model=nuc_asr_model,
                )
                validate_remote_asr_segments(segments)
                save_segments_as_txt(base.with_suffix(".txt"), segments)
                save_segments_as_srt(base.with_suffix(".srt"), segments)
            elif self.mode.backend == "nuc_qwen3_asr_1p7b":
                self.status.emit("Transcribing with NUC remote Qwen3-ASR 1.7B (high-quality offline)...")
                if source.startswith(("http://", "https://")):
                    segments = self._transcribe_nuc_qwen3_asr_1p7b_chunked(wav, base_url=str(self.mode.model))
                else:
                    segments = self._transcribe_local_file_via_nuc_qwen_job(
                        wav,
                        base_url=str(self.mode.model),
                    )
                save_segments_as_txt(base.with_suffix(".txt"), segments)
                save_segments_as_srt(base.with_suffix(".srt"), segments)
            else:
                self._run(
                    [
                        WHISPER_CLI,
                        "-m", str(self.mode.model),
                        "-f", str(wav),
                        *self.config.cpp_args(),
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
            self.history.upsert(
                source,
                title=source_title,
                audio_cache_key=wav.parent.name if wav.parent.parent == LOCAL_AUDIO_CACHE_DIR else "",
                audio_cache_wav=str(wav) if wav.exists() else "",
                last_mode_key=self.mode.key,
                last_mode_label=self.mode.label,
                output_base=str(base),
                status="ready",
            )
            return True
        except Exception as exc:
            self.status.emit(f"Failed {source}: {exc}")
            self.history.upsert(
                source,
                title=source_title,
                last_mode_key=self.mode.key,
                last_mode_label=self.mode.label,
                status="failed",
            )
            return False
        finally:
            _cleanup_temp_paths(self._temp_paths)

    def _transcribe_local_qwen3_asr_chunked(
        self,
        wav: Path,
        on_chunk_ready: Callable[[dict[str, Any], list[SubtitleSegment]], None] | None = None,
        tasks_override: list[dict[str, Any]] | None = None,
    ) -> list[SubtitleSegment]:
        duration = self._get_duration(wav)
        chunk_seconds = self.config.qwen_chunk_seconds
        tasks = tasks_override or [
            {
                "label": str(index),
                "start": start,
                "duration": min(chunk_seconds, duration - start),
                "root": True,
                "split": False,
                "attempt": 0,
            }
            for index, start in enumerate(
                index * chunk_seconds for index in range(math.ceil(duration / chunk_seconds))
            )
            if start < duration
        ]
        replicas = self.config.qwen_replicas if self.config.qwen_parallel_enabled else 1
        pending = list(tasks)
        futures: dict[Future, dict[str, Any]] = {}
        started: dict[Future, float] = {}
        cancel_events: dict[Future, threading.Event] = {}
        process_holders: dict[Future, dict[str, Any]] = {}
        result_groups: list[tuple[float, str, list[SubtitleSegment]]] = []
        successful_root_times: list[float] = []
        done = 0
        total = len(tasks)
        splits = 0

        def split_task(task: dict[str, Any]) -> None:
            nonlocal total, splits
            half = float(task["duration"]) / 2.0
            pending.extend(
                [
                    {
                        "label": f"{task['label']}a",
                        "start": float(task["start"]),
                        "duration": half,
                        "root": False,
                        "split": False,
                        "attempt": 0,
                    },
                    {
                        "label": f"{task['label']}b",
                        "start": float(task["start"]) + half,
                        "duration": float(task["duration"]) - half,
                        "root": False,
                        "split": False,
                        "attempt": 0,
                    },
                ]
            )
            total += 1
            splits += 1

        def submit(executor: ThreadPoolExecutor, task: dict[str, Any]) -> None:
            cancel_event = threading.Event()
            holder: dict[str, Any] = {}
            future = executor.submit(self._run_qwen_chunk_task, wav, task, cancel_event, holder)
            futures[future] = task
            started[future] = time.monotonic()
            cancel_events[future] = cancel_event
            process_holders[future] = holder

        with ThreadPoolExecutor(max_workers=replicas, thread_name_prefix="qwen-asr") as executor:
            while (pending or futures) and not self._stop:
                while pending and len(futures) < replicas:
                    submit(executor, pending.pop(0))
                threshold = None
                if len(successful_root_times) >= 3:
                    threshold = max(10.0, min(successful_root_times[:3]) * ADAPTIVE_SPLIT_MULTIPLIER)
                if self.config.adaptive_split_enabled and threshold:
                    now = time.monotonic()
                    for future, task in list(futures.items()):
                        if not task["root"] or task["split"] or future.done():
                            continue
                        if now - started[future] <= threshold:
                            continue
                        task["split"] = True
                        cancel_events[future].set()
                        _terminate_process(process_holders[future].get("proc"), timeout=1.0)
                        split_task(task)
                        self.status.emit(
                            f"Adaptive split {task['label']} at {threshold:.1f}s into two child chunks"
                        )
                completed, _ = wait(list(futures), timeout=0.2, return_when=FIRST_COMPLETED)
                for future in completed:
                    task = futures.pop(future)
                    elapsed = time.monotonic() - started.pop(future)
                    cancel_events.pop(future, None)
                    process_holders.pop(future, None)
                    if task["split"]:
                        try:
                            future.result()
                        except Exception:
                            pass
                        continue
                    try:
                        chunk_segments = future.result()
                    except Exception as chunk_error:
                        if self._stop:
                            raise RuntimeError("Queue stopped") from chunk_error
                        if int(task.get("attempt", 0)) < 1:
                            retry_task = dict(task)
                            retry_task["attempt"] = 1
                            self.status.emit(
                                f"Qwen3-ASR chunk {task['label']} failed; retrying once: {chunk_error}"
                            )
                            submit(executor, retry_task)
                            continue
                        if (
                            self.config.adaptive_split_enabled
                            and task.get("root", False)
                            and float(task["duration"]) >= 20.0
                        ):
                            split_task(task)
                            self.status.emit(
                                f"Qwen3-ASR chunk {task['label']} failed after retry; "
                                "falling back to two child chunks"
                            )
                            continue
                        for event in cancel_events.values():
                            event.set()
                        for holder in process_holders.values():
                            _terminate_process(holder.get("proc"), timeout=1.0)
                        pending.clear()
                        for active_future in futures:
                            active_future.cancel()
                        raise RuntimeError(
                            f"Qwen3-ASR chunk {task['label']} failed after retry: {chunk_error}"
                        ) from chunk_error
                    result_groups.append((float(task["start"]), str(task["label"]), chunk_segments))
                    if on_chunk_ready is not None:
                        on_chunk_ready(task, chunk_segments)
                    done += 1
                    if task["root"]:
                        successful_root_times.append(elapsed)
                    self.chunk_progress.emit(
                        {
                            "done": done,
                            "total": total,
                            "finished": done == total,
                            "inflight": len(futures),
                            "splits": splits,
                        }
                    )
        if self._stop:
            raise RuntimeError("Queue stopped")
        return self._merge_qwen_chunk_groups(result_groups)

    def _run_qwen_chunk_task(
        self,
        wav: Path,
        task: dict[str, Any],
        cancel_event: threading.Event,
        holder: dict[str, Any],
    ) -> list[SubtitleSegment]:
        label = task["label"]
        token = uuid.uuid4().hex
        chunk_wav = Path(tempfile.gettempdir()) / f"{wav.stem}-qwen3-{label}-{token}.wav"
        chunk_out = Path(tempfile.gettempdir()) / f"{wav.stem}-qwen3-{label}-{token}"
        with self._process_lock:
            self._temp_paths.extend([chunk_wav, chunk_out.with_suffix(".txt")])
        self._run_chunk_command(
            [
                FFMPEG,
                "-hide_banner",
                "-y",
                "-ss",
                str(task["start"]),
                "-t",
                str(task["duration"]),
                "-i",
                str(wav),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(chunk_wav),
            ],
            f"Preparing Qwen3-ASR chunk {label}",
            cancel_event,
            holder,
        )
        if cancel_event.is_set() or self._stop:
            raise RuntimeError("Chunk cancelled")
        cmd = [
            MLX_AUDIO_STT,
            "--model",
            str(self.mode.model),
            "--audio",
            str(chunk_wav),
            "--output-path",
            str(chunk_out),
            "--format",
            "txt",
            "--language",
            "zh",
            "--chunk-duration",
            str(max(1, int(task["duration"]))),
            "--max-tokens",
            "512",
        ]
        output = self._run_chunk_command(
            cmd,
            f"Transcribing Qwen3-ASR chunk {label}",
            cancel_event,
            holder,
            timeout=max(120.0, float(task["duration"]) * 4.0),
        )
        chunk_txt = chunk_out.with_suffix(".txt")
        chunk_text = (
            chunk_txt.read_text(encoding="utf-8", errors="ignore").strip()
            if chunk_txt.exists()
            else output.strip()
        )
        return [
            SubtitleSegment(
                segment.start + task["start"],
                segment.end + task["start"],
                segment.text,
            )
            for segment in pseudo_timestamp_qwen3_text(chunk_text, task["duration"])
        ]

    def _run_chunk_command(
        self,
        cmd: list[str],
        label: str,
        cancel_event: threading.Event,
        holder: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(OUTPUT_DIR),
        )
        self._register_process(proc)
        holder["proc"] = proc
        output_lines: list[str] = []

        def consume_output() -> None:
            if not proc.stdout:
                return
            for line in proc.stdout:
                output_lines.append(line.rstrip())

        output_thread = threading.Thread(target=consume_output, daemon=True)
        output_thread.start()
        try:
            deadline = time.monotonic() + timeout if timeout is not None else None
            while proc.poll() is None:
                if self._stop or cancel_event.wait(0.1):
                    _terminate_process(proc, timeout=1.0)
                    raise RuntimeError("Chunk cancelled")
                if deadline is not None and time.monotonic() >= deadline:
                    _terminate_process(proc, timeout=1.0)
                    raise TimeoutError(f"{label} timed out after {timeout:.0f}s")
            output_thread.join(timeout=2)
            output = "\n".join(output_lines)
            if proc.returncode != 0:
                raise RuntimeError(f"worker exited with code {proc.returncode}: {output[-1000:]}")
            return output
        finally:
            output_thread.join(timeout=2)
            if proc.stdout:
                proc.stdout.close()
            if holder.get("proc") is proc:
                holder.pop("proc", None)
            self._unregister_process(proc)

    @staticmethod
    def _merge_qwen_chunk_groups(
        groups: list[tuple[float, str, list[SubtitleSegment]]],
    ) -> list[SubtitleSegment]:
        merged: list[SubtitleSegment] = []
        for _start, _label, segments in sorted(groups, key=lambda item: (item[0], item[1])):
            ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
            if merged and ordered:
                previous = merged[-1]
                current = ordered[0]
                if (
                    previous.text.strip() == current.text.strip()
                    and current.start - previous.end <= 1.5
                ):
                    merged[-1] = SubtitleSegment(
                        previous.start,
                        max(previous.end, current.end),
                        previous.text,
                    )
                    ordered = ordered[1:]
            merged.extend(ordered)
        return merged

    def _transcribe_nuc_qwen3_asr_1p7b_chunked(self, wav: Path, base_url: str) -> list[SubtitleSegment]:
        duration = self._get_duration(wav)
        chunk_seconds = 30.0
        overlap_seconds = 2.0
        all_segments: list[SubtitleSegment] = []
        offset = 0.0
        chunk_index = 0
        while offset < duration and not self._stop:
            actual_start = max(0.0, offset - (overlap_seconds if chunk_index > 0 else 0.0))
            leading_trim = offset - actual_start
            remaining = min(chunk_seconds + leading_trim + overlap_seconds, duration - actual_start)
            chunk_wav = self._track_temp_path(Path(tempfile.gettempdir()) / f"{wav.stem}-nuc-qwen3-chunk{chunk_index}.wav")
            self._run(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss", str(actual_start),
                    "-t", str(remaining),
                    "-i", str(wav),
                    "-ac", "1",
                    "-ar", "16000",
                    str(chunk_wav),
                ],
                f"Preparing NUC Qwen3-ASR 1.7B chunk {chunk_index}",
            )
            request_wav = chunk_wav
            vad_offset = 0.0
            request_duration = remaining
            if self.config.remote_vad_enabled:
                vad_result = self._prepare_vad_window(chunk_wav, remaining, str(chunk_index))
                if vad_result is None:
                    self.status.emit(
                        f"Chunk {chunk_index}: no stable voice window detected, skipping NUC Qwen3-ASR request"
                    )
                    offset += chunk_seconds
                    chunk_index += 1
                    continue
                request_wav, vad_offset, request_duration = vad_result
            raw_segments = _transcribe_via_nuc_qwen3_asr_1p7b(
                request_wav, base_url=base_url, timeout=900
            )
            raw_segments = [
                SubtitleSegment(
                    segment.start + vad_offset,
                    segment.end + vad_offset,
                    segment.text,
                )
                for segment in raw_segments
            ]
            trimmed = self._trim_overlap_segments(
                raw_segments,
                leading_trim=leading_trim,
                trailing_trim=overlap_seconds if actual_start + remaining < duration else 0.0,
                chunk_duration=remaining,
            )
            all_segments.extend(
                SubtitleSegment(segment.start + actual_start, segment.end + actual_start, segment.text)
                for segment in trimmed
            )
            offset += chunk_seconds
            chunk_index += 1
        return self._merge_near_duplicate_segments(all_segments)

    def _prepare_vad_window(
        self,
        chunk_wav: Path,
        duration: float,
        label: str,
    ) -> tuple[Path, float, float] | None:
        try:
            window = self._detect_voice_window(chunk_wav, duration, label)
            if window is None:
                return None
            if window.start <= 0.001 and window.duration >= duration - 0.001:
                return chunk_wav, 0.0, duration
            trimmed = self._track_temp_path(
                Path(tempfile.gettempdir()) / f"{chunk_wav.stem}-vad-{uuid.uuid4().hex}.wav"
            )
            self._run(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss",
                    str(window.start),
                    "-t",
                    str(window.duration),
                    "-i",
                    str(chunk_wav),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(trimmed),
                ],
                f"Trimming speech window for chunk {label}",
            )
            self.status.emit(
                f"Chunk {label}: VAD trimmed {window.start:.2f}s leading, "
                f"{max(0.0, duration - window.start - window.duration):.2f}s trailing"
            )
            return trimmed, window.start, window.duration
        except Exception as exc:
            self.status.emit(f"Chunk {label}: VAD failed, using full chunk: {exc}")
            return chunk_wav, 0.0, duration

    def _detect_voice_window(
        self,
        chunk_wav: Path,
        duration: float,
        label: str,
    ) -> VoiceWindow | None:
        output = self._run_capture(
            [
                FFMPEG,
                "-hide_banner",
                "-i",
                str(chunk_wav),
                "-af",
                "silencedetect=noise=-35dB:d=0.3",
                "-f",
                "null",
                "-",
            ],
            f"Detecting speech window for chunk {label}",
        )
        return parse_silencedetect_voice_window(
            output,
            duration,
            leading_guard=0.0,
        )

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
    def _trim_overlap_segments(
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
    def _merge_near_duplicate_segments(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
        if not segments:
            return []
        ordered = sorted(segments, key=lambda item: (item.start, item.end))
        merged: list[SubtitleSegment] = [ordered[0]]
        for segment in ordered[1:]:
            previous = merged[-1]
            if (
                segment.text == previous.text
                and abs(segment.start - previous.start) <= 1.5
                and abs(segment.end - previous.end) <= 1.5
            ):
                merged[-1] = SubtitleSegment(
                    min(previous.start, segment.start),
                    max(previous.end, segment.end),
                    previous.text,
                )
                continue
            merged.append(segment)
        return merged

    @staticmethod
    def _get_duration(audio_path: Path) -> float:
        return _probe_audio_duration(audio_path)

    def _run(self, cmd: list[str], label: str) -> None:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(OUTPUT_DIR)
        )
        self._register_process(proc)
        try:
            output_lines = _stream_process_output(
                proc,
                status_signal=self.status,
                stop_flag=lambda: self._stop,
                stop_message="Queue stopped",
            )
            if proc.wait() != 0:
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
                raise RuntimeError(f"command failed: {cmd[0]}\n\nOutput:\n{error_context}")
        finally:
            self._unregister_process(proc)

    def _run_capture(self, cmd: list[str], label: str) -> str:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(OUTPUT_DIR)
        )
        self._register_process(proc)
        try:
            output_lines = _stream_process_output(
                proc,
                status_signal=self.status,
                stop_flag=lambda: self._stop,
                stop_message="Queue stopped",
            )
            if proc.wait() != 0:
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
                raise RuntimeError(f"command failed: {cmd[0]}\n\nOutput:\n{error_context}")
            return "\n".join(output_lines)
        finally:
            self._unregister_process(proc)


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
    quality_updated = Signal(object)
    gemini_fusion_blocked = Signal(str)  # reason string
    finished = Signal()

    def __init__(
        self, url: str, mode: CaptionMode, chunk_seconds: int = 30,
        llm_provider: Optional[LLMProvider] = None, llm_api_key: str = "",
        llm_api_url: str = "", llm_model_id: str = "",
        remote_vad_enabled: bool | None = None,
        run_config: QueueRunConfig | None = None,
        gemini_fusion_enabled: bool = False,
        gemini_api_key: str = "",
    ) -> None:
        super().__init__()
        self.url = url
        self.mode = mode
        self.chunk_seconds = chunk_seconds
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_api_url = llm_api_url
        self.llm_model_id = llm_model_id
        self.run_config = run_config or QueueRunConfig.from_environment()
        self.remote_vad_enabled = (
            _env_bool("WHISPER_CAPTIONER_REMOTE_VAD", False)
            if remote_vad_enabled is None
            else remote_vad_enabled
        )
        self.gemini_fusion_enabled = gemini_fusion_enabled
        self.gemini_api_key = gemini_api_key
        self._stop = False
        self.proc: Optional[subprocess.Popen[str]] = None
        self._temp_paths: list[Path] = []
        self._used_nuc_asr = False
        self._has_emitted_segments = False
        self._parallel_qwen_worker: QueueWorker | None = None
        self._last_asr_result: ASRResult | None = None
        self._retry_records: list[dict] = []
        explicit_language = next(
            (value for flag, value in zip(self.mode.args, self.mode.args[1:]) if flag == "-l"),
            "auto",
        )
        self._language_pin = LanguagePin(explicit_language=explicit_language)

    def run(self) -> None:
        try:
            self._do_rolling_prefetch()
        except Exception as exc:
            self.status.emit(f"Rolling prefetch failed: {exc}")
        finally:
            if self._used_nuc_asr:
                try:
                    result = _request_json_url(
                        f"{self.mode.model}/release/asr",
                        timeout=10,
                        method="POST",
                    )
                    self.status.emit(f"NUC ASR release: {result.get('status', result)}")
                except Exception as exc:
                    self.status.emit(f"NUC ASR release unavailable: {exc}")
            self.finished.emit()

    def stop(self) -> None:
        self._stop = True
        if self._parallel_qwen_worker is not None:
            self._parallel_qwen_worker.stop()
        _terminate_process(self.proc)
        self.proc = None

    def _track_temp_path(self, path: Path) -> Path:
        self._temp_paths.append(path)
        return path

    def _emit_incremental_segments(self, segments: list[SubtitleSegment]) -> None:
        if not segments:
            return
        if self._has_emitted_segments:
            self.more_segments.emit(segments)
        else:
            self.first_segments.emit(segments)
            self._has_emitted_segments = True

    def _transcribe_local_qwen_parallel(
        self,
        audio: Path,
        job_cache_dir: Path,
        duration: float,
    ) -> bool:
        chunk_seconds = self.run_config.qwen_chunk_seconds
        tasks = [
            {
                "label": str(index),
                "start": start,
                "duration": min(chunk_seconds, duration - start),
                "root": True,
                "split": False,
                "attempt": 0,
            }
            for index, start in enumerate(
                index * chunk_seconds for index in range(math.ceil(duration / chunk_seconds))
            )
            if start < duration
        ]
        _write_json_local(
            job_cache_dir / "chunk-index.json",
            {
                "source_audio_wav": str(audio),
                "chunk_seconds": chunk_seconds,
                "duration_seconds": duration,
                "chunks": [
                    {
                        "label": task["label"],
                        "start_seconds": task["start"],
                        "end_seconds": task["start"] + task["duration"],
                        "duration_seconds": task["duration"],
                        "cache_file": f"chunk-{int(task['label']):04d}-raw.json",
                    }
                    for task in tasks
                ],
            },
        )
        worker = QueueWorker([], self.mode, self.run_config)
        self._parallel_qwen_worker = worker
        worker.status.connect(self.status.emit)
        pending_by_start: dict[float, tuple[dict[str, Any], list[SubtitleSegment]]] = {}
        next_start = 0.0
        cached_count = 0
        missing_tasks: list[dict[str, Any]] = []

        for task in tasks:
            cache_path = job_cache_dir / f"chunk-{int(task['label']):04d}-raw.json"
            if cache_path.exists():
                try:
                    cached_segments = load_segments(cache_path)
                except Exception as exc:
                    self.status.emit(f"Chunk {task['label']}: invalid Qwen cache; rebuilding: {exc}")
                    cache_path.unlink(missing_ok=True)
                    missing_tasks.append(task)
                    continue
                pending_by_start[float(task["start"])] = (task, cached_segments)
                cached_count += 1
                self.status.emit(f"Chunk {task['label']}: loaded Qwen cache")
                continue

            half = float(task["duration"]) / 2.0
            child_tasks = [
                {
                    "label": f"{task['label']}a",
                    "start": float(task["start"]),
                    "duration": half,
                    "root": False,
                    "split": False,
                    "attempt": 0,
                },
                {
                    "label": f"{task['label']}b",
                    "start": float(task["start"]) + half,
                    "duration": float(task["duration"]) - half,
                    "root": False,
                    "split": False,
                    "attempt": 0,
                },
            ]
            child_paths = [
                job_cache_dir / f"chunk-{child['label']}-raw.json" for child in child_tasks
            ]
            if not any(path.exists() for path in child_paths):
                missing_tasks.append(task)
                continue
            for child, child_path in zip(child_tasks, child_paths):
                if not child_path.exists():
                    missing_tasks.append(child)
                    continue
                try:
                    cached_segments = load_segments(child_path)
                except Exception as exc:
                    self.status.emit(
                        f"Chunk {child['label']}: invalid split cache; rebuilding: {exc}"
                    )
                    child_path.unlink(missing_ok=True)
                    missing_tasks.append(child)
                    continue
                pending_by_start[float(child["start"])] = (child, cached_segments)
                cached_count += 1
                self.status.emit(f"Chunk {child['label']}: loaded split Qwen cache")

        worker.chunk_progress.connect(
            lambda snapshot: self.progress.emit(
                cached_count + int(snapshot.get("done", 0)),
                cached_count + int(snapshot.get("total", 0)),
            )
        )

        def on_chunk_ready(task: dict[str, Any], segments: list[SubtitleSegment]) -> None:
            nonlocal next_start
            start = float(task["start"])
            pending_by_start[start] = (task, segments)
            while next_start in pending_by_start:
                ready_task, ready = pending_by_start.pop(next_start)
                label = str(ready_task["label"])
                cache_label = f"{int(label):04d}" if label.isdigit() else label
                save_segments(job_cache_dir / f"chunk-{cache_label}-raw.json", ready)
                self._emit_incremental_segments(ready)
                next_start += float(ready_task["duration"])

        try:
            while next_start in pending_by_start:
                ready_task, ready = pending_by_start.pop(next_start)
                self._emit_incremental_segments(ready)
                next_start += float(ready_task["duration"])
            self.progress.emit(cached_count, len(tasks))
            if not missing_tasks:
                self.status.emit(f"Loaded all {cached_count} Qwen3-ASR chunks from cache")
                return False

            replicas = (
                self.run_config.qwen_replicas
                if self.run_config.qwen_parallel_enabled
                else 1
            )
            self.status.emit(
                f"Controlled Qwen3-ASR parallel transcription: {replicas} replica(s), "
                f"{self.run_config.qwen_chunk_seconds:.0f}s chunks; "
                f"{cached_count} cached, {len(missing_tasks)} pending"
            )
            worker._transcribe_local_qwen3_asr_chunked(
                audio,
                on_chunk_ready=on_chunk_ready,
                tasks_override=missing_tasks,
            )
            return True
        finally:
            self._parallel_qwen_worker = None
            _cleanup_temp_paths(worker._temp_paths)

    # ---- internal pipeline ----

    def _do_rolling_prefetch(self) -> None:
        self._temp_paths = []
        self._has_emitted_segments = False
        try:
            if not self.mode.available:
                raise RuntimeError(f"Missing model: {self.mode.model}")

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.cache_url = canonical_media_url(self.url)
            job_cache_dir = controlled_cache_dir(
                self.cache_url,
                self.mode.backend,
                self.mode.model_name,
                self.chunk_seconds,
            )
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
                native_output_base = self._native_output_base()
                self._export_subtitles(native_segments, native_output_base)
                self.status.emit(
                    f"Native subtitles written: {native_output_base.with_suffix('.srt')}"
                )
                self.native_subtitles_detected.emit(
                    native_segments,
                    "检测到视频自带中文字幕，已下载、保存并载入。"
                    f"\n\n字幕文件：{native_output_base.with_suffix('.srt')}"
                    "\n\n已跳过 Whisper 和 LLM 识别以节省资源。"
                )
                return

            # 1. Pause Chrome
            self.status.emit("Pausing Chrome while captions are prepared")
            from whisper_captioner.chrome_control import chrome_pause, chrome_pause_url
            if chrome_pause_url(self.cache_url) is None:
                chrome_pause()
            if self._stop:
                return

            # 2. Reuse or build the persistent URL audio cache.
            audio = prepare_url_audio_cache(
                self.url,
                run_command=self._run_cmd,
                status_signal=self.status,
            )
            audio_link = job_cache_dir / "source-audio.wav"
            audio_link.unlink(missing_ok=True)
            try:
                audio_link.symlink_to(audio)
            except OSError:
                pass
            _write_json_local(
                job_cache_dir / "source-info.json",
                {
                    "url": self.url,
                    "canonical_url": self.cache_url,
                    "audio_cache_wav": str(audio),
                    "backend": self.mode.backend,
                    "model": self.mode.model_name,
                    "pipeline_chunk_seconds": self.chunk_seconds,
                    "asr_chunk_seconds": (
                        self.run_config.qwen_chunk_seconds
                        if qwen3_asr_mode(self.mode)
                        else self.chunk_seconds
                    ),
                },
            )
            if self._stop:
                return

            # 3. Get duration
            duration = self._get_duration(audio)
            num_chunks = int(duration // self.chunk_seconds) + (1 if duration % self.chunk_seconds else 0)
            self.status.emit(
                f"Audio duration: {duration:.1f}s — processing in {num_chunks} chunk(s) of {self.chunk_seconds}s"
            )

            final_cache = self._final_subtitle_cache_path(job_cache_dir)
            cached_segments = self._load_current_final_cache(final_cache)
            if cached_segments is not None:
                try:
                    raw_segments = self._load_all_raw_segments(job_cache_dir)
                    quality_report = self._audit_and_cache_quality(
                        audio,
                        duration,
                        job_cache_dir,
                        raw_segments or cached_segments,
                    )
                    self.quality_updated.emit(quality_report_to_dict(quality_report))
                    if quality_report.status == "incomplete_speech_coverage":
                        self.status.emit(
                            "Cached subtitles failed current quality audit; rebuilding affected chunks"
                        )
                        cached_segments = None
                    else:
                        self.status.emit(f"Loaded final subtitle cache: {final_cache}")
                        if raw_segments:
                            self._export_subtitles(raw_segments, raw_output_base)
                            self.status.emit(
                                f"Raw subtitles written: {raw_output_base.with_suffix('.srt')}"
                            )
                        elif not self._llm_polish_ready():
                            self._export_subtitles(cached_segments, raw_output_base)
                            self.status.emit(
                                f"Raw subtitles written: {raw_output_base.with_suffix('.srt')}"
                            )
                        if self._llm_polish_ready():
                            self._export_final_subtitles(cached_segments, output_base)
                        self.first_segments.emit(cached_segments)
                        self.all_done.emit()
                        return
                except Exception as exc:
                    self.status.emit(f"Final subtitle cache unreadable, rebuilding: {exc}")

            # 4. Chunk → transcribe → cache. LLM proofreading runs once on the full transcript.
            offset = 0.0
            chunk_index = 0
            rebuilt_raw_cache = False

            if qwen3_asr_mode(self.mode) and self.run_config.qwen_parallel_enabled:
                rebuilt_raw_cache = self._transcribe_local_qwen_parallel(
                    audio,
                    job_cache_dir,
                    duration,
                )
                offset = duration

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
                    cached_segments = self._load_segment_cache(
                        raw_cache,
                        f"Chunk {chunk_index} raw subtitle cache",
                    )
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
                    self._last_asr_result = None
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
                                chunk_wav,
                                chunk_out,
                                chunk_index,
                                remaining,
                                original_segments=raw_segments,
                            )
                            if repaired_segments:
                                raw_segments = repaired_segments
                                if self._last_asr_result is None:
                                    self._last_asr_result = ASRResult(
                                        language=self._language_pin.language,
                                        words=[],
                                        segments=repaired_segments,
                                        diagnostics={
                                            "capability_warnings": [
                                                "word timestamps unavailable after local repair"
                                            ],
                                            "locally_repaired": True,
                                        },
                                    )
                        # Shift timestamps to absolute position in the full audio.
                        segments = [
                            SubtitleSegment(s.start + offset, s.end + offset, s.text)
                            for s in raw_segments
                        ]
                        save_segments(raw_cache, segments)
                        result_for_cache = self._last_asr_result or ASRResult(
                            language=self._language_pin.language,
                            words=[],
                            segments=raw_segments,
                            diagnostics={
                                "capability_warnings": ["segment-only backend or legacy result"]
                            },
                        )
                        save_asr_result(
                            job_cache_dir / f"chunk-{chunk_index:04d}-asr-v2.json",
                            ASRResult(
                                language=result_for_cache.language,
                                words=[
                                    SubtitleWord(
                                        word.start + offset,
                                        word.end + offset,
                                        word.text,
                                        word.probability,
                                    )
                                    for word in result_for_cache.words
                                ],
                                segments=segments,
                                diagnostics=result_for_cache.diagnostics,
                            ),
                        )
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

                self._emit_incremental_segments(segments)

                self.progress.emit(chunk_index + 1, num_chunks)
                offset += remaining  # Use actual chunk duration, not fixed chunk_seconds
                chunk_index += 1

            if not self._stop:
                if rebuilt_raw_cache:
                    self._discard_derived_subtitle_caches(job_cache_dir)
                all_raw_segments = self._load_all_raw_segments(job_cache_dir)
                quality_report = self._audit_and_cache_quality(
                    audio,
                    duration,
                    job_cache_dir,
                    all_raw_segments,
                )
                self.quality_updated.emit(quality_report_to_dict(quality_report))
                if all_raw_segments:
                    self._export_subtitles(all_raw_segments, raw_output_base)
                    self.status.emit(f"Raw subtitles written: {raw_output_base.with_suffix('.srt')}")
                final_segments = self._run_full_document_polish(job_cache_dir, output_base)
                if final_segments and not self._has_emitted_segments:
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

    def _load_segment_cache(self, path: Path, context: str) -> list[SubtitleSegment]:
        try:
            return load_segments(path)
        except Exception as exc:
            raise ValueError(f"{context} unreadable ({path.name}): {exc}") from exc

    def _load_all_raw_segments(self, job_cache_dir: Path) -> list[SubtitleSegment]:
        raw_files = sorted(job_cache_dir.glob("chunk-*-raw.json"))
        all_segments: list[SubtitleSegment] = []
        for path in raw_files:
            all_segments.extend(self._load_segment_cache(path, "Raw subtitle cache"))
        all_segments.sort(key=lambda item: (item.start, item.end))
        return all_segments

    def _final_subtitle_cache_path(self, job_cache_dir: Path) -> Path:
        return job_cache_dir / "final-subtitles-current.json"

    def _discard_derived_subtitle_caches(self, job_cache_dir: Path) -> None:
        for path in [self._final_subtitle_cache_path(job_cache_dir), self._full_polish_cache_path(job_cache_dir)]:
            if path.exists():
                path.unlink()
                self.status.emit(f"Discarded derived subtitle cache after raw rebuild: {path.name}")

    def _transcribe_chunk(self, chunk_wav: Path, chunk_out: Path, chunk_label: int | str) -> Path:
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
                    f"Transcribing chunk {chunk_label} with Qwen3-ASR",
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
                f"Transcribing chunk {chunk_label} with MLX-Audio",
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
                f"Transcribing chunk {chunk_label} with MLX Whisper",
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
                f"Transcribing chunk {chunk_label} with SenseVoice.cpp",
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
        if self.mode.backend == "nuc_asr":
            self._used_nuc_asr = True
            nuc_asr_model = _nuc_asr_model_for_mode(self.mode)
            self.status.emit(
                f"Transcribing chunk {chunk_label} with NUC remote ASR ({nuc_asr_model})..."
            )
            request_wav, vad_offset = self._prepare_remote_vad_chunk(chunk_wav, chunk_label)
            if request_wav is None:
                segments = []
                self._last_asr_result = ASRResult(
                    language=self._language_pin.language,
                    words=[],
                    segments=[],
                    diagnostics={"capability_warnings": ["ffmpeg found no stable voice window"]},
                )
            else:
                result = _transcribe_via_nuc_asr_result(
                    request_wav,
                    base_url=str(self.mode.model),
                    model=nuc_asr_model,
                    language=self._language_pin.request_language,
                    vad_filter=False,
                    status_signal=self.status,
                )
                shifted_words = [
                    SubtitleWord(
                        word.start + vad_offset,
                        word.end + vad_offset,
                        word.text,
                        word.probability,
                    )
                    for word in result.words
                ]
                shifted_segments = [
                    SubtitleSegment(
                        segment.start + vad_offset,
                        segment.end + vad_offset,
                        segment.text,
                    )
                    for segment in result.segments
                ]
                self._language_pin.observe(
                    result.language,
                    result.diagnostics.get("language_probability"),
                    _probe_audio_duration(request_wav),
                )
                segments, cue_warnings = build_cues(shifted_words, shifted_segments)
                diagnostics = dict(result.diagnostics)
                diagnostics.setdefault("capability_warnings", []).extend(cue_warnings)
                diagnostics["language_policy"] = {
                    "requested": self._language_pin.request_language,
                    "detected": result.language,
                    "pinned": self._language_pin.language,
                    "confidence": self._language_pin.confidence,
                }
                self._last_asr_result = ASRResult(
                    language=result.language,
                    words=shifted_words,
                    segments=segments,
                    diagnostics=diagnostics,
                )
            srt_path = chunk_out.with_suffix(".srt")
            save_segments_as_srt(srt_path, segments)
            return srt_path
        if self.mode.backend == "nuc_qwen3_asr_1p7b":
            self.status.emit(f"Transcribing chunk {chunk_label} with NUC remote Qwen3-ASR 1.7B...")
            request_wav, vad_offset = self._prepare_remote_vad_chunk(chunk_wav, chunk_label)
            if request_wav is None:
                segments = []
            else:
                segments = [
                    SubtitleSegment(segment.start + vad_offset, segment.end + vad_offset, segment.text)
                    for segment in _transcribe_via_nuc_qwen3_asr_1p7b(
                        request_wav, base_url=str(self.mode.model)
                    )
                ]
            segments = self._trim_overlap_segments(
                segments,
                leading_trim=0.0,
                trailing_trim=0.0,
                chunk_duration=float(self.chunk_seconds),
            )
            srt_path = chunk_out.with_suffix(".srt")
            save_segments_as_srt(srt_path, segments)
            return srt_path
        self._run_cmd(
            [
                WHISPER_CLI,
                "-m", str(self.mode.model),
                "-f", str(chunk_wav),
                *self.run_config.cpp_args(),
                "-l", "zh",
                "-osrt",
                "-of", str(chunk_out),
            ],
            f"Transcribing chunk {chunk_label}",
        )
        return chunk_out.with_suffix(".srt")

    def _audit_and_cache_quality(
        self,
        audio: Path,
        duration: float,
        job_cache_dir: Path,
        fallback_segments: list[SubtitleSegment],
    ):
        result_files = sorted(job_cache_dir.glob("chunk-*-asr-v2.json"))
        words: list[SubtitleWord] = []
        segments: list[SubtitleSegment] = []
        warnings: list[str] = []
        languages: list[str] = []
        for path in result_files:
            try:
                result = load_asr_result(path)
            except Exception as exc:
                warnings.append(f"{path.name} unreadable: {exc}")
                continue
            words.extend(result.words)
            segments.extend(result.segments)
            if result.language:
                languages.append(result.language)
            warnings.extend(result.diagnostics.get("capability_warnings", []))
        if not segments:
            segments = fallback_segments
            warnings.append("quality audit used legacy segment cache")
        combined = ASRResult(
            language=self._language_pin.language or (languages[0] if languages else ""),
            words=sorted(words, key=lambda item: (item.start, item.end)),
            segments=sorted(segments, key=lambda item: (item.start, item.end)),
            diagnostics={"capability_warnings": list(dict.fromkeys(warnings))},
        )

        # --- Gemini + Whisper dual-model fusion ---
        if self.gemini_fusion_enabled and self.gemini_api_key and combined.words:
            gemini_result = gemini_transcribe_audio(
                audio, self.gemini_api_key, model="gemini-2.5-flash",
            )
            combined.diagnostics["gemini_fusion"] = {
                "status": gemini_result.status,
                "model": gemini_result.model,
                "elapsed": gemini_result.elapsed,
            }
            if gemini_result.status == "completed" and gemini_result.lines:
                fused = fuse_gemini_with_whisper(gemini_result.lines, combined.words)
                if fused:
                    combined.diagnostics["gemini_fusion"]["fused_segments"] = len(fused)
                    combined.diagnostics["gemini_fusion"]["source"] = "gemini-text + whisper-timestamps"
                    combined.segments = fused
                    self.status.emit(
                        f"Gemini+Whisper fusion: {len(gemini_result.lines)} Gemini lines → "
                        f"{len(fused)} fused segments "
                        f"({gemini_result.elapsed:.1f}s)"
                    )
            elif gemini_result.status == "failed":
                combined.diagnostics["gemini_fusion"]["warning"] = gemini_result.warning
                self.status.emit(f"Gemini fusion failed: {gemini_result.warning}")
                self.gemini_fusion_blocked.emit(
                    f"Gemini 转写 API 调用失败：\n{gemini_result.warning}\n\n"
                    "将继续使用纯 Whisper 字幕。"
                )
        elif self.gemini_fusion_enabled and not combined.words:
            combined.diagnostics["gemini_fusion"] = {
                "status": "skipped", "warning": "word timestamps unavailable",
            }
            self.gemini_fusion_blocked.emit(
                "Gemini 双模型融合已启用，但 Whisper ASR 未返回逐词时间戳。\n"
                "融合需要词级时间戳才能将 Gemini 文本精确对齐到时间轴。"
            )
        try:
            silencedetect_output = self._run_cmd_capture(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-i",
                    str(audio),
                    "-af",
                    "silencedetect=noise=-35dB:d=0.3",
                    "-f",
                    "null",
                    "-",
                ],
                "Auditing speech coverage",
            )
            speech_regions = parse_silencedetect_regions(silencedetect_output, duration)
        except Exception as exc:
            speech_regions = []
            combined.diagnostics["capability_warnings"].append(f"ffmpeg VAD audit failed: {exc}")
        _write_json_local(
            job_cache_dir / "speech-regions.json",
            {
                "schema_version": 2,
                "source": "ffmpeg",
                "parameters": {"noise": "-35dB", "minimum_silence": 0.3},
                "regions": [
                    {
                        "start": region.start,
                        "end": region.end,
                        "confidence": region.confidence,
                        "source": region.source,
                    }
                    for region in speech_regions
                ],
            },
        )
        report = audit_asr_result(combined, speech_regions, duration=duration)
        report.retries = list(self._retry_records)

        shadow = run_omnivad_shadow(audio, job_cache_dir / "omnivad")
        report.diagnostics["omnivad_shadow"] = {
            "status": shadow.status,
            "region_count": len(shadow.regions),
            "warning": shadow.warning,
        }
        if shadow.warning:
            report.warnings.append(shadow.warning)
            if report.status == "passed":
                report.status = "passed_with_warnings"

        report.diagnostics["alignment"] = {"status": "abandoned", "reason": "LattifAI tested and rejected"}
        report.diagnostics["language"] = combined.language
        report.diagnostics["language_policy"] = (
            "explicit" if self._language_pin.explicit_language not in {"", "auto"} else "detect-once-then-pin"
        )
        _write_json_local(
            job_cache_dir / "quality-report.json",
            {
                "schema_version": 2,
                "pipeline_version": SUBTITLE_PIPELINE_VERSION,
                **quality_report_to_dict(report),
            },
        )
        self.status.emit(
            f"Subtitle quality: {report.status}; speech coverage={report.speech_coverage:.1%}; "
            f"suspicious={len(report.suspicious_regions)} uncovered={len(report.uncovered_regions)}"
        )
        return report

    def _prepare_remote_vad_chunk(
        self,
        chunk_wav: Path,
        chunk_label: int | str,
    ) -> tuple[Path | None, float]:
        if not self.remote_vad_enabled:
            return chunk_wav, 0.0
        try:
            duration = _probe_audio_duration(chunk_wav)
            output = self._run_cmd_capture(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-i",
                    str(chunk_wav),
                    "-af",
                    "silencedetect=noise=-35dB:d=0.3",
                    "-f",
                    "null",
                    "-",
                ],
                f"Detecting speech window for chunk {chunk_label}",
            )
            window = parse_silencedetect_voice_window(output, duration)
            if window is None:
                self.status.emit(
                    f"Chunk {chunk_label}: no stable voice window detected, skipping remote ASR request"
                )
                return None, 0.0
            if window.start <= 0.001 and window.duration >= duration - 0.001:
                return chunk_wav, 0.0
            trimmed = self._track_temp_path(
                chunk_wav.with_name(f"{chunk_wav.stem}-vad-{uuid.uuid4().hex}.wav")
            )
            self._run_cmd(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-y",
                    "-ss",
                    str(window.start),
                    "-t",
                    str(window.duration),
                    "-i",
                    str(chunk_wav),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(trimmed),
                ],
                f"Trimming speech window for chunk {chunk_label}",
            )
            return trimmed, window.start
        except Exception as exc:
            self.status.emit(f"Chunk {chunk_label}: VAD failed, using full chunk: {exc}")
            return chunk_wav, 0.0

    def _repair_sparse_chunk_with_subchunks(
        self,
        chunk_wav: Path,
        chunk_out: Path,
        chunk_index: int,
        chunk_duration: float,
        original_segments: list[SubtitleSegment] | None = None,
    ) -> list[SubtitleSegment]:
        if self.mode.backend not in {"mlx_whisper", "mlx_audio", "nuc_asr"} or chunk_duration <= 12:
            return []
        self.status.emit(
            f"Chunk {chunk_index}: retrying sparse result with smaller subchunks"
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
            srt_path = self._transcribe_chunk(part_wav, part_out, _subchunk_label(chunk_index, part_index))
            if not srt_path.exists():
                continue
            sub_segments = self._clamp_chunk_segments(parse_srt(srt_path), part_duration)
            repaired.extend(
                SubtitleSegment(s.start + start_offset, s.end + start_offset, s.text)
                for s in sub_segments
            )
        repaired = self._clamp_chunk_segments(repaired, chunk_duration)
        suspicious_regions = []
        for segment in original_segments or []:
            span = segment.end - segment.start
            density = len(re.sub(r"\s+", "", segment.text)) / span if span > 0 else 0.0
            if span >= 6.0 and density < 1.5:
                suspicious_regions.append(
                    RetryRegion(segment.start, segment.end, "low text density")
                )
            elif span > 5.25:
                suspicious_regions.append(RetryRegion(segment.start, segment.end, "long cue"))
        if suspicious_regions:
            retry_regions = merge_retry_regions(
                suspicious_regions,
                guard=2.0,
                duration=chunk_duration,
            )
            repaired = replace_segments_in_regions(
                original_segments or [],
                repaired,
                retry_regions,
            )
        else:
            retry_regions = [RetryRegion(0.0, chunk_duration, "sparse chunk")]
        self._retry_records.extend(
            {
                "chunk": chunk_index,
                "start": region.start,
                "end": region.end,
                "reason": region.reason,
                "attempt": 1,
                "model": _nuc_asr_model_for_mode(self.mode)
                if self.mode.backend == "nuc_asr"
                else self.mode.model_name,
            }
            for region in retry_regions
        )
        if (
            self.mode.backend == "nuc_asr"
            and _nuc_asr_model_for_mode(self.mode) != "large-v3"
            and self._chunk_cache_looks_bad(repaired, chunk_duration)
        ):
            self.status.emit(
                f"Chunk {chunk_index}: turbo retry remains suspicious; reviewing with large-v3"
            )
            reviewed = _transcribe_via_nuc_asr_result(
                chunk_wav,
                base_url=str(self.mode.model),
                model="large-v3",
                language=self._language_pin.request_language,
                vad_filter=False,
                status_signal=self.status,
            )
            reviewed_cues, _ = build_cues(reviewed.words, reviewed.segments)
            adopted_review = False
            if (
                not self._chunk_cache_looks_bad(reviewed_cues, chunk_duration)
                and self._candidate_text_score(reviewed_cues)
                >= self._candidate_text_score(repaired)
            ):
                repaired = reviewed_cues
                adopted_review = True
            else:
                self.status.emit(
                    f"Chunk {chunk_index}: retaining turbo repair because large-v3 review is not better"
                )
            reviewed.diagnostics["review_model"] = "large-v3"
            if adopted_review:
                self._last_asr_result = ASRResult(
                    reviewed.language,
                    reviewed.words,
                    repaired,
                    reviewed.diagnostics,
                )
            self._retry_records.extend(
                {
                    "chunk": chunk_index,
                    "start": region.start,
                    "end": region.end,
                    "reason": region.reason,
                    "attempt": 2,
                    "model": "large-v3",
                }
                for region in retry_regions
            )
        return repaired

    @staticmethod
    def _candidate_text_score(segments: list[SubtitleSegment]) -> tuple[int, int, int]:
        compact = "".join(re.sub(r"\s+", "", segment.text) for segment in segments)
        replacement_chars = compact.count("�")
        repeated = sum(
            1
            for previous, current in zip(segments, segments[1:])
            if previous.text.strip() == current.text.strip()
        )
        return (-replacement_chars, -repeated, len(compact))

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
        for segment in segments:
            segment_duration = segment.end - segment.start
            compact_text = re.sub(r"\s+", "", segment.text)
            if (
                segment_duration >= 6.0
                and len(compact_text) >= 2
                and len(compact_text) / segment_duration < 1.5
            ):
                return True
        for previous, current in zip(segments, segments[1:]):
            if current.start - previous.end >= 6.0:
                return True
        last_text = segments[-1].text.strip()
        if len(last_text) > 30:
            words = last_text.split()
            if len(words) >= 6 and len(set(words[-6:])) <= 2:
                return True
            if any(last_text.count(token * 2) for token in set(words[: min(6, len(words))])):
                return True
        return False

    def _llm_polish_ready(self) -> bool:
        return bool(
            self.llm_provider
            and llm_provider_ready(self.llm_provider, self.llm_api_key)
        )

    def _pipeline_signature(self) -> dict:
        llm_ready = self._llm_polish_ready()
        return {
            "pipeline_version": SUBTITLE_PIPELINE_VERSION,
            "whisper_backend": self.mode.backend,
            "whisper_model": self.mode.model_name,
            "chunk_seconds": self.chunk_seconds,
            "native_subtitles": "zh-only-skip",
            "llm_provider": self.llm_provider.key if llm_ready else "raw",
            "llm_model": (
                self.llm_model_id or self.llm_provider.model_id
                if llm_ready
                else ""
            ),
            "llm_api_url": self.llm_api_url if llm_ready else "",
            "proofread_scope": "full-document",
            "ai_zh_reference": False,
            "term_extraction": False,
        }

    def _load_current_final_cache(self, cache_path: Path) -> Optional[list[SubtitleSegment]]:
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid final subtitle cache at {cache_path}: malformed JSON ({exc})"
            ) from exc
        if isinstance(data, list):
            self.status.emit("Ignoring legacy final subtitle cache without pipeline signature")
            return None
        if data.get("pipeline_signature") != self._pipeline_signature():
            self.status.emit("Final subtitle cache belongs to a different pipeline; rebuilding")
            return None
        quality_path = cache_path.parent / "quality-report.json"
        if not quality_path.exists():
            self.status.emit("Final subtitle cache has no v2 quality report; rebuilding")
            return None
        try:
            quality_data = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.status.emit("Final subtitle quality report is unreadable; rebuilding")
            return None
        if (
            quality_data.get("schema_version") != 2
            or quality_data.get("pipeline_version") != SUBTITLE_PIPELINE_VERSION
        ):
            self.status.emit("Final subtitle quality report is stale; rebuilding")
            return None
        self.quality_updated.emit(quality_data)
        segments_data = data.get("segments", [])
        if not isinstance(segments_data, list):
            raise ValueError(
                f"Invalid final subtitle cache at {cache_path}: 'segments' must be a list"
            )
        return [
            segment_from_dict(item, path=cache_path, index=index)
            for index, item in enumerate(segments_data)
        ]

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
        return source_output_dir(GENERATED_DIR, title) / f"{title}-{self._output_variant_suffix()}-{stamp}"

    def _raw_output_base(self, stamp: str) -> Path:
        title = clean_title_for_filename(self._source_title())
        return source_output_dir(GENERATED_DIR, title) / f"{title}-{self._asr_output_suffix()}-原始识别字幕"

    def _optimized_output_base(self, stamp: str) -> Path:
        title = clean_title_for_filename(self._source_title())
        return source_output_dir(GENERATED_DIR, title) / f"{title}-{self._output_variant_suffix()}-LLM优化字幕"

    def _native_output_base(self) -> Path:
        title = clean_title_for_filename(self._source_title())
        return source_output_dir(GENERATED_DIR, title) / f"{title}-视频原生中文字幕"

    def _output_variant_suffix(self) -> str:
        parts = [self._asr_output_suffix()]
        if self.llm_provider:
            parts.append(clean_title_for_filename(self.llm_provider.key, fallback="llm"))
            model_name = self.llm_model_id or self.llm_provider.model_id
            if model_name:
                parts.append(clean_title_for_filename(model_name, fallback="model"))
        return "-".join(parts)

    def _asr_output_suffix(self) -> str:
        return clean_title_for_filename(self.mode.key, fallback="mode")

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
                cwd=str(OUTPUT_DIR),
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
            all_segments.extend(self._load_segment_cache(path, "Raw subtitle cache"))
        all_segments.sort(key=lambda item: (item.start, item.end))
        try:
            llm_ready = self._llm_polish_ready()
            if llm_ready and cache_path.exists():
                polished = self._load_segment_cache(cache_path, "Full-document LLM polish cache")
                self.status.emit("Loaded full-document LLM polish cache")
            elif llm_ready:
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
                self.status.emit("No LLM provider/API key configured; exporting raw Whisper subtitles")
                self._save_current_final_cache(
                    self._final_subtitle_cache_path(job_cache_dir),
                    all_segments,
                )
                self.status.emit("Updated final subtitle cache for raw pipeline")
                return all_segments
            if polished:
                self._save_current_final_cache(self._final_subtitle_cache_path(job_cache_dir), polished)
                self.status.emit("Updated final subtitle cache for current pipeline")
                self._export_final_subtitles(polished, output_base)
            return polished
        except Exception as exc:
            self.status.emit(f"Full-document LLM polish failed: {exc}; exporting raw Whisper subtitles")
            return all_segments

    def _load_or_fetch_native_subtitles(self, job_cache_dir: Path) -> tuple[list[SubtitleSegment], str]:
        preferred = [
            ("zh", "zh.*"),
        ]
        for cache_name, lang_expr in preferred:
            cache_path = job_cache_dir / f"native-subtitles-{cache_name}.json"
            if cache_path.exists():
                try:
                    segments = self._load_segment_cache(
                        cache_path,
                        f"Native subtitle cache ({cache_name})",
                    )
                    self.status.emit(
                        f"Loaded native subtitle cache ({cache_name}) from {cache_path.name}"
                    )
                    return segments, cache_name
                except Exception as exc:
                    self.status.emit(str(exc))

            subs_dir = job_cache_dir / f"native-subs-{cache_name}"
            subs_dir.mkdir(parents=True, exist_ok=True)
            out_template = subs_dir / "native.%(ext)s"
            for subtitle_kind, write_flag in (
                ("manual", "--write-subs"),
                ("automatic", "--write-auto-subs"),
            ):
                for stale_path in subs_dir.glob("native.*"):
                    if stale_path.is_file():
                        stale_path.unlink()
                try:
                    self._run_cmd(
                        [
                            YT_DLP,
                            "--skip-download",
                            write_flag,
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
                        f"Fetching {subtitle_kind} native subtitles ({lang_expr})",
                    )
                except Exception as exc:
                    self.status.emit(
                        f"No {subtitle_kind} native subtitles fetched for "
                        f"{lang_expr} ({cache_name}, {subs_dir.name}): {exc}"
                    )
                subtitle_files = sorted(
                    p for p in subs_dir.glob("native.*")
                    if p.suffix.lower() in {".srt", ".vtt"}
                )
                if subtitle_files:
                    self.status.emit(
                        f"Downloaded {subtitle_kind} native subtitles: "
                        f"{', '.join(path.name for path in subtitle_files)}"
                    )
                    break

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
                    f"Saved native subtitle cache ({cache_name}) from {len(subtitle_files)} file(s) for {lang_expr}"
                )
                return segments, cache_name

        self.status.emit("No usable native subtitle segments found for preferred native subtitle languages")
        return [], "none"

    def _extract_terms_with_mlx(self, job_cache_dir: Path, segments: list[SubtitleSegment]) -> list[str]:
        cache_path = job_cache_dir / "mlx-terms.json"
        if cache_path.exists():
            try:
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid MLX term cache at {cache_path}: malformed JSON ({exc})") from exc
                terms_data = data.get("terms", [])
                if not isinstance(terms_data, list):
                    raise ValueError(f"Invalid MLX term cache at {cache_path}: 'terms' must be a list")
                return [str(item["term"]) for item in terms_data if isinstance(item, dict) and item.get("term")]
            except Exception as exc:
                self.status.emit(f"MLX term cache unreadable ({cache_path.name}): {exc}")
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
        return _probe_audio_duration(audio_path)

    def _run_cmd(self, cmd: list[str], label: str) -> None:
        for attempt in range(2):
            self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(OUTPUT_DIR)
            )
            try:
                output_lines = _stream_process_output(
                    self.proc,
                    status_signal=self.status,
                    stop_flag=lambda: self._stop,
                    stop_message="Rolling prefetch stopped",
                )
                returncode = self.proc.wait()
                if returncode == 0 or self._stop:
                    return
                if attempt == 0 and _should_retry_yt_dlp_cookie_read(cmd, output_lines):
                    self.status.emit(
                        "yt-dlp received an incomplete Chrome cookie snapshot; retrying once"
                    )
                    time.sleep(1.0)
                    continue

                cmd_name = cmd[0]
                error_context = "\n".join(output_lines[-10:]) if output_lines else "(no output)"
                if "yt-dlp" in cmd_name or "yt-dlp" in " ".join(cmd):
                    error_msg = f"yt-dlp failed to download the URL.\n\nLast output:\n{error_context}\n\nPossible reasons:\n- URL is not supported by yt-dlp (e.g., GitHub docs, Wikipedia)\n- Video is region-restricted or requires login\n- Chrome cookies are unavailable, expired, or locked by the browser\n- URL format is incorrect"
                    raise RuntimeError(error_msg)

                raise RuntimeError(f"command failed: {cmd_name}\n\nOutput:\n{error_context}")
            finally:
                self.proc = None

    def _run_cmd_capture(self, cmd: list[str], label: str) -> str:
        self.status.emit(f"{label}: {' '.join(shlex.quote(part) for part in cmd)}")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(OUTPUT_DIR)
        )
        try:
            output_lines = _stream_process_output(
                self.proc,
                status_signal=self.status,
                stop_flag=lambda: self._stop,
                stop_message="Rolling prefetch stopped",
            )
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
