#!/usr/bin/env python3
"""Standalone ASR entry points used by the ForensicSubtitle.command TUI."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gemini_youtube_url_asr_smoke import save_result, transcribe_youtube_url
from whisper_captioner.config import FFMPEG, FFPROBE, GENERATED_DIR, NUC_OLLAMA_HOST
from whisper_captioner.credentials import load_secret, save_secret
from whisper_captioner.external_backends import gemini_transcribe_audio
from whisper_captioner.models import ASRResult, RetryRegion, SubtitleSegment, SubtitleWord
from whisper_captioner.subtitle_io import save_asr_result, save_segments_as_srt
from whisper_captioner.subtitle_reliability import (
    audit_asr_result,
    build_cues,
    merge_retry_regions,
    parse_silencedetect_regions,
    parse_verbose_asr_response,
)

KEYCHAIN_SERVICE = "WhisperCaptioner"
GEMINI_KEYCHAIN_ACCOUNT = "gemini-api-key"
NUC_SSH_KEYCHAIN_ACCOUNT = "nuc-ssh-sudo-password"
DEFAULT_NUC_SSH_TARGET = "jack@192.168.31.196"
DEFAULT_NUC_ASR_BACKEND_CONTAINER = "nuc-asr-backend"
DEFAULT_NUC_WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEFAULT_NUC_NATIVE_BATCH_SIZE = max(
    0,
    int(os.environ.get("WHISPER_CAPTIONER_NUC_NATIVE_BATCH_SIZE", "8")),
)


@dataclass(frozen=True)
class ChunkWindow:
    label: str
    start: float
    duration: float
    timeline_start: float
    leading_trim: float
    trailing_trim: float
    depth: int = 0
    attempts: int = 0


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def safe_name(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "-", value).strip(" .-")
    return (value or "asr-job")[:120]


def remote_asr_quality_issue(segments: list[SubtitleSegment]) -> str | None:
    texts = [" ".join(segment.text.split()) for segment in segments if segment.text.strip()]
    for text in texts:
        tokens = re.findall(r"\w+(?:['’-]\w+)*", text.lower(), flags=re.UNICODE)
        if len(tokens) < 12:
            continue
        for offset in range(min(12, len(tokens) - 5)):
            for phrase_length in range(2, min(12, (len(tokens) - offset) // 3) + 1):
                phrase = tokens[offset:offset + phrase_length]
                repeats = 1
                cursor = offset + phrase_length
                while tokens[cursor:cursor + phrase_length] == phrase:
                    repeats += 1
                    cursor += phrase_length
                if repeats >= 3 and (cursor - offset) / len(tokens) >= 0.75:
                    return f"one segment repeats {repeats} times ({' '.join(phrase[:8])!r})"

    if len(texts) < 20:
        return None

    counts = Counter(texts)
    most_common_text, most_common_count = counts.most_common(1)[0]
    repeated_share = most_common_count / len(texts)
    unique_share = len(counts) / len(texts)
    if most_common_count >= 8 and repeated_share >= 0.30:
        return f"one phrase occupies {repeated_share:.0%} of {len(texts)} segments ({most_common_text[:60]!r})"
    if len(texts) >= 40 and unique_share <= 0.12:
        return f"only {len(counts)} unique texts across {len(texts)} segments ({unique_share:.0%} unique)"
    return None


def mask_secret(value: str) -> str:
    if not value:
        return "未保存"
    if len(value) <= 24:
        return f"已保存（长度 {len(value)}）"
    return f"{value[:10]}...{value[-10:]}"


def youtube_identity(url: str) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [parsed.path.rsplit("/", 1)[-1]])[0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    key = load_secret(KEYCHAIN_SERVICE, GEMINI_KEYCHAIN_ACCOUNT)
    if key:
        return key
    if not sys.stdin.isatty():
        raise RuntimeError("GEMINI_API_KEY is missing and no interactive terminal is available")
    key = getpass.getpass("Gemini API Key（输入不会显示）: ").strip()
    if not key:
        raise RuntimeError("Gemini API Key is required")
    answer = input("保存到 macOS Keychain，供以后使用？[Y/n]: ").strip().lower()
    if answer not in {"n", "no"}:
        save_secret(KEYCHAIN_SERVICE, GEMINI_KEYCHAIN_ACCOUNT, key)
    return key


def manifest_start(path: Path, **values: Any) -> dict[str, Any]:
    manifest = {
        "kind": "standalone-asr",
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        **values,
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("source") == manifest.get("source"):
            manifest["created_at"] = previous.get("created_at", manifest["created_at"])
            manifest["stages"] = previous.get("stages", {})
    write_json(path, manifest)
    return manifest


def manifest_write(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now()
    write_json(path, manifest)


def default_gemini_output(url: str) -> Path:
    identity = safe_name(youtube_identity(url) or "youtube")
    return GENERATED_DIR / f"Gemini-URL-ASR [{identity}]"


def completed_gemini_job(
    manifest_path: Path,
    *,
    source: str,
    model: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("source") != source
        or manifest.get("model") != model
    ):
        return None
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not all(
        isinstance(outputs.get(key), str)
        and Path(outputs[key]).is_file()
        and Path(outputs[key]).stat().st_size > 0
        for key in ("transcript", "metadata")
    ):
        return None
    return manifest


def run_gemini_url(args: argparse.Namespace) -> int:
    if urlparse(args.url).scheme not in {"http", "https"}:
        raise RuntimeError("A public YouTube HTTP(S) URL is required")
    output_dir = (args.output_dir or default_gemini_output(args.url)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "asr-manifest.json"
    completed = completed_gemini_job(
        manifest_path,
        source=args.url,
        model=args.model,
    )
    if completed is not None:
        print(f"复用已完成的 Gemini URL 全文：{completed['outputs']['transcript']}")
        return 0
    manifest = manifest_start(
        manifest_path,
        mode="gemini-youtube-url-audio-only",
        title=f"Gemini URL ASR [{youtube_identity(args.url)}]",
        source=args.url,
        model=args.model,
        visual_analysis_requested=False,
    )
    try:
        print("Gemini URL 离线转写：仅请求语音全文，不请求视觉分析或时间戳。", flush=True)
        result = transcribe_youtube_url(
            url=args.url,
            api_key=gemini_api_key(),
            model=args.model,
            timeout=args.timeout,
        )
        outputs = save_result(result, output_dir)
        manifest.update(
            status="completed",
            outputs=outputs,
            characters=len(result.text),
            elapsed_seconds=result.metadata["elapsed_seconds"],
            recommended_output=outputs["transcript"],
        )
        manifest_write(manifest_path, manifest)
        print(f"\n完成。全文：{outputs['transcript']}", flush=True)
        return 0
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc) or type(exc).__name__,
        )
        manifest_write(manifest_path, manifest)
        raise


def default_gemini_local_output(source: Path) -> Path:
    return GENERATED_DIR / f"{safe_name(source.stem)}-Gemini-Local-ASR"


def save_gemini_local_result(result: Any, output_dir: Path, *, audio_path: Path, source: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript = output_dir / "gemini-local-audio-asr-transcript.txt"
    metadata = output_dir / "gemini-local-audio-asr-metadata.json"
    if result.text.strip():
        transcript.write_text(result.text.strip() + "\n", encoding="utf-8")
    write_json(
        metadata,
        {
            "input": "local-audio-ogg-file-api",
            "source": str(source),
            "audio": str(audio_path),
            "model": result.model,
            "status": result.status,
            "elapsed": result.elapsed,
            "characters": len(result.text),
            "diagnostics": result.diagnostics,
            "warning": result.warning,
        },
    )
    outputs = {"metadata": str(metadata)}
    if transcript.is_file() and transcript.stat().st_size > 0:
        outputs["transcript"] = str(transcript)
    return outputs


def run_gemini_local_audio(
    source: Path,
    output_dir: Path,
    *,
    model: str,
    timeout: int,
    upload_timeout: int,
    processing_timeout: int,
) -> dict[str, Any]:
    ogg = gemini_audio_upload_path(source, output_dir)
    print(f"Gemini OGG：{ogg.name}（{ogg.stat().st_size / 1024 / 1024:.1f} MiB）", flush=True)
    result = gemini_transcribe_audio(
        ogg,
        gemini_api_key(),
        model=model,
        timeout=timeout,
        upload_timeout=upload_timeout,
        processing_timeout=processing_timeout,
        force_file_api=True,
        progress_callback=lambda message: print(f"Gemini: {message}", flush=True),
    )
    outputs = save_gemini_local_result(result, output_dir, audio_path=ogg, source=source)
    if result.status != "completed" or not result.text.strip():
        raise RuntimeError(f"Gemini local ASR failed: {result.warning or result.status}")
    return {
        "model": result.model,
        "characters": len(result.text),
        "ogg": str(ogg),
        "transcript": outputs["transcript"],
        "metadata": outputs["metadata"],
        "elapsed": result.elapsed,
        "diagnostics": result.diagnostics,
    }


def completed_gemini_local_stage(stage: dict[str, Any]) -> bool:
    return all(
        isinstance(stage.get(key), str)
        and Path(stage[key]).is_file()
        and Path(stage[key]).stat().st_size > 0
        for key in ("transcript", "metadata")
    )


def run_gemini_local(args: argparse.Namespace) -> int:
    source = args.audio.expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Local audio file not found: {source}")
    output_dir = (args.output_dir or default_gemini_local_output(source)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "asr-manifest.json"
    manifest = manifest_start(
        manifest_path,
        mode="gemini-local-audio-asr",
        title=f"Gemini local ASR: {source.name}",
        source=str(source),
        model=args.model,
        stages={},
    )
    try:
        stages = manifest.setdefault("stages", {})
        if completed_gemini_local_stage(stages.get("gemini", {})):
            print("\n[复用已完成的 Gemini 本地音频 ASR 产物]", flush=True)
        else:
            print("\n[Gemini OGG/File API 本地音频全文 ASR]", flush=True)
            stages["gemini"] = {"status": "running", "updated_at": now()}
            manifest_write(manifest_path, manifest)
            gemini = run_gemini_local_audio(
                source,
                output_dir,
                model=args.model,
                timeout=args.timeout,
                upload_timeout=args.upload_timeout,
                processing_timeout=args.processing_timeout,
            )
            stages["gemini"] = {"status": "completed", "updated_at": now(), **gemini}
            manifest.update(status="completed", recommended_output=gemini["transcript"])
            manifest_write(manifest_path, manifest)
        print(f"\n完成。产物目录：{output_dir}", flush=True)
        return 0
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc) or type(exc).__name__,
        )
        manifest_write(manifest_path, manifest)
        raise


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}]", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def media_stream_info(source: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not inspect media streams: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    if not audio_streams:
        raise RuntimeError(f"Media file contains no audio stream: {source}")
    audio = audio_streams[0]
    return {
        "suffix": source.suffix.lower(),
        "audio_streams": len(audio_streams),
        "video_streams": len(video_streams),
        "selected_audio_stream": audio.get("index"),
        "selected_audio_codec": audio.get("codec_name"),
        "selected_audio_sample_rate": audio.get("sample_rate"),
        "selected_audio_channels": audio.get("channels"),
    }


def print_media_stream_info(info: dict[str, Any]) -> None:
    print(
        "媒体检测："
        f"音频流 {info['audio_streams']} 个，视频流 {info['video_streams']} 个；"
        f"使用音频流 #{info['selected_audio_stream']} "
        f"({info.get('selected_audio_codec') or 'unknown'}, "
        f"{info.get('selected_audio_sample_rate') or '?'} Hz, "
        f"{info.get('selected_audio_channels') or '?'} ch)",
        flush=True,
    )


def normalized_wav(source: Path, output_dir: Path) -> Path:
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio-16k-mono.wav"
    cache_meta = work / "audio-16k-mono.json"
    stream_info = media_stream_info(source)
    identity = {
        "source": str(source),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "stream_info": stream_info,
    }
    print_media_stream_info(stream_info)
    if wav.is_file() and wav.stat().st_size > 0 and cache_meta.is_file():
        cached = json.loads(cache_meta.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in identity.items()):
            print(f"复用 16 kHz 单声道 WAV：{wav}", flush=True)
            return wav
    run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        "将本地媒体首个音频流规范化为 16 kHz/mono WAV",
    )
    write_json(cache_meta, {**identity, "wav": str(wav), "created_at": now()})
    return wav


def gemini_audio_upload_path(source: Path, output_dir: Path) -> Path:
    stream_info = media_stream_info(source)
    if (
        source.suffix.lower() in {".ogg", ".oga"}
        and stream_info.get("video_streams") == 0
        and str(stream_info.get("selected_audio_codec") or "").lower() == "opus"
    ):
        print(f"直接使用原始 OGG/Opus 上传 Gemini：{source}", flush=True)
        return source
    return normalized_gemini_ogg(source, output_dir, stream_info=stream_info)


def normalized_gemini_ogg(
    source: Path,
    output_dir: Path,
    *,
    stream_info: dict[str, Any] | None = None,
) -> Path:
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    ogg = work / "gemini-audio.ogg"
    cache_meta = work / "gemini-audio.json"
    if stream_info is None:
        stream_info = media_stream_info(source)
    identity = {
        "source": str(source),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "stream_info": stream_info,
        "codec": "libopus",
        "bitrate": "64k",
    }
    if ogg.is_file() and ogg.stat().st_size > 0 and cache_meta.is_file():
        cached = json.loads(cache_meta.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in identity.items()):
            print(f"复用 Gemini OGG/Opus：{ogg}", flush=True)
            return ogg
    run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            str(ogg),
        ],
        "将本地媒体首个音频流转为 Gemini OGG/Opus",
    )
    write_json(cache_meta, {**identity, "ogg": str(ogg), "created_at": now()})
    return ogg


def probe_audio_duration(audio_path: Path) -> float:
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not determine audio duration: {completed.stderr.strip()}")
    try:
        return float(completed.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not determine audio duration: {completed.stdout.strip()}") from exc


def extract_wav_chunk(
    source: Path,
    target: Path,
    start: float,
    duration: float,
    label: str = "ASR",
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        f"准备 {label} 分块 {target.stem}",
    )


def detect_voice_window(wav: Path, duration: float, label: str) -> tuple[float, float] | None:
    completed = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(wav),
            "-af",
            "silencedetect=noise=-35dB:d=0.3",
            "-f",
            "null",
            "-",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        print(f"VAD 检测失败，保留完整分块 {label}: {completed.stderr.strip()[:200]}", flush=True)
        return 0.0, duration
    regions = parse_silencedetect_regions(
        completed.stdout + "\n" + completed.stderr,
        duration,
        source="ffmpeg",
        minimum_voice=0.10,
    )
    if not regions:
        return None
    start = max(0.0, regions[0].start - 0.10)
    end = min(duration, regions[-1].end + 0.15)
    return start, max(0.0, end - start)


def prepare_vad_trimmed_chunk(chunk_wav: Path, duration: float, label: str) -> tuple[Path, float, float] | None:
    window = detect_voice_window(chunk_wav, duration, label)
    if window is None:
        return None
    start, speech_duration = window
    if start <= 0.001 and speech_duration >= duration - 0.001:
        return chunk_wav, 0.0, duration
    trimmed = chunk_wav.with_name(f"{chunk_wav.stem}-vad-{uuid.uuid4().hex}.wav")
    extract_wav_chunk(chunk_wav, trimmed, start, speech_duration, label=f"{label} VAD")
    print(
        f"分块 {label}: VAD 裁掉开头 {start:.2f}s，结尾 {max(0.0, duration - start - speech_duration):.2f}s",
        flush=True,
    )
    return trimmed, start, speech_duration


def shift_segments(segments: list[SubtitleSegment], offset: float) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            round(segment.start + offset, 3),
            round(segment.end + offset, 3),
            segment.text,
        )
        for segment in segments
    ]


def shift_words(words: list[SubtitleWord], offset: float) -> list[SubtitleWord]:
    return [
        SubtitleWord(
            round(word.start + offset, 3),
            round(word.end + offset, 3),
            word.text,
            word.probability,
        )
        for word in words
    ]


def trim_segments_to_window(
    segments: list[SubtitleSegment],
    *,
    leading_trim: float,
    trailing_trim: float,
    chunk_duration: float,
) -> list[SubtitleSegment]:
    upper_bound = max(0.0, chunk_duration - trailing_trim)
    trimmed: list[SubtitleSegment] = []
    for segment in segments:
        start = max(leading_trim, min(upper_bound, segment.start))
        end = max(leading_trim, min(upper_bound, segment.end))
        text = " ".join(segment.text.split()).strip()
        if text and end > start:
            trimmed.append(SubtitleSegment(round(start, 3), round(end, 3), text))
    return trimmed


def trim_words_to_window(
    words: list[SubtitleWord],
    *,
    leading_trim: float,
    trailing_trim: float,
    chunk_duration: float,
) -> list[SubtitleWord]:
    upper_bound = max(0.0, chunk_duration - trailing_trim)
    trimmed: list[SubtitleWord] = []
    for word in words:
        start = max(leading_trim, min(upper_bound, word.start))
        end = max(leading_trim, min(upper_bound, word.end))
        text = word.text.strip()
        if text and end > start:
            trimmed.append(SubtitleWord(round(start, 3), round(end, 3), text, word.probability))
    return trimmed


def merge_near_duplicate_segments(segments: list[SubtitleSegment], max_gap: float = 1.5) -> list[SubtitleSegment]:
    merged: list[SubtitleSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        text = " ".join(segment.text.split()).strip()
        if not text:
            continue
        current = SubtitleSegment(segment.start, segment.end, text)
        if (
            merged
            and merged[-1].text == current.text
            and current.start - merged[-1].end <= max_gap
        ):
            previous = merged[-1]
            merged[-1] = SubtitleSegment(previous.start, max(previous.end, current.end), previous.text)
            continue
        merged.append(current)
    return merged


def multipart_body(
    fields: list[tuple[str, str]],
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def read_json_url(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json_url(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, data=b"{}", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def nuc_ssh_password() -> str:
    value = os.environ.get("NUC_SSH_PASSWORD", "").strip()
    if value:
        print(f"NUC SSH/sudo 密码：使用环境变量 {mask_secret(value)}", flush=True)
        return value
    try:
        saved = load_secret(KEYCHAIN_SERVICE, NUC_SSH_KEYCHAIN_ACCOUNT)
    except Exception as exc:
        print(f"Keychain read warning for NUC SSH password: {exc}", flush=True)
        saved = ""
    if not sys.stdin.isatty():
        return saved
    if saved:
        print(f"NUC SSH/sudo 密码：{mask_secret(saved)}", flush=True)
        prompt = "NUC SSH/sudo 密码（按 Enter 沿用；输入新密码将更新保存，输入不显示）: "
    else:
        prompt = "NUC SSH/sudo 密码（未保存；可留空跳过 SSH kill，输入不显示）: "
    entered = getpass.getpass(prompt).strip()
    if not entered:
        return saved
    try:
        save_secret(KEYCHAIN_SERVICE, NUC_SSH_KEYCHAIN_ACCOUNT, entered)
    except Exception as exc:
        print(f"Keychain save warning for NUC SSH password: {exc}", flush=True)
    return entered


def restart_nuc_asr_backend(
    *,
    ssh_target: str,
    password: str,
    scheduler_url: str,
    container: str = DEFAULT_NUC_ASR_BACKEND_CONTAINER,
) -> dict[str, Any]:
    if ssh_target:
        try:
            payload = ssh_nuc_scheduler(
                ssh_target=ssh_target,
                password=password,
                path="/restart/asr",
                method="POST",
            )
            return {"method": "ssh-scheduler", **payload}
        except Exception as exc:
            scheduler_error = str(exc)
        print(
            "SSH 调用 NUC scheduler 重启 ASR 失败，尝试 SSH 直接重启容器："
            f"{scheduler_error}",
            flush=True,
        )

        if shutil.which("sshpass") and password:
            ssh_prefix = [
                "sshpass",
                "-e",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                ssh_target,
            ]
            env = dict(os.environ)
            env["SSHPASS"] = password
            sudo_command = (
                f"printf '%s\\n' {shlex.quote(password)} "
                f"| sudo -S docker restart {shlex.quote(container)}"
            )
            completed = subprocess.run(
                [*ssh_prefix, sudo_command],
                text=True,
                capture_output=True,
                check=False,
                timeout=45,
                env=env,
            )
        else:
            ssh_prefix = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                ssh_target,
            ]
            completed = subprocess.run(
                [*ssh_prefix, f"sudo -n docker restart {shlex.quote(container)}"],
                text=True,
                capture_output=True,
                check=False,
                timeout=45,
            )
        if completed.returncode == 0:
            return {"method": "ssh-docker-restart", "status": "restarted", "container": container}
        detail = completed.stderr.strip() or completed.stdout.strip()
        if scheduler_url:
            print(
                "SSH 直接重启 NUC faster-whisper backend 失败，尝试显式配置的 HTTP scheduler："
                f"{detail}",
                flush=True,
            )
        else:
            print(
                "SSH 直接重启 NUC faster-whisper backend 失败；未配置 Mac 直连 HTTP scheduler fallback。",
                flush=True,
            )
    if scheduler_url:
        return {"method": "http", **post_json_url(f"{scheduler_url.rstrip('/')}/restart/asr", timeout=30)}
    raise RuntimeError("No NUC restart method is available")


def ssh_nuc_scheduler(
    *,
    ssh_target: str,
    password: str,
    path: str,
    method: str = "GET",
) -> dict[str, Any]:
    if not ssh_target:
        raise RuntimeError("NUC SSH target is required")
    curl_method = "-X POST " if method.upper() == "POST" else ""
    remote_command = f"curl -fsS {curl_method}http://127.0.0.1:8010/{path.lstrip('/')}"
    if shutil.which("sshpass") and password:
        command = [
            "sshpass",
            "-e",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=8",
            ssh_target,
            remote_command,
        ]
    else:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            ssh_target,
            remote_command,
        ]
    env = dict(os.environ)
    if password:
        env["SSHPASS"] = password
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "ssh scheduler request failed")
    try:
        return json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"stdout": completed.stdout.strip()}


def submit_nuc_job(
    *,
    wav: Path,
    base_url: str,
    fields: list[tuple[str, str]],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body, boundary = multipart_body(fields, wav)
    print(f"上传到 NUC：{wav.name}（{wav.stat().st_size / 1024 / 1024:.1f} MiB）", flush=True)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/jobs/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=min(timeout, 600)) as response:
        task = json.loads(response.read().decode("utf-8"))
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not task_id:
        raise RuntimeError(f"NUC upload did not return a task id: {task}")
    print(f"NUC 作业：{task_id}", flush=True)

    started = time.monotonic()
    last_status = ""
    last_heartbeat = 0.0
    while True:
        current = read_json_url(f"{base_url.rstrip('/')}/jobs/{task_id}")
        status = str(current.get("status") or "unknown")
        elapsed = time.monotonic() - started
        if status != last_status or elapsed - last_heartbeat >= 20:
            print(f"NUC 状态：{status}，已等待 {elapsed:.0f}s", flush=True)
            last_status = status
            last_heartbeat = elapsed
        if status == "completed":
            result = current.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"NUC job completed without a result: {current}")
            return result, current
        if status == "failed":
            raise RuntimeError(f"NUC job failed: {current.get('error')}")
        if elapsed >= timeout:
            raise TimeoutError(f"NUC job timed out after {timeout:.0f}s: {task_id}")
        time.sleep(5)


def qwen_segments(result: dict[str, Any]) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    for item in result.get("segments") or []:
        text = str(item.get("text") or "").strip()
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and end > start:
            segments.append(SubtitleSegment(start, end, text))
    return segments


def qwen_text_and_segments(result: dict[str, Any]) -> tuple[str, list[SubtitleSegment]]:
    text = str(result.get("text") or "").strip()
    segments = qwen_segments(result)
    if not text:
        text = "".join(segment.text for segment in segments).strip()
    return text, segments


def task_without_result(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != "result"}


def run_qwen_single(wav: Path, base_url: str, timeout: float) -> tuple[str, list[SubtitleSegment], dict[str, Any], dict[str, Any]]:
    result, task = submit_nuc_job(
        wav=wav,
        base_url=base_url,
        fields=[
            ("model", "qwen3-asr-1p7b"),
            ("language", "zh"),
            ("response_format", "verbose_json"),
        ],
        timeout=timeout,
    )
    text, segments = qwen_text_and_segments(result)
    if not text:
        raise RuntimeError("NUC Qwen3-ASR returned no transcript text")
    return text, segments, result, task


def run_qwen(wav: Path, output_dir: Path, base_url: str, timeout: float, chunk_seconds: float = 600.0) -> dict[str, Any]:
    duration = probe_audio_duration(wav)
    chunk_dir = output_dir / "work" / "qwen-chunks"
    chunk_count = max(1, math.ceil(duration / chunk_seconds))
    text_parts: list[str] = []
    segments: list[SubtitleSegment] = []
    raw_chunks: list[dict[str, Any]] = []

    for chunk_index in range(chunk_count):
        chunk_start = chunk_index * chunk_seconds
        chunk_duration = min(chunk_seconds, duration - chunk_start)
        chunk_wav = wav
        if chunk_count > 1:
            chunk_wav = chunk_dir / f"qwen-chunk-{chunk_index + 1:04d}.wav"
            if not chunk_wav.is_file() or chunk_wav.stat().st_size == 0:
                extract_wav_chunk(wav, chunk_wav, chunk_start, chunk_duration, label="Qwen")
            print(
                f"\n[Qwen 分块 {chunk_index + 1}/{chunk_count}："
                f"{chunk_start:.1f}s - {chunk_start + chunk_duration:.1f}s]",
                flush=True,
            )

        chunk_text, chunk_segments, result, task = run_qwen_single(
            chunk_wav,
            base_url,
            timeout,
        )
        text_parts.append(chunk_text)
        segments.extend(shift_segments(chunk_segments, chunk_start if chunk_count > 1 else 0.0))
        raw_chunks.append(
            {
                "index": chunk_index + 1,
                "start": round(chunk_start, 3),
                "duration": round(chunk_duration, 3),
                "wav": str(chunk_wav),
                "task": task_without_result(task),
                "result": result,
            }
        )

    text = "\n".join(part for part in text_parts if part).strip()
    prefix = output_dir / "nuc-qwen3-asr-1.7b"
    transcript = prefix.with_name(prefix.name + "-transcript.txt")
    pseudo_srt = prefix.with_name(prefix.name + "-pseudo-timeline.srt")
    asr_json = prefix.with_name(prefix.name + "-asr.json")
    raw_json = prefix.with_name(prefix.name + "-raw-response.json")
    transcript.write_text(text + "\n", encoding="utf-8")
    save_segments_as_srt(pseudo_srt, segments)
    save_asr_result(
        asr_json,
        ASRResult(
            language="zh",
            words=[],
            segments=segments,
            diagnostics={
                "model": "Qwen/Qwen3-ASR-1.7B",
                "timeline_kind": "pseudo",
                "timeline_is_acoustic_authority": False,
                "chunked_by_tui": chunk_count > 1,
                "chunk_seconds": chunk_seconds if chunk_count > 1 else None,
                "chunks": chunk_count,
            },
        ),
    )
    write_json(
        raw_json,
        {
            "chunked_by_tui": chunk_count > 1,
            "chunk_seconds": chunk_seconds if chunk_count > 1 else None,
            "duration": duration,
            "chunks": raw_chunks,
        },
    )
    return {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "timeline_kind": "pseudo",
        "timeline_is_acoustic_authority": False,
        "chunked_by_tui": chunk_count > 1,
        "chunk_seconds": chunk_seconds if chunk_count > 1 else None,
        "chunks": chunk_count,
        "characters": len(text),
        "segments": len(segments),
        "transcript": str(transcript),
        "srt": str(pseudo_srt),
        "asr_json": str(asr_json),
        "raw_response": str(raw_json),
    }


def validate_word_timeline(result: ASRResult) -> None:
    if not result.words:
        raise RuntimeError("NUC faster-whisper large-v3-turbo returned no word timestamps")
    previous_start = 0.0
    for index, word in enumerate(result.words):
        if word.start < previous_start - 0.001 or word.end <= word.start:
            raise RuntimeError(f"Invalid word timestamp at index {index}")
        previous_start = word.start


def rebuild_native_batch_segments(result: ASRResult, batch_size: int) -> ASRResult:
    if batch_size <= 1:
        return result
    if not result.words:
        raise RuntimeError("NUC native batch ASR returned no word timestamps")
    segments, warnings = build_cues(result.words, result.segments)
    diagnostics = dict(result.diagnostics)
    diagnostics["native_batch_size"] = batch_size
    diagnostics["batched_segments_rebuilt_from_words"] = True
    diagnostics["upstream_segment_count"] = len(result.segments)
    diagnostics["rebuilt_segment_count"] = len(segments)
    if warnings:
        diagnostics.setdefault("capability_warnings", []).extend(warnings)
    return ASRResult(result.language, result.words, segments, diagnostics)


def long_word_gap_regions(
    result: ASRResult,
    *,
    minimum_gap: float = 3.0,
    duration: float | None = None,
) -> list[RetryRegion]:
    words = sorted(result.words, key=lambda item: (item.start, item.end))
    regions: list[RetryRegion] = []
    for previous, current in zip(words, words[1:]):
        gap_start = previous.end
        gap_end = current.start
        if gap_end - gap_start >= minimum_gap:
            regions.append(RetryRegion(gap_start, gap_end, "word timestamp gap"))
    if duration is not None and words:
        if words[0].start >= minimum_gap:
            regions.append(RetryRegion(0.0, words[0].start, "leading word timestamp gap"))
        if duration - words[-1].end >= minimum_gap:
            regions.append(RetryRegion(words[-1].end, duration, "trailing word timestamp gap"))
    return regions


def audit_speech_regions(wav: Path, duration: float) -> list[Any]:
    completed = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(wav),
            "-af",
            "silencedetect=noise=-35dB:d=0.3",
            "-f",
            "null",
            "-",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg silencedetect failed")
    return parse_silencedetect_regions(
        completed.stdout + "\n" + completed.stderr,
        duration,
    )


def repair_faster_whisper_gaps(
    wav: Path,
    result: ASRResult,
    *,
    base_url: str,
    timeout: float,
    model: str,
    output_dir: Path,
    duration: float,
    max_repairs: int = 24,
) -> ASRResult:
    try:
        speech_regions = audit_speech_regions(wav, duration)
        report = audit_asr_result(result, speech_regions, duration=duration)
        uncovered = report.uncovered_regions
        speech_coverage = report.speech_coverage
    except Exception as exc:
        uncovered = []
        speech_coverage = None
        print(f"NUC native batch speech coverage audit unavailable: {exc}", flush=True)

    target_regions = merge_retry_regions(
        long_word_gap_regions(result, duration=duration),
        guard=0.0,
        merge_gap=0.75,
        duration=duration,
    )
    target_regions = [
        region
        for region in target_regions
        if region.end - region.start >= 0.5
    ][:max_repairs]
    if not target_regions:
        diagnostics = dict(result.diagnostics)
        diagnostics["speech_gap_repair"] = {
            "status": "not-needed",
            "speech_coverage": speech_coverage,
        }
        return ASRResult(result.language, result.words, result.segments, diagnostics)

    print(f"NUC native batch gap repair: {len(target_regions)} region(s)", flush=True)
    repair_dir = output_dir / "work" / "faster-whisper-gap-repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repaired_segments: list[SubtitleSegment] = []
    repaired_words: list[SubtitleWord] = []
    attempts: list[dict[str, Any]] = []
    for index, target_region in enumerate(target_regions, 1):
        request = RetryRegion(
            max(0.0, target_region.start - 2.0),
            min(duration, target_region.end + 2.0),
            target_region.reason,
        )
        repair_wav = repair_dir / (
            f"gap-{index:03d}-{request.start:.2f}-{request.end:.2f}.wav"
        )
        if not repair_wav.exists() or repair_wav.stat().st_size == 0:
            extract_wav_chunk(
                wav,
                repair_wav,
                request.start,
                request.end - request.start,
                label="faster-whisper gap repair",
            )
        try:
            local, _raw, task = run_faster_whisper_single_with_vad(
                repair_wav,
                base_url,
                timeout=max(120, min(timeout, (request.end - request.start) * 8)),
                model=model,
            )
        except Exception as exc:
            attempts.append(
                {
                    "start": target_region.start,
                    "end": target_region.end,
                    "request_start": request.start,
                    "request_end": request.end,
                    "reason": target_region.reason,
                    "accepted": False,
                    "error": str(exc) or type(exc).__name__,
                }
            )
            print(f"Gap repair {index}/{len(target_regions)} failed: {exc}", flush=True)
            continue

        local_segments = [
            SubtitleSegment(
                max(target_region.start, request.start + segment.start),
                min(target_region.end, request.start + segment.end),
                segment.text,
            )
            for segment in local.segments
            if (
                segment.text.strip()
                and request.start + segment.end > target_region.start
                and request.start + segment.start < target_region.end
            )
        ]
        local_words = [
            SubtitleWord(
                max(target_region.start, request.start + word.start),
                min(target_region.end, request.start + word.end),
                word.text,
                word.probability,
            )
            for word in local.words
            if (
                word.text.strip()
                and request.start + word.end > target_region.start
                and request.start + word.start < target_region.end
            )
        ]
        local_segments = [segment for segment in local_segments if segment.end > segment.start]
        local_words = [word for word in local_words if word.end > word.start]
        accepted = bool(local_words)
        if accepted:
            repaired_segments.extend(local_segments)
            repaired_words.extend(local_words)
        attempts.append(
            {
                "start": target_region.start,
                "end": target_region.end,
                "request_start": request.start,
                "request_end": request.end,
                "reason": target_region.reason,
                "accepted": accepted,
                "segment_count": len(local_segments),
                "word_count": len(local_words),
                "task": task_without_result(task),
            }
        )
        print(
            f"Gap repair {index}/{len(target_regions)}: "
            f"{'accepted' if accepted else 'no usable words'} "
            f"({len(local_words)} word(s))",
            flush=True,
        )

    accepted_regions = [
        region
        for region, attempt in zip(target_regions, attempts)
        if attempt.get("accepted")
    ]
    diagnostics = dict(result.diagnostics)
    diagnostics["speech_gap_repair"] = {
        "status": "completed" if accepted_regions else "no-improvement",
        "speech_coverage_before": speech_coverage,
        "attempts": attempts,
    }
    if not accepted_regions:
        return ASRResult(result.language, result.words, result.segments, diagnostics)

    original_words = sorted(result.words, key=lambda item: (item.start, item.end))
    filtered_repairs = [
        repair
        for repair in repaired_words
        if not any(
            repair.end > original.start
            and repair.start < original.end
            and repair.text.strip() == original.text.strip()
            for original in original_words
        )
    ]
    words = [*original_words, *filtered_repairs]
    words.sort(key=lambda item: (item.start, item.end))
    segments, warnings = build_cues(words, result.segments)
    if warnings:
        diagnostics.setdefault("capability_warnings", []).extend(warnings)
    diagnostics["speech_gap_repair"]["accepted_regions"] = len(accepted_regions)
    diagnostics["speech_gap_repair"]["words_after"] = len(words)
    diagnostics["speech_gap_repair"]["segments_after"] = len(segments)
    return ASRResult(result.language, words, segments, diagnostics)


def run_faster_whisper(
    wav: Path,
    output_dir: Path,
    base_url: str,
    timeout: float,
    *,
    chunk_seconds: float = 60.0,
    overlap_seconds: float = 2.0,
    replicas: int = 2,
    adaptive_parallel: bool = True,
    remote_vad: bool = True,
    ssh_target: str = DEFAULT_NUC_SSH_TARGET,
    ssh_password: str = "",
    scheduler_url: str = "",
    model: str = DEFAULT_NUC_WHISPER_MODEL,
    native_batch_size: int = DEFAULT_NUC_NATIVE_BATCH_SIZE,
) -> dict[str, Any]:
    duration = probe_audio_duration(wav)
    if native_batch_size > 1:
        print(
            f"启用 NUC native batch_size={native_batch_size}；跳过 Mac 端分块并发，整文件上传。",
            flush=True,
        )
        adaptive_parallel = False
    if not adaptive_parallel or duration <= chunk_seconds * 1.2:
        try:
            asr, result, task = run_faster_whisper_single(
                wav,
                base_url,
                timeout,
                model=model,
                native_batch_size=native_batch_size,
            )
            asr = rebuild_native_batch_segments(asr, native_batch_size)
            if native_batch_size > 1:
                asr = repair_faster_whisper_gaps(
                    wav,
                    asr,
                    base_url=base_url,
                    timeout=timeout,
                    model=model,
                    output_dir=output_dir,
                    duration=duration,
                )
        except Exception as exc:
            if native_batch_size <= 1:
                raise
            print(
                "NUC native batch ASR failed quality checks; falling back to chunked upload: "
                f"{exc}",
                flush=True,
            )
            asr = run_faster_whisper_chunked(
                wav,
                output_dir,
                base_url,
                timeout,
                duration=duration,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                replicas=replicas,
                remote_vad=remote_vad,
                ssh_target=ssh_target,
                ssh_password=ssh_password,
                scheduler_url=scheduler_url,
                model=model,
            )
            asr.diagnostics["native_batch_fallback"] = {
                "batch_size": native_batch_size,
                "reason": str(exc) or type(exc).__name__,
                "fallback": "chunked_upload",
            }
            result = {"text": "".join(segment.text for segment in asr.segments)}
            task = {}
        return save_faster_whisper_outputs(
            output_dir,
            asr,
            result,
            {
                "chunked_by_tui": False,
                "native_batch_size": native_batch_size if native_batch_size > 1 else None,
                "task": task_without_result(task),
                "result": result,
            },
            model=model,
        )

    asr = run_faster_whisper_chunked(
        wav,
        output_dir,
        base_url,
        timeout,
        duration=duration,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        replicas=replicas,
        remote_vad=remote_vad,
        ssh_target=ssh_target,
        ssh_password=ssh_password,
        scheduler_url=scheduler_url,
        model=model,
    )
    return save_faster_whisper_outputs(
        output_dir,
        asr,
        {
            "text": "".join(segment.text for segment in asr.segments),
        },
        asr.diagnostics.get("raw_response", {}),
        model=model,
    )


def run_faster_whisper_single(
    wav: Path,
    base_url: str,
    timeout: float,
    model: str = DEFAULT_NUC_WHISPER_MODEL,
    native_batch_size: int = 0,
) -> tuple[ASRResult, dict[str, Any], dict[str, Any]]:
    fields = [
        ("model", model),
        ("language", "auto"),
        ("response_format", "verbose_json"),
        ("vad_filter", "true" if native_batch_size > 1 else "false"),
        ("timestamp_granularities[]", "word"),
    ]
    if native_batch_size > 1:
        fields.append(("batch_size", str(native_batch_size)))
    result, task = submit_nuc_job(
        wav=wav,
        base_url=base_url,
        fields=fields,
        timeout=timeout,
    )
    asr = parse_verbose_asr_response(result, requested_words=True)
    validate_word_timeline(asr)
    return asr, result, task


def save_faster_whisper_outputs(
    output_dir: Path,
    asr: ASRResult,
    result: dict[str, Any],
    raw_payload: dict[str, Any],
    model: str = DEFAULT_NUC_WHISPER_MODEL,
) -> dict[str, Any]:
    text = str(result.get("text") or "").strip()
    if not text:
        text = "".join(segment.text for segment in asr.segments).strip()
    prefix = output_dir / "nuc-faster-whisper-large-v3"
    transcript = prefix.with_name(prefix.name + "-transcript.txt")
    srt = prefix.with_name(prefix.name + "-original-segments.srt")
    word_json = prefix.with_name(prefix.name + "-word-timestamps.json")
    raw_json = prefix.with_name(prefix.name + "-raw-response.json")
    asr.diagnostics.pop("upstream_response", None)
    asr.diagnostics["raw_response_path"] = str(raw_json)
    transcript.write_text(text + "\n", encoding="utf-8")
    save_segments_as_srt(srt, asr.segments)
    save_asr_result(word_json, asr)
    write_json(raw_json, raw_payload)
    return {
        "model": model,
        "timeline_kind": "acoustic-word-timestamps",
        "timeline_is_acoustic_authority": True,
        "characters": len(text),
        "segments": len(asr.segments),
        "words": len(asr.words),
        "transcript": str(transcript),
        "srt": str(srt),
        "asr_json": str(word_json),
        "raw_response": str(raw_json),
    }


def run_faster_whisper_chunk_task(
    source_wav: Path,
    chunk_dir: Path,
    task: ChunkWindow,
    base_url: str,
    timeout: float,
    model: str,
    remote_vad: bool,
) -> tuple[dict[str, Any], ASRResult, dict[str, Any], dict[str, Any], float]:
    started = time.monotonic()
    start = task.start
    duration = task.duration
    label = task.label
    chunk_wav = chunk_dir / f"whisper-chunk-{safe_name(label)}.wav"
    if not chunk_wav.is_file() or chunk_wav.stat().st_size == 0:
        extract_wav_chunk(source_wav, chunk_wav, start, duration, label="faster-whisper")
    print(
        f"\n[faster-whisper 分块 {label}：实际 {start:.1f}s - {start + duration:.1f}s，"
        f"保留 {task.timeline_start:.1f}s - {task.timeline_start + duration - task.leading_trim - task.trailing_trim:.1f}s]",
        flush=True,
    )
    request_wav = chunk_wav
    vad_offset = 0.0
    request_duration = duration
    if remote_vad:
        vad_result = prepare_vad_trimmed_chunk(chunk_wav, duration, label)
        if vad_result is None:
            empty = ASRResult(
                language="",
                words=[],
                segments=[],
                diagnostics={"capability_warnings": ["ffmpeg found no stable voice window"]},
            )
            return task.__dict__, empty, {"text": ""}, {}, time.monotonic() - started
        request_wav, vad_offset, request_duration = vad_result
    asr, result, nuc_task = run_faster_whisper_single(
        request_wav,
        base_url,
        timeout=max(180, min(timeout, request_duration * 8)),
        model=model,
    )
    local_segments = shift_segments(asr.segments, vad_offset)
    issue = remote_asr_quality_issue(local_segments)
    if issue and not remote_vad:
        print(f"分块 {label} 疑似不稳定（{issue}），使用 faster-whisper vad_filter=true 重试", flush=True)
        result_obj, result, nuc_task = run_faster_whisper_single_with_vad(
            chunk_wav,
            base_url,
            timeout=max(180, min(timeout, duration * 8)),
            model=model,
        )
        asr = result_obj
        local_segments = asr.segments
    elif issue:
        raise RuntimeError(f"faster-whisper chunk {label} unstable after VAD trim: {issue}")
    local_words = shift_words(asr.words, vad_offset)
    local_segments = trim_segments_to_window(
        local_segments,
        leading_trim=task.leading_trim,
        trailing_trim=task.trailing_trim,
        chunk_duration=duration,
    )
    local_words = trim_words_to_window(
        local_words,
        leading_trim=task.leading_trim,
        trailing_trim=task.trailing_trim,
        chunk_duration=duration,
    )
    shifted = ASRResult(
        language=asr.language,
        words=shift_words(local_words, start),
        segments=shift_segments(local_segments, start),
        diagnostics=dict(asr.diagnostics),
    )
    return task.__dict__, shifted, result, nuc_task, time.monotonic() - started


def run_faster_whisper_single_with_vad(
    wav: Path,
    base_url: str,
    timeout: float,
    model: str = DEFAULT_NUC_WHISPER_MODEL,
) -> tuple[ASRResult, dict[str, Any], dict[str, Any]]:
    result, task = submit_nuc_job(
        wav=wav,
        base_url=base_url,
        fields=[
            ("model", model),
            ("language", "auto"),
            ("response_format", "verbose_json"),
            ("vad_filter", "true"),
            ("timestamp_granularities[]", "word"),
        ],
        timeout=timeout,
    )
    asr = parse_verbose_asr_response(result, requested_words=True)
    validate_word_timeline(asr)
    return asr, result, task


def split_whisper_task(task: ChunkWindow) -> list[ChunkWindow]:
    duration = task.duration
    half = duration / 2
    depth = task.depth + 1
    return [
        ChunkWindow(
            label=f"{task.label}a",
            start=task.start,
            duration=half,
            timeline_start=task.timeline_start,
            leading_trim=task.leading_trim,
            trailing_trim=0.0,
            depth=depth,
        ),
        ChunkWindow(
            label=f"{task.label}b",
            start=task.start + half,
            duration=duration - half,
            timeline_start=task.start + half,
            leading_trim=0.0,
            trailing_trim=task.trailing_trim,
            depth=depth,
        ),
    ]


def whisper_chunk_windows(duration: float, chunk_seconds: float, overlap_seconds: float) -> list[ChunkWindow]:
    if duration <= 0:
        return []
    chunk_seconds = max(1.0, chunk_seconds)
    overlap_seconds = max(0.0, min(overlap_seconds, chunk_seconds / 2))
    windows: list[ChunkWindow] = []
    for index in range(math.ceil(duration / chunk_seconds)):
        timeline_start = index * chunk_seconds
        timeline_duration = min(chunk_seconds, duration - timeline_start)
        actual_start = max(0.0, timeline_start - (overlap_seconds if index > 0 else 0.0))
        leading_trim = timeline_start - actual_start
        actual_end = min(duration, timeline_start + timeline_duration + overlap_seconds)
        trailing_trim = max(0.0, actual_end - (timeline_start + timeline_duration))
        windows.append(
            ChunkWindow(
                label=str(index + 1),
                start=actual_start,
                duration=actual_end - actual_start,
                timeline_start=timeline_start,
                leading_trim=leading_trim,
                trailing_trim=trailing_trim,
            )
        )
    return windows


def run_faster_whisper_chunked(
    wav: Path,
    output_dir: Path,
    base_url: str,
    timeout: float,
    *,
    duration: float,
    chunk_seconds: float,
    overlap_seconds: float,
    replicas: int,
    remote_vad: bool,
    ssh_target: str,
    ssh_password: str,
    scheduler_url: str,
    model: str = DEFAULT_NUC_WHISPER_MODEL,
) -> ASRResult:
    chunk_dir = output_dir / "work" / "faster-whisper-chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending = whisper_chunk_windows(duration, chunk_seconds, overlap_seconds)
    active: dict[Future, ChunkWindow] = {}
    started_at: dict[Future, float] = {}
    segments: list[SubtitleSegment] = []
    words: list[SubtitleWord] = []
    raw_chunks: list[dict[str, Any]] = []
    successful_root_times: list[float] = []
    max_workers = max(1, min(4, replicas))
    minimum_split_seconds = 20.0
    threshold_multiplier = 1.5

    def submit(executor: ThreadPoolExecutor, task: ChunkWindow) -> None:
        future = executor.submit(
            run_faster_whisper_chunk_task,
            wav,
            chunk_dir,
            task,
            base_url,
            timeout,
            model,
            remote_vad,
        )
        active[future] = task
        started_at[future] = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nuc-whisper") as executor:
        while pending or active:
            while pending and len(active) < max_workers:
                submit(executor, pending.pop(0))

            threshold = 0.0
            if len(successful_root_times) >= 3:
                threshold = max(10.0, min(successful_root_times[:3]) * threshold_multiplier)
            if threshold:
                for future, task in list(active.items()):
                    elapsed = time.monotonic() - started_at[future]
                    if task.depth == 0 and not future.done() and elapsed > threshold:
                        print(
                            f"分块 {task.label} 超过动态阈值 {threshold:.1f}s；"
                            "SSH 优先重启 NUC faster-whisper backend 并拆分重试。",
                            flush=True,
                        )
                        try:
                            restart = restart_nuc_asr_backend(
                                ssh_target=ssh_target,
                                password=ssh_password,
                                scheduler_url=scheduler_url,
                            )
                            print(f"NUC ASR backend restart: {restart}", flush=True)
                        except Exception as exc:
                            print(f"NUC ASR backend restart failed: {exc}", flush=True)
                        active.pop(future, None)
                        started_at.pop(future, None)
                        if task.duration >= minimum_split_seconds:
                            pending[0:0] = split_whisper_task(task)
                        else:
                            pending.insert(0, ChunkWindow(**{**task.__dict__, "attempts": task.attempts + 1}))
                        break

            if not active:
                continue
            done, _not_done = wait(active.keys(), timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                started_at.pop(future, None)
                try:
                    completed_task, asr, result, nuc_task, elapsed = future.result()
                except Exception as exc:
                    attempts = task.attempts
                    if attempts < 1:
                        print(f"分块 {task.label} 失败，重试一次：{exc}", flush=True)
                        pending.insert(0, ChunkWindow(**{**task.__dict__, "attempts": attempts + 1}))
                    elif task.duration >= minimum_split_seconds:
                        print(f"分块 {task.label} 重试失败，拆分继续：{exc}", flush=True)
                        pending[0:0] = split_whisper_task(task)
                    else:
                        for active_future in active:
                            active_future.cancel()
                        raise RuntimeError(f"faster-whisper chunk {task.label} failed: {exc}") from exc
                    continue
                if int(completed_task.get("depth", 0)) == 0:
                    successful_root_times.append(elapsed)
                segments.extend(asr.segments)
                words.extend(asr.words)
                raw_chunks.append(
                    {
                        "task": completed_task,
                        "elapsed_seconds": round(elapsed, 3),
                        "nuc_task": task_without_result(nuc_task),
                        "result": result,
                    }
                )

    combined = ASRResult(
        language="",
        words=sorted(words, key=lambda item: (item.start, item.end)),
        segments=merge_near_duplicate_segments(segments),
        diagnostics={
            "model": model,
            "timeline_kind": "acoustic-word-timestamps",
            "timeline_is_acoustic_authority": True,
            "chunked_by_tui": True,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "replicas": max_workers,
            "remote_vad": remote_vad,
            "adaptive_split": True,
            "restart_policy": "ssh_first_http_scheduler_fallback",
            "raw_response": {
                "chunked_by_tui": True,
                "duration": duration,
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": overlap_seconds,
                "replicas": max_workers,
                "remote_vad": remote_vad,
                "chunks": raw_chunks,
            },
        },
    )
    validate_word_timeline(combined)
    return combined


def default_nuc_output(source: Path) -> Path:
    return GENERATED_DIR / f"{safe_name(source.stem)}-本地NUC-ASR"


def completed_stage(stage: dict[str, Any]) -> bool:
    if stage.get("status") != "completed":
        return False
    outputs = [stage.get(key) for key in ("transcript", "srt", "asr_json", "raw_response")]
    return all(
        isinstance(value, str) and Path(value).is_file() and Path(value).stat().st_size > 0
        for value in outputs
    )


def run_nuc_local(args: argparse.Namespace) -> int:
    source = args.audio.expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Local audio file not found: {source}")
    output_dir = (args.output_dir or default_nuc_output(source)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "asr-manifest.json"
    manifest = manifest_start(
        manifest_path,
        mode=f"nuc-local-{args.backend}",
        title=f"Local NUC ASR: {source.name}",
        source=str(source),
        backend=args.backend,
        stages={},
    )
    try:
        wav = normalized_wav(source, output_dir)
        duration = probe_audio_duration(wav)
        print(f"音频时长：{format_seconds(duration)}（{duration:.3f}s）", flush=True)
        manifest["normalized_wav"] = str(wav)
        manifest["duration_seconds"] = round(duration, 3)
        manifest_write(manifest_path, manifest)

        stages = manifest.setdefault("stages", {})
        if args.gemini_asr:
            if completed_gemini_local_stage(stages.get("gemini", {})):
                print("\n[复用已完成的 Gemini OGG/File API 全文 ASR 产物]", flush=True)
            else:
                print("\n[Gemini OGG/File API 全文 ASR：文字覆盖补充]", flush=True)
                stages["gemini"] = {"status": "running", "updated_at": now()}
                manifest_write(manifest_path, manifest)
                try:
                    gemini = run_gemini_local_audio(
                        source,
                        output_dir,
                        model=args.gemini_model,
                        timeout=args.gemini_timeout,
                        upload_timeout=args.gemini_upload_timeout,
                        processing_timeout=args.gemini_processing_timeout,
                    )
                except BaseException as exc:
                    stages["gemini"] = {
                        "status": "failed",
                        "updated_at": now(),
                        "error": str(exc) or type(exc).__name__,
                    }
                    manifest_write(manifest_path, manifest)
                    raise
                stages["gemini"] = {"status": "completed", "updated_at": now(), **gemini}
                manifest_write(manifest_path, manifest)
        if args.backend in {"qwen", "both"}:
            if completed_stage(stages.get("qwen", {})):
                print("\n[复用已完成的 NUC Qwen3-ASR 产物]", flush=True)
            else:
                print("\n[NUC Qwen3-ASR 1.7B：文字准确优先，时间轴仅为伪时间轴]", flush=True)
                stages["qwen"] = {"status": "running", "updated_at": now()}
                manifest_write(manifest_path, manifest)
                try:
                    qwen = run_qwen(
                        wav,
                        output_dir,
                        args.qwen_url,
                        args.timeout,
                        chunk_seconds=args.qwen_chunk_seconds,
                    )
                except BaseException as exc:
                    stages["qwen"] = {
                        "status": "failed",
                        "updated_at": now(),
                        "error": str(exc) or type(exc).__name__,
                    }
                    manifest_write(manifest_path, manifest)
                    raise
                stages["qwen"] = {"status": "completed", "updated_at": now(), **qwen}
                manifest_write(manifest_path, manifest)
        if args.backend in {"whisper", "both"}:
            if completed_stage(stages.get("whisper", {})):
                print("\n[复用已完成的 NUC faster-whisper large-v3-turbo 产物]", flush=True)
            else:
                print("\n[NUC faster-whisper large-v3-turbo：词级时间轴优先]", flush=True)
                stages["whisper"] = {"status": "running", "updated_at": now()}
                manifest_write(manifest_path, manifest)
                try:
                    ssh_password = ""
                    if args.whisper_adaptive_parallel and args.nuc_ssh_target:
                        ssh_password = nuc_ssh_password()
                    whisper = run_faster_whisper(
                        wav,
                        output_dir,
                        args.whisper_url,
                        args.timeout,
                        chunk_seconds=args.whisper_chunk_seconds,
                        overlap_seconds=args.whisper_overlap_seconds,
                        replicas=args.whisper_replicas,
                        adaptive_parallel=args.whisper_adaptive_parallel,
                        remote_vad=args.whisper_remote_vad,
                        ssh_target=args.nuc_ssh_target,
                        ssh_password=ssh_password,
                        scheduler_url=args.scheduler_url,
                        model=args.whisper_model,
                        native_batch_size=args.whisper_native_batch_size,
                    )
                except BaseException as exc:
                    stages["whisper"] = {
                        "status": "failed",
                        "updated_at": now(),
                        "error": str(exc) or type(exc).__name__,
                    }
                    manifest_write(manifest_path, manifest)
                    raise
                stages["whisper"] = {
                    "status": "completed",
                    "updated_at": now(),
                    **whisper,
                }
                manifest_write(manifest_path, manifest)

        if args.backend in {"whisper", "both"}:
            recommended = stages.get("whisper", {}).get("asr_json")
        else:
            recommended = stages.get("qwen", {}).get("transcript")
        manifest.update(status="completed", recommended_output=recommended)
        manifest_write(manifest_path, manifest)
        print(f"\n完成。产物目录：{output_dir}", flush=True)
        return 0
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc) or type(exc).__name__,
        )
        manifest_write(manifest_path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gemini = subparsers.add_parser(
        "gemini-url",
        help="Transcribe a public YouTube URL without yt-dlp or visual analysis output",
    )
    gemini.add_argument("url")
    gemini.add_argument("--output-dir", type=Path)
    gemini.add_argument("--model", default="gemini-2.5-flash")
    gemini.add_argument("--timeout", type=float, default=1200.0)
    gemini.set_defaults(handler=run_gemini_url)

    gemini_local = subparsers.add_parser(
        "gemini-local",
        help="Transcribe a local audio/video file through Gemini OGG/File API",
    )
    gemini_local.add_argument("audio", type=Path, metavar="MEDIA")
    gemini_local.add_argument("--output-dir", type=Path)
    gemini_local.add_argument("--model", default="gemini-2.5-flash")
    gemini_local.add_argument("--timeout", type=int, default=900)
    gemini_local.add_argument("--upload-timeout", type=int, default=300)
    gemini_local.add_argument("--processing-timeout", type=int, default=1200)
    gemini_local.set_defaults(handler=run_gemini_local)

    nuc = subparsers.add_parser("nuc-local", help="Transcribe a local audio/video media file on the NUC")
    nuc.add_argument("audio", type=Path, metavar="MEDIA")
    nuc.add_argument("--backend", choices=("qwen", "whisper", "both"), default="whisper")
    nuc.add_argument("--output-dir", type=Path)
    nuc.add_argument("--qwen-url", default=f"http://{NUC_OLLAMA_HOST}:8001")
    nuc.add_argument("--whisper-url", default=f"http://{NUC_OLLAMA_HOST}:8000")
    nuc.add_argument("--gemini-asr", action="store_true", help="Also run local OGG/File API Gemini full-text ASR in this output directory.")
    nuc.add_argument("--gemini-model", default="gemini-2.5-flash")
    nuc.add_argument("--gemini-timeout", type=int, default=900)
    nuc.add_argument("--gemini-upload-timeout", type=int, default=300)
    nuc.add_argument("--gemini-processing-timeout", type=int, default=1200)
    nuc.add_argument("--qwen-chunk-seconds", type=float, default=600.0)
    nuc.add_argument("--whisper-chunk-seconds", type=float, default=60.0)
    nuc.add_argument("--whisper-overlap-seconds", type=float, default=2.0)
    nuc.add_argument("--whisper-replicas", type=int, default=2)
    nuc.add_argument("--whisper-model", default=DEFAULT_NUC_WHISPER_MODEL)
    nuc.add_argument(
        "--whisper-native-batch-size",
        type=int,
        default=DEFAULT_NUC_NATIVE_BATCH_SIZE,
        help="Use a NUC server-side native batch size and disable Mac-side chunked parallel upload. 0 uses the legacy chunked path.",
    )
    nuc.add_argument("--whisper-adaptive-parallel", action=argparse.BooleanOptionalAction, default=True)
    nuc.add_argument("--whisper-remote-vad", action=argparse.BooleanOptionalAction, default=True)
    nuc.add_argument("--nuc-ssh-target", default=DEFAULT_NUC_SSH_TARGET)
    nuc.add_argument(
        "--scheduler-url",
        default="",
        help="Optional direct HTTP scheduler URL. Empty by default because NUC scheduler normally binds to 127.0.0.1.",
    )
    nuc.add_argument("--timeout", type=float, default=7200.0)
    nuc.set_defaults(handler=run_nuc_local)

    scheduler = subparsers.add_parser(
        "nuc-scheduler",
        help="Access the NUC-local scheduler through SSH without exposing port 8010",
    )
    scheduler.add_argument("action", choices=("status", "restart-asr"))
    scheduler.add_argument("--nuc-ssh-target", default=DEFAULT_NUC_SSH_TARGET)
    scheduler.set_defaults(handler=run_nuc_scheduler)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("\n已中断。NUC 已接受的后台作业可能仍会完成。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1


def run_nuc_scheduler(args: argparse.Namespace) -> int:
    password = nuc_ssh_password() if args.nuc_ssh_target else ""
    if args.action == "status":
        payload = ssh_nuc_scheduler(
            ssh_target=args.nuc_ssh_target,
            password=password,
            path="/status",
            method="GET",
        )
    elif args.action == "restart-asr":
        payload = ssh_nuc_scheduler(
            ssh_target=args.nuc_ssh_target,
            password=password,
            path="/restart/asr",
            method="POST",
        )
    else:
        raise RuntimeError(f"Unsupported scheduler action: {args.action}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
