#!/usr/bin/env python3
"""Resumable TUI worker for the final forensic subtitle pipeline.

The user-facing menu lives in ``ForensicSubtitle.command``. This module runs
the full pipeline and exposes status/doctor helpers for that shell TUI.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.config import FFMPEG, FFPROBE, GENERATED_DIR, NUC_OLLAMA_HOST, YT_DLP
from whisper_captioner.credentials import load_secret, save_secret
from whisper_captioner.external_backends import gemini_transcribe_audio
from whisper_captioner.subtitle_io import save_asr_result, save_segments_as_srt
from whisper_captioner.workers import _transcribe_via_nuc_asr_result
from scripts.chrome_cookie_snapshot import YtDlpCookieSession, yt_dlp_cookie_session

KEYCHAIN_SERVICE = "WhisperCaptioner"
GEMINI_KEYCHAIN_ACCOUNT = "gemini-api-key"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def safe_name(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "-", value).strip(" .-")
    return (value or "forensic-subtitle")[:120]


def fallback_identity(source: str) -> str:
    if not is_url(source):
        return Path(source).stem
    parsed = urlparse(source)
    return parse_qs(parsed.query).get("v", [parsed.path.rsplit("/", 1)[-1]])[0]


def run(command: list[str], label: str, cwd: Path | None = None) -> None:
    print(f"\n[{label}]", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd or PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now()
    write_json(path, manifest)


def mark_stage(
    manifest_path: Path,
    manifest: dict[str, Any],
    key: str,
    status: str,
    **details: Any,
) -> None:
    manifest.setdefault("stages", {})[key] = {
        "status": status,
        "updated_at": now(),
        **details,
    }
    write_manifest(manifest_path, manifest)


def stage_ready(manifest: dict[str, Any], key: str, paths: list[Path]) -> bool:
    stage = manifest.get("stages", {}).get(key, {})
    return stage.get("status") == "completed" and all(
        path.exists() and (not path.is_file() or path.stat().st_size > 0)
        for path in paths
    )


def source_metadata(source: str, cookie_args: list[str]) -> dict[str, Any]:
    if not is_url(source):
        path = Path(source).expanduser().resolve()
        return {"id": path.stem, "title": path.stem, "duration": None}
    command = [YT_DLP, "--no-playlist", "--dump-single-json", "--skip-download"]
    command.extend(cookie_args)
    command.append(source)
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def resolve_output_dir(
    source: str,
    requested: Path | None,
    cookie_args: list[str],
) -> tuple[Path, dict[str, Any]]:
    metadata = source_metadata(source, cookie_args)
    if requested is not None:
        return requested.expanduser().resolve(), metadata
    identity = str(metadata.get("id") or fallback_identity(source))
    title = safe_name(str(metadata.get("title") or identity))
    return (GENERATED_DIR / f"{title} [{identity}]").resolve(), metadata


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        key = load_secret(KEYCHAIN_SERVICE, GEMINI_KEYCHAIN_ACCOUNT)
    except Exception as exc:
        print(f"Keychain read warning: {exc}", flush=True)
    if key:
        return key
    if not sys.stdin.isatty():
        raise RuntimeError("GEMINI_API_KEY is missing and no interactive terminal is available")
    key = getpass.getpass("Gemini API Key（输入不会显示）: ").strip()
    if not key:
        raise RuntimeError("Gemini API Key is required")
    answer = input("保存到 macOS Keychain，供以后续跑使用？[Y/n]: ").strip().lower()
    if answer not in {"n", "no"}:
        save_secret(KEYCHAIN_SERVICE, GEMINI_KEYCHAIN_ACCOUNT, key)
    return key


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


def prepare_audio(source: str, work: Path, cookie_args: list[str]) -> tuple[Path, Path]:
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio-16k-mono.wav"
    ogg = work / "gemini-audio.ogg"
    if wav.exists() and ogg.exists():
        return wav, ogg
    if is_url(source):
        template = work / "source-audio.%(ext)s"
        command = [YT_DLP, "--no-playlist", "-f", "bestaudio[ext=webm]/bestaudio"]
        command.extend(cookie_args)
        command.extend(["-o", str(template), source])
        run(command, "下载一次原始音频", work)
        candidates = sorted(work.glob("source-audio.*"))
        source_audio = next((item for item in candidates if item.is_file()), None)
        if source_audio is None:
            raise RuntimeError("yt-dlp did not produce source audio")
    else:
        source_audio = Path(source).expanduser().resolve()
        if not source_audio.is_file():
            raise RuntimeError(f"Local source not found: {source_audio}")
    stream_info = media_stream_info(source_audio)
    print_media_stream_info(stream_info)
    if not wav.exists():
        run(
            [
                FFMPEG,
                "-hide_banner",
                "-y",
                "-i",
                str(source_audio),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav),
            ],
            "生成 NUC 16kHz WAV",
            work,
        )
    if not ogg.exists():
        run(
            [
                FFMPEG,
                "-hide_banner",
                "-y",
                "-i",
                str(source_audio),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "libopus",
                "-b:a",
                "64k",
                str(ogg),
            ],
            "生成 Gemini OGG",
            work,
        )
    return wav, ogg


def validate_words(asr: Any) -> None:
    if not asr.words:
        raise RuntimeError("NUC ASR did not return word timestamps")
    previous = 0.0
    for index, word in enumerate(asr.words):
        if word.start < previous - 0.001 or word.end < word.start:
            raise RuntimeError(f"Invalid NUC word timeline at index {index}")
        previous = word.start


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


def validate_final(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cues = payload.get("cues", [])
    if not cues:
        raise RuntimeError("Final timeline contains no cues")
    previous_end = 0.0
    for index, cue in enumerate(cues):
        start, end = float(cue["start"]), float(cue["end"])
        if not str(cue.get("text", "")).strip() or start < previous_end - 0.001 or end <= start:
            raise RuntimeError(f"Invalid final cue at index {index}")
        previous_end = end
    return {"cues": len(cues), "end_seconds": previous_end}


def run_pipeline(args: argparse.Namespace, cookie_session: YtDlpCookieSession) -> int:
    source = args.source.strip()
    if not source:
        raise RuntimeError("Source URL/path is required")
    cookies = args.cookies_from_chrome
    cookie_args = cookie_session.arguments
    output_dir, metadata = resolve_output_dir(source, args.output_dir, cookie_args)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pipeline-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "pipeline": "final-forensic-subtitle-v1",
            "source": source,
            "title": metadata.get("title"),
            "video_id": metadata.get("id"),
            "created_at": now(),
            "stages": {},
        }
    args._manifest_path = manifest_path
    args._manifest = manifest
    manifest.pop("error", None)
    manifest.update(
        status="running",
        ocr_mode=args.ocr,
        cookies_from_chrome=cookies,
        chrome_profile=args.chrome_profile if cookies else None,
    )
    write_manifest(manifest_path, manifest)
    print(f"\n作业目录: {output_dir}", flush=True)

    probe_dir = output_dir / "00-hard-subtitle-probe"
    probe_report = probe_dir / "hard-subtitle-probe.json"
    if args.ocr == "off":
        hard_subtitles = False
        mark_stage(manifest_path, manifest, "hard_subtitle_probe", "skipped", reason="ocr_off")
    elif args.ocr == "on":
        hard_subtitles = True
        mark_stage(manifest_path, manifest, "hard_subtitle_probe", "completed", forced=True)
    else:
        if not stage_ready(manifest, "hard_subtitle_probe", [probe_report]):
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/forensic_subtitle_command.py"),
                "probe-hard-subs",
                source,
                "--output-dir",
                str(probe_dir),
            ]
            if cookie_session.browser_spec:
                command.extend(["--cookies-browser-spec", cookie_session.browser_spec])
            run(command, "阶段 0/6：低清短片段硬字幕预检")
            decision = json.loads(probe_report.read_text(encoding="utf-8"))
            mark_stage(
                manifest_path,
                manifest,
                "hard_subtitle_probe",
                "completed",
                report=str(probe_report),
                burned_subtitles_present=bool(decision.get("burned_subtitles_present")),
            )
        hard_subtitles = bool(
            json.loads(probe_report.read_text(encoding="utf-8")).get("burned_subtitles_present")
        )
    print(f"硬字幕预检: {'检测到' if hard_subtitles else '未检测到/已跳过'}", flush=True)

    work = output_dir / "work"
    wav, ogg = prepare_audio(source, work, cookie_args)
    audio_duration = probe_audio_duration(wav)
    print(f"音频时长：{format_seconds(audio_duration)}（{audio_duration:.3f}s）", flush=True)
    mark_stage(
        manifest_path,
        manifest,
        "audio",
        "completed",
        wav=str(wav),
        ogg=str(ogg),
        duration_seconds=round(audio_duration, 3),
    )

    asr_path = output_dir / "01-nuc-word-asr.json"
    nuc_srt = output_dir / "01-nuc-original.srt"
    if not stage_ready(manifest, "nuc_asr", [asr_path, nuc_srt]):
        print("\n[阶段 1/6：NUC 高精度词级 ASR]", flush=True)
        asr = _transcribe_via_nuc_asr_result(
            wav,
            base_url=args.nuc_url,
            model="deepdml/faster-whisper-large-v3-turbo-ct2",
            timeout=7200,
        )
        validate_words(asr)
        save_asr_result(asr_path, asr)
        save_segments_as_srt(nuc_srt, asr.segments)
        mark_stage(manifest_path, manifest, "nuc_asr", "completed", json=str(asr_path), srt=str(nuc_srt), words=len(asr.words))

    gemini_path = output_dir / "02-gemini-transcript.txt"
    gemini_meta = output_dir / "02-gemini-request.json"
    if not stage_ready(manifest, "gemini_asr", [gemini_path, gemini_meta]):
        print("\n[阶段 2/6：Gemini OGG 全文 ASR]", flush=True)
        result = gemini_transcribe_audio(
            ogg,
            gemini_api_key(),
            model=args.gemini_model,
            timeout=900,
            upload_timeout=300,
            processing_timeout=1200,
            progress_callback=lambda message: print(f"Gemini: {message}", flush=True),
        )
        if result.status != "completed" or not result.text.strip():
            raise RuntimeError(f"Gemini ASR failed: {result.warning or result.status}")
        gemini_path.write_text(result.text.strip() + "\n", encoding="utf-8")
        write_json(gemini_meta, {"model": result.model, "elapsed": result.elapsed, "diagnostics": result.diagnostics})
        mark_stage(manifest_path, manifest, "gemini_asr", "completed", transcript=str(gemini_path), metadata=str(gemini_meta))

    draft_dir = output_dir / "03-gemini-backfill"
    differences = draft_dir / "transcript-differences.json"
    if not stage_ready(manifest, "gemini_backfill", [draft_dir / "gemini-backfilled-local-timeline.srt", differences]):
        run(
            [sys.executable, str(PROJECT_ROOT / "scripts/forensic_subtitle_command.py"), "finalize", "--nuc-asr", str(asr_path), "--gemini", str(gemini_path), "--output-dir", str(draft_dir)],
            "阶段 3/6：Gemini 全文回填本地时间轴并定位争议",
        )
        mark_stage(manifest_path, manifest, "gemini_backfill", "completed", srt=str(draft_dir / "gemini-backfilled-local-timeline.srt"), differences=str(differences))

    final_transcript = output_dir / "05-final-transcript.txt"
    if hard_subtitles:
        ocr_dir = output_dir / "04-targeted-ocr"
        plan = ocr_dir / "targeted-ocr-plan.json"
        frames = ocr_dir / "frames"
        raw = ocr_dir / "apple-vision-raw.jsonl"
        merged = ocr_dir / "merged"
        ocr_json = merged / "burned-subtitle-ocr.json"
        adjudications = ocr_dir / "ocr-adjudications.json"
        if not stage_ready(manifest, "ocr_adjudication", [ocr_json, adjudications, final_transcript]):
            ocr_dir.mkdir(parents=True, exist_ok=True)
            run(
                [sys.executable, str(PROJECT_ROOT / "scripts/plan_targeted_ocr_frames.py"), "--ogg-transcript", str(gemini_path), "--nuc-asr", str(asr_path), "--differences", str(differences), "--output", str(plan)],
                "阶段 4/6：规划争议窗口 OCR",
            )
            extract_command = [sys.executable, str(PROJECT_ROOT / "scripts/extract_url_targeted_ocr_frames.py"), "--source", source, "--plan", str(plan), "--output-dir", str(frames)]
            if cookie_session.browser_spec:
                extract_command.extend(["--cookies-browser-spec", cookie_session.browser_spec])
            run(extract_command, "按争议窗口远程取帧（不下载全片）")
            binary = ocr_dir / "apple-vision-ocr"
            module_cache = ocr_dir / "swift-module-cache"
            module_cache.mkdir(exist_ok=True)
            run(["swiftc", "-O", "-Xcc", f"-fmodules-cache-path={module_cache}", str(PROJECT_ROOT / "scripts/apple_vision_ocr.swift"), "-o", str(binary)], "编译 Apple Vision OCR")
            run([str(binary), "--frames", str(frames), "--output", str(raw), "--fps", "6", "--timestamps", str(frames / "timestamps.json")], "Apple Vision 精细 OCR")
            asr_payload = json.loads(asr_path.read_text(encoding="utf-8"))
            duration = max((float(word.get("end", 0.0)) for word in asr_payload.get("words", [])), default=0.0)
            run([sys.executable, str(PROJECT_ROOT / "scripts/evaluate_burned_subtitle_ocr.py"), "--raw", str(raw), "--asr", str(asr_path), "--output-dir", str(merged), "--fps", "6", "--duration", f"{duration:.3f}"], "聚合 OCR 文字证据")
            run([sys.executable, str(PROJECT_ROOT / "scripts/adjudicate_nuc_gemini_with_ocr.py"), "--gemini", str(gemini_path), "--differences", str(differences), "--ocr", str(ocr_json), "--output-transcript", str(final_transcript), "--output-report", str(adjudications)], "OCR 仅裁决文字分歧")
            mark_stage(manifest_path, manifest, "ocr_adjudication", "completed", plan=str(plan), raw=str(raw), ocr=str(ocr_json), adjudications=str(adjudications), transcript=str(final_transcript))
    else:
        if not final_transcript.exists():
            shutil.copyfile(gemini_path, final_transcript)
        mark_stage(manifest_path, manifest, "ocr_adjudication", "skipped", reason="burned_subtitles_not_detected", transcript=str(final_transcript))

    final_dir = output_dir / "06-final"
    final_srt = final_dir / "final.srt"
    final_json = final_dir / "final-timeline.json"
    if not stage_ready(manifest, "finalize", [final_srt, final_json]):
        run([sys.executable, str(PROJECT_ROOT / "scripts/forensic_subtitle_command.py"), "finalize", "--nuc-asr", str(asr_path), "--gemini", str(final_transcript), "--output-dir", str(final_dir)], "阶段 5-6/6：算法断句并映射 NUC words 时间")
        quality = validate_final(final_json)
        mark_stage(manifest_path, manifest, "finalize", "completed", srt=str(final_srt), timeline=str(final_json), quality=quality)

    manifest["status"] = "completed"
    manifest["recommended_output"] = str(final_srt)
    write_manifest(manifest_path, manifest)
    print(f"\n完成。最终字幕：{final_srt}", flush=True)
    return 0


def pipeline(args: argparse.Namespace) -> int:
    with yt_dlp_cookie_session(
        enabled=args.cookies_from_chrome,
        chrome_profile=args.chrome_profile,
    ) as cookie_session:
        if cookie_session.source_database:
            print(
                "Chrome Cookie: 已隔离读取主数据库 "
                f"{cookie_session.source_database} "
                f"（总计 {cookie_session.total_cookies}，YouTube {cookie_session.youtube_cookies}）",
                flush=True,
            )
        return run_pipeline(args, cookie_session)


def mark_job_exit(args: argparse.Namespace, status_value: str, error: str) -> None:
    manifest_path = getattr(args, "_manifest_path", None)
    manifest = getattr(args, "_manifest", None)
    if not isinstance(manifest_path, Path) or not isinstance(manifest, dict):
        return
    manifest["status"] = status_value
    manifest["error"] = error
    write_manifest(manifest_path, manifest)


def latest_manifests(limit: int = 8) -> list[Path]:
    if not GENERATED_DIR.exists():
        return []
    manifests = [
        *GENERATED_DIR.glob("*/pipeline-manifest.json"),
        *GENERATED_DIR.glob("*/asr-manifest.json"),
    ]
    return sorted(
        manifests,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


def status() -> int:
    manifests = latest_manifests()
    if not manifests:
        print("暂无取证字幕作业。")
        return 0
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{data.get('status', 'running'):10}  {data.get('title') or data.get('source')}")
        print(f"  {path.parent}")
        if data.get("recommended_output"):
            print(f"  final: {data['recommended_output']}")
    return 0


def doctor() -> int:
    failures = 0
    for label, path in (("yt-dlp", YT_DLP), ("ffmpeg", FFMPEG), ("ffprobe", FFPROBE), ("swiftc", "/usr/bin/swiftc")):
        ok = Path(path).exists()
        print(f"{'OK' if ok else 'MISSING':7} {label}: {path}")
        failures += 0 if ok else 1
    for module in ("google.genai", "PySide6"):
        completed = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True)
        ok = completed.returncode == 0
        print(f"{'OK' if ok else 'MISSING':7} Python module: {module}")
        failures += 0 if ok else 1
    for label, url in (
        ("NUC faster-whisper", f"http://{NUC_OLLAMA_HOST}:8000/busy"),
        ("NUC Qwen3-ASR", f"http://{NUC_OLLAMA_HOST}:8001/healthz"),
    ):
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                ok = 200 <= response.status < 300
        except Exception:
            ok = False
        print(f"{'OK' if ok else 'WARN':7} {label}: {url}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("source")
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--ocr", choices=("auto", "on", "off"), default="auto")
    run_parser.add_argument("--cookies-from-chrome", action="store_true")
    run_parser.add_argument(
        "--chrome-profile",
        default=os.environ.get("FORENSIC_CHROME_PROFILE", "Default"),
    )
    run_parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    run_parser.add_argument("--nuc-url", default="")
    run_parser.set_defaults(handler=pipeline)
    sub.add_parser("status").set_defaults(handler=lambda _args: status())
    sub.add_parser("doctor").set_defaults(handler=lambda _args: doctor())
    args = parser.parse_args()
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        mark_job_exit(args, "interrupted", "Interrupted by user")
        print("\n已中断；再次运行同一 URL 可从已完成阶段续跑。", file=sys.stderr)
        return 130
    except Exception as exc:
        mark_job_exit(args, "failed", str(exc))
        print(f"\n失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
