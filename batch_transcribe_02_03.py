#!/usr/bin/env python3
"""
两阶段批量转录脚本：
  阶段一：对所有 MP4 依次 ASR → 原始 JSON 暂存到 ~/Documents/temp/batch_asr_staging/
  阶段二：所有 ASR 完成后，统一 LLM 规整 → SRT 写回原视频同目录

断点续跑：
  - 已有 .srt 的视频直接跳过全流程
  - 已有暂存 JSON 的视频跳过 ASR，直接进入阶段二队列
  - 阶段二从 staging 目录的 manifest.json 恢复对应关系
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# ─── 配置 ───────────────────────────────────────────────────────────────────

BASE_DIR = Path("/Volumes/02_HDD_unTar_NFS/华清元宇宙/yyzlab_05_第五阶段：进阶实战-大模型实战应用")
TARGET_DIRS = [
    BASE_DIR / "02 大模型的部署与应用基础",
    BASE_DIR / "03 大模型的RAG与MCP Agent设计",
]

# 暂存目录（~/Documents/temp/batch_asr_staging/）
STAGING_DIR = Path.home() / "Documents" / "temp" / "batch_asr_staging"
MANIFEST_PATH = STAGING_DIR / "manifest.json"  # {slug: {"mp4": str, "json": str}}

NUC_HOST = "192.168.31.196"
NUC_QWEN_PORT = "8001"
NUC_QWEN_BASE_URL = f"http://{NUC_HOST}:{NUC_QWEN_PORT}"

GEMINI_API_KEY = "AIzaSyADl6hpoxdZUVdEqvLylzEwV7lvdr93Jdk"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = "gemini-2.5-flash"

FFMPEG = "/opt/homebrew/bin/ffmpeg"
LLM_BATCH_SIZE = 80  # 每次 LLM 调用最多处理的段数

# ─── SRT / 工具函数 ──────────────────────────────────────────────────────────

def format_srt_timestamp(seconds: float) -> str:
    millis_total = max(0, int(round(seconds * 1000)))
    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    secs = total_seconds % 60
    minutes_total = total_seconds // 60
    minutes = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_srt(path: Path, segments: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, seg in enumerate(segments, 1):
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(seg['start'])} --> {format_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def mp4_slug(mp4_path: Path) -> str:
    """将 MP4 路径转为文件系统安全的唯一标识符。"""
    rel = str(mp4_path).replace("/", "_").replace(" ", "_")
    return re.sub(r"[^\w\-.]", "_", rel)[-180:]


# ─── Manifest 管理 ────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


# ─── 音频提取 ─────────────────────────────────────────────────────────────────

def extract_audio_wav(mp4_path: Path, wav_path: Path) -> None:
    cmd = [
        FFMPEG, "-y",
        "-i", str(mp4_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败:\n{result.stderr[-2000:]}")


# ─── NUC Qwen3-ASR 1.7B ─────────────────────────────────────────────────────

def _multipart_body(wav_path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    audio_data = wav_path.read_bytes()
    filename = wav_path.name
    parts = []
    for field_name, field_value in [
        ("model", "qwen3-asr-1p7b"),
        ("language", "zh"),
        ("response_format", "verbose_json"),
    ]:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
            f"{field_value}\r\n"
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    )
    body = b""
    for p in parts:
        body += p.encode("utf-8")
    body += audio_data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, boundary


def _parse_nuc_response(data: dict) -> list[dict]:
    segments = []
    raw_segs = data.get("segments", [])
    if raw_segs:
        for seg in raw_segs:
            text = seg.get("text", "").strip()
            if text:
                segments.append({
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", 0)),
                    "text": text,
                })
    else:
        full_text = data.get("text", "").strip()
        duration = float(data.get("duration", 60.0))
        if full_text:
            segments = _pseudo_timestamp(full_text, duration)
    return segments


def qwen3_event_label(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[\(\[（【][^)\]）】]{1,24}[\)\]）】]", stripped))


def _pseudo_timestamp(text: str, duration: float) -> list[dict]:
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
        return [{"start": 0.0, "end": duration, "text": clean}]
        
    char_weights = [
        max(1, len(re.sub(r"[，。！？!?；;、,（）()【】\[\]\s]", "", part)))
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
                    max(1, len(re.sub(r"[，。！？!?；;、,（）()【】\[\]\s]", "", sub_part)))
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
    segments: list[dict] = []
    for index, part in enumerate(parts):
        segment_duration = effective_duration * (weights[index] / total_weight)
        if qwen3_event_label(part):
            segment_duration = min(segment_duration, 1.2)
        end = effective_duration if index == len(parts) - 1 else min(effective_duration, cursor + segment_duration)
        segments.append({"start": cursor, "end": end, "text": part})
        cursor = end
        
    if tail_padding > 0 and segments:
        last = segments[-1]
        segments[-1] = {"start": last["start"], "end": duration, "text": last["text"]}
        
    return merge_short_qwen3_segments(segments)


def merge_short_qwen3_segments(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    merged: list[dict] = []
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if (
            merged
            and duration < 1.5
            and not qwen3_event_label(segment["text"])
            and not qwen3_event_label(merged[-1]["text"])
        ):
            previous = merged[-1]
            joiner = "" if previous["text"].endswith(("，", "。", "！", "？", ",", ".", "!", "?")) else " "
            merged[-1] = {
                "start": previous["start"],
                "end": segment["end"],
                "text": f"{previous['text']}{joiner}{segment['text']}".strip()
            }
            continue
        merged.append(segment)
    return merged


def _upload_and_poll(wav_path: Path, base_url: str, timeout: int = 1800) -> list[dict]:
    body, boundary = _multipart_body(wav_path)
    upload_url = f"{base_url}/jobs/upload"
    req = urllib.request.Request(
        upload_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    print(f"    → Job上传 ({len(body)/1024/1024:.1f} MB)...", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp:
        upload_resp = json.loads(resp.read().decode("utf-8"))
    task_id = upload_resp.get("task_id") or upload_resp.get("id")
    if not task_id:
        raise RuntimeError(f"上传响应中无 task_id: {upload_resp}")
    print(f"    → task_id={task_id}，轮询中...", flush=True)
    poll_url = f"{base_url}/jobs/{task_id}"
    deadline = time.monotonic() + timeout
    interval = 5.0
    while time.monotonic() < deadline:
        time.sleep(interval)
        with urllib.request.urlopen(poll_url, timeout=30) as r:
            status_data = json.loads(r.read().decode("utf-8"))
        state = status_data.get("status", "")
        print(f"    → 状态: {state}    ", end="\r", flush=True)
        if state in ("completed", "done", "finished", "success"):
            print(flush=True)
            break
        if state in ("failed", "error"):
            raise RuntimeError(f"NUC 任务失败: {status_data}")
        interval = min(interval * 1.2, 30.0)
    else:
        raise TimeoutError(f"NUC 任务超时（{timeout}s）")
    result = status_data.get("result") or status_data
    return _parse_nuc_response(result)


def _transcribe_direct(wav_path: Path, base_url: str, timeout: int = 900) -> list[dict]:
    body, boundary = _multipart_body(wav_path)
    url = f"{base_url}/v1/audio/transcriptions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    print(f"    → 直接转录 ({len(body)/1024/1024:.1f} MB)...", flush=True)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _parse_nuc_response(data)


def transcribe_wav(wav_path: Path) -> list[dict]:
    """使用与原项目一致的 30s 分块 + 2s 重叠策略，确保 Qwen3-ASR 的断句和时间戳准确。"""
    cmd_dur = [FFMPEG, "-i", str(wav_path)]
    result = subprocess.run(cmd_dur, capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    else:
        duration = 0.0

    chunk_seconds = 30.0
    overlap_seconds = 2.0
    all_segments: list[dict] = []
    offset = 0.0
    chunk_index = 0
    
    chunks_dir = wav_path.with_suffix(".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"    → 音频总时长: {duration:.1f}s，开始 30s 分块转录(2s重叠)...", flush=True)

    while offset < duration:
        actual_start = max(0.0, offset - (overlap_seconds if chunk_index > 0 else 0.0))
        leading_trim = offset - actual_start
        remaining = min(chunk_seconds + leading_trim + overlap_seconds, duration - actual_start)
        
        if remaining <= 0.1:
            break

        chunk_wav = chunks_dir / f"chunk_{chunk_index:04d}.wav"
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(actual_start),
            "-t", str(remaining),
            "-i", str(wav_path),
            "-ac", "1",
            "-ar", "16000",
            str(chunk_wav),
        ]
        subprocess.run(cmd, check=True)
        
        try:
            raw_segs = _transcribe_direct(chunk_wav, NUC_QWEN_BASE_URL)
        except Exception as e:
            print(f"    ⚠ 切片 {chunk_index} 转录失败，跳过: {e}", flush=True)
            raw_segs = []

        # Trim overlaps
        trailing_trim = overlap_seconds if actual_start + remaining < duration else 0.0
        for seg in raw_segs:
            # 这里的 seg["start"] 是基于当前 chunk 的相对时间
            if seg["start"] < leading_trim and chunk_index > 0:
                continue
            if seg["end"] > remaining - trailing_trim and actual_start + remaining < duration:
                continue
            
            # 调整时间到完整音频
            adjusted_start = actual_start + seg["start"]
            adjusted_end = actual_start + seg["end"]
            all_segments.append({"start": adjusted_start, "end": adjusted_end, "text": seg["text"]})
            
        offset += chunk_seconds
        chunk_index += 1
        if chunk_index % 10 == 0:
            print(f"    → 已处理 {offset:.1f}s / {duration:.1f}s", flush=True)
            
    return all_segments


# ─── LLM 规整 ────────────────────────────────────────────────────────────────

LLM_SYSTEM_PROMPT = (
    "你是一个专业的中文字幕校对专家。你的任务是对 ASR 识别的字幕文本进行错别字修正、"
    "同音字修正、去除重复词，以及修正技术术语（如大模型、RAG、MCP、Agent等）。\n"
    "保持原意。输出格式严格为：\"序号: 规整后的文本\"，每行一个，不要添加解释。\n"
    "你不能合并或者删除任何行，必须对每一个序号进行逐行返回修正后的结果。\n"
    "技术词汇如 LangChain、RAG、MCP、FastAPI、LLM、API、Transformer、vLLM、Ollama 等保持英文原样。"
)

_LLM_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")


def _parse_llm_lines(reply: str, expected_count: int) -> dict[int, str]:
    corrected: dict[int, str] = {}
    fallback_lines: list[str] = []
    for line in reply.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LLM_LINE_RE.match(line)
        if m:
            corrected[int(m.group(1)) - 1] = m.group(2).strip()
        elif not re.match(r"^(Whisper|Native|Final|Output)\s*[:=]", line, re.I):
            fallback_lines.append(line)
    if not corrected and len(fallback_lines) == expected_count:
        corrected = {i: text for i, text in enumerate(fallback_lines)}
    return corrected


def llm_polish(segments: list[dict]) -> list[dict]:
    """对 segments 分批调用 Gemini 2.5 Flash，返回规整后的 segments。"""
    if not segments:
        return segments
    result = list(segments)
    for batch_start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[batch_start: batch_start + LLM_BATCH_SIZE]
        lines = [f"{i + 1}: {seg['text']}" for i, seg in enumerate(batch)]
        user_text = "\n".join(lines)
        body = {
            "model": GEMINI_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
            "max_tokens": 8192,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
        }
        reply = ""
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    GEMINI_API_URL,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"]["content"]
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace").strip()
                print(f"    ⚠ Gemini HTTP {exc.code}（第{attempt+1}次）: {detail[:200]}", flush=True)
                if attempt == 2:
                    print("    ⚠ LLM 失败，保留原始文本", flush=True)
                time.sleep(8 * (attempt + 1))
            except Exception as exc:
                print(f"    ⚠ Gemini 异常（第{attempt+1}次）: {exc}", flush=True)
                if attempt == 2:
                    print("    ⚠ LLM 失败，保留原始文本", flush=True)
                time.sleep(8 * (attempt + 1))
        if reply:
            corrected = _parse_llm_lines(reply, len(batch))
            for local_i, seg in enumerate(batch):
                if local_i in corrected:
                    result[batch_start + local_i] = {**seg, "text": corrected[local_i]}
        end_idx = batch_start + len(batch)
        print(f"    → LLM [{batch_start+1}~{end_idx}/{len(segments)}] ✓", flush=True)
    return result


# ─── 阶段一：全量 ASR ─────────────────────────────────────────────────────────

def phase1_asr(mp4_files: list[Path]) -> None:
    """对所有 MP4 做 ASR，原始结果 JSON 存入 STAGING_DIR，更新 manifest。"""
    print("\n" + "=" * 60)
    print("【阶段一】ASR 转录（全量）")
    print(f"暂存目录: {STAGING_DIR}")
    print("=" * 60)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    total = len(mp4_files)
    asr_done = 0
    asr_skip = 0
    asr_fail = []

    for i, mp4_path in enumerate(mp4_files, 1):
        slug = mp4_slug(mp4_path)
        srt_path = mp4_path.with_suffix(".srt")
        json_path = STAGING_DIR / f"{slug}.json"

        # 最终 SRT 已存在 → 全跳过
        if srt_path.exists():
            print(f"\n[{i}/{total}] ⏩ SRT已存在，跳过: {mp4_path.name}", flush=True)
            asr_skip += 1
            # 确保 manifest 中也记录（以便阶段二幂等）
            if slug not in manifest:
                manifest[slug] = {"mp4": str(mp4_path), "json": str(json_path), "srt_done": True}
                save_manifest(manifest)
            continue

        # 暂存 JSON 已存在 → ASR 已完成，跳过
        if json_path.exists():
            print(f"\n[{i}/{total}] ⏩ ASR已完成，跳过: {mp4_path.name}", flush=True)
            asr_skip += 1
            if slug not in manifest:
                manifest[slug] = {"mp4": str(mp4_path), "json": str(json_path), "srt_done": False}
                save_manifest(manifest)
            continue

        print(f"\n[{i}/{total}] 🎙 ASR: {mp4_path.name}", flush=True)

        with tempfile.TemporaryDirectory(prefix="asr_wav_") as tmpdir:
            wav_path = Path(tmpdir) / "audio-16k-mono.wav"

            # 提取音频
            print(f"  → 提取音频...", flush=True)
            try:
                extract_audio_wav(mp4_path, wav_path)
            except Exception as e:
                print(f"  ✗ 音频提取失败: {e}", flush=True)
                asr_fail.append(str(mp4_path))
                continue

            wav_mb = wav_path.stat().st_size / 1024 / 1024
            print(f"  → WAV: {wav_mb:.1f} MB", flush=True)

            # NUC ASR
            try:
                segments = transcribe_wav(wav_path)
            except Exception as e:
                print(f"  ✗ ASR 失败: {e}", flush=True)
                asr_fail.append(str(mp4_path))
                continue

        if not segments:
            print(f"  ✗ ASR 返回空结果", flush=True)
            asr_fail.append(str(mp4_path))
            continue

        print(f"  → ASR 完成，{len(segments)} 段", flush=True)

        # 保存原始 JSON 到暂存目录
        json_path.write_text(
            json.dumps({"mp4": str(mp4_path), "segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 更新 manifest
        manifest[slug] = {"mp4": str(mp4_path), "json": str(json_path), "srt_done": False}
        save_manifest(manifest)

        asr_done += 1

    print(f"\n{'='*60}")
    print(f"【阶段一完成】成功={asr_done}  跳过={asr_skip}  失败={len(asr_fail)}")
    if asr_fail:
        print("失败列表：")
        for f in asr_fail:
            print(f"  - {f}")
    print("=" * 60, flush=True)


# ─── 阶段二：统一 LLM 规整 ───────────────────────────────────────────────────

def phase2_llm() -> None:
    """读取 manifest，对所有未规整的条目做 LLM 规整，写 SRT 回原路径。"""
    print("\n" + "=" * 60)
    print("【阶段二】LLM 规整（统一批量）")
    print("=" * 60)

    manifest = load_manifest()
    if not manifest:
        print("manifest 为空，没有需要规整的条目。", flush=True)
        return

    pending_all = [(slug, info) for slug, info in manifest.items() if not info.get("srt_done")]
    already_done = sum(1 for info in manifest.values() if info.get("srt_done"))

    # 按原 MP4 路径排序，确保阶段二也按章节顺序输出 SRT（用户可边生成边观看）
    pending = sorted(pending_all, key=lambda x: x[1].get("mp4", ""))

    print(f"待规整: {len(pending)}  已完成: {already_done}  （按章节顺序处理）", flush=True)

    llm_done = 0
    llm_fail = []

    for idx, (slug, info) in enumerate(pending, 1):
        mp4_path = Path(info["mp4"])
        json_path = Path(info["json"])
        srt_path = mp4_path.with_suffix(".srt")

        print(f"\n[{idx}/{len(pending)}] ✏ LLM: {mp4_path.name}", flush=True)

        # SRT 已存在（中途重跑保护）
        if srt_path.exists():
            print(f"  ⏩ SRT已存在，标记完成", flush=True)
            manifest[slug]["srt_done"] = True
            save_manifest(manifest)
            llm_done += 1
            continue

        # 加载暂存 JSON
        if not json_path.exists():
            print(f"  ✗ 暂存 JSON 不存在: {json_path}", flush=True)
            llm_fail.append(str(mp4_path))
            continue
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            segments: list[dict] = cached["segments"]
        except Exception as e:
            print(f"  ✗ 读取暂存 JSON 失败: {e}", flush=True)
            llm_fail.append(str(mp4_path))
            continue

        if not segments:
            print(f"  ✗ 暂存 JSON 中 segments 为空", flush=True)
            llm_fail.append(str(mp4_path))
            continue

        print(f"  → {len(segments)} 段，开始 LLM 规整...", flush=True)
        try:
            polished = llm_polish(segments)
        except Exception as e:
            print(f"  ⚠ LLM 异常（保留原始）: {e}", flush=True)
            polished = segments

        # 写 SRT
        try:
            save_srt(srt_path, polished)
            print(f"  ✓ SRT 已保存: {srt_path.name}", flush=True)
        except Exception as e:
            print(f"  ✗ SRT 写入失败: {e}", flush=True)
            llm_fail.append(str(mp4_path))
            continue

        # 更新 manifest
        manifest[slug]["srt_done"] = True
        save_manifest(manifest)
        llm_done += 1

    print(f"\n{'='*60}")
    print(f"【阶段二完成】成功={llm_done}  失败={len(llm_fail)}")
    if llm_fail:
        print("失败列表：")
        for f in llm_fail:
            print(f"  - {f}")
    print("=" * 60, flush=True)


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def find_mp4_files(dirs: list[Path]) -> list[Path]:
    """按章节顺序（目录名+文件名字母序）收集所有 MP4，确保 02→03 章节顺序。"""
    files = []
    for d in dirs:
        # rglob 后按完整路径排序，保证子目录章节顺序
        files.extend(sorted(d.rglob("*.mp4"), key=lambda p: str(p)))
    return files


def check_nuc() -> bool:
    for endpoint in ("/healthz", "/busy"):
        try:
            with urllib.request.urlopen(f"{NUC_QWEN_BASE_URL}{endpoint}", timeout=5) as r:
                return r.status == 200
        except Exception:
            continue
    return False


def main() -> None:
    print("=" * 60)
    print("两阶段批量 MP4 → SRT 转录脚本")
    print(f"  阶段一 ASR  : NUC Qwen3-ASR 1.7B ({NUC_QWEN_BASE_URL})")
    print(f"  阶段二 LLM  : Gemini 2.5 Flash")
    print(f"  暂存目录    : {STAGING_DIR}")
    print("=" * 60, flush=True)

    # NUC 健康检查
    print("\n检查 NUC ASR 服务...", flush=True)
    if check_nuc():
        print("  ✓ NUC ASR 服务正常", flush=True)
    else:
        print(f"  ⚠ NUC 不可达，ASR 阶段每个文件失败时会有错误提示", flush=True)

    # 扫描 MP4
    print("\n扫描 MP4 文件...", flush=True)
    mp4_files = find_mp4_files(TARGET_DIRS)
    print(f"  找到 {len(mp4_files)} 个 MP4", flush=True)

    start = time.time()

    # ── 阶段一 ──
    phase1_asr(mp4_files)

    elapsed_total = time.time() - start
    print(f"\n全部 ASR 阶段一已处理完毕！总耗时: {elapsed_total/60:.1f} 分钟", flush=True)
    print(f"暂存 JSON 位置: {STAGING_DIR}", flush=True)


if __name__ == "__main__":
    main()
