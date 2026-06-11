from __future__ import annotations

import asyncio
import array
from collections import Counter
import io
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any
import wave

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel


UPSTREAM_URL = os.environ.get(
    "UPSTREAM_URL",
    "http://nuc-qwen3-asr-1p7b-vllm:8000/v1/audio/transcriptions",
)
SCHEDULER_URL = os.environ.get(
    "SCHEDULER_URL",
    "http://nuc-service-scheduler:8010",
)
UPSTREAM_MODEL = "Qwen/Qwen3-ASR-1.7B"
QUEUE_CONCURRENCY = 1
IDLE_STOP_SECONDS = int(os.environ.get("IDLE_STOP_SECONDS", "180"))
RESULT_DIR = Path(os.environ.get("QWEN_ASR_RESULT_DIR", "/app/qwen-asr-results"))
STAGING_DIR = Path(os.environ.get("QWEN_ASR_STAGING_DIR", "/app/qwen-asr-staging"))
ADMISSION_RETRY_SECONDS = float(os.environ.get("QWEN_ADMISSION_RETRY_SECONDS", "5"))
ADMISSION_MAX_WAIT_SECONDS = float(os.environ.get("QWEN_ADMISSION_MAX_WAIT_SECONDS", "1800"))
MAX_DIRECT_UPLOAD_MB = float(os.environ.get("QWEN_MAX_DIRECT_UPLOAD_MB", "64"))
CHUNK_SECONDS = float(os.environ.get("QWEN_CHUNK_SECONDS", "30"))
CHUNK_OVERLAP_SECONDS = float(os.environ.get("QWEN_CHUNK_OVERLAP_SECONDS", "2"))
EMPTY_RETRY_MIN_DBFS = float(os.environ.get("QWEN_EMPTY_RETRY_MIN_DBFS", "-50"))
TEXT_OVERLAP_MAX_CHARS = int(os.environ.get("QWEN_TEXT_OVERLAP_MAX_CHARS", "120"))

app = FastAPI(title="NUC Qwen3-ASR 1.7B Proxy")
semaphore = asyncio.Semaphore(QUEUE_CONCURRENCY)
idle_stop_task: asyncio.Task | None = None
idle_stop_lock = asyncio.Lock()
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
ACTIVE_REQUESTS = 0
CURRENT_REQUEST: dict[str, Any] | None = None


async def _scheduler_post(path: str) -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(f"{SCHEDULER_URL}{path}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"scheduler unavailable: {exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)


async def _scheduler_get(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(f"{SCHEDULER_URL}{path}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"scheduler unavailable: {exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


def _safe_name(value: str, fallback: str = "audio") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", value).strip("._")
    return cleaned[:120] or fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_task_dir(filename: str) -> tuple[str, Path]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    task_id = f"{stamp}-{uuid.uuid4().hex[:8]}-{_safe_name(filename)}"
    return task_id, STAGING_DIR / task_id


def _task_set(task_id: str, **updates: Any) -> dict[str, Any]:
    with TASKS_LOCK:
        task = TASKS.setdefault(task_id, {})
        task.update(updates)
        return dict(task)


def _task_snapshot(task_id: str) -> dict[str, Any]:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return dict(task)


async def _track_request(metadata: dict[str, Any]):
    global ACTIVE_REQUESTS
    global CURRENT_REQUEST
    ACTIVE_REQUESTS += 1
    CURRENT_REQUEST = {**metadata, "started_at": time.time()}
    try:
        yield
    finally:
        ACTIVE_REQUESTS = max(0, ACTIVE_REQUESTS - 1)
        if ACTIVE_REQUESTS == 0:
            CURRENT_REQUEST = None


async def _cancel_idle_stop() -> None:
    global idle_stop_task
    async with idle_stop_lock:
        if idle_stop_task and not idle_stop_task.done():
            idle_stop_task.cancel()
        idle_stop_task = None


async def _schedule_idle_stop() -> None:
    global idle_stop_task

    async def _worker() -> None:
        try:
            await asyncio.sleep(IDLE_STOP_SECONDS)
            await _scheduler_post("/stop/qwen")
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async with idle_stop_lock:
        if idle_stop_task and not idle_stop_task.done():
            idle_stop_task.cancel()
        idle_stop_task = asyncio.create_task(_worker())


def _current_request_status() -> dict[str, Any] | None:
    if not CURRENT_REQUEST:
        return None
    started_at = float(CURRENT_REQUEST.get("started_at", 0.0))
    return {
        **CURRENT_REQUEST,
        "elapsed_seconds": max(0.0, time.time() - started_at),
    }


async def _wait_for_qwen_admission() -> None:
    deadline = time.monotonic() + ADMISSION_MAX_WAIT_SECONDS
    last_error: str | None = None
    while True:
        try:
            await _scheduler_post("/admit/qwen")
            await _scheduler_post("/ensure/qwen")
            return
        except HTTPException as exc:
            last_error = str(exc.detail)
            if exc.status_code not in {429, 503}:
                raise
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=f"Timed out waiting for Qwen admission: {last_error}",
                ) from exc
            await asyncio.sleep(ADMISSION_RETRY_SECONDS)


def _pseudo_segments(text: str) -> list[dict[str, Any]]:
    clean = " ".join(text.split()).strip()
    if not clean:
        return []
    return [{"start": 0.0, "end": 30.0, "text": clean}]


def _clean_transcript(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return ""
    clean = re.sub(r"^\s*language\s+[A-Za-z]+\s*<asr_text>\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\s*<asr_text>\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def _repetition_hallucination_reason(text: str) -> str | None:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) < 160:
        return None
    parts = [
        part.strip()
        for part in re.split(r"[，,。！？!?；;、\s]+", clean)
        if part.strip()
    ]
    if len(parts) < 12:
        return None
    counts = Counter(parts)
    phrase, count = counts.most_common(1)[0]
    ratio = count / len(parts)
    if len(phrase) <= 24 and count >= 8 and ratio >= 0.6:
        return (
            f"dominant repeated short phrase filtered: phrase={phrase!r} "
            f"count={count} total_parts={len(parts)} ratio={ratio:.3f}"
        )
    return None


def _sanitize_chunk_text(text: str) -> tuple[str, str | None]:
    clean = _clean_transcript(text)
    reason = _repetition_hallucination_reason(clean)
    if reason:
        return "", reason
    return clean, None


def _bounded_edit_distance(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_char in enumerate(right, 1):
            value = min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _text_overlap_length(left: str, right: str) -> tuple[int, int]:
    maximum = min(TEXT_OVERLAP_MAX_CHARS, len(left), len(right))
    best: tuple[int, int, int] | None = None
    for right_length in range(maximum, 3, -1):
        allowed_errors = 0
        if right_length >= 8:
            allowed_errors = 1
        if right_length >= 16:
            allowed_errors = 2
        minimum_left = max(4, right_length - allowed_errors)
        maximum_left = min(len(left), right_length + allowed_errors)
        for left_length in range(maximum_left, minimum_left - 1, -1):
            distance = _bounded_edit_distance(
                left[-left_length:],
                right[:right_length],
                allowed_errors,
            )
            if distance <= allowed_errors:
                score = min(left_length, right_length) - distance * 3
                candidate = (score, right_length, -distance)
                if best is None or candidate > best:
                    best = candidate
    if best is not None:
        _score, right_length, negative_distance = best
        return right_length, -negative_distance
    return 0, 0


def _merge_overlapping_text(left: str, right: str) -> tuple[str, int, int]:
    left = left.strip()
    right = right.strip()
    if not left:
        return right, 0, 0
    if not right:
        return left, 0, 0
    overlap, errors = _text_overlap_length(left, right)
    if not overlap:
        return f"{left}\n{right}", 0, 0
    return f"{left}{right[overlap:]}", overlap, errors


def _pseudo_segments_for_text(text: str, duration: float, offset: float) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。？！!?；;……])\s*", clean) if part.strip()]
    if len(parts) <= 1:
        return [{"start": offset, "end": offset + max(duration, 0.1), "text": clean}]
    weights = [
        max(1, len(re.sub(r"[，。！？!?；;、,（）()【】\[\]\s]", "", part)))
        for part in parts
    ]
    total_weight = sum(weights) or len(parts)
    segments: list[dict[str, Any]] = []
    cursor = offset
    for index, part in enumerate(parts):
        seg_duration = duration * (weights[index] / total_weight)
        end = offset + duration if index == len(parts) - 1 else cursor + seg_duration
        segments.append({"start": cursor, "end": end, "text": part})
        cursor = end
    return segments


def _wav_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as reader:
        frame_rate = reader.getframerate()
        if frame_rate <= 0:
            return 0.0
        return reader.getnframes() / frame_rate


def _wav_bytes_dbfs(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        sample_width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    if not frames:
        return float("-inf")
    if sample_width == 1:
        samples = array.array("B", frames)
        squared = sum((sample - 128) ** 2 for sample in samples)
        peak = 127
    elif sample_width == 2:
        samples = array.array("h")
        samples.frombytes(frames)
        if os.sys.byteorder != "little":
            samples.byteswap()
        squared = sum(sample * sample for sample in samples)
        peak = 32767
    else:
        return 0.0
    rms = math.sqrt(squared / max(1, len(samples)))
    if rms <= 0:
        return float("-inf")
    return 20.0 * math.log10(rms / peak)


def _wav_slice_bytes(audio_path: Path, start_seconds: float, duration_seconds: float) -> bytes:
    with wave.open(str(audio_path), "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        if frame_rate <= 0:
            raise RuntimeError(f"Invalid WAV frame rate: {audio_path}")
        start_frame = min(reader.getnframes(), max(0, int(start_seconds * frame_rate)))
        frame_count = min(
            max(1, int(duration_seconds * frame_rate)),
            reader.getnframes() - start_frame,
        )
        reader.setpos(start_frame)
        frames = reader.readframes(frame_count)
    chunk_buffer = io.BytesIO()
    with wave.open(chunk_buffer, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return chunk_buffer.getvalue()


def _iter_wav_chunks(audio_path: Path, chunk_seconds: float, overlap_seconds: float = 0.0):
    with wave.open(str(audio_path), "rb") as reader:
        frame_rate = reader.getframerate()
        if frame_rate <= 0:
            raise RuntimeError(f"Invalid WAV frame rate: {audio_path}")
        total_frames = reader.getnframes()
    total_duration = total_frames / frame_rate
    chunk_index = 0
    nominal_start = 0.0
    while nominal_start < total_duration:
        nominal_duration = min(chunk_seconds, total_duration - nominal_start)
        request_start = max(0.0, nominal_start - overlap_seconds)
        request_end = min(
            total_duration,
            nominal_start + nominal_duration + overlap_seconds,
        )
        request_duration = request_end - request_start
        yield {
            "chunk_index": chunk_index,
            "offset_seconds": nominal_start,
            "duration_seconds": nominal_duration,
            "request_offset_seconds": request_start,
            "request_duration_seconds": request_duration,
            "leading_context_seconds": nominal_start - request_start,
            "trailing_context_seconds": request_end - (nominal_start + nominal_duration),
            "audio_bytes": _wav_slice_bytes(audio_path, request_start, request_duration),
        }
        nominal_start += nominal_duration
        chunk_index += 1


def _split_wav_bytes(wav_bytes: bytes, overlap_seconds: float = 1.0) -> list[tuple[float, float, bytes]]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()
        frames = reader.readframes(total_frames)
    if frame_rate <= 0 or total_frames <= 1:
        return []
    midpoint = total_frames // 2
    overlap_frames = max(0, int(overlap_seconds * frame_rate))
    windows = [
        (0, min(total_frames, midpoint + overlap_frames)),
        (max(0, midpoint - overlap_frames), total_frames),
    ]
    parts: list[tuple[float, float, bytes]] = []
    for start_frame, end_frame in windows:
        frame_bytes = frames[
            start_frame * params.sampwidth * params.nchannels:
            end_frame * params.sampwidth * params.nchannels
        ]
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frame_bytes)
        parts.append((
            start_frame / frame_rate,
            (end_frame - start_frame) / frame_rate,
            buffer.getvalue(),
        ))
    return parts


async def _post_upstream_bytes(
    *,
    audio_bytes: bytes,
    filename: str,
    language: str,
) -> dict[str, Any]:
    files = {
        "file": (filename, audio_bytes, "audio/wav"),
    }
    data_payload = {
        "model": UPSTREAM_MODEL,
        "language": language,
    }
    async with httpx.AsyncClient(timeout=900) as client:
        response = await client.post(UPSTREAM_URL, data=data_payload, files=files)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


async def _transcribe_chunk_with_empty_retry(
    *,
    audio_bytes: bytes,
    filename: str,
    language: str,
) -> tuple[str, str | None, dict[str, Any]]:
    data = await _post_upstream_bytes(
        audio_bytes=audio_bytes,
        filename=filename,
        language=language,
    )
    raw_text = _clean_transcript(data.get("text", ""))
    text, filtered_reason = _sanitize_chunk_text(raw_text)
    dbfs = _wav_bytes_dbfs(audio_bytes)
    retry = {
        "attempted": False,
        "reason": None,
        "part_count": 0,
        "part_text_lengths": [],
    }
    if text or filtered_reason or dbfs < EMPTY_RETRY_MIN_DBFS:
        return text, filtered_reason, {
            "raw_text": raw_text,
            "audio_dbfs": dbfs,
            "empty_retry": retry,
        }

    retry["attempted"] = True
    retry["reason"] = "empty_non_silent_chunk"
    merged = ""
    for part_index, (_offset, _duration, part_bytes) in enumerate(
        _split_wav_bytes(audio_bytes)
    ):
        part_data = await _post_upstream_bytes(
            audio_bytes=part_bytes,
            filename=f"{Path(filename).stem}-retry{part_index}.wav",
            language=language,
        )
        part_text, _part_filter = _sanitize_chunk_text(part_data.get("text", ""))
        retry["part_text_lengths"].append(len(part_text))
        if part_text:
            merged, _overlap, _errors = _merge_overlapping_text(merged, part_text)
    retry["part_count"] = len(retry["part_text_lengths"])
    return merged, None, {
        "raw_text": raw_text,
        "audio_dbfs": dbfs,
        "empty_retry": retry,
    }


async def _run_qwen_transcription(
    *,
    audio_path: Path,
    filename: str,
    model: str,
    language: str,
    response_format: str,
    result_dir: Path,
) -> dict[str, Any]:
    total_duration = _wav_duration_seconds(audio_path)
    metadata = {
        "filename": filename,
        "model": model,
        "language": language,
        "response_format": response_format,
        "audio_path": str(audio_path),
        "duration": total_duration,
        "result_dir": str(result_dir),
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(result_dir / "metadata.json", metadata)
    await _cancel_idle_stop()
    async with semaphore:
        await _wait_for_qwen_admission()
        try:
            async for _ in _track_request({"filename": filename, "result_dir": str(result_dir)}):
                if audio_path.stat().st_size <= MAX_DIRECT_UPLOAD_MB * 1024 * 1024:
                    data = await _post_upstream_bytes(
                        audio_bytes=audio_path.read_bytes(),
                        filename=filename,
                        language=language,
                    )
                    text, filtered_reason = _sanitize_chunk_text(data.get("text", ""))
                    result = {
                        "text": text,
                        "segments": _pseudo_segments_for_text(text, total_duration, 0.0),
                        "model": model,
                        "duration": total_duration,
                    }
                    if filtered_reason:
                        result["filtered_chunk_reason"] = filtered_reason
                else:
                    chunk_results: list[dict[str, Any]] = []
                    merged_text = ""
                    merged_segments: list[dict[str, Any]] = []
                    for chunk in _iter_wav_chunks(
                        audio_path,
                        CHUNK_SECONDS,
                        CHUNK_OVERLAP_SECONDS,
                    ):
                        chunk_index = chunk["chunk_index"]
                        offset = chunk["offset_seconds"]
                        duration = chunk["duration_seconds"]
                        chunk_name = f"{Path(filename).stem}-chunk{chunk_index:04d}.wav"
                        text, filtered_reason, diagnostics = await _transcribe_chunk_with_empty_retry(
                            audio_bytes=chunk["audio_bytes"],
                            filename=chunk_name,
                            language=language,
                        )
                        merged_text, overlap_chars, overlap_errors = _merge_overlapping_text(
                            merged_text,
                            text,
                        )
                        if text:
                            novel_text = text[overlap_chars:]
                        else:
                            novel_text = ""
                        chunk_record = {
                            "chunk_index": chunk_index,
                            "offset_seconds": offset,
                            "duration_seconds": duration,
                            "request_offset_seconds": chunk["request_offset_seconds"],
                            "request_duration_seconds": chunk["request_duration_seconds"],
                            "leading_context_seconds": chunk["leading_context_seconds"],
                            "trailing_context_seconds": chunk["trailing_context_seconds"],
                            "text": text,
                            "novel_text": novel_text,
                            "overlap_chars": overlap_chars,
                            "overlap_errors": overlap_errors,
                            "audio_dbfs": diagnostics["audio_dbfs"],
                            "empty_retry": diagnostics["empty_retry"],
                        }
                        if filtered_reason:
                            chunk_record["filtered_reason"] = filtered_reason
                            chunk_record["raw_text_preview"] = diagnostics["raw_text"][:500]
                        chunk_results.append(chunk_record)
                        merged_segments.extend(
                            _pseudo_segments_for_text(novel_text, duration, offset)
                        )
                    result = {
                        "text": merged_text.strip(),
                        "segments": merged_segments,
                        "model": model,
                        "duration": total_duration,
                        "chunked": True,
                        "chunk_seconds": CHUNK_SECONDS,
                        "chunk_overlap_seconds": CHUNK_OVERLAP_SECONDS,
                        "chunk_count": len(chunk_results),
                    }
                    _write_json(result_dir / "chunks.json", chunk_results)
        except HTTPException as exc:
            _write_json(
                result_dir / "error.json",
                {
                    **metadata,
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            raise
    _write_json(result_dir / "response.json", result)
    _write_json(result_dir / "metadata.json", {**metadata, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    await _schedule_idle_stop()
    return result


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    scheduler = await _scheduler_get("/status-lite")
    if scheduler.get("qwen_proxy") not in {"running", "created"}:
        raise HTTPException(status_code=503, detail="proxy container not healthy")
    # Under the new cold-start design, the backend may be intentionally stopped.
    return {
        "status": "ok",
        "qwen_backend": str(scheduler.get("qwen_backend", "unknown")),
        "asr": str(scheduler.get("asr", "unknown")),
    }


@app.get("/busy")
async def busy() -> dict[str, Any]:
    return {
        "active_requests": ACTIVE_REQUESTS,
        "busy": ACTIVE_REQUESTS > 0,
        "current_request": _current_request_status(),
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("qwen3-asr-1p7b"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
) -> Any:
    filename = file.filename or "audio.wav"
    task_id, task_dir = _stage_task_dir(filename)
    task_dir.mkdir(parents=True, exist_ok=True)
    audio_path = task_dir / filename
    audio_path.write_bytes(await file.read())
    result_dir = RESULT_DIR / task_id
    result = await _run_qwen_transcription(
        audio_path=audio_path,
        filename=filename,
        model=model,
        language=language,
        response_format=response_format,
        result_dir=result_dir,
    )
    if response_format == "text":
        return result["text"]
    return result


@app.post("/jobs/upload")
async def upload_job(
    file: UploadFile = File(...),
    model: str = Form("qwen3-asr-1p7b"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
) -> dict[str, Any]:
    filename = file.filename or "audio.wav"
    task_id, task_dir = _stage_task_dir(filename)
    task_dir.mkdir(parents=True, exist_ok=True)
    audio_path = task_dir / filename
    audio_path.write_bytes(await file.read())
    result_dir = RESULT_DIR / task_id
    task = _task_set(
        task_id,
        id=task_id,
        status="queued",
        filename=filename,
        audio_path=str(audio_path),
        model=model,
        language=language,
        response_format=response_format,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        result_dir=str(result_dir),
    )
    _write_json(
        task_dir / "upload.json",
        {
            "filename": filename,
            "audio_path": str(audio_path),
            "size_bytes": audio_path.stat().st_size,
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    async def runner() -> None:
        _task_set(
            task_id,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        try:
            result = await _run_qwen_transcription(
                audio_path=audio_path,
                filename=filename,
                model=model,
                language=language,
                response_format=response_format,
                result_dir=result_dir,
            )
            _task_set(
                task_id,
                status="completed",
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                response_path=str(result_dir / "response.json"),
                metadata_path=str(result_dir / "metadata.json"),
                result=result,
            )
        except HTTPException as exc:
            _task_set(
                task_id,
                status="failed",
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                error={"status_code": exc.status_code, "detail": exc.detail},
                error_path=str(result_dir / "error.json"),
            )
        except Exception as exc:  # noqa: BLE001
            _task_set(
                task_id,
                status="failed",
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                error={"detail": str(exc)},
            )

    asyncio.create_task(runner())
    return task


class LocalFileTaskRequest(BaseModel):
    audio_path: str
    filename: str
    model: str = "qwen3-asr-1p7b"
    language: str = "zh"
    response_format: str = "verbose_json"


@app.get("/jobs/{task_id}")
async def get_job(task_id: str) -> dict[str, Any]:
    task = _task_snapshot(task_id)
    result_dir = Path(task.get("result_dir", ""))
    if task.get("status") == "completed":
        response_path = result_dir / "response.json"
        metadata_path = result_dir / "metadata.json"
        if response_path.exists():
            task["result"] = _read_json(response_path)
        if metadata_path.exists():
            task["metadata"] = _read_json(metadata_path)
    elif task.get("status") == "failed":
        error_path = result_dir / "error.json"
        if error_path.exists():
            task["error"] = _read_json(error_path)
    return task
