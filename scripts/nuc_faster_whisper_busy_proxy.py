from __future__ import annotations

import asyncio
import os
import json
import re
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel


UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://nuc-asr-backend:8000")
UPSTREAM_HEALTH_URL = os.environ.get("UPSTREAM_HEALTH_URL", f"{UPSTREAM_BASE_URL}/health")
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "http://127.0.0.1:8010")
RESULT_DIR = Path(os.environ.get("ASR_RESULT_DIR", "/app/asr-results"))
STAGING_DIR = Path(os.environ.get("ASR_STAGING_DIR", "/app/asr-staging"))
HOST_SERVICE_DIR = Path(os.environ.get("ASR_HOST_SERVICE_DIR", "/srv/qwen3-asr-1p7b"))
CONTAINER_SERVICE_DIR = Path(os.environ.get("ASR_CONTAINER_SERVICE_DIR", "/app"))
TRANSCRIBE_PATH = "/v1/audio/transcriptions"
ACTIVE_REQUESTS = 0
CURRENT_REQUEST: dict[str, Any] | None = None
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


def _safe_name(value: str, fallback: str = "audio") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", value).strip("._")
    return cleaned[:120] or fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _request_result_dir(filename: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return RESULT_DIR / f"{stamp}-{uuid.uuid4().hex[:8]}-{_safe_name(filename)}"


def _stage_task_dir(filename: str) -> tuple[str, Path]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    task_id = f"{stamp}-{uuid.uuid4().hex[:8]}-{_safe_name(filename)}"
    return task_id, STAGING_DIR / task_id


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_snapshot(task_id: str) -> dict[str, Any]:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return dict(task)


def _task_set(task_id: str, **updates: Any) -> dict[str, Any]:
    with TASKS_LOCK:
        task = TASKS.setdefault(task_id, {})
        task.update(updates)
        return dict(task)


def _task_update(task_name: str, **updates: Any) -> dict[str, Any]:
    return _task_set(task_name, **updates)


def _resolve_audio_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.exists():
        return raw
    try:
        relative = raw.relative_to(HOST_SERVICE_DIR)
    except ValueError:
        return raw
    mapped = CONTAINER_SERVICE_DIR / relative
    return mapped


@asynccontextmanager
async def _track_active_requests(metadata: dict[str, Any]):
    global ACTIVE_REQUESTS
    global CURRENT_REQUEST
    ACTIVE_REQUESTS += 1
    CURRENT_REQUEST = {
        **metadata,
        "started_at": time.time(),
    }
    try:
        yield
    finally:
        ACTIVE_REQUESTS = max(0, ACTIVE_REQUESTS - 1)
        if ACTIVE_REQUESTS == 0:
            CURRENT_REQUEST = None


def _current_request_status() -> dict[str, Any] | None:
    if not CURRENT_REQUEST:
        return None
    started_at = float(CURRENT_REQUEST.get("started_at", 0.0))
    return {
        **CURRENT_REQUEST,
        "elapsed_seconds": max(0.0, time.time() - started_at),
    }


async def _get_upstream_health() -> tuple[bool, str]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(UPSTREAM_HEALTH_URL)
            return response.status_code < 400, response.text
        except httpx.HTTPError as exc:
            return False, str(exc)


async def _scheduler_post(path: str) -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(f"{SCHEDULER_URL}{path}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"scheduler unavailable: {exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)


async def _transcribe_local_file_to_result(
    *,
    audio_path: Path,
    filename: str,
    model: str,
    language: str,
    response_format: str,
    result_dir: Path,
) -> dict[str, Any]:
    await _scheduler_post("/admit/asr")
    await _scheduler_post("/ensure/asr")
    metadata = {
        "filename": filename,
        "model": model,
        "language": language,
        "response_format": response_format,
        "audio_path": str(audio_path),
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "result_dir": str(result_dir),
    }
    _write_json(result_dir / "metadata.json", metadata)
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "-X",
        "POST",
        f"{UPSTREAM_BASE_URL}{TRANSCRIBE_PATH}",
        "-F",
        f"model={model}",
        "-F",
        f"language={language}",
        "-F",
        f"response_format={response_format}",
        "-F",
        f"file=@{audio_path};type=audio/wav",
    ]
    async with _track_active_requests({
        "filename": filename,
        "model": model,
        "language": language,
        "result_dir": str(result_dir),
        "audio_path": str(audio_path),
    }):
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="ignore").strip() or f"curl exited {proc.returncode}"
        _write_json(
            result_dir / "error.json",
            {**metadata, "error": detail, "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        )
        raise HTTPException(status_code=503, detail=f"faster-whisper upstream error: {detail}")
    try:
        result = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        detail = stdout.decode("utf-8", errors="ignore")[:1000]
        _write_json(
            result_dir / "error.json",
            {**metadata, "error": f"invalid JSON from upstream: {detail}", "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        )
        raise HTTPException(status_code=503, detail=f"invalid upstream JSON: {detail}") from exc
    _write_json(result_dir / "response.json", result)
    _write_json(result_dir / "metadata.json", {**metadata, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if isinstance(result, dict):
        result.setdefault("nuc_result_dir", str(result_dir))
    return result


class LocalFileTaskRequest(BaseModel):
    audio_path: str
    filename: str
    model: str = "large-v3"
    language: str = "zh"
    response_format: str = "verbose_json"


def _create_task_record(
    *,
    task_id: str,
    filename: str,
    audio_path: Path,
    model: str,
    language: str,
    response_format: str,
    result_dir: Path,
) -> dict[str, Any]:
    return _task_update(
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


def _start_local_file_task(
    *,
    task_id: str,
    filename: str,
    audio_path: Path,
    model: str,
    language: str,
    response_format: str,
    result_dir: Path,
) -> None:
    async def runner() -> None:
        _task_update(
            task_id,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        try:
            result = await _transcribe_local_file_to_result(
                audio_path=audio_path,
                filename=filename,
                model=model,
                language=language,
                response_format=response_format,
                result_dir=result_dir,
            )
            _task_update(
                task_id,
                status="completed",
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                response_path=str(result_dir / "response.json"),
                metadata_path=str(result_dir / "metadata.json"),
                result=result,
            )
        except HTTPException as exc:
            _task_update(
                task_id,
                status="failed",
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                error={"status_code": exc.status_code, "detail": exc.detail},
                error_path=str(result_dir / "error.json"),
            )
        except Exception as exc:  # noqa: BLE001
            _task_update(
                task_id,
                status="failed",
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                error={"detail": str(exc)},
            )

    asyncio.create_task(runner())


app = FastAPI(title="NUC faster-whisper Busy Proxy")


@app.get("/health")
async def health() -> Any:
    healthy, body = await _get_upstream_health()
    return {
        "status": "ok",
        "active_requests": ACTIVE_REQUESTS,
        "upstream": "healthy" if healthy else "stopped_or_unhealthy",
        "upstream_message": "ok" if healthy else body,
        "current_request": _current_request_status(),
    }


@app.get("/busy")
async def busy() -> dict[str, Any]:
    healthy, message = await _get_upstream_health()
    return {
        "active_requests": ACTIVE_REQUESTS,
        "busy": ACTIVE_REQUESTS > 0,
        "upstream_healthy": healthy,
        "upstream_message": message if not healthy else "ok",
        "current_request": _current_request_status(),
    }


@app.post(TRANSCRIBE_PATH)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
) -> Any:
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"
    result_dir = _request_result_dir(filename)
    result_dir.mkdir(parents=True, exist_ok=True)
    audio_path = result_dir / filename
    audio_path.write_bytes(audio_bytes)
    metadata = {
        "filename": filename,
        "model": model,
        "language": language,
        "response_format": response_format,
        "content_type": file.content_type or "audio/wav",
        "audio_bytes": len(audio_bytes),
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "result_dir": str(result_dir),
        "audio_path": str(audio_path),
    }
    _write_json(result_dir / "metadata.json", metadata)
    return await _transcribe_local_file_to_result(
        audio_path=audio_path,
        filename=filename,
        model=model,
        language=language,
        response_format=response_format,
        result_dir=result_dir,
    )


@app.post("/jobs/local-file")
async def submit_local_file_job(payload: LocalFileTaskRequest) -> dict[str, Any]:
    audio_path = _resolve_audio_path(payload.audio_path)
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found on NUC: {audio_path}")
    task_id, task_dir = _stage_task_dir(payload.filename)
    result_dir = task_dir / "result"
    task = _create_task_record(
        task_id=task_id,
        filename=payload.filename,
        audio_path=audio_path,
        model=payload.model,
        language=payload.language,
        response_format=payload.response_format,
        result_dir=result_dir,
    )
    _start_local_file_task(
        task_id=task_id,
        filename=payload.filename,
        audio_path=audio_path,
        model=payload.model,
        language=payload.language,
        response_format=payload.response_format,
        result_dir=result_dir,
    )
    return task


@app.post("/jobs/upload")
async def upload_local_file_job(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
) -> dict[str, Any]:
    filename = file.filename or "audio.wav"
    task_id, task_dir = _stage_task_dir(filename)
    task_dir.mkdir(parents=True, exist_ok=True)
    audio_path = task_dir / filename
    audio_path.write_bytes(await file.read())
    result_dir = task_dir / "result"
    task = _create_task_record(
        task_id=task_id,
        filename=filename,
        audio_path=audio_path,
        model=model,
        language=language,
        response_format=response_format,
        result_dir=result_dir,
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
    _start_local_file_task(
        task_id=task_id,
        filename=filename,
        audio_path=audio_path,
        model=model,
        language=language,
        response_format=response_format,
        result_dir=result_dir,
    )
    return task


@app.get("/jobs/{task_id}")
async def get_job(task_id: str) -> dict[str, Any]:
    task = _task_snapshot(task_id)
    result_dir = Path(task.get("result_dir", ""))
    if task.get("status") == "completed":
        response_path = result_dir / "response.json"
        metadata_path = result_dir / "metadata.json"
        if response_path.exists():
            task["result"] = _read_json(response_path)
            task["response_path"] = str(response_path)
        if metadata_path.exists():
            task["metadata"] = _read_json(metadata_path)
    elif task.get("status") == "failed":
        error_path = result_dir / "error.json"
        if error_path.exists():
            task["error"] = _read_json(error_path)
            task["error_path"] = str(error_path)
    return task
