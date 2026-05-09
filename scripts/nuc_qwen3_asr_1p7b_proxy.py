from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


UPSTREAM_URL = os.environ.get(
    "UPSTREAM_URL",
    "http://nuc-qwen3-asr-1p7b-vllm:8000/v1/audio/transcriptions",
)
SCHEDULER_URL = os.environ.get(
    "SCHEDULER_URL",
    "http://nuc-service-scheduler:8010",
)
UPSTREAM_MODEL = "Qwen/Qwen3-ASR-1.7B"
GPU_MIN_FREE_MB = 1800
QUEUE_CONCURRENCY = 1
IDLE_STOP_SECONDS = int(os.environ.get("IDLE_STOP_SECONDS", "180"))

app = FastAPI(title="NUC Qwen3-ASR 1.7B Proxy")
semaphore = asyncio.Semaphore(QUEUE_CONCURRENCY)
idle_stop_task: asyncio.Task | None = None
idle_stop_lock = asyncio.Lock()


async def _gpu_free_mb() -> int:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return int(stdout.decode("utf-8").strip().splitlines()[0])
    except Exception:
        return 0


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


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("qwen3-asr-1p7b"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
) -> Any:
    if semaphore.locked():
        raise HTTPException(status_code=429, detail="NUC Qwen3-ASR 1.7B queue is full")
    await _cancel_idle_stop()
    await _scheduler_post("/admit/qwen")
    await _scheduler_post("/ensure/qwen")
    free_mb = await _gpu_free_mb()
    if free_mb < GPU_MIN_FREE_MB:
        raise HTTPException(status_code=503, detail=f"GPU free memory too low: {free_mb} MiB")

    async with semaphore:
        audio_bytes = await file.read()
        files = {
            "file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav"),
        }
        data_payload = {
            "model": UPSTREAM_MODEL,
            "language": language,
        }
        async with httpx.AsyncClient(timeout=900) as client:
            response = await client.post(UPSTREAM_URL, data=data_payload, files=files)
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            data = response.json()
        text = _clean_transcript(data.get("text", ""))
        await _schedule_idle_stop()
        if response_format == "text":
            return text
        return {
            "text": text,
            "segments": _pseudo_segments(text),
            "model": model,
            "duration": 30.0,
        }
