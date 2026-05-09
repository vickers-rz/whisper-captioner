from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://nuc-asr-backend:8000")
UPSTREAM_HEALTH_URL = os.environ.get("UPSTREAM_HEALTH_URL", f"{UPSTREAM_BASE_URL}/health")
TRANSCRIBE_PATH = "/v1/audio/transcriptions"
ACTIVE_REQUESTS = 0


@asynccontextmanager
async def _track_active_requests():
    global ACTIVE_REQUESTS
    ACTIVE_REQUESTS += 1
    try:
        yield
    finally:
        ACTIVE_REQUESTS = max(0, ACTIVE_REQUESTS - 1)


async def _get_upstream_health() -> tuple[bool, str]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(UPSTREAM_HEALTH_URL)
            return response.status_code < 400, response.text
        except httpx.HTTPError as exc:
            return False, str(exc)


app = FastAPI(title="NUC faster-whisper Busy Proxy")


@app.get("/health")
async def health() -> Any:
    healthy, body = await _get_upstream_health()
    if not healthy:
        raise HTTPException(status_code=503, detail={"upstream": "unhealthy", "message": body})
    return {"status": "ok", "active_requests": ACTIVE_REQUESTS, "upstream": "healthy"}


@app.get("/busy")
async def busy() -> dict[str, Any]:
    healthy, message = await _get_upstream_health()
    return {
        "active_requests": ACTIVE_REQUESTS,
        "busy": ACTIVE_REQUESTS > 0,
        "upstream_healthy": healthy,
        "upstream_message": message if not healthy else "ok",
    }


@app.post(TRANSCRIBE_PATH)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
) -> Any:
    audio_bytes = await file.read()
    files = {
        "file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav"),
    }
    data = {
        "model": model,
        "language": language,
        "response_format": response_format,
    }

    async with _track_active_requests():
        async with httpx.AsyncClient(timeout=900) as client:
            try:
                response = await client.post(
                    f"{UPSTREAM_BASE_URL}{TRANSCRIBE_PATH}",
                    data=data,
                    files=files,
                )
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=503, detail=f"faster-whisper upstream error: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
