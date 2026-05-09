from __future__ import annotations

import os
import time
from typing import Any

import docker
import httpx
from fastapi import FastAPI, HTTPException
import asyncio


DOCKER_BASE_URL = os.environ.get("DOCKER_BASE_URL", "unix://var/run/docker.sock")
QWEN_BACKEND_CONTAINER = os.environ.get("QWEN_BACKEND_CONTAINER", "nuc-qwen3-asr-1p7b-vllm")
QWEN_PROXY_CONTAINER = os.environ.get("QWEN_PROXY_CONTAINER", "nuc-qwen3-asr-1p7b-proxy")
ASR_CONTAINER = os.environ.get("ASR_CONTAINER", "nuc-asr")
QWEN_BACKEND_HEALTH_URL = os.environ.get(
    "QWEN_BACKEND_HEALTH_URL",
    "http://nuc-qwen3-asr-1p7b-vllm:8000/v1/models",
)
QWEN_BACKEND_START_TIMEOUT = float(os.environ.get("QWEN_BACKEND_START_TIMEOUT", "120"))
STOP_TIMEOUT_SECONDS = int(os.environ.get("STOP_TIMEOUT_SECONDS", "12"))
MIN_FREE_MB_TO_START_QWEN = int(os.environ.get("MIN_FREE_MB_TO_START_QWEN", "2600"))
GPU_UTILIZATION_BUSY_THRESHOLD = int(os.environ.get("GPU_UTILIZATION_BUSY_THRESHOLD", "40"))
ASR_HEALTH_URL = os.environ.get("ASR_HEALTH_URL", "http://127.0.0.1:8000/health")
ASR_BUSY_URL = os.environ.get("ASR_BUSY_URL", "http://127.0.0.1:8000/busy")

app = FastAPI(title="NUC Service Scheduler")
docker_client = docker.DockerClient(base_url=DOCKER_BASE_URL)


def _get_container(name: str):
    try:
        return docker_client.containers.get(name)
    except docker.errors.NotFound as exc:
        raise HTTPException(status_code=404, detail=f"Container not found: {name}") from exc


def _container_status(name: str) -> str:
    try:
        container = docker_client.containers.get(name)
    except docker.errors.NotFound:
        return "missing"
    container.reload()
    return container.status


def _gpu_snapshot() -> dict[str, int]:
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"memory_used_mb": -1, "memory_free_mb": -1, "gpu_utilization": -1}
    try:
        used, free, util = [int(part.strip()) for part in result.stdout.strip().splitlines()[0].split(",")]
        return {
            "memory_used_mb": used,
            "memory_free_mb": free,
            "gpu_utilization": util,
        }
    except Exception:
        return {"memory_used_mb": -1, "memory_free_mb": -1, "gpu_utilization": -1}


async def _asr_health_ok() -> bool:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(ASR_HEALTH_URL)
            return response.status_code < 400
        except Exception:
            return False


async def _asr_busy_snapshot() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(ASR_BUSY_URL)
            if response.status_code >= 400:
                return {"available": False, "active_requests": 0, "busy": False, "upstream_healthy": False}
            data = response.json()
            return {
                "available": True,
                "active_requests": int(data.get("active_requests", 0) or 0),
                "busy": bool(data.get("busy", False)),
                "upstream_healthy": bool(data.get("upstream_healthy", False)),
            }
        except Exception:
            return {"available": False, "active_requests": 0, "busy": False, "upstream_healthy": False}


async def _wait_http_ready(url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code < 400:
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            await asyncio.sleep(1)
    raise HTTPException(status_code=503, detail=f"Timed out waiting for upstream readiness: {last_error or 'unknown error'}")


def _stop_container(name: str) -> dict[str, str]:
    container = _get_container(name)
    container.reload()
    if container.status != "running":
        return {"status": "already_stopped", "container": name}
    try:
        container.stop(timeout=STOP_TIMEOUT_SECONDS)
        return {"status": "stopped", "container": name}
    except Exception:  # noqa: BLE001
        try:
            container.kill()
            return {"status": "killed", "container": name}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not stop or kill {name}: {exc}") from exc


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, Any]:
    asr_busy = await _asr_busy_snapshot()
    return {
        "qwen_backend": _container_status(QWEN_BACKEND_CONTAINER),
        "qwen_proxy": _container_status(QWEN_PROXY_CONTAINER),
        "asr": _container_status(ASR_CONTAINER),
        "asr_busy": asr_busy,
        "gpu": _gpu_snapshot(),
    }


@app.get("/status-lite")
async def status_lite() -> dict[str, Any]:
    asr_busy = await _asr_busy_snapshot()
    return {
        "qwen_backend": _container_status(QWEN_BACKEND_CONTAINER),
        "qwen_proxy": _container_status(QWEN_PROXY_CONTAINER),
        "asr": _container_status(ASR_CONTAINER),
        "asr_busy": asr_busy,
    }


@app.post("/admit/qwen")
async def admit_qwen() -> dict[str, Any]:
    gpu = _gpu_snapshot()
    asr_running = _container_status(ASR_CONTAINER) == "running"
    asr_healthy = await _asr_health_ok() if asr_running else False
    asr_busy = await _asr_busy_snapshot() if asr_running else {"available": False, "active_requests": 0, "busy": False}

    if asr_running and asr_busy.get("busy"):
        raise HTTPException(
            status_code=429,
            detail=(
                "faster-whisper currently has active requests; "
                "defer Qwen request to protect realtime/default ASR lane"
            ),
        )

    if gpu["memory_free_mb"] >= 0 and gpu["memory_free_mb"] < MIN_FREE_MB_TO_START_QWEN:
        raise HTTPException(
            status_code=503,
            detail=f"GPU free memory too low for Qwen admission: {gpu['memory_free_mb']} MiB",
        )

    if (
        asr_running
        and asr_healthy
        and not asr_busy.get("available")
        and gpu["gpu_utilization"] >= GPU_UTILIZATION_BUSY_THRESHOLD
    ):
        raise HTTPException(
            status_code=429,
            detail=(
                "faster-whisper busy endpoint unavailable and GPU utilization is high; "
                "defer Qwen request to protect realtime/default ASR lane"
            ),
        )

    return {
        "status": "admitted",
        "gpu": gpu,
        "asr_running": asr_running,
        "asr_healthy": asr_healthy,
        "asr_busy": asr_busy,
    }


@app.post("/ensure/qwen")
async def ensure_qwen() -> dict[str, str]:
    container = _get_container(QWEN_BACKEND_CONTAINER)
    container.reload()
    if container.status != "running":
        try:
            container.start()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not start {QWEN_BACKEND_CONTAINER}: {exc}") from exc
    await _wait_http_ready(QWEN_BACKEND_HEALTH_URL, QWEN_BACKEND_START_TIMEOUT)
    return {"status": "ready", "container": QWEN_BACKEND_CONTAINER}


@app.post("/stop/qwen")
async def stop_qwen() -> dict[str, Any]:
    result = {
        "proxy": _container_status(QWEN_PROXY_CONTAINER),
        "backend": _stop_container(QWEN_BACKEND_CONTAINER),
    }
    return result


@app.post("/restart/asr")
async def restart_asr() -> dict[str, str]:
    container = _get_container(ASR_CONTAINER)
    try:
        container.restart(timeout=STOP_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not restart {ASR_CONTAINER}: {exc}") from exc
    return {"status": "restarted", "container": ASR_CONTAINER}
