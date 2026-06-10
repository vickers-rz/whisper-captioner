from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import docker
import httpx
from fastapi import FastAPI, HTTPException


DOCKER_BASE_URL = os.environ.get("DOCKER_BASE_URL", "unix://var/run/docker.sock")
QWEN_BACKEND_CONTAINER = os.environ.get("QWEN_BACKEND_CONTAINER", "nuc-qwen3-asr-1p7b-vllm")
QWEN_PROXY_CONTAINER = os.environ.get("QWEN_PROXY_CONTAINER", "nuc-qwen3-asr-1p7b-proxy")
ASR_CONTAINER = os.environ.get("ASR_CONTAINER", "nuc-asr")
ASR_BACKEND_CONTAINER = os.environ.get("ASR_BACKEND_CONTAINER", "nuc-asr-backend")
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
ASR_BACKEND_HEALTH_URL = os.environ.get("ASR_BACKEND_HEALTH_URL", "http://nuc-asr-backend:8000/health")
ASR_BACKEND_START_TIMEOUT = float(os.environ.get("ASR_BACKEND_START_TIMEOUT", "120"))
ASR_IDLE_SECONDS = float(os.environ.get("ASR_IDLE_SECONDS", "180"))
QWEN_BUSY_URL = os.environ.get("QWEN_BUSY_URL", "http://nuc-qwen3-asr-1p7b-proxy:8000/busy")

app = FastAPI(title="NUC Service Scheduler")
docker_client = docker.DockerClient(base_url=DOCKER_BASE_URL)
_asr_idle_deadline: float | None = None
_asr_idle_task: asyncio.Task[None] | None = None


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


def _gpu_snapshot() -> dict[str, Any]:
    import subprocess

    try:
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
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "error": str(exc),
            "memory_used_mb": -1,
            "memory_free_mb": -1,
            "gpu_utilization": -1,
        }
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "available": False,
            "error": result.stderr.strip() or f"nvidia-smi exited {result.returncode}",
            "memory_used_mb": -1,
            "memory_free_mb": -1,
            "gpu_utilization": -1,
        }
    try:
        used, free, util = [int(part.strip()) for part in result.stdout.strip().splitlines()[0].split(",")]
        return {
            "available": True,
            "error": None,
            "memory_used_mb": used,
            "memory_free_mb": free,
            "gpu_utilization": util,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"Could not parse nvidia-smi output: {exc}",
            "memory_used_mb": -1,
            "memory_free_mb": -1,
            "gpu_utilization": -1,
        }


def _require_gpu_snapshot() -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    if not snapshot.get("available"):
        raise HTTPException(
            status_code=503,
            detail=f"GPU status unavailable: {snapshot.get('error') or 'unknown nvidia-smi error'}",
        )
    return snapshot


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


async def _qwen_busy_snapshot() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(QWEN_BUSY_URL)
            if response.status_code >= 400:
                return {"available": False, "active_requests": 0, "busy": False}
            data = response.json()
            return {
                "available": True,
                "active_requests": int(data.get("active_requests", 0) or 0),
                "busy": bool(data.get("busy", False)),
            }
        except Exception:
            return {"available": False, "active_requests": 0, "busy": False}


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


async def _wait_for_gpu_memory_free(min_free_mb: int, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_snapshot: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        snapshot = _require_gpu_snapshot()
        last_snapshot = snapshot
        if snapshot["memory_free_mb"] >= min_free_mb:
            return
        await asyncio.sleep(1)
    raise HTTPException(
        status_code=503,
        detail=(
            "Timed out waiting for GPU memory to free after stopping a container: "
            f"last_snapshot={last_snapshot or {}}"
        ),
    )


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


def _cancel_asr_idle_stop() -> None:
    global _asr_idle_deadline
    global _asr_idle_task
    _asr_idle_deadline = None
    if _asr_idle_task and not _asr_idle_task.done():
        _asr_idle_task.cancel()
    _asr_idle_task = None


async def _schedule_asr_idle_stop() -> None:
    global _asr_idle_deadline
    global _asr_idle_task
    _cancel_asr_idle_stop()
    _asr_idle_deadline = time.monotonic() + ASR_IDLE_SECONDS

    async def worker() -> None:
        global _asr_idle_deadline
        global _asr_idle_task
        try:
            while _asr_idle_deadline is not None:
                remaining = _asr_idle_deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, 5.0))
            asr_busy = await _asr_busy_snapshot()
            if asr_busy.get("active_requests", 0) > 0 or asr_busy.get("busy"):
                return
            try:
                _stop_container(ASR_BACKEND_CONTAINER)
            except HTTPException:
                return
        finally:
            _asr_idle_deadline = None
            _asr_idle_task = None

    _asr_idle_task = asyncio.create_task(worker())


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, Any]:
    asr_busy = await _asr_busy_snapshot()
    qwen_busy = await _qwen_busy_snapshot()
    return {
        "qwen_backend": _container_status(QWEN_BACKEND_CONTAINER),
        "qwen_proxy": _container_status(QWEN_PROXY_CONTAINER),
        "qwen_busy": qwen_busy,
        "asr": _container_status(ASR_CONTAINER),
        "asr_backend": _container_status(ASR_BACKEND_CONTAINER),
        "asr_busy": asr_busy,
        "asr_idle_seconds": ASR_IDLE_SECONDS,
        "asr_idle_deadline_epoch": (
            time.time() + max(0.0, _asr_idle_deadline - time.monotonic())
            if _asr_idle_deadline
            else None
        ),
        "gpu": _gpu_snapshot(),
    }


@app.get("/status-lite")
async def status_lite() -> dict[str, Any]:
    asr_busy = await _asr_busy_snapshot()
    qwen_busy = await _qwen_busy_snapshot()
    return {
        "qwen_backend": _container_status(QWEN_BACKEND_CONTAINER),
        "qwen_proxy": _container_status(QWEN_PROXY_CONTAINER),
        "qwen_busy": qwen_busy,
        "asr": _container_status(ASR_CONTAINER),
        "asr_backend": _container_status(ASR_BACKEND_CONTAINER),
        "asr_busy": asr_busy,
        "asr_idle_seconds": ASR_IDLE_SECONDS,
    }


@app.post("/admit/qwen")
async def admit_qwen() -> dict[str, Any]:
    gpu = _require_gpu_snapshot()
    asr_running = _container_status(ASR_CONTAINER) == "running"
    asr_backend_running = _container_status(ASR_BACKEND_CONTAINER) == "running"
    qwen_backend_running = _container_status(QWEN_BACKEND_CONTAINER) == "running"
    asr_healthy = await _asr_health_ok() if asr_running else False
    asr_busy = await _asr_busy_snapshot() if asr_running else {"available": False, "active_requests": 0, "busy": False}

    if asr_backend_running and asr_busy.get("busy"):
        raise HTTPException(
            status_code=429,
            detail="faster-whisper currently has active requests; Qwen must wait",
        )

    if (
        not qwen_backend_running
        and not asr_backend_running
        and gpu["memory_free_mb"] < MIN_FREE_MB_TO_START_QWEN
    ):
        raise HTTPException(
            status_code=503,
            detail=f"GPU free memory too low for Qwen admission: {gpu['memory_free_mb']} MiB",
        )

    return {
        "status": "admitted",
        "gpu": gpu,
        "asr_running": asr_running,
        "asr_backend_running": asr_backend_running,
        "asr_healthy": asr_healthy,
        "asr_busy": asr_busy,
        "qwen_backend_running": qwen_backend_running,
        "priority": "qwen",
    }


@app.post("/admit/asr")
async def admit_asr() -> dict[str, Any]:
    _cancel_asr_idle_stop()
    gpu = _require_gpu_snapshot()
    qwen_backend_running = _container_status(QWEN_BACKEND_CONTAINER) == "running"
    qwen_busy = await _qwen_busy_snapshot() if qwen_backend_running else {"available": False, "active_requests": 0, "busy": False}

    if qwen_backend_running and qwen_busy.get("busy"):
        raise HTTPException(
            status_code=429,
            detail=(
                "Qwen3-ASR currently has active requests; "
                "defer faster-whisper request because Qwen now has higher scheduling priority"
            ),
        )

    if (
        qwen_backend_running
        and not qwen_busy.get("available")
        and gpu["gpu_utilization"] >= GPU_UTILIZATION_BUSY_THRESHOLD
    ):
        raise HTTPException(
            status_code=429,
            detail=(
                "Qwen busy endpoint unavailable while GPU utilization is high; "
                "defer faster-whisper request because Qwen now has higher scheduling priority"
            ),
        )

    return {
        "status": "admitted",
        "gpu": gpu,
        "qwen_backend_running": qwen_backend_running,
        "qwen_busy": qwen_busy,
        "priority": "realtime_asr",
    }


@app.post("/ensure/qwen")
async def ensure_qwen() -> dict[str, str]:
    _cancel_asr_idle_stop()
    asr_running = _container_status(ASR_BACKEND_CONTAINER) == "running"
    asr_busy = await _asr_busy_snapshot() if asr_running else {"busy": False, "active_requests": 0}
    _require_gpu_snapshot()

    if asr_running and asr_busy.get("busy"):
        raise HTTPException(
            status_code=429,
            detail="faster-whisper currently has active requests; Qwen must wait",
        )
    if asr_running:
        _stop_container(ASR_BACKEND_CONTAINER)
        await _wait_for_gpu_memory_free(MIN_FREE_MB_TO_START_QWEN, timeout_seconds=30)

    container = _get_container(QWEN_BACKEND_CONTAINER)
    container.reload()
    if container.status != "running":
        try:
            container.start()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not start {QWEN_BACKEND_CONTAINER}: {exc}") from exc
    await _wait_http_ready(QWEN_BACKEND_HEALTH_URL, QWEN_BACKEND_START_TIMEOUT)
    return {"status": "ready", "container": QWEN_BACKEND_CONTAINER}


@app.post("/ensure/asr")
async def ensure_asr() -> dict[str, str]:
    _cancel_asr_idle_stop()
    qwen_backend_running = _container_status(QWEN_BACKEND_CONTAINER) == "running"
    qwen_busy = await _qwen_busy_snapshot() if qwen_backend_running else {"busy": False, "active_requests": 0}
    if qwen_backend_running and qwen_busy.get("busy"):
        raise HTTPException(
            status_code=429,
            detail="Qwen3-ASR currently has active requests; faster-whisper must wait",
        )

    _require_gpu_snapshot()
    if qwen_backend_running:
        _stop_container(QWEN_BACKEND_CONTAINER)
        await _wait_for_gpu_memory_free(MIN_FREE_MB_TO_START_QWEN, timeout_seconds=30)

    container = _get_container(ASR_BACKEND_CONTAINER)
    container.reload()
    if container.status != "running":
        try:
            container.start()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not start {ASR_BACKEND_CONTAINER}: {exc}") from exc
    await _wait_http_ready(ASR_BACKEND_HEALTH_URL, ASR_BACKEND_START_TIMEOUT)
    return {"status": "ready", "container": ASR_BACKEND_CONTAINER}


@app.post("/stop/qwen")
async def stop_qwen() -> dict[str, Any]:
    result = {
        "proxy": _container_status(QWEN_PROXY_CONTAINER),
        "backend": _stop_container(QWEN_BACKEND_CONTAINER),
    }
    return result


@app.post("/stop/asr")
async def stop_asr() -> dict[str, Any]:
    _cancel_asr_idle_stop()
    result = {
        "proxy": _container_status(ASR_CONTAINER),
        "backend": _stop_container(ASR_BACKEND_CONTAINER),
    }
    return result


@app.post("/restart/asr")
async def restart_asr() -> dict[str, str]:
    _cancel_asr_idle_stop()
    container = _get_container(ASR_BACKEND_CONTAINER)
    try:
        container.restart(timeout=STOP_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not restart {ASR_BACKEND_CONTAINER}: {exc}") from exc
    return {"status": "restarted", "container": ASR_BACKEND_CONTAINER}


@app.post("/release/asr")
async def release_asr() -> dict[str, Any]:
    asr_busy = await _asr_busy_snapshot()
    if asr_busy.get("active_requests", 0) > 0 or asr_busy.get("busy"):
        return {
            "status": "busy",
            "detail": "faster-whisper realtime still has active requests",
            "asr_busy": asr_busy,
        }
    await _schedule_asr_idle_stop()
    return {
        "status": "idle_timer_started",
        "idle_seconds": ASR_IDLE_SECONDS,
        "asr_busy": asr_busy,
    }
