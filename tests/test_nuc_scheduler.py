from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


try:
    import docker as docker_module
except ModuleNotFoundError:
    docker_stub = types.ModuleType("docker")
    docker_stub.DockerClient = mock.Mock
    docker_stub.errors = types.SimpleNamespace(NotFound=type("NotFound", (Exception,), {}))
    docker_stub.types = types.SimpleNamespace(DeviceRequest=mock.Mock)
    sys.modules["docker"] = docker_stub
    docker_module = docker_stub

for missing_name in ("httpx", "fastapi", "pydantic"):
    if missing_name in sys.modules:
        continue
    if missing_name == "httpx":
        module = types.ModuleType("httpx")
        module.AsyncClient = mock.Mock
        module.HTTPError = Exception
    elif missing_name == "fastapi":
        module = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def get(self, *_args, **_kwargs):
                return lambda func: func

            def post(self, *_args, **_kwargs):
                return lambda func: func

        module.FastAPI = FastAPI

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str = "") -> None:
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        module.HTTPException = HTTPException
    else:
        module = types.ModuleType("pydantic")

        class BaseModel:
            pass

        module.BaseModel = BaseModel
    sys.modules[missing_name] = module


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "nuc_service_scheduler.py"
SPEC = importlib.util.spec_from_file_location("nuc_scheduler_for_tests", MODULE_PATH)
assert SPEC and SPEC.loader
scheduler = importlib.util.module_from_spec(SPEC)
with mock.patch.object(docker_module, "DockerClient", return_value=mock.Mock()):
    SPEC.loader.exec_module(scheduler)


class FakeContainer:
    def __init__(self, model: str, status: str = "exited") -> None:
        self.status = status
        self.attrs = {"Config": {"Env": [f"WHISPER__MODEL={model}"]}}
        self.started = False

    def reload(self) -> None:
        return None

    def start(self) -> None:
        self.status = "running"
        self.started = True


class NucSchedulerModelTests(unittest.TestCase):
    def test_container_model_is_read_from_environment(self) -> None:
        container = FakeContainer("large-v3")
        self.assertEqual(scheduler._container_asr_model(container), "large-v3")

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(scheduler.HTTPException) as raised:
            scheduler._normalize_asr_model("unknown/model")
        self.assertEqual(raised.exception.status_code, 400)

    def test_backend_command_bypasses_uv_run_lock_sync(self) -> None:
        self.assertIn(".venv/bin/uvicorn", scheduler.ASR_BACKEND_COMMAND[0])
        self.assertNotIn("uv", scheduler.ASR_BACKEND_COMMAND[:1])

    def test_ensure_reuses_matching_container_without_recreate(self) -> None:
        container = FakeContainer("large-v3")
        scheduler.docker_client.containers.get.return_value = container

        async def run() -> dict:
            with (
                mock.patch.object(scheduler, "_container_status", return_value="exited"),
                mock.patch.object(scheduler, "_require_gpu_snapshot", return_value={"available": True}),
                mock.patch.object(scheduler, "_wait_http_ready", new=mock.AsyncMock()),
                mock.patch.object(scheduler, "_recreate_asr_backend") as recreate,
            ):
                result = await scheduler.ensure_asr(types.SimpleNamespace(model="large-v3"))
                recreate.assert_not_called()
                return result

        result = asyncio.run(run())
        self.assertTrue(container.started)
        self.assertFalse(result["switched"])
        self.assertEqual(result["model"], "large-v3")

    def test_ensure_recreates_container_for_different_model(self) -> None:
        current = FakeContainer("large-v3")
        replacement = FakeContainer(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            status="running",
        )
        scheduler.docker_client.containers.get.return_value = current

        async def run() -> dict:
            with (
                mock.patch.object(scheduler, "_container_status", return_value="exited"),
                mock.patch.object(scheduler, "_require_gpu_snapshot", return_value={"available": True}),
                mock.patch.object(scheduler, "_wait_http_ready", new=mock.AsyncMock()),
                mock.patch.object(
                    scheduler,
                    "_recreate_asr_backend",
                    return_value=replacement,
                ) as recreate,
            ):
                result = await scheduler.ensure_asr(
                    types.SimpleNamespace(
                        model="deepdml/faster-whisper-large-v3-turbo-ct2"
                    )
                )
                recreate.assert_called_once_with(
                    "deepdml/faster-whisper-large-v3-turbo-ct2"
                )
                return result

        result = asyncio.run(run())
        self.assertTrue(result["switched"])


if __name__ == "__main__":
    unittest.main()
