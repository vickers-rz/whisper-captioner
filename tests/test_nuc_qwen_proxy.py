from __future__ import annotations

import asyncio
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave


try:
    import fastapi  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = types.ModuleType("fastapi")

    class DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, _name):
            def decorator(*args, **kwargs):
                def wrap(function):
                    return function
                return wrap
            return decorator

    class DummyHTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_stub.FastAPI = DummyFastAPI
    fastapi_stub.File = lambda *args, **kwargs: None
    fastapi_stub.Form = lambda default=None, *args, **kwargs: default
    fastapi_stub.HTTPException = DummyHTTPException
    fastapi_stub.UploadFile = object
    sys.modules["fastapi"] = fastapi_stub

try:
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = object
    sys.modules["pydantic"] = pydantic_stub

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.AsyncClient = object
    httpx_stub.HTTPError = Exception
    sys.modules["httpx"] = httpx_stub


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "nuc_qwen3_asr_1p7b_proxy.py"
SPEC = importlib.util.spec_from_file_location("nuc_qwen_proxy_for_tests", MODULE_PATH)
assert SPEC and SPEC.loader
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def wav_bytes(seconds: float, amplitude: int = 1000, sample_rate: int = 1000) -> bytes:
    frame_count = int(seconds * sample_rate)
    sample = int(amplitude).to_bytes(2, "little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(sample * frame_count)
    return buffer.getvalue()


class QwenProxyChunkTests(unittest.TestCase):
    def test_iter_chunks_adds_context_without_changing_nominal_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(wav_bytes(65))
            chunks = list(proxy._iter_wav_chunks(path, 30.0, 2.0))

        self.assertEqual(3, len(chunks))
        self.assertEqual(0.0, chunks[0]["offset_seconds"])
        self.assertEqual(32.0, chunks[0]["request_duration_seconds"])
        self.assertEqual(28.0, chunks[1]["request_offset_seconds"])
        self.assertEqual(34.0, chunks[1]["request_duration_seconds"])
        self.assertEqual(58.0, chunks[2]["request_offset_seconds"])
        self.assertEqual(7.0, chunks[2]["request_duration_seconds"])

    def test_merge_overlap_tolerates_one_character_difference(self) -> None:
        merged, overlap, errors = proxy._merge_overlapping_text(
            "前文这是一个跨越切块边界的完整句子",
            "这是一个跨越切块边界得完整句子后文",
        )

        self.assertGreaterEqual(overlap, 12)
        self.assertEqual(1, errors)
        self.assertEqual("前文这是一个跨越切块边界的完整句子后文", merged)

    def test_merge_overlap_tolerates_missing_character(self) -> None:
        merged, overlap, errors = proxy._merge_overlapping_text(
            "前文这是一个跨越切块边界的完整句子",
            "这是一个跨越切块边界完整句子后文",
        )

        self.assertGreaterEqual(overlap, 12)
        self.assertEqual(1, errors)
        self.assertEqual("前文这是一个跨越切块边界的完整句子后文", merged)

    def test_silence_dbfs_is_negative_infinity(self) -> None:
        self.assertEqual(float("-inf"), proxy._wav_bytes_dbfs(wav_bytes(1, amplitude=0)))


class QwenProxyRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_silent_empty_chunk_is_not_retried(self) -> None:
        with mock.patch.object(
            proxy,
            "_post_upstream_bytes",
            new=mock.AsyncMock(return_value={"text": ""}),
        ) as post:
            text, reason, diagnostics = await proxy._transcribe_chunk_with_empty_retry(
                audio_bytes=wav_bytes(2, amplitude=0),
                filename="silent.wav",
                language="zh",
            )

        self.assertEqual("", text)
        self.assertIsNone(reason)
        self.assertFalse(diagnostics["empty_retry"]["attempted"])
        self.assertEqual(1, post.await_count)

    async def test_non_silent_empty_chunk_retries_two_halves(self) -> None:
        responses = [
            {"text": ""},
            {"text": "这是前半段共同内容"},
            {"text": "共同内容以及后半段"},
        ]
        with mock.patch.object(
            proxy,
            "_post_upstream_bytes",
            new=mock.AsyncMock(side_effect=responses),
        ) as post:
            text, reason, diagnostics = await proxy._transcribe_chunk_with_empty_retry(
                audio_bytes=wav_bytes(4, amplitude=2000),
                filename="speech.wav",
                language="zh",
            )

        self.assertEqual("这是前半段共同内容以及后半段", text)
        self.assertIsNone(reason)
        self.assertTrue(diagnostics["empty_retry"]["attempted"])
        self.assertEqual(2, diagnostics["empty_retry"]["part_count"])
        self.assertEqual(3, post.await_count)


if __name__ == "__main__":
    unittest.main()
