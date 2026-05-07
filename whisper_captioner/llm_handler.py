"""LLM integration for subtitle proofreading and fusion with reference subtitles."""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from whisper_captioner.config import (
    RAPIDMLX_8B_MODEL,
    RAPIDMLX_8B_PORT,
    RAPIDMLX_8B_SERVED_MODEL,
    RAPIDMLX_BIN,
    RAPIDMLX_HOST,
    RAPIDMLX_MODEL,
    RAPIDMLX_PORT,
    RAPIDMLX_SERVED_MODEL,
)
from whisper_captioner.models import LLMProvider, SubtitleSegment
from whisper_captioner.subtitle_io import overlapping_segments

LLM_SYSTEM_PROMPT = (
    "You are a Chinese subtitle corrector. Your task is to proofread and improve subtitle text "
    "while preserving original meaning. The subtitle index and the corrected text should be in format: "
    "\"序号: 修正后的文本\". Only output corrected lines, no explanations."
)

LLM_FUSION_PROMPT = (
    "你是一个中英文字幕融合专家。你的任务是根据 Whisper 识别的字幕(Whisper)、原始视频字幕(Native)和术语表，"
    "融合出最佳的中文字幕。规则如下：\n"
    "1. 最重要的是输出格式：\"序号: 融合后的字幕文本\"，每行一个，不要添加其他内容。\n"
    "2. 优先保留与音频语义一致、自然、简洁的中文表达。\n"
    "3. 使用视频自带字幕纠正 Whisper 的同音字、专名、术语、英文名、漏词和断句。\n"
    "4. 若视频自带字幕是英文，可将其作为语义参考，不要生硬直译；输出仍以中文为主，除非原句本身就是英文。\n"
    "5. 如果提供术语表，优先保留术语表中的拼写、大小写和符号。\n"
    "6. 如果 Native 明显修正了 Whisper 的错字、术语或专名，应采用 Native 的对应表达。\n"
    "7. 输出格式只能是 \"序号: 融合后的字幕文本\"，不得输出 Whisper、Native、Final 等字段名。\n"
    "8. 不要添加两边都没有的信息，不要输出解释。\n"
)

_LLM_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")
_RAPIDMLX_SERVER_PROCS: dict[str, subprocess.Popen[bytes]] = {}


def _parse_llm_lines(reply: str, expected_count: int) -> dict[int, str]:
    """Parse LLM response into numbered lines."""
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


def _llm_request(api_url: str, body: dict, headers: dict, timeout: int = 15) -> str:
    """Send a JSON POST and return the response body as string."""
    req = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _rapidmlx_models_url(port: str) -> str:
    """Construct Rapid-MLX models endpoint URL."""
    return f"http://{RAPIDMLX_HOST}:{port}/v1/models"


def _rapidmlx_is_ready(port: str, timeout: float = 0.8) -> bool:
    """Check if Rapid-MLX server is ready to accept requests."""
    try:
        with urllib.request.urlopen(_rapidmlx_models_url(port), timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def ensure_local_rapidmlx_server(
    model: str = RAPIDMLX_MODEL,
    served_model: str = RAPIDMLX_SERVED_MODEL,
    port: str = RAPIDMLX_PORT,
    timeout: float = 60.0,
) -> None:
    """Start the local Rapid-MLX OpenAI-compatible server if it is not running."""
    if _rapidmlx_is_ready(port):
        return
    if not Path(RAPIDMLX_BIN).exists():
        raise RuntimeError(f"Rapid-MLX binary not found: {RAPIDMLX_BIN}")
    proc = _RAPIDMLX_SERVER_PROCS.get(port)
    if proc is None or proc.poll() is not None:
        _RAPIDMLX_SERVER_PROCS[port] = subprocess.Popen(
            [
                RAPIDMLX_BIN,
                "serve",
                model,
                "--host",
                RAPIDMLX_HOST,
                "--port",
                port,
                "--log-level",
                "ERROR",
                "--no-thinking",
                "--default-temperature",
                "0.1",
                "--served-model-name",
                served_model,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _rapidmlx_is_ready(port, timeout=1.0):
            return
        time.sleep(0.5)
    raise RuntimeError("Rapid-MLX server did not become ready in time")


def _build_llm_call(
    provider: LLMProvider,
    api_key: str,
    user_text: str,
    api_url_override: str = "",
    model_id_override: str = "",
    max_tokens: int = 4096,
    system_prompt: str = LLM_SYSTEM_PROMPT,
) -> tuple[str, dict, dict]:
    """Return (url, body, headers) for an LLM API call."""
    if provider.key == "local_rapidmlx":
        ensure_local_rapidmlx_server()
    elif provider.key == "local_rapidmlx_8b":
        ensure_local_rapidmlx_server(RAPIDMLX_8B_MODEL, RAPIDMLX_8B_SERVED_MODEL, RAPIDMLX_8B_PORT)
    url = api_url_override or provider.api_url
    model = model_id_override or provider.model_id
    fmt = provider.format

    if fmt == "anthropic":
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_text}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        if provider.key.startswith("local_rapidmlx"):
            body["no_thinking"] = True
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return url, body, headers


def llm_provider_ready(provider: LLMProvider, api_key: str) -> bool:
    """Check if LLM provider is configured and ready."""
    return bool(provider and (api_key or not provider.requires_api_key))


def _extract_llm_reply(data: dict, fmt: str) -> str:
    """Extract reply text from LLM response based on format."""
    if fmt == "anthropic":
        return data["content"][0]["text"]
    return data["choices"][0]["message"]["content"]


def llm_proofread(
    segments: list[SubtitleSegment],
    provider: LLMProvider,
    api_key: str,
    api_url_override: str = "",
    model_id_override: str = "",
    timeout: int = 15,
    max_tokens: int = 4096,
) -> list[SubtitleSegment]:
    """Send segment texts to an LLM for proofreading. Returns corrected segments."""
    if not segments:
        return segments
    lines = [f"{i + 1}: {s.text}" for i, s in enumerate(segments)]
    user_text = "\n".join(lines)

    url, body, headers = _build_llm_call(
        provider, api_key, user_text, api_url_override, model_id_override, max_tokens=max_tokens,
    )
    raw = _llm_request(url, body, headers, timeout=timeout)
    data = json.loads(raw)
    reply = _extract_llm_reply(data, provider.format)

    corrected = _parse_llm_lines(reply, len(segments))

    return [
        SubtitleSegment(s.start, s.end, corrected.get(i, s.text))
        for i, s in enumerate(segments)
    ]


def llm_generate_text(
    user_text: str,
    provider: LLMProvider,
    api_key: str,
    api_url_override: str = "",
    model_id_override: str = "",
    system_prompt: str = "你是一个严谨的视频内容分析助手。",
    timeout: int = 180,
    max_tokens: int = 16000,
) -> str:
    """Generate free-form text from a transcript or prompt."""
    url, body, headers = _build_llm_call(
        provider,
        api_key,
        user_text,
        api_url_override,
        model_id_override,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    raw = _llm_request(url, body, headers, timeout=timeout)
    data = json.loads(raw)
    return _extract_llm_reply(data, provider.format).strip()


def llm_fuse_with_reference(
    whisper_segments: list[SubtitleSegment],
    reference_segments: list[SubtitleSegment],
    provider: LLMProvider,
    api_key: str,
    api_url_override: str = "",
    model_id_override: str = "",
    timeout: int = 20,
    terms: Optional[list[str]] = None,
) -> list[SubtitleSegment]:
    """Fuse whisper segments with reference subtitles using LLM."""
    if not whisper_segments or not reference_segments:
        return whisper_segments
    lines = []
    for i, segment in enumerate(whisper_segments):
        refs = overlapping_segments(reference_segments, segment.start, segment.end)
        ref_text = " / ".join(s.text for s in refs[:4]) if refs else "(无匹配原字幕)"
        lines.append(
            f"[{i + 1}]\n"
            f"Whisper: {segment.text}\n"
            f"Native: {ref_text}\n"
            f"Output: {i + 1}: "
        )
    term_text = "、".join(terms or []) or "无"
    user_text = "术语表：" + term_text + "\n\n" + "\n".join(lines)
    url, body, headers = _build_llm_call(
        provider,
        api_key,
        user_text,
        api_url_override,
        model_id_override,
        system_prompt=LLM_FUSION_PROMPT,
    )
    raw = _llm_request(url, body, headers, timeout=timeout)
    data = json.loads(raw)
    reply = _extract_llm_reply(data, provider.format)

    corrected = _parse_llm_lines(reply, len(whisper_segments))
    return [
        SubtitleSegment(s.start, s.end, corrected.get(i, s.text))
        for i, s in enumerate(whisper_segments)
    ]


def test_llm_connection(
    provider: LLMProvider,
    api_key: str,
    api_url_override: str = "",
    model_id_override: str = "",
) -> tuple[bool, str]:
    """Send a minimal request to verify API key and model."""
    try:
        url, body, headers = _build_llm_call(
            provider, api_key, "Hello", api_url_override, model_id_override,
            max_tokens=10,
        )
        raw = _llm_request(url, body, headers, timeout=15)
        json.loads(raw)
        return True, "Connection successful ✓"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return False, str(exc)
