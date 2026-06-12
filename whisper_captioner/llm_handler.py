"""
大语言模型 (LLM) 交互与字幕处理模块

负责与各种 LLM 提供商（OpenAI, Anthropic, Ollama, 兼容 OpenAI 格式的本地服务等）进行通信。
主要职责包括：
1. 构建通用的 API 请求（适配不同厂商的格式）。
2. 提供字幕校对（润色）、基于字幕的自由文本生成（如总结、长文）和参考字幕融合功能。
3. 自动管理和唤醒局域网内的推理节点（如发送 WOL 唤醒 NUC，自动拉起本地 Rapid-MLX 服务）。
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import binascii
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel
    HAS_GOOGLE_GENAI = True

    class RefinedChapter(BaseModel):
        text: str
except ImportError:
    genai = None
    genai_types = None
    BaseModel = object
    HAS_GOOGLE_GENAI = False

from whisper_captioner.config import (
    RAPIDMLX_8B_MODEL,
    RAPIDMLX_8B_PORT,
    RAPIDMLX_8B_SERVED_MODEL,
    RAPIDMLX_BIN,
    RAPIDMLX_HOST,
    RAPIDMLX_MODEL,
    RAPIDMLX_PORT,
    RAPIDMLX_SERVED_MODEL,
    NUC_MAC_ADDRESS,
    NUC_OLLAMA_HOST,
    NUC_OLLAMA_PORT,
    OUTPUT_DIR,
)
from whisper_captioner.models import LLMProvider, SubtitleSegment
from whisper_captioner.subtitle_io import overlapping_segments

LLM_SYSTEM_PROMPT = (
    "你是中文视频字幕的严格校订编辑。你的任务是对字幕进行忠实去噪和规整。\n\n"
    "这是“忠实校订”，不是摘要、改写、提纲或出版式压缩。\n\n"
    "只允许：\n"
    "1. 删除“嗯、啊、呃”等没有语义的语气词。\n"
    "2. 删除结巴造成的连续重复，如“这个这个这个”。\n"
    "3. 删除完全相同且相邻的重复句。\n"
    "4. 修正明显病句、标点、错别字和技术术语。\n\n"
    "不得删除：\n"
    "1. 包含任何新增信息的近义重复。\n"
    "2. 教学或讲解中的强调、解释、因果关系和过渡。\n"
    "3. 例子、数字、代码标识、条件、否定表达和操作步骤。\n"
    "4. 任何不能确定是冗余的内容。\n\n"
    "不得总结、扩写、改变观点、重排论证或添加原文没有的知识。\n"
    "输出长度原则上不得低于输入正文的 80%。不确定是否应删除时，保留原文。\n"
    "输出格式必须为 \"序号: 规整后的文本\"，每行一个，不能包含多余的解释。如果不需要修改，直接输出原句。"
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        if detail:
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
        raise


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
            cwd=str(OUTPUT_DIR),
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _rapidmlx_is_ready(port, timeout=1.0):
            return
        time.sleep(0.5)
    raise RuntimeError("Rapid-MLX server did not become ready in time")


def _strip_think_blocks(text: str) -> str:
    """Remove any residual <think>...</think> blocks from model output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or text


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
    elif fmt == "ollama":
        is_nuc_gemma4 = provider.key == "nuc_ollama_gemma4"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0.1 if is_nuc_gemma4 else 0.3,
                "num_predict": min(max_tokens, 8192) if is_nuc_gemma4 else max_tokens,
            },
        }
        if is_nuc_gemma4:
            body["keep_alive"] = "10m"
            body["options"]["num_ctx"] = 16384
        headers = {"Content-Type": "application/json"}
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


def wake_on_lan_nuc() -> None:
    """Send the NUC magic packet to global and local-subnet broadcasts."""
    mac = NUC_MAC_ADDRESS.replace(":", "").replace("-", "")
    if len(mac) != 12:
        raise ValueError(f"Invalid NUC MAC address: {NUC_MAC_ADDRESS}")
    mac_bytes = binascii.unhexlify(mac)
    magic_packet = b"\xff" * 6 + mac_bytes * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for target in ("255.255.255.255", "192.168.31.255"):
            sock.sendto(magic_packet, (target, 9))


def _nuc_ollama_is_ready(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{NUC_OLLAMA_HOST}:{NUC_OLLAMA_PORT}/api/tags",
            timeout=timeout,
        ) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def ensure_nuc_ollama_ready(timeout: float = 120.0) -> None:
    if _nuc_ollama_is_ready():
        return
    wake_on_lan_nuc()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _nuc_ollama_is_ready(timeout=2.0):
            return
        time.sleep(2.0)
    raise TimeoutError(f"NUC Ollama did not become ready within {timeout:.0f}s after WOL")


def _ensure_provider_ready(provider: LLMProvider) -> None:
    if provider.key.startswith("nuc_ollama"):
        ensure_nuc_ollama_ready()


def _extract_llm_reply(data: dict, fmt: str) -> str:
    """Extract reply text from LLM response based on format."""
    if fmt == "anthropic":
        blocks = data.get("content", [])
        if isinstance(blocks, list):
            text_parts = [
                str(block.get("text", "")).strip()
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip()
            ]
            if text_parts:
                return "\n".join(text_parts)
        return data["content"][0]["text"]
    if fmt == "ollama":
        content = data.get("message", {}).get("content", "")
        return _strip_think_blocks(content)
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
    """
    使用大语言模型对字幕片段进行校对和润色。

    发送带编号的字幕行到 LLM，并解析返回的编号行，替换原始字幕文本。
    
    参数:
        segments: 待校对的字幕片段列表。
        provider: LLM 服务提供商配置。
        api_key: API 密钥。
        api_url_override: 自定义的 API 地址（如果适用）。
        model_id_override: 自定义的模型 ID（如果适用）。
        timeout: API 请求超时时间（秒）。
        max_tokens: 允许生成的最大 token 数量。

    返回:
        校对后的 SubtitleSegment 列表。时间轴与输入保持一致。
    """
    if not segments:
        return segments
    _ensure_provider_ready(provider)
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
    """
    使用大语言模型基于输入文本自由生成内容。

    主要用于全文字幕总结、长文转写或基于字幕的问答。
    
    参数:
        user_text: 用户的输入文本或提示词（通常包含全部字幕）。
        provider: LLM 服务提供商配置。
        api_key: API 密钥。
        api_url_override: 自定义的 API 地址。
        model_id_override: 自定义的模型 ID。
        system_prompt: 系统级提示词，用于设定模型的行为角色。
        timeout: API 请求超时时间（秒）。
        max_tokens: 允许生成的最大 token 数量。

    返回:
        LLM 生成的回复文本。
    """
    _ensure_provider_ready(provider)
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
    """
    使用大语言模型将 Whisper 识别结果与参考字幕进行融合修正。

    这通常用于结合官方英文字幕纠正 Whisper 中文识别的专有名词、术语或断句。
    
    参数:
        whisper_segments: Whisper 识别的基准中文字幕片段。
        reference_segments: 原视频的参考字幕（可能是英文或带有正确术语的其他字幕）。
        provider: LLM 服务提供商配置。
        api_key: API 密钥。
        api_url_override: 自定义的 API 地址。
        model_id_override: 自定义的模型 ID。
        timeout: API 请求超时时间（秒）。
        terms: 可选的专有名词术语表。

    返回:
        融合修正后的 SubtitleSegment 列表。
    """
    if not whisper_segments or not reference_segments:
        return whisper_segments
    _ensure_provider_ready(provider)
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
        _ensure_provider_ready(provider)
        url, body, headers = _build_llm_call(
            provider, api_key, "Hello", api_url_override, model_id_override,
            max_tokens=10,
        )
        timeout = 120 if provider.key == "nuc_ollama_gemma4" else 15
        raw = _llm_request(url, body, headers, timeout=timeout)
        json.loads(raw)
        return True, "Connection successful ✓"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        if provider.key.startswith("nuc_ollama"):
            return False, f"WOL packet sent. Error: {str(exc)}. Please wait a moment and try again."
        return False, str(exc)
