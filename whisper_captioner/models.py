"""
数据模型模块

定义了 Whisper Captioner 项目中使用的核心数据类和常量配置，
包括支持的字幕生成模式 (CaptionMode)、大语言模型提供商 (LLMProvider)
以及字幕片段结构 (SubtitleSegment)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import (
    MLX_WHISPER_FP16_MODEL,
    MLX_WHISPER_Q5_MODEL,
    MODELS_DIR,
    NUC_OLLAMA_HOST,
    NUC_OLLAMA_PORT,
    NUC_ASR_PORT,
    NUC_QWEN3_ASR_PORT,
    QWEN3_ASR_06B_4BIT_MLX_MODEL,
    QWEN3_ASR_1P7B_8BIT_MLX_MODEL,
    RAPIDMLX_HOST,
    RAPIDMLX_PORT,
    RAPIDMLX_8B_PORT,
    RAPIDMLX_SERVED_MODEL,
    RAPIDMLX_8B_SERVED_MODEL,
    SENSEVOICE_SMALL_MLX_MODEL,
    SENSE_VOICE_CPP_FP16_MODEL,
)


@dataclass(frozen=True)
class CaptionMode:
    """
    字幕生成模式配置类
    
    用于定义不同 ASR（自动语音识别）后端及其运行参数。
    包含本地运行模式（如 whisper.cpp, mlx_audio）和远程调用模式（如 NUC 节点）。
    """
    key: str
    label: str
    model: Path | str
    realtime: bool
    args: tuple[str, ...]
    backend: str = "whisper_cpp"

    @property
    def model_name(self) -> str:
        return self.model.name if isinstance(self.model, Path) else self.model

    @property
    def available(self) -> bool:
        """检查模型文件是否在本地存在（对于远程模型始终返回 True）"""
        return not isinstance(self.model, Path) or self.model.exists()


@dataclass(frozen=True)
class LLMProvider:
    """
    大语言模型提供商配置类
    
    定义了可用于文本后处理（如总结、问答、润色）的 LLM 服务节点。
    包含 API 地址、模型 ID 及其所需的调用格式（如 openai 兼容格式）。
    """
    key: str
    label: str
    api_url: str
    model_id: str
    format: str
    requires_api_key: bool = True


@dataclass(frozen=True)
class SubtitleSegment:
    """
    单条字幕片段数据类
    
    表示字幕文件（如 SRT）中的一个最小时间轴单元。
    """
    start: float
    end: float
    text: str


LLM_API_KEY_ENV_VARS = {
    "gpt4o_mini": "OPENAI_API_KEY",
    "gpt4o": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini_flash": "GEMINI_API_KEY",
    "gemini_pro": "GEMINI_API_KEY",
    "minimax_m27": "MINIMAX_API_KEY",
    "claude_sonnet": "ANTHROPIC_API_KEY",
}


def resolved_llm_api_key(provider_key: str, saved_key: str = "") -> str:
    """Return the provider environment key when set, otherwise the saved key."""
    env_name = LLM_API_KEY_ENV_VARS.get(provider_key, "")
    env_key = os.environ.get(env_name, "").strip() if env_name else ""
    return env_key or saved_key.strip()


MODES = [
    CaptionMode(
        "realtime_small",
        "实时字幕 whisper.cpp small（SoundSource/Loopback）",
        MODELS_DIR / "ggml-small.bin",
        True,
        ("-l", "zh", "--step", "2000", "--length", "8000", "--keep", "500"),
    ),
    CaptionMode(
        "realtime_turbo",
        "实时字幕 whisper.cpp q5_0（large-v3-turbo）",
        MODELS_DIR / "ggml-large-v3-turbo-q5_0.bin",
        True,
        ("-l", "zh", "--step", "3000", "--length", "10000", "--keep", "500"),
    ),
    CaptionMode(
        "realtime_nuc",
        "实时字幕 NUC large-v3（远程 CUDA，3s延迟）",
        f"http://{NUC_OLLAMA_HOST}:{NUC_ASR_PORT}",
        True,
        ("-l", "auto"),
        "nuc_asr",
    ),
    CaptionMode(
        "qwen3_asr_06b_4bit_mlx",
        "Qwen3-ASR 0.6B 4bit（默认）",
        QWEN3_ASR_06B_4BIT_MLX_MODEL,
        False,
        ("-l", "zh"),
        "mlx_audio",
    ),
    CaptionMode(
        "mlx_turbo_q5",
        "MLX-Audio 5bit（whisper-large-v3-turbo-asr-5bit）",
        MLX_WHISPER_Q5_MODEL,
        False,
        ("-l", "zh"),
        "mlx_audio",
    ),
    CaptionMode(
        "sensevoice_small_mlx",
        "SenseVoice-Small-mlx",
        SENSEVOICE_SMALL_MLX_MODEL,
        False,
        ("-l", "zh"),
        "mlx_audio",
    ),
    CaptionMode(
        "qwen3_asr_1p7b_8bit_mlx",
        "Qwen3-ASR 1.7B 8bit（高质量）",
        QWEN3_ASR_1P7B_8BIT_MLX_MODEL,
        False,
        ("-l", "zh"),
        "mlx_audio",
    ),
    CaptionMode(
        "sensevoice_cpp_fp16",
        "SenseVoice.cpp FP16",
        Path(SENSE_VOICE_CPP_FP16_MODEL),
        False,
        (),
        "sense_voice_cpp",
    ),
    CaptionMode(
        "mlx_turbo_fp16",
        "MLX Whisper FP16（whisper-large-v3-turbo）",
        MLX_WHISPER_FP16_MODEL,
        False,
        ("-l", "zh"),
        "mlx_whisper",
    ),
    CaptionMode(
        "hq_turbo",
        "whisper.cpp q5_0（large-v3-turbo）",
        MODELS_DIR / "ggml-large-v3-turbo-q5_0.bin",
        False,
        ("-l", "zh", "-osrt", "-otxt"),
    ),
    CaptionMode(
        "small",
        "whisper.cpp small",
        MODELS_DIR / "ggml-small.bin",
        False,
        ("-l", "zh", "-osrt", "-otxt"),
    ),
    CaptionMode(
        "hq_large",
        "whisper.cpp 高精度 q5_0（large-v3）",
        MODELS_DIR / "ggml-large-v3-q5_0.bin",
        False,
        ("-l", "zh", "-osrt", "-otxt"),
    ),
    CaptionMode(
        "nuc_asr_turbo",
        "NUC faster-whisper large-v3-turbo（远程 CUDA，快速）",
        f"http://{NUC_OLLAMA_HOST}:{NUC_ASR_PORT}",
        False,
        ("-l", "auto"),
        "nuc_asr",
    ),
    CaptionMode(
        "nuc_asr",
        "NUC faster-whisper large-v3（远程 CUDA，高质量）",
        f"http://{NUC_OLLAMA_HOST}:{NUC_ASR_PORT}",
        False,
        ("-l", "auto"),
        "nuc_asr",
    ),
    CaptionMode(
        "nuc_qwen3_asr_1p7b",
        "NUC Qwen3-ASR 1.7B（远程高质量离线）",
        f"http://{NUC_OLLAMA_HOST}:{NUC_QWEN3_ASR_PORT}",
        False,
        ("-l", "zh"),
        "nuc_qwen3_asr_1p7b",
    ),
]


LLM_PROVIDERS = [
    LLMProvider(
        "local_rapidmlx_8b",
        "Local Rapid-MLX Qwen3-8B",
        f"http://{RAPIDMLX_HOST}:{RAPIDMLX_8B_PORT}/v1/chat/completions",
        RAPIDMLX_8B_SERVED_MODEL,
        "openai",
        False,
    ),
    LLMProvider(
        "local_rapidmlx",
        "Local Rapid-MLX Qwen2.5-3B",
        f"http://{RAPIDMLX_HOST}:{RAPIDMLX_PORT}/v1/chat/completions",
        RAPIDMLX_SERVED_MODEL,
        "openai",
        False,
    ),
    LLMProvider(
        "local_ollama_qwen35_4b",
        "Local Ollama Qwen3.5-4B",
        "http://127.0.0.1:11434/api/chat",
        "qwen3.5:4b",
        "ollama",
        False,
    ),
    LLMProvider(
        "nuc_ollama_8b",
        "NUC Ollama Qwen3-8B",
        f"http://{NUC_OLLAMA_HOST}:{NUC_OLLAMA_PORT}/api/chat",
        "qwen3:8b",
        "ollama",
        False,
    ),
    LLMProvider(
        "nuc_ollama_14b",
        "NUC Ollama Qwen3-14B-8K",
        f"http://{NUC_OLLAMA_HOST}:{NUC_OLLAMA_PORT}/api/chat",
        "qwen3-14b-8k",
        "ollama",
        False,
    ),
    LLMProvider(
        "nuc_ollama_gemma4",
        "NUC Ollama Gemma 4 E4B（16K）",
        f"http://{NUC_OLLAMA_HOST}:{NUC_OLLAMA_PORT}/api/chat",
        "gemma4:latest",
        "ollama",
        False,
    ),
    LLMProvider("gpt4o_mini", "OpenAI GPT-4o-mini",
                "https://api.openai.com/v1/chat/completions", "gpt-4o-mini", "openai"),
    LLMProvider("gpt4o", "OpenAI GPT-4o",
                "https://api.openai.com/v1/chat/completions", "gpt-4o", "openai"),
    LLMProvider("deepseek", "DeepSeek V3",
                "https://api.deepseek.com/v1/chat/completions", "deepseek-chat", "openai"),
    LLMProvider("gemini_flash", "Gemini 2.5 Flash",
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "gemini-2.5-flash", "openai"),
    LLMProvider("gemini_pro", "Gemini 2.5 Pro",
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                "gemini-2.5-pro", "openai"),
    LLMProvider("minimax_m27", "MiniMAX M2.7",
                "https://api.minimaxi.com/v1/chat/completions", "MiniMax-M2.7", "openai"),
    LLMProvider("claude_sonnet", "Claude Sonnet 4",
                "https://api.anthropic.com/v1/messages", "claude-sonnet-4-20250514", "anthropic"),
    LLMProvider("custom", "Custom (OpenAI-compatible)", "", "", "openai"),
]
