from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import (
    MLX_WHISPER_FP16_MODEL,
    MLX_WHISPER_Q5_MODEL,
    MODELS_DIR,
    QWEN3_ASR_06B_4BIT_MLX_MODEL,
    QWEN3_ASR_17B_8BIT_MLX_MODEL,
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
        return not isinstance(self.model, Path) or self.model.exists()


@dataclass(frozen=True)
class LLMProvider:
    key: str
    label: str
    api_url: str
    model_id: str
    format: str
    requires_api_key: bool = True


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


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
        "qwen3_asr_17b_8bit_mlx",
        "Qwen3-ASR 1.7B 8bit（高质量）",
        QWEN3_ASR_17B_8BIT_MLX_MODEL,
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
                "https://api.minimaxi.com/anthropic/v1/messages", "MiniMax-M2.7", "anthropic"),
    LLMProvider("claude_sonnet", "Claude Sonnet 4",
                "https://api.anthropic.com/v1/messages", "claude-sonnet-4-20250514", "anthropic"),
    LLMProvider("custom", "Custom (OpenAI-compatible)", "", "", "openai"),
]
