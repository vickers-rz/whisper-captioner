"""
全局配置与常量模块

定义 Whisper Captioner 项目的路径、环境变量读取和所有的默认常量。
主要职责包括：
1. 设置并读取输出目录、缓存目录和资源目录。
2. 配置外部依赖的路径，如 ffmpeg, yt-dlp, whisper.cpp, mlx_whisper。
3. 集中管理 NUC 服务器相关环境变量和各个端点服务的默认端口。
"""
from __future__ import annotations

import os
from pathlib import Path


HOME = Path.home()
PROJECT_ROOT = Path(__file__).parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser()


def _first_existing_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


DEFAULT_T7_MOVIES_DIR = Path("/Volumes/T7_APFS/MacBackup/Movies")
LOCAL_MODELS_DIR = _env_path(
    "WHISPER_CAPTIONER_LOCAL_MODELS_DIR",
    DEFAULT_T7_MOVIES_DIR / "whisper-captioner_APP_Resource" / "local-models",
)

OUTPUT_DIR = _env_path(
    "WHISPER_CAPTIONER_OUTPUT_DIR",
    DEFAULT_T7_MOVIES_DIR / "WhisperCaptioner",
)
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
GENERATED_DIR = ARTIFACTS_DIR / "generated"
APP_RESOURCE_DIR = _env_path(
    "WHISPER_CAPTIONER_RESOURCE_DIR",
    DEFAULT_T7_MOVIES_DIR / "whisper-captioner_APP_Resource",
)
MODELS_DIR = APP_RESOURCE_DIR / "whisper-models"
THIRD_PARTY_DIR = APP_RESOURCE_DIR / "third_party"
HF_CACHE_DIR = APP_RESOURCE_DIR / "huggingface-cache"
CACHE_DIR = OUTPUT_DIR / "cache"
REALTIME_DIR = OUTPUT_DIR / "realtime"
LOG_DIR = ARTIFACTS_DIR / "logs"
NOTES_DIR = ARTIFACTS_DIR / "notes"
QWEN_CHAT_DIR = OUTPUT_DIR / "qwen-chat"
HALLUCINATION_BLOCKLIST_PATH = OUTPUT_DIR / "hallucination_blocklist.txt"

REALTIME_CHUNK_SECONDS = 3.0
REALTIME_POLISH_BATCH_SECONDS = 30.0

WHISPER_STREAM = "/opt/homebrew/bin/whisper-stream"
WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
PYSIDE6_ENV_DIR = _env_path(
    "WHISPER_CAPTIONER_PYSIDE6_ENV_DIR",
    HOME / "miniforge3" / "envs" / "whishperapp_pyside6",
)
MLX_WHISPER = str(PYSIDE6_ENV_DIR / "bin" / "mlx_whisper")
MLX_AUDIO_STT = str(PYSIDE6_ENV_DIR / "bin" / "mlx_audio.stt.generate")
FFMPEG = "/opt/homebrew/bin/ffmpeg"
YT_DLP = "/opt/homebrew/bin/yt-dlp"
FFPROBE = "/opt/homebrew/bin/ffprobe"
SENSE_VOICE_CPP_DIR = _env_path(
    "WHISPER_CAPTIONER_SENSEVOICE_DIR",
    LOCAL_MODELS_DIR / "SenseVoice.cpp",
)
T7_SENSE_VOICE_CPP_DIR = THIRD_PARTY_DIR / "SenseVoice.cpp"
SENSE_VOICE_CPP_MAIN_PATH = _first_existing_path(
    SENSE_VOICE_CPP_DIR / "build/bin/sense-voice-main",
    T7_SENSE_VOICE_CPP_DIR / "build/bin/sense-voice-main",
)
SENSE_VOICE_CPP_FP16_MODEL_PATH = _first_existing_path(
    SENSE_VOICE_CPP_DIR / "models/sense-voice-gguf/sense-voice-small-fp16.gguf",
    T7_SENSE_VOICE_CPP_DIR / "models/sense-voice-gguf/sense-voice-small-fp16.gguf",
)
SENSE_VOICE_CPP_MAIN = str(SENSE_VOICE_CPP_MAIN_PATH)
SENSE_VOICE_CPP_FP16_MODEL = str(SENSE_VOICE_CPP_FP16_MODEL_PATH)

RAPIDMLX_PYTHON = str(HOME / "miniforge3/envs/rapidmlx/bin/python")
RAPIDMLX_BIN = str(HOME / "miniforge3/envs/rapidmlx/bin/rapid-mlx")
RAPIDMLX_HOST = "127.0.0.1"
RAPIDMLX_PORT = "8765"
RAPIDMLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
RAPIDMLX_SERVED_MODEL = "qwen2.5-3b-mlx"
RAPIDMLX_8B_PORT = "8766"
RAPIDMLX_8B_MODEL = "mlx-community/Qwen3-8B-4bit"
RAPIDMLX_8B_SERVED_MODEL = "qwen3-8b-mlx"

NUC_OLLAMA_HOST = "192.168.31.196"
NUC_OLLAMA_PORT = "11434"
NUC_MAC_ADDRESS = os.environ.get("WHISPER_CAPTIONER_NUC_MAC", "1c:69:7a:d9:49:23")
NUC_ASR_PORT = "8000"
NUC_QWEN3_ASR_PORT = "8001"
NUC_SSH_USER = os.environ.get("WHISPER_CAPTIONER_NUC_SSH_USER", "jack")
NUC_SSH_PORT = os.environ.get("WHISPER_CAPTIONER_NUC_SSH_PORT", "22")
NUC_REMOTE_ASR_ROOT = os.environ.get("WHISPER_CAPTIONER_NUC_ASR_ROOT", "/srv/qwen3-asr-1p7b")
LOCAL_AUDIO_CACHE_DIR = CACHE_DIR / "local-audio"
ASR_HISTORY_PATH = CACHE_DIR / "asr-history.json"

MLX_TERMS_SCRIPT = Path(__file__).with_name("mlx_terms.py")

BUFFER_PAUSE_MARGIN = 3.0
BUFFER_RESUME_MARGIN = 10.0
DEFAULT_SUBTITLE_OFFSET = 0.0
SENSE_VOICE_CHUNK_OVERLAP_SECONDS = 1.0
SUBTITLE_PIPELINE_VERSION = "gemini-whisper-fusion-v3"
MLX_WHISPER_Q5_MODEL = "mlx-community/whisper-large-v3-turbo-asr-5bit"
MLX_WHISPER_FP16_MODEL = "mlx-community/whisper-large-v3-turbo"
SENSEVOICE_SMALL_MLX_MODEL = "mlx-community/SenseVoiceSmall"
QWEN3_ASR_06B_4BIT_MLX_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"
QWEN3_ASR_1P7B_8BIT_MLX_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"


def apply_resource_environment() -> None:
    APP_RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
