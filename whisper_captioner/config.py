from __future__ import annotations

from pathlib import Path


HOME = Path.home()
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = HOME / "whisper-models"
OUTPUT_DIR = HOME / "Movies" / "WhisperCaptioner"
CACHE_DIR = OUTPUT_DIR / "cache"
REALTIME_DIR = OUTPUT_DIR / "realtime"
LOG_DIR = OUTPUT_DIR / "logs"
NOTES_DIR = OUTPUT_DIR / "notes"
QWEN_CHAT_DIR = OUTPUT_DIR / "qwen-chat"
HALLUCINATION_BLOCKLIST_PATH = OUTPUT_DIR / "hallucination_blocklist.txt"

REALTIME_CHUNK_SECONDS = 3.0
REALTIME_POLISH_BATCH_SECONDS = 30.0

WHISPER_STREAM = "/opt/homebrew/bin/whisper-stream"
WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
MLX_WHISPER = "/opt/anaconda3/envs/pyside6/bin/mlx_whisper"
MLX_AUDIO_STT = "/opt/anaconda3/envs/pyside6/bin/mlx_audio.stt.generate"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
YT_DLP = "/opt/homebrew/bin/yt-dlp"
FFPROBE = "/opt/homebrew/bin/ffprobe"
SENSE_VOICE_CPP_MAIN = str(PROJECT_ROOT / "third_party/SenseVoice.cpp/build/bin/sense-voice-main")
SENSE_VOICE_CPP_FP16_MODEL = str(PROJECT_ROOT / "third_party/SenseVoice.cpp/models/sense-voice-gguf/sense-voice-small-fp16.gguf")

RAPIDMLX_PYTHON = "/opt/anaconda3/envs/rapidmlx/bin/python"
RAPIDMLX_BIN = "/opt/anaconda3/envs/rapidmlx/bin/rapid-mlx"
RAPIDMLX_HOST = "127.0.0.1"
RAPIDMLX_PORT = "8765"
RAPIDMLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
RAPIDMLX_SERVED_MODEL = "qwen2.5-3b-mlx"
RAPIDMLX_8B_PORT = "8766"
RAPIDMLX_8B_MODEL = "mlx-community/Qwen3-8B-4bit"
RAPIDMLX_8B_SERVED_MODEL = "qwen3-8b-mlx"

NUC_OLLAMA_HOST = "192.168.31.196"
NUC_OLLAMA_PORT = "11434"
NUC_ASR_PORT = "8000"

MLX_TERMS_SCRIPT = Path(__file__).with_name("mlx_terms.py")

BUFFER_PAUSE_MARGIN = 3.0
BUFFER_RESUME_MARGIN = 10.0
DEFAULT_SUBTITLE_OFFSET = 0.0
SENSE_VOICE_CHUNK_OVERLAP_SECONDS = 1.0
SUBTITLE_PIPELINE_VERSION = "gemini-full-document-v1"
MLX_WHISPER_Q5_MODEL = "mlx-community/whisper-large-v3-turbo-asr-5bit"
MLX_WHISPER_FP16_MODEL = "mlx-community/whisper-large-v3-turbo"
SENSEVOICE_SMALL_MLX_MODEL = "mlx-community/SenseVoiceSmall"
QWEN3_ASR_06B_4BIT_MLX_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"
QWEN3_ASR_17B_8BIT_MLX_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
