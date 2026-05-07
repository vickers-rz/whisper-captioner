# Copilot Instructions for Whisper Captioner

## Project Overview

Whisper Captioner is a macOS desktop application for real-time caption generation from audio streams. It integrates with the Loopback audio device and supports multiple transcription backends (whisper-stream, whisper-cli) and LLM providers for subtitle proofreading and fusion with reference captions.

## Run and Test

### Launch the App

```bash
conda run -n pyside6 python /Users/vickers/whisper-captioner/whisper_captioner/app.py
```

Or use the shell script shortcut:
```bash
bash /Users/vickers/whisper-captioner/run.sh
```

### Benchmarks

The `benchmark_local_terms.py` script measures term extraction performance across GGUF and MLX backends:

```bash
cd /Users/vickers/whisper-captioner
conda run -n rapidmlx python benchmark_local_terms.py
```

Results are saved to `benchmark_local_terms_results.json`.

## Architecture

### High-Level Flow

1. **Main Application** (`app.py`)
   - Entry point: `App` class orchestrates the Qt6 UI and worker threads
   - Main window contains tabs for different caption modes and control panels
   - System tray icon provides global keyboard shortcuts and quick access

2. **Caption Modes** (defined as `CaptionMode` dataclass)
   - Real-time transcription via whisper-stream (default: large-v3-turbo-q5_0)
   - Delayed high-quality transcription (large-v3-q5_0 with buffering)
   - URL playback with synced captions (yt-dlp + Chrome control + realtime overlay)
   - Each mode specifies a model file, transcription args, and realtime flag

3. **LLM Proofreading Pipeline**
   - `LLMProvider` dataclass configures available LLM backends (local Rapid-MLX, OpenAI, Claude, Gemini, DeepSeek)
   - Three proofreading operations:
     - **LLM Proofread**: Corrects Whisper recognition errors (homophones, segmentation, proper nouns)
     - **LLM Fuse**: Merges Whisper transcription with reference captions (e.g., video's embedded subtitles)
     - **Extract Terms**: Uses `mlx_terms.py` to identify proper nouns and terminology for custom dictionaries

4. **Subtitle Processing**
   - Segments are stored as `SubtitleSegment` dataclass (start_sec, end_sec, text)
   - Supports SRT, VTT, and plain text export
   - JSON serialization for caching and state persistence
   - Parsing functions: `parse_srt()`, `parse_vtt()`, `parse_subtitle_file()`

5. **Worker Threads**
   - `RealtimeWorker`: Captures Loopback audio stream in chunks, feeds to whisper-stream
   - `DelayedCaptureWorker`: Buffers audio for high-quality transcription
   - `QueueWorker`: Processes batch URLs from a queue with delayed LLM proofreading
   - `RollingPrefetchWorker`: Prepares subtitles from URLs before playback, manages Chrome pause/resume

6. **Chrome Integration**
   - AppleScript-based automation for pause/resume/seek operations
   - Functions: `chrome_pause()`, `chrome_play_from()`, `chrome_resume()`
   - URL extraction and playback control via `run_chrome_script()`

7. **Subtitle Overlay**
   - `SubtitleOverlay` widget renders captions on screen during video playback
   - Docked or floating window; supports font/style customization via context menu
   - Synced to Chrome's `currentTime` during controlled playback

### Key Configuration Constants

All paths and service URLs are defined at the module level in `app.py`:

- **Paths**: `MODELS_DIR`, `OUTPUT_DIR`, `CACHE_DIR`, `WHISPER_STREAM`, `WHISPER_CLI`, `FFMPEG`, `YT_DLP`, `FFPROBE`
- **Local LLM**: `RAPIDMLX_PYTHON`, `RAPIDMLX_BIN`, `RAPIDMLX_HOST`, `RAPIDMLX_PORT`, and separate ports/models for 3B and 8B variants
- **Subtitle Parameters**: `BUFFER_PAUSE_MARGIN`, `BUFFER_RESUME_MARGIN`, `DEFAULT_SUBTITLE_OFFSET`
- **Pipeline Version**: `SUBTITLE_PIPELINE_VERSION` for tracking subtitle processing workflows

### Term Extraction Module

`mlx_terms.py` handles specialized terminology extraction using a local LLM:

- Calls Rapid-MLX OpenAI-compatible API at `/v1/chat/completions`
- Parses JSON output with schema: `{"terms": [{"term": "...", "type": "person|brand|product|model|acronym|term"}]}`
- Normalizes and deduplicates extracted terms
- Used in proofreading workflows to preserve proper nouns and technical terminology

## Key Conventions

### Qt6 Signal/Slot Pattern

All worker threads inherit `QObject` and use PySignal for thread-safe communication:

- Workers define custom signals (e.g., `progress`, `finished`, `error`)
- Main thread connects signals to slots (e.g., `update_display()`)
- Use `QThread` to move workers to background threads; connect before `moveToThread()`
- Always emit signals; do not call UI methods directly from worker threads

### Subprocess Management

Subprocess calls use context managers and return code checks:

- `subprocess.run(..., capture_output=True, text=True)` for synchronous commands
- `subprocess.Popen(..., stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)` for daemon servers (e.g., Rapid-MLX)
- Timeout handling: Use `timeout` parameter in `urllib.request.urlopen()` for API calls
- Process tracking: `_RAPIDMLX_SERVER_PROCS` dict prevents duplicate server starts

### Settings Persistence

`QSettings` (Qt framework) stores user preferences:

- Scope: application name "Whisper Captioner"
- Used implicitly in `MainWindow` for window geometry, selected LLM, etc.
- Accessed via `QSettings()` constructor (auto-loads from OS keychain/defaults)

### Regex and Parsing

- `_LLM_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")` parses numbered subtitle lines from LLM output
- SRT parsing: `_LLM_LINE_RE` matches fallback lines if JSON output fails
- VTT parsing: Handles `00:00:00.000 --> 00:00:10.000` timestamps

### Path Handling

- All paths use `pathlib.Path` for cross-platform compatibility
- Constants like `MODELS_DIR` and `OUTPUT_DIR` resolved once at module load
- Cache slugs: `cache_slug(*parts)` creates safe filenames from mixed inputs

### Error Handling in LLM Requests

- `_llm_request()` uses `urllib.request.urlopen()` with 15-second timeout
- JSON parsing: `_parse_llm_lines()` includes fallback logic (extracts first `N` non-empty lines if structured parsing fails)
- API key validation: `llm_provider_ready()` pings the provider before actual requests
- Server startup: `ensure_local_rapidmlx_server()` blocks up to 60 seconds waiting for readiness

### Data Serialization

- Segments saved as JSON: `segment_to_dict()` → `save_segments()`
- VTT/SRT export: Text-only representation for external subtitle players
- Benchmarks: Results logged to JSON with timing and stderr for debugging

## Important Files

- **`whisper_captioner/app.py`** (2591 lines): Main application, UI, workers, LLM integrations
- **`whisper_captioner/mlx_terms.py`** (125 lines): Term extraction using local LLM
- **`benchmark_local_terms.py`**: Performance comparison of GGUF vs MLX backends
- **`run.sh`**: Convenience script to launch the app via conda environment

## MCP Servers

Two MCP servers enhance Copilot's capabilities in this repository:

1. **Chrome DevTools Protocol (CDP)** — WebSocket-based
   - Connects to Chrome instance running with `--remote-debugging-port=9222`
   - Enables Copilot to: inspect DOM, control video playback, manage tabs, debug web interactions
   - Replaces AppleScript-based Chrome control (`run_chrome_script()`, `chrome_pause()`, etc.)
   - Setup: `.github/copilot-setup-steps.yml` automatically starts Chrome with debugging enabled

2. **LLM Testing** — Stdio-based
   - Tests connections to local Rapid-MLX and cloud LLM providers (OpenAI, Claude, Gemini, DeepSeek)
   - Validates proofreading prompts and term extraction workflows
   - Helps debug API key issues and LLM provider configuration

## Dependencies

The application requires:

- **Python 3.8+** with PySide6 (Qt6 bindings)
- **Conda environments**: `pyside6` (main app), `rapidmlx` (optional, for local LLM)
- **External tools** (on PATH or homebrew):
  - `whisper-stream` (real-time transcription)
  - `whisper-cli` (batch transcription)
  - `ffmpeg`, `ffprobe` (audio/video processing)
  - `yt-dlp` (URL handling)
  - `llama-cli` (GGUF model inference, optional)
  - `rapid-mlx` (local LLM server, optional)
- **Chrome**: Must run with `--remote-debugging-port=9222` for CDP MCP integration
- **Audio device**: Loopback virtual audio device (name "Whisper Captions") expected by `AVFoundation`
- **Models**:
  - Whisper models: `~/whisper-models/ggml-large-v3-turbo-q5_0.bin`, etc.
  - LLM models: Hosted on HuggingFace (Qwen, Llama) or via API

## macOS-Specific Details

- **AppleScript**: Chrome automation via `osascript` subprocess calls
- **Audio I/O**: `AVFoundation` framework (via PySide6) detects Loopback device
- **System Tray**: `QSystemTrayIcon` integrates with macOS menu bar
- **Settings**: Stored in `~/Library/Preferences/com.vickers.WhisperCaptioner.plist` (Qt framework)
- **Output Directory**: `~/Movies/WhisperCaptioner/` for captions and cache

## Prompt Engineering Notes

- **Whisper correction prompts**: Chinese prompts use specific keywords (同音字, 近音字, 专有名词) to guide LLM output format
- **Subtitle fusion prompts**: Two-source merging prioritizes Whisper's timeline accuracy but uses reference captions for terminology and spelling
- **Term extraction schema**: Strict JSON output required; LLM is configured with `temperature=0.1` and Chinese instruction text
- **No-thinking mode**: Rapid-MLX server launched with `--no-thinking` flag to avoid extended reasoning in fast mode

## Common Workflows

- **Real-time captioning**: Default large-v3-turbo mode, Loopback device, system tray shortcuts
- **Video URL batch processing**: Queue tab, yt-dlp parsing, LLM proofreading, synced playback overlay
- **Terminology refinement**: Extract terms tab → MLX term extraction → custom dictionary for next run
- **Caption fusion**: Delayed mode + reference SRT/VTT → LLM fusion → output SRT
