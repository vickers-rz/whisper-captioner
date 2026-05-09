# Whisper Captioner

Local macOS caption helper for Loopback audio and controlled web-video subtitle playback.

## Run

```bash
conda run -n pyside6 python /Users/vickers/whisper-captioner/whisper_captioner/app.py
```

Or:

```bash
bash /Users/vickers/whisper-captioner/run.sh
```

## Current Architecture

The project has been split out of the original single-file prototype:

- `whisper_captioner/app.py`: Qt main window, tray menu, high-level orchestration, and signal wiring.
- `whisper_captioner/config.py`: paths, model names, tool paths, pipeline version.
- `whisper_captioner/models.py`: shared dataclasses and configured caption/LLM modes.
- `whisper_captioner/subtitle_io.py`: SRT/VTT parsing, JSON segment cache, SRT/TXT export.
- `whisper_captioner/cache.py`: canonical media URL handling, Bilibili page-aware cache keys, and URL validation.
- `whisper_captioner/chrome_control.py`: Chrome AppleScript video control helpers. Polling helpers read video time without activating Chrome; playback helpers can intentionally activate the target tab.
- `whisper_captioner/overlay.py`: floating subtitle overlay, pin button, drag/resize, font, opacity, and playback controls.
- `whisper_captioner/workers.py`: realtime loopback capture (`RealtimeWorker`, `NUCRealtimeWorker`), realtime session polishing/re-recognition workers, controlled URL subtitle processing, local queue workers, and hallucination phrase filtering backed by an external blocklist file.
- `whisper_captioner/llm_handler.py`: Native Ollama API, Gemini, OpenAI-compatible, Anthropic, and Rapid-MLX subtitle proofreading calls.
- `whisper_captioner/mlx_terms.py`: local Rapid-MLX/MLX term extraction helper, currently not part of the main Gemini full-document pipeline.
- `whisper_captioner/qwen_chat_service.py`: local subtitle post-processing web workspace for Qwen3-8B / Gemini chat, manual subtitle upload, and second-stage-only cleanup or article rewriting.

## Current Stability Notes

Recent maintenance focused on the controlled URL path and smaller-screen usability:

- The main window now opens at a smaller default size and uses a scrollable central layout, so the full GUI remains reachable on a 24-inch 1080p secondary display.
- Controlled URL playback no longer exits the app when manually provided `zh.*` subtitles are found. The app loads those subtitle segments, refreshes the transcript list, and starts controlled playback without running Whisper or LLM work.
- Qwen3-ASR pseudo timestamping is shared by local queue processing and controlled URL processing, avoiding a previous controlled-mode crash when the Qwen3-ASR backend was selected.
- Controlled subtitle lookup uses a cached current index plus a cached subtitle-start index for `bisect` fallback, avoiding a full subtitle scan every 250 ms on long videos.
- Controlled SenseVoice.cpp chunking now calls the correct `RollingPrefetchWorker` command helpers.

## Modes

- `NUC faster-whisper large-v3（远程 CUDA）`: highest-performance remote inference via NUC RTX 3080 Ti.
- `实时字幕 NUC large-v3（远程 CUDA，3s延迟）`: low-latency realtime mode offloaded to the NUC, with full session persistence.
- `实时字幕 whisper.cpp small（SoundSource/Loopback）`: lowest-latency local realtime mode for Loopback-routed Chrome or local player audio.
- `实时字幕 whisper.cpp q5_0（large-v3-turbo）`: higher-quality local realtime mode when you can tolerate a bit more latency.
- `Qwen3-ASR 0.6B 4bit（默认）`: recommended MLX-Audio workflow for transcript-first transcription plus light normalization.
- `Qwen3-ASR 1.7B 8bit（高质量）`: higher-quality MLX-Audio workflow when you want better transcript polish.
- `MLX-Audio 5bit（whisper-large-v3-turbo-asr-5bit）`: experimental MLX-Audio backend for Whisper.
- `SenseVoice-Small-mlx`: experimental MLX-Audio mode backed by `mlx-community/SenseVoiceSmall`.
- `SenseVoice.cpp FP16`: local GGUF/Metal backend powered by SenseVoice.cpp. Fast and fluent, but chunk-boundary handling is still under tuning for long files.
- `MLX Whisper FP16（whisper-large-v3-turbo）`: higher-precision MLX fallback using `mlx-whisper`.
- `whisper.cpp q5_0（large-v3-turbo）`: current default controlled URL backend and strongest subtitle-style baseline.
- `whisper.cpp small`: fastest local whisper.cpp batch mode.
- `whisper.cpp 高精度 q5_0（large-v3）`: slower, higher-accuracy whisper.cpp mode for batch work.

## Controlled Playback Pipeline

For web videos where the URL can be processed by `yt-dlp`, use `Controlled URL captions`.

Current flow:

1. Canonicalize the media URL, for example Bilibili URLs become `https://www.bilibili.com/video/<BV...>` and preserve `?p=` for multi-part videos.
2. Check for manually provided `zh.*` built-in subtitles. If present, load those subtitles directly, skip local Whisper and LLM work, and start controlled playback. Auto subtitles are not treated as a safe early-exit source.
3. Pause Chrome and look for a final subtitle cache matching the current pipeline signature.
4. If no valid final cache exists, download audio with `yt-dlp`.
5. Split audio into 30-second chunks with `ffmpeg`.
6. Transcribe each chunk with the currently selected backend and shift chunk-local timestamps onto the full-video timeline.
7. Run Gemini 2.5 Flash once over the full transcript when enabled.
8. Save `final-subtitles-current.json` plus exported `.srt` and `.txt`.
9. Start Chrome at 0 seconds and render subtitles by polling the controlled video's `currentTime` without repeatedly stealing focus from the user.

Cache identity uses the canonical media URL, Whisper model, chunk duration, LLM provider/model, and `SUBTITLE_PIPELINE_VERSION`.
The cache key also includes the Whisper backend, so `mlx-audio`, `mlx-whisper`, and `whisper.cpp` outputs do not overwrite each other.

Known cache follow-ups:

- `b23.tv` short links are not yet expanded before cache-key generation.
- Native subtitle caches are still plain segment JSON files and do not have their own metadata/signature.

Post-processing outputs are stored next to the current video's cache:

- `video-summary-analysis.md`: video summary, structure, argument analysis, keywords, and one-line conclusion.
- `video-article.md`: a polished long-form article rewritten from the transcript.

## Subtitle Post-Processing Workspace

The local Qwen web entry has been expanded into a subtitle post-processing workspace.

What it can do:

- Upload third-party `.srt`, `.vtt`, or `.txt` files without running the transcription pipeline.
- Scan previously generated subtitle files under `~/Movies/WhisperCaptioner` and show them in a history sidebar.
- Attach any uploaded or historical subtitle file to a work conversation.
- Chat with the LLM grounded on the attached subtitle content.
- Run one-click `语句规整` or `转写成文稿` actions on the attached subtitle.
- Switch between `NUC Ollama Qwen3-14B`, `Local Rapid-MLX Qwen3-8B`, `Gemini 2.5 Flash`, and `Gemini 2.5 Pro`.

Notes:

- Long subtitle payloads now show an explicit warning when they are likely beyond the more comfortable single-shot range for local Qwen3-8B.
- When that warning appears, switch to `Gemini 2.5 Pro` if you have already configured its API key in the desktop app Settings pane.
- Action exports are written under `~/Movies/WhisperCaptioner/qwen-chat/exports/`.

Sync controls:

- `Sub -0.5s` / `Sub +0.5s`: adjust and persist subtitle offset for the current canonical video cache.
- `Sync line`: align the currently displayed subtitle line to the controlled Chrome video's current time.

## Hallucination Blocklist

To suppress recurring obviously unrelated subtitle hallucinations, the app filters a small set of known bad phrases before they are written into raw caches or exported subtitles.

- Built-in defaults cover several recurring phrases already observed in this project, such as `优优独播剧场——YoYo Television Series Exclusive`.
- You can extend the filter without editing Python code by appending phrases to:
  `~/Movies/WhisperCaptioner/hallucination_blocklist.txt`
- Add one phrase per line.
- Lines starting with `#` are treated as comments.

This blocklist is intentionally a conservative text-layer safeguard. It does not change model decoding parameters; it only removes exact recurring garbage phrases after transcription and before cache/export persistence.

## Runtime Artifacts

Some third-party speech tools may emit transient sidecar files such as `fbank_lfr_cmvn_feature.json`.

- These runtime artifacts are not treated as source files.
- The repository ignores known generated artifacts.
- Selected subprocesses are launched with `cwd=~/Movies/WhisperCaptioner` so those files land in the output area instead of polluting the source tree.

## Local Benchmark Notes

On this Apple M2 Mac mini:

- `large-v3-turbo-q5_0`: benchmark total about 7.6s.
- `large-v3-q5_0`: benchmark total about 16.1s.

Recommended defaults:

- Default subtitles: `whisper.cpp large-v3-turbo-q5_0`.
- Non-live video where delay is acceptable: `Controlled URL captions`.
- Maximum accuracy batch work: `Delayed max accuracy (large-v3 q5_0)`.

Backend benchmark note:

- On a 30-second sample, `whisper.cpp q5_0` averaged about `4.64s`, `mlx-whisper FP16` about `6.42s`, and `mlx-audio q5` about `7.82s`.
- On a 70-second mixed technical sample, `whisper.cpp q5_0` averaged about `8.18s`, `mlx-whisper FP16` about `9.35s`, and `mlx-audio q5` about `10.25s`.
- Because `whisper.cpp q5_0` is faster and has cleaner chunk timestamps on these local samples, it remains the default.

### 30s Comparison

Sample used below:

- `/tmp/sensevoice-test-30s.wav`

Results on this machine:

- `whisper.cpp large-v3-turbo-q5_0`: `4.22s`
- `SenseVoice.cpp FP16`: `5.37s`
- `mlx-community/Qwen3-ASR-0.6B-4bit`: `6.46s`
- `SenseVoice.cpp q8_0`: `17.43s`
- `mlx-community/Qwen3-ASR-1.7B-8bit`: `9.50s`
- `SenseVoiceSmall via mlx-audio`: `20.43s`

### NUC 3080 Ti Remote Inference Performance

For short 5.5s mixed Chinese/English/Tech audio chunks (typical of realtime chunk size):

- `NUC faster-whisper large-v3 CUDA`: `0.37s` (RTF `0.067x`) -> 🏆 Fastest, 15x faster than real-time, 100% accurate.
- `Mac M2 SenseVoice.cpp FP16`: `0.46s` (RTF `0.083x`) -> Second fastest, but suffered from homophone errors on tech terms.
- `Mac M2 whisper.cpp turbo-q5_0`: `2.43s` (RTF `0.438x`)

**Conclusion**: The NUC backend completely outclasses local M2 inference in both speed and context-awareness (large-v3).

## Remote NUC Inference

The app natively integrates with an external Intel NUC equipped with an NVIDIA RTX 3080 Ti via LAN (e.g., `192.168.31.196`) to dramatically accelerate inference.

- **NUC ASR**: Powered by `faster-whisper-server` (Docker) exposing an OpenAI-compatible `/v1/audio/transcriptions` endpoint on port `8000`.
- **NUC LLM**: Powered by native `Ollama` exposing `/api/chat` on port `11434`. Models like `qwen3:14b` run 6.5x faster than local Rapid-MLX on the Mac.

To use this, simply select a `nuc_asr` or `nuc_ollama` provider from the app UI dropdowns. The app implements automatic timeouts and fallbacks to ensure local Mac usability if the NUC is offline.

## Realtime Session Persistence & Review

Realtime Loopback capture (especially via NUC) now features a full "Two-Stage" persistence and correction pipeline:

1. **Capture & Chunking**: Audio is captured in 3s chunks and transcribed immediately for low-latency floating captions.
2. **Session Persistence**: Chunks are saved locally to `~/Movies/WhisperCaptioner/realtime/YYYYMMDD-HHMMSS/audio/`.
3. **Review Tab**: A new `实时回顾` (Realtime Review) tab lists all historical sessions, allowing you to view the raw generated segments.
4. **Offline Polish & Re-recognition**: 
   - Use the **"LLM 校对"** (LLM Polish) button to trigger a background job (`RealtimePolishWorker`) that chunks segments into 30s semantic batches and sends them to the configured LLM (e.g., NUC Ollama Qwen3-14B) for high-quality contextual correction.
   - Use the **"重新识别"** (Re-recognize) button to combine all saved 3s audio chunks into one `full-audio.wav` and run a full-context re-transcription against the NUC, fixing edge-case truncation issues.

## Realtime Captions With SoundSource

For live captions from Chrome or a local video player, use SoundSource as the routing console:

```text
Chrome / video player
  -> SoundSource routes the app output to Loopback
  -> Loopback exposes a virtual input device
  -> whisper-stream captures that input device
  -> floating realtime subtitles
```

To hear the video at the same time, route the app to a SoundSource output group or a macOS multi-output device:

```text
Chrome / video player
  -> SoundSource output group
      -> speakers / headphones
      -> Loopback virtual device
```

In the app, select a realtime mode such as `Realtime small`, set the Loopback capture device ID, and click `实时字幕`.
Use `列出音频输入设备` in Settings to inspect AVFoundation device IDs if Loopback is not device `0`.

Current app behavior:

- `实时字幕` will automatically switch to the `实时字幕 whisper.cpp small（SoundSource/Loopback）` mode if you are not already on a realtime mode.
- `Loopback 输入` is the `whisper-stream -c` capture device ID.
- The `?` help buttons next to realtime controls explain routing and capture ID usage directly in the UI.

## Backend Notes

### Qwen3-ASR

- Implemented through `mlx_audio`.
- Best used when you care about transcript readability and light normalization more than subtitle-grade timestamp stability.
- The current app path uses 30-second chunking plus pseudo timestamps:
  each chunk is normalized by Qwen3-ASR, split into sentence-like units, and converted into a usable `.srt` by character-weighted time allocation.
- These timestamps are intentionally approximate rather than forced-alignment timestamps. In practice they work well for transcript-first playback review, journaling, and semantic recall.
- Current default recommendation inside this family is `Qwen3-ASR 0.6B 4bit（默认）`.

### SenseVoice.cpp

- Implemented through the local `SenseVoice.cpp` binary plus GGUF fp16 model.
- Uses Metal on Apple Silicon and performed very well on short samples.
- Local long-file batch processing now uses a chunked pipeline with overlap, but boundary smoothing still needs more tuning before it should replace the default whisper.cpp path for every workload.

## Requirements

- Loopback input device: `Whisper Captions`, visible to AVFoundation as audio device `0`.
- Whisper models in `/Users/vickers/whisper-models`.
- `whisper-stream`, `whisper-cli`, `ffmpeg`, `ffprobe`, and `yt-dlp`.
- Conda env `pyside6` for the GUI.
- Local SenseVoice.cpp checkout and GGUF model under `/Users/vickers/whisper-captioner/third_party/SenseVoice.cpp` if you want the `SenseVoice.cpp FP16` backend.

### SenseVoice.cpp Setup

`third_party/` is intentionally not tracked in this repository.

If you want the `SenseVoice.cpp FP16` backend, clone and build SenseVoice.cpp locally first:

- Upstream project: [lovemefan/SenseVoice.cpp](https://github.com/lovemefan/SenseVoice.cpp)
- Upstream SenseVoice model project: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)

Recommended local path for this project:

```bash
git clone https://github.com/lovemefan/SenseVoice.cpp /Users/vickers/whisper-captioner/third_party/SenseVoice.cpp
cd /Users/vickers/whisper-captioner/third_party/SenseVoice.cpp
cmake -B build
cmake --build build -j
```

You also need a compatible GGUF model under the local SenseVoice.cpp models directory.

## License

This repository uses a source-available non-commercial copyleft license.

Allowed:

- Personal learning and research use
- Reuse in open-source projects with attribution

Required:

- Derivative code must remain source-available
- License and copyright notices must be retained

Not allowed without prior written authorization:

- Closed-source commercial use
- Commercial redistribution

Important notes:

- This is not an OSI open-source license.
- Third-party tools, bundled code, and model files remain under their own separate licenses or terms.
- See `LICENSE` and `THIRD_PARTY_NOTICES.md` for details.
