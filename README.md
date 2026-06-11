# Whisper Captioner

Local macOS caption helper for Loopback audio and controlled web-video subtitle playback.

## Run

```bash
conda run -n whishperapp_pyside6 python /Users/vickers/Documents/whisper-captioner/whisper_captioner/app.py
```

Or:

```bash
bash /Users/vickers/Documents/whisper-captioner/run.sh
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

Current architecture notes:

- `app.py`, `workers.py`, and `qwen_chat_service.py` are the main growth hotspots.
- New transcription backend logic should be routed toward a shared transcriber service rather than duplicated between queue and controlled-playback workers.
- New NUC lifecycle or priority decisions should live in the scheduler layer, with proxies kept as thin job/admission clients.
- New subtitle post-processing workspace features should avoid expanding the single-file web service further; split storage, asset indexing, prompts, and HTTP routing first.
- See `ARCHITECTURE_AUDIT_v2026-05-07.md` for the current refactor map and 2026-05-14 architecture review update.

Runtime layout:

- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/`: generated subtitle, transcript, and shared Markdown outputs grouped by source title.
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/logs/`: application logs.
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/notes/`: standalone note exports that are not tied to a generated subtitle folder.
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/`: per-source processing caches and final segment JSON.
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/local-audio/`: extracted `16 kHz / mono / wav` cache for local media files. The same source file is reused across retries until you manually clear it or the source file changes.
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/qwen-chat/`: web workspace uploads, storage, and action exports.
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/realtime/`: persisted realtime session audio and review manifests.
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/whisper-models/`: local Whisper model binaries.
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/third_party/`: local third-party source checkouts and built binaries used by optional backends.
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/huggingface-cache/`: Hugging Face / MLX model cache used by `huggingface_hub`, `mlx-audio`, and related downloads.
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models/SenseVoice.cpp/`: default SenseVoice.cpp FP16 runtime path.

Path overrides:

- `WHISPER_CAPTIONER_OUTPUT_DIR`: runtime output root; defaults to `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner`.
- `WHISPER_CAPTIONER_RESOURCE_DIR`: model/resource root; defaults to `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource`.
- `WHISPER_CAPTIONER_LOCAL_MODELS_DIR`: local model root; defaults to `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models`.
- `WHISPER_CAPTIONER_SENSEVOICE_DIR`: SenseVoice.cpp runtime root; defaults to `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models/SenseVoice.cpp`.

To prepare the local SenseVoice.cpp runtime:

```bash
bash scripts/migrate_local_sensevoice_runtime.sh
```

## Current Stability Notes

Recent maintenance focused on the controlled URL path and smaller-screen usability:

- The main window now opens at a smaller default size and uses a scrollable central layout, so the full GUI remains reachable on a 24-inch 1080p secondary display.
- Controlled URL playback no longer exits the app when manually provided `zh.*` subtitles are found. The app loads those subtitle segments, refreshes the transcript list, and starts controlled playback without running Whisper or LLM work.
- Qwen3-ASR pseudo timestamping is shared by local queue processing and controlled URL processing, avoiding a previous controlled-mode crash when the Qwen3-ASR backend was selected.
- Controlled subtitle lookup uses a cached current index plus a cached subtitle-start index for `bisect` fallback, avoiding a full subtitle scan every 250 ms on long videos.
- Controlled SenseVoice.cpp chunking now calls the correct `RollingPrefetchWorker` command helpers.

## Modes

- `NUC faster-whisper large-v3（远程 CUDA）`: highest-performance remote inference via NUC RTX 3080 Ti.
- `NUC Qwen3-ASR 1.7B（远程高质量离线）`: remote high-quality offline mode for long audio, queued on the NUC and kept separate from realtime ASR.
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
7. Run full-document LLM proofreading when enabled and the selected provider is ready. The default configured provider is Gemini 2.5 Flash, but the pipeline uses the provider selected in the UI.
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
- Scan previously generated subtitle files under `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/` and show them in a history sidebar.
- Attach any uploaded or historical subtitle file to a work conversation.
- Chat with the LLM grounded on the attached subtitle content.
- Run one-click `语句规整` or `转写成文稿` actions on the attached subtitle.
- Switch between `NUC Ollama Gemma 4 E4B (16K)`, `NUC Ollama Qwen3-14B`, `Local Rapid-MLX Qwen3-8B`, `Gemini 2.5 Flash`, and `Gemini 2.5 Pro`.

Notes:

- Long subtitle payloads now show an explicit warning when they are likely beyond the more comfortable single-shot range for local Qwen3-8B.
- The NUC Gemma 4 provider explicitly requests a 16K context window, disables thinking, caps output at 8192 tokens, and keeps the model loaded for 10 minutes between requests.
- When that warning appears, switch to `Gemini 2.5 Pro` if you have already configured its API key in the desktop app Settings pane.
- Action exports are written under `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/qwen-chat/exports/`.

Sync controls:

- `Sub -0.5s` / `Sub +0.5s`: adjust and persist subtitle offset for the current canonical video cache.
- `Sync line`: align the currently displayed subtitle line to the controlled Chrome video's current time.

## Hallucination Blocklist

To suppress recurring obviously unrelated subtitle hallucinations, the app filters a small set of known bad phrases before they are written into raw caches or exported subtitles.

- Built-in defaults cover several recurring phrases already observed in this project, such as `优优独播剧场——YoYo Television Series Exclusive`.
- You can extend the filter without editing Python code by appending phrases to:
  `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/hallucination_blocklist.txt`
- Add one phrase per line.
- Lines starting with `#` are treated as comments.

This blocklist is intentionally a conservative text-layer safeguard. It does not change model decoding parameters; it only removes exact recurring garbage phrases after transcription and before cache/export persistence.

## Runtime Artifacts

Some third-party speech tools may emit transient sidecar files such as `fbank_lfr_cmvn_feature.json`.

- These runtime artifacts are not treated as source files.
- The repository ignores known generated artifacts.
- Selected subprocesses are launched with `cwd=/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner` so those files land in the output area instead of polluting the source tree.

## Artifact Migration

Generated subtitle and transcript outputs now default to `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/`.

- New queue exports, controlled subtitle exports, and shared per-video Markdown copies are written there.
- The web workspace scans this location directly for historical generated subtitle assets.
- `hallucination_blocklist.txt`, `cache/`, `qwen-chat/`, and `realtime/` remain at the top level because they are runtime state, not exported deliverables.
- Optional local models, third-party backends, and Hugging Face cache now live under `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/`.

## Local Benchmark Notes

On this Apple M2 Mac mini:

- `large-v3-turbo-q5_0`: benchmark total about 7.6s.
- `large-v3-q5_0`: benchmark total about 16.1s.

Recommended defaults:

- Default subtitles: `whisper.cpp large-v3-turbo-q5_0`.
- Non-live video where delay is acceptable: `Controlled URL captions`.
- Maximum accuracy batch work: `whisper.cpp 高精度 q5_0（large-v3）`.

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

- **NUC ASR**: App-facing `:8000` remains an OpenAI-compatible `/v1/audio/transcriptions` endpoint. In the current NUC layout it is fronted by a thin busy-aware proxy, which forwards real transcription to the internal `faster-whisper-server` backend on `:18000`.
- **NUC High-Quality ASR**: Optional `Qwen3-ASR 1.7B` path exposed through a separate proxy on port `8001`, intended for single-concurrency long-audio offline jobs rather than realtime chunks.
- **NUC LLM**: Powered by native `Ollama` exposing `/api/chat` on port `11434`. Models like `qwen3:14b` run 6.5x faster than local Rapid-MLX on the Mac.

To use this, simply select a `nuc_asr` or `nuc_ollama` provider from the app UI dropdowns. The app implements automatic timeouts and fallbacks to ensure local Mac usability if the NUC is offline.

Current NUC port map:

- `:8000`: app-facing realtime/default ASR lane; persistent proxy exposing `/health`, `/busy`, `/v1/models`, and transcription/job endpoints.
- `:18000`: internal `faster-whisper-server` backend used by the `:8000` proxy.
- `:8001`: app-facing `Qwen3-ASR 1.7B` high-quality offline proxy.
- `:8002`: debug-accessible `Qwen3-ASR 1.7B` backend served by official `qwen-asr-serve`.
- `:11434`: Ollama LLM API.

### Local File Remote ASR Cache

For local media files, the app now avoids repeating `ffmpeg` extraction on every retry:

- The Mac first converts the source file once into `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/local-audio/<cache-key>/audio-16k-mono.wav`.
- Retries reuse that cached WAV instead of extracting audio again.
- The UI now exposes `删除本地音频缓存`, so the cache can be cleared manually for the current local file.
- A different source file naturally gets a different cache key because the key includes the resolved path, file size, and mtime.

This applies to both `NUC faster-whisper large-v3（远程 CUDA）` and `NUC Qwen3-ASR 1.7B（远程高质量离线）` when the source is a local file.

### NUC Runtime Deployment

Use the checked-in sync script instead of editing anonymous copies under `/srv`:

```bash
bash /Users/vickers/Documents/whisper-captioner/scripts/sync_nuc_runtime.sh --sync-only
bash /Users/vickers/Documents/whisper-captioner/scripts/sync_nuc_runtime.sh --deploy
```

`--sync-only` is the default. It verifies SHA-256 checksums, backs up the current
NUC sources and container definitions, and installs the candidate scripts without
rebuilding services. `--deploy` additionally recreates the scheduler, proxies,
and both GPU backends. It never deletes staging, results, or model caches.

The deployed layout includes:

- a local-only service scheduler sidecar that can start/stop the `Qwen3-ASR 1.7B` backend on demand
- an official `qwen-asr-serve` backend on port `8002`
- a lightweight serializing proxy on port `8001`
- single-concurrency admission with a GPU-free-memory guard and `faster-whisper` busy-awareness
- automatic backend warm-start before a request and idle backend shutdown after the request window expires
- a `180s` faster-whisper idle timer started through `POST :8010/release/asr`
- persistent host-side staging/results directories under `/srv/qwen3-asr-1p7b/qwen-asr-staging` and `/srv/qwen3-asr-1p7b/qwen-asr-results`

This keeps `:8000` as the app-facing endpoint, but turns it into a tiny front proxy that:

- counts active `faster-whisper` requests
- exposes `/busy` for the scheduler
- exposes an OpenAI-compatible static `/v1/models` response for client probing
- forwards real transcription work to an internal backend on `:18000`
- asks the scheduler to release the backend after the final active request

Admission behavior:

- An active Qwen offline job blocks new faster-whisper admission.
- The `realtime_asr` priority value identifies the admitted request lane; it does not preempt an active Qwen job.
- The two GPU backends are mutually exclusive: channel switches stop the idle backend and wait for VRAM release before starting the other.
- An already active faster-whisper request is allowed to finish before a queued Qwen job switches the GPU lane.
- When Qwen is idle, `faster-whisper` starts normally.
- The `:8001` proxy retries Qwen admission internally instead of immediately surfacing transient scheduler `429` or `503` responses.
- The Mac app's local-file NUC paths prefer `POST /jobs/upload` plus `GET /jobs/{id}` polling, so long jobs can keep reporting progress even when the original upload request would otherwise be fragile.
- When the busy endpoint is unavailable, the scheduler falls back to GPU utilization as a conservative signal.
- When `nvidia-smi` is unavailable or free GPU memory is below the threshold, admission fails with HTTP `503`.
- After a `Qwen` request finishes, the backend is stopped after the idle timeout so it does not sit on VRAM.
- After the final faster-whisper request, its backend is stopped after `180s`; the `:8000` proxy remains available.
- Queue and controlled-caption workers call `POST :8000/release/asr` after actually using NUC
  faster-whisper. The ASR proxy forwards only this safe operation to the localhost-only scheduler;
  busy, offline, and timeout responses are logged without changing completed results.

### ASR history and recovered chunk processing

The **ASR 历史** tab is backed by `CACHE_DIR / "asr-history.json"`. It supports cached-WAV
reruns when the original file is missing, atomic writes, corrupt-file preservation, and old output
path migration. Deleting a history row never deletes its WAV, subtitle cache, or output files.

Recovered Qwen3-ASR controls include 1-4 local process replicas, a default `45s` root chunk,
one-level adaptive splitting (`max(10.0, fastest_of_first_3 * 1.5)`), and FFmpeg remote-chunk
VAD (`-35dB`, `0.3s`). Two-replica local Qwen processing is enabled by default after machine
acceptance; adaptive splitting and remote VAD remain disabled by default.

Environment variables override `QSettings`:

```text
WHISPER_CAPTIONER_QWEN_PARALLEL=1
WHISPER_CAPTIONER_QWEN_REPLICAS=2
WHISPER_CAPTIONER_QWEN_CHUNK_SECONDS=45
WHISPER_CAPTIONER_ADAPTIVE_SPLIT=0
WHISPER_CAPTIONER_REMOTE_VAD=0
```

Current large-file local-file flow for `NUC Qwen3-ASR 1.7B`:

1. Mac extracts and caches a full `audio-16k-mono.wav` once.
2. Mac uploads that full cached WAV to `http://<NUC>:8001/jobs/upload`.
3. The proxy writes the original upload to `/srv/qwen3-asr-1p7b/qwen-asr-staging/<task-id>/`.
4. If the WAV is small enough, the proxy sends it upstream directly and spreads the returned text over the real WAV duration with approximate sentence-level timestamps.
5. If the WAV is larger than the direct-upload threshold, the proxy splits it on the NUC into `30s` WAV chunks with Python `wave`, uploads those chunks serially to the Qwen backend, and merges the transcript on the NUC side.
6. Before merge, the proxy filters obvious repetition-hallucination chunks, for example a long chunk dominated by one short phrase repeated hundreds of times in a low-information audio window.
7. The proxy writes `metadata.json`, `response.json`, and `chunks.json` into `/srv/qwen3-asr-1p7b/qwen-asr-results/<task-id>/`. Failed jobs also write `error.json`.

Validated on 2026-05-13:

- synthetic test file: `test-huge.wav`
- size on NUC: about `68 MB`
- duration: `2200s` (`36m40s`)
- task id: `20260513-143855-test-huge.wav` in the original validation run. New task IDs include a short random suffix, for example `YYYYMMDD-HHMMSS-<8hex>-filename.wav`, to avoid collisions when multiple uploads start in the same second.
- result: completed successfully after NUC-side chunking
- chunk count: `74`
- elapsed wall time from saved metadata: about `4m34s`

Useful inspection paths on the NUC:

- faster-whisper staging/results: `/srv/qwen3-asr-1p7b/asr-staging`, `/srv/qwen3-asr-1p7b/asr-results`
- faster-whisper home shortcuts: `/home/jack/whisper-captioner-asr-files/staging`, `/home/jack/whisper-captioner-asr-files/results`
- Qwen staging/results: `/srv/qwen3-asr-1p7b/qwen-asr-staging`, `/srv/qwen3-asr-1p7b/qwen-asr-results`

Current retention policy:

- These staging/results files are kept until manually deleted.
- There is currently no TTL or automatic pruning job.
- For Qwen, the NUC currently persists JSON metadata/results; final `.srt` and `.txt` are still exported by the Mac app.
- When a Qwen chunk is filtered as obvious repetition hallucination, `chunks.json` keeps the chunk entry and records `filtered_reason`; the merged transcript treats that chunk as empty instead of exporting the repeated garbage text.

Validated coexistence check:

```bash
curl -fsS http://127.0.0.1:8000/busy
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/healthz
curl -fsS http://127.0.0.1:8010/status
```

`GET :8000/health` reports proxy health with HTTP 200 even when the backend is
intentionally cold. Inspect its `upstream` field to distinguish `healthy` from
`stopped_or_unhealthy`. During an active `:8000` transcription, `/busy` should
report `active_requests: 1`. During an active Qwen job, the scheduler defers new
faster-whisper admission.

### NUC GPU Guard Helper

When the NUC GPU gets pinned by `Qwen3-ASR 1.7B` or `faster-whisper`, use:

```bash
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh status
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh auto-clean
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh prep-asr
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh release-asr
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh idle-watch
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh unload-all
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh start-qwen
```

Notes:

- `auto-clean` only stops the `Qwen3-ASR 1.7B` proxy/backend containers when free GPU memory drops below the threshold.
- `prep-asr` first frees Qwen GPU occupancy if needed, then starts the faster-whisper backend through the scheduler.
- `release-asr` starts the scheduler's faster-whisper idle timer.
- `idle-watch` stops idle Qwen occupancy and nudges the scheduler's faster-whisper release timer.
- `unload-all` stops both ASR lanes and frees GPU memory without deleting the containers.
- `start-qwen` first tries `docker start` on the existing `Qwen3-ASR 1.7B` containers, and only falls back to redeploy if they no longer exist.
- If a container does not stop cleanly, the script escalates from `docker stop` to `docker kill`, but it does not remove containers.

The first deploy may take a while because the official image is large and Docker needs to pull it before the backend can start.

Implementation note:

- The `faster-whisper` busy proxy already uses a baked image (`scripts/nuc_asr_busy_proxy.Dockerfile`) and no longer does runtime `pip install`.
- The current `Qwen3-ASR 1.7B` deploy script still launches `python:3.11-slim` containers and installs proxy dependencies at container start. That works, but it is not yet the fixed-image version.

Operational intent:

- keep `:8000` `faster-whisper` as the default realtime / daily ASR lane
- use `NUC Qwen3-ASR 1.7B（远程高质量离线）` only for batch-style long audio
- let the service scheduler handle ordinary Qwen warm-start, admission, and idle-stop behavior automatically
- do not assume `Qwen3-ASR 1.7B` can stably coexist with heavy `14B` Ollama traffic under peak GPU load

## Realtime Session Persistence & Review

Realtime Loopback capture (especially via NUC) now features a full "Two-Stage" persistence and correction pipeline:

1. **Capture & Chunking**: Audio is captured in 3s chunks and transcribed immediately for low-latency floating captions.
2. **Session Persistence**: Chunks are saved locally to `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/realtime/YYYYMMDD-HHMMSS/audio/`.
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

In the app, select a realtime mode such as `实时字幕 whisper.cpp small（SoundSource/Loopback）`, set the Loopback capture device ID, and click `实时字幕`.
Use `列出音频输入设备` in Settings to inspect AVFoundation device IDs. Loopback is often device `0`, but the app can save whichever numeric capture ID is detected or entered.

Current app behavior:

- `实时字幕` will automatically switch to `实时字幕 NUC large-v3（远程 CUDA，3s延迟）` when that mode exists in the mode list; if not, it falls back to `实时字幕 whisper.cpp small（SoundSource/Loopback）`.
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

- Loopback input device: usually named `Whisper Captions` or `Loopback` in AVFoundation. Device `0` is common on this setup, but the app should use the numeric ID reported by `列出音频输入设备`.
- Whisper models in `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/whisper-models`.
- Hugging Face cache in `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/huggingface-cache`.
- `whisper-stream`, `whisper-cli`, `ffmpeg`, `ffprobe`, and `yt-dlp`.
- Conda env `whishperapp_pyside6` for the GUI.
- Local SenseVoice.cpp checkout and GGUF model under `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp` if you want the `SenseVoice.cpp FP16` backend.

### SenseVoice.cpp Setup

The local SenseVoice.cpp checkout is intentionally not tracked in this repository and now lives under the separate app resource area instead of inside the repo tree.

If you want the `SenseVoice.cpp FP16` backend, clone and build SenseVoice.cpp locally first:

- Upstream project: [lovemefan/SenseVoice.cpp](https://github.com/lovemefan/SenseVoice.cpp)
- Upstream SenseVoice model project: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)

Recommended local path for this project:

```bash
git clone https://github.com/lovemefan/SenseVoice.cpp /Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp
cd /Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp
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
