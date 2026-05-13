# NUC ASR Notes

This file summarizes the current remote ASR layout used by `Whisper Captioner`.

## What Runs on the NUC

- `:8000`: app-facing `faster-whisper` proxy
- `:18000`: internal `faster-whisper-server` backend
- `:8001`: app-facing `Qwen3-ASR 1.7B` proxy
- `:8002`: `Qwen3-ASR 1.7B` backend
- `:11434`: Ollama for subtitle cleanup / LLM work

Current scheduler priority:

- `Qwen3-ASR 1.7B` first
- `faster-whisper` second

That means active Qwen offline work now blocks new `faster-whisper` admission instead of yielding to it.

## Local-file Transcription Flow

For local files from the Mac app:

1. The Mac extracts one cached `audio-16k-mono.wav`.
2. The Mac uploads that WAV once to the NUC.
3. The NUC proxy writes the original upload into a staging directory.
4. The proxy runs ASR locally on the NUC side.
5. The proxy writes result JSON files into a result directory.

This avoids repeating `ffmpeg` extraction on the Mac and avoids wasting GPU work when the Mac-side HTTP request times out.

The Mac app now uses the proxy job endpoints for local-file NUC transcription:

- upload: `POST /jobs/upload`
- poll: `GET /jobs/{task_id}`

Task IDs include a timestamp, a short random suffix, and the safe filename, for example:

```text
YYYYMMDD-HHMMSS-<8hex>-audio-16k-mono.wav
```

The random suffix avoids collisions when repeated uploads happen in the same second.

## NUC Paths

### faster-whisper

- staging: `/srv/qwen3-asr-1p7b/asr-staging`
- results: `/srv/qwen3-asr-1p7b/asr-results`
- home shortcuts:
  - `/home/jack/whisper-captioner-asr-files/staging`
  - `/home/jack/whisper-captioner-asr-files/results`

### Qwen3-ASR 1.7B

- staging: `/srv/qwen3-asr-1p7b/qwen-asr-staging`
- results: `/srv/qwen3-asr-1p7b/qwen-asr-results`

Each Qwen task directory usually contains:

- `metadata.json`
- `response.json`
- `chunks.json` for large files that were split on the NUC
- `error.json` only when the task failed

For filtered bad chunks:

- `chunks.json` may include `filtered_reason`
- filtered chunks are treated as empty during merge
- this is meant to suppress obvious repetition hallucinations in low-information audio windows

Each staging directory usually contains:

- the uploaded WAV
- `upload.json`

Small Qwen uploads are sent upstream as a single WAV and the returned text is spread across the real WAV duration with approximate sentence-level timestamps.
Large Qwen uploads are split into `30s` WAV chunks on the NUC side, then merged back into one response.

## Current Retention Policy

- Files remain on disk until manually deleted.
- There is no automatic cleanup yet.
- Mac exports the final `.srt` and `.txt`; the NUC currently persists JSON metadata/results.

## Health Checks

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/busy
curl -fsS http://127.0.0.1:8001/healthz
curl -fsS http://127.0.0.1:8001/busy
```

Useful GPU check:

```bash
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,fan.speed --format=csv
```

## Validated Large-file Qwen Test

Verified on 2026-05-13:

- task id: `20260513-143855-test-huge.wav`
- staged WAV: `/srv/qwen3-asr-1p7b/qwen-asr-staging/20260513-143855-test-huge.wav/test-huge.wav`
- size: about `68 MB`
- duration: `2200s` (`36m40s`)
- result dir: `/srv/qwen3-asr-1p7b/qwen-asr-results/20260513-143855-test-huge.wav`
- chunking: enabled
- chunk count: `74`
- outcome: completed

Newer task IDs include an 8-character random suffix, so current result directories no longer use only `timestamp-filename`.

## Known Current Boundaries

- The Qwen proxy can now absorb large local-file uploads by chunking on the NUC side, but transcript merge quality is still simpler than a true timestamped ASR engine.
- A conservative repetition-hallucination filter now drops chunks dominated by one short repeated phrase instead of exporting that garbage into the final subtitle.
- `faster-whisper` proxy already uses a baked image and does not install Python deps at runtime.
- `Qwen3-ASR 1.7B` proxy still starts from `python:3.11-slim` and installs Python deps when the container starts. It works, but it is not yet the fixed-image version.
