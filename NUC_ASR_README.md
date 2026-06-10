# NUC ASR Notes

This file summarizes the current remote ASR layout used by `Whisper Captioner`.

## What Runs on the NUC

- `:8000`: app-facing `faster-whisper` proxy
- `:18000`: internal `faster-whisper-server` backend
- `:8001`: app-facing `Qwen3-ASR 1.7B` proxy
- `:8002`: `Qwen3-ASR 1.7B` backend
- `:11434`: Ollama for subtitle cleanup / LLM work

Current scheduler priority:

- active `Qwen3-ASR 1.7B` work blocks new faster-whisper admission
- otherwise faster-whisper is admitted as the `realtime_asr` lane

Both GPU backends are cold-started on demand. The scheduler stops Qwen after its
idle window and stops faster-whisper `180s` after the last request. The three
lightweight containers (`:8000`, `:8001`, and `:8010`) remain running.
The GPU backends are mutually exclusive. A channel switch waits for the current
request to finish, stops the idle backend, confirms VRAM release, and only then
starts the other backend.

## Deployment

Run deployment from the Git checkout on the Mac:

```bash
bash scripts/sync_nuc_runtime.sh --sync-only
bash scripts/sync_nuc_runtime.sh --deploy
```

The default `--sync-only` mode verifies SHA-256 checksums and creates a timestamped
backup under `/srv/qwen3-asr-1p7b/backups/`. `--deploy` also rebuilds all five
containers. Neither mode deletes staging, results, or model caches. Do not edit
the runtime aliases `proxy.py`, `scheduler.py`, or `asr_busy_proxy.py` manually.

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
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/healthz
curl -fsS http://127.0.0.1:8001/busy
curl -fsS http://127.0.0.1:8010/status
```

The `:8000` proxy deliberately returns HTTP 200 while its GPU backend is cold.
Read the `/health` response's `upstream` field to determine whether the backend
is loaded. Scheduler status includes `gpu.available` and `gpu.error`; both ASR
lanes reject admission with HTTP 503 when NVML or `nvidia-smi` is unavailable.

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
- Staging and result directories have no automatic TTL and must be pruned manually.
