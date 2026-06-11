# Whisper Captioner T7 Runtime Log Recovery Investigation

Date: 2026-06-11

## Scope

Searched:

- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner`
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource`
- the current Git object database and reflog

The highest-value interval is 2026-05-14 through 2026-05-27. The repository was
cloned again on 2026-05-27 00:58:50, after the last pre-crash commit from
2026-05-14.

No recoverable application `.py`, `.pyc`, editor backup, archive, or app bundle
was found under the T7 Movies tree. The remaining reconstruction evidence is in
runtime logs, cache metadata, generated outputs, and one ASR history database.

## High-Confidence Missing Features

### 1. ASR history browser and cached-audio reprocessing

Evidence:

- `cache/asr-history.json` stores source identity, canonical source, title,
  source kind, audio cache key/path, last ASR mode, output path, status, and
  timestamps.
- Logs repeatedly contain `Loaded ASR history into input`,
  `Reprocessing ASR history with cached WAV`, and `Reusing history audio cache`.
- The feature worked for both URLs and local files even when the original local
  source was unavailable, provided the cached WAV still existed.

Current state:

- Local WAV caching still exists in `workers.py`.
- The history database, history UI, load action, and reprocess path are absent
  from the current source.

### 2. Multi-process Qwen3-ASR 0.6B batch transcription

Evidence:

- Logs show a configurable replica count of 2, 3, and 4 workers.
- The missing constant was named
  `QWEN3_ASR_06B_4BIT_MLX_REPLICA_COUNT`.
- A 2026-05-23 02:50 run first failed with a `NameError` because
  `QWEN3_ASR_06B_4BIT_MLX_REPLICA_COUNT` was not defined. This was a temporary
  configuration regression, not evidence that the implementation never ran.
- Successful runs then processed a 10,158.2-second file with 226 45-second
  chunks:
  - 4 replicas ran from 02:56:10 to 03:06:32 and logged `All chunks
    transcribed`, `Done`, and `Queue worker finished`.
  - 2 replicas ran from 03:16:34 to 03:27:01 and logged the same successful
    completion sequence.
- Both successful runs produced matching `.txt`, `.srt`, and `.json` artifacts
  at their logged completion timestamps.
- Logs report out-of-order completion and active in-flight chunk IDs.

Current state:

- `_transcribe_local_qwen3_asr_chunked()` is strictly sequential and uses fixed
  30-second chunks.

### 3. Adaptive slow-chunk splitting

Evidence:

- The scheduler measured the first three completed root chunks.
- It selected a 10.0-second slow-chunk timeout from the fastest early results.
- A slow root chunk was cancelled/replaced by two subchunks such as `10a/10b`.
- The progress total increased dynamically after splitting and tracked recent
  split relationships.
- This was tested with 45-, 60-, and 90-second root chunks.

Current state:

- No adaptive timeout, straggler detection, dynamic task graph, or subchunk
  naming remains in the current source.

### 4. VAD trimming before remote NUC requests

Evidence:

- Logs contain per-chunk leading/trailing silence removal, for example
  `VAD trimmed 0.12s leading / 1.84s trailing`.
- Fully silent/unstable chunks were skipped with
  `no stable voice window detected`.
- The behavior was used by both NUC faster-whisper and NUC Qwen3-ASR paths.

Current state:

- Current remote chunking uses overlap trimming after inference, but it does not
  perform the logged pre-request voice-window detection and VAD crop.

### 5. Automatic remote NUC lifecycle tied to queue work

Evidence:

- Logs show separate SSH stop operations for NUC ASR and NUC Qwen services when
  a queue finished, failed, or was stopped.
- The feature used remote HTTP control commands over SSH and tolerated network,
  host-down, and already-stopped states.

Current state:

- The June recovery restored scheduler/proxy orchestration and Wake-on-LAN
  support, including `/stop/asr` and `/stop/qwen` scheduler endpoints.
- The current APP queue code no longer contains the old SSH stop calls or the
  queue-to-runtime lifecycle binding. The old behavior should be reviewed
  separately before restoring it because it may conflict with the newer
  scheduler architecture.

## Runtime Artifacts That Preserve Behavior

- `cache/asr-history.json`: direct schema for the missing history feature.
- `cache/local-audio/*/metadata.json`: source identity and cached WAV metadata.
- `cache/*/manifest.json`: URL, ASR backend/model, LLM provider/model,
  chunk size, cache URL, and pipeline version.
- `artifacts/generated/*`: output naming proves combined ASR/LLM mode labels,
  including `nuc_asr_queue`, `nuc_ollama_8b`, `nuc_ollama_gemma4`,
  `minimax_m27`, original transcript, LLM optimized transcript, and LLM
  normalized transcript variants.

## Already Present Or Subsequently Restored

- Persistent local audio cache.
- Basic local Qwen3-ASR chunking.
- SenseVoice.cpp chunking and Metal runtime.
- NUC job polling and heartbeat logging.
- Subtitle post-processing workspace and provider selection.
- Opening the active cache directory in Finder.
- Remote NUC runtime scheduler/proxy scripts and current Wake-on-LAN support.

## Git Recovery Result

The current clone's reflog begins at 2026-05-27 00:58:50. No pre-crash branch,
commit, or worker/app source blob is present.

Unreachable Git blobs were inspected. They contain README/config/runtime script
variants and unrelated MCP files, but none contain the missing history,
parallel-replica, adaptive-split, or VAD implementation strings.

## Independent Review Correction

A later review proposed lowering the multi-replica feature's confidence because
it found only the undefined-constant error. That conclusion came from searching
English `replica` terms while the successful runtime message was Chinese:
`Qwen3-ASR 0.6B 高吞吐并发：N 个副本并行处理 chunk`.

Direct inspection of the complete log tails and generated artifacts confirms
that the 4-replica and 2-replica runs both completed successfully. The correct
interpretation is:

- the feature was implemented and successfully used;
- one earlier launch had a missing-constant regression;
- the current source has lost the implementation and configuration.

The same review described queue-linked NUC shutdown as "code still exists".
Only the newer scheduler stop endpoints still exist. The APP-side queue-linked
SSH invocation visible in the May logs is absent from the current Python source.

## Recommended Reconstruction Order

1. Restore the ASR history repository and UI from `asr-history.json`.
2. Add a tested worker-pool implementation for local Qwen3-ASR replicas.
3. Add adaptive subchunk splitting behind explicit configuration.
4. Restore VAD pre-trimming as a reusable helper for remote ASR backends.
5. Decide whether queue-coupled NUC shutdown is still desirable under the
   current scheduler architecture.

The first item is the safest recovery because its persisted schema and user
workflow are directly observable. Items 2 through 4 affect transcription
correctness and process lifecycle and should be restored with focused tests
rather than by reproducing log strings alone.

## Recovery Implementation Status

Implemented on 2026-06-11:

- typed ASR history storage, migration, metadata relocation, atomic writes, and corrupt JSON recovery;
- independent history tab, cached-WAV reruns, model restore, filtering, and record-only deletion;
- prepared WAV injection, structured chunk progress, and history state updates in `QueueWorker`;
- scheduler `POST /release/asr` without restoring SSH force-stop behavior;
- opt-in FFmpeg VAD for chunked NUC Qwen and faster-whisper paths;
- opt-in 1-4 replica local Qwen scheduling, retry, stop cleanup, ordered merge, and adaptive splitting.

The 227.718-second machine acceptance run completed with identical output for 1, 2, and 4 replicas:

| Replicas | Wall time | Segments | Characters | Output SHA-256 |
|---:|---:|---:|---:|---|
| 1 | 90.86s | 57 | 2015 | `d16d1135e140a2646baf0305d4973374ed75f45114722ad488795e3dfc9503bb` |
| 2 | 27.67s | 57 | 2015 | same |
| 4 | 26.41s | 57 | 2015 | same |

Stopping a four-replica run terminated four active child processes in 0.11 seconds and left no
`mlx_audio.stt.generate` process behind. Two replicas are therefore enabled by default. Four
replicas provide negligible additional throughput on this machine. VAD and adaptive splitting
remain disabled by default; adaptive mode was also run against the same audio and correctly made
zero splits because no root chunk exceeded the calibrated threshold.

## NUC Provenance Correction

The NUC runtime is a mixed source and must not be treated as one authoritative snapshot:

- `/srv/qwen3-asr-1p7b/backups/20260610-224051/` preserves the pre-sync May implementation.
- The May backup contains the uncommitted persistent job APIs, staging/result/error artifacts,
  Qwen large-WAV 30-second chunking, busy snapshots, scheduler admission, and ASR idle release.
- The active June 10 files were synchronized from the Git checkout and then adapted, but they
  retained and hardened much of that May runtime behavior.
- NUC result directories from May 16 and May 22 independently confirm Mac-side per-chunk Qwen
  requests before the June 10 synchronization.

Therefore the recovery uses the pre-sync backup and runtime artifacts as evidence for interfaces,
while keeping reconstructed Mac-side concurrency, VAD, and adaptive splitting behind feature flags.

## 2026-06-11 NUC ASR Optimization

- Restored safe large-file Qwen chunking with 30-second nominal windows and 2 seconds of context on each available edge.
- Added fuzzy Chinese prefix/suffix deduplication with up to two edit errors for sufficiently long overlaps.
- Added a two-half retry when Qwen returns empty text for a non-silent chunk; true silence remains empty.
- Added a separate NUC faster-whisper Turbo mode while retaining large-v3 as the high-quality mode.
- Aligned scheduler idle shutdown and faster-whisper model TTL to 900 seconds.
- A 227.718-second WAV completed in 3.68 seconds with warm Turbo versus 15.54 seconds with large-v3. Turbo produced fewer characters, so it was not made a silent replacement for the quality mode.
- A forced chunk-path Qwen test completed in 94.52 seconds including backend cold start and produced eight overlapped chunks. Runtime diagnostics confirmed 32/34-second request windows and fuzzy boundary matches with up to two edit errors.
