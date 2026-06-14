# Subtitle Pipeline Reference Review

Date: 2026-06-13

Reviewed projects:

- argmaxinc/argmax-oss-swift
- jianshuo/claude-skills/wjs-transcribing-audio
- lifeiteng/OmniVAD-Kit
- lattifai/lattifai-python
- lattifai/lattifai-skills

## Executive decision

The most valuable improvement is not another ASR backend. It is a backend-neutral
pipeline with observable intermediate artifacts:

1. detect speech regions;
2. transcribe into words/segments rather than final SRT;
3. assemble readable cues deterministically;
4. compare subtitle coverage against detected speech;
5. retry only uncovered speech regions;
6. optionally run forced alignment and diarization.

This design directly addresses the observed failure mode where audible speech
around 11:17-11:20 had no subtitle. A successful ASR request is not sufficient;
the pipeline needs a coverage invariant.

## Priority recommendations

### P0: Add a subtitle coverage audit

Use VAD speech intervals as an independent reference and calculate whether every
meaningful speech interval overlaps at least one subtitle segment.

Suggested outputs:

- `speech_regions.json`
- `asr_raw.json`
- `subtitle_cues.json`
- `quality_report.json`
- final `.srt`

The report should include:

- uncovered speech intervals;
- subtitle intervals with no detected speech;
- overlap or non-monotonic timestamps;
- suspiciously long cues;
- repeated-text/hallucination signals;
- speech coverage ratio by duration.

Uncovered intervals longer than a configurable threshold should be retried with
guard audio on both sides. This is safer than globally enabling aggressive VAD,
which can itself delete quiet speech.

### P0: Request word timestamps and build SRT locally

The `wjs-transcribing-audio` workflow correctly separates recognition from
subtitle rendering. It requests structured ASR output with word timestamps,
then deterministically assembles cues using punctuation, length, duration and
word gaps.

For the NUC faster-whisper path:

- request `verbose_json`;
- forward `timestamp_granularities[]=word` when the upstream supports it;
- retain words in the internal cache;
- render SRT locally instead of treating upstream segments as final cues.

The cue builder should support:

- hard punctuation breaks;
- soft punctuation breaks after a minimum length;
- maximum duration and characters;
- split on large word gaps;
- minimum cue duration;
- monotonic, non-overlapping timestamps;
- CJK and Latin-specific line-length policies.

### P1: Evaluate OmniVAD in shadow mode

OmniVAD-Kit is the best direct fit among the reviewed repositories:

- Apache-2.0;
- small FireRedVAD models;
- macOS arm64 support;
- non-stream and 10 ms streaming VAD;
- speech/singing/music event detection;
- long-audio overlap aggregation;
- Whisper-oriented 30-second chunk packing.

Initially run it alongside the current ffmpeg `silencedetect` implementation
without changing production decisions. Record both outputs on the existing
failure corpus, especially quiet speech, music, English/Chinese transitions and
the 11:17-11:20 omission.

If the benchmark is favorable, use OmniVAD speech intervals for:

- speech-aware chunk boundaries;
- coverage auditing;
- targeted retries;
- real-time speech gating;
- suppressing music-only hallucinations with AED.

Keep ffmpeg detection as a fallback. Model VAD should not become a hard
dependency until wheel availability, startup cost and real files are verified.

### P1: Detect language once, then pin it

The reviewed workflows often pin language for stability. This project also
needs multilingual input, so globally forcing `zh` is incorrect.

Recommended policy:

1. auto-detect language from the first speech-rich window;
2. pin the detected language for subsequent chunks;
3. allow explicit user override;
4. re-detect only after strong evidence of a language transition.

This avoids both the old forced-Chinese failure and per-chunk language drift.

### P2: Add forced alignment as an optional second pass

LattifAI's useful concept is the separation of transcription and alignment.
Forced alignment is particularly valuable for:

- Qwen ASR outputs that currently receive pseudo timestamps;
- externally supplied transcripts;
- polished text whose cue boundaries changed;
- word-level or karaoke exports.

Do not add `lattifai-python` to the core environment. Its alignment path
requires LattifAI authentication, uses custom/private package distribution, and
introduces a large dependency surface with potential conflicts.

Prefer an optional CLI adapter:

```text
raw transcript -> temporary JSON/SRT -> lai alignment align -> aligned JSON
```

Store the raw and aligned artifacts separately. Before product integration,
verify model licensing, offline behavior, API/quota requirements, MPS speed and
Chinese mixed-language quality.

### P2: Add optional diarization

Argmax SpeakerKit demonstrates a clean post-ASR design: run diarization
independently, then assign speakers to transcript words/segments by temporal
overlap. It also preserves an RTTM artifact.

Because this application is Python/PySide, do not rewrite the pipeline in
Swift. Potential integration options are:

- invoke `argmax-cli` for Apple-native diarization;
- use its local OpenAI-compatible server as an optional WhisperKit backend;
- implement the same RTTM-plus-temporal-join contract with another Python
  diarizer.

Diarization should remain optional and targeted at interviews, meetings and
courses.

## Project-by-project assessment

### argmax-oss-swift

Absorb:

- OpenAI-compatible local ASR service boundary;
- streaming partial results;
- word and segment timestamp support;
- independent diarization plus temporal merge;
- explicit model lifecycle and lazy loading.

Do not absorb directly:

- Swift/Core ML implementation into the Python application;
- TTS features unrelated to the current subtitle reliability problem.

Recommended use: optional external Apple-native backend and architecture
reference. License: MIT.

### wjs-transcribing-audio

Absorb:

- structured ASR JSON as the source of truth;
- deterministic cue assembly;
- word timestamp repair;
- punctuation-aware CJK segmentation;
- retries for long/cloud jobs;
- invariant-preserving post-processing.

Adapt rather than copy:

- its 10-minute cloud streaming chunks are not suitable defaults for local
  Whisper;
- its cue builder is Chinese-specific and needs duration/gap/CPS policies;
- filler deletion should be optional because it changes transcript content.

The repository subtree has no standalone license file, so copy concepts and
write an original implementation unless repository-level licensing is
confirmed.

### OmniVAD-Kit

Absorb after benchmark:

- model VAD as an independent signal;
- overlap-aware long-file processing;
- 30-second speech-region packing;
- streaming VAD for live captioning;
- AED speech/singing/music labels.

Risk:

- beta status;
- native binary packaging;
- VAD thresholds can still remove low-energy speech if used aggressively.

Recommended use: shadow-mode benchmark, then optional primary VAD with ffmpeg
fallback. License: Apache-2.0.

### lattifai-python

> **2026-06-14: 已测试并放弃。** 对齐质量不满足本项目需求，相关代码已从
> `external_backends.py`、`workers.py` 和测试中移除。

Absorb:

- forced alignment as a separate stage;
- word-level canonical JSON;
- long-file segmented alignment;
- alignment score/sanity checks;
- preservation of source cue IDs and boundaries;
- caption standardization profiles.

Do not absorb as a core dependency:

- authenticated/proprietary alignment path;
- custom package index and broad dependency graph;
- full workflow, translation and summarization stack.

Recommended use: optional CLI integration and design reference. Repository
license: MIT; model/service terms require separate verification.

### lattifai-skills

Absorb:

- small, composable pipeline stages;
- stable artifact naming (`transcript`, `aligned`, `diarized`, `translated`);
- validators that compare source and output invariants;
- raw acoustic speaker IDs separated from inferred names;
- JSON as the lossless interchange format, with SRT/VTT/ASS as render targets.

Recommended use: product workflow and QA design reference, not runtime
dependency. License: MIT.

## Proposed internal data model

Extend the current segment-only cache without breaking SRT output:

```json
{
  "media": {"duration": 680.2, "language": "en"},
  "speech_regions": [{"start": 676.8, "end": 680.1, "confidence": 0.82}],
  "words": [{"start": 677.1, "end": 677.6, "text": "Questions"}],
  "segments": [{"start": 677.1, "end": 680.0, "text": "Questions 16 to 20."}],
  "quality": {"speech_coverage": 0.997, "uncovered_regions": []}
}
```

Keep LLM-polished text separate from acoustic timing data. LLM stages may
change text, but must not silently change segment count, ordering or timestamps
unless an explicit realignment stage follows.

## Implementation order

1. Add word-capable internal schema, deterministic cue builder and validators.
2. Add VAD-to-subtitle coverage audit and targeted uncovered-region retries.
3. Run OmniVAD in shadow mode and benchmark against ffmpeg/faster-whisper VAD.
4. Implement detect-once-then-pin language behavior.
5. Add optional forced-alignment CLI adapter.
6. Add optional diarization and richer export formats only after reliability
   metrics are stable.

## Acceptance tests

The first two phases should not be considered complete until:

- the known 11:17-11:20 speech interval is either captioned or reported as
  uncovered;
- no existing subtitle segment is lost during cue rendering;
- timestamps remain monotonic and non-overlapping;
- raw ASR, rendered cues and quality report are reproducible from cache;
- VAD-disabled and VAD-fallback paths still work;
- multilingual samples do not regress because of a forced language.

## Sources

- https://github.com/argmaxinc/argmax-oss-swift
- https://github.com/jianshuo/claude-skills/tree/main/wjs-transcribing-audio
- https://github.com/lifeiteng/OmniVAD-Kit
- https://github.com/lattifai/lattifai-python
- https://github.com/lattifai/lattifai-skills
