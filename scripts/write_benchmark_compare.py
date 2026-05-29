from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.subtitle_io import parse_sense_voice_output, save_segments_as_srt, save_segments_as_txt


OUTPUT_DIR = Path(os.environ.get("WHISPER_CAPTIONER_OUTPUT_DIR", "/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner")).expanduser()
COMPARE_DIR = OUTPUT_DIR / "benchmark-30s-model-compare"
SENSE_STDOUT = COMPARE_DIR / "sensevoicecpp-fp16.stdout.txt"
SENSE_TXT = COMPARE_DIR / "sensevoicecpp-fp16.txt"
SENSE_SRT = COMPARE_DIR / "sensevoicecpp-fp16.srt"
SUMMARY = COMPARE_DIR / "benchmark-summary.md"


def main() -> None:
    stdout_text = SENSE_STDOUT.read_text(encoding="utf-8")
    segments = parse_sense_voice_output(stdout_text)
    save_segments_as_txt(SENSE_TXT, segments)
    save_segments_as_srt(SENSE_SRT, segments)

    whisper_text = (COMPARE_DIR / "whispercpp-large-v3-turbo-q5_0.txt").read_text(encoding="utf-8")
    sense_text = SENSE_TXT.read_text(encoding="utf-8")
    qwen_text = (COMPARE_DIR / "qwen3-asr-0.6b-4bit.txt").read_text(encoding="utf-8")

    summary = """# 30s Benchmark Compare

- Sample: `/tmp/sensevoice-test-30s.wav`
- whisper.cpp large-v3-turbo-q5_0: 4.22s
- SenseVoice.cpp fp16: 5.37s
- mlx-community/Qwen3-ASR-0.6B-4bit: 6.46s

## Precision Notes

- whisper.cpp large-v3-turbo-q5_0:
  Accurate word boundaries and the most stable timestamps. Text is slightly more fragmented and subtitle-like.
- SenseVoice.cpp fp16:
  Fast and quite fluent, but it occasionally rewrites short spans more aggressively, so punctuation and wording can drift a bit.
- mlx-community/Qwen3-ASR-0.6B-4bit:
  The most naturally normalized prose-like output of the three, but it does not emit timestamped subtitles through the current mlx-audio path, so it is better suited to transcript-first workflows.

## Text Heads

### whisper.cpp large-v3-turbo-q5_0
{whisper_head}

### SenseVoice.cpp fp16
{sense_head}

### mlx-community/Qwen3-ASR-0.6B-4bit
{qwen_head}
""".format(
        whisper_head=whisper_text[:500].strip(),
        sense_head=sense_text[:500].strip(),
        qwen_head=qwen_text[:500].strip(),
    )
    SUMMARY.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
