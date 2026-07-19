#!/usr/bin/env bash
# Shell scheduler for the deterministic subtitle-forensics pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FFPROBE_BIN="${FFPROBE_BIN:-/opt/homebrew/bin/ffprobe}"

usage() {
  cat <<'EOF'
Usage:
  forensic_subtitle_pipeline.sh probe URL OUTPUT_DIR [--cookies-from-chrome]
  forensic_subtitle_pipeline.sh finalize NUC_ASR_JSON GEMINI_TRANSCRIPT OUTPUT_DIR
  forensic_subtitle_pipeline.sh targeted-ocr VIDEO GEMINI_TRANSCRIPT NUC_ASR_JSON DIFFERENCES_JSON OUTPUT_DIR

Commands:
  probe         Download only low-resolution, short samples and decide whether
                burned subtitles are present. It never downloads the full video.
  finalize      Write Gemini text back into the fixed NUC timeline, create a
                timed dispute report, then deterministically re-segment final.srt.
  targeted-ocr  Use dispute windows to sample an already downloaded local video,
                run Apple Vision OCR, and preserve raw visual evidence. This
                command does not alter timestamps or subtitle text automatically.

Environment overrides: PYTHON_BIN, FFMPEG_BIN, FFPROBE_BIN.
EOF
}

run_python() {
  "$PYTHON_BIN" "$@"
}

command_name="${1:-}"
case "$command_name" in
  probe)
    [[ $# -ge 3 ]] || { usage >&2; exit 2; }
    source_url="$2"
    output_dir="$3"
    shift 3
    run_python "$ROOT/scripts/forensic_subtitle_command.py" \
      probe-hard-subs "$source_url" --output-dir "$output_dir" "$@"
    ;;
  finalize)
    [[ $# -eq 4 ]] || { usage >&2; exit 2; }
    run_python "$ROOT/scripts/forensic_subtitle_command.py" finalize \
      --nuc-asr "$2" --gemini "$3" --output-dir "$4"
    ;;
  targeted-ocr)
    [[ $# -eq 6 ]] || { usage >&2; exit 2; }
    video="$2"
    gemini="$3"
    nuc_asr="$4"
    differences="$5"
    output_dir="$6"
    mkdir -p "$output_dir"
    plan="$output_dir/targeted-ocr-frame-plan.json"
    frames="$output_dir/frames"
    raw="$output_dir/apple-vision-raw.jsonl"
    ocr_dir="$output_dir/ocr"
    duration="$($FFPROBE_BIN -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video")"
    run_python "$ROOT/scripts/plan_targeted_ocr_frames.py" \
      --ogg-transcript "$gemini" --nuc-asr "$nuc_asr" \
      --differences "$differences" --output "$plan"
    run_python "$ROOT/scripts/extract_targeted_ocr_frames.py" \
      --video "$video" --plan "$plan" --output-dir "$frames"
    ocr_binary="$output_dir/apple-vision-ocr"
    swift_cache="$output_dir/swift-module-cache"
    mkdir -p "$swift_cache"
    swiftc -O -Xcc "-fmodules-cache-path=$swift_cache" \
      "$ROOT/scripts/apple_vision_ocr.swift" -o "$ocr_binary"
    "$ocr_binary" --frames "$frames" --output "$raw" --fps 6 \
      --timestamps "$frames/timestamps.json"
    run_python "$ROOT/scripts/evaluate_burned_subtitle_ocr.py" \
      --raw "$raw" --asr "$nuc_asr" --output-dir "$ocr_dir" \
      --fps 6 --duration "$duration"
    ;;
  -h|--help|help|'')
    usage
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    usage >&2
    exit 2
    ;;
esac
