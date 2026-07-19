#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)"
DEFAULT_PYTHON="${HOME}/miniforge3/envs/whishperapp_pyside6/bin/python"
PYTHON_BIN="${FORENSIC_PYTHON:-$DEFAULT_PYTHON}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/webm_to_ogg_gemini_asr.sh INPUT.webm|INPUT.ogg [OUTPUT_DIR]

Environment:
  GEMINI_API_KEY       Gemini API key. If unset, the script tries macOS Keychain
                       service "WhisperCaptioner", account "gemini-api-key".
  GEMINI_ASR_MODEL     Optional Gemini model. Default: gemini-2.5-flash
  FORENSIC_PYTHON      Optional Python interpreter override.

Output:
  OUTPUT_DIR/work/gemini-audio.ogg for transcoded input, or original OGG upload
  OUTPUT_DIR/gemini-local-audio-asr-transcript.txt
  OUTPUT_DIR/gemini-local-audio-asr-metadata.json
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage >&2
  exit 2
fi

INPUT="$1"
OUTPUT_DIR="${2:-}"
MODEL="${GEMINI_ASR_MODEL:-gemini-2.5-flash}"

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "python3 not found. Set FORENSIC_PYTHON to a valid interpreter." >&2
  exit 127
fi

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT" >&2
  exit 2
fi

case "${INPUT##*.}" in
  webm|WEBM|ogg|OGG|oga|OGA) ;;
  *)
    echo "Input should be a .webm or .ogg file for this wrapper: $INPUT" >&2
    exit 2
    ;;
esac

if [ -z "${GEMINI_API_KEY:-}" ]; then
  GEMINI_API_KEY="$(/usr/bin/security find-generic-password \
    -s "WhisperCaptioner" \
    -a "gemini-api-key" \
    -w 2>/dev/null || true)"
  export GEMINI_API_KEY
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY is not set and no saved Keychain key was found." >&2
  echo "Run with: GEMINI_API_KEY=... scripts/webm_to_ogg_gemini_asr.sh INPUT.webm" >&2
  exit 2
fi

ARGS=(gemini-local "$INPUT" --model "$MODEL" --timeout 900 --upload-timeout 300 --processing-timeout 1200)
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/asr_entrypoints.py" "${ARGS[@]}"
