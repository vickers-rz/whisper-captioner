#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${WHISPER_CAPTIONER_HF_HOME:-/Volumes/T7/MacBackup/Movies/whisper-captioner_APP_Resource/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
conda run -n whishperapp_pyside6 python "$SCRIPT_DIR/whisper_captioner/app.py"
