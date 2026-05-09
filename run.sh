#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$HOME/Movies/whisper-captioner_APP_Resource/huggingface-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
conda run -n pyside6 python /Users/vickers/whisper-captioner/whisper_captioner/app.py
