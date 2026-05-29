#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${WHISPER_CAPTIONER_T7_SENSEVOICE_DIR:-/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp}"
TARGET_DIR="${WHISPER_CAPTIONER_SENSEVOICE_DIR:-/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models/SenseVoice.cpp}"
OLD_RPATH="${WHISPER_CAPTIONER_OLD_SENSEVOICE_RPATH:-$HOME/Movies/whisper-captioner_APP_Resource/third_party/SenseVoice.cpp/build/lib}"

SOURCE_BIN="$SOURCE_DIR/build/bin/sense-voice-main"
SOURCE_LIB_DIR="$SOURCE_DIR/build/lib"
SOURCE_MODEL="$SOURCE_DIR/models/sense-voice-gguf/sense-voice-small-fp16.gguf"

TARGET_BIN_DIR="$TARGET_DIR/build/bin"
TARGET_LIB_DIR="$TARGET_DIR/build/lib"
TARGET_MODEL_DIR="$TARGET_DIR/models/sense-voice-gguf"
TARGET_BIN="$TARGET_BIN_DIR/sense-voice-main"
TARGET_MODEL="$TARGET_MODEL_DIR/sense-voice-small-fp16.gguf"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing directory: $path" >&2
    exit 1
  fi
}

require_file "$SOURCE_BIN"
require_dir "$SOURCE_LIB_DIR"
require_file "$SOURCE_MODEL"

mkdir -p "$TARGET_BIN_DIR" "$TARGET_LIB_DIR" "$TARGET_MODEL_DIR"

echo "Copying SenseVoice.cpp runtime from:"
echo "  $SOURCE_DIR"
echo "to:"
echo "  $TARGET_DIR"

cp -p "$SOURCE_BIN" "$TARGET_BIN"
cp -p "$SOURCE_LIB_DIR"/*.dylib "$TARGET_LIB_DIR"/
cp -p "$SOURCE_MODEL" "$TARGET_MODEL"
chmod +x "$TARGET_BIN"

if command -v install_name_tool >/dev/null 2>&1; then
  if otool -l "$TARGET_BIN" | grep -Fq "$OLD_RPATH"; then
    install_name_tool -rpath "$OLD_RPATH" "$TARGET_LIB_DIR" "$TARGET_BIN"
  elif ! otool -l "$TARGET_BIN" | grep -Fq "$TARGET_LIB_DIR"; then
    install_name_tool -add_rpath "$TARGET_LIB_DIR" "$TARGET_BIN"
  fi
else
  echo "WARNING: install_name_tool not found; rpath was not updated." >&2
fi

echo
echo "SenseVoice.cpp SSD runtime:"
ls -lh "$TARGET_BIN" "$TARGET_MODEL"
du -sh "$TARGET_LIB_DIR"

echo
echo "Linked libraries:"
otool -L "$TARGET_BIN"

echo
echo "Runtime search paths:"
otool -l "$TARGET_BIN" | awk '
  /cmd LC_RPATH/ { in_rpath=1; next }
  in_rpath && /path / { print; in_rpath=0 }
'
