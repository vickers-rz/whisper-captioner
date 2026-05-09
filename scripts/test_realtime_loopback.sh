#!/usr/bin/env bash
set -u

WHISPER_STREAM="${WHISPER_STREAM:-/opt/homebrew/bin/whisper-stream}"
FFMPEG="${FFMPEG:-/opt/homebrew/bin/ffmpeg}"
MODEL="${MODEL:-$HOME/Movies/whisper-captioner_APP_Resource/whisper-models/ggml-small.bin}"

RUN_PIDS=()
TMP_FILES=()

kill_tree() {
  local pid="$1"
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 0.2
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  for pid in "${RUN_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill_tree "$pid"
    fi
  done
  for path in "${TMP_FILES[@]:-}"; do
    rm -f "$path"
  done
}

trap 'echo; echo "Interrupted. Cleaning up..."; cleanup; exit 130' INT TERM
trap cleanup EXIT

run_limited() {
  local label="$1"
  local seconds="$2"
  local max_lines="$3"
  shift 3

  local output
  output="$(mktemp "${TMPDIR:-/tmp}/whisper-captioner-probe.XXXXXX")"
  TMP_FILES+=("$output")

  echo "$label"
  "$@" >"$output" 2>&1 &
  local pid=$!
  RUN_PIDS+=("$pid")

  (
    sleep "$seconds"
    if kill -0 "$pid" 2>/dev/null; then
      {
        echo
        echo "[probe timeout after ${seconds}s; stopping command]"
      } >>"$output"
      kill_tree "$pid"
    fi
  ) &
  local watcher=$!

  wait "$pid" 2>/dev/null || true
  kill "$watcher" 2>/dev/null || true
  wait "$watcher" 2>/dev/null || true

  sed -n "1,${max_lines}p" "$output"
  echo
}

echo "== Whisper Captioner realtime audio probe =="
echo "Date: $(date)"
echo "Shell: ${SHELL:-unknown}"
echo "whisper-stream: $WHISPER_STREAM"
echo "ffmpeg: $FFMPEG"
echo "model: $MODEL"
echo

if [[ ! -x "$WHISPER_STREAM" ]]; then
  echo "ERROR: whisper-stream not executable: $WHISPER_STREAM"
  exit 1
fi
if [[ ! -x "$FFMPEG" ]]; then
  echo "ERROR: ffmpeg not executable: $FFMPEG"
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Whisper model not found: $MODEL"
  exit 1
fi

run_limited \
  "== 1) AVFoundation devices via ffmpeg ==" \
  5 \
  80 \
  "$FFMPEG" -hide_banner -f avfoundation -list_devices true -i ""

SWIFT_FILE="$(mktemp "${TMPDIR:-/tmp}/whisper-captioner-audio-devices.XXXXXX.swift")"
TMP_FILES+=("$SWIFT_FILE")
cat >"$SWIFT_FILE" <<'SWIFT'
import AVFoundation

let session = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.microphone, .externalUnknown],
    mediaType: .audio,
    position: .unspecified
)
let devices = session.devices
print("Swift AVFoundation audio device count: \(devices.count)")
for (index, device) in devices.enumerated() {
    print("[\(index)] \(device.localizedName) uniqueID=\(device.uniqueID)")
}
SWIFT

run_limited \
  "== 2) Audio devices via Swift AVFoundation ==" \
  8 \
  80 \
  swift "$SWIFT_FILE"

run_limited \
  "== 3) whisper-stream default capture probe ==" \
  6 \
  80 \
  "$WHISPER_STREAM" \
    -c -1 \
    -m "$MODEL" \
    -t 1 \
    -l zh \
    --step 1000 \
    --length 1000 \
    --keep 100

echo "== 4) Try explicit capture IDs 0..8 =="
for id in 0 1 2 3 4 5 6 7 8; do
  run_limited \
    "-- capture id $id --" \
    5 \
    28 \
    "$WHISPER_STREAM" \
      -c "$id" \
      -m "$MODEL" \
      -t 1 \
      -l zh \
      --step 1000 \
      --length 1000 \
      --keep 100
done

cat <<'EOF'
== Reading the result ==
- This script intentionally stops whisper-stream probes after a few seconds. If a probe times out after model init, that may mean the device opened successfully and was waiting for audio.
- If every probe says "found 0 capture devices", the current app/Terminal cannot see macOS audio input devices.
- On macOS, grant Microphone permission to the app you run this script from:
  System Settings -> Privacy & Security -> Microphone -> Terminal / iTerm / Codex.
- If Swift or ffmpeg lists Loopback/Whisper Captions as [N], use N in the app's "Loopback 输入" dropdown.
- If SoundSource sends Chrome/player to Loopback but you also want to hear it, route to a SoundSource output group or macOS multi-output device containing both speakers/headphones and Loopback.
EOF
