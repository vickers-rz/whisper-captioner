#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="${SERVICE_DIR:-/srv/qwen3-asr-1p7b}"
NETWORK_NAME="${NETWORK_NAME:-qwen3-asr-net}"
ASR_BACKEND_CONTAINER="${ASR_BACKEND_CONTAINER:-nuc-asr-backend}"
ASR_PROXY_CONTAINER="${ASR_PROXY_CONTAINER:-nuc-asr}"
ASR_IMAGE="${ASR_IMAGE:-fedirz/faster-whisper-server:latest-cuda}"
ASR_PROXY_IMAGE="${ASR_PROXY_IMAGE:-nuc-asr-busy-proxy:latest}"
UPSTREAM_PROXY_PORT="${UPSTREAM_PROXY_PORT:-8000}"
UPSTREAM_BACKEND_PORT="${UPSTREAM_BACKEND_PORT:-18000}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"

sudo mkdir -p "${SERVICE_DIR}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SOURCE_DIR}/nuc_faster_whisper_busy_proxy.py" ]]; then
  SOURCE_PROXY="${SOURCE_DIR}/nuc_faster_whisper_busy_proxy.py"
else
  SOURCE_PROXY="${SOURCE_DIR}/asr_busy_proxy.py"
fi
if [[ "${SOURCE_PROXY}" != "${SERVICE_DIR}/asr_busy_proxy.py" ]]; then
  sudo cp "${SOURCE_PROXY}" "${SERVICE_DIR}/asr_busy_proxy.py"
fi
if [[ -f "${SOURCE_DIR}/nuc_asr_busy_proxy.Dockerfile" ]]; then
  sudo cp "${SOURCE_DIR}/nuc_asr_busy_proxy.Dockerfile" "${SERVICE_DIR}/Dockerfile.asr-busy-proxy"
fi
sudo docker build -t "${ASR_PROXY_IMAGE}" -f "${SERVICE_DIR}/Dockerfile.asr-busy-proxy" "${SERVICE_DIR}"
sudo docker network create "${NETWORK_NAME}" >/dev/null 2>&1 || true

if sudo docker ps --format '{{.Names}}' | grep -qx "${ASR_PROXY_CONTAINER}"; then
  echo "Stopping current ${ASR_PROXY_CONTAINER} so host port ${UPSTREAM_PROXY_PORT} can move to the busy proxy..."
  sudo docker stop -t 20 "${ASR_PROXY_CONTAINER}" >/dev/null
fi

if sudo docker ps -a --format '{{.Names}}' | grep -qx "${ASR_BACKEND_CONTAINER}"; then
  echo "Removing stale ${ASR_BACKEND_CONTAINER} before recreating it..."
  sudo docker rm "${ASR_BACKEND_CONTAINER}" >/dev/null
fi

if sudo docker ps -a --format '{{.Names}}' | grep -qx "${ASR_PROXY_CONTAINER}"; then
  echo "Removing old direct-exposed ${ASR_PROXY_CONTAINER} container to recreate it as the front proxy..."
  sudo docker rm "${ASR_PROXY_CONTAINER}" >/dev/null
fi

sudo docker run -d \
  --name "${ASR_BACKEND_CONTAINER}" \
  --gpus all \
  --restart no \
  --network "${NETWORK_NAME}" \
  -p "${UPSTREAM_BACKEND_PORT}:8000" \
  -e WHISPER__DEVICE=cuda \
  -e WHISPER__COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE}" \
  -e WHISPER__MODEL="${WHISPER_MODEL}" \
  -e WHISPER__INFERENCE_DEVICE=auto \
  -e UVICORN_HOST=0.0.0.0 \
  -e UVICORN_PORT=8000 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v /srv/ai-models/whisper-cache:/root/.cache/huggingface \
  "${ASR_IMAGE}" \
  uv run uvicorn --factory faster_whisper_server.main:create_app

sudo docker run -d \
  --name "${ASR_PROXY_CONTAINER}" \
  --restart unless-stopped \
  --network host \
  -e SCHEDULER_URL=http://127.0.0.1:8010 \
  -e ASR_RESULT_DIR=/app/asr-results \
  -e ASR_STAGING_DIR=/app/asr-staging \
  -e UPSTREAM_BASE_URL="http://127.0.0.1:${UPSTREAM_BACKEND_PORT}" \
  -e UPSTREAM_HEALTH_URL="http://127.0.0.1:${UPSTREAM_BACKEND_PORT}/health" \
  -v "${SERVICE_DIR}:/app" \
  "${ASR_PROXY_IMAGE}" \
  uvicorn asr_busy_proxy:app --app-dir /app --host 0.0.0.0 --port "${UPSTREAM_PROXY_PORT}"

echo "NUC faster-whisper realtime proxy is exposed on :${UPSTREAM_PROXY_PORT}, backend on :${UPSTREAM_BACKEND_PORT}"
