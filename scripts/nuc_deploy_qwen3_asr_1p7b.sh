#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="${SERVICE_DIR:-/srv/qwen3-asr-1p7b}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
QWEN_ASR_IMAGE="${QWEN_ASR_IMAGE:-qwenllm/qwen3-asr:latest}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
NETWORK_NAME="${NETWORK_NAME:-qwen3-asr-net}"
SCHEDULER_CONTAINER="${SCHEDULER_CONTAINER:-nuc-service-scheduler}"
ASR_IDLE_SECONDS="${ASR_IDLE_SECONDS:-900}"
QWEN_MAX_DIRECT_UPLOAD_MB="${QWEN_MAX_DIRECT_UPLOAD_MB:-64}"
QWEN_CHUNK_SECONDS="${QWEN_CHUNK_SECONDS:-30}"
QWEN_CHUNK_OVERLAP_SECONDS="${QWEN_CHUNK_OVERLAP_SECONDS:-2}"
QWEN_EMPTY_RETRY_MIN_DBFS="${QWEN_EMPTY_RETRY_MIN_DBFS:--50}"

sudo mkdir -p "${SERVICE_DIR}"
sudo cp "${SCRIPT_DIR}/nuc_qwen3_asr_1p7b_proxy.py" "${SERVICE_DIR}/proxy.py"
sudo cp "${SCRIPT_DIR}/nuc_service_scheduler.py" "${SERVICE_DIR}/scheduler.py"
sudo docker network create "${NETWORK_NAME}" >/dev/null 2>&1 || true

sudo docker rm -f nuc-qwen3-asr-7b-vllm >/dev/null 2>&1 || true
sudo docker rm -f nuc-qwen3-asr-7b-proxy >/dev/null 2>&1 || true
sudo docker rm -f nuc-qwen3-asr-1p7b-vllm >/dev/null 2>&1 || true
sudo docker rm -f nuc-qwen3-asr-1p7b-proxy >/dev/null 2>&1 || true
sudo docker rm -f "${SCHEDULER_CONTAINER}" >/dev/null 2>&1 || true

sudo docker run -d \
  --name "${SCHEDULER_CONTAINER}" \
  --gpus all \
  --restart unless-stopped \
  --network "${NETWORK_NAME}" \
  -p 127.0.0.1:8010:8010 \
  --add-host host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "${SERVICE_DIR}:/app" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e ASR_HEALTH_URL=http://host.docker.internal:8000/health \
  -e ASR_BUSY_URL=http://host.docker.internal:8000/busy \
  -e ASR_BACKEND_HEALTH_URL=http://nuc-asr-backend:8000/health \
  -e ASR_IDLE_SECONDS="${ASR_IDLE_SECONDS}" \
  python:3.11-slim \
  bash -lc "pip install --no-cache-dir fastapi uvicorn httpx docker && uvicorn scheduler:app --app-dir /app --host 0.0.0.0 --port 8010"

sudo docker run -d \
  --name nuc-qwen3-asr-1p7b-vllm \
  --gpus all \
  --restart no \
  --network "${NETWORK_NAME}" \
  -p 8002:8000 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v /srv/ai-models/qwen3-asr-cache:/root/.cache/huggingface \
  "${QWEN_ASR_IMAGE}" \
  qwen-asr-serve "${MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 1

sudo docker run -d \
  --name nuc-qwen3-asr-1p7b-proxy \
  --gpus all \
  --restart unless-stopped \
  --network "${NETWORK_NAME}" \
  -e SCHEDULER_URL=http://nuc-service-scheduler:8010 \
  -e UPSTREAM_URL=http://nuc-qwen3-asr-1p7b-vllm:8000/v1/audio/transcriptions \
  -e QWEN_MAX_DIRECT_UPLOAD_MB="${QWEN_MAX_DIRECT_UPLOAD_MB}" \
  -e QWEN_CHUNK_SECONDS="${QWEN_CHUNK_SECONDS}" \
  -e QWEN_CHUNK_OVERLAP_SECONDS="${QWEN_CHUNK_OVERLAP_SECONDS}" \
  -e QWEN_EMPTY_RETRY_MIN_DBFS="${QWEN_EMPTY_RETRY_MIN_DBFS}" \
  -p 8001:8000 \
  -v "${SERVICE_DIR}:/app" \
  python:3.11-slim \
  bash -lc "pip install --no-cache-dir fastapi uvicorn httpx python-multipart && uvicorn proxy:app --app-dir /app --host 0.0.0.0 --port 8000"

echo "Qwen3-ASR 1.7B proxy exposed on :8001"
