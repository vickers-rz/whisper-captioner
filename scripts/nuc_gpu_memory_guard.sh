#!/usr/bin/env bash
set -euo pipefail

NUC_HOST="${NUC_HOST:-192.168.31.196}"
NUC_USER="${NUC_USER:-jack}"
SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=8
)

QWEN_VLLM_CONTAINER="${QWEN_VLLM_CONTAINER:-nuc-qwen3-asr-1p7b-vllm}"
QWEN_PROXY_CONTAINER="${QWEN_PROXY_CONTAINER:-nuc-qwen3-asr-1p7b-proxy}"
ASR_CONTAINER="${ASR_CONTAINER:-nuc-asr}"
QWEN_SERVICE_DIR="${QWEN_SERVICE_DIR:-/srv/qwen3-asr-1p7b}"
MIN_FREE_MB="${MIN_FREE_MB:-1800}"
ASR_IDLE_SECONDS="${ASR_IDLE_SECONDS:-360}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/nuc_gpu_memory_guard.sh status
  bash scripts/nuc_gpu_memory_guard.sh unload-qwen
  bash scripts/nuc_gpu_memory_guard.sh start-qwen
  bash scripts/nuc_gpu_memory_guard.sh restart-asr
  bash scripts/nuc_gpu_memory_guard.sh auto-clean
  bash scripts/nuc_gpu_memory_guard.sh prep-asr
  bash scripts/nuc_gpu_memory_guard.sh unload-all
  bash scripts/nuc_gpu_memory_guard.sh start-all
  bash scripts/nuc_gpu_memory_guard.sh idle-watch

Environment overrides:
  NUC_HOST=192.168.31.196
  NUC_USER=jack
  MIN_FREE_MB=1800
  ASR_IDLE_SECONDS=360

Commands:
  status
    Print GPU memory, GPU utilization, and current container status.

  unload-qwen
    Stop the Qwen3-ASR 1.7B proxy/backend containers to free GPU memory.

  start-qwen
    Start existing Qwen3-ASR 1.7B containers when possible, otherwise redeploy them.

  restart-asr
    Restart the faster-whisper container only.

  auto-clean
    If free GPU memory is below MIN_FREE_MB, stop the Qwen3-ASR 1.7B containers.
    This is intended as the safest first cleanup step before touching faster-whisper.

  prep-asr
    Unload Qwen if free GPU memory is below MIN_FREE_MB, then check faster-whisper
    health and restart only that service if needed.

  unload-all
    Stop both Qwen3-ASR 1.7B and faster-whisper containers to release GPU memory.

  start-all
    Start faster-whisper and Qwen3-ASR 1.7B containers.

  idle-watch
    Run a lightweight remote loop that stops Qwen when it stays healthy and idle,
    and restarts faster-whisper only if its health check fails after Qwen cleanup.
EOF
}

remote() {
  ssh "${SSH_OPTS[@]}" "${NUC_USER}@${NUC_HOST}" "$@"
}

remote_bash() {
  remote "bash -lc $(printf '%q' "$1")"
}

stop_or_kill() {
  local container_name="$1"
  local timeout_seconds="${2:-12}"
  remote_bash "
    if ! docker ps -a --format '{{.Names}}' | grep -qx '${container_name}'; then
      echo '${container_name}: not found'
      exit 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx '${container_name}'; then
      echo '${container_name}: already stopped'
      exit 0
    fi
    if docker stop -t '${timeout_seconds}' '${container_name}' >/dev/null 2>&1; then
      echo '${container_name}: stopped'
      exit 0
    fi
    echo '${container_name}: stop timed out, escalating to kill'
    docker kill '${container_name}' >/dev/null 2>&1 || true
    if docker ps --format '{{.Names}}' | grep -qx '${container_name}'; then
      echo '${container_name}: still running after kill' >&2
      exit 1
    fi
    echo '${container_name}: killed'
  "
}

status_cmd() {
  remote_bash "
    echo '=== GPU ==='
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    echo
    echo '=== Containers ==='
    docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E '${QWEN_VLLM_CONTAINER}|${QWEN_PROXY_CONTAINER}|${ASR_CONTAINER}' || true
    echo
    echo '=== Health ==='
    printf '8000 faster-whisper: '
    curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null && echo ok || echo fail
    printf '8001 qwen proxy: '
    curl -fsS --max-time 5 http://127.0.0.1:8001/healthz >/dev/null && echo ok || echo fail
    printf '8002 qwen backend: '
    curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null && echo ok || echo fail
  "
}

unload_qwen_cmd() {
  stop_or_kill "${QWEN_PROXY_CONTAINER}"
  stop_or_kill "${QWEN_VLLM_CONTAINER}"
  remote_bash "
    echo 'Stopped Qwen3-ASR 1.7B containers.'
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  "
}

start_qwen_cmd() {
  remote_bash "
    started=1
    if docker ps -a --format '{{.Names}}' | grep -qx '${QWEN_VLLM_CONTAINER}'; then
      docker start '${QWEN_VLLM_CONTAINER}' >/dev/null 2>&1 || started=0
    else
      started=0
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx '${QWEN_PROXY_CONTAINER}'; then
      docker start '${QWEN_PROXY_CONTAINER}' >/dev/null 2>&1 || started=0
    else
      started=0
    fi
    if [ \"\${started}\" = '1' ] \
      && docker ps --format '{{.Names}}' | grep -qx '${QWEN_VLLM_CONTAINER}' \
      && docker ps --format '{{.Names}}' | grep -qx '${QWEN_PROXY_CONTAINER}'; then
      echo 'Started existing Qwen3-ASR 1.7B containers.'
      exit 0
    fi
    if [ -x '${QWEN_SERVICE_DIR}/nuc_deploy_qwen3_asr_1p7b.sh' ]; then
      bash '${QWEN_SERVICE_DIR}/nuc_deploy_qwen3_asr_1p7b.sh'
    elif [ -f '${QWEN_SERVICE_DIR}/nuc_deploy_qwen3_asr_1p7b.sh' ]; then
      bash '${QWEN_SERVICE_DIR}/nuc_deploy_qwen3_asr_1p7b.sh'
    elif [ -f '${QWEN_SERVICE_DIR}/deploy.sh' ]; then
      bash '${QWEN_SERVICE_DIR}/deploy.sh'
    else
      echo 'No Qwen deploy script found under ${QWEN_SERVICE_DIR}.' >&2
      exit 1
    fi
  "
}

restart_asr_cmd() {
  remote_bash "
    docker restart '${ASR_CONTAINER}' >/dev/null
    echo 'Restarted ${ASR_CONTAINER}.'
    sleep 2
    printf '8000 faster-whisper: '
    curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null && echo ok || echo fail
  "
}

auto_clean_cmd() {
  local cleanup_needed
  cleanup_needed="$(remote_bash "
    free_mb=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo \"Current free GPU memory: \${free_mb} MiB\"
    if [ \"\${free_mb}\" -ge '${MIN_FREE_MB}' ]; then
      echo no
    else
      echo yes
    fi
  " | tail -n 1)"
  if [ "${cleanup_needed}" = "yes" ]; then
    echo "Free GPU memory is below threshold; stopping Qwen3-ASR 1.7B containers."
    unload_qwen_cmd
  else
    echo "No cleanup needed."
  fi
}

prep_asr_cmd() {
  auto_clean_cmd
  remote_bash "
    if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null; then
      echo 'faster-whisper is already healthy.'
      exit 0
    fi
    echo 'faster-whisper health check failed; restarting only ${ASR_CONTAINER}.'
    docker restart '${ASR_CONTAINER}' >/dev/null
    sleep 2
    printf '8000 faster-whisper: '
    curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null && echo ok || echo fail
  "
}

unload_all_cmd() {
  stop_or_kill "${QWEN_PROXY_CONTAINER}"
  stop_or_kill "${QWEN_VLLM_CONTAINER}"
  stop_or_kill "${ASR_CONTAINER}"
  remote_bash "
    echo 'Stopped Qwen3-ASR 1.7B and faster-whisper containers.'
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  "
}

start_all_cmd() {
  remote_bash "
    if docker ps -a --format '{{.Names}}' | grep -qx '${ASR_CONTAINER}'; then
      docker start '${ASR_CONTAINER}' >/dev/null 2>&1 || true
    fi
    printf '8000 faster-whisper: '
    curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null && echo ok || echo fail
  "
  start_qwen_cmd
}

idle_watch_cmd() {
  remote_bash "
    set -euo pipefail
    echo 'Starting one-shot idle watch...'
    free_mb=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo \"Current free GPU memory: \${free_mb} MiB\"

    qwen_active=0
    if docker ps --format '{{.Names}}' | grep -qx '${QWEN_VLLM_CONTAINER}'; then
      qwen_active=1
    fi

    if [ \"\${qwen_active}\" = '1' ]; then
      echo 'Qwen3-ASR 1.7B containers are active; stopping them after this check to avoid idle GPU residency.'
      if docker stop -t 12 '${QWEN_PROXY_CONTAINER}' >/dev/null 2>&1; then :; else docker kill '${QWEN_PROXY_CONTAINER}' >/dev/null 2>&1 || true; fi
      if docker stop -t 12 '${QWEN_VLLM_CONTAINER}' >/dev/null 2>&1; then :; else docker kill '${QWEN_VLLM_CONTAINER}' >/dev/null 2>&1 || true; fi
    else
      echo 'Qwen3-ASR 1.7B is already stopped.'
    fi

    if docker ps --format '{{.Names}}' | grep -qx '${ASR_CONTAINER}'; then
      echo 'faster-whisper container is present; relying on its internal idle offload window.'
      if ! curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null; then
        echo 'faster-whisper health check failed; restarting it.'
        docker restart '${ASR_CONTAINER}' >/dev/null
      fi
    else
      echo 'faster-whisper container is not running.'
    fi

    echo 'Post-watch GPU status:'
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  "
}

main() {
  if [ "$#" -ne 1 ]; then
    usage
    exit 1
  fi

  case "$1" in
    status) status_cmd ;;
    unload-qwen) unload_qwen_cmd ;;
    start-qwen) start_qwen_cmd ;;
    restart-asr) restart_asr_cmd ;;
    auto-clean) auto_clean_cmd ;;
    prep-asr) prep_asr_cmd ;;
    unload-all) unload_all_cmd ;;
    start-all) start_all_cmd ;;
    idle-watch) idle_watch_cmd ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: $1" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
