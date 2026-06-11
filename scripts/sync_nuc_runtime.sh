#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUC_HOST="${NUC_HOST:-192.168.31.196}"
NUC_USER="${NUC_USER:-jack}"
SERVICE_DIR="${SERVICE_DIR:-/srv/qwen3-asr-1p7b}"
MODE="sync-only"
SSH_OPTS=(
  -o ConnectTimeout=8
  -o StrictHostKeyChecking=accept-new
)
FILES=(
  nuc_asr_busy_proxy.Dockerfile
  nuc_deploy_qwen3_asr_1p7b.sh
  nuc_enable_asr_busy_proxy.sh
  nuc_faster_whisper_busy_proxy.py
  nuc_gpu_memory_guard.sh
  nuc_qwen3_asr_1p7b_proxy.py
  nuc_service_scheduler.py
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_nuc_runtime.sh [--sync-only|--deploy]

Defaults to --sync-only. Set NUC_HOST, NUC_USER, or SERVICE_DIR to override
the target. Passwords are never stored; use SSH keys, an interactive prompt,
or provide SSHPASS/NUC_SUDO_PASSWORD only in the invoking process environment.
EOF
}

while (($#)); do
  case "$1" in
    --sync-only) MODE="sync-only" ;;
    --deploy) MODE="deploy" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SSH=(ssh "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH=(sshpass -e ssh "${SSH_OPTS[@]}")
fi

REMOTE="${NUC_USER}@${NUC_HOST}"
STAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_STAGE="/tmp/whisper-captioner-nuc-${STAMP}"
LOCAL_STAGE="$(mktemp -d)"
trap 'rm -rf "${LOCAL_STAGE}"' EXIT

for file in "${FILES[@]}"; do
  cp "${SCRIPT_DIR}/${file}" "${LOCAL_STAGE}/${file}"
done
(
  cd "${LOCAL_STAGE}"
  shasum -a 256 "${FILES[@]}" > SHA256SUMS
)

"${SSH[@]}" "${REMOTE}" "mkdir -p '${REMOTE_STAGE}'"
tar -C "${LOCAL_STAGE}" -cf - "${FILES[@]}" SHA256SUMS |
  "${SSH[@]}" "${REMOTE}" "tar -C '${REMOTE_STAGE}' -xf -"
"${SSH[@]}" "${REMOTE}" "cd '${REMOTE_STAGE}' && sha256sum -c SHA256SUMS"

read -r -d '' REMOTE_SCRIPT <<'EOF' || true
set -euo pipefail
stage="$1"
service_dir="$2"
mode="$3"
stamp="$4"
backup_dir="${service_dir}/backups/${stamp}"

mkdir -p "${service_dir}" "${backup_dir}"
find "${service_dir}" -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.sh' -o -name 'Dockerfile*' \) \
  -exec cp -p {} "${backup_dir}/" \;
sha256sum "${service_dir}"/*.py "${service_dir}"/*.sh "${service_dir}"/Dockerfile* \
  > "${backup_dir}/SHA256SUMS.before" 2>/dev/null || true
for container in \
  nuc-service-scheduler \
  nuc-qwen3-asr-1p7b-proxy \
  nuc-qwen3-asr-1p7b-vllm \
  nuc-asr \
  nuc-asr-backend; do
  docker inspect "${container}" > "${backup_dir}/${container}.inspect.json" 2>/dev/null || true
done
docker images --digests --format '{{json .}}' > "${backup_dir}/docker-images.json"

install -m 0644 "${stage}/nuc_asr_busy_proxy.Dockerfile" "${service_dir}/Dockerfile.asr-busy-proxy"
install -m 0755 "${stage}/nuc_deploy_qwen3_asr_1p7b.sh" "${service_dir}/nuc_deploy_qwen3_asr_1p7b.sh"
install -m 0755 "${stage}/nuc_enable_asr_busy_proxy.sh" "${service_dir}/nuc_enable_asr_busy_proxy.sh"
install -m 0644 "${stage}/nuc_faster_whisper_busy_proxy.py" "${service_dir}/nuc_faster_whisper_busy_proxy.py"
install -m 0755 "${stage}/nuc_gpu_memory_guard.sh" "${service_dir}/nuc_gpu_memory_guard.sh"
install -m 0644 "${stage}/nuc_qwen3_asr_1p7b_proxy.py" "${service_dir}/nuc_qwen3_asr_1p7b_proxy.py"
install -m 0644 "${stage}/nuc_service_scheduler.py" "${service_dir}/nuc_service_scheduler.py"

if [[ "${mode}" = deploy ]]; then
  SERVICE_DIR="${service_dir}" ASR_IDLE_SECONDS=900 bash "${service_dir}/nuc_deploy_qwen3_asr_1p7b.sh"
  SERVICE_DIR="${service_dir}" bash "${service_dir}/nuc_enable_asr_busy_proxy.sh"
fi

sha256sum \
  "${service_dir}/nuc_faster_whisper_busy_proxy.py" \
  "${service_dir}/nuc_qwen3_asr_1p7b_proxy.py" \
  "${service_dir}/nuc_service_scheduler.py" \
  "${service_dir}/nuc_enable_asr_busy_proxy.sh" \
  "${service_dir}/nuc_gpu_memory_guard.sh" \
  > "${backup_dir}/SHA256SUMS.after"
echo "Backup: ${backup_dir}"
echo "Mode: ${mode}"
EOF

if [[ -n "${NUC_SUDO_PASSWORD:-}" ]]; then
  {
    printf '%s\n' "${NUC_SUDO_PASSWORD}"
    printf '%s\n' "${REMOTE_SCRIPT}"
  } | "${SSH[@]}" "${REMOTE}" \
    "sudo -S -p '' bash -s -- '${REMOTE_STAGE}' '${SERVICE_DIR}' '${MODE}' '${STAMP}'"
else
  printf '%s\n' "${REMOTE_SCRIPT}" |
    "${SSH[@]}" -t "${REMOTE}" \
      "sudo bash -s -- '${REMOTE_STAGE}' '${SERVICE_DIR}' '${MODE}' '${STAMP}'"
fi

"${SSH[@]}" "${REMOTE}" "rm -rf '${REMOTE_STAGE}'"
