#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_dir}"

backend="${BACKEND:-openai}"
gpuid="${GPUID:-0}"
port="${PORT:-7860}"
help_message="Usage: $0 [options]

Options:
  --backend <openai|local>  Backend to expose (default: ${backend})
  --gpuid <id>              NVIDIA GPU index for Local Ollama/MiniCPM-o (default: ${gpuid})
  --port <1-65535>          Web server port (default: ${port})
  --help                    Show this help message"

. ./parse_options.sh

if [[ $# -ne 0 ]]; then
  echo "$0: unexpected positional arguments: $*" >&2
  exit 2
fi
if [[ "${backend}" != "openai" && "${backend}" != "local" ]]; then
  echo "$0: --backend must be openai or local." >&2
  exit 2
fi
if [[ ! "${gpuid}" =~ ^[0-9]+$ ]]; then
  echo "$0: --gpuid must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "$0: --port must be an integer from 1 to 65535." >&2
  exit 2
fi

export BACKEND="${backend}"
export GPUID="${gpuid}"
export LOCAL_GPU_ID="${gpuid}"
export LOCAL_DUPLEX_GPU_ID="${LOCAL_DUPLEX_GPU_ID:-${gpuid}}"
export PORT="${port}"

# shellcheck source=path.sh
source ./path.sh
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before starting the app}"

if [ "${BACKEND}" == "local" ]; then
  # shellcheck source=start_local_services.sh
  source ./start_local_services.sh
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. See https://developers.cloudflare.com/tunnel/downloads/" >&2
  exit 1
fi

bind_host="${PUBLIC_BIND_HOST:-127.0.0.1}"
origin_url="http://${bind_host}:${port}"
app_pid=""
tunnel_pid=""

cleanup() {
  if [[ -n "${app_pid}" ]] && kill -0 "${app_pid}" 2>/dev/null; then
    kill "${app_pid}" 2>/dev/null || true
    wait "${app_pid}" 2>/dev/null || true
  fi
  if [[ -n "${tunnel_pid}" ]] && kill -0 "${tunnel_pid}" 2>/dev/null; then
    kill "${tunnel_pid}" 2>/dev/null || true
    wait "${tunnel_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONUNBUFFERED=1 python -m uvicorn dual_mode.main:app \
  --host "${bind_host}" \
  --port "${port}" &
app_pid=$!

if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  echo "Starting the configured Cloudflare named tunnel."
  TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN}" \
    TUNNEL_TRANSPORT_PROTOCOL=http2 \
    cloudflared tunnel --no-autoupdate run &
else
  echo "Starting a temporary Quick Tunnel. Copy the trycloudflare.com URL printed below."
  echo "The URL is public and unauthenticated. Stop it with Ctrl+C when testing is complete."
  TUNNEL_TRANSPORT_PROTOCOL=http2 cloudflared tunnel --url "${origin_url}" &
fi
tunnel_pid=$!

status=0
wait -n "${app_pid}" "${tunnel_pid}" || status=$?
exit "${status}"
