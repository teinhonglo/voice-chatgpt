#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BACKEND="${BACKEND:-openai}"

# shellcheck source=path.sh
source "${project_dir}/path.sh"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before starting the app}"

if [ "${BACKEND}" == "local" ]; then
  # shellcheck source=start_local_services.sh
  source "${project_dir}/start_local_services.sh"
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. See https://developers.cloudflare.com/tunnel/downloads/" >&2
  exit 1
fi

bind_host="${PUBLIC_BIND_HOST:-127.0.0.1}"
port="${PORT:-7860}"
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

cd "${project_dir}"
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
