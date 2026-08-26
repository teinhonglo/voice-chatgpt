#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend="${1:-${BACKEND:-openai}}"
if [[ "$#" -gt 1 ]]; then
  echo "Usage: BACKEND=openai|local ./run_public.sh [openai|local]" >&2
  exit 2
fi

# shellcheck source=path.sh
source "${project_dir}/path.sh" "${backend}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before starting the public app}"

if [[ "${BACKEND}" == "local" ]]; then
  # shellcheck source=start_local_services.sh
  source "${project_dir}/start_local_services.sh"
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. See https://developers.cloudflare.com/tunnel/downloads/" >&2
  exit 1
fi

python_bin="${PYTHON_BIN}"
bind_host="${PUBLIC_BIND_HOST:-127.0.0.1}"
port="${PORT:-7860}"
origin_url="http://${bind_host}:${port}"
app_log="$(mktemp -t voice-chatgpt.XXXXXX.log)"
app_pid=""

cleanup() {
  if [[ -n "${app_pid}" ]] && kill -0 "${app_pid}" 2>/dev/null; then
    kill "${app_pid}" 2>/dev/null || true
    wait "${app_pid}" 2>/dev/null || true
  fi
  rm -f "${app_log}"
}
trap cleanup EXIT INT TERM

cd "${project_dir}"
"${python_bin}" -m uvicorn dual_mode.main:app \
  --host "${bind_host}" \
  --port "${port}" >"${app_log}" 2>&1 &
app_pid=$!

if ! "${python_bin}" - "${origin_url}/api/health" <<'PY'
import json
import sys
import time
import urllib.request

health_url = sys.argv[1]
for _ in range(30):
    try:
        with urllib.request.urlopen(health_url, timeout=1) as response:
            payload = json.load(response)
        if response.status == 200 and payload.get("ok") is True:
            raise SystemExit(0)
    except Exception:
        time.sleep(0.5)
raise SystemExit(1)
PY
then
  echo "The voice app did not become ready. Server log:" >&2
  tail -n 50 "${app_log}" >&2
  exit 1
fi

echo "Voice app is ready at ${origin_url}."

if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  echo "Starting the configured Cloudflare named tunnel."
  cloudflared tunnel --no-autoupdate run --token "${CLOUDFLARE_TUNNEL_TOKEN}"
else
  echo "Starting a temporary Quick Tunnel. Copy the trycloudflare.com URL printed below."
  echo "The URL is public and unauthenticated. Stop it with Ctrl+C when testing is complete."
  cloudflared tunnel --url "${origin_url}"
fi
