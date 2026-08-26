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

cleanup() {
  if [[ -n "${app_pid}" ]] && kill -0 "${app_pid}" 2>/dev/null; then
    kill "${app_pid}" 2>/dev/null || true
    wait "${app_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${project_dir}"
PYTHONUNBUFFERED=1 python -m uvicorn dual_mode.main:app \
  --host "${bind_host}" \
  --port "${port}" &
app_pid=$!

if ! python - "${origin_url}/api/health" "${app_pid}" <<'PY'
import json
import os
import sys
import time
import urllib.request

health_url = sys.argv[1]
app_pid = int(sys.argv[2])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
last_error = "no response"
for _ in range(30):
    try:
        os.kill(app_pid, 0)
    except ProcessLookupError:
        print("Uvicorn exited before the health check succeeded.", file=sys.stderr)
        raise SystemExit(1)
    try:
        with opener.open(health_url, timeout=1) as response:
            payload = json.load(response)
        if response.status == 200 and payload.get("ok") is True:
            raise SystemExit(0)
        last_error = f"unexpected response: HTTP {response.status} {payload!r}"
    except Exception as error:
        last_error = f"{type(error).__name__}: {error}"
        time.sleep(0.5)
print(f"Health check failed for {health_url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
then
  echo "The voice app did not become ready. Review the Uvicorn error above." >&2
  exit 1
fi

echo "Voice app is ready at ${origin_url}."

if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  echo "Starting the configured Cloudflare named tunnel."
  TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN}" cloudflared tunnel --no-autoupdate run
else
  echo "Starting a temporary Quick Tunnel. Copy the trycloudflare.com URL printed below."
  echo "The URL is public and unauthenticated. Stop it with Ctrl+C when testing is complete."
  cloudflared tunnel --url "${origin_url}"
fi
