#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend="${1:-${BACKEND:-openai}}"
if [[ "$#" -gt 1 ]]; then
  echo "Usage: BACKEND=openai|local ./run_dual_mode.sh [openai|local]" >&2
  exit 2
fi

# shellcheck source=path.sh
source "${project_dir}/path.sh" "${backend}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before starting the app}"

if [[ "${BACKEND}" == "local" ]]; then
  # shellcheck source=start_local_services.sh
  source "${project_dir}/start_local_services.sh"
fi

cd "${project_dir}"
exec "${PYTHON_BIN}" -m uvicorn dual_mode.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7860}"
