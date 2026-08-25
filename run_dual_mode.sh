#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before starting the app}"

exec uvicorn dual_mode.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7860}"

