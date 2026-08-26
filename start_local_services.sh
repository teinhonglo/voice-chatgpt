#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="${project_dir}/docker-compose.local.yml"
local_llm_model="${LOCAL_LLM_MODEL:-qwen3:8b}"
local_embedding_model="${LOCAL_EMBEDDING_MODEL:-bge-m3}"

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker with the Compose plugin is required for BACKEND=local." >&2
  exit 1
fi

docker compose -f "${compose_file}" up -d

if [[ "${SKIP_LOCAL_MODEL_PULL:-0}" != "1" ]]; then
  if ! docker compose -f "${compose_file}" exec -T ollama ollama show "${local_llm_model}" >/dev/null 2>&1; then
    docker compose -f "${compose_file}" exec -T ollama ollama pull "${local_llm_model}"
  fi
  if ! docker compose -f "${compose_file}" exec -T ollama ollama show "${local_embedding_model}" >/dev/null 2>&1; then
    docker compose -f "${compose_file}" exec -T ollama ollama pull "${local_embedding_model}"
  fi
fi

export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
export LOCAL_EMBEDDING_BASE_URL="${LOCAL_EMBEDDING_BASE_URL:-http://127.0.0.1:11434/v1}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
