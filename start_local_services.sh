#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="${project_dir}/docker-compose.local.yml"
local_llm_model="${LOCAL_LLM_MODEL:-qwen3:8b}"
local_embedding_model="${LOCAL_EMBEDDING_MODEL:-bge-m3}"

if [[ "${MANAGE_LOCAL_SERVICES:-1}" == "0" ]]; then
  export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
  export LOCAL_EMBEDDING_BASE_URL="${LOCAL_EMBEDDING_BASE_URL:-${LOCAL_LLM_BASE_URL}}"
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  echo "Using externally managed local LLM, embedding, and Qdrant services."
  return 0 2>/dev/null || exit 0
fi

if [[ "${MANAGE_LOCAL_SERVICES:-1}" != "1" ]]; then
  echo "MANAGE_LOCAL_SERVICES must be 0 or 1." >&2
  return 2 2>/dev/null || exit 2
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker with the Compose plugin is required for BACKEND=local." >&2
  exit 1
fi

docker compose -f "${compose_file}" up -d

ollama_ready=0
for _ in $(seq 1 45); do
  if docker compose -f "${compose_file}" exec -T ollama ollama list >/dev/null 2>&1; then
    ollama_ready=1
    break
  fi
  sleep 1
done
if [[ "${ollama_ready}" != "1" ]]; then
  echo "Ollama did not become ready within 45 seconds." >&2
  return 1 2>/dev/null || exit 1
fi
unset ollama_ready

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
