#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="${project_dir}/docker-compose.local.yml"
local_llm_model="${LOCAL_LLM_MODEL:-qwen3:8b}"
local_embedding_model="${LOCAL_EMBEDDING_MODEL:-bge-m3}"

docker compose -f "${compose_file}" up -d
docker compose -f "${compose_file}" exec ollama ollama pull "${local_llm_model}"
docker compose -f "${compose_file}" exec ollama ollama pull "${local_embedding_model}"

export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
export LOCAL_EMBEDDING_BASE_URL="${LOCAL_EMBEDDING_BASE_URL:-http://127.0.0.1:11434/v1}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
exec "${project_dir}/run_dual_mode.sh"
