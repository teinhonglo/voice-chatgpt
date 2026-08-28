#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="${project_dir}/docker-compose.local.yml"
local_llm_model="${LOCAL_LLM_MODEL:-qwen3.5:9b}"
local_embedding_model="${LOCAL_EMBEDDING_MODEL:-bge-m3}"
local_duplex_model="${LOCAL_DUPLEX_MODEL:-openbmb/MiniCPM-o-4_5-GPTQ}"
local_duplex_timeout="${LOCAL_DUPLEX_STARTUP_TIMEOUT_SECONDS:-900}"
ref_audio_dir="${project_dir}/runtime/ref_audio"
ref_audio_path="${LOCAL_DUPLEX_REF_AUDIO:-${ref_audio_dir}/ref_minicpm_signature.wav}"
ref_audio_url="${LOCAL_DUPLEX_REF_AUDIO_URL:-https://raw.githubusercontent.com/OpenBMB/MiniCPM-o-Demo/main/assets/ref_audio/ref_minicpm_signature.wav}"

if [[ "${MANAGE_LOCAL_SERVICES:-1}" == "0" ]]; then
  export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
  export LOCAL_EMBEDDING_BASE_URL="${LOCAL_EMBEDDING_BASE_URL:-${LOCAL_LLM_BASE_URL}}"
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export LOCAL_DUPLEX_WS_URL="${LOCAL_DUPLEX_WS_URL:-ws://127.0.0.1:8099/v1/realtime}"
  export LOCAL_DUPLEX_REF_AUDIO="${ref_audio_path}"
  echo "Using externally managed local text LLM, speech LLM, embedding, and Qdrant services."
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

if [[ ! "${local_duplex_timeout}" =~ ^[0-9]+$ ]] || (( local_duplex_timeout < 30 )); then
  echo "LOCAL_DUPLEX_STARTUP_TIMEOUT_SECONDS must be an integer of at least 30." >&2
  return 2 2>/dev/null || exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the managed Local GPU services." >&2
  return 1 2>/dev/null || exit 1
fi

duplex_gpu_id="${LOCAL_DUPLEX_GPU_ID:-${LOCAL_GPU_ID:-0}}"
duplex_driver_version="$(
  nvidia-smi --id="${duplex_gpu_id}" \
    --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -n 1 | tr -d '[:space:]'
)"
if [[ "${duplex_driver_version}" =~ ^([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]]; then
  driver_major="${BASH_REMATCH[1]}"
  driver_minor="${BASH_REMATCH[2]}"
  driver_patch="${BASH_REMATCH[4]:-0}"
else
  echo "GPU ${duplex_gpu_id} is unavailable or its NVIDIA driver version could not be read." >&2
  return 1 2>/dev/null || exit 1
fi
if (( driver_major < 525 \
      || (driver_major == 525 && driver_minor < 60) \
      || (driver_major == 525 && driver_minor == 60 && driver_patch < 13) )); then
  echo "MiniCPM-o requires NVIDIA driver 525.60.13 or newer for CUDA 12.x compatibility; GPU ${duplex_gpu_id} uses ${duplex_driver_version}." >&2
  return 1 2>/dev/null || exit 1
fi

# The vLLM 0.26 CUDA 12.9 build can run on R525-R579 through NVIDIA's
# documented CUDA 12.x minor-version compatibility. The container image's
# conservative cuda>=12.9 metadata would otherwise reject those drivers before
# vLLM starts, so disable only that metadata check in the supported range.
if (( driver_major < 580 )); then
  export LOCAL_DUPLEX_DISABLE_CUDA_REQUIRE=true
  echo "Using GPU ${duplex_gpu_id} with NVIDIA driver ${duplex_driver_version} in CUDA 12.x compatibility mode."
else
  export LOCAL_DUPLEX_DISABLE_CUDA_REQUIRE=false
  echo "Using GPU ${duplex_gpu_id} with NVIDIA driver ${duplex_driver_version}."
fi
unset driver_major driver_minor driver_patch

if [[ ! -s "${ref_audio_path}" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download the MiniCPM-o reference voice." >&2
    return 1 2>/dev/null || exit 1
  fi
  mkdir -p "$(dirname "${ref_audio_path}")"
  echo "Downloading the official MiniCPM-o reference voice."
  curl -fL --retry 3 --connect-timeout 15 "${ref_audio_url}" -o "${ref_audio_path}"
fi

export LOCAL_DUPLEX_MODEL="${local_duplex_model}"
export LOCAL_DUPLEX_REF_AUDIO="${ref_audio_path}"
export LOCAL_DUPLEX_GPU_ID="${duplex_gpu_id}"

docker compose -f "${compose_file}" up -d --build

ollama_host_binding="$(
  docker compose -f "${compose_file}" port ollama 11434 2>/dev/null | head -n 1
)"
if [[ ! "${ollama_host_binding}" =~ ^127\.0\.0\.1:([0-9]+)$ ]]; then
  echo "Could not determine the Docker Ollama host port: ${ollama_host_binding:-no mapping returned}." >&2
  return 1 2>/dev/null || exit 1
fi
ollama_host_port="${BASH_REMATCH[1]}"
managed_ollama_base_url="http://127.0.0.1:${ollama_host_port}/v1"

duplex_host_binding="$(
  docker compose -f "${compose_file}" port minicpmo 8099 2>/dev/null | head -n 1
)"
if [[ ! "${duplex_host_binding}" =~ ^127\.0\.0\.1:([0-9]+)$ ]]; then
  echo "Could not determine the MiniCPM-o host port: ${duplex_host_binding:-no mapping returned}." >&2
  return 1 2>/dev/null || exit 1
fi
duplex_host_port="${BASH_REMATCH[1]}"
managed_duplex_http_url="http://127.0.0.1:${duplex_host_port}"
managed_duplex_ws_url="ws://127.0.0.1:${duplex_host_port}/v1/realtime"

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

duplex_ready=0
for _ in $(seq 1 "${local_duplex_timeout}"); do
  if curl --noproxy '*' -fsS "${managed_duplex_http_url}/health" >/dev/null 2>&1; then
    duplex_ready=1
    break
  fi
  if ! docker compose -f "${compose_file}" ps --status running --services | grep -qx minicpmo; then
    echo "MiniCPM-o stopped before becoming ready. Recent log:" >&2
    docker compose -f "${compose_file}" logs --tail 120 minicpmo >&2 || true
    return 1 2>/dev/null || exit 1
  fi
  sleep 1
done
if [[ "${duplex_ready}" != "1" ]]; then
  echo "MiniCPM-o did not become ready within ${local_duplex_timeout} seconds. Recent log:" >&2
  docker compose -f "${compose_file}" logs --tail 120 minicpmo >&2 || true
  return 1 2>/dev/null || exit 1
fi
unset duplex_ready

if [[ "${SKIP_LOCAL_MODEL_PULL:-0}" != "1" ]]; then
  if ! docker compose -f "${compose_file}" exec -T ollama ollama show "${local_llm_model}" >/dev/null 2>&1; then
    docker compose -f "${compose_file}" exec -T ollama ollama pull "${local_llm_model}"
  fi
  if ! docker compose -f "${compose_file}" exec -T ollama ollama show "${local_embedding_model}" >/dev/null 2>&1; then
    docker compose -f "${compose_file}" exec -T ollama ollama pull "${local_embedding_model}"
  fi
fi

export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-${managed_ollama_base_url}}"
export LOCAL_EMBEDDING_BASE_URL="${LOCAL_EMBEDDING_BASE_URL:-${managed_ollama_base_url}}"
export LOCAL_DUPLEX_WS_URL="${LOCAL_DUPLEX_WS_URL:-${managed_duplex_ws_url}}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
echo "Docker Ollama is available at ${managed_ollama_base_url}."
echo "Docker MiniCPM-o Full Duplex is available at ${managed_duplex_http_url}."
