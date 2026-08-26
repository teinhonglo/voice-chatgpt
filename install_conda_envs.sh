#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
requested_backend="${BACKEND:-all}"
cd "${project_dir}"

case "${requested_backend}" in
  openai)
    environment_specs=("environment.openai.yml:voice-chatgpt-openai")
    ;;
  local)
    environment_specs=("environment.local.yml:voice-chatgpt-local")
    ;;
  all)
    environment_specs=(
      "environment.openai.yml:voice-chatgpt-openai"
      "environment.local.yml:voice-chatgpt-local"
    )
    ;;
  *)
    echo "Unsupported BACKEND='${requested_backend}'. Use openai, local, or all." >&2
    exit 2
    ;;
esac

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_executable="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  conda_executable="$(command -v conda)"
else
  echo "Conda was not found. Install Miniconda/Anaconda or expose it through CONDA_EXE." >&2
  exit 1
fi

for environment_spec in "${environment_specs[@]}"; do
  environment_file="${environment_spec%%:*}"
  environment_name="${environment_spec#*:}"
  environment_path="${project_dir}/${environment_file}"
  if [[ -z "${environment_name}" ]]; then
    echo "The Conda environment name for ${environment_path} is empty." >&2
    exit 1
  fi

  if "${conda_executable}" run --name "${environment_name}" python --version >/dev/null 2>&1; then
    echo "Updating Conda environment: ${environment_name}"
    "${conda_executable}" env update --name "${environment_name}" --file "${environment_path}"
  else
    echo "Creating Conda environment: ${environment_name}"
    "${conda_executable}" env create --name "${environment_name}" --file "${environment_path}"
  fi
done

echo "Conda environment setup completed for BACKEND=${requested_backend}."
