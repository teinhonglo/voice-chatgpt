#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
requested_backend="${BACKEND:-all}"
cd "${project_dir}"

case "${requested_backend}" in
  openai)
    environment_files=("environment.openai.yml")
    ;;
  local)
    environment_files=("environment.local.yml")
    ;;
  all)
    environment_files=("environment.openai.yml" "environment.local.yml")
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

for environment_file in "${environment_files[@]}"; do
  environment_path="${project_dir}/${environment_file}"
  environment_name="$(awk '/^name:/ {print $2; exit}' "${environment_path}")"
  if [[ -z "${environment_name}" ]]; then
    echo "Missing environment name in ${environment_path}." >&2
    exit 1
  fi

  if "${conda_executable}" run --name "${environment_name}" python --version >/dev/null 2>&1; then
    echo "Updating Conda environment: ${environment_name}"
    "${conda_executable}" env update --name "${environment_name}" --file "${environment_path}"
  else
    echo "Creating Conda environment: ${environment_name}"
    "${conda_executable}" env create --file "${environment_path}"
  fi
done

echo "Conda environment setup completed for BACKEND=${requested_backend}."
