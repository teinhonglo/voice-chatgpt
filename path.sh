#!/usr/bin/env bash

# This file must be sourced. The first argument overrides the BACKEND variable.
requested_backend="${1:-${BACKEND:-openai}}"

case "${requested_backend}" in
  openai)
    target_conda_env="${OPENAI_CONDA_ENV:-voice-chatgpt-openai}"
    ;;
  local)
    target_conda_env="${LOCAL_CONDA_ENV:-voice-chatgpt-local}"
    ;;
  *)
    echo "Unsupported BACKEND='${requested_backend}'. Use BACKEND=openai or BACKEND=local." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_executable="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  conda_executable="$(command -v conda)"
else
  echo "Conda was not found. Install Miniconda/Anaconda or expose it through CONDA_EXE." >&2
  return 1 2>/dev/null || exit 1
fi

conda_base="$("${conda_executable}" info --base)"
conda_profile="${conda_base}/etc/profile.d/conda.sh"
if [[ ! -r "${conda_profile}" ]]; then
  echo "Conda shell integration was not found at ${conda_profile}." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "${conda_profile}"
if ! conda activate "${target_conda_env}"; then
  echo "Could not activate '${target_conda_env}'. Create it from environment.${requested_backend}.yml first." >&2
  return 1 2>/dev/null || exit 1
fi

export BACKEND="${requested_backend}"
export VOICE_CHATGPT_CONDA_ENV="${target_conda_env}"
export PYTHON_BIN="$(command -v python)"

unset requested_backend target_conda_env conda_executable conda_base conda_profile
