#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_dir}"

stage=0
stop_stage=3
source_prompt=""
scenarios="experiments/auto_prompt/scenarios.example.jsonl"
exp_dir="experiments/auto_prompt/exp"
source_model="${AUTO_PROMPT_SOURCE_MODEL:-${OPENAI_TEXT_MODEL:-gpt-5.6-luna}}"
target_model="${AUTO_PROMPT_TARGET_MODEL:-${OPENAI_REALTIME_MODEL:-gpt-realtime-2}}"
judge_model="${AUTO_PROMPT_JUDGE_MODEL:-gpt-5.6-sol}"
reflection_model="${AUTO_PROMPT_REFLECTION_MODEL:-gpt-5.6-sol}"
max_metric_calls=60
reasoning_effort=low

help_message="Usage: $0 [options]

Stages:
  0  Generate source Text-LLM tutor references
  1  Evaluate direct prompt transfer on the dev split
  2  Optimize a Full-Duplex prompt with GEPA
  3  Compare direct transfer and adapted prompt on held-out test scenarios

Options:
  --stage <int>
  --stop_stage <int>
  --source_prompt <path>       Optional; defaults to dual_mode.core.DEFAULT_SYSTEM_PROMPT
  --scenarios <jsonl>
  --exp_dir <dir>
  --source_model <id>
  --target_model <id>
  --judge_model <id>
  --reflection_model <id>
  --max_metric_calls <int>
  --reasoning_effort <minimal|low|medium|high|xhigh>"

. ./parse_options.sh

if [[ $# -ne 0 ]]; then
  echo "$0: unexpected positional arguments: $*" >&2
  exit 2
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running the experiment}"
mkdir -p "${exp_dir}"

common=(
  --source-model "${source_model}"
  --target-model "${target_model}"
  --judge-model "${judge_model}"
  --reflection-model "${reflection_model}"
  --reasoning-effort "${reasoning_effort}"
)
prompt_arg=()
if [[ -n "${source_prompt}" ]]; then
  prompt_arg=(--source-prompt "${source_prompt}")
fi

if (( stage <= 0 && stop_stage >= 0 )); then
  python experiments/auto_prompt/experiment.py prepare \
    "${common[@]}" \
    "${prompt_arg[@]}" \
    --scenarios "${scenarios}" \
    --references "${exp_dir}/references.jsonl"
fi

if (( stage <= 1 && stop_stage >= 1 )); then
  baseline_prompt="${source_prompt}"
  if [[ -z "${baseline_prompt}" ]]; then
    python - <<'PY' > "${exp_dir}/source_system_prompt.txt"
from dual_mode.core import DEFAULT_SYSTEM_PROMPT
print(DEFAULT_SYSTEM_PROMPT)
PY
    baseline_prompt="${exp_dir}/source_system_prompt.txt"
  fi
  python experiments/auto_prompt/experiment.py evaluate \
    "${common[@]}" \
    --prompt "${baseline_prompt}" \
    --references "${exp_dir}/references.jsonl" \
    --split dev \
    --judge-repeats 1 \
    --output "${exp_dir}/direct_transfer_dev.jsonl"
fi

if (( stage <= 2 && stop_stage >= 2 )); then
  python experiments/auto_prompt/experiment.py optimize \
    "${common[@]}" \
    "${prompt_arg[@]}" \
    --references "${exp_dir}/references.jsonl" \
    --max-metric-calls "${max_metric_calls}" \
    --output-dir "${exp_dir}/gepa"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
  baseline_prompt="${source_prompt}"
  if [[ -z "${baseline_prompt}" ]]; then
    baseline_prompt="${exp_dir}/source_system_prompt.txt"
    if [[ ! -s "${baseline_prompt}" ]]; then
      python - <<'PY' > "${baseline_prompt}"
from dual_mode.core import DEFAULT_SYSTEM_PROMPT
print(DEFAULT_SYSTEM_PROMPT)
PY
    fi
  fi
  python experiments/auto_prompt/experiment.py evaluate \
    "${common[@]}" \
    --prompt "${baseline_prompt}" \
    --references "${exp_dir}/references.jsonl" \
    --split test \
    --judge-repeats 3 \
    --output "${exp_dir}/direct_transfer_test.jsonl"
  python experiments/auto_prompt/experiment.py evaluate \
    "${common[@]}" \
    --prompt "${exp_dir}/gepa/best_system_prompt.txt" \
    --references "${exp_dir}/references.jsonl" \
    --split test \
    --judge-repeats 3 \
    --output "${exp_dir}/gepa/final_test.jsonl"
fi

printf '\nFinal Full-Duplex prompt:\n  %s\n' "${exp_dir}/gepa/best_system_prompt.txt"
