#!/usr/bin/env bash
set -eo pipefail

stage=-1
stop_stage=3
backend=openai
exp_config=prompt_adaptation/config.json
trajectory_source=prompt_adaptation/private_data/SR_prompt_adaptation_trajectories.jsonl
overwrite_audio=false

help_message="Usage: $0 [options]

Options:
  --stage <int>                 Start stage (default: -1)
  --stop-stage <int>            Stop stage (default: 3)
  --backend <openai|local>      Backend passed to path.sh (default: openai)
  --exp-config <path>           Prompt-adaptation JSON config
  --trajectory-source <path>    Curated trajectory JSONL to prepare
  --overwrite-audio <true|false> Regenerate existing PCM files

Stages:
  -1  Install prompt-adaptation dependencies
   0  Prepare/verify trajectories and initial prompt
   1  Generate fixed learner audio
   2  Run GEPA prompt adaptation
   3  Run held-out test evaluation
"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

. ./parse_options.sh

export BACKEND="$backend"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. ./path.sh
set -u

if ! [[ "$stage" =~ ^-?[0-9]+$ && "$stop_stage" =~ ^-?[0-9]+$ ]]; then
  echo "$0: --stage and --stop-stage must be integers" >&2
  exit 1
fi
if (( stage > stop_stage )); then
  echo "$0: --stage must be <= --stop-stage" >&2
  exit 1
fi
if [[ ! -f "$exp_config" ]]; then
  echo "$0: missing experiment config: $exp_config" >&2
  exit 1
fi

mapfile -t cfg_values < <(
  python - "$exp_config" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
cfg = json.loads(path.read_text(encoding="utf-8"))
for key in ("data_path", "source_prompt", "output_dir"):
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"Missing or invalid {key!r} in {path}")
    print(value)
PY
)

data_path="${cfg_values[0]}"
source_prompt="${cfg_values[1]}"
output_dir="${cfg_values[2]}"
log_dir="$output_dir/logs"
mkdir -p "$log_dir"

if (( stage <= -1 && stop_stage >= -1 )); then
  echo "===== Stage -1: install dependencies ====="
  python -m pip install -r requirements.prompt_adaptation.txt
fi

if (( stage <= 0 && stop_stage >= 0 )); then
  echo "===== Stage 0: prepare data and prompt ====="
  mkdir -p "$(dirname "$data_path")"

  if [[ -f "$trajectory_source" ]]; then
    if [[ "$trajectory_source" != "$data_path" ]]; then
      cp -f "$trajectory_source" "$data_path"
      echo "Prepared trajectories: $trajectory_source -> $data_path"
    else
      echo "Using trajectories: $data_path"
    fi
  elif [[ -f "$data_path" ]]; then
    echo "Trajectory source not found; reusing existing: $data_path"
  else
    echo "$0: no trajectory JSONL found." >&2
    echo "Expected either:" >&2
    echo "  $trajectory_source" >&2
    echo "or:" >&2
    echo "  $data_path" >&2
    exit 1
  fi

  if [[ ! -f "$source_prompt" ]]; then
    echo "$0: missing initial prompt: $source_prompt" >&2
    exit 1
  fi

  echo "Initial prompt: $source_prompt"
  python - "$data_path" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

counts = Counter()
total = 0
with Path(sys.argv[1]).open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        counts[str(item.get("split", "unknown"))] += 1
        total += 1

print(f"Trajectories: total={total}, " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
PY
fi

if (( stop_stage >= 1 && stage <= 3 )); then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "$0: OPENAI_API_KEY is not set" >&2
    exit 1
  fi
fi

if (( stage <= 1 && stop_stage >= 1 )); then
  echo "===== Stage 1: prepare fixed learner audio ====="
  cmd=(python -m prompt_adaptation.prepare_audio --config "$exp_config")
  if [[ "$overwrite_audio" == "true" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}" 2>&1 | tee "$log_dir/stage1_prepare_audio.log"
fi

if (( stage <= 2 && stop_stage >= 2 )); then
  echo "===== Stage 2: GEPA prompt adaptation ====="
  python -m prompt_adaptation.run     --config "$exp_config"     2>&1 | tee "$log_dir/stage2_prompt_adaptation.log"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
  echo "===== Stage 3: held-out test evaluation ====="
  python -m prompt_adaptation.evaluate     --config "$exp_config"     2>&1 | tee "$log_dir/stage3_test_evaluation.log"
fi

echo
echo "Done."
echo "Final prompt:        $output_dir/system_prompt.full_duplex.txt"
echo "Optimization report: $output_dir/optimization_report.json"
echo "Test JSON:           $output_dir/test_evaluation.json"
echo "Readable report:     $output_dir/test_report.md"
echo "Logs:                $log_dir/"
echo "GEPA intermediates:  $output_dir/gepa/"
