#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BACKEND=local
exec "${project_dir}/run_dual_mode.sh" local
