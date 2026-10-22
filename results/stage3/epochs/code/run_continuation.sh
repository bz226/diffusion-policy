#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/epochs/code/environment.sh
exec python -B "$EP_OUTPUT/code/continue_after_gates.py" "$@"
