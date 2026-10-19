#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/code/environment.sh
exec python -B "$DL1_OUTPUT/code/continue_after_gates.py" "$@"
