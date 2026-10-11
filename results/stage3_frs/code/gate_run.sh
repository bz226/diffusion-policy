#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3_frs/code/environment.sh
exec python -B "$STUDY_ROOT/code/gate_frs.py" "$@"
