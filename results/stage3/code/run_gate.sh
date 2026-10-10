#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/code/environment.sh
exec python -B "$STUDY_ROOT/code/gate_stage3.py" --timeout-seconds 2640
