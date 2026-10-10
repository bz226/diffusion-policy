#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
exec python "$STUDY_ROOT/code/supervise_recovery.py" "$@"
