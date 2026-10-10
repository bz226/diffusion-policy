#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
exec python "$STUDY_ROOT/code/verify_full_update.py" \
  --output "$STUDY_ROOT/runs/verification_full_update" --timeout-seconds 1710
