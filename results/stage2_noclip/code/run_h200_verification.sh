#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
# The shell cap also terminates a stuck C/CUDA call before the 600-second allocation cap.
exec timeout --signal=TERM --kill-after=15s 570s \
  python "$STUDY_ROOT/code/verify_h200.py" \
  --output "$STUDY_ROOT/runs/verification_h200" --timeout-seconds 570
