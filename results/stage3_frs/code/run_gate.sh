#!/usr/bin/env bash
# One bounded allocation; no training, new trajectories, or retries.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3_frs/code/environment.sh
exec python -B "$STUDY_ROOT/code/gate_frs.py" --device both --timeout-seconds 3540
