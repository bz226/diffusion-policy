#!/usr/bin/env bash
# One Slurm allocation, one approved run. Slurm B:TERM reaches the exec'd supervisor.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
exec python "$STUDY_ROOT/code/supervise_run.py" "$@"
