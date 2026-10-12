#!/usr/bin/env bash
# Submitted only after terminal training/evaluation jobs; CPU Slurm allocation.
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/code/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec python -B "$CRITIC_OUTPUT/code/analyze_critic.py" "$@"
