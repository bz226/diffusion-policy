#!/usr/bin/env bash
# Postprocessing only: CPU threading changes never enter collection workers.
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/code/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec python -B "$GRADMEAN_OUTPUT/analyze.py" "$@"
