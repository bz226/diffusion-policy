#!/usr/bin/env bash
# One allocation owns only its own run; the supervisor never signals other jobs.
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/code/environment.sh
exec python -B "$STUDY_ROOT/code/monitor.py" worker "$@"
