#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/code/environment.sh
exec python -B "$GRADMEAN_OUTPUT/code/campaign.py" worker "$@"
