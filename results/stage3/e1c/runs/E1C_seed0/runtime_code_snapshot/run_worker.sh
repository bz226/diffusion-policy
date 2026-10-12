#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/e1c/code/environment.sh
exec python -B "$E1C_OUTPUT/code/campaign.py" worker "$@"
