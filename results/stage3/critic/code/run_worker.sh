#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/code/environment.sh
exec python -B "$CRITIC_OUTPUT/code/campaign.py" worker "$@"
