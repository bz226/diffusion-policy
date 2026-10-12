#!/usr/bin/env bash
# Bounded lightweight controller; no GPU. Do not allocate an extra Slurm CPU
# controller while G is running: that would consume its admission headroom.
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/code/environment.sh
exec python -B "$CRITIC_OUTPUT/code/campaign.py" run "$@"
