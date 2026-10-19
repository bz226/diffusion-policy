#!/usr/bin/env bash
set -euo pipefail
set -f
exec /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/env/bin/python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/noise_without_ppo/code/campaign.py "$@"
