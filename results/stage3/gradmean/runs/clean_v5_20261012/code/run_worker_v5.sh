#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/environment_v5.sh
exec python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/queue_v5.py worker "$@"
