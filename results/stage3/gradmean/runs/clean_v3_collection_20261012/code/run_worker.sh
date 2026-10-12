#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/code/environment.sh
exec python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v3_collection_20261012/code/collection.py worker "$@"
