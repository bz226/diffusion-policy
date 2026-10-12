#!/usr/bin/env bash
source /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/code/environment.sh
export GRADMEAN_SOURCE=/soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v5/results/stage3_gradmean/dppo
export PYTHONPATH="$GRADMEAN_SOURCE"
cd "$GRADMEAN_SOURCE"
