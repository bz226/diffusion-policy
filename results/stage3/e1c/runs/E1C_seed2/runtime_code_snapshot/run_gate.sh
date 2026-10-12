#!/usr/bin/env bash
# Saved-batch CPU gate only. No Slurm submission, rollout, or optimizer update.
set -euo pipefail
E1C_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/e1c
STAGE1_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah
E1C_SOURCE=/soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-e1c/results/stage3_e1c/dppo
export PATH="$STAGE1_ROOT/env/bin:$PATH"
export PYTHONPATH="$E1C_SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia
export DPPO_DATA_DIR="$STAGE1_ROOT/dppo/data" DPPO_LOG_DIR="$STAGE1_ROOT/dppo/log"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$E1C_ROOT/runs/gate/tmp" XDG_CACHE_HOME="$E1C_ROOT/runs/gate/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export D4RL_DATASET_DIR="$XDG_CACHE_HOME/d4rl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$D4RL_DATASET_DIR"
cd "$E1C_SOURCE"
exec timeout --signal=TERM --kill-after=30s 1200 python -B "$E1C_ROOT/code/gate_e1c.py"
