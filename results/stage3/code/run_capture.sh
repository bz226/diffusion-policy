#!/usr/bin/env bash
set -euo pipefail
STAGE3_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/stage3
STAGE1_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah
export PATH="$STAGE1_ROOT/env/bin:$PATH"
export PYTHONPATH="$STAGE3_ROOT/reference_stage2"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$STAGE3_ROOT/tmp"
export PIP_CACHE_DIR="$STAGE3_ROOT/cache/pip"
export XDG_CACHE_HOME="$STAGE3_ROOT/cache"
export MPLCONFIGDIR="$STAGE3_ROOT/cache/matplotlib"
export TORCH_HOME="$STAGE3_ROOT/cache/torch"
export TRITON_CACHE_DIR="$STAGE3_ROOT/cache/triton"
export CUDA_CACHE_PATH="$STAGE3_ROOT/cache/cuda"
export D4RL_DATASET_DIR="$STAGE3_ROOT/cache/d4rl"
export LD_LIBRARY_PATH="/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export DPPO_DATA_DIR="$STAGE1_ROOT/dppo/data"
export DPPO_LOG_DIR="$STAGE1_ROOT/dppo/log"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
cd "$STAGE3_ROOT/reference_stage2"
exec python -B "$STAGE3_ROOT/code/capture_stage2_batch.py" --timeout-seconds 840
