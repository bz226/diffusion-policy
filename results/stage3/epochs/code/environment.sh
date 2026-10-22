#!/usr/bin/env bash
# Same packages, MuJoCo, data and thread settings as the Stage-1/3 baseline.
export EP_OUTPUT=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/epochs
export EP_PARENT=/soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-epochs
export EP_SOURCE="$EP_PARENT/results/stage3/epochs/dppo"
STAGE1_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah
export PATH="$STAGE1_ROOT/env/bin:$PATH"
export PYTHONPATH="$EP_SOURCE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$EP_OUTPUT/tmp" XDG_CACHE_HOME="$EP_OUTPUT/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export D4RL_DATASET_DIR="$XDG_CACHE_HOME/d4rl"
export LD_LIBRARY_PATH=/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia
export DPPO_DATA_DIR="$STAGE1_ROOT/dppo/data" DPPO_LOG_DIR="$STAGE1_ROOT/dppo/log"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$D4RL_DATASET_DIR"
cd "$EP_SOURCE"
