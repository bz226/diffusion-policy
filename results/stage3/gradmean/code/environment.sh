#!/usr/bin/env bash
# Reuse Stage 1 without changing its environment, data, or thread settings.
export GRADMEAN_OUTPUT=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean
export GRADMEAN_SOURCE=/soalnas/share/data/zbao7/diffusion_policy/results/stage3_gradmean/dppo
STAGE1_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah
export PATH="$STAGE1_ROOT/env/bin:$PATH"
export PYTHONPATH="$GRADMEAN_SOURCE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$GRADMEAN_OUTPUT/tmp"
export XDG_CACHE_HOME="$GRADMEAN_OUTPUT/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export D4RL_DATASET_DIR="$XDG_CACHE_HOME/d4rl"
export LD_LIBRARY_PATH=/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia
export DPPO_DATA_DIR="$STAGE1_ROOT/dppo/data"
export DPPO_LOG_DIR="$STAGE1_ROOT/dppo/log"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$D4RL_DATASET_DIR"
cd "$GRADMEAN_SOURCE"
