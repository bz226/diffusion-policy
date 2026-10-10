#!/usr/bin/env bash
# Process-local setup; reuse the frozen Stage-1 environment without modifying it.
export STUDY_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/stage3
STAGE1_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah
export PATH="$STAGE1_ROOT/env/bin:$PATH"
export PYTHONPATH="$STUDY_ROOT/dppo"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$STUDY_ROOT/tmp"
export PIP_CACHE_DIR="$STUDY_ROOT/cache/pip"
export XDG_CACHE_HOME="$STUDY_ROOT/cache"
export MPLCONFIGDIR="$STUDY_ROOT/cache/matplotlib"
export TORCH_HOME="$STUDY_ROOT/cache/torch"
export TRITON_CACHE_DIR="$STUDY_ROOT/cache/triton"
export CUDA_CACHE_PATH="$STUDY_ROOT/cache/cuda"
export D4RL_DATASET_DIR="$STUDY_ROOT/cache/d4rl"
export LD_LIBRARY_PATH="/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export DPPO_DATA_DIR="$STAGE1_ROOT/dppo/data"
export DPPO_LOG_DIR="$STAGE1_ROOT/dppo/log"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$D4RL_DATASET_DIR"
cd "$STUDY_ROOT/dppo"
