#!/usr/bin/env bash
# Process-local Stage 2 setup; never source the Stage 1 script or modify shell startup.
export STUDY_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip
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
cd "$STUDY_ROOT/dppo"
