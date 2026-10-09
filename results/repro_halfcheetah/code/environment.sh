#!/usr/bin/env bash
# Session-only paths; do not source from shell startup files.
REPRO_ROOT=/soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah
export REPRO_ROOT
export PATH="$REPRO_ROOT/env/bin:$PATH"
export TMPDIR="$REPRO_ROOT/tmp"
export PIP_CACHE_DIR="$REPRO_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$REPRO_ROOT/cache/conda"
export CONDA_ENVS_PATH="$REPRO_ROOT/conda_envs"
export CONDA_REGISTER_ENVS=false
export XDG_CACHE_HOME="$REPRO_ROOT/cache"
export MPLCONFIGDIR="$REPRO_ROOT/cache/matplotlib"
export TORCH_HOME="$REPRO_ROOT/cache/torch"
export D4RL_DATASET_DIR="$REPRO_ROOT/dppo/data/d4rl"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPRO_ROOT/dppo"
export DPPO_DATA_DIR="$PWD/data"
export DPPO_LOG_DIR="$PWD/log"
