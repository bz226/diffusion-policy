#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=e1cg-an-cpu-gates
set -euo pipefail
set -f
PROJECT=/soalnas/share/data/zbao7/diffusion_policy
STUDY=$PROJECT/results/stage3/e1cg_an
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$STUDY/tmp"
export XDG_CACHE_HOME="$STUDY/cache" MPLCONFIGDIR="$STUDY/cache/matplotlib"
export TORCH_HOME="$STUDY/cache/torch" TRITON_CACHE_DIR="$STUDY/cache/triton"
export CUDA_CACHE_PATH="$STUDY/cache/cuda" D4RL_DATASET_DIR="$STUDY/cache/d4rl"
export LD_LIBRARY_PATH=/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia
export DPPO_DATA_DIR="$PROJECT/results/repro_halfcheetah/dppo/data"
export DPPO_LOG_DIR="$PROJECT/results/repro_halfcheetah/dppo/log"
cd "$PROJECT"
exec "$PROJECT/results/repro_halfcheetah/env/bin/python" -B \
  "$STUDY/code/continue_after_calibration.py" --execute
