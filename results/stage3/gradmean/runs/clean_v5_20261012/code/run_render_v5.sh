#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-clean-v5
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/code/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/plot_clean.py "$@"
