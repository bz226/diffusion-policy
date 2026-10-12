#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:20:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-v3-winddown
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/code/environment.sh
exec python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/monitor_v3.py --monitor
