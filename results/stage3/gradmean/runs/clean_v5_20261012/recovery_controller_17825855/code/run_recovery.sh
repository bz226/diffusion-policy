#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-v5-recovery
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/environment_v5.sh
exec python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/recovery_controller_17825855/code/recover_controller.py run
