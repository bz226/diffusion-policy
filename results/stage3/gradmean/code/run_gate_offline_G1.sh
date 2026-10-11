#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-G1-G5
#SBATCH --chdir=/soalnas/share/data/zbao7/diffusion_policy
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/setup/G1-G5-%j.out
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/code/environment.sh
exec python -B "$GRADMEAN_OUTPUT/code/gate_offline.py" --g2 "$GRADMEAN_OUTPUT/runs/gate_G2_G1/gate_summary.json" --output "$GRADMEAN_OUTPUT/runs/gate_offline_G1"
