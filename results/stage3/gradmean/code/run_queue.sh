#!/usr/bin/env bash
# CPU-only durable controller. Submission remains an explicit operator action.
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-controller
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/code/environment.sh
exec python -B "$GRADMEAN_OUTPUT/code/campaign.py" run "$@"
