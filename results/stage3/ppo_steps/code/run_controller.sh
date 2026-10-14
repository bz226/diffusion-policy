#!/usr/bin/env bash
# Submission is an explicit root-agent action after scientific gates pass.
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --job-name=ppo-steps-controller
set -euo pipefail
set -f
exec "$1" -B "$3" run "$2"
