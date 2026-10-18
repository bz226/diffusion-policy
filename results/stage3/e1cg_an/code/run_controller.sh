#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --job-name=e1cg-an-controller
set -euo pipefail
set -f
exec "$1" -B "$2" run "$3"
