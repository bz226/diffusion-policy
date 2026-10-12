#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:10:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-v5-gate-monitor
set -euo pipefail
exec /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/env/bin/python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/monitor_gate_v5.py
