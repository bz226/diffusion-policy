#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=gm-gate-G2
#SBATCH --chdir=/soalnas/share/data/zbao7/diffusion_policy
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/setup/G2-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/env/bin/python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/setup/g2_source/gate_synthetic.py --out /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/gate_G2 --workers 8
