#!/usr/bin/env bash
#SBATCH --job-name=gm_v5_cpu_gate
#SBATCH --partition=soal
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/gate_scheduler.out
#SBATCH --error=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/gate_scheduler.err
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/environment_v5.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec python -B /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/code/gate_v5.py \
  --source /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v5/results/stage3_gradmean/dppo \
  --reference /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3_gradmean/dppo \
  --parent /soalnas/share/data/zbao7/diffusion_policy \
  --base-commit eabff898011503ad2635b0efdbc59e0b90c8dc5c \
  --prepare /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/prepare/D1/grad_mean.json \
  --checkpoint /soalnas/share/data/zbao7/diffusion_policy/results/stage3/e1c/E1C_seed0/checkpoint/state_0.pt \
  --divisor-audit /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/log_audit.json \
  --batch /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/gate_v6/old/saved_batch.pt \
  --output /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/gate_cpu
