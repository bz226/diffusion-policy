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
#SBATCH --job-name=critic-gate
#SBATCH --nice=1000
#SBATCH --chdir=/soalnas/share/data/zbao7/diffusion_policy
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/setup/gate-%j.out
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/code/environment.sh
exec python -B "$CRITIC_OUTPUT/code/gate_critic.py" --output "$CRITIC_OUTPUT/runs/gate"
