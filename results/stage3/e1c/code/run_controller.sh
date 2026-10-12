#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --job-name=e1c-controller
#SBATCH --export=PATH,HOME,USER,LOGNAME,LANG
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/e1c/setup/controller-%j.out
set -euo pipefail
set -f
exec bash /soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-e1c/results/stage3/run_queue.sh e1c run "$@"
