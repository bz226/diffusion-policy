#!/usr/bin/env bash
# Root submits this with --dependency=afterany:17810633. This one CPU-only
# allocation replaces the expiring interactive job in the shared job count.
# calG is admitted by this controller; no direct calG submission is needed.
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --nice=1000
#SBATCH --job-name=critic-controller
#SBATCH --export=PATH,HOME,USER,LOGNAME,LANG
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/setup/controller-%j.out
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic/code/environment.sh
exec python -B "$CRITIC_OUTPUT/code/campaign.py" run --controller-cap-seconds=172800 "$@"
