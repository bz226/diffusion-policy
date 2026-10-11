#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --export=PATH,HOME,USER,LOGNAME,LANG
#SBATCH --job-name=frs-controller
#SBATCH --chdir=/soalnas/share/data/zbao7/diffusion_policy/results/stage3_frs
#SBATCH --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3_frs/setup/controller-%j.out
#SBATCH --signal=B:TERM@60
# Controller owns no GPU; GPU workers keep their independent stopping limits.
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3_frs/code/environment.sh
exec python -B "$STUDY_ROOT/code/campaign.py" run "$@"
