#!/usr/bin/env bash
# Submit this wrapper as a CPU-only Slurm controller; it never trains locally.
# Review/preparation: bash results/stage3/run_queue.sh prepare
# Execution: sbatch ... results/stage3/run_queue.sh run --commit HASH --gate /abs/gate.json --approval /abs/approval.json
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/code/environment.sh
exec python -B "$STUDY_ROOT/code/campaign.py" "$@"
