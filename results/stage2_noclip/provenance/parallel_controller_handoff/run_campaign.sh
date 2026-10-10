#!/usr/bin/env bash
# CPU-only Slurm controller; each training job owns its GPU/CPU allocation.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
python "$STUDY_ROOT/code/campaign.py" "$@" &
controller_pid=$!
trap 'kill -TERM "$controller_pid" 2>/dev/null || true; wait "$controller_pid" || true; exit 143' TERM INT
wait "$controller_pid"
trap - TERM INT
exec python "$STUDY_ROOT/code/finalize_campaign.py" \
  --matrix "$STUDY_ROOT/provenance/run_matrix_revised.json" \
  --campaign "$STUDY_ROOT/runs/campaign_revised.json"
