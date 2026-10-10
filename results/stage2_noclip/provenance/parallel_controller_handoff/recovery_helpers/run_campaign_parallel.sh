#!/usr/bin/env bash
# Adopt a stopped CPU controller; existing training allocations remain untouched.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
python "$STUDY_ROOT/code/campaign_parallel.py" "$@" &
controller_pid=$!
trap 'kill -TERM "$controller_pid" 2>/dev/null || true; wait "$controller_pid" || true; exit 143' TERM INT
wait "$controller_pid"
trap - TERM INT
exec python "$STUDY_ROOT/code/finalize_campaign.py" \
  --matrix "$STUDY_ROOT/provenance/run_matrix_revised.json" \
  --campaign "$STUDY_ROOT/runs/campaign_revised.json"
