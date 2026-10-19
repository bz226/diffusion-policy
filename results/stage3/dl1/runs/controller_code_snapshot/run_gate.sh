#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/code/environment.sh
PHASE=$1
if [[ "$PHASE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=
  exec python -B "$DL1_OUTPUT/code/gate_dl1.py" --phase cpu --source "$DL1_SOURCE" --output "$DL1_OUTPUT/runs/gates/cpu"
elif [[ "$PHASE" == live ]]; then
  exec python -B "$DL1_OUTPUT/code/gate_dl1.py" --phase live --source "$DL1_SOURCE" --output "$DL1_OUTPUT/runs/gates/live" --cpu-result "$DL1_OUTPUT/runs/gates/cpu/gate_summary.json"
else
  echo 'Gate phase must be cpu or live' >&2
  exit 2
fi
