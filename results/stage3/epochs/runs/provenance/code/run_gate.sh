#!/usr/bin/env bash
set -euo pipefail
set -f
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/epochs/code/environment.sh
PHASE=$1
if [[ "$PHASE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=
  exec python -B "$EP_OUTPUT/code/gate_epochs.py" --phase cpu --source "$EP_SOURCE" --output "$EP_OUTPUT/runs/gates/cpu"
elif [[ "$PHASE" == live ]]; then
  exec python -B "$EP_OUTPUT/code/gate_epochs.py" --phase live --source "$EP_SOURCE" --output "$EP_OUTPUT/runs/gates/live" --cpu-result "$EP_OUTPUT/runs/gates/cpu/gate_summary.json"
else
  echo 'Gate phase must be cpu or live' >&2
  exit 2
fi
