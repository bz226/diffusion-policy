#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/stage2_noclip/code/environment.sh
exec python "$STUDY_ROOT/code/verify_changes.py" \
  --repo "$STUDY_ROOT/dppo" --output "$STUDY_ROOT/runs/verification" \
  --device cuda:0 --timeout-seconds 1740 \
  --checkpoint "$DPPO_LOG_DIR/gym-pretrain/halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-04-42/checkpoint/state_3000.pt"
