#!/usr/bin/env bash
# Study-local DPPO skeleton. The shared Stage-3 COMMON runner is unchanged.
set -euo pipefail
set -f
if [[ "${SKELETON:-dppo}" != dppo ]]; then
  echo 'The epochs study requires SKELETON=dppo.' >&2
  exit 2
fi
source /soalnas/share/data/zbao7/diffusion_policy/results/stage3/epochs/code/environment.sh
exec python -B "$EP_OUTPUT/code/campaign.py" "$@"
