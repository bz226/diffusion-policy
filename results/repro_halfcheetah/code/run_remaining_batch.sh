#!/usr/bin/env bash
# Continue the authorized seed list after seed 0's external watchdog handoff.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/code/environment.sh
exec > >(tee -a "$REPRO_ROOT/setup/remaining-batch-console.log") 2>&1
set -x
trap 'batch_rc=$?; printf "{\"exit_code\":%s,\"status\":\"terminal\"}\n" "$batch_rc" > "$REPRO_ROOT/runs/remaining_batch_status.json"' EXIT
python "$REPRO_ROOT/code/run_logged.py" --label seed0_console_reconciliation --timeout 120 --cwd "$PWD" -- python "$REPRO_ROOT/code/adopt_running_run.py" --reconcile-only
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['REPRO_ROOT'])
manifest = json.loads((root / 'runs/seed0_handoff/manifest.json').read_text())
assert manifest['status'] == 'complete', manifest
assert manifest['verified_rows'] == 140, manifest
assert manifest['checkpoint_tensors_finite'] is True, manifest
assert manifest['final_training_env_steps'] == 10080000, manifest
assert manifest['requires_original_console_reconciliation'] is False, manifest
reconciliation = json.loads((root / 'runs/seed0_handoff/console_reconciliation.json').read_text())
assert reconciliation['status'] == 'passed', reconciliation
assert all(not (root / 'runs' / run).exists() for run in ('seed1', 'seed2')), 'Never duplicate a seed'
print('Verified seed 0; starting only the two remaining authorized seeds.')
PY
printf '{"status":"running"}\n' > "$REPRO_ROOT/runs/remaining_batch_status.json"
nvidia-smi
nproc
for seed in 1 2; do
    estimate=$(python "$REPRO_ROOT/code/estimate_runtime.py")
    python "$REPRO_ROOT/code/supervise_run.py" --run-id "seed$seed" --timeout-seconds 8100 --estimate-seconds "$estimate" -- python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=140 train.save_model_freq=35 "seed=$seed" wandb=null
done
git diff --exit-code HEAD --
python "$REPRO_ROOT/code/finalize_candidates.py"
date -u
