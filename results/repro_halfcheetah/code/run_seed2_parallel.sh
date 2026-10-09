#!/usr/bin/env bash
# Run the one remaining authorized seed on its own GPU and 40 CPUs.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/code/environment.sh
exec > >(tee -a "$REPRO_ROOT/setup/seed2-parallel-console.log") 2>&1
set -x
trap 'batch_rc=$?; printf "{\"exit_code\":%s,\"status\":\"terminal\"}\n" "$batch_rc" > "$REPRO_ROOT/runs/parallel_batch_status.json"' EXIT
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['REPRO_ROOT'])
assignment = json.loads((root / 'provenance/parallel_seed2.json').read_text())
assert assignment['status'] == 'assigned'
assert str(assignment['parallel_job_id']) == os.environ['SLURM_JOB_ID']
assert int(os.environ['SLURM_CPUS_PER_TASK']) == 40
assert not (root / 'runs/seed2').exists(), 'Never duplicate seed 2'
first = json.loads((root / 'runs/seed1/manifest.json').read_text())
assert first['status'] in ('running', 'complete'), first
assert str(first['slurm_job_id']) == str(assignment['original_job_id'])
print('Parallel assignment and unchanged seed 1 verified.')
PY
printf '{"status":"running"}\n' > "$REPRO_ROOT/runs/parallel_batch_status.json"
nvidia-smi
nproc
date -u
estimate=$(python "$REPRO_ROOT/code/estimate_runtime.py")
python "$REPRO_ROOT/code/supervise_run.py" --run-id seed2 --timeout-seconds 8100 --estimate-seconds "$estimate" -- python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=140 train.save_model_freq=35 seed=2 wandb=null
git diff --exit-code HEAD --
python "$REPRO_ROOT/code/finalize_candidates.py"
date -u
