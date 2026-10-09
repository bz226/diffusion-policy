#!/usr/bin/env bash
# Run only after installation succeeds. Slurm supplies the GPU and CPU allocation.
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/code/environment.sh
exec > >(tee -a "$REPRO_ROOT/setup/batch-console.log") 2>&1
set -x
trap 'batch_rc=$?; printf "{\"exit_code\":%s,\"status\":\"terminal\"}\n" "$batch_rc" > "$REPRO_ROOT/runs/batch_status.json"' EXIT
printf '{"status":"running"}\n' > "$REPRO_ROOT/runs/batch_status.json"
setup_remaining=$(python -c "import datetime,time; print(int(datetime.datetime(2026,10,9,4,33,tzinfo=datetime.timezone.utc).timestamp()-time.time()))")
test "$setup_remaining" -gt 0
timeout --kill-after=15 "$setup_remaining" bash "$REPRO_ROOT/code/check_install.sh"
python "$REPRO_ROOT/code/supervise_run.py" --run-id smoke --timeout-seconds 1800 --estimate-seconds 60 -- python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=2 wandb=null
for seed in 0 1 2; do
    estimate=$(python "$REPRO_ROOT/code/estimate_runtime.py")
    python "$REPRO_ROOT/code/supervise_run.py" --run-id "seed$seed" --timeout-seconds 7200 --estimate-seconds "$estimate" -- python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=140 train.save_model_freq=35 "seed=$seed" wandb=null
done
git diff --exit-code HEAD --
date -u
