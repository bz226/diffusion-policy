# Approved reproduction methods

This is a reproduction, not a hyperparameter search or algorithm modification. The authoritative algorithm and all fixed parameters are the [pinned YAML](https://github.com/irom-princeton/dppo/blob/cc7234a/cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp.yaml), [training loop](https://github.com/irom-princeton/dppo/blob/cc7234a/agent/finetune/train_ppo_diffusion_agent.py), and Ren et al., [arXiv:2409.00588, §4, Fig. 5, Appendix C Table A1](https://arxiv.org/html/2409.00588v1). These sources were inspected read-only before approval. Repository owner `irom-lab` redirects to `irom-princeton`; preserve the requested commit.

## Parameters and approval

| Code key / quantity | Meaning | Value | Rationale |
|---|---|---|---|
| env.n_envs | Parallel environment count | 40 | Constraint: unchanged pinned YAML |
| train.n_steps | Chunk decisions per environment/iteration | 500 | Constraint: pinned YAML |
| act_steps | Executed physical actions/chunk | 4 | Constraint: pinned YAML |
| denoising_steps / ft_denoising_steps | Total / optimized denoising steps | 20 / 10 | Constraint: pinned YAML |
| seed | Independent fine-tuning runs | 0, 1, 2 | User's design, three repetitions |
| train.n_train_itr | Total iterations/full run | 140 | User's budget: 126 training × 80,000 physical environment steps |
| train.save_model_freq | Checkpoint interval | 35 | User's requested override |
| train.val_freq | Evaluation-only iteration interval | 10 | Constraint: pinned YAML |
| wandb | External experiment tracking | null | User's required override |
| initial evaluation range | Stop gate on pretrained return | 3850–4650 | User's approximate Fig. 5 acceptance interval |
| final mean evaluation range | Approximate reproduction criterion | 4550–4900 | User's approximate Fig. 5 acceptance interval; apply to labeled last saved point |

All other parameters remain exactly as checked in. Config uses the D4RL ID `halfcheetah-medium-v2`; the separately prescribed import check makes `HalfCheetah-v2`. Neither is changed. Installation pins may repair dependencies only and must be documented individually. No source/config file edits, extra baselines, sweeps, training continuations, evaluation runs, or data transformations are permitted.

## Execution and verification

Clone below this study's `dppo/`; verify full commit and clean tracked source/config before and after execution. Install isolated conda Python 3.8, then `pip install -e .` and `pip install -e '.[gym]'`, torch 2.4.0 with supported CUDA wheel. MuJoCo 2.1.0 installs at the explicitly requested `~/.mujoco/mujoco210`; this installation path is the task's explicit exception to the study-only artifact location. Do not run `script/set_path.sh` or change shell startup files. Export `DPPO_DATA_DIR=$PWD/data DPPO_LOG_DIR=$PWD/log` from the cloned repository and export MuJoCo's bin on `LD_LIBRARY_PATH`. Keep package caches and temporary build files inside this study when configurable.

Run the user's import/reset check once the dependencies are installed. Then execute exactly:

```bash
python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=2 wandb=null
python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=140 train.save_model_freq=35 seed=0 wandb=null
python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=140 train.save_model_freq=35 seed=1 wandb=null
python script/run.py --config-dir=cfg/gym/finetune/halfcheetah-v2 --config-name=ft_ppo_diffusion_mlp train.n_train_itr=140 train.save_model_freq=35 seed=2 wandb=null
```

Pseudocode: verify installation and source → run smoke → inspect itr-0 gate and finite outputs → for each seed initialize from released weights, evaluate at 0,10,...,130 and train at all other iterations → stop at 139 → verify result.pkl, source, and checkpoints → aggregate the three saved series → pause for artifact decisions. An external supervisor may enforce timeout and failure gates without altering DPPO. A failed seed terminates the batch without restart. Save timestamps, commands, exact resolved config, dependency versions, data filenames/source URLs, clean-source status, execution status, and every log/checkpoint. Existing checkpoints contain model weights/iteration, not complete optimizer state.

The user explicitly approved a revised full-seed wall-clock cap of 8,100 seconds (2h15m), after the smoke projected approximately 2h04m. This changes external supervision only; run counts, steps, exact DPPO commands, and all hyperparameters remain fixed. The initial seed-0 watchdog has a resident 7,200-second deadline without reload support; preserve its original source and records when transferring supervision of the same running DPPO process.

Seed 0 retained its original DPPO process through that transfer. After DPPO exited normally, the retired watchdog recorded its superseded 7,200-second timeout; this is preserved as a controller outcome. The kernel and original parent both reported DPPO exit code zero, and all 140 result rows, five checkpoints, and the complete console stream passed verification before seed 1 launched. [Reconciliation evidence](runs/seed0_handoff/console_reconciliation.json). Seeds 1 and 2 use the 8,100-second cap directly; their initially sequential allocation is recorded in [allocation record](provenance/remaining_allocation.json). No DPPO process was restarted.

## Measurements and aggregation

For a completed episode e, raw undiscounted return is R_e = sum over its 1000 physical steps of reward. The native per-iteration metric is the average of completed-episode returns; it precedes training reward scaling. Training collection and evaluation sampling differ, so plot them separately. Evaluation remains stochastic under the native diffusion sampling algorithm. Episodes within a training run are not independent training repetitions; the independent repetition unit is the seed. Retain native episode counts where available in saved records; do not invent them.

For aligned native step x, report mean m(x) = (R_0(x)+R_1(x)+R_2(x))/3 and population standard deviation s(x) = sqrt(sum_i (R_i(x)-m(x))²/3). Shading is across-seed variability, not a confidence interval. No smoothing, reward normalization, interpolation, extrapolation, or seed exclusions. The table uses target→actual steps: 0→0, 2.5m→2.16m, 5m→5.04m, 7.5m→7.20m, 10m→9.36m. Add itr-1 training return at 80,000 steps in the same table. Individual runs and all failures remain visible. The exact 10m evaluation criterion is unmeasured; only approximate endpoint agreement can be assessed.

Validate exactly 140 rows per completed full run, iteration/step schedule, finite metrics, and checkpoints at 0/35/70/105/139 before analysis. Record native iteration durations and external subprocess wall time separately; native durations omit installation and agent initialization. Preserve requested output captions adjacent to the figures. Candidate outputs belong under a fresh run's `candidates/`; no curated copy or completed-study checklist mutation outside the permitted study directory is authorized.

## Parallel execution requested during seed 1

The user asked to run seeds in parallel. Seed 2 is assigned job 17770172 on a separate A6000 GPU and 40 CPU cores, while seed 1 remains in its original job 17769259 and process. Future-only external dispatch guards make the original launcher skip seed 2 and delegate analysis to the new job; all DPPO code, commands, steps and per-seed caps remain unchanged. The seed 2 scheduled monitor writes a separate status fragment to avoid competing writes with the resident seed 1 monitor. Runtime estimation reads only verified completed runs. Analysis requires all three completed manifests and waits only within seed 1’s existing deadline if needed. No additional repetition, restart or scientific output is authorized. [Assignment and resource limits](provenance/parallel_seed2.json).
