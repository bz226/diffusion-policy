# Stage 2 methods

This design was approved by the user's “approve” reply. Execution stopped when R0 failed the required matching-value replay check at iteration 6; no ablations launched. See [experiment.md](experiment.md) and [terminal records](provenance/terminal_status.json). The remaining sections preserve the approved protocol, not a claim that the full campaign executed. The user-specified conditions are the scientific scope. The retained Stage 1 source and completed records were inspected read-only; the fresh clone is pinned to the full revision below. The retained Stage 1 source lacks its own Git metadata and was not used as a Git remote or worktree parent.

## Fixed setup and source isolation

Create `dppo/` inside this study by cloning the official repository and running `git checkout -b stage2-noclip cc7234a`. Use the full expected revision `cc7234ad7ff39a8f32de3af903606723a16f0648` for verification. Keep the clone pristine throughout both determinism probes and wait for both processes to finish before any scientific source edit. Then modify and commit only `model/diffusion/diffusion_ppo.py` and `agent/finetune/train_ppo_diffusion_agent.py`. Keep supervision, verification, analysis and provenance helpers elsewhere in this study. Save the commit and `git diff cc7234a`; do not hash artifacts. Verify that no YAML or other scientific source file changes.

Reuse `../repro_halfcheetah/env/bin/python` and MuJoCo 2.1.0. Do not reinstall the editable DPPO package or modify Stage 1's environment script. Instead, prepend the Stage 2 source directory to the process-local import path and verify that imported DPPO scientific modules originate there, despite the environment's existing Stage 1 editable installation. If routing cannot be verified without an additional implementation change, stop and report.

Retain Stage 1's absolute `DPPO_DATA_DIR` and `DPPO_LOG_DIR` values so the released pretrained checkpoint and normalization statistics are read from their existing locations. Preflight both assets and refuse fallback downloads into Stage 1. The proposed extra `logdir` CLI override places Hydra and native outputs under a unique Stage 2 condition/seed path; Hydra's existing `run.dir` already follows it. Do not edit YAML. Use Stage 2-only output, cache, temporary and plotting paths. Stage 1 recorded no `OMP_NUM_THREADS`, `MKL_NUM_THREADS` or `OPENBLAS_NUM_THREADS` override; keep these unset.

## Conditions and parameters

Every full run uses `train.n_train_itr=140 train.save_model_freq=35 seed=<seed> wandb=null`, plus the approved output-location override. P1 and P2 instead use `train.n_train_itr=20`, both seed 0, and the same other settings, on pristine source. No new scientific flags are passed to the probes or R0. New model/train keys for ablations use Hydra `+` syntax.

| Condition | Seeds | Additional scientific overrides | Source |
|---|---|---|---|
| P1, P2 | 0 for both | Pristine source; 20 iterations each | User revised regression design |
| R0 | 0 | None; new flags at defaults | User regression design |
| NC1 | 0, 1, 2 | `model.clip_ploss_coef=1e6 model.clip_ploss_coef_base=1e6 train.target_kl=null` | User design |
| NC2 | 0, 1, 2 | NC1 plus `+model.clamp_logprob=false model.randn_clip_value=100` | User design |
| NC3 | 0, 1, 2 | NC2 plus `+model.logprob_reduce=sum` | User design |
| NC4, LR 1e-4 | 0, 1, 2 | NC2 plus `+train.actor_single_step=true train.actor_lr=1e-4 train.actor_lr_scheduler.min_lr=1e-4` | User design |
| NC4, LR 1e-3 | 0, 1, 2 | NC2 plus `+train.actor_single_step=true train.actor_lr=1e-3 train.actor_lr_scheduler.min_lr=1e-3` | User design |
| NC4, LR 3e-3 | 0, 1, 2 | NC2 plus `+train.actor_single_step=true train.actor_lr=3e-3 train.actor_lr_scheduler.min_lr=3e-3` | User design |

| Fixed key / quantity | Value | Rationale |
|---|---|---|
| Environment / action dimensions | `halfcheetah-medium-v2`, 6 | Unchanged pinned YAML |
| `env.n_envs`, `train.n_steps` | 40, 500 | User constraint |
| Action chunk / total denoising steps / fine-tuned steps | 4 / 20 / 10 | User constraint |
| `train.batch_size`, `train.update_epochs` | 50,000 pairs, 5 | User constraint |
| Pair count per training iteration | 200,000 | 500 × 40 × 10 |
| Physical environment steps per training iteration | 80,000 | 500 × 40 × 4 |
| Sampling and log-probability noise floors | 0.1, 0.1 | User constraint |
| Actor single-step default / log-probability clamp default / reduction default | false / true / mean | Preserve baseline scientific behavior |
| Diagnostic subsample | 20,000 pairs, without replacement | User size; uniform sampling design |
| Diagnostic RNG | Separate CPU `torch.Generator`, seeded with the run seed | Reproducible diagnostics without global RNG draws |
| Replay reference | All original log/result values at iterations 0–19 of pristine P1 and P2 | User revised regression design |
| Nondeterministic fallback red flag | Stage 1 same-step evaluation mean ± 5 sample standard deviations, using all 3 seeds | User threshold; not a seed-band test |
| Proposed runtime caps | 1,800 seconds per probe; 10,800 per full run | Bounded budget with diagnostic overhead allowance |

All other configuration values stay unchanged, including denoised-action clipping, advantage normalization, GAE, denoising discount, AdamW, critic, schedules outside the specified NC4 LR overrides, and the actor/critic architecture. Capture resolved configurations and effective startup values for every run. Validate the exact command-key set and compare printed effective flags with the intended condition, so misspelled new keys cannot silently pass.

## Algorithm changes

The original PPO implementation is [PPODiffusion.loss](https://github.com/irom-princeton/dppo/blob/cc7234a/model/diffusion/diffusion_ppo.py). Add typed constructor settings `clamp_logprob=True` and `logprob_reduce="mean"`. Gate only the old/new PPO coordinate clamps; select mean or sum over the executed action coordinates, rejecting any other reduction. Leave the unrelated BC branch unchanged; BC is disabled in this config.

In [TrainPPODiffusionAgent](https://github.com/irom-princeton/dppo/blob/cc7234a/agent/finetune/train_ppo_diffusion_agent.py), read `cfg.train.get("actor_single_step", False)`. Preserve the original default update path and RNG draw order. With the flag true, keep the existing five epochs and four minibatches per epoch for critic updates only, using the unchanged value loss and coefficient. Then perform exactly one actor optimizer step: reuse the last critic epoch's existing permutation, traverse all 200,000 pairs in four 50,000-pair chunks, and accumulate each actor objective multiplied by its chunk size divided by 200,000. The same PPO loss retains per-chunk advantage normalization. Exclude value loss from actor backward; zero actor gradients once and step once. Keep the existing scheduler placement.

Reusing the final permutation makes the chunk choice explicit and adds no global RNG draw. Verify complete coverage, unchanged actor parameters throughout critic updates, 20 critic optimizer steps and one actor optimizer step. Report the accumulated actor step's sample-weighted pre-update mean ratio and clip fraction; require literal 1 and 0. Cached old log-probabilities use a different forward batch shape than actor chunks, so floating-point differences are possible even with unchanged parameters. Do not round, force ratios, replace cached old probabilities, or relax the invariant silently; stop and report if this check fails.

## Diagnostics and verification

Diagnostics run after every training update, including R0, without changing model mode, optimizers, parameters, or global Torch/NumPy/Python RNG states. They use the existing raw `get_logprobs_subsample` method; no VPG implementation changes are needed.

For a uniform sample of 20,000 flattened environment/denoising pairs, gather stored raw old coordinate log-probabilities and evaluate new raw log-probabilities. Cast coordinates to float64 before summing the 4 × 6 coordinates. For pair i, let d_i = sum(new_i) − sum(old_i). Save and print:

- `kl_true_per_action` = 10 × mean(expm1(d_i) − d_i).
- `logratio_p99` = the 0.99 quantile of |d_i|.
- `clamp_hit_frac` = the count of new coordinate log-probabilities strictly outside [−5, 2], divided by 20,000 × 24.

Use the prescribed formulas without stabilizing clamps, clipping d, or replacing nonfinite values. The name `kl_true_per_action` is retained as requested, but the quantity is a Monte Carlo Gaussian-transition estimate. It is not the exact KL of the final action marginal; with clipped sampling noise, it is also not the exact KL of that clipped sampler.

Also persist DPPO's existing metrics: `approx_kl` from the last optimized minibatch and `clipfrac` averaged over the optimized minibatches, preserving their definitions. For NC4, explicitly label the weighted actor pre-step ratio/clip fraction so they are not confused with the post-update diagnostic KL. Record actual actor and critic step counts and learning rates for verification.

Before R0, perform bounded fixed-batch checks of pinned-versus-new default losses, gradients and updates; invalid-reduction rejection and flag effectiveness; RNG-state preservation and independent diagnostic arithmetic; and the single-step coverage/optimizer/invariant checks. Use no additional scientific training runs. Proposed GPU verification budget: at most 30 minutes. R0's two initial rewards alone cannot verify default gradient-update equivalence because both precede the first actor update.

## Exact replay regression protocol

The user's revised Section 2 replaces the earlier two-reward check and all Stage 1 seed-band acceptance rules. After setup, launch P1 and P2 simultaneously on separate A6000 GPUs, seed 0 and 20 iterations, with the pinned source unmodified. Save root consoles as `P1_seed0.out` and `P2_seed0.out`, and native outputs under `runs/P1_seed0/native` and `runs/P2_seed0/native`. Do not enable deterministic-algorithm settings or change threading, libraries, configuration or computation to make replay pass. Require both probes to finish successfully, with all 20 rows, before touching either scientific source file. Preserve their source revision, configurations and complete logs. Each probe has 18 training iterations, 2 evaluation iterations and 1,440,000 training environment steps.

After the Section 1 edits and fixed-batch checks, launch R0 alone with defaults, seed 0 and 140 iterations on the same GPU type and software stack. Its console is `R0_seed0.out`. Compare iterations 0–19 before releasing any ablation. Inspection of the retained pinned source found `n_train_itr` controls termination and final checkpoint saving; it does not change the scheduler's fixed `first_cycle_steps=1000`. Therefore the requested run-length difference does not itself alter the first 20 scientific updates.

Use P1's and P2's complete original result schemas and ordered iteration log payloads as the reference, not an unrestricted intersection with R0. Check every original key, value, shape and ordered message, including all minibatch `approx_kl`/epoch/batch messages and final train/eval summaries. Compare saved numeric values exactly without rounding, tolerance or smoothing; printed numeric tokens must also match exactly where required. Missing original fields/messages are mismatches. Strip wall-clock timestamp prefixes, the summary `t:` token and the result `time` field; preserve raw evidence.

Three structural exceptions require explicit approval because literal whole-output equality is otherwise impossible: run-specific checkpoint path prefixes; the probes' final `Saved model ... state_19.pt` message, absent in R0 because it continues; and R0's requested additive metrics/flag printouts. Inventory each exception in the comparison record. Normalize only the artifact root of corresponding checkpoint messages, retain/check checkpoint iteration numbers, and verify the expected probe-only iteration-19 save separately. Preserve the original iteration-summary payload when adding diagnostics. Compare all original fields; separately validate the declared new diagnostic/verification fields for presence, arithmetic and finiteness, labeling them unavailable in pristine probes. An unexpected added/deleted field or message is not silently ignored.

Decision logic:

1. Compare P1/P2 at every field/message of iterations 0–19. If identical, the platform passed this finite determinism probe; require exact R0=P1 equality on every original compared value. Any mismatch fails R0. Report the earliest iteration, field/message, and P1/P2/R0 values. Passing 20 iterations is evidence for this run configuration, not a proof of universal determinism.
2. If P1/P2 differ, report their first difference and retain a per-value equality mask over all 20 iterations, including matches after the first divergent iteration. R0 must equal P1 wherever P1=P2; any violation fails R0. At nonmatching values, do not impose approximate equality or a seed envelope. Missing/misaligned original output structure requires a stop/report rather than an invented alignment rule.
3. In the nondeterministic case, check R0 evaluation returns against the three unchanged Stage 1 seeds at the same saved environment step. Let xbar be their arithmetic mean and s their sample standard deviation (denominator 2). Flag only returns outside xbar ± 5s as gross deviations; continue this check through R0 completion. This is the user's approximate pointwise prediction threshold under independent normal-run assumptions, not a simultaneous confidence band or a determinism certificate. Report these red flags separately from exact-match failures; do not silently widen the threshold or invent an additional acceptance test.

Archive the comparison result and first-mismatch details under the study. R0 failures stop the campaign and are reported; no automatic retry. Probes lack the new diagnostics and are replay controls only, not extra independent ablation seeds or replacements for Stage 1.

## Execution, monitoring and failures

First complete both parallel pristine probes, then implement/verify and run R0 alone. Once the full 20-iteration replay gate above passes, leave R0 running and launch all three seeds of NC1 concurrently. Run later condition groups sequentially, with three concurrent seeds per group. Prefer one A6000, 40 CPUs and 64 GiB per process: two GPUs for the probes and a peak of four during R0 overlap. Record actual GPU sharing and per-condition elapsed times; no additional condition overlap or retries are proposed. Do not wait for the periodic monitor to enforce the iteration-19 gate or detect a replay failure; the run supervisor also checks newly saved progress.

Stage 1's measured full runtimes were 7414.73, 7380.34 and 7512.23 seconds. Its approximately 57–58-second training and 13-second evaluation iterations imply about 18 minutes per 20-iteration probe including startup. Nineteen full runs imply roughly 40 GPU-hours before diagnostic overhead; the two probes add roughly 0.6 GPU-hours. Proposed caps total 57 GPU-hours for full runs, 1 GPU-hour for the two probes, and 0.5 GPU-hours for fixed-batch verification: 58.5 GPU-hours. Approximately six three-seed waves plus the parallel probe window determine elapsed time, with queue time additional. Estimate each wave from comparable completed runs and start scheduled monitoring at max(30 minutes, estimated runtime / 6), initially 30 minutes for both probes and full runs. Record completion immediately even if it precedes the first scheduled check. Save dated counters, errors, resource allocation and terminal status; pause monitors at completion/failure. Never extend caps automatically. Runtime is a resource detail, not a scientific acceptance criterion.

Use exclusive run directories and preserve every console, resolved config, `result.pkl`, checkpoint and failure record. A crash/nonfinite condition stops that run and is recorded with the failing iteration/phase and the last available diagnostics (explicitly distinguishing an earlier iteration when necessary); remaining conditions continue. A failed R0 or a required change outside the authorized source scope stops the campaign for a decision. Distinguish resource timeout, finite crash, nonfinite failure and reward collapse. No failing seed is rerun or excluded.

No R0 decision uses containment in the Stage 1 seed band or mean ± population-standard-deviation band. The same-step mean ± five sample-standard-deviation threshold is used only for the user's nondeterministic fallback.

## Analysis and deliverables

Independent repetitions are training seeds. At each common saved step, use the arithmetic mean and population standard deviation (denominator equal to the three requested seeds), without smoothing or extrapolation. R0 is one regression run, so show it individually without an invented uncertainty band. Reuse all three original baseline runs unchanged.

Produce exactly two return figures, each with two panels: baseline versus NC1–NC3; baseline versus the three NC4 learning rates. Produce one five-panel diagnostic figure for R0 and the six three-seed conditions: prescribed KL on a log scale, absolute-log-ratio p99, clamp-hit fraction, native approximate KL and native clip fraction. Draw individual runs faintly and means/bands where all three condition seeds have measurements. After a failed seed stops, show remaining individual trajectories without presenting a changing-survivor average as a three-seed mean. Annotate failures; never replace zero/nonfinite KL values with a positive plotting epsilon.

The condition-by-seed table includes R0, all 18 ablation runs, and separately identified reused Stage 1 rows. Use iteration 0, 70 and 130 for the requested 0, approximately 5m and approximately 10m evaluations, at actual steps 0, 5.04m and 9.36m. A missing required iteration is NA, not a substituted earlier endpoint. Also list minimum observed evaluation, median logged diagnostic KL, and the requested collapse flag: any saved evaluation below half that seed's initial evaluation, or any observed NaN/inf. Do not silently omit nonfinite values in medians; report undefined statistics explicitly. Stage 1 diagnostic medians are unavailable because those diagnostics were not collected. Distinguish a finite crash from a demonstrated collapse, and flag an unassessable initial-reference case explicitly.

Save figures/table with captions in `runs/<analysis-id>/candidates/`, with native source paths and all run statuses. Report commit/diff, effective-flag verification, R0 outcome, timing/resource usage and factual effects of each cumulative removal. Do not infer statistical equivalence or claim an exact final-action KL. Scientific outputs and curation remain separate approvals; no curated files or completed-experiment catalog entry are created before the study actually completes.
