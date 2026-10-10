# Stage 2 methods

The user approved the Stage 2 design and explicitly replaced its regression rule after the initial R0 was stopped. The current gate is the saved-batch, full-update differential check below; trajectory differences never decide code equivalence. The [revised authorization](provenance/revised_regression_authorization.json) preserves budgets and scope. Prior probes, the stopped R0 and their reports remain [archived](provenance/retired_replay_rule/). Current execution is recorded in [experiment.md](experiment.md).

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
| Code-equivalence reference | Pinned and edited native update code on one identical saved training batch | User replacement regression rule |
| Non-stopping R0 evaluation red flag | Stage 1 same-step evaluation mean ± 5 sample standard deviations, using all 3 seeds | User threshold; not a seed-band test |
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

## Fixed-batch differential regression gate

This section replaces all trajectory-based acceptance rules. P1/P2 already differed at iteration 0; matching rounded values after divergence can occur by chance. Their original records and the first R0 are retained as historical evidence, not acceptance references. Do not enable deterministic-algorithm settings or change numerical settings to make the check pass.

The earlier verification compared a 128-pair synthetic minibatch and one optimizer step. Its separate production-size check covered the NC4 single-step path, not the complete default update. Those checks remain useful but are insufficient for this gate.

Capture one real, pristine first-training-iteration batch immediately before the policy/critic update, using the unchanged baseline environment and config: 500 rollout steps × 40 environments, with ten denoising pairs per observation. Save observations, complete stored chains, advantages, returns, values and raw old log-probabilities, plus the model state/modes, actor and critic optimizer states, effective update attributes and Python/NumPy/torch CPU/all-visible-CUDA RNG states. Empty Adam moments before the first update are part of that state. Capture is verification data collection, not an extra training-condition seed. It uses the same bounded verification allocation.

Extract the actual update body from pristine and edited agent source without modifying either source file. Restore the identical saved inputs, states and device-appropriate RNG for each version; execute the full five-epoch, four-minibatch update separately on CPU and GPU. Record and compare every permutation, each minibatch's loss components, actor and critic gradients immediately before optimizer steps, and final actor/critic parameters and optimizer states. Require all 20 actor and critic steps; a truncated or early-stopped update does not satisfy this fixture's required coverage. Compare versions on each device; CPU-versus-GPU equality is not required.

CPU comparisons require bitwise equality, including tensor bytes. GPU comparisons use the predeclared elementwise tolerance `abs(new − reference) <= 1e-7 + 1e-5 * abs(reference)`; discrete indices and RNG states require exact equality. Report maximum absolute difference and maximum relative difference for each category and minibatch. The report defines its treatment of zero reference values; no tolerance is loosened after observing results.

For sampling, supply the same fixed noise tape to both versions: one initial draw and 20 denoising draws. Compare the complete returned chain and final action trajectory, and verify complete, matching tape consumption. After invoking the real diagnostics on the saved batch, compare Python, NumPy, torch CPU and all visible CUDA RNG states before/after; only the dedicated diagnostics generator may advance. Verify deterministic-algorithm and related numerical settings remain unchanged.

A passed, source-revision-bound gate authorizes immediate launch of the new R0 and all three NC1 seeds. Preserve the old R0 directory and use `runs/R0_rerun_seed0/` and `R0_rerun_seed0.out`. Start from the released pretrained policy, not its stopped checkpoint. No P1/P2 log token or saved trajectory comparison is used for stopping.

R0 now supplies baseline diagnostics only. For each evaluation, compare its return with the unchanged three Stage 1 seeds at the same saved step. Let xbar be their mean and s their sample standard deviation (denominator 2); a return outside xbar ± 5s is a reported, non-stopping red flag. This is the user's pointwise prediction heuristic, not a simultaneous confidence band, seed-envelope test or code-equivalence certificate.

## Execution, monitoring and failures

The user subsequently approved one hardware exception: retarget the still-pending `NC1_seed2` job 17777249 to one H200 NVL, retaining its job identity, 40 CPUs, 64 GiB, three-hour cap and scientific configuration. Before release, replay the existing saved batch on H200 with the same GPU differential tolerance, sampling and RNG checks; this uses at most ten minutes from the existing 30-minute total verification allowance. No new rollout or deterministic setting is introduced. Other training jobs retain A6000 requests. The [specific authorization](provenance/h200_seed2_authorization.json) and actual run hardware supplement the original run matrix. GPU hardware becomes a recorded covariate for this seed; same-device code equivalence does not establish identical learning trajectories across devices.

After the revised fixed-batch gate passes, launch the new R0 and the three NC1 seeds without waiting for trajectory comparisons. Run later condition groups sequentially, each with its three seeds in parallel. Allocate one A6000, 40 CPUs and 64 GiB per process, at most four concurrent GPUs during R0 overlap. Record actual GPU sharing and per-condition elapsed times. No automatic retries or additional condition overlap are authorized.

Stage 1's measured full runtimes were 7414.73, 7380.34 and 7512.23 seconds. Its approximately 57–58-second training and 13-second evaluation iterations imply about 18 minutes per 20-iteration probe including startup. Nineteen full runs imply roughly 40 GPU-hours before diagnostic overhead; the two probes add roughly 0.6 GPU-hours. Proposed caps total 57 GPU-hours for full runs, 1 GPU-hour for the two probes, and 0.5 GPU-hours for fixed-batch verification: 58.5 GPU-hours. Approximately six three-seed waves plus the parallel probe window determine elapsed time, with queue time additional. Estimate each wave from comparable completed runs and start scheduled monitoring at max(30 minutes, estimated runtime / 6), initially 30 minutes for both probes and full runs. Record completion immediately even if it precedes the first scheduled check. Save dated counters, errors, resource allocation and terminal status; pause monitors at completion/failure. Never extend caps automatically. Runtime is a resource detail, not a scientific acceptance criterion.

The revised launch keeps the original 58.5 GPU-hour ceiling. Completed probes consumed 1073 + 1081 GPU-seconds and the stopped R0 367 GPU-seconds. Prior GPU verification consumed 38 seconds; the new allocation is capped at 1740 seconds, so verification stays below 1800 seconds total. Even charging the full 1800-second verification allowance and all 19 remaining runs at three hours yields 58.2003 GPU-hours cumulatively. R0's explicitly requested restart is the sole authorized rerun. The CPU-only campaign controller is bounded to 24 hours including queue delays; individual jobs retain their three-hour limits.


Use exclusive run directories and preserve every console, resolved config, `result.pkl`, checkpoint and failure record. A crash/nonfinite condition stops that run and is recorded with the failing iteration/phase and the last available diagnostics (explicitly distinguishing an earlier iteration when necessary); remaining conditions continue. A failed fixed-batch gate or a required source change outside the authorized scope stops the campaign. R0 numerical failures are recorded like other run failures; trajectory deviations never stop remaining conditions. Distinguish resource timeout, finite crash, nonfinite failure and reward collapse. No failing seed is rerun or excluded.

No R0 decision uses containment in the Stage 1 seed band or mean ± population-standard-deviation band. The same-step mean ± five sample-standard-deviation threshold produces informational R0 red flags only.

## Analysis and deliverables

Independent repetitions are training seeds. At each common saved step, use the arithmetic mean and population standard deviation (denominator equal to the three requested seeds), without smoothing or extrapolation. R0 is one diagnostic baseline run, so show it individually without an invented uncertainty band. Reuse all three original baseline runs unchanged.

Produce exactly two return figures, each with two panels: baseline versus NC1–NC3; baseline versus the three NC4 learning rates. Produce one five-panel diagnostic figure for R0 and the six three-seed conditions: prescribed KL on a log scale, absolute-log-ratio p99, clamp-hit fraction, native approximate KL and native clip fraction. Draw individual runs faintly and means/bands where all three condition seeds have measurements. After a failed seed stops, show remaining individual trajectories without presenting a changing-survivor average as a three-seed mean. Annotate failures; never replace zero/nonfinite KL values with a positive plotting epsilon.

The condition-by-seed table includes R0, all 18 ablation runs, and separately identified reused Stage 1 rows. Use iteration 0, 70 and 130 for the requested 0, approximately 5m and approximately 10m evaluations, at actual steps 0, 5.04m and 9.36m. A missing required iteration is NA, not a substituted earlier endpoint. Also list minimum observed evaluation, median logged diagnostic KL, and the requested collapse flag: any saved evaluation below half that seed's initial evaluation, or any observed NaN/inf. Do not silently omit nonfinite values in medians; report undefined statistics explicitly. Stage 1 diagnostic medians are unavailable because those diagnostics were not collected. Distinguish a finite crash from a demonstrated collapse, and flag an unassessable initial-reference case explicitly.

Save figures/table with captions in `runs/<analysis-id>/candidates/`, with native source paths and all run statuses. Report commit/diff, effective-flag verification, R0 outcome, timing/resource usage and factual effects of each cumulative removal. Do not infer statistical equivalence or claim an exact final-action KL. Scientific outputs and curation remain separate approvals; no curated files or completed-experiment catalog entry are created before the study actually completes.

H200 execution verification: the existing pristine saved batch and update state were replayed on H200 NVL using the same differential harness. All 78 comparison records were bitwise identical (maximum absolute and relative differences zero), including all 20 minibatch updates and fixed-noise sampling; Python, NumPy, torch CPU and CUDA RNG states were preserved. The check took 62.84 seconds, allocated 65 seconds under Slurm job `17777408`. Driver `610.43.02` differs from the A6000 driver; the installed torch `2.4.0+cu121`, dependencies and numerical settings were unchanged. Same seed job `17777249` was retargeted to `gpu:h200nvl:1` and released after validation. See [gate](runs/verification_h200/verification.json), [launcher verification](provenance/h200_actual_gate_guard_review.json), [scheduler change](provenance/NC1_seed2_h200_retarget.json), and [release](provenance/NC1_seed2_h200_release.json). This is the sole hardware exception; the original run matrix remains the pre-exception plan, with actual hardware in each run manifest.
