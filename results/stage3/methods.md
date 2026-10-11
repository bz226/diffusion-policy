# Stage 3 methods and execution boundary

This is the user-specified Stage-3 design, approved with all three optional training conditions and a 200 GPU-hour cap. It is not a claim that empirical checks establish a convergence theorem. The analyzed algorithm is defined by the task; no separate theoretical-paper citation was supplied. DPPO source provenance is upstream `cc7234ad7ff39a8f32de3af903606723a16f0648`, followed by the clean Stage-2 tip `ab46b150fa34b5a5b457cd4062cd4c5ad830d964`. The new branch is `stage3-theory`; its passed source revision will be recorded with the gate.

## Objective and sampling units

One decision executes four HalfCheetah environment steps. A complete episode has H=250 decisions, each with raw reward R_k equal to the sum of those four rewards. The finite-horizon objective is the expected sum of gamma^k R_k, with gamma=0.99 per decision. Every ordinary batch contains 500 decisions from each of 40 environments: 80 complete episodes and 200,000 (decision, fine-tuned-denoising-step) pairs. BATCH4 collects 2,000 decisions per environment, 320 episodes and 800,000 pairs.

MC reward-to-go is computed in float64 by backward recursion, cutting at `firsts[t+1]`. Episode boundaries must be exactly 0, 250, ..., T for every environment. The within-episode index is k=t mod 250. The LOO baseline at k averages G_k over all other episodes in that batch, including the other complete episode from the same environment. Actor advantages are formed independently of the unchanged GAE critic targets.

The score is the sum of the Gaussian transition scores over all 24 action coordinates and the last ten denoising transitions. The first ten of twenty transitions remain frozen. With E2 flags the code minimizes the average of `-gamma^k (G_k-b_k) logp` over all pairs. Its loss gradient is minus the task's episode-average estimator divided by 2,500; SGD's theoretical step size is therefore eta/2,500, including BATCH4. The x0 clamp and both 0.1 noise floors remain in place.

## Training comparisons

| Setting | Value and unit | Basis |
|---|---|---|
| Environment, software, pretrained files | Same HalfCheetah setup and Python environment as Stage 1/2 | Task constraint |
| Parallel environments | 40 per run | Task constraint |
| Episode horizon / action chunk | 250 decisions / 4 environment steps | Task constraint, asserted in captured data |
| Discount | 0.99 per decision | Task objective |
| Fine-tuned transitions / coordinates | 10 / 24 | Task constraint |
| Iterations / saved checkpoints | 140 / 0, 35, 70, 105, 139 | Task constraint |
| Independent training repetitions | Seeds 0, 1, 2 per condition | Task design |
| Actor gradient chunks | 50,000 pairs; weight each by its fraction of the full batch | NC4 implementation and task constraint |
| Critic updates | Five epochs, unchanged GAE, optimizer and targets | Task constraint |
| AdamW | lr=1e-4, beta1=0.9 except E2_beta0, beta2=0.999, wd=0 | Task design |
| SGD | Momentum, dampening and weight decay zero; constant calibrated eta | Task design |
| Projection | Euclidean ball centered on the initial actor_ft | Task design |
| Noise measurement | 25 batches × 40 environments per policy; five variants | Task design |
| Bootstrap | 2,000 whole-batch resamples; percentile 90% intervals | Numerical convention fixed before execution |
| Calibration | Five fresh batches, seed 3000, fixed 20,000 pairs per batch | Task design |

Every training condition uses the NC4 skeleton: one actor step, ratio bounds 1e6, no KL stop, no coordinate-logprob clamp, noise bound 100, score loss and Stage-3 diagnostics. E0 retains GAE, scaled rewards, per-chunk advantage normalization, mean logprob and denoising discount 0.99. E1 changes actor advantages to raw MC returns with LOO. E2 also adds gamma^k, sets denoising weights to one, sums coordinate logprobs and disables advantage normalization. E2_beta0 removes Adam's first-moment averaging; E2_nobase removes LOO. SGD1/3/10/03 use E2 with eta*, 3eta*, 10eta* and eta*/3.

PROJ3/10 match SGD3/10 and add the fixed radius. A projection condition is skipped only if all three corresponding complete SGD runs have maximum pre-projection distance below 0.8R. An incomplete SGD seed cannot establish this skip. BATCH4 matches SGD3 while increasing the sample count fourfold. It consequently has four times as many environment steps at equal iteration count; it is not an equal-sample-budget comparison.

The algorithm order is:

1. Collect a fresh batch using the existing sampler; copy raw rewards before scaling.
2. Compute existing GAE/value targets. Form the selected actor coefficients separately.
3. At the old actor, measure split gradients and saturation without changing model state, gradients or global RNG.
4. Run the unchanged critic epoch/minibatch loop. Accumulate the one actor gradient over the final permutation's chunks and apply one optimizer step.
5. Project actor_ft if requested; log actual gradient, displacement, distance and post-update transition diagnostics.
6. Save the existing checkpoint/result formats and advance the unchanged scheduler. Equal scheduler min/max must give the configured constant actor lr.

## Defaults and gate

Defaults remain ratio loss, GAE, no decision weighting, AdamW beta1=0.9, no projection, no Stage-3 diagnostics and train mode. Existing Stage-2 flags and its dedicated advancing diagnostic generator are preserved. All new active settings are printed as `STAGE3_SETTINGS`; the worker checks them against the requested overrides. YAML files are unchanged.

The fresh capture at the unchanged Stage-2 tip includes default and NC4 batches, raw/scaled rewards, firsts, terminated/truncated histories, actions, model/optimizer/scheduler/scaler state and Python/NumPy/CPU/CUDA RNG state. The source was not edited until both captures finished. Their 40 environments each had boundaries [0,250,500], with 80 terminated and 80 truncated episode ends per batch.

The gate checks both complete update paths on CPU and GPU, each minibatch loss, actor/critic gradients, final parameters and optimizer states, logged values, all denoising transitions from a fixed noise tape and global RNG states. CPU old logprobs are recomputed once by the unchanged CPU reference from the saved chains and shared identically by both replays; this avoids treating a device change as a policy change. CPU requires bitwise equality; GPU uses atol=1e-7, rtol=1e-5. The specified B1–B10 mathematical, projection, scheduler and inert-diagnostic checks are additional requirements; B2b and B6 are report-only. A gate failure blocks the campaign and is not retried automatically.

## Fixed-policy noise measurements and uncertainty

At the pretrained EMA policy and NC4 1e-4 seed 0's final checkpoint, warm the critic for 20 and five batches respectively with the actor frozen. Then freeze both critic and reward-scaler statistics and collect 25 fresh train-sampler batches. Each environment contributes a gradient averaged over its 5,000 pairs for each variant:

| Variant | Actor coefficient and reduction |
|---|---|
| V1 nc4 | Scaled GAE(0.95), full-batch normalized, mean coordinates, denoising weight 0.99 |
| V2 mc_critic | Scaled MC return minus the warmed critic, otherwise V1 |
| V3 mc_loo | Raw MC minus full-batch LOO, otherwise V1 |
| V4 theory_loo | Raw MC minus LOO, gamma^k, sum coordinates, no normalization, weights one |
| V5 theory_nobase | V4 without LOO |

Full-batch normalization uses the sample standard deviation of the repeated pair-level advantage array. Split-gradient diagnostics use these same full-batch coefficients/statistics, partition the environments into halves, and normalize each half by its own sample count. They do not recompute the LOO baseline separately within each half.

For n=1,000 environment gradients xi, compute their mean gbar, trace covariance `(sum ||xi||² - n||gbar||²)/(n-1)` and signal estimate `||gbar||² - trCov/n`. Cross-signal inner products use the corresponding cross-covariance correction. Report B_env=trCov/signal, B_ep=2B_env, training SNR=40/B_env and expected cosine `(1+B_env/40)^(-1/2)` when positive signal is resolved.

**Dependence limitation:** shared normalization couples V1–V3 across environments, and LOO couples V3–V4. The specified n-based moment correction is approximate for those variants. Resampling whole batches preserves within-batch coupling for uncertainty but does not remove centering bias. Store per-batch vector sums and scalar cross-products, and bootstrap through their Gram matrix; do not store all individual parameter gradients.

Retain signed signal estimates. If the signal interval includes zero or positive signal is otherwise unresolved, derived point ratios/cosines are null, not epsilon-clipped. The reported B_env lower bound uses the nonnegative variance fifth percentile divided by the positive signal 95th percentile, combining two one-sided 95% percentile bounds. Undefined cosine quantities have an explicit explanation. Estimated direction cosines outside [-1,1] are preserved as estimation uncertainty, not silently clipped.

**Observed uncertainty limitation and reporting.** The [post-run audit](runs/noise_ci_review.json) found that these percentile intervals are strongly upward shifted; the confidence-based resolution flags, intervals and lower bounds above are not reliable for these measurements. The [curated Phase-1 analysis](curated/phase1-results/analysis.md) therefore displays saved signed moments and point-derived ratios only, without error bars or confidence claims. Negative squared-signal estimates remain negative; corresponding ratios are unavailable, and corrected cosines outside [-1,1] are explicitly marked. No replacement uncertainty estimator or new sampling has been used. The source field `critic_explained_var_after_warmup` describes the first fresh frozen-critic measurement batch, not a rescore of the final warm-up batch.

Along training, ten-iteration windows estimate B_env as `10 * sum(split_diff_sq)/sum(split_dot)`. A nonpositive denominator is marked unresolved. The multiplier remains 40/4 because the split is over 40 environment rollouts, including BATCH4.

## Calibration, projection and evaluation

KL*=0.000584517336086918 is the pooled median of 378 NC4 1e-4 training-iteration transition-KL estimates. Each calibration trial temporarily applies the E2 gradient to the pretrained actor, uses the same 20,000 pairs for that batch, then restores every actor parameter. Begin with displacement norm 1e-4 times the actor norm, bracket KL in [1e-6,1e-2] by decades, predict the matched step quadratically, and use at most one log-log secant correction. Report the decade exponent and five eta values; warnings outside the prescribed exponent/spread limits do not trigger tuning.

Stage-1 final actor distances are 6.858459063553909, 6.935898417571779 and 6.948820395760235. Thus R=8.686025494700294, with no fallback. The initial actor is the pretrained checkpoint's EMA `network.*`, matched to final `model.actor_ft.*`; all six checked Stage-1/NC4 frozen actors match it exactly. Distances cast tensors to float64 before subtraction. The NC4 distances are 0.9097471263899602, 0.8784519424117502 and 0.8951921299172264.

Final-checkpoint evaluation uses seed 5000 and 80 episodes for each sampler, with no optimizer steps. The pretrained policy gets three train/eval rollout pairs; the first pair shares the policy-RNG prefix with the final-checkpoint jobs. The existing DDPM eval flag still adds noise before t=0. **Shared seeds do not fully pair Gym initial conditions:** the unchanged wrapper seeds global NumPy rather than the underlying Gym environment's private generator. This limitation is reported; environment seeding is not changed to hide it.

Discounted-return standard errors use episode sample SD divided by sqrt(episode count). Across independent training seeds, summaries and bands use sample SD (ddof=1). Differences smaller than twice the pooled seed SD are called unresolved; no further significance tests are run. Collapses are any eval return below half that seed's initial return, or an actual NaN/inf. Structurally undefined first-iteration/zero-gradient cosines use null with a reason and do not count as numerical collapse. Incomplete seeds remain explicit; no survivor-only condition mean is presented as a three-seed result.

Saturation measures the pre-clamp x0 prediction, averaged equally over ten steps on a dedicated 20,000-pair subsample, and separately at t=0. `action_oor_frac` refers to actual float32 actions passed to Gym after the unchanged affine wrapper transformation; `policy_action_oor_frac` additionally records normalized policy coordinates. These transformations only affect diagnostics.

## Execution, records and artifacts

All new files are under `results/stage3/`. The Python/MuJoCo installation and pretrained data are reused read-only. OMP/MKL/OPENBLAS thread variables stay unset. Separate Slurm jobs each receive one A6000, 40 CPUs and 64 GiB; at most nine GPU jobs are admitted together. The controller includes other allocations in admission checks, submits each condition's three seeds together, reserves all submitted caps within 200 GPU-hours and never automatically retries failures. See [approval](provenance/approval.json), [queue plan](runs/queue_plan.json), [anchors](provenance/anchors.json) and the generated full job files.

Central duration estimates of three hours per normal run and twelve hours for BATCH4 give scheduled monitoring intervals of 30 minutes and two hours. Each monitor checks progress, failures, GPU use and limits; two scheduled checks without progress flag a suspected stall. Workers signal only their own process groups. Controller interruption stops new admissions without cancelling worker allocations. Terminal outputs and finite checkpoints are verified before success is recorded.

The requested final report, all-run CSV, condition tables and figures will contain gate evidence, noise results, calibration, Q1–Q8, the deviations table, paths and exact launch commands. Figures go to the explicitly requested `figs/` destination. No artifact is copied into `curated/` without a separate decision. Large checkpoints, gate tensors, caches and noise vector sums stay on disk but outside Git.
