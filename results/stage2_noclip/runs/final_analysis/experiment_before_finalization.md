# Stage 2: DPPO clipping ablations

Which clipping mechanisms support stable HalfCheetah-v2 fine-tuning, and does their effect depend on reusing a batch for multiple actor updates?

**All NC2 and NC3 seeds crossed the collapse threshold; NC3 recovered to higher returns than NC2.** NC4 was stable at LR 1e-4, degraded at 1e-3 and collapsed at 3e-3. These completed subsets are plotted; the full comparison awaits the R0 and NC1 seed-1 replacements.

## Analysis: NC2 and NC3

| Condition | Last eval, mean ± SD | Collapsed seeds |
|---|---:|---:|
| Stage 1 baseline | 4,758 ± 88 | 0/3 |
| NC2 | −13 ± 833 | 3/3 |
| NC3 | 3,043 ± 497 | 3/3 |

![NC2 and NC3 evaluation and training returns](curated/halfcheetah-nc2-nc3-returns.png)

Evaluation and exploratory training returns use three independent seeds per condition: faint lines are seeds, bold lines arithmetic means, and bands ± population SD, not confidence intervals. All nine runs completed 140 iterations; last evaluation is at 9.36 million training steps and final training at 10.08 million, without smoothing or extrapolation. NC3 ends higher but does not restore baseline performance.

NC2 widens ratio clipping to 1e6 and removes KL early stopping, log-probability clamping and practical sampling-noise clipping. NC3 additionally replaces coordinate-mean with coordinate-sum log-probabilities in the PPO ratio. Collapse means any evaluation below half that seed's initial return or any scientific NaN/inf; all six ablation runs remained finite.

The [diagnostic plot](curated/halfcheetah-nc2-nc3-diagnostics.png) preserves one exception to the anticipated zero clip fraction: NC3 seed 1, iteration 31, approximately 1e-6. The prescribed finite bound was exceeded; this alone does not prove the clipped loss branch changed gradients. Post-update KL is a sampled denoising-transition statistic, not exact final-action KL. Its 52 exact zeros remain in summaries and are omitted only from logarithmic rendering.

[Per-seed table](curated/halfcheetah-nc2-nc3-seed-results.csv), [captions](curated/halfcheetah-nc2-nc3-captions.md), [verification](runs/nc23_analysis/verification.json), and [sources/command](runs/nc23_analysis/analysis_record.json) accompany the curated artifacts. [Methods](methods.md#nc2-and-nc3-plots-before-full-stage-2-analysis) document the unchanged aggregation and checks. NC2's joint removals do not isolate individual mechanisms. The approved figures and table are [curated](curated/INDEX.md#nc2-and-nc3); source candidates remain preserved.

## Analysis: NC4 learning rates

Last evaluation means ± SD were 4,604 ± 16 at LR 1e-4, 3,726 ± 404 at 1e-3 and −98 ± 477 at 3e-3; collapse counts were 0/3, 0/3 and 3/3. All 1,134 actor updates had pre-step ratio exactly 1 and clip fraction 0. Avoiding repeated updates did not prevent collapse at the largest learning rate; NC4 also changes update count and gradient aggregation, so reuse alone is not isolated.

Approved [return plots](curated/halfcheetah-nc4-returns.png), [diagnostics](curated/halfcheetah-nc4-diagnostics.png), [table](curated/halfcheetah-nc4-seed-results.csv), [captions](curated/halfcheetah-nc4-captions.md), [verification](runs/nc4_analysis/verification.json) and [methods](methods.md#nc4-subset-analysis-requested-before-campaign-completion) remain preserved in the [curated index](curated/INDEX.md).

## Execution and remaining work

My earlier scheduler handoff [interrupted R0 and NC1 seed 1](provenance/parallel_handoff_failure.json) at 138/140 iterations. Their checkpoints lacked the state needed for faithful continuation; raw outputs survive. Two [authorized fresh replacements](provenance/interrupted_recovery_authorization.json) started October 9 at 10:15 p.m. PDT, capped at 2h15m each; current completion estimate is October 10 around 12:17–12:20 a.m. Completed seeds were not repeated.

Scientific source remains `ab46b150fa34b5a5b457cd4062cd4c5ad830d964`; the [fixed-batch gate](runs/verification_full_update/verification.json) passed with zero CPU/A6000 differences and isolated diagnostic RNG. NC2–NC4 used A6000; only NC1 seed 2 used the verified H200 exception. The [58.5 GPU-hour budget](provenance/interrupted_recovery_budget.json), deadline, 40 workers and 30-minute monitors remain unchanged.

<!-- campaign-status-start -->
Original campaign terminal: 17 ablations completed; two interrupted attempts remain preserved. Original allocations ended before recovery began; their monitors are paused. [Verified status](runs/monitor_checks/20261010T052335_status.json).
<!-- campaign-status-end -->

<!-- recovery-status-start -->
Recovery (2026-10-10T07:19:02+00:00): runs_terminal_analysis_pending; R0_recovery_seed0: complete, NC1_recovery_seed1: complete; next: review terminal records. [Run record](runs/recovery_campaign.json).
<!-- recovery-status-end -->
