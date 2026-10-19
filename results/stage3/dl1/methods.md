# Methods: the missing PPO × advantage cell

## Question, algorithm and controls

This implements the user's Stage 3 addendum DL1. The comparison is a 2×2 of the entire PPO mechanism bundle (reuse, ratio clipping, KL stop, log-probability clamp and sampling-noise clamp) against GAE or Monte Carlo critic advantages. It does not isolate reuse from the other bundle components. No reference training is repeated.

Each batch has 40 environments × 500 decisions: 80 terminal episodes of 250 decisions, four environment steps per decision. Copy raw decision rewards before running reward scaling. Compute raw G_k=Σ_(t=k)^249 0.99^(t−k)R_t in float64, stopping at each episode boundary. The actor advantage is G_k−cV(s_k), where c is the reward scaler divisor at iteration start and V is the pre-update critic. Cast once to the normal training dtype. Normalize inside each shuffled 50,000-pair minibatch, then apply 0.99^(9−i) denoising weights. The actor uses the ordinary PPO ratio loss, not the score loss. There is no decision discount γ^k.

The critic uses the unchanged scaled-reward GAE λ=0.95 returns. Each epoch shuffles all 200,000 pairs; there are five epochs and four minibatches per epoch, unless the unchanged KL stop fires. The live gate must execute all 20 steps; this does not disable early stopping in training. AdamW, the learning-rate schedules, noise floors, x̂₀ clamp, ratio clipping ε=.01, coordinate log-probability clamp [−5,2], and sampling noise clip ±3σ all retain the Stage-1 configuration. `max_grad_norm` is unset and `reset_at_iteration=false`.

| Setting | Value / rationale |
|---|---|
| Source | `stage3-e1c` tip `35cb89ba46236860cc01738942d7e5d2dbad74e3`; already gated `mc_critic`, required by the brief |
| Added configuration | `+train.adv_estimator=mc_critic +train.baseline_diag=true`; requested definition |
| Seeds | 0, 1, 2; three independent training runs |
| Length and saves | 140 iterations; save every 35; checkpoints 0,35,70,105,139 |
| Rollouts | 80 episodes per training or evaluation iteration; evaluation every 10 iterations |
| New logging | Detached PPO-step scalars and rollout/distance/saturation diagnostics under `baseline_diag` in the multiple-step path; no change to defaults or one-step behavior |
| Runtime | 16 GPU-hours total; 4h/training attempt; 1 GPU-hour gates; 2h aggregate CPU-gate wall |

Pseudocode: capture pre-batch critic/scaler → collect the native DPPO rollout → calculate ordinary critic returns and separate MC actor advantages → perform native epoch/minibatch updates while observing detached scalars → save native checkpoints and result records. Evaluation does not update parameters. Diagnostics use a dedicated generator, never the training random stream, and do not modify gradients or optimizer state.

## Verification and isolation

Development uses a separate `stage3-dl1` worktree, while the main workspace remains on `main`. The snapshot helper exports ordinary committed source; no nested Git repository is created. Outputs are absolute paths under the main repository's `results/stage3/dl1`. Save the exact source revision, source snapshot, configurations, commands, seeds and dependency versions. Existing studies and source trees remain untouched.

G1 diffs resolved configurations with matched approved runtime/path overrides: only the two requested flags may differ. G2 checks float64 pre-cast advantages against explicit G−cV at relative 1e-12 and the existing implementation, plus exact float32 casts. G3 requires bitwise-identical λ=.95 critic returns and one critic minibatch step. G4 runs one full live training iteration and requires 20 steps, 50,000 pairs each, and nonzero mean |ρ−1| at steps 2–20. G5 directly compares a saved-batch CPU update against original `cc7234a`, including parameters, both optimizer states and all global RNG states. G6 runs the full PPO loop with diagnostics on/off on CPU and requires bitwise equality of those states. No deterministic-algorithm settings are enabled. Explicit vector-environment/worker cleanup must succeed.

## Metrics, analysis and boundaries

For reward curves, x is the number of batch updates **before** the rollout: itr1 has x=0 and itr130 evaluation has x=117 (9.36M training environment steps). Checkpoints at iterations 0/35/70/105/139 contain 0/32/63/95/126 updates. A training record's logged distance after the update belongs to x+1; pair its rollout J_disc with the pre-update distance instead.

The outcome table reports per-seed values and three-seed means ± sample SD (ddof=1). Own-start discounted gain is each run's last-ten-update J_disc mean minus its own itr1 J_disc; it does not use earlier reports' shared pretrained anchor. Last-five-evaluation means and last-ten-training-update means follow actual evaluation/training rows, not ten chronological iterations. Final distance can be read from checkpoints if not logged. Preserve distinct true post-update KL and PPO approximate KL definitions; do not silently compare them as one metric. Missing historical values are “not logged.”

PPO diagnostics retain per-step clip fraction, approximate KL, the fraction switched off by clipping, and mean absolute ratio displacement. Aggregate over executed steps; record early stops and counts. A pair is switched off when A>0 and ρ>1+ε, or A<0 and ρ<1−ε, using the normalized and denoising-weighted advantage. Legacy last-minibatch approximate KL keeps its original meaning; the new mean is a separate field.

For each two-cell contrast, pooled seed SD is sqrt((s_A²+s_B²)/2); a difference below twice that value is unresolved. The interaction contrasts two independent reuse differences; its equivalent pooled SD is sqrt((s_DPPO²+s_without_PPO²+s_without_GAE²+s_without_PPO_and_GAE²)/2). No artificial paired covariance or additional significance tests are introduced. The three runs per cell, not rollout episodes, are the independent units for these comparisons. Ratios, missingness and uncertainty remain visible in the tables; figures show requested point estimates only.

Produce the five requested figures (seed-0/three-seed reward curves, corresponding distance plots, and PPO diagnostic curves), captions, tables, Q1–Q4 and REPORT.md into candidates before publishing study-level links/copies. No new final-checkpoint evaluations, reference runs, sweeps or automatic curation are authorized. Stop on scientific gate failure. A failed training attempt can only be retried with the same seed after diagnosis and a documented fix, reserving the full attempt within the remaining budget.
