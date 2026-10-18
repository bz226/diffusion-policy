# Methods: AdaGrad-Norm with post-normalization decision discount

## Question and scope

Does replacing AdamW by a single-scalar adaptive step preserve stable fine-tuning of `without_PPO_and_GAE + γ^k`? The direct comparison holds the estimator and its normalization fixed. Completed SGD3/SGD10 and E2 are contextual references: their LOO baseline and other estimator settings differ, so that comparison alone cannot isolate optimizer geometry.

The source is parent revision `03f82611746954839169a64305777c3d7f91b139`, the exact discounted-run revision, exported as ordinary files into this study. Development uses the separate `stage3-e1cg-an` worktree. Existing studies and their outputs are read only. [Approval and limits](provenance/approval.json).

## Actor update

For the accumulated actor loss gradient g, maintain one scalar b in float64 across all actor_ft parameters. On each actor update, compute b_new² = b_old² + sum_i g_i², then θ_new = θ_old − η g / b_new. Use the updated divisor immediately. There is no momentum, weight decay, per-coordinate rescaling or gradient clipping. Evaluation does not update b. Save b at every checkpoint, including its initial value at iteration 0, and preserve float64 when restoring it.

This is Algorithm 1 of [Ward, Wu and Bottou, JMLR 21 (2020)](https://www.jmlr.org/papers/volume21/18-352/18-352.pdf). Theorem 2.1 additionally assumes an unbiased gradient, smoothness and bounded variance/gradient. This empirical normalized, denoising-weighted estimator has not been shown to satisfy those assumptions; the experiment does not establish the theorem's applicability.

The unchanged estimator is γ^k · norm(G − c_start V_pre) · 0.99^(9−i), with mean coordinate log-probability. Normalization is within the same shuffled 50,000-pair chunks used by the reference update; the split diagnostic retains its existing full-batch convention. Rewards, critic targets and optimizer, sample generation, action chunks, noise floors, and diagnostics are inherited unchanged. [Reference methods](../e1cg/methods.md).

| Setting | Value | Rationale |
|---|---|---|
| Environments × decisions | 40 × 500 | User-fixed baseline |
| Episodes per fresh batch | 80, each 250 decisions | User-fixed sampling unit |
| Actor updates / run | 126 in 140 iterations | Evaluation-only iterations 0,10,…,130 |
| New actor optimizer | AdaGrad-Norm | User intervention; Algorithm 1 |
| η grid | d*, 3d*, 10d* | User-fixed grid; no further tuning |
| b₀ | 0.001 × median calibration gradient norm | User rule |
| Calibration | Seed 3000, 5 fresh batches | User protocol |
| KL* | 0.000584517336086918 | Fixed Stage-3 anchor |
| Training seeds | 0, 1, 2 | Independent runs; retained if unsuccessful |
| Final checkpoint evaluation | Seed 5000, 2 rollouts per sampler, 160 episodes each | Approved comparison with discounted-run reference |

## Calibration and gates

Use the existing calibrated-SGD trial procedure at the pretrained policy, with the configured estimator and frozen critic/scaler. Restore the actor after every trial. For each batch, retain η_b, norm(g_b), realized KL, trial history and exponent. Set d_b = η_b norm(g_b), d* = median(d_b), and report max(d_b)/min(d_b). The amendment fixes the executable η values to 0.00263955, 0.00791865, 0.0263955 and b₀ to 3.22129e-6; raw full-precision calibration values remain recorded, with no recalibration. Calibration does not train a policy. Exponents outside [1.5,2.5] and η spread above 3 trigger the inherited warning, not tuning.

G1 compares 20 CPU steps on 1,000-dimensional fixed random gradients with the float64 formula (relative 1e-6), including b₀ larger than the gradient. G2 requires identical CPU actor gradients under the old and new optimizers on saved inputs. The amended G3 replaces the original displacement check: G3a takes the saved first step with float64 actor/gradient/state and requires relative step-length error ≤1e-10. G3b checks the native float32 pre-add update length to relative 1e-6 and bounds every coordinate’s addition-rounding error by one float32 spacing at the resulting parameter. Report sqrt(sum(u² spacing²/12))/norm(u)² beside the original observed 2.308655985808447e-6 norm discrepancy. The original failed gate and calibration are preserved; G1/G2/G4 are reused. [Approved amendment](provenance/g3_amendment.json). G4 replays the default and discounted-reference CPU updates, comparing parameters, gradients, optimizer state and RNG. G5 runs 12 actual CPU iterations with the full environment/batch settings; lr must remain constant and evaluation must leave b unchanged. Stop on any failed gate.

## Execution and monitoring

The approved cap is 40 GPU-hours and 2 CPU gate wall-hours. Each training job has a 4-hour cap; calibration has 1 hour; each final evaluation has 10 minutes. Submit seed 0 of the three conditions first, then seeds 1 and 2, using up to nine separate A6000/40-CPU allocations as slots become available. At most one same-seed retry after fixing a failed job is allowed, within the remaining aggregate cap. Never change a run's scientific configuration to fit resources.

The reference runs took 8,617–8,708 seconds per seed (about 2h25). Nine runs project about 21.65 GPU-hours plus calibration/evaluation. Training is monitored every 30 minutes, max(30 minutes, estimated run duration / 6), with hard job/aggregate limits, explicit environment cleanup, and completion verification. Queue waiting is not a promised start time. A stall needs two consecutive unchanged progress checks. Stop and report gate failures; preserve failed training outputs and report the last iteration/diagnostics.

## Analysis fixed before execution

Reward x-values use training steps before the rollout, giving 9.36M at iteration 130. Checkpoints contain 0,32,63,95,126 updates. Report the requested three figures as point estimates only, PNG 300 dpi and PDF, with no error bars or bands; tables retain mean ± sample SD and all three seed values. Use pink #CC79A7 for AdamW and one light-to-dark blue hue for the AdaGrad-Norm grid.

For each run, average the last ten training updates. J_disc gain subtracts the existing pretrained train-sampler mean 1369.0240467260296, exactly as the reference report. Collapse means any ten consecutive training-update returns average more than 300 below the Stage-3 pretrained checkpoint-evaluation train-sampler return. Non-finite/crashed runs remain explicit. The differences against AdamW and SGD3 use pooled-SD × sqrt(2/3) standard errors; effects smaller than twice pooled seed SD are unresolved. Existing reference checkpoint evaluations keep their original sample counts.

For seed 0, inspect norm(Δθ_j) sqrt(j+1)/η, first KL relative to KL* × (η/d*)², subsequent KL versus 1/(j+1), and final distance versus η sqrt(H_126). The last expression is a noise-dominated displacement heuristic, not total path length or a guaranteed radius. Report observed deviations. All candidate artifacts require a separate curation decision.
