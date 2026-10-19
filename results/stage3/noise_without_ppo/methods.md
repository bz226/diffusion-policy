# Methods and approved boundaries

## Question and comparison

Measure the loss gradient handed to AdamW at without_PPO's original seed-0 checkpoints. On each batch compute GAE + critic and raw Monte Carlo return minus the same pre-update critic in raw units. The actor, critic, rollout, chunk permutation and sampling budget are paired; no parameter is updated. The second comparison uses the existing without_PPO_and_GAE and DPPO measurements at their own parameters, so it does not isolate the advantage estimator.

The user supplied this protocol and approved the normalization and scaler limitations below. This is a discrete 20-step diffusion-policy measurement, not an exact continuous-time gradient or ground truth for the mean direction.

| Setting | Value and units | Rationale |
|---|---|---|
| Source run | Original Stage-2 seed 0 at actor lr 0.0001 | Requested run; all five checkpoints exist |
| Checkpoint iteration / updates | 0/0, 35/32, 70/63, 105/95, 139/126 | Evaluation iterations do not update |
| Per batch | 40 environments × 500 decisions; 80 terminal episodes of 250 decisions | Original configuration |
| Per actor gradient | 200,000 decision/denoising pairs; four shuffled chunks of 50,000 | Native one-step update |
| Norm and weights | Native per-chunk sample mean/std; mean of 24 log-prob coordinates; denoising weights 0.99^(9−i) | Native source, replacing the brief's full-batch normalization |
| Discount | Decision gamma 0.99; GAE lambda 0.95; no outer gamma^k | Requested estimators |
| Sampling | Training noise floor 0.1, noise clamp 100, log-prob clamp off | Original without_PPO |
| Measurement size | 500 batches per point; ten independent groups of 50 | Requested fixed design |
| Preparation | No critic warm-up; five separate fitting batches at each later checkpoint | User-authorized fallback |
| Precision | Model/gradients float32; sums, inner products and statistics float64 | Preserve native computation and stable aggregation |
| Resources | ≤5 GPU jobs; ≤30 min/job; ≤30 GPU-hours overall | Approved cap; measured preceding collection used 10.22 GPU-hours |

## State and normalization

The source checkpoint restores the actor, frozen actor and critic. A requested critic hash check must match the checkpoint. Neither the original checkpoints nor their logs preserve the running reward scaler. At update 0 its initial state can be reconstructed; each fresh batch starts from that state and updates it before scaling the GAE rewards. The Monte Carlo baseline uses the divisor before that update. At later points fit a scalar divisor c by the existing least-squares, no-intercept fit of raw Monte Carlo returns to frozen V(s), over five fresh batches. Freeze c for collection and mark every affected measurement. This cannot reconstruct the historical scaler's mean/count/variance or its update on each batch. Intervals condition on the fitted c and omit its estimation uncertainty.

The actual training code normalizes each shuffled 50,000-pair chunk separately. It uses the fifth critic-epoch permutation for its one actor step. Both estimators use that same permutation and four chunk weights of one quarter. The older green reference used full-batch statistics after a looser approximation check; it is not an exact chunk-normalized reference. Saved-batch gates keep the requested relative L2 tolerance of 1e-6.

## Independence and rollout starts

Native `reset_at_iteration=false`, with auto-reset at every 250 decisions. Each 500-decision batch nevertheless starts at an episode boundary. Measurement explicitly reseeds the private MuJoCo generator and resets at that boundary, preserving the initial-state distribution while ensuring auditable independent streams. The native wrapper's `.seed()` alone does not reach that private generator. Record each environment, diffusion and shuffle seed; use disjoint job/batch blocks, including fitting and gate batches. The two gradients on a batch share data and permutation. Groups use independent streams.

## Aggregation

For K batch loss gradients g, store only float64 group vector sums and per-batch squared norms and cross products. Let S be the sum and Q the sum of squared norms. The estimates are

`mean_sq = (dot(S,S) − Q)/(K*(K−1))`

`trace_cov = (Q − dot(S,S)/K)/(K−1)`

`R = trace_cov/mean_sq`, `B = 80*R`, and expected batch-gradient cosine `(1+R)^(-1/2)`.

Use delete-one-group jackknife intervals with Student t at 90% and 9 degrees of freedom for ten completed groups. Recompute both components for ratios. Resolution uses the interval for mean_sq, not its square root: if it reaches zero, report the plug-in lower bound `R >= trace_cov/mean_sq_upper`. The corresponding expected-cosine bound is an upper bound. This bound fixes variance at its point estimate and is distinct from a joint ratio interval. Nonpositive point estimates remain signed in tables and cannot be drawn on logarithmic axes; report that limitation without adding samples.

For paired mean inner products exclude same-group products: sum over distinct groups of their vector-sum inner products, divided by sum over distinct groups of the group-size products. Use this same group estimator for both denominator squared norms of a cosine. Nonpositive norms in the full estimate or a jackknife replicate leave that cosine unresolved. Report paired trace ratios, R ratios only where both means resolve, and the average of per-batch gradient cosines.

## Gates and execution

1. On the full saved batch, run native without_PPO and native mc_critic updates up to the optimizer step, read actor gradients, and compare each with the measurement (relative L2 ≤1e-6).
2. Verify critic equality, known initial scaler behavior or explicitly flagged fallback, distinct streams and different first-batch episode returns, and explicit worker cleanup.
3. Replay original/default source and new source on identical CPU inputs; parameters, optimizer and RNG states must match bitwise.
4. Only after all gates pass, collect the fixed 50 jobs. A scientific gate failure halts admission. A failed job is retained; any corrected retry uses new seeds and must fit the approved cap.

Pseudocode: load one checkpoint; prepare its fixed state; for each independent group and batch, seed and collect 80 episodes, compute both actor advantages, replay identical chunk normalization and discounting, accumulate two gradient sums and three scalar products; close all environments; verify outputs; aggregate groups and render candidates.

## Files, preservation and interpretation

Scientific source is an ordinary-file snapshot in `dppo/`, developed in the dedicated detached worktree outside the study directory. The initial import is recorded in `provenance/dppo_source_snapshot.json`; final source revision and the exact gated snapshot will be recorded before collection. Results, logs and executable job specifications belong under this study's `runs/`.

The explicitly requested before/after hashes cover all existing PNG/PDF files in the two old noise-figure folders. Existing files are never overwritten. New candidates use the old shared whole-decade y-limits where possible; if needed, wider-axis comparison copies live only in this study. User-facing method names are without_PPO, without_PPO_and_GAE and DPPO. The paired labels are GAE + critic and Monte Carlo + critic. Curation is not part of run approval.
