# Methods: post-normalization decision discount

This record implements the user's Stage-3 E1CG addendum and the approval of 2026-10-17. It specifies a comparison, not a hyperparameter search. The exact reference source is recorded in [snapshot provenance](provenance/dppo_source_snapshot.json); the previous study's [methods](../e1c/methods.md) define the unchanged estimator and critic.

## Estimator and comparison

Each fresh batch has 40 environments × 500 decisions = 80 episodes of 250 decisions. A decision executes four environment actions. Raw reward-to-go G is cut at episode ends and uses gamma = 0.99 per decision. The actor advantage before normalization is A = G − c_start V_pre(s), with both critic weights and the reward scaler divisor taken before the batch update.

The reference actor normalizes A separately within each shuffled 50,000-pair chunk (sample standard deviation plus 1e-8), multiplies by 0.99^(9−i), and averages coordinate log-probabilities. One optimizer step accumulates four weighted chunk gradients. The new actor uses 0.99^k × norm(A) × 0.99^(9−i), keeping the same chunk statistics and permutation. The existing split diagnostic retains its full-batch normalization convention; it applies the decision weight afterwards. Neither convention is silently changed to match the other.

The state-only critic continues its existing scaled-reward GAE training. Its initialization is untrained. Noise truncation is effectively disabled (100 standard deviations); log-probability clipping is off, PPO clipping is 1e6 and target KL is null. Actor loss is score; AdamW lr and scheduler minimum are both 1e-4. New flags default to the exact reference behavior. No deterministic-algorithm setting is enabled.

| Quantity / setting | Value and unit | Basis |
|---|---|---|
| Seeds | 0, 1, 2 per new condition | User's comparison design |
| Training | 140 iterations; evaluation every tenth iteration | User's fixed budget |
| Actor updates | 126 fresh batches | Derived from evaluation-only iterations |
| Final logged evaluation | Iteration 130, 117 updates, 9.36M environment steps | Pre-rollout update convention |
| Saved checkpoints | Iterations 0, 35, 70, 105, 139 | User's save frequency 35 |
| Checkpoint updates | 0, 32, 63, 95, 126 | Verify from source and completed logs |
| Decision discount | 0.99^k, k = 0…249 | User's analyzed finite-horizon objective |
| Primary comparison | Only the post-normalization decision weight changes | User's one-change design |
| Optional condition | mc_critic + discount, denoising weights 1, summed log-probability, no normalization | User's E2C scope; no new tuning |

The normalized update has a random scale and mean shift. Euclidean mean-gradient cosines from frozen-policy measurements indicate direction but do not predict the exact AdamW update or prove unbiasedness of the normalized algorithm.

## Part A: existing paired measurements

Use all completed groups with paired V2/V6 (and V4/V5) data. The five exact-critic checkpoints each have 10 groups × 50 batches = 500 batches. The warmed P0/D1/D2/N1/N2 cohorts also have 500 paired batches each. The older warmed C1 cohort has 9 groups / 450 batches; no extra collection is authorized.

Every cohort stores same-batch cross products, so use U(a,b) = (dot(sum s_a, sum s_b) − sum dot(s_a,b, s_b,b)) / (K(K−1)) for the numerator and both denominator terms. All arithmetic is float64. A cosine is U(a,b)/sqrt(U(a,a)U(b,b)). Delete one independent 50-batch group at a time; 90% intervals use the Student-t quantile at 0.95 with G−1 degrees of freedom. If either diagonal is nonpositive in the full sample or any deletion, the cosine is unresolved. Do not interpret the V2/V6 direction at a point if either available common-mean sanity interval (V4/V6 or V5/V6) excludes 1.

The own-critic collection's V2 uses raw G−cV with full-batch normalization. The approved earlier two-batch check found full/chunk direction cosines 0.999991 and 0.999994, with chunk/full norm ratios 1.000176 and 0.999930. This remains a proxy, not a replay of each training chunk. The older warmed collection instead uses MC returns of clipped scaled rewards minus V; report it separately. The own-critic final checkpoint uses a fitted divisor c = 438.775862; other divisors were logged. Do not pool these cohorts. Report existing point estimates above 1 without clipping, and label them as noisy U-statistic ratios.

## Verification gates

G1 checks explicit float64 normalized/discounted coefficients against the float32 update and split paths (relative 1e-6), preserving each pass's actual normalization. G2 verifies k = t mod 250 and rejection of misaligned boundaries. G3 uses test-only all-ones post-normalization weights to require bitwise CPU actor gradients and a complete update identical to the reference. G4 compares raw discounted coefficients with the explicit V6 definition (relative 1e-12); V6 is absent from the old source. G5 repeats default-path fixed-batch replay and reference-condition CPU parameter, optimizer and RNG equality. G6 compares existing checkpoint-evaluation outputs and RNG with segment logging on/off. Optional E2C additionally requires the earlier critic C1 check.

Reuse the saved Stage-3 batch `results/stage3/runs/gate_capture/nc4/training_batch.pt` and its accepted CPU old-log-probability replay. It contains rewards, boundaries, terminal flags, pre-update model/optimizer/RNG/scaler state. It is a saved baseline batch evaluated with the reference flags, not a rollout archived from the completed reference training run. Stop before GPU submission on any scientific gate failure.

## Part C: checkpoint evaluations

At checkpoints 35/70/105/139 of both primary methods and all three seeds, collect two train-sampler rollouts (160 episodes per checkpoint), plus six train-sampler rollouts at the pretrained policy. This totals 54 train batches. Add eval-sampler rollouts at final checkpoints for the requested final table. The checkpoint-evaluation extension accepts the requested two/six repeats and sampler selection while keeping old defaults intact. Optional E2C receives final checkpoint evaluation only. Existing E1/E2 final measurements are reused and their episode counts disclosed.

For each episode and each of the five 50-decision bins, average raw decision rewards. Report the across-episode mean and standard error. Subtract the pretrained bin mean, and define tilt T as the average gain in the first two bins minus that in the final two. Preserve existing checkpoint-evaluation outputs exactly when the new flag is off.

## Reporting and interpretation

Reward curves use pre-rollout training environment steps in millions; checkpoint comparisons use the number of updates contained in each checkpoint. The primary names in figures are `without_PPO_and_GAE` and `without_PPO_and_GAE + γ^k`; reuse green #009E73 and use purple #CC79A7 for the new method. Figures show point estimates only, with filled circles and no uncertainty bands; tables retain uncertainty.

For each training seed, compute last-ten-update train reward and J_disc gain. Gain subtracts the same pretrained training-sampler mean used by the previous report: 1369.024047 (240 episodes). Summaries use sample SD across three seeds. For a difference of method means, SE = pooled SD × sqrt(2/3); differences smaller than twice pooled SD are unresolved. No further significance tests or tuning. Keep failed runs and non-finite values explicit.

## Resource and file limits

Approved cap: 32 GPU-hours including optional E2C, gates if any GPU is required, checkpoint evaluations and same-seed retries. Each training seed reserves at most four hours; checkpoint jobs reserve bounded minutes. CPU verification is capped at one hour. A durable controller monitors at 30-minute intervals, enforces allocation accounting before submissions, and never modifies unrelated jobs. Existing analogous runs took 2h22–2h26 per seed; queue delay is unknown.

All new source and selected records live under `results/stage3/e1cg/`; development is isolated in the requested worktree. Existing studies are read-only references. Candidate outputs are not curated automatically. Record exact commands, configuration, seeds, source revision and relevant source snapshots with each run. Runtime logs remain local under explicit ignore rules.
