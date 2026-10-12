# Clean figures, version 5

Reward figures are complete; gradient measurements are pending at updates 0, 32, 63, 95, 126.

Checkpoint state_0/35/70/105/139 contains 0/32/63/95/126 actor updates, verified from native counters and save order. The final evaluation is iteration130, 117 updates, 9.36M training env steps. DPPO uses the original Stage-1 seed0 run.

Final seed0 evaluation matches each native log: DPPO 4636.2733; without_PPO 4624.08735; without_PPO_and_GAE 4490.68811. Pooled original evaluation/training means are 4298.783494/3806.733700 (240 episodes each). Episode spreads were not saved, so the requested starting-point standard errors are unavailable. DPPO retains its original ±3σ truncation of nominal Gaussian noise.

The critic has no warm-up. Scales come from the next training iteration's start log; the final scale uses the authorized through-origin least-squares fit over five fresh batches. The native initial c is 1.000000005 (nominal1 plus the scaler epsilon). All final-point estimates are marked †. Legacy V2 used clipped scaled returns; this collection explicitly uses raw Monte Carlo returns minus c times the frozen critic.

CPU gates and normalization checks: [gate record](../runs/clean_v5_20261012/gate_cpu/) and [collection record](../runs/clean_v5_20261012/). F3 uses the approved identical-input old/new CPU replay with identical provenance. Each plotted point requires ten complete50-batch jobs. No unfinished groups or older warmed-critic data enter these figures.

| Updates | Predicted consecutive-gradient cosine | Logged mean ± s.e. | Logged n | Measured critic EV | Logged critic EV |
|---:|---:|---:|---:|---:|---:|
| 0 | N/A | -0.0017594 ± 0.020855 | 10 | N/A | -0.19543 |
| 32 | N/A | -0.0068248 ± 0.0079753 | 21 | N/A | 0.47399 |
| 63 | N/A | 0.0016482 ± 0.010766 | 21 | N/A | 0.24997 |
| 95 | N/A | -0.0054056 ± 0.0095877 | 21 | N/A | 0.25543 |
| 126 | N/A | -0.0047856 ± 0.02155 | 10 | N/A | N/A |

The prediction is 1/(1+B_as_run/80). Log windows include ±10 pre-rollout update indices; endpoints are truncated to available data. The logged standard error is descriptive: successive gradients can be dependent and training changes the policy and critic. Fresh-batch EV uses the same GAE-target definition as training; update126 has no matching logged rollout.

| Updates | K | m | Jobs used |
|---:|---:|---:|---|
| 0 | 0 | 0 | pending |
| 32 | 0 | 0 | pending |
| 63 | 0 | 0 | pending |
| 95 | 0 | 0 | pending |
| 126 | 0 | 0 | pending |

The superseded warmed-critic collection finished its already-running work: nine middle-checkpoint groups (450 batches) and two preparations. Those results remain separate and are not used here. No further jobs from that plan are launched. Earlier figures remain preserved in the figures history; curated artifacts are unchanged.

[Evaluation](eval_reward.png) · [Training](train_reward.png) · [As run](gradient_noise_as_run.png) · [Paper form](gradient_noise_paper_form.png) · [Table](points_table.md). Captions accompany each figure.
