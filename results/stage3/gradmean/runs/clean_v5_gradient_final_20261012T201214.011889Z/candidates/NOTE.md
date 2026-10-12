# Clean figures, version 5

All five checkpoints have 500 measured batches. Both gradient figures use exactly the same checkpoints, critics, scales and batches.

Checkpoint state_0/35/70/105/139 contains 0/32/63/95/126 actor updates, verified from native counters and save order. The final evaluation is iteration130, 117 updates, 9.36M training env steps. DPPO uses the original Stage-1 seed0 run.

Final seed0 evaluation matches each native log: DPPO 4636.2733; without_PPO 4624.08735; without_PPO_and_GAE 4490.68811. Each curve starts from its own saved evaluation or first-training mean (80 episodes), without pooling. Episode spreads were not saved, so the requested starting-point standard errors are unavailable. DPPO retains its original ±3σ truncation of nominal Gaussian noise.

The critic has no warm-up. Scales come from the next training iteration's start log; the final scale uses the authorized through-origin least-squares fit over five fresh batches. The native initial c is 1.000000005 (nominal1 plus the scaler epsilon). All final-point estimates are marked †. Legacy V2 used clipped scaled returns; this collection explicitly uses raw Monte Carlo returns minus c times the frozen critic.

CPU gates and normalization checks: [gate record](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/gate_cpu/gate_summary.json) and [collection record](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012/experiment.md). F3 uses the approved identical-input old/new CPU replay with identical provenance. Each plotted point requires ten complete50-batch jobs. No unfinished groups or older warmed-critic data enter these figures.

| Updates | Predicted consecutive-gradient cosine | Logged mean ± s.e. | Logged n | Measured critic EV | Logged critic EV |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00058916 [0.0003552, 0.00082312] | -0.0017594 ± 0.020855 | 10 | -0.00060692 [-0.00061219, -0.00060165] | -0.19543 |
| 32 | N/A (upper 0.0012726) | -0.0068248 ± 0.0079753 | 21 | 0.37206 [0.35762, 0.3865] | 0.47399 |
| 63 | 0.00079974 [9.2857e-05, 0.0015066] | 0.0016482 ± 0.010766 | 21 | 0.31213 [0.30285, 0.32141] | 0.24997 |
| 95 | N/A (upper 0.00093814) | -0.0054056 ± 0.0095877 | 21 | 0.29268 [0.28027, 0.30509] | 0.25543 |
| 126 | N/A (upper 0.00064988)† | -0.0047856 ± 0.02155 | 10 | 0.29401 [0.28069, 0.30733]† | N/A |

The prediction is 1/(1+B_as_run/80). Log windows include ±10 pre-rollout update indices; endpoints are truncated to available data. The logged standard error is descriptive: successive gradients can be dependent and training changes the policy and critic. Fresh-batch EV uses the same GAE-target definition as training; update126 has no matching logged rollout.

| Updates | K | m | Jobs used |
|---:|---:|---:|---|
| 0 | 500 | 40000 | W0_000_j02;W0_000_j00;W0_000_j03;W0_000_j01;W0_000_j04;W0_000_j07;W0_000_j08;W0_000_j05;W0_000_j06;W0_000_j09 |
| 32 | 500 | 40000 | W0_035_j00;W0_035_j01;W0_035_j02;W0_035_j03;W0_035_j04;W0_035_j06;W0_035_j05;W0_035_j07;W0_035_j08;W0_035_j09 |
| 63 | 500 | 40000 | W0_070_j00;W0_070_j01;W0_070_j02;W0_070_j03;W0_070_j04;W0_070_j05;W0_070_j06;W0_070_j07;W0_070_j08;W0_070_j09 |
| 95 | 500 | 40000 | W0_105_j00;W0_105_j01;W0_105_j02;W0_105_j03;W0_105_j04;W0_105_j05;W0_105_j07;W0_105_j06;W0_105_j08;W0_105_j09 |
| 126 | 500 | 40000 | W0_139_j00;W0_139_j01;W0_139_j02;W0_139_j03;W0_139_j04;W0_139_j05;W0_139_j06;W0_139_j07;W0_139_j09;W0_139_j08 |

F1–F3 passed on CPU: checkpoint critic and supplied scales are exact; old/new final states and frozen-file bytes agree. Native quarter-gradient relative L2 difference: 3.0337206e-07.

Normalization selected **full**: batch0 cosine 0.999991242, quarter/full norm ratio 1.00017582; batch1 cosine 0.999994031, quarter/full norm ratio 0.999930496.

The superseded warmed-critic collection finished its already-running work: nine middle-checkpoint groups (450 batches) and two preparations. Those results remain separate and are not used here. No further jobs from that plan are launched. Earlier figures remain preserved in the figures history; curated artifacts are unchanged.

[Evaluation](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/figs_clean/eval_reward.png) · [Training](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/figs_clean/train_reward.png) · [As run](gradient_noise_as_run.png) · [Paper form](gradient_noise_paper_form.png) · [Table](points_table.md). Captions accompany each figure.

The existing reward artifacts were preserved byte for byte. Single-seed curves retain each run's individual starting value; [the three-seed reward comparison](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/figs_clean/REWARD_3SEED_NOTE.md) retains mean ± sample-SD bands. This finalization rendered and published only the gradient figures, table and this note.
