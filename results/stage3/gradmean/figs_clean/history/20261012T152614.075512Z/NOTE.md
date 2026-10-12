# Clean figures, version2

**Mapping.** P0: state_3000.pt → 0 updates; D1: state_70.pt → 63 updates; D2: state_139.pt → 126 updates; N1: state_70.pt → 63 updates; N2: state_139.pt → 126 updates. Checkpoints save after the update phase; reward rollouts precede that phase. Final evaluation: iteration 130 → 117 updates → 9.36 million training environment steps.

**Sanity check.** DPPO final evaluation 4758.371 ± 108.293 (sample sd, three seeds); without_PPO final evaluation 4603.524 ± 19.143 (sample sd, three seeds).

**Sample sizes.** P0: V4/V5 K=3700,m=296000; V6 K=500,m=40000; D1: V4/V5 K=2800,m=224000; V6 K=500,m=40000; D2: V4/V5 K=3700,m=296000; V6 K=500,m=40000; N1: V4/V5 K=2800,m=224000; V6 K=500,m=40000; N2: V4/V5 K=3550,m=284000; V6 K=500,m=40000.

**Availability and limits.** All five points have V6 measurements. V1 is a warmed/frozen, full-batch-normalized measurement proxy for the evolving, per-chunk-normalized training estimator. The original adaptive stopping rule was replaced by exactly 10 new V6 groups per point for compute reasons, before seeing new V6 data; already-running jobs also count. The table distinguishes the V6 cohort from the larger shared-signal cohort. Figures and tables are uncurated candidates.

[Evaluation](eval_reward.png) · [Training](train_reward.png) · [Critic gradient noise](gradient_noise_V6.png) · [LOO](gradient_noise_V4.png) · [No baseline](gradient_noise_V5.png) · [Table](points_table.md). Captions accompany every PNG/PDF; [render record](render_record.json) lists exact inputs.
