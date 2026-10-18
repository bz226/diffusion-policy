# AdaGrad-Norm on the discounted Monte Carlo critic estimator

**Question.** Does a single-scalar adaptive step train `without_PPO_and_GAE + γ^k` stably, and how does it compare with AdamW and the completed SGD references?

**Status: completed.** All nine training runs reached 140 iterations (126 actor updates), and all nine final checkpoints were evaluated with both samplers. G1–G5 passed. The calibrated d* is 0.0026396 and b₀ is 3.2213e-06.

**Results.** AdaGrad-Norm, η = d*: 18.302 ± 6.3185 J_disc gain; 0/3 collapsed; AdaGrad-Norm, η = 3d*: 37.211 ± 8.0505 J_disc gain; 0/3 collapsed; AdaGrad-Norm, η = 10d*: 51.779 ± 12.395 J_disc gain; 0/3 collapsed. Values are means ± sample SD across three seeds; the [report](REPORT.md) gives the comparisons and limitations.

[Methods](methods.md) · [Outcome table](an_table.md) · [Seed-0 curves](figs/an_curves_seed0.png) · [Three-seed curves](figs/an_curves_3seed.png) · [Steps, KL and distance](figs/an_steps.png) · [Verified analysis inputs](analysis_record.json).

**Curated results.** The user approved artifacts A–E on 2026-10-18. [Curated index](curated/INDEX.md) links the three figures (PNG/PDF), captions, results table and report. No further sampling is scheduled.

<!-- AN runtime monitor -->
**Execution status (2026-10-18T18:10:35.954339+00:00):** complete; 18/18 jobs verified complete in training.

GPU time: 20.406 hours used; 0.000 hours reserved; 40-hour combined cap. Active or queued: none.

[Live status](runs/status.json) · [Execution record](runs/campaign_training/campaign.json).
<!-- end AN runtime monitor -->
