# Discounting decisions after advantage normalization

**Question.** Does weighting decision k by 0.99^k after advantage normalization improve the discounted objective or move gains toward earlier decisions?

**Answer.** The three-seed experiment found no resolved change in discounted-return gain, evaluation reward, training reward, or the early-versus-late reward tilt under the agreed reading rule. The discounted run made smaller parameter changes and had lower mean KL per update. This is inconclusive about a small performance effect; it does not establish equivalence.

| Metric | without_PPO_and_GAE | without_PPO_and_GAE + γ^k |
|---|---:|---:|
| Evaluation at 9.36M steps | 4503.9 ± 13.4 | 4496.8 ± 48.4 |
| Training return, last 10 updates | 4207.5 ± 48.4 | 4185.5 ± 62.7 |
| Discounted-return gain, last 10 updates | 112.6 ± 14.1 | 105.9 ± 21.3 |

Values are mean ± sample SD over three seeds. The discounted-minus-reference differences are −7.0 ± 29.0, −22.1 ± 45.7 and −6.7 ± 14.7 respectively (standard errors); all are smaller than twice pooled seed SD. [Full table](e1cg_table.md) · [Three-seed curves](figs/e1cg_curves_3seed.png) · [Q1–Q5 report](REPORT.md).

**Completed execution.** All three seeds reached 140 iterations, 126 updates and 10.08M training steps, with all five checkpoints saved and no crashes, non-finite results or collapsed seeds. All 25 checkpoint-evaluation jobs and the final analysis completed by **2026-10-17 07:25 PDT (14:25 UTC)**. Total use was **7.529 GPU-hours** of the 32-hour cap; training took 2h24–2h25 per seed. Seeds 0/1 shared the node on separate A6000s; seed 2 started after a slot freed. Optional E2C was skipped because the approved three-extra-slot condition was never met. The monitor is paused. [Completion verification](runs/completion_verification.json).

**Essential setup.** The primary intervention changes only the post-normalization decision weight. Raw Monte Carlo returns, the pre-update critic and reward divisor, normalization within 50,000-pair chunks, denoising weights, and one AdamW step at 1e-4 are fixed. Final-checkpoint comparisons use 160 episodes per sampler. [Methods, approved budget and deviations](methods.md).

**Validity and limits.** All six CPU gates and optional C1 passed; default replay and the unit-weight check were bitwise equal. Part A's five own-critic alignment estimates were 0.783/1.013/1.015/1.140/1.065, with all 90% intervals including 1. Those uncertain Euclidean estimates do not establish the proposed ≥0.9 alignment premise at every checkpoint or predict AdamW's trajectory. [Gate evidence](runs/gates/README.md) · [Alignment table](alignment_table.md) · [Source revision](provenance/execution_revision.json).

**Decision.** The user approved artifacts A–H for curation on 2026-10-17. The report, methods, four figure sets and three tables are saved with their captions and provenance in the [curated index](curated/INDEX.md). No further sampling is scheduled. [Artifact review and selection](ARTIFACT_REVIEW.md).

<!-- E1CG runtime monitor -->
**Execution status:** complete; 28 jobs verified complete, 6 optional jobs skipped. Monitor paused after completion. GPU time: 7.529 hours; no active or queued jobs.

[Live status](runs/status.json) · [Execution record](runs/campaign/campaign.json).
<!-- end E1CG runtime monitor -->
