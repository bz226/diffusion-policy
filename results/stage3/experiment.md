# Stage 3: from NC4 to projected SGD

**Question.** How do the estimator, optimizer and projection affect gradient noise, discounted return and fine-tuning stability?

**Answer.** All 30 training runs and 31 checkpoint evaluations are complete, with no observed collapses. E1 improved final scheduled eval return to 4670 ± 28 versus E0's 4532 ± 67 and E2's 4542 ± 81 (sample SD across three seeds); switching E1→E2 lowered both discounted and undiscounted training return. Calibrated SGD1 improved the final train-sampler discounted objective less than E2. The [curated full report](curated/stage3-final-results/REPORT.md) answers Q1–Q8; Phase-1 uncertainty remains inconclusive.

The user approved all final artifact groups A–C under the proposed names. The [curated index](curated/INDEX.md#final-comparison) links the final plots, report and tables; earlier curated snapshots remain available.

**Phase-1 uncertainty is unreliable.** The saved raw percentile bootstrap intervals for signal norm² are strongly upward shifted. CI-based positivity/resolution claims, derived confidence intervals and noise-scale lower bounds must not be used. Signed point moments and saved point-derived ratios/cosines below are provisional; no replacement intervals were computed. Training curves and across-seed sample SDs are unaffected. [Read-only statistical audit](runs/noise_ci_review.json).

The [curated phase comparison](curated/stage3-final-comparison.png) and [BATCH4 comparison](curated/stage3-batch4-comparison.png) show the completed evidence. All three seed values are required for means and sample-SD bands; undefined noise windows remain gaps. Noise estimates are approximate because normalization/LOO couples observations within batches, and shared evaluation seeds do not pair the underlying Gym initial states.

**Review.** Numerical reconstruction passed 2,118 checks with zero differences. Reporting fixes exclude evaluation rows from training action-range curves, preserve undefined noise-window gaps, and label projection skips explicitly. Original automatic outputs and previously curated artifacts remain unchanged. [Review record](runs/final_analysis_review/review_record.json).

**Setup.** HalfCheetah, pretrained DPPO actor, 140 iterations, three training seeds per condition; the final scheduled eval is itr130 and final checkpoints are additionally evaluated with both samplers. [Methods](methods.md), [gate](runs/gate/gate_summary.json), [per-seed table](curated/stage3-final-results/seed_results.csv), [condition table](curated/stage3-final-results/condition_results.csv), [run/command record](analysis_record.json).

**Decision.** Approved final figures, tables, captions and report are curated. [Approval and copy record](runs/final_analysis_review/curation_record.json). Phase-1 uncertainty remains unresolved; no replacement analysis or further experiments were performed.

<!-- stage3-status-start -->
Last verified: 2026-10-11T07:29:54.771441+00:00; campaign **complete**. All 30 training runs and 31 checkpoint-evaluation jobs finished; 64 jobs complete and six projection slots skipped. No missing outputs, non-finite results or observed collapse flags. Monitor paused. [Completion check](runs/manual_monitor_20261011T072954Z.json); [command journal](runs/campaign_commands.jsonl).
<!-- stage3-status-end -->
