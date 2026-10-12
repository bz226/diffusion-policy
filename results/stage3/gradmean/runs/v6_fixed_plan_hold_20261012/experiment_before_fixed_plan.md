# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** G1–G5 passed and collection resumed on October 12 under controller 17821138. The [administrative recovery](runs/administrative_recovery_20261012_e1c/README.md) accepts seven verified complete groups rejected by the obsolete branch check, preserving their failed wrapper records; two jobs that failed before sampling use the approved replacement seeds. The requested [Round 2 analysis](curated/round2-gradient-analysis.md) freezes all five policies at 1,400 batches each. V4's gradient signal is resolved at every point, but its estimated mean-direction cosine is only 0.75–0.81, below the 0.9 target. The [plot](curated/round2-gradient-signal-and-noise.png) and [estimate table](curated/round2-gradient-estimates.csv) are [curated with user approval](curated/INDEX.md). [Offline gate evidence](runs/gate_offline_G1/gate_summary.json), [amended G2](runs/gate_G2_G1/gate_summary.json), and [live gates](runs/gate_live_G1/gate_summary.json).

**Scope.** All five required checkpoints exist. The scientific source is an ordinary snapshot; this study never updates the actor. Sampling uncertainty is conditional on these seed-0 training paths. Round 2 supports smaller mean-gradient norms than at P0 and persistently large noise relative to an 80-episode training batch. [Methods](methods.md), [amendment](provenance/amendment_G1.json), [revisions](provenance/revisions.json), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json). The [original failed G2 evidence](provenance/report_before_G1_amendment.md) is preserved.

**Resources and monitoring.** E1C has priority on three separate A6000 allocations. Gradient-mean admissions are limited to five concurrent GPU jobs and further constrained by live free A6000, CPU and memory resources; no running job was stopped. The hard cap remains 125 GPU-hours, counting actual usage and outstanding reservations. Each collection job reserves one A6000 and 40 CPU cores for at most one hour, with explicit child cleanup. One-hour jobs have 30-minute monitors. The [continuation estimate](runs/administrative_recovery_20261012_e1c/monitor_interval_verification.json) is at most 121 remaining groups × 724 seconds per group ÷ five parallel jobs, approximately 4.87 hours; the campaign monitor now runs every 50 minutes, rounding up `max(30 minutes, remaining estimate/6)`. Current counters and monitor status are below. Rendering the Round 2 review used 1.4 CPU seconds and no new scientific samples.

**Next step.** Continue the approved collection and stopping checks at round boundaries. Round 2 artifacts are curated; retain this snapshot separately from later rounds. The critic study remains paused at the user's request.

<!-- gradmean-bootstrap-start -->
Gate transition monitor: 2026-10-12T05:56:36.874086+00:00. **blocked**; phase: stopped. Monitor paused.
<!-- gradmean-bootstrap-end -->

<!-- gradmean-status-start -->
Last monitor: 2026-10-12T08:13:32.718576+00:00. **blocked**. P0: closed_rho, 3200 batches; D1: open, 2300 batches; D2: open, 3200 batches; N1: open, 2300 batches; N2: open, 2800 batches. GPU hours used: 56.641; outstanding reserved: 5.000; cap:125. Monitor paused. Next: Review terminal records; no automatic expansion.
<!-- gradmean-status-end -->
