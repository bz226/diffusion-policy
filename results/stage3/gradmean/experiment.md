# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** G1–G5 passed and production collection is running. The requested [Round 2 analysis](curated/round2-gradient-analysis.md) freezes all five policies at 1,400 batches each. V4's gradient signal is resolved at every point, but its estimated mean-direction cosine is only 0.75–0.81, below the 0.9 target. The [plot](curated/round2-gradient-signal-and-noise.png) and [estimate table](curated/round2-gradient-estimates.csv) are [curated with user approval](curated/INDEX.md). [Offline gate evidence](runs/gate_offline_G1/gate_summary.json), [amended G2](runs/gate_G2_G1/gate_summary.json), and [live gates](runs/gate_live_G1/gate_summary.json).

**Scope.** All five required checkpoints exist. The scientific source is an ordinary snapshot; this study never updates the actor. Sampling uncertainty is conditional on these seed-0 training paths. Round 2 supports smaller mean-gradient norms than at P0 and persistently large noise relative to an 80-episode training batch. [Methods](methods.md), [amendment](provenance/amendment_G1.json), [revisions](provenance/revisions.json), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json). The [original failed G2 evidence](provenance/report_before_G1_amendment.md) is preserved.

**Resources and monitoring.** The hard cap remains 125 GPU-hours, counting actual usage and outstanding reservations. Each collection job reserves one A6000 and 40 CPU cores for at most one hour, with explicit child cleanup. One-hour jobs have 30-minute monitors; the campaign's original 12–16 hour estimate gives a 2 hour 40 minute monitor, following `max(30 minutes, estimate/6)`. Current counters and monitor status are below. Rendering the Round 2 review used 1.4 CPU seconds and no new scientific samples.

**Next step.** Continue the approved collection and stopping checks at round boundaries. Round 2 artifacts are curated; retain this snapshot separately from later rounds. The critic study remains paused at the user's request.

<!-- gradmean-bootstrap-start -->
Gate transition monitor: 2026-10-12T05:56:36.874086+00:00. **blocked**; phase: stopped. Monitor paused.
<!-- gradmean-bootstrap-end -->

<!-- gradmean-status-start -->
Last monitor: 2026-10-12T05:56:37.404136+00:00. **blocked**. P0: closed_rho, 3200 batches; D1: open, 2300 batches; D2: open, 3100 batches; N1: open, 2300 batches; N2: open, 2300 batches. GPU hours used: 54.188; outstanding reserved: 9.000; cap:125. Monitor paused. Next: Review terminal records; no automatic expansion.
<!-- gradmean-status-end -->
