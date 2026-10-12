# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** Collection completed: all 50 planned V6 groups succeeded, giving 500 new batches at each point. No further sampling is queued. E1C also completed all three seeds. The final numerical analysis ran, but report formatting failed with an undefined `fixed_plan` name. A two-line context fix passed 21 checks and a replay of the saved report inputs; CPU-only postprocessing job `17825116` is now rerunning into a new directory. [Recovery record](runs/postprocess_recovery_20261012/recovery.json). The failed output and traceback are preserved. The [curated Round 2 analysis](curated/round2-gradient-analysis.md), [plot](curated/round2-gradient-signal-and-noise.png) and [table](curated/round2-gradient-estimates.csv) remain unchanged.

**Scope.** All five required checkpoints exist. The scientific source is an ordinary snapshot; this study never updates the actor. Sampling uncertainty is conditional on these seed-0 training paths. Round 2 supports smaller mean-gradient norms than at P0 and persistently large noise relative to an 80-episode training batch. [Methods](methods.md), [amendment](provenance/amendment_G1.json), [revisions](provenance/revisions.json), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json). The [original failed G2 evidence](provenance/report_before_G1_amendment.md) is preserved.

**Resources and monitoring.** The user replaced adaptive stopping for compute reasons at 08:13 UTC on October 12, before new V6 data: stop old admissions, retain all running jobs, then collect exactly ten new 50-batch V6 groups per point in order P0, D2, N2, D1, N1. There are no further adaptive rounds. [Decision record](provenance/fixed_v6_plan_20261012.json). All five previously running collection jobs finished naturally; their data count. E1C keeps priority on three A6000s; gradient-mean uses at most five further allocations, each one A6000/40 CPUs with a one-hour cap. The aggregate 125 GPU-hour limit remains. Existing usage is 57.673 GPU-hours; all 50 one-hour reservations fit below that limit. Comparable completed jobs predict about 10.1 additional GPU-hours and 2.1 hours elapsed at five concurrent workers. A 30-minute scheduled monitor checks jobs, counters, errors and resources, satisfying `max(30 minutes, estimate/6)`.

**Next step.** Verify CPU postprocessing, generate clean figures v2, and update completion records. Sampling monitors are paused after collection ended; the analysis recovery is being checked directly. All new outputs remain uncurated pending selection.

<!-- gradmean-bootstrap-start -->
Gate transition monitor: 2026-10-12T05:56:36.874086+00:00. **blocked**; phase: stopped. Monitor paused.
<!-- gradmean-bootstrap-end -->

<!-- gradmean-status-start -->
Last monitor: 2026-10-12T10:51:23.054548+00:00. **complete_with_analysis_failures**. P0: closed_rho, 3200 legacy batches, 500/500 new V6 batches; D1: open, 2300 legacy batches, 500/500 new V6 batches; D2: open, 3200 legacy batches, 500/500 new V6 batches; N1: open, 2300 legacy batches, 500/500 new V6 batches; N2: open, 3050 legacy batches, 500/500 new V6 batches. GPU hours used: 68.054; outstanding reserved: 0.000; cap:125. Monitor paused. Next: Review terminal records; no automatic expansion.
**Analysis incomplete:** 1 CPU analysis job(s) failed; see campaign records and saved tracebacks.
<!-- gradmean-status-end -->
