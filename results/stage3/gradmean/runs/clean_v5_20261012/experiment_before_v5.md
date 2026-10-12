# Mean gradient and noise along DPPO and without_PPO paths

**Question.** How do rewards and gradient noise change along the DPPO, without_PPO and without_PPO_and_GAE seed-0 paths when every gradient point uses the same measurement sample size?

**Status.** The original five-point collection and version-2 analysis are complete and preserved in [curated results](curated/INDEX.md#clean-figures-v2). Version 3 adds the third method's middle and final checkpoints, then redraws the five requested figures and checkpoint table. The new collection is bounded to two preparations and twenty 50-batch jobs; it never updates the actor. [Version-3 plan and exact scope](runs/clean_v3_collection_20261012/plan.json).

**Evidence and limits.** All three seed-0 training runs finished. Checkpoints at iterations 70 and 139 contain 63 and 126 actor updates; the last evaluation precedes update 118, at 117 updates / 9.36M steps. Final seed-0 evaluation returns are 4636.27, 4624.09 and 4490.69, respectively. Original episode dispersion was not saved, so the pooled pretrained reward means can be drawn but their requested standard-error bars cannot be recovered. Fixed-checkpoint uncertainty does not measure training-seed variability.

**Essential setup.** Version 3 uses the earliest-launched ten complete V6-enabled groups at every point: 500 batches, 40,000 episodes. Every main panel and table row uses that same cohort, with 90% delete-one-group Student-t jackknife intervals. All-available V4 values are separate references only. The new points restore their own checkpoint critics, warm them for five batches and freeze critic and scaler. The verified collector source is unchanged; explicit C1→D1/C2→D2 administrative aliases use separate checkpoints, output folders and seeds. [Original methods](methods.md), [source and runtime verification](runs/clean_v3_collection_20261012/runtime_checks.json).

**Resources and monitoring.** Existing gradient-mean use is 68.054 GPU-hours of the approved 125-hour cap. The new work reserves at most 22 additional GPU-hours (one-hour cap per job) and is expected to use 4.2–4.5 GPU-hours, about 60–70 minutes at MAX_PAR=5 plus scheduling. Each worker receives one A6000 and 40 CPUs. The controller checks progress and records a scheduled monitor every 30 minutes; terminal failure stops new admissions without changing running jobs. No further adaptive rounds or automatic replacements are authorized.

**Next step.** Finish the two new 500-batch cohorts and regenerate [version-3 figures and table](figs_clean/NOTE.md). The new artifacts remain uncurated candidates. Earlier approved artifacts remain unchanged: [Round-2 analysis](curated/round2-gradient-analysis.md), [clean figures v2](curated/INDEX.md#clean-figures-v2).

<!-- gradmean-status-start -->
Verified 2026-10-12T15:28:28.625135+00:00. **Complete**: all five policies have 500/500 new V6 batches; final analysis and clean figures passed. Total gradient-mean usage: 68.054 GPU-hours of 125. Monitor paused; no further collection queued.
<!-- gradmean-status-end -->

<!-- clean-v3-status-start -->
**Clean figures version3 collection (2026-10-12T16:31:14.885784+00:00): blocked.** 6/22 jobs verified. The 30-minute monitor is paused. [Collection record](runs/clean_v3_collection_20261012/experiment.md).
<!-- clean-v3-status-end -->
