# Rewards and gradient noise with the run's own critic

**Question.** How does gradient noise evolve along without_PPO_and_GAE when measured in its training form and in the paper's form using the same checkpoint critic, reward scale and batches?

**Status.** Complete: the five exact-critic points each have 500 fresh batches (40,000 episodes). CPU gates, normalization selection, all 56 jobs, worker cleanup and the independently checked figures/table passed. [Results and limitations](figs_clean/NOTE.md); earlier outputs remain preserved.

**Existing evidence.** The as-run variance decreases from 0.004496 to 0.004209 in training-loss units² per episode (about 6.4%). At update63, relative variance is 99,952 episodes for the as-run estimator and 105,550 for the paper form, with overlapping 90% intervals [10,950,188,954] and [16,702,194,398]; their ordering is unresolved. The squared-mean signal is resolved only at updates0/63 for the as-run estimator and32/63 for the paper form. Positive norm estimates at other points do not resolve the signal; their B values are reported as plug-in lower bounds, not confidence intervals. [Figures and interpretation](figs_clean/NOTE.md), [complete estimates](figs_clean/points_table.md). The final point uses the authorized fitted reward scale, and its gradient intervals condition on that fitted value. Both reward-figure versions and all curated files remain unchanged.

**Design.** Exactly500 fresh batches per checkpoint,40,000episodes, critic warm-up0. The as-run gradient uses raw MC minus the critic, normalized advantages and averaged log-probability. The paper-form gradient adds decision discount and removes normalization/coordinate averaging/denoising weights. Each uses its own pairwise mean estimate, with90% group-jackknife t intervals. Only relative variance B is directly comparable between their different units. [Methods](methods.md#clean-figures-version-5), [run record](runs/clean_v5_20261012/).

**Resources and next step.** Aggregate gradient-mean use is 80.4122 GPU-hours against the approved 125-hour cap. The normalization choice was **full**, frozen before the 50 collection groups. The bounded queue is complete and its monitor is paused. The user approved both gradient-figure bundles, the estimates table and the note for [curation](curated/INDEX.md#clean-figures-v5-gradient-noise); their uncertainty and fitted-scale limitations are retained. [Verification record](runs/clean_v5_20261012/final_analysis/manifest.json). [Earlier curated results](curated/INDEX.md).

<!-- gradmean-status-start -->
Verified 2026-10-12T15:28:28.625135+00:00. **Complete**: all five policies have 500/500 new V6 batches; final analysis and clean figures passed. Total gradient-mean usage: 68.054 GPU-hours of 125. Monitor paused; no further collection queued.
<!-- gradmean-status-end -->

<!-- clean-v3-status-start -->
**Superseded by version 5.** Further admissions stopped at the user's request. All nine admitted middle-checkpoint groups and two preparations finished cleanly; 450 collected batches remain preserved and excluded from version-5 figures. [Final accounting](runs/clean_v5_20261012/v3_final_ledger.json).
<!-- clean-v3-status-end -->

<!-- clean-v5-status-start -->
**Clean figures version 5: complete.** All 56 jobs and final gradient figures/table passed checks. [Results](figs_clean/NOTE.md). Reward and curated artifacts were preserved; the collection monitor is paused.
<!-- clean-v5-status-end -->
