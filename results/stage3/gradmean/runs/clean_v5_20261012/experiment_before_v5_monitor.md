# Rewards and gradient noise with the run's own critic

**Question.** How does gradient noise evolve along without_PPO_and_GAE when measured in its training form and in the paper's form using the same checkpoint critic, reward scale and batches?

**Status.** All three seed-0 training runs are complete. The new five-point measurement is in verification: CPU gate17825755 compares old/new default preparation and checks the zero-warm-up path. New scientific collection starts only after it passes and the normalization comparison selects a permitted path. Version3 is superseded; its finished outputs remain preserved.

**Existing evidence.** Final evaluations at9.36M training steps are DPPO4636.27, without_PPO4624.09 and without_PPO_and_GAE4490.69, matching native logs. The pooled starting means use240 original episodes, but their standard errors cannot be recovered because episode spreads were not saved. The five checkpoints contain0/32/63/95/126updates; four starting reward divisors are logged and the final one requires the authorized five-batch fit.

**Design.** Exactly500 fresh batches per checkpoint,40,000episodes, critic warm-up0. The as-run gradient uses raw MC minus the critic, normalized advantages and averaged log-probability. The paper-form gradient adds decision discount and removes normalization/coordinate averaging/denoising weights. Each uses its own pairwise mean estimate, with90% group-jackknife t intervals. Only relative variance B is directly comparable between their different units. [Methods](methods.md#clean-figures-version-5), [run record](runs/clean_v5_20261012/).

**Resources and next step.** Existing use is70.0706GPU-hours. The fixed plan reserves at most28 more within the approved125-hour cap, with five parallel A6000workers and30-minute job caps. Expected collection time is about2.1hours plus gates and scheduling; monitoring runs every30minutes. Finish gates, freeze the normalization choice, collect and verify all50groups, then regenerate the figures/table. New artifacts remain uncurated candidates. [Earlier curated results](curated/INDEX.md).

<!-- gradmean-status-start -->
Verified 2026-10-12T15:28:28.625135+00:00. **Complete**: all five policies have 500/500 new V6 batches; final analysis and clean figures passed. Total gradient-mean usage: 68.054 GPU-hours of 125. Monitor paused; no further collection queued.
<!-- gradmean-status-end -->

<!-- clean-v3-status-start -->
**Superseded by version 5.** Further admissions stopped at the user's request. All nine admitted middle-checkpoint groups and two preparations finished cleanly; 450 collected batches remain preserved and excluded from version-5 figures. [Final accounting](runs/clean_v5_20261012/v3_final_ledger.json).
<!-- clean-v3-status-end -->
