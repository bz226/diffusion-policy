# Rewards and gradient noise with the run's own critic

**Question.** How does gradient noise evolve along without_PPO_and_GAE when measured in its training form and in the paper's form using the same checkpoint critic, reward scale and batches?

**Status.** All three seed-0 training runs, CPU gates, five zero-warm-up preparations and the normalization check are complete. Full-batch normalization was frozen before collection. The first five collection jobs completed and passed checks: 250 batches, 20,000 episodes. Controller 17825855 failed only during a filesystem progress read; the scientific jobs exited successfully. The tested administrative recovery (17826164) is running with its 30-minute monitor active. The next five jobs are queued; it will admit only the remaining 45 jobs, with finalizer 17826166 waiting for its success. No scientific job is retried and no settings or budgets change. [Recovery evidence](runs/clean_v5_20261012/recovery_controller_17825855/). Version 3 outputs remain preserved.

**Existing evidence.** Final evaluations at 9.36M training steps are DPPO 4636.27, without_PPO 4624.09 and without_PPO_and_GAE 4490.69, matching native logs. The pooled starting means use 240 original episodes, but their standard errors cannot be recovered because episode spreads were not saved. The five checkpoints contain 0/32/63/95/126 updates; four starting reward divisors are logged and the final one requires the authorized five-batch fit.

**Design.** Exactly500 fresh batches per checkpoint,40,000episodes, critic warm-up0. The as-run gradient uses raw MC minus the critic, normalized advantages and averaged log-probability. The paper-form gradient adds decision discount and removes normalization/coordinate averaging/denoising weights. Each uses its own pairwise mean estimate, with90% group-jackknife t intervals. Only relative variance B is directly comparable between their different units. [Methods](methods.md#clean-figures-version-5), [run record](runs/clean_v5_20261012/).

**Resources and next step.** Existing use is70.0706GPU-hours. The fixed plan reserves at most28 more within the approved125-hour cap, with five parallel A6000workers and30-minute job caps. At 10:25 PDT on October 12, the completed jobs average 12.3 minutes. With five jobs running in parallel, collection is estimated to finish at 12:15–12:30 PDT and figures at 12:30–12:45 PDT, assuming no further delays. The recovered queue and 30-minute monitor are active; verify all 50 groups, then regenerate the figures/table. New artifacts remain uncurated candidates. [Earlier curated results](curated/INDEX.md).

<!-- gradmean-status-start -->
Verified 2026-10-12T15:28:28.625135+00:00. **Complete**: all five policies have 500/500 new V6 batches; final analysis and clean figures passed. Total gradient-mean usage: 68.054 GPU-hours of 125. Monitor paused; no further collection queued.
<!-- gradmean-status-end -->

<!-- clean-v3-status-start -->
**Superseded by version 5.** Further admissions stopped at the user's request. All nine admitted middle-checkpoint groups and two preparations finished cleanly; 450 collected batches remain preserved and excluded from version-5 figures. [Final accounting](runs/clean_v5_20261012/v3_final_ledger.json).
<!-- clean-v3-status-end -->

<!-- clean-v5-status-start -->
**Clean figures version 5 collection (2026-10-12T18:28:08.679591+00:00): running.** 34/56 fixed jobs verified; 30-minute monitor active. [Current collection record](runs/clean_v5_20261012/experiment.md).
<!-- clean-v5-status-end -->
