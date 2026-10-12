# Gradient noise of without_PPO_and_GAE

Each row uses the same 500 batches (40,000 episodes, ten independent jobs) for both estimators. Brackets are 90% delete-one-group jackknife intervals using t(0.95,9)=1.833113, with joint deletions for ratios and cosines. Each estimator uses its own pairwise squared-mean estimate. Only B is comparable across their different units. A dagger marks every estimate affected by the fitted scale at update126; intervals condition on that scale, so do not include its fitting uncertainty. Bounds indicate an unresolved signal; they are upper-signal plug-in bounds, not two-sided confidence intervals.

| Updates | Checkpoint | K | m | As run: ‖u‖ | Trace/episode | B | B/80 | Paper form: ‖u‖ | Trace/episode | B | B/80 | Mean-direction cosine | Critic EV measured | Critic EV logged | c and source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | state_0.pt | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -0.19543 | 1; exact_next_training_iteration_start |
| 32 | state_35.pt | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.47399 | 476.77; exact_next_training_iteration_start |
| 63 | state_70.pt | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.24997 | 480.38; exact_next_training_iteration_start |
| 95 | state_105.pt | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.25543 | 481.73; exact_next_training_iteration_start |
| 126 | state_139.pt | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A; requires_authorized_five_batch_least_squares_fallback |

As-run norm/trace use training-loss units/squared units; paper-form norm/trace use paper units/squared units; B is in episodes. Exact logged c and logged EV are fixed reference values without sampling intervals. Measured EV uses DPPO's scaled-reward GAE targets. The final policy has no next training rollout, so its logged EV is unavailable. Selected jobs and normalization are in [render_record.json](render_record.json).
