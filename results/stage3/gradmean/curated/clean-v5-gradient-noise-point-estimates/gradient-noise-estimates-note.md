# Gradient-noise figures: point estimates

Both figures now show five filled-circle point estimates with identical two-decade log spans in every panel. All ten squared-mean point estimates are positive. The collection is unchanged: 500 batches / 40,000 episodes / ten groups per checkpoint.

**Uncertainty.** The 90% squared-mean intervals reach zero at updates 32, 95, 126 for as run, and 0, 95, 126 for paper’s form. Their B estimates remain plotted; all six B intervals are [0, ∞). The table reports squared-mean intervals directly, including negative endpoints. Intervals use the same delete-one-group jackknife and t(0.95, 9) = 1.833113. For an unresolved denominator, the upper ratio endpoint is infinite and the joint-ratio lower endpoint is intersected with B ≥ 0; the previous positive plug-in bounds are not used.

| Updates | Logged consecutive-gradient cosine ± s.e. | n | Prediction 1/(1 + B/80) | Measured critic EV (90% interval) | Logged critic EV |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.001759414 ± 0.02085489 | 10 | 0.0005891604 | -0.00060692 [-0.00061219, -0.00060165] | -0.19543 |
| 32 | -0.006824841 ± 0.007975331 | 21 | 0.0004076213 | 0.37206 [0.35762, 0.3865] | 0.47399 |
| 63 | 0.001648222 ± 0.01076613 | 21 | 0.0007997443 | 0.31213 [0.30285, 0.32141] | 0.24997 |
| 95 | -0.00540565 ± 0.009587713 | 21 | 0.000387516 | 0.29268 [0.28027, 0.30509] | 0.25543 |
| 126 | -0.004785618 ± 0.02155022 | 10 | 0.0002243435 | 0.29401 [0.28069, 0.30733] | N/A |

Each prediction lies within one logged standard error. Windows are ±10 pre-rollout update indices, truncated at the endpoints; the first undefined cosine is excluded. These standard errors are descriptive because successive gradients can be dependent and training changes the policy and critic. Measured EV uses the training GAE-target definition; update 126 has no next logged rollout.

**Scope and limitations.** Checkpoints state_0/35/70/105/139 correspond to updates 0/32/63/95/126. The final checkpoint retains its authorized five-batch fitted reward scale; its gradient intervals condition on that scale. Full-batch normalization retains the previously passed comparison with the training quarters. Only B is comparable between the two estimators’ different units. No jobs were launched, no batches were added, and reward code, reward artifacts, raw collection results and curated copies were preserved. No deviations from this request.

[As run](gradient-noise-as-run.png) · [Paper’s form](gradient-noise-paper-form.png) · [Estimates table](gradient-noise-estimates.md) · [Selected batches and calculation record](../../runs/gradient_point_estimates_20261012T205936.853570Z/candidates/render_record.json).
