# Final fixed-policy gradient measurements

**Question.** How accurately can the mean gradient and its noise be measured along the DPPO and NC4 checkpoint paths?

**Answer.** Completed measurements cover 5 of five policies. 1 measured policies meet the prescribed V4 split-correlation threshold of 0.68; 4 remain below it or unresolved at the current sample count. The user replaced adaptive collection with a fixed compute plan; completion does not establish a resolved direction.

| Policy | Independent batches | V4 squared signal | V4 B in episodes | Split correlation |
| --- | ---: | ---: | ---: | ---: |
| P0 | 3700 | 4.2e+06 | 53,052 | 0.734 |
| D1 | 2800 | 1.34e+06 | 73,713 | 0.604 |
| D2 | 3700 | 1.35e+06 | 79,595 | 0.649 |
| N1 | 2800 | 1.11e+06 | 1e+05 | 0.531 |
| N2 | 3550 | 9.74e+05 | 87,714 | 0.622 |

The signed squared signal uses V4 in paper gradient units. The ≥ entries use the trace estimate divided by the upper endpoint of the signal's 90% interval; they are plug-in lower bounds, without a joint confidence guarantee. Undefined values are preserved rather than set to zero.

**Uncertainty.** Every measurement interval deletes one independent group of 50 whole batches, recomputes the statistic, and uses ±1.645 standard errors. Each batch contains 80 episodes. Groups are paired across variants at one policy and independent across policies; the 20 random half-splits are not 20 independent experiments.

**Comparisons.** [Path alignment](path_alignment.csv) uses endpoint gradients and checkpoint displacements; its trapezoid approximates a chord integral, not the complete training trajectory. [Existing SGD drift](drift_summary.csv) uses all 126 training returns per seed and is a path-averaged proxy, particularly approximate for SGD10. [Earlier Phase-1 estimates](phase1_comparison.csv) remain provisional because their uncertainty was not valid.

**Boundary.** V1–V3 publish noise scales and cosines only; V4/V5 norms and covariance traces use paper units. The [variance-lemma comparison](score_variance_lemma.csv) uses an observed reward maximum and 200 decision-score samples from one group, so it has no group-jackknife interval and is not a proven bound.

[V4 path figure](figures/v4_signal_noise_paths.png) · [All results and intervals](main_results.csv) · [Per-policy returns and diagnostics](per_point.csv) · [Table definitions](table_captions.md) · [Analysis record](../analysis_status.json).

These candidates are uncurated. No training or additional scientific sampling was performed by this analysis.
