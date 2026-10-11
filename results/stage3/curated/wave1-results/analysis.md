# Completed Wave-1 runs: estimator and SGD comparison

**Question.** Which estimator and optimizer choices improve HalfCheetah returns without clipping or sample reuse?

**Answer.** All 27 Wave-1 runs completed without collapse, non-finite values or zero actor updates. E1 has the highest mean return; removing LOO hurts performance, while removing Adam momentum leaves endpoint returns unresolved relative to E2. SGD10 remains stable, but its apparent improvement over SGD3 is unresolved.

| Condition | Eval @9.36M | Train, last 10 | Discounted train J |
| --- | --- | --- | --- |
| E0 | 4532 ± 67 | 4303 ± 39 | 1510 ± 15 |
| E1 | 4670 ± 28 | 4429 ± 17 | 1553 ± 4 |
| E2 | 4542 ± 81 | 4307 ± 14 | 1510 ± 7 |
| E2_beta0 | 4588 ± 42 | 4301 ± 18 | 1511 ± 8 |
| E2_nobase | 4306 ± 109 | 3987 ± 91 | 1412 ± 29 |
| SGD03 | 4339 ± 14 | 3940 ± 24 | 1399 ± 8 |
| SGD1 | 4364 ± 67 | 3981 ± 12 | 1413 ± 4 |
| SGD3 | 4438 ± 84 | 4092 ± 39 | 1450 ± 14 |
| SGD10 | 4458 ± 106 | 4202 ± 111 | 1480 ± 44 |

Mean ± sample SD (ddof=1) across three independent seeds. Eval is iteration 130; training averages iterations 129 and 131–139, ending at 10.08M steps. J discounts raw four-step decision rewards by 0.99. The comparison rule calls differences smaller than twice pooled seed SD unresolved; it is not a significance test.

[Return curves](../wave1-returns.png) · [Update diagnostics](../wave1-update-diagnostics.png) · [Noise and saturation](../wave1-noise-saturation.png) · [Captions](captions.md) · [Condition table](condition_results.csv) · [Per-seed table](seed_results.csv) · [Table definitions](table_caption.md)

**Estimator changes.** E1 exceeds E0 in eval, train and J. E2 lowers both return measures relative to E1, offering no evidence of a discounted-up/undiscounted-down tradeoff; the E1–E2 eval contrast only narrowly exceeds the rule. These are bundled interventions, not an isolated test of decision discounting.

**Momentum and baseline.** E2_beta0 versus E2 is unresolved on all three return endpoints, despite approximately 14.7× larger mean seed-median KL (0.00555 versus 0.000377). Removing LOO reduces eval by 236 and J by 97.9, both beyond the descriptive threshold; it does not cause collapse.

**SGD sweep.** Mean training J is 1,399, 1,413, 1,450 and 1,480 at η*/3, η*, 3η* and 10η*, respectively. SGD10 versus SGD3 and E2 remains unresolved on all three endpoints. Its mean seed-median KL is 0.0214, 36.6× the calibration target, yet its final parameter distance is only 0.242 versus E2’s 0.876. Thus the initial calibration does not keep realized training KL matched. No SGD run collapsed or froze.

**Diagnostic limits.** Split-noise denominators are nonpositive in 207/378 seed windows; only nine of 126 condition windows support a complete three-seed B_env mean. Conditional medians from the remaining positive windows cannot establish a noise-scale ordering. Mean x0 saturation spans 6.64–12.18%, but no zero updates occur and saturation alone does not identify a cause of return differences.

**Scope.** AdamW uses 1e-4; η*=0.0001210723567 for SGD, with η_theory=η/2500. Calibration’s 3.40× spread triggered the prescribed warning. BATCH4 is ongoing; PROJ3/PROJ10 were skipped by the specified distance rule. Final-checkpoint evaluations are pending, so no claim about final-policy discounted improvement is made. Known Phase-1 bootstrap confidence-interval problems do not enter these seed-SD comparisons. Earlier curated results and all native outputs remain unchanged.

**Curation.** The user approved all four artifact groups with “keep in curated.” [Methods](../../methods.md) · [Reproduction and sources](../../runs/completed_wave1_analysis/analysis_record.json).
