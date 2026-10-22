# DPPO batch-average gradient: stopped at G3

**The required estimator-agreement gate failed.** Two of 63 covariance comparisons have nonoverlapping 90% intervals. The analysis stopped before the frozen-Adam metric, epoch/batch comparisons, turning geometry or figures. Q1–Q6 remain unanswered; no gate was relaxed and no run was launched.

## Failed comparisons

Variances below are in **10⁻⁴ training-loss units²**. Ratios are dimensionless. Intervals use the requested delete-one-group jackknife and t quantile on nine degrees of freedom.

| DPPO updates | PPO step | Group-based covariance trace | Batch-based covariance trace | Group / batch |
|---:|---:|---|---|---|
| 32 | 19 | 1.98243 [1.80249, 2.16236] | 1.77282 [1.75566, 1.78997] | 1.11824 [1.01957, 1.21691] |
| 63 | 9 | 1.96733 [1.90106, 2.03360] | 2.04502 [2.03512, 2.05493] | 0.96201 [0.93108, 0.99294] |

Thus the group estimator is 11.82% higher in the first case and 3.80% lower in the second. These discrepancies do not by themselves establish a coding error or estimator bias. Nevertheless, they fail the specified stopping gate, even under an interval-overlap interpretation.

The sum G passes at all checkpoints. Its covariance group/batch ratios are 0.99249 [0.95667, 1.02831], 1.00142 [0.97472, 1.02812] and 0.98900 [0.96687, 1.01114] at 32, 63 and 95 updates. These gate checks do not replace the unexecuted batch-update analysis.

## Verification and scope

- **G1 passed:** all 30 groups, 500 batches per checkpoint, all step sums, G sums and squared norms are present. No G reconstruction fallback was needed.
- **G2 passed:** the maximum relative L2 discrepancy between the sum of step sums and stored G is 1.73×10⁻¹⁶, below 10⁻⁸.
- **G3 failed:** 61/63 covariance comparisons and 63/63 signed squared-mean comparisons have overlapping intervals. All ratios and intervals are in the [gate record](runs/analysis_20261019/candidates/gates.json).
- **G4 passed:** full and deleted-group M matrices match the prior clip-timing cache bitwise.
- **G5 passed:** the final [preservation record](provenance/existing_files_after.json) confirms all 2,241 pre-existing files are unchanged.

A separate read-only code review confirmed the factor of 50, the 10/9-group covariance denominators, the 500/450-batch formulas and joint jackknife calculations. Deterministic small-array arithmetic checks passed. No simulation, model construction, gradient recomputation or GPU operation occurred.

The epoch-1 reference would be only first-order on-policy: its gradients are evaluated at sequentially moved parameters. G/20 is an average raw loss gradient, not AdamW's parameter displacement. The planned fixed checkpoint preconditioner would not reconstruct changing moments. These distinctions remain in the [methods](methods.md).

## Reproducibility and boundary

Inputs: `../runs/C35_j00..09`, `C70_j00..09`, `C105_j00..09`, their `batches.csv`, `batch_totals.csv`, `sums_step*.npy`, `sums_G.npy`, and the [clip-timing cache](../clip_timing/runs/analysis_20261019/candidates/grouped_moments.npz). See the [exact inventory](runs/analysis_20261019/candidates/inventory.json).

Parent commit: `a428d582ae05133f8026b81a35a699b6f07ebfac` on `main`; input measurement commit: `838189eac4794ac921217c50e133534ce95dc6ba`. New source is preserved in the [ordinary source snapshot](runs/analysis_20261019/source/) with its [execution record](provenance/execution.json). The Git metadata is read-only; no new commit is claimed.

The required staging-helper dry run passed and staged nothing. To stage from a normal terminal, run `python3 scripts/git_add_all.py` at the repository root.

The analysis and plotting scripts were written but not executed. The without_PPO table was unavailable at inspection; no fallback comparison was executed. The requested figures and `batch_update_table` were not generated because of G3. Further analysis requires an explicit revised instruction; existing measurements remain intact.
