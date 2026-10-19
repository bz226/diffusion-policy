# Gradient noise per optimizer gradient

The six figures retain every saved mean-norm point estimate. Covariance traces and relative variances are divided by 80, including their saved intervals. No sampling, jobs, gradients or jackknife estimates were recomputed.

| Quantity | Lower y-limit | Upper y-limit |
|---|---:|---:|
| mean_norm | 1e-05 | 0.01 |
| variance | 1e-05 | 0.001 |
| relative_variance | 10 | 1e+06 |

All new figures share these limits by quantity, at whole decades with at least 0.25 decade of margin.

**Unresolved means.** without_PPO_and_GAE: updates 32, 95 and 126. DPPO: steps 1–2 at update 32, steps 1–5 at update 63, and steps 1–4 at update 95. Positive ratio point estimates remain plotted. The table gives the requested plug-in R lower bounds and therefore upper bounds on expected cosine. Those bounds hold variance at its estimate; they do not replace the saved joint ratio intervals [0, ∞).

| Updates | DPPO step-1 covariance / without_PPO_and_GAE update covariance |
|---:|---:|
| 32 | 4.0970007 |
| 63 | 4.229155 |
| 95 | 4.1468063 |

**Interpretation and inherited limitations.** These are different policies at matched update counts. The normalization choices put both collections in training-loss units, but the saved without_PPO_and_GAE collection uses full-batch normalization that previously passed its check against training quarters; it is not a bitwise replay of those quarters. Its update-126 scale is the previously authorized five-batch least-squares fit, and its intervals condition on that scale.

Norm intervals retain each source convention: direct norm jackknife for without_PPO_and_GAE (including its signed negative lower endpoint at update 126), and transformed squared-mean endpoints for DPPO. Resolution always uses the squared-mean interval. No direct batch-covariance field was stored; traces use /80. The saved B_update fields agree with B/80 for all five as-run points and interval endpoints.

**Preservation.** All 88 existing PNG/PDF files in the two figure folders, including their history subfolders, match their explicitly requested before/after SHA-256 values. The hash records are in [the render record folder](../runs/noise_actual_20261019T172317Z/). Existing code, tables and figures were not edited. New tables are copied identically into both figure folders so captions can refer to a local noise_actual_table. Long x-axis labels are shared once per figure for legibility. There are no scientific changes.

[Table](noise_actual_table.md) · [CSV](noise_actual_table.csv)
