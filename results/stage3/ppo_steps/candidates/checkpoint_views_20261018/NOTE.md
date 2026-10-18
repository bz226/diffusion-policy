# DPPO per-step gradient noise: checkpoint views

The figures use the saved Part C `u_norm`, `trace_ep` and `B` point estimates unchanged: 20 PPO steps at 32/63/95 updates, 500 batches per checkpoint. Each column uses the same centered four-decade limits in all four figures.

| Quantity | Lower y-limit | Upper y-limit |
|---|---:|---:|
| Mean gradient norm (loss units) | 4.32123372219e-06 | 0.0432123372219 |
| Variance per episode (loss units²) | 0.000158313356189 | 1.58313356189 |
| Relative variance B (episodes) | 921.815691691 | 9218156.91691 |

Original Figure 2 PNG/PDF SHA-256 values agree before and after; all existing copies and the source table are byte-identical. The requested hashes and command are in [render_record.json](render_record.json). No jobs were launched and no statistical estimates or intervals were recomputed.

File-location detail: existing Figure 2 and its table were stored under `curated/` and `candidates/final/`, rather than the previously requested study-root `figs/`. The new plots are published to the requested `figs/` path. One shared x-axis label per figure avoids repeated long labels at single-column width. There are no scientific deviations.
