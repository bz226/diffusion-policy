# Artifact review

These outputs use existing DPPO measurements only. The user approved all A–C artifacts on 2026-10-19 ("keep curated").

| ID | What it answers / important limitation | Suggested human-readable name |
|---|---|---|
| A | [Timing figure](figs/dppo_clip_timing_grid.png): clip engagement, mean norm and variance; temporal association is not causation. | dppo-clip-timing.png / .pdf |
| B | [Shape and direction](figs/dppo_reuse_mean_shape_grid.png): tests a fixed-direction cap; unresolved points are omitted and listed in its caption. | dppo-reuse-mean-shape-and-direction.png / .pdf |
| C | [Table](clip_timing_table.md) and [report](REPORT.md): joint jackknife intervals, sensitivities and all checkpoint verdicts; one training seed. | dppo-clip-timing-table.csv / .md; dppo-clip-timing-report.md |

Decision: keep A–C with the proposed names. See the [curated index](curated/INDEX.md), including captions and methods links.
