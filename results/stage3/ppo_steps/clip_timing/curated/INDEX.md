# DPPO clipping timing and the extra mean gradient

Selected by the user on 2026-10-19: both figures, the full table and report (A–C), under the proposed names.

**Question:** does clipping engage when DPPO’s mean step gradient stops following the reuse model? Clipping rises during epoch 2, but the extra mean rotates through significantly negative alignment with its final direction. This opposes a fixed-direction cap; it tests timing and shape, not causation.

| Artifact | Files | Source |
|---|---|---|
| Clip activity, mean norm and variance | [PNG](dppo-clip-timing.png), [PDF](dppo-clip-timing.pdf), [caption](dppo-clip-timing.caption.md) | [Timing figure](../figs/dppo_clip_timing_grid.png) |
| Extra mean’s shape and direction | [PNG](dppo-reuse-mean-shape-and-direction.png), [PDF](dppo-reuse-mean-shape-and-direction.pdf), [caption](dppo-reuse-mean-shape-and-direction.caption.md) | [Shape figure](../figs/dppo_reuse_mean_shape_grid.png) |
| Per-step estimates and 90% intervals | [Markdown](dppo-clip-timing-table.md), [CSV](dppo-clip-timing-table.csv) | [Source table](../clip_timing_table.csv) |
| Q1–Q5, checkpoint verdicts and gates | [Report](dppo-clip-timing-report.md) | [Source report](../REPORT.md) |

**Scope and limits:** one DPPO training seed, checkpoints at 32, 63 and 95 updates; 500 batches and ten independent groups at each checkpoint. Intervals use the delete-one-group jackknife with t₉. Direction panels omit unresolved points, listed in their caption, and use the approved −1 to 1 cosine axis. No new gradients, rollouts or GPU jobs were used. Curation retains the counterevidence and unresolved estimates.

[Methods and design](../methods.md) · [Run analysis_20261019](../runs/analysis_20261019/candidates/) · [Input inventory](../runs/analysis_20261019/candidates/input_inventory.json) · [Code and provenance](../provenance/source.json) · [Execution record](../provenance/execution.json)

Analysis source: `97b3ceb10529404d48df91dcdf1eec1e5946bcac`; completed results: `69b2ab2c1b5d71614a6382f511b8c81d67c1e55b`, branch `analysis-dppo-clip-timing`. Figure, caption and table copies are byte-identical to their sources. The curated report changes relative links and curation status only.
