# Curated DPPO per-step gradient-noise measurements

How does the gradient noise DPPO passes to AdamW change across its 20 PPO steps and checkpoints at 32, 63 and 95 updates?

User-approved selection: the completed figures, table and note (“keep in curated”). All 60 collection jobs finished, with 500 batches per checkpoint and part; no scientific run crashed and no KL early stop fired. The original-run first-step figure is omitted: at update 95, the squared-mean estimate is −3.33×10⁻⁸ (90% interval [−8.48×10⁻⁸, 1.83×10⁻⁸]), so its signal is unresolved. That evidence remains in the table and note.

| Artifact | Curated files | Source |
|---|---|---|
| Noise in each PPO step | [PNG](dppo_per_step_noise.png), [PDF](dppo_per_step_noise.pdf), [caption](dppo_per_step_noise.caption.md) | [Revised layout](../candidates/layout_review/render_provenance.json) |
| Pairs switched off by clipping | [PNG](dppo_per_step_inactive.png), [PDF](dppo_per_step_inactive.pdf), [caption](dppo_per_step_inactive.caption.md) | [Final candidates](../candidates/final/figs/dppo_per_step_inactive.png) |
| Rerun evaluation sanity check | [PNG](dppo_rerun_sanity.png), [PDF](dppo_rerun_sanity.pdf), [caption](dppo_rerun_sanity.caption.md), [curve CSV](rerun_evaluation_curve.csv) | [Final candidates](../candidates/final/figs/dppo_rerun_sanity.png) |
| Estimates and 90% intervals | [Markdown](ppo_steps_table.md), [CSV](ppo_steps_table.csv) | [Analysis record](../candidates/final/analysis_record.json) |
| Interpretation and limitations | [Analysis note](NOTE.md) | [Original note](../candidates/final/NOTE.md) |

The measurements condition on one training seed. Several early-step signal intervals include zero even when their point estimates are positive. Part A uses fitted reward scaling and is approximate; Part C restores full training state. Per-episode variance after step 1 uses the requested 80× batch-variance convention. The original combined noise figure uses a 3.5-decade span; intervals are in the table, not the plotted points.

[Methods](../methods.md) · [Study record](../experiment.md) · [Exact job plan](../plan.json) · [Verification](../runs/final_review.json) · [Curation record](../runs/curation_20261014T213600Z/manifest.json).

## Per-checkpoint views

How do DPPO's 20 per-step gradients compare at 32, 63 and 95 updates on identical axes? These four views were approved on 2026-10-18 ("keep curated"). They use the same saved Part C estimates, with no new sampling or recomputation. Each quantity has identical limits across checkpoints, and all three quantities span four decades. Point estimates alone do not show uncertainty; the existing table retains the 90% intervals.

| Artifact | Curated files |
|---|---|
| DPPO after 32 updates | [PNG](dppo-per-step-noise-32-updates.png), [PDF](dppo-per-step-noise-32-updates.pdf), [caption](dppo-per-step-noise-32-updates.caption.md) |
| DPPO after 63 updates | [PNG](dppo-per-step-noise-63-updates.png), [PDF](dppo-per-step-noise-63-updates.pdf), [caption](dppo-per-step-noise-63-updates.caption.md) |
| DPPO after 95 updates | [PNG](dppo-per-step-noise-95-updates.png), [PDF](dppo-per-step-noise-95-updates.pdf), [caption](dppo-per-step-noise-95-updates.caption.md) |
| Three-checkpoint grid | [PNG](dppo-per-step-noise-checkpoint-grid.png), [PDF](dppo-per-step-noise-checkpoint-grid.pdf), [caption](dppo-per-step-noise-checkpoint-grid.caption.md) |

[Axis limits and verification note](dppo-per-step-noise-checkpoint-views-note.md) · [Source candidates](../candidates/checkpoint_views_20261018/) · [Rendering record](../candidates/checkpoint_views_20261018/render_record.json) · [Plotting code](../code/plot_per_checkpoint.py) · [Curation decision](../runs/curation_checkpoint_views_20261018/decision.json). The original combined figure remains unchanged.
