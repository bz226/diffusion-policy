# DPPO actor epochs: curated results

Does reuse improve DPPO beyond the movement obtained by increasing the learning rate with fewer actor epochs?

The four seed-0 runs and final analysis are complete. The matched 1-epoch run's final evaluation is 4,609 versus DPPO's 4,636; the difference is unresolved under the specified historical seed-SD yardstick. This is compatible with the movement-only explanation, but does not establish equivalence.

[Report](REPORT.md) · [Results table](epochs_table.md) ([CSV](epochs_table.csv)) · [Methods and design](methods.md)

## Figures

| Display | Files | Limitation |
| --- | --- | --- |
| Reward curves: same and matched learning rates | [PNG](figs/epochs_curves_seed0.png) · [PDF](figs/epochs_curves_seed0.pdf) · [Caption](figs/epochs_curves_seed0.caption.md) | Seed 0 only; historical DPPO and without_PPO did not log discounted returns. |
| Discounted return versus parameter distance | [PNG](figs/epochs_distance_seed0.png) · [PDF](figs/epochs_distance_seed0.pdf) · [Caption](figs/epochs_distance_seed0.caption.md) | Only the four new runs have both quantities; the comparison with original DPPO is unavailable. |
| Last-actor-step PPO diagnostics | [PNG](figs/epochs_ppo_diagnostics_seed0.png) · [PDF](figs/epochs_ppo_diagnostics_seed0.pdf) · [Caption](figs/epochs_ppo_diagnostics_seed0.caption.md) | Historical DPPO supplies approximate KL only; its switched-off fraction and ratio displacement were not logged. |

[Reward contrasts and reading thresholds](gaps_to_dppo.csv) · [Checkpoint distances](checkpoint_distances.csv)

## Sources and reproducibility

The user selected the completed EP report, plots and tables with “save in curated”; existing artifact names are retained, with the methods included for context. All requested comparisons are included, including unresolved outcomes. No new analysis, training or sampling was performed for curation.

| Curated artifact | Source |
| --- | --- |
| Report | [Published report](../REPORT.md) |
| Methods | [Study methods](../methods.md) |
| Results table | [Markdown](../epochs_table.md) · [CSV](../epochs_table.csv) |
| Reward contrasts | [CSV](../gaps_to_dppo.csv) |
| Checkpoint distances | [CSV](../checkpoint_distances.csv) |
| Figures and captions | [Published figure folder](../figs/) |

New run IDs: `EP1_seed0`, `EP1M_seed0`, `EP2_seed0`, `EP2M_seed0`; jobs 18011574, 18011572, 18011575 and 18011573. Each completed 140 iterations and 126 fresh-data updates. Source revision: `e0db307b35debdb42762e0ef1b206def3e61ac41`, branch `stage3-epochs`.

[Experiment record](../experiment.md) · [Exact inputs](../analysis_record.json) · [Execution records](../execution_records.json) · [Gate results](../runs/gates/complete/gate_summary.json) · [Verified completion](../runs/completion_training.json) · [Original candidates](../runs/campaign_training/candidates/final/)

PNG, PDF, CSV and caption files are unchanged copies. Report links are adjusted for this folder; table-header pipes are escaped for Markdown rendering. Original measurements, checkpoints and source artifacts remain in place.

**Limits.** These are single-seed comparisons. Matching epochs × learning rate is only a first-order movement premise under AdamW. Historical missing discounted-return and clipping measurements remain missing. The uncertainty yardstick uses the original three DPPO seeds; it cannot establish equivalence.
