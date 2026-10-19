# PPO pieces and advantage estimator: curated results

Does replacing GAE with Monte Carlo return minus the critic change performance when DPPO keeps its PPO reuse and clipping mechanisms?

Selected by the user on 2026-10-19 (“keep in curated”), covering all five approved figure sets, the report, results table and contrasts under the proposed names. Figures retain PNG/PDF versions and their captions. Raw measurements, original candidates and checkpoints remain in the study folder.

**Finding.** Over the last five evaluations, DPPO exceeds without_GAE by 128 reward points (pooled seed SD 48), above the prescribed two-SD threshold. The interaction between advantage choice and the PPO bundle is unresolved. These results compare three seeds per method; they do not isolate reuse from the other PPO mechanisms.

[Report](ppo-gae-report.md) · [Results table](ppo-gae-results.md) ([CSV](ppo-gae-results.csv)) · [Contrasts](ppo-gae-contrasts.md) ([CSV](ppo-gae-contrasts.csv))

## Figures

| Question / display | Files | Important limitation |
|---|---|---|
| Reward curves, seed 0 | [PNG](ppo-gae-rewards-seed0.png) · [PDF](ppo-gae-rewards-seed0.pdf) · [Caption](ppo-gae-rewards-seed0.caption.md) | Single seed; historical discounted returns are missing. |
| Reward curves, three-seed means | [PNG](ppo-gae-rewards-three-seeds.png) · [PDF](ppo-gae-rewards-three-seeds.pdf) · [Caption](ppo-gae-rewards-three-seeds.caption.md) | Point estimates only; uncertainty is in the results table. |
| Discounted return versus distance, seed 0 | [PNG](ppo-gae-distance-seed0.png) · [PDF](ppo-gae-distance-seed0.pdf) · [Caption](ppo-gae-distance-seed0.caption.md) | Only two methods logged discounted returns; the four-method direction comparison is unavailable. |
| Discounted return versus distance, three-seed means | [PNG](ppo-gae-distance-three-seeds.png) · [PDF](ppo-gae-distance-three-seeds.pdf) · [Caption](ppo-gae-distance-three-seeds.caption.md) | Distances and returns are averaged at matching updates, not matched distances; only two methods have both. |
| PPO clipping diagnostics | [PNG](ppo-gae-clipping.png) · [PDF](ppo-gae-clipping.pdf) · [Caption](ppo-gae-clipping.caption.md) | Original DPPO clipping and switched-off fractions were not logged; only its approximate KL can be compared. |

## Sources and reproducibility

New runs: `DL1_seed0`, `DL1_seed1`, `DL1_seed2`, jobs 17959932–17959934, all 140 iterations and 126 updates complete. References are the original DPPO, without_PPO and without_PPO_and_GAE runs, three seeds each. Exact directories, commands and revisions are in the [analysis record](../analysis_record.json).

[Experiment record](../experiment.md) · [Methods](../methods.md) · [Gate results](../runs/gates/complete/gate_summary.json) · [Completed run records](../runs/status.json)

| Curated artifact | Published source |
|---|---|
| Reward curves, seed 0 (PNG, PDF, caption) | [`ladder2x2_curves_seed0`](../figs/ladder2x2_curves_seed0.png) and companion files |
| Reward curves, three-seed means (PNG, PDF, caption) | [`ladder2x2_curves_3seed`](../figs/ladder2x2_curves_3seed.png) and companion files |
| Discounted return versus distance, seed 0 (PNG, PDF, caption) | [`ladder2x2_distance_seed0`](../figs/ladder2x2_distance_seed0.png) and companion files |
| Discounted return versus distance, three-seed means (PNG, PDF, caption) | [`ladder2x2_distance_3seed`](../figs/ladder2x2_distance_3seed.png) and companion files |
| PPO clipping diagnostics (PNG, PDF, caption) | [`without_gae_ppo_steps`](../figs/without_gae_ppo_steps.png) and companion files |
| Report | [REPORT.md](../REPORT.md) |
| Results table | [Markdown](../ladder2x2_table.md) · [CSV](../ladder2x2_table.csv) |
| Contrasts | [Markdown](../contrasts.md) · [CSV](../contrasts.csv) |

Original analysis candidates are preserved in [the final candidate folder](../candidates/final_20261019T114811_890017Z/). The published report/table include the documented wording corrections; curation changes only filenames and relative links. All copied PNG/PDF/CSV files and captions are byte-identical to their published sources. No new analysis or sampling was run.

**Limits.** Historical DPPO and without_PPO runs lack discounted-return trajectories and saturation; original DPPO also lacks clipping fractions. Missing measurements stay missing. True post-update KL and PPO approximate KL are distinct quantities. The curated set includes all requested outputs, including unresolved comparisons.
