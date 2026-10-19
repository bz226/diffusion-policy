# Gradient noise per optimizer gradient

**Question.** How do the saved gradient-noise figures compare in the units of one gradient passed to AdamW?

**Completed.** Six figure sets and a 65-row table were generated from saved statistics only. DPPO’s step-1 covariance is 4.097, 4.229 and 4.147 times the without_PPO_and_GAE update covariance at updates 32, 63 and 95. These compare different policies at matched update counts, not identical parameters.

Mean norms are unchanged. Covariance traces, relative variances and their saved intervals are divided by 80. Fourteen means remain unresolved; the table retains ratio point estimates and intervals, with the requested plug-in bounds separately labelled. The fitted scale at update 126 and normalization approximation are disclosed.

[Conversion, limits and limitations](../../figs/NOISE_ACTUAL_NOTE.md) · [Table](../../figs/noise_actual_table.md) · [DPPO grid](../../figs/dppo_per_step_noise_actual_grid.png) · [Grid with reference](../../figs/dppo_per_step_noise_actual_grid_ref.png) · [without_PPO_and_GAE](../../../gradmean/figs_clean/without_ppo_and_gae_noise_actual.png).

**Verification.** All 88 pre-existing PNG/PDF files match before and after. [Execution record](render_record.json) · [Before hashes](figure_hashes_before.json) · [After hashes](figure_hashes_after.json). No jobs or gradients were run; all code executed under Python 3.8. The approved CPU cap was ten minutes. All six figure sets, captions, the table and note are now [curated](../../curated/INDEX.md#per-optimizer-gradient-comparison), as explicitly approved by the user.
