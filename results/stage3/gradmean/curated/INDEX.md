# Curated results

## Clean figures v2

How do rewards and gradient signal/noise compare between DPPO and without_PPO as actor updates accumulate?

User-approved selection: all five clean figures and the checkpoint table, under their proposed names ("keep in curated"). Each image is preserved in PNG and PDF, with its original caption; the table is preserved in CSV and Markdown.

| Artifact | Files |
|---|---|
| Evaluation reward | [PNG](eval_reward.png), [PDF](eval_reward.pdf), [caption](eval_reward.caption.md) |
| Training reward | [PNG](train_reward.png), [PDF](train_reward.pdf), [caption](train_reward.caption.md) |
| Main: frozen-critic gradient noise (V6) | [PNG](gradient_noise_V6.png), [PDF](gradient_noise_V6.pdf), [caption](gradient_noise_V6.caption.md) |
| Secondary: LOO gradient noise (V4) | [PNG](gradient_noise_V4.png), [PDF](gradient_noise_V4.pdf), [caption](gradient_noise_V4.caption.md) |
| Secondary: no-baseline gradient noise (V5) | [PNG](gradient_noise_V5.png), [PDF](gradient_noise_V5.pdf), [caption](gradient_noise_V5.caption.md) |
| Checkpoint measurements and intervals | [CSV](points_table.csv), [Markdown and column definitions](points_table.md) |

Source analysis: [final_v6_fixed, report recovery](../runs/analyses/reportfix_20261012/final_v6_fixed/analysis_status.json). [Original candidates and update mapping](../figs_clean/NOTE.md) · [Exact render inputs](../figs_clean/render_record.json) · [Plot source snapshot](../runs/postprocess_recovery_20261012/final_plot_source.py) · [Rerunnable plot script](../plot_clean.py) · [Methods](../methods.md) · [Curation record](../runs/curation_clean_v2_20261012/manifest.json).

Reward bands are ±1 sample SD over three training seeds. Gradient intervals are 90% group-jackknife intervals conditional on the measured seed-0 checkpoints, not training-seed uncertainty. Each point has 500 V6 batches (40,000 measurement episodes); the shared V4 signal uses all completed batches. V1 is a frozen, warmed-critic measurement proxy with full-batch normalization. The fixed compute stop replaced adaptive continuation before V6 data were observed; unresolved directions remain unresolved. Point returns in the table use the stochastic measurement sampler, unlike the deterministic evaluation curves. This selection preserves the complete requested comparison and its limitations.

## Round 2 curated results

How do the mean discounted-return gradient and its noise change along the DPPO and NC4 paths?

User-approved selection: both artifacts A and B, with the proposed names ("keep curated"). These are the fixed Round 2 measurements: 1,400 batches per policy, not the full campaign's final results.

| Artifact | Files |
|---|---|
| A: gradient signal and noise | [PNG](round2-gradient-signal-and-noise.png), [PDF](round2-gradient-signal-and-noise.pdf), [caption](round2-gradient-signal-and-noise.caption.md) |
| B: interpretation and estimates | [Analysis](round2-gradient-analysis.md), [CSV](round2-gradient-estimates.csv), [column definitions](round2-gradient-estimates.caption.md) |

Source run: [round2_review_20261012](../runs/analyses/round2_review_20261012/manifest.json), from [round_1400](../runs/analyses/round_1400/candidates/REPORT.md). [Original candidates](../runs/analyses/round2_review_20261012/candidates/round2_analysis.md) · [Methods](../methods.md) · [Study status](../experiment.md).

The intervals describe sampling uncertainty at seed-0 checkpoints. Mean-direction accuracy remains below 0.9; V5's signal remains unresolved. Curation preserves these limitations and does not mark the study complete.
