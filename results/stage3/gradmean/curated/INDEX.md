# Curated results

## Clean figures v5: gradient noise point estimates

How do the as-run and paper-form gradient estimates change along without_PPO_and_GAE on the same 500 batches per checkpoint?

Latest user-approved presentation ("save to curated"): both figures in PNG/PDF with captions, the estimates table and the comparison note. Each panel shows five filled circles on a two-decade log span; uncertainty is retained in the table.

| Artifact | Files |
|---|---|
| As-run gradient noise | [PNG](clean-v5-gradient-noise-point-estimates/gradient-noise-as-run.png), [PDF](clean-v5-gradient-noise-point-estimates/gradient-noise-as-run.pdf), [caption](clean-v5-gradient-noise-point-estimates/gradient-noise-as-run.caption.md) |
| Paper-form gradient noise | [PNG](clean-v5-gradient-noise-point-estimates/gradient-noise-paper-form.png), [PDF](clean-v5-gradient-noise-point-estimates/gradient-noise-paper-form.pdf), [caption](clean-v5-gradient-noise-point-estimates/gradient-noise-paper-form.caption.md) |
| Estimates and log comparison | [CSV](clean-v5-gradient-noise-point-estimates/gradient-noise-estimates.csv), [table](clean-v5-gradient-noise-point-estimates/gradient-noise-estimates.md), [note](clean-v5-gradient-noise-point-estimates/gradient-noise-estimates-note.md) |

All ten squared-mean point estimates are positive. Their 90% intervals reach zero at updates 32/95/126 for as run and 0/95/126 for paper form; the corresponding B intervals are [0, ∞). Each point uses 40,000 episodes and ten measurement groups from one training seed. Only B is directly comparable between the two forms. Final-checkpoint intervals condition on the fitted reward scale. The previous figure version and all reward artifacts remain preserved below.

[Saved statistics](../runs/gradient_point_estimates_20261012T205936.853570Z/candidates/render_record.json) · [Verification](../runs/gradient_point_estimates_20261012T205936.853570Z/verification.json) · [Plot source](../runs/gradient_point_estimates_20261012T205936.853570Z/plot_clean.py) · [Interval calculation](../runs/gradient_point_estimates_20261012T205936.853570Z/clean_v5_point_estimates.py) · [Methods](../methods.md#clean-figures-version-5) · [Curation record](../runs/curation_gradient_point_estimates_20261012T210120Z/manifest.json).

## Clean figures v5: gradient noise

How do the as-run and paper-form gradient estimators compare along without_PPO_and_GAE, using the same checkpoints, critic, reward scale and measurement batches?

User-approved selection: both gradient-figure bundles, the estimates table and the note ("keep in curated"), under the proposed names in `clean-v5-gradient-noise/`.

| Artifact | Files |
|---|---|
| As-run gradient noise | [PNG](clean-v5-gradient-noise/gradient-noise-as-run.png), [PDF](clean-v5-gradient-noise/gradient-noise-as-run.pdf), [caption](clean-v5-gradient-noise/gradient-noise-as-run.caption.md) |
| Paper-form gradient noise | [PNG](clean-v5-gradient-noise/gradient-noise-paper-form.png), [PDF](clean-v5-gradient-noise/gradient-noise-paper-form.pdf), [caption](clean-v5-gradient-noise/gradient-noise-paper-form.caption.md) |
| Estimates and limitations | [CSV](clean-v5-gradient-noise/gradient-noise-estimates.csv), [table](clean-v5-gradient-noise/gradient-noise-estimates.md), [note](clean-v5-gradient-noise/gradient-noise-estimates-note.md) |

Every checkpoint uses 500 batches (40,000 episodes), with 90% delete-one-group jackknife intervals using t(9) quantiles. Each estimator uses its own mean-gradient estimate. Only relative variance B is comparable between the two forms' units. Three of five squared-mean signals remain unresolved for each form; open triangles are plug-in lower bounds rather than confidence limits. The final reward scale is fitted, and gradient intervals condition on it. All measurements concern one training seed; earlier curated results remain preserved.

[Analysis record](../runs/clean_v5_20261012/final_analysis/manifest.json) · [Saved statistics](../runs/clean_v5_gradient_final_20261012T201214.011889Z/candidates/render_record.json) · [Independent checks](../runs/clean_v5_20261012/final_analysis/artifact_check.json) · [Plot source](../runs/clean_v5_20261012/final_analysis/source/plot_clean.py) · [Layout source and review](../runs/clean_v5_20261012/final_analysis/layout_review.json) · [Methods](../methods.md#clean-figures-version-5) · [Curation record](../runs/curation_gradient_noise_v5_20261012T202235Z/manifest.json).

## Three-seed reward comparison

How do evaluation and training rewards compare across three seeds for DPPO, without_PPO and without_PPO_and_GAE?

User-approved selection: both three-seed plot bundles and their statistics note ("keep in curated"), using the proposed names under `clean-v5-3seed/`.

| Artifact | Files |
|---|---|
| Evaluation reward | [PNG](clean-v5-3seed/evaluation-reward-three-seeds.png), [PDF](clean-v5-3seed/evaluation-reward-three-seeds.pdf), [caption](clean-v5-3seed/evaluation-reward-three-seeds.caption.md) |
| Training reward | [PNG](clean-v5-3seed/training-reward-three-seeds.png), [PDF](clean-v5-3seed/training-reward-three-seeds.pdf), [caption](clean-v5-3seed/training-reward-three-seeds.caption.md) |
| Statistics and source runs | [Note](clean-v5-3seed/reward-three-seed-note.md) |

Lines are means over seeds 0–2; bands show ±1 sample SD (ddof = 1), not confidence intervals. Each seed contributes 80 episodes per point. The x-axis is training environment steps in millions, and each method starts from its own three-seed mean. Original Stage-1 runs are used; DPPO's final evaluation is 4,758 ± 108. All nine runs are complete. These are three-seed comparisons only; the single-seed figures and earlier curated versions remain preserved.

[Original candidate note](../runs/clean_rewards_3seed_20261012T193300.153408Z/candidates/REWARD_3SEED_NOTE.md) · [Exact inputs and checks](../runs/clean_rewards_3seed_20261012T193300.153408Z/candidates/reward_3seed_record.json) · [Plot source snapshot](../runs/clean_rewards_3seed_20261012T193300.153408Z/plot_clean.py) · [Artifact review](../runs/clean_rewards_3seed_20261012T193300.153408Z/artifact_review.json) · [Curation record](../runs/curation_rewards_three_seed_20261012T193616Z/manifest.json).

## Clean figures v5: individual starting values

How do evaluation and training rewards compare for DPPO, without_PPO and without_PPO_and_GAE when each curve starts from its own run's value?

User-approved selection: both revised reward figures ("keep in curated"), with PNG, PDF and original captions under `clean-v5-own-starts/`.

| Artifact | Files |
|---|---|
| Evaluation reward | [PNG](clean-v5-own-starts/eval_reward.png), [PDF](clean-v5-own-starts/eval_reward.pdf), [caption](clean-v5-own-starts/eval_reward.caption.md) |
| Training reward | [PNG](clean-v5-own-starts/train_reward.png), [PDF](clean-v5-own-starts/train_reward.pdf), [caption](clean-v5-own-starts/train_reward.caption.md) |

Each curve is seed 0, with 80 episodes per point and training environment steps in millions. Each starts at its own recorded mean; there is no pooling, smoothing or seed band. Original episode spreads were not saved, so starting-point standard errors are unavailable. All 420 curve values and the final evaluations at 9.36M steps match native results and logs. Earlier curated versions remain below; no gradient-noise artifact is included in this selection.

[Original candidates](../runs/clean_v5_reward_origins_20261012T180336.159971Z/candidates/) · [Starting values and limitations](../runs/clean_v5_reward_origins_20261012T180336.159971Z/candidates/REWARD_NOTE.md) · [Render inputs](../runs/clean_v5_reward_origins_20261012T180336.159971Z/candidates/reward_render_record.json) · [Plot source snapshot](../runs/clean_v5_reward_origins_20261012T180336.159971Z/plot_clean.py) · [Methods](../methods.md#clean-figures-version-5) · [Curation record](../runs/curation_reward_own_starts_20261012T182702Z/manifest.json).

## Clean figures v5: three-method reward curves

How do evaluation and training rewards compare for DPPO, without_PPO and without_PPO_and_GAE on seed 0?

User-approved selection: the two version-5 reward plots ("keep this two plot in curated"), retained as PNG and PDF with their captions. Earlier curated versions are preserved.

| Artifact | Files |
|---|---|
| Evaluation reward | [PNG](clean-v5-rewards/eval_reward.png), [PDF](clean-v5-rewards/eval_reward.pdf), [caption](clean-v5-rewards/eval_reward.caption.md) |
| Training reward | [PNG](clean-v5-rewards/train_reward.png), [PDF](clean-v5-rewards/train_reward.pdf), [caption](clean-v5-rewards/train_reward.caption.md) |

Both figures use training environment steps in millions, one seed per method and 80 episodes per point, with a shared starting mean over 240 original episodes. There are no seed bands; the starting-point standard error is unavailable because episode spreads were not saved. These curves describe the selected runs, not uncertainty across training seeds. This selection contains no gradient-noise figures or table.

[Original candidates](../runs/clean_v5_render_20261012T164704.933028Z/candidates/) · [Render inputs](../runs/clean_v5_render_20261012T164704.933028Z/candidates/render_record.json) · [Plot source snapshot](../runs/clean_v5_render_20261012T164704.933028Z/plot_clean.py) · [Methods](../methods.md#clean-figures-version-5) · [Curation record](../runs/curation_clean_v5_rewards_20261012T173535Z/manifest.json).

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

Source analysis: [final_v6_fixed, report recovery](../runs/analyses/reportfix_20261012/final_v6_fixed/analysis_status.json). [Original candidates and update mapping](../figs_clean/history/20261012T161233.062570Z/NOTE.md) · [Exact render inputs](../figs_clean/history/20261012T161233.062570Z/render_record.json) · [Plot source snapshot](../runs/postprocess_recovery_20261012/final_plot_source.py) · [Version-2 plot script](../runs/postprocess_recovery_20261012/final_plot_source.py) · [Methods](../methods.md) · [Curation record](../runs/curation_clean_v2_20261012/manifest.json).

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
