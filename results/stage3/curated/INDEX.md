# Curated Stage 3 results

## Completed Wave-1 comparison

Which estimator and optimizer choices improve HalfCheetah returns without clipping or sample reuse, relative to Stage-1 DPPO and Stage-2 NC4?

The user approved artifacts **A–D** and their proposed names with “keep in curated.” This snapshot includes all 27 completed Wave-1 runs: seeds 0–2 for E0, E1, E2, E2_beta0, E2_nobase, SGD03, SGD1, SGD3 and SGD10. BATCH4 and final-checkpoint evaluations were unfinished at the snapshot; the six earlier reference runs are retained where metrics exist.

| ID | Curated artifact | What it answers / limitation | Original source |
| --- | --- | --- | --- |
| A | Returns: [PNG](wave1-returns.png), [PDF](wave1-returns.pdf) | Eval, train and discounted train returns; scheduled eval ends at 9.36M steps. | [Candidate](../runs/completed_wave1_analysis/candidates/performance.png) |
| B | Update diagnostics: [PNG](wave1-update-diagnostics.png), [PDF](wave1-update-diagnostics.pdf) | Sampled transition KL, parameter movement and gradient cosine; these do not establish a causal mechanism. | [Candidate](../runs/completed_wave1_analysis/candidates/diagnostics.png) |
| C | Noise and saturation: [PNG](wave1-noise-saturation.png), [PDF](wave1-noise-saturation.pdf) | Split-gradient noise, denoised saturation and action range; 207/378 noise windows are undefined, preventing reliable noise ranking. | [Candidate](../runs/completed_wave1_analysis/candidates/noise_saturation.png) |
| D | [Analysis](wave1-results/analysis.md), [condition table](wave1-results/condition_results.csv), [per-seed table](wave1-results/seed_results.csv) | All nine conditions and 27 runs; final-policy checkpoint comparisons remain outside this snapshot. | [Candidate analysis](../runs/completed_wave1_analysis/candidates/analysis.md) |

Keep the [figure captions](wave1-results/captions.md) and [table definitions](wave1-results/table_caption.md) with exports. Bands are sample SD (ddof=1) across three seeds, not confidence intervals. Undefined noise windows remain gaps; no means use only surviving positive-denominator seeds. Phase-1 bootstrap intervals are excluded. E1/E2 bundle estimator changes, so decision discounting is not isolated.

The bundle retains [all iteration records](wave1-results/all_runs_long.csv), [descriptive comparisons](wave1-results/comparisons.csv), [noise windows](wave1-results/noise_windows.csv) and [structured summaries](wave1-results/summary.json). Original candidates and the earlier curated comparison below are preserved.

[Methods](../methods.md#training-comparisons) · [Analysis sources and command](../runs/completed_wave1_analysis/analysis_record.json) · [Plot sources and command](../runs/completed_wave1_analysis/plot_record.json) · [Independent numerical checks](../runs/completed_wave1_analysis/independent_verification.json) · [Approval and copy record](../runs/completed_wave1_analysis/curation_record.json).

## Completed E0, E1, E2, SGD1 and SGD3 comparison

How do the estimator ladder and calibrated SGD change HalfCheetah returns and policy updates relative to the Stage-1 DPPO and Stage-2 NC4 baselines?

The user approved artifacts **A–C**, including the proposed names, with “keep in curated.” This completed interim analysis includes all 15 runs: seeds 0–2 for each of E0, E1, E2, SGD1 and SGD3, with six earlier reference runs. The broader Stage-3 campaign remains in progress.

| ID | Curated artifact | What it answers / limitation | Original source |
| --- | --- | --- | --- |
| A | Returns: [PNG](estimator-sgd-returns.png), [PDF](estimator-sgd-returns.pdf) | Evaluation, training and discounted training return; evaluation ends at 9.36M steps. | [Candidate](../runs/completed_five_analysis/candidates/performance.png) |
| B | Diagnostics: [PNG](estimator-sgd-diagnostics.png), [PDF](estimator-sgd-diagnostics.pdf) | Sampled transition KL, parameter movement and consecutive gradient cosine; these do not establish a causal mechanism. | [Candidate](../runs/completed_five_analysis/candidates/diagnostics.png) |
| C | [Analysis](estimator-sgd-results/analysis.md), [condition table](estimator-sgd-results/condition_results.csv), [per-seed table](estimator-sgd-results/seed_results.csv) | All five conditions and 15 seeds, with mean and sample SD; final-checkpoint evaluations were pending at this snapshot. | [Candidate analysis](../runs/completed_five_analysis/candidates/analysis.md) |

Keep the [figure captions](estimator-sgd-results/captions.md) and [table definitions](estimator-sgd-results/table_caption.md) with exports. Bands represent sample SD (ddof=1) across three seeds, not confidence intervals. All selected seeds and outcomes are retained; E1/E2 change multiple estimator components together, so discounting alone is not isolated. Phase-1 bootstrap intervals are not used here.

The results bundle also preserves [all iteration records](estimator-sgd-results/all_runs_long.csv), [descriptive comparisons](estimator-sgd-results/comparisons.csv), [noise windows](estimator-sgd-results/noise_windows.csv) and [structured summaries](estimator-sgd-results/summary.json). Noise-window medians use defined positive-denominator windows; undefined windows remain recorded.

[Methods](../methods.md#training-comparisons) · [Analysis sources and command](../runs/completed_five_analysis/analysis_record.json) · [Plot sources and command](../runs/completed_five_analysis/plot_record.json) · [Approval and copy record](../runs/completed_five_analysis/curation_record.json) · [Current study](../experiment.md).
