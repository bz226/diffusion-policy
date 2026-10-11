# Curated FRS1 results

Can frozen pretrained-gradient scales match E2_beta0's progress and stability on HalfCheetah-v2?

The user approved **A–C** and the names below with “keep in curated.” All three FRS1 seeds completed training and final evaluation but collapsed in reward: scheduled final eval was **−447 ± 221**, versus **4,588 ± 42** for E2_beta0 (mean ± sample SD). This selection preserves all three negative outcomes and the completed reference runs.

| ID | Curated artifact | What it answers / limitation | Original source |
| --- | --- | --- | --- |
| A | Training comparison: [PNG](frs-training-comparison.png), [PDF](frs-training-comparison.pdf) | Returns, discounted objective, movement, noise and saturation versus E2, E2_beta0 and SGD10; three seeds, sample-SD bands. | [Reviewed candidate](../runs/final_analysis/candidates/frs_comparison.png) |
| B | Update instability: [PNG](frs-update-instability.png), [PDF](frs-update-instability.pdf) | Coordinate-gradient ratios and KL excursions; temporal coincidence does not establish causation. | [Reviewed candidate](../runs/final_analysis/candidates/frs_ratios_kl.png) |
| C | [Reviewed report](frs-results/REPORT.md), [condition table](frs-results/condition_results.csv), [per-seed table](frs-results/seed_results.csv) | Final evaluations, collapse, gradient scales and descriptive comparisons; conclusions apply to the tested LR and floor. | [Reviewed study report](../REPORT.md) |

Keep the [captions and definitions](frs-results/captions.md) with exports. Logarithmic axes omit zero values: late FRS1 gaps reflect frozen updates, not missing iterations. Undefined split-noise windows remain gaps; their independence approximation is limited by LOO coupling. Means require three finite seeds; bands are sample SD (ddof=1), not confidence intervals.

The results bundle includes [all iteration records](frs-results/all_runs_long.csv), [descriptive comparisons](frs-results/comparisons.csv), [noise windows](frs-results/noise_windows.csv), [ratio/KL extrema](frs-results/ratio_extrema.csv), and frozen-scale statistics in [CSV](frs-results/rescale_statistics.csv) and [JSON](frs-results/rescale_statistics.json). Scales reuse 25 saved Phase-1 V4 batches; this does not validate the earlier Phase-1 confidence intervals.

Run ID: `final_analysis`; training: `FRS1_seed0`, `FRS1_seed1`, `FRS1_seed2`. Source/run revision: `650e4f1665a6384dce1fb6920e53e790333318a6`.

[Methods](../methods.md#analysis-and-deliverables) · [Analysis source and command](../runs/final_analysis/analysis_record.json) · [Exact campaign commands](../runs/campaign_commands.jsonl) · [Numerical and visual review](../runs/final_analysis/review_record.json) · [Approval and copy record](../runs/final_analysis/curation_record.json) · [Current study](../experiment.md). Original candidates, logs and checkpoints are preserved.
