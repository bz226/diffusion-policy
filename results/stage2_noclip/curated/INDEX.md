# Curated Stage 2 results

## Full Stage 2 comparison

Which clipping removals preserve stable fine-tuning, and how does one actor update per batch change the outcome?

The user approved all full-analysis plots and the table with “keep the plots and tables in curated.” **final_analysis** includes all 19 selected Stage 2 runs and three unchanged Stage 1 baselines; original candidate filenames are retained.

| Artifact | What it answers / limitation | Original source |
|---|---|---|
| [Evaluation returns](eval_return_vs_env_steps.png) | Baseline versus NC1–NC3 and NC4 rates; three seeds, last eval at 9.36M steps. | [Candidate](../runs/final_analysis/candidates/eval_return_vs_env_steps.png) |
| [Training returns](train_return_vs_env_steps.png) | Returns under exploration noise; differs from evaluation. | [Candidate](../runs/final_analysis/candidates/train_return_vs_env_steps.png) |
| [Update diagnostics](diagnostics_vs_iteration.png) | R0 and all ablations; sampled transition KL, one R0 seed. | [Candidate](../runs/final_analysis/candidates/diagnostics_vs_iteration.png) |
| [Condition × seed table](condition_seed_results.csv) | All 22 rows, endpoints, minima, diagnostic KL and collapse. | [Candidate](../runs/final_analysis/candidates/condition_seed_results.csv) |

Keep the [captions](stage2-captions.md) with exports. They define population-SD bands and diagnostic caveats. Authorized fresh replacements occupy the R0 and NC1 seed-1 slots; interrupted attempts remain preserved and excluded. All unfavorable results are retained.

[Methods](../methods.md#analysis-and-deliverables) · [Source paths and analysis](../runs/final_analysis/analysis_record.json) · [Verification](../runs/final_analysis/curation_verification.json) · [Approval and copy record](../runs/final_analysis/curation_record.json).

## NC4 learning rates

Does one actor update per batch preserve stable HalfCheetah-v2 fine-tuning after the specified clipping removals, and how does the learning rate affect it?

Approved artifacts A–C from analysis **nc4_analysis** include all nine NC4 runs: seeds 0, 1 and 2 at learning rates 1e-4, 1e-3 and 3e-3, plus the three unchanged Stage 1 runs where baseline comparisons apply. The user approved the proposed names with “keep in curated.” This subset was curated before full-campaign completion and remains available alongside the full comparison.

| ID | Artifact | What it answers / limitation | Original source |
|---|---|---|---|
| A | Return curves: [PNG](halfcheetah-nc4-returns.png), [PDF](halfcheetah-nc4-returns.pdf) | Eval/train performance versus learning rate; three seeds, last eval at 9.36M steps. | [Candidate](../runs/nc4_analysis/candidates/nc4_returns.png) |
| B | Diagnostics: [PNG](halfcheetah-nc4-diagnostics.png), [PDF](halfcheetah-nc4-diagnostics.pdf) | Policy changes after the actor step; sampled transition KL is not exact final-action KL. | [Candidate](../runs/nc4_analysis/candidates/nc4_diagnostics.png) |
| C | [Per-seed CSV](halfcheetah-nc4-seed-results.csv) | Individual endpoints, minima and collapse flags; no extrapolation to 10M steps. | [Candidate](../runs/nc4_analysis/candidates/nc4_seed_results.csv) |

Keep the [captions](halfcheetah-nc4-captions.md) with exported artifacts. Bands are mean ± population standard deviation, not confidence intervals. All specified seeds and unfavorable outcomes are retained; this is no evidence that sample reuse alone causes instability because update count and gradient aggregation also change.

[Methods](../methods.md#nc4-subset-analysis-requested-before-campaign-completion) · [Source paths, configurations and command](../runs/nc4_analysis/analysis_record.json) · [Independent verification](../runs/nc4_analysis/verification.json) · [Approval and copy record](../runs/nc4_analysis/curation_record.json) · [Current study report](../experiment.md).

## NC2 and NC3

How does HalfCheetah-v2 fine-tuning change when the specified clipping mechanisms are removed, and when the coordinate-mean ratio becomes a coordinate-sum ratio?

Approved artifacts A–C from **nc23_analysis** include all six NC2/NC3 runs (seeds 0, 1 and 2) and the three unchanged Stage 1 runs for return comparisons. The user approved the proposed names with “keep in curated.” This subset was curated before full-campaign completion and remains available alongside the full comparison.

| ID | Artifact | What it answers / limitation | Original source |
|---|---|---|---|
| A | Return curves: [PNG](halfcheetah-nc2-nc3-returns.png), [PDF](halfcheetah-nc2-nc3-returns.pdf) | Eval/train performance; three seeds per condition, last eval at 9.36M steps. | [Candidate](../runs/nc23_analysis/candidates/nc23_returns.png) |
| B | Diagnostics: [PNG](halfcheetah-nc2-nc3-diagnostics.png), [PDF](halfcheetah-nc2-nc3-diagnostics.pdf) | Post-update changes; sampled transition KL is not exact final-action KL. | [Candidate](../runs/nc23_analysis/candidates/nc23_diagnostics.png) |
| C | [Per-seed CSV](halfcheetah-nc2-nc3-seed-results.csv) | Endpoints, minima, diagnostic summaries and collapse flags; no endpoint extrapolation. | [Candidate](../runs/nc23_analysis/candidates/nc23_seed_results.csv) |

Keep the [captions](halfcheetah-nc2-nc3-captions.md) with exported artifacts. Bands show mean ± population SD, not confidence intervals; all specified seeds and unfavorable outcomes are retained. NC2 removes multiple mechanisms jointly, and all six ablation seeds crossed the collapse threshold despite NC3 recovering more by the end. The rare nonzero clip fraction in NC3 seed 1, iteration 31, remains visible: the prescribed finite ratio bound of 1e6 was exceeded. Exact-zero KL values remain in summaries and are omitted only from logarithmic rendering.

[Methods](../methods.md#nc2-and-nc3-plots-before-full-stage-2-analysis) · [Source paths and exact command](../runs/nc23_analysis/analysis_record.json) · [Independent verification](../runs/nc23_analysis/verification.json) · [Approval and copy record](../runs/nc23_analysis/curation_record.json) · [Current study report](../experiment.md).
