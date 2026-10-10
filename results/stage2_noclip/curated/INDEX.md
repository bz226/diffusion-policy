# Curated Stage 2 results

## NC4 learning rates

Does one actor update per batch preserve stable HalfCheetah-v2 fine-tuning after the specified clipping removals, and how does the learning rate affect it?

Approved artifacts A–C from analysis **nc4_analysis** include all nine NC4 runs: seeds 0, 1 and 2 at learning rates 1e-4, 1e-3 and 3e-3, plus the three unchanged Stage 1 runs where baseline comparisons apply. The user approved the proposed names with “keep in curated.” This completed subset does not mark the full Stage 2 campaign complete.

| ID | Artifact | What it answers / limitation | Original source |
|---|---|---|---|
| A | Return curves: [PNG](halfcheetah-nc4-returns.png), [PDF](halfcheetah-nc4-returns.pdf) | Eval/train performance versus learning rate; three seeds, last eval at 9.36M steps. | [Candidate](../runs/nc4_analysis/candidates/nc4_returns.png) |
| B | Diagnostics: [PNG](halfcheetah-nc4-diagnostics.png), [PDF](halfcheetah-nc4-diagnostics.pdf) | Policy changes after the actor step; sampled transition KL is not exact final-action KL. | [Candidate](../runs/nc4_analysis/candidates/nc4_diagnostics.png) |
| C | [Per-seed CSV](halfcheetah-nc4-seed-results.csv) | Individual endpoints, minima and collapse flags; no extrapolation to 10M steps. | [Candidate](../runs/nc4_analysis/candidates/nc4_seed_results.csv) |

Keep the [captions](halfcheetah-nc4-captions.md) with exported artifacts. Bands are mean ± population standard deviation, not confidence intervals. All specified seeds and unfavorable outcomes are retained; this is no evidence that sample reuse alone causes instability because update count and gradient aggregation also change.

[Methods](../methods.md#nc4-subset-analysis-requested-before-campaign-completion) · [Source paths, configurations and command](../runs/nc4_analysis/analysis_record.json) · [Independent verification](../runs/nc4_analysis/verification.json) · [Approval and copy record](../runs/nc4_analysis/curation_record.json) · [Current study report](../experiment.md).

## NC2 and NC3

How does HalfCheetah-v2 fine-tuning change when the specified clipping mechanisms are removed, and when the coordinate-mean ratio becomes a coordinate-sum ratio?

Approved artifacts A–C from **nc23_analysis** include all six NC2/NC3 runs (seeds 0, 1 and 2) and the three unchanged Stage 1 runs for return comparisons. The user approved the proposed names with “keep in curated.” This is a completed subset, not a declaration that the full Stage 2 campaign is complete.

| ID | Artifact | What it answers / limitation | Original source |
|---|---|---|---|
| A | Return curves: [PNG](halfcheetah-nc2-nc3-returns.png), [PDF](halfcheetah-nc2-nc3-returns.pdf) | Eval/train performance; three seeds per condition, last eval at 9.36M steps. | [Candidate](../runs/nc23_analysis/candidates/nc23_returns.png) |
| B | Diagnostics: [PNG](halfcheetah-nc2-nc3-diagnostics.png), [PDF](halfcheetah-nc2-nc3-diagnostics.pdf) | Post-update changes; sampled transition KL is not exact final-action KL. | [Candidate](../runs/nc23_analysis/candidates/nc23_diagnostics.png) |
| C | [Per-seed CSV](halfcheetah-nc2-nc3-seed-results.csv) | Endpoints, minima, diagnostic summaries and collapse flags; no endpoint extrapolation. | [Candidate](../runs/nc23_analysis/candidates/nc23_seed_results.csv) |

Keep the [captions](halfcheetah-nc2-nc3-captions.md) with exported artifacts. Bands show mean ± population SD, not confidence intervals; all specified seeds and unfavorable outcomes are retained. NC2 removes multiple mechanisms jointly, and all six ablation seeds crossed the collapse threshold despite NC3 recovering more by the end. The rare nonzero clip fraction in NC3 seed 1, iteration 31, remains visible: the prescribed finite ratio bound of 1e6 was exceeded. Exact-zero KL values remain in summaries and are omitted only from logarithmic rendering.

[Methods](../methods.md#nc2-and-nc3-plots-before-full-stage-2-analysis) · [Source paths and exact command](../runs/nc23_analysis/analysis_record.json) · [Independent verification](../runs/nc23_analysis/verification.json) · [Approval and copy record](../runs/nc23_analysis/curation_record.json) · [Current study report](../experiment.md).
