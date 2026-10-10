# Curated NC4 results

Does one actor update per batch preserve stable HalfCheetah-v2 fine-tuning after the specified clipping removals, and how does the learning rate affect it?

Approved artifacts A–C from analysis **nc4_analysis** include all nine NC4 runs: seeds 0, 1 and 2 at learning rates 1e-4, 1e-3 and 3e-3, plus the three unchanged Stage 1 runs where baseline comparisons apply. The user approved the proposed names with “keep in curated.” This completed subset does not mark the full Stage 2 campaign complete.

| ID | Artifact | What it answers / limitation | Original source |
|---|---|---|---|
| A | Return curves: [PNG](halfcheetah-nc4-returns.png), [PDF](halfcheetah-nc4-returns.pdf) | Eval/train performance versus learning rate; three seeds, last eval at 9.36M steps. | [Candidate](../runs/nc4_analysis/candidates/nc4_returns.png) |
| B | Diagnostics: [PNG](halfcheetah-nc4-diagnostics.png), [PDF](halfcheetah-nc4-diagnostics.pdf) | Policy changes after the actor step; sampled transition KL is not exact final-action KL. | [Candidate](../runs/nc4_analysis/candidates/nc4_diagnostics.png) |
| C | [Per-seed CSV](halfcheetah-nc4-seed-results.csv) | Individual endpoints, minima and collapse flags; no extrapolation to 10M steps. | [Candidate](../runs/nc4_analysis/candidates/nc4_seed_results.csv) |

Keep the [captions](halfcheetah-nc4-captions.md) with exported artifacts. Bands are mean ± population standard deviation, not confidence intervals. All specified seeds and unfavorable outcomes are retained; this is no evidence that sample reuse alone causes instability because update count and gradient aggregation also change.

[Methods](../methods.md#nc4-subset-analysis-requested-before-campaign-completion) · [Source paths, configurations and command](../runs/nc4_analysis/analysis_record.json) · [Independent verification](../runs/nc4_analysis/verification.json) · [Approval and copy record](../runs/nc4_analysis/curation_record.json) · [Current study report](../experiment.md).
