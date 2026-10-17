# Decision discount after advantage normalization

**Question.** Does adding γ^k after advantage normalization improve the discounted objective or move gains earlier in the episode?

The three-seed experiment found no resolved performance or episode-tilt change under the agreed reading rule. The discounted run had smaller parameter displacement and lower mean KL per update. These results do not establish equivalence or a general benefit of removing discount weights.

Artifacts A–H were approved on 2026-10-17 with “keep curated,” using the names proposed in the [artifact review](../ARTIFACT_REVIEW.md). All three completed seeds are included; optional E2C was not run because capacity did not meet the approved rule.

| ID | Artifact | Question / principal limitation | Source |
|---|---|---|---|
| A | [PNG](decision-discount-gradient-alignment.png) · [PDF](decision-discount-gradient-alignment.pdf) · [Caption](decision-discount-gradient-alignment.caption.md) | Mean-gradient alignment; noisy Euclidean estimates do not predict AdamW exactly. | [Figure](../figs/e1cg_alignment.png) |
| B | [PNG](decision-discount-seed0-curves.png) · [PDF](decision-discount-seed0-curves.pdf) · [Caption](decision-discount-seed0-curves.caption.md) | Learning curves for seed 0; one run per method. | [Figure](../figs/e1cg_curves_seed0.png) |
| C | [PNG](decision-discount-three-seed-curves.png) · [PDF](decision-discount-three-seed-curves.pdf) · [Caption](decision-discount-three-seed-curves.caption.md) | Three-seed mean curves; uncertainty is in the outcome table. | [Figure](../figs/e1cg_curves_3seed.png) |
| D | [PNG](decision-discount-final-segment-gains.png) · [PDF](decision-discount-final-segment-gains.pdf) · [Caption](decision-discount-final-segment-gains.caption.md) | Final gains by episode segment; 160 episodes per method and seed. | [Figure](../figs/e1cg_gain_by_segment.png) |
| E | [Table](decision-discount-gradient-alignment-table.md) · [CSV](decision-discount-gradient-alignment-table.csv) | Alignment estimates and 90% intervals; unresolved cases remain explicit. | [Table](../alignment_table.md) |
| F | [Table](decision-discount-outcomes.md) · [CSV](decision-discount-outcomes.csv) | Performance and diagnostic contrasts; three seeds and the agreed reading rule. | [Table](../e1cg_table.md) |
| G | [Table](decision-discount-segment-gains.md) · [CSV](decision-discount-segment-gains.csv) | Segment gains and early/late tilt by checkpoint; shared pretrained reference. | [Table](../segments_table.md) |
| H | [Report](decision-discount-report.md) · [Methods](decision-discount-methods.md) | Q1–Q5, design and deviations; conclusions are specific to this setup. | [Report](../REPORT.md) · [Methods](../methods.md) |

**Reproducibility.** Study/run ID: `e1cg`; training runs `E1CG_seed0`, `E1CG_seed1`, `E1CG_seed2`. Part A uses existing paired gradient collections. [Analysis record](../analysis_record.json), [Part A record](../runs/part_a/candidates/alignment_record.json), [exact job plan](../runs/plan.json), [gate results](../runs/gates/README.md), [completion verification](../runs/completion_verification.json), and [source revision](../provenance/execution_revision.json) retain inputs, seeds and commands. The final figures include the [layout-only correction](../runs/plot_layout_fix/record.json).

Figures retain their captions; tables retain the uncertainty and reading rules. Numerical artifacts are copied unchanged, Markdown links are rebased for this folder, and original results remain in place. This selection is the approved A–H set, not a claim that the evidence supports a performance gain.
