# Artifact review

All required runs and analysis are complete. On 2026-10-17, “keep curated” approved all artifacts A–H with the proposed names below. [Curated index](curated/INDEX.md) · [Selection and source record](runs/curation/decision.json).

| ID | What it answers / important limitation | Source | Suggested curated name |
|---|---|---|---|
| A | Mean-gradient alignment along one path; uncertain Euclidean estimates do not predict AdamW exactly | [Figure](figs/e1cg_alignment.png) | `decision-discount-gradient-alignment.{png,pdf}` |
| B | Reward and discounted-return curves for seed 0 only | [Figure](figs/e1cg_curves_seed0.png) | `decision-discount-seed0-curves.{png,pdf}` |
| C | Three-seed mean reward curves; seed uncertainty is in the table, not bands | [Figure](figs/e1cg_curves_3seed.png) | `decision-discount-three-seed-curves.{png,pdf}` |
| D | Final episode-segment gains, 160 episodes per method and seed | [Figure](figs/e1cg_gain_by_segment.png) | `decision-discount-final-segment-gains.{png,pdf}` |
| E | Alignment estimates and 90% intervals; some cohorts or sanity checks unresolved | [Table](alignment_table.md) | `decision-discount-gradient-alignment-table.{csv,md}` |
| F | Outcomes, diagnostics and contrasts; only three seeds | [Table](e1cg_table.md) | `decision-discount-outcomes.{csv,md}` |
| G | Segment gains and early/late tilt at each checkpoint | [Table](segments_table.md) | `decision-discount-segment-gains.{csv,md}` |
| H | Q1–Q5 conclusions and complete methods, including deviations | [Report](REPORT.md), [methods](methods.md) | `decision-discount-report.md`, `decision-discount-methods.md` |

Kept figures include their caption files and links to the source run and methods. Curation follows the explicit approval; original artifacts remain in place.
