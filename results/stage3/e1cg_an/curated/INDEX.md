# AdaGrad-Norm on the discounted critic-baseline estimator

Does a single-scalar adaptive step train without_PPO_and_GAE + γ^k stably? All nine runs completed without collapse. The largest mean discounted-return gain was 51.8 ± 12.4, versus AdamW’s 105.9 ± 21.3 (three-seed mean ± sample SD).

The user approved artifacts A–E and the proposed names with “save to curated” on 2026-10-18.

| ID | Artifact | Question / limitation | Original source |
|---|---|---|---|
| A | [PNG](adagrad-norm-seed0-curves.png) · [PDF](adagrad-norm-seed0-curves.pdf) · [Caption](adagrad-norm-seed0-curves.caption.md) | Learning curves; seed 0 only. | [Figure](../figs/an_curves_seed0.png) |
| B | [PNG](adagrad-norm-three-seed-curves.png) · [PDF](adagrad-norm-three-seed-curves.pdf) · [Caption](adagrad-norm-three-seed-curves.caption.md) | Mean learning curves; seed uncertainty is in the table. | [Figure](../figs/an_curves_3seed.png) |
| C | [PNG](adagrad-norm-step-diagnostics.png) · [PDF](adagrad-norm-step-diagnostics.pdf) · [Caption](adagrad-norm-step-diagnostics.caption.md) | Step length, KL and distance; seed 0 only. | [Figure](../figs/an_steps.png) |
| D | [Table](adagrad-norm-results-table.md) · [CSV](adagrad-norm-results-table.csv) | Outcomes and variability; three seeds. | [Table](../an_table.md) |
| E | [Report](adagrad-norm-report.md) | Q1–Q4 and gates; SGD references use a different estimator. | [Report](../REPORT.md) |

Run IDs: `AN1_seed0–2`, `AN3_seed0–2`, `AN10_seed0–2`. [Methods](../methods.md), [analysis inputs](../analysis_record.json), [job plan](../runs/plan_training.json), [gates](../runs/gates/amended_complete/gate_summary.json), and [completion record](../runs/completion_training.json) retain provenance. Source revision: `2acb18e9629c577474ee7f1652c7cc6f5a3829ac`.

η remains KL-calibrated; the theorem’s unbiased-gradient assumption is not established. Originals are preserved; figures and numeric tables are unchanged. Markdown links are adjusted for this folder, and captions accompany every figure.
