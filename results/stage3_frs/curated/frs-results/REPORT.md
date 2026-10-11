# Frozen rescaling: all three FRS1 seeds collapsed

**Question.** Can fixed pretrained-gradient scales reproduce the progress and stability of E2_beta0, which adapts its scales?

**Answer.** No, under the tested LR 1e-4 and floor 1e-3. All three FRS1 seeds crossed the reward-collapse threshold, first observed at evaluation iterations 80, 70 and 30. All runs nevertheless completed 140 iterations without NaN/inf; completion is not evidence of stability.

**Performance.** FRS1 ended far below E2, E2_beta0 and SGD10. These differences in return and discounted objective exceed twice the pooled seed SD; this is a descriptive three-seed comparison, not a significance test or a general claim about frozen scaling.

| Condition | Eval return, itr130 | Train return, last10 | Discounted J, last10 | Final distance from θ₀ |
|---|---:|---:|---:|---:|
| FRS1 | −447 ± 221 | −412 ± 182 | −144 ± 57 | 584 ± 877 |
| E2 | 4,542 ± 81 | 4,307 ± 14 | 1,510 ± 7 | 0.876 ± 0.012 |
| E2_beta0 | 4,588 ± 42 | 4,301 ± 18 | 1,511 ± 8 | 0.856 ± 0.028 |
| SGD10 | 4,458 ± 106 | 4,202 ± 111 | 1,480 ± 44 | 0.242 ± 0.022 |

Values are mean ± sample SD over seeds 0–2. Eval is at 9.36M training steps; training ends at 10.08M. “Last10” means the final ten training iterations. [Condition](condition_results.csv) and [per-seed tables](seed_results.csv) include checkpoint evaluations and paths.

![Training comparison](../frs-training-comparison.png)

Thin lines are individual seeds; bold lines and bands are means ± sample SD, without smoothing. The panels compare returns, discounted J, KL, parameter movement, approximate split-gradient noise, gradient direction and saturation; [captions](captions.md) define each quantity. Logarithmic panels omit zero values, including the late frozen portions of FRS1. SD bands may extend beyond physical bounds; they do not represent additional observations.

Final-checkpoint eval-sampler return was **−448 ± 223**; train-sampler return was **−417 ± 175**. Their discounted J values were −157 ± 73 and −144 ± 51, respectively.

**Instability.** Peak per-action KL was **1,384 / 2,181 / 2,317** for seeds 0/1/2. Peak coordinate-gradient/denominator ratios were **385,171 / 237,650 / 7,679,169**. The final 45/64/89 training updates were exactly zero, out of 126 updates each; final checkpoint x₀ saturation was 100% for both samplers. Per-seed median KL of 0.0803/0/0 therefore hides the instability. E2_beta0's mean seed-median KL was 0.00555. [Ratio/KL plot](../frs-update-instability.png), [extrema](ratio_extrema.csv).

**Frozen scales.** Raw r quantiles (min, 1%, 10%, 50%, 90%, 99%, max) were **0, 0, 1.453e−5, 6.267e−4, 0.01044, 0.09360, 2.20775**. There were **47,798 exact zeros** and **49,850 floored coordinates** among 568,392 parameters. The top 1% carried **79.02% of Σr²**. This used the approved reuse of 25 saved V4 batches at θ₀ (measurement seed1000, processing seed4000); no new scale-measurement sampling occurred.

**Verification and execution.** CPU/GPU numerical gates passed, including unchanged defaults and inert diagnostics; the rescaled formula's relative L2 error was below 2.20e−8. Source/run revision: `650e4f1665a6384dce1fb6920e53e790333318a6`. All 15 checkpoints and three final evaluations were verified. Training took 2h04m–2h07m per seed, about 58 seconds per training iteration. Total: **6.664/14 GPU-hours**. Monitors are paused. [Gate](../../runs/gate/gate_summary.json), [review](../../runs/final_analysis/review_record.json), [exact commands](../../runs/campaign_commands.jsonl), [methods and limitations](../../methods.md).

**Decision.** The campaign is complete. The user approved figures, tables and this reviewed report for curation; see the [curated index](../INDEX.md). All three collapsed seeds and the original evidence are retained.
