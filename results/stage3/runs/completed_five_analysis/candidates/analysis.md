# Completed E0, E1, E2, SGD1 and SGD3: interim analysis

**Question.** How do the estimator ladder and calibrated SGD change return and stability relative to NC4?

**Answer.** E1 has the highest final scheduled evaluation and last-ten-iteration training returns among these five conditions. E2 does not show the proposed discounted-versus-undiscounted tradeoff: both measures are lower than E1. All 15 runs finished without non-finite values, reward-collapse flags or zero actor updates; final-checkpoint evaluation is still pending.

| Condition | Eval @9.36M | Train, last 10 | Discounted train J | Mean seed median KL | Mean final distance |
| --- | --- | --- | --- | --- | --- |
| E0 | 4532 ± 67 | 4303 ± 39 | 1510 ± 15 | 0.00056 | 0.889 |
| E1 | 4670 ± 28 | 4429 ± 17 | 1553 ± 4 | 0.00052 | 0.949 |
| E2 | 4542 ± 81 | 4307 ± 14 | 1510 ± 7 | 0.00038 | 0.876 |
| SGD1 | 4364 ± 67 | 3981 ± 12 | 1413 ± 4 | 0.00034 | 0.030 |
| SGD3 | 4438 ± 84 | 4092 ± 39 | 1450 ± 14 | 0.0026 | 0.088 |

Returns show mean ± sample SD (ddof=1) across three independent training seeds. Eval is iteration 130; training averages use iterations 129 and 131–139, ending at 10.08M environment steps. J uses raw rewards discounted by 0.99 per four-step decision. KL averages each seed’s median sampled transition KL; distance is the actor parameter norm relative to the pretrained actor.

[Performance curves](performance.png) · [Update diagnostics](diagnostics.png) · [Figure captions](captions.md) · [Per-seed values](seed_results.csv) · [Condition table](condition_results.csv) · [Table definitions](table_caption.md)

**Evidence.** E0→E1 improves eval by 137, train return by 126 and J by 43.3; all exceed twice the pooled seed SD. E1→E2 reduces eval by 128 and J by 43.1; the eval comparison only narrowly exceeds that rule (threshold 122). These are joint estimator interventions: E1 changes returns, baseline and actor reward units; E2 adds decision discounting while changing normalization, denoising weights and log-prob reduction.

SGD1→SGD3 raises J by 37.9, exceeding the same rule, while the 74-point eval difference remains unresolved. E2 exceeds SGD1 in eval and J; E2 versus SGD3 is unresolved in eval but E2 has higher J. SGD1/3 travel only 0.030/0.088 in parameter space versus E2’s 0.876. Their mean seed-median KLs are 0.58×/4.46× the calibration target, so matching initial calibration does not enforce matched KL throughout training.

E0’s eval difference from NC4 1e-4 (4,604 ± 19) remains inside the prescribed two-pooled-SD sanity band. E1’s difference from Stage-1 DPPO (4,758 ± 108) is unresolved under the same descriptive rule; this does not establish equivalence.

**Setup and limits.** AdamW uses 1e-4. SGD1 uses η=0.0001210723567 (η_theory=4.8429e-8); SGD3 uses three times that. Calibration’s 3.40× batch spread triggered the prescribed warning; the median was retained. All use one actor step per fresh batch, with unchanged critic training. Three seeds and training-sampler J are insufficient to claim a final-policy objective improvement; checkpoint evaluation will provide that comparison. No significance tests, smoothing, resampling or extra scientific sampling were added. The projection conditions and remaining campaign runs are outside this interim analysis.

**Decision.** Review these candidates for curation; no artifact has been curated. [Methods](../../../methods.md) · [Reproduction and sources](../analysis_record.json).
