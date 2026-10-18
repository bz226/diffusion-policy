# AdaGrad-Norm on the discounted critic-baseline estimator

Can a single-scalar adaptive step train without_PPO_and_GAE + γ^k stably, and how does it compare with AdamW and calibrated SGD?

**Status.** All requested training runs and final checkpoint evaluations are complete.

**Q1 — Stability.**
- AdaGrad-Norm, η = d*: J_disc gain 18.302 ± 6.3185 (seed SD); collapse flags [false, false, false]; first crossing updates [null, null, null].
- AdaGrad-Norm, η = 3d*: J_disc gain 37.211 ± 8.0505 (seed SD); collapse flags [false, false, false]; first crossing updates [null, null, null].
- AdaGrad-Norm, η = 10d*: J_disc gain 51.779 ± 12.395 (seed SD); collapse flags [false, false, false]; first crossing updates [null, null, null].

**Q2 — Calibrated SGD.** The largest observed mean gain in this fixed grid is AdaGrad-Norm, η = 10d*. Against SGD at 3η*: J_disc gain, last 10 updates -29.613 ± 10.801 SE (exceeds reading-rule threshold); Eval at 9.36M -133.48 ± 50.481 SE (exceeds reading-rule threshold); Train last 10 updates -91.053 ± 31.456 SE (exceeds reading-rule threshold); Final distance from θ₀ -0.025356 ± 0.0018436 SE (exceeds reading-rule threshold). The available SGD references use LOO and other estimator settings; this comparison cannot by itself isolate the step schedule. The adaptive grid also uses KL calibration, so its performance cannot establish calibration-free selection of η. SGD, 3η* (LOO reference) has mean final distance 0.087553; the closest adaptive condition is AdaGrad-Norm, η = 10d* at 0.062197; SGD, 10η* (LOO reference) has mean final distance 0.24169; the closest adaptive condition is AdaGrad-Norm, η = 10d* at 0.062197.

**Q3 — AdamW.**
- AdaGrad-Norm, η = d* minus AdamW: J_disc gain, last 10 updates -87.565 ± 12.829 SE (exceeds reading-rule threshold); Eval at 9.36M -218.96 ± 31.513 SE (exceeds reading-rule threshold); Train last 10 updates -288.94 ± 37.943 SE (exceeds reading-rule threshold).
- AdaGrad-Norm, η = 3d* minus AdamW: J_disc gain, last 10 updates -68.657 ± 13.148 SE (exceeds reading-rule threshold); Eval at 9.36M -167.68 ± 28.471 SE (exceeds reading-rule threshold); Train last 10 updates -219.5 ± 40.859 SE (exceeds reading-rule threshold).
- AdaGrad-Norm, η = 10d* minus AdamW: J_disc gain, last 10 updates -54.088 ± 14.23 SE (exceeds reading-rule threshold); Eval at 9.36M -192.28 ± 31.081 SE (exceeds reading-rule threshold); Train last 10 updates -184.09 ± 42.429 SE (exceeds reading-rule threshold).

SGD, 3η* (LOO reference) minus AdamW gives J_disc gain difference -24.475 ± 14.721 SE (unresolved); SGD, 10η* (LOO reference) minus AdamW gives J_disc gain difference 4.8036 ± 28.356 SE (unresolved). These contextual SGD gaps are confounded by baseline, normalization and denoising-weight choices; agreement with an adaptive-optimizer gap would not alone establish geometry as its cause.

**Q4 — Mechanism, seed 0.** Step normalization uses j+1 = 1,…,126. A KL log-log slope of −1 corresponds to the proposed inverse-update decay. The distance ratio uses η√H_N, a noise-dominated heuristic, not a bound.

| Optimizer | Normalized step: median [min,max] | First KL / prediction | KL log-log slope | Final distance / η√H_N | RMS step length |
|---|---:|---:|---:|---:|---:|
| AdaGrad-Norm, η = d* | 0.98375 [0.92548, 1.1072] | 1.6671 | -1.0811 | 0.96979 | 0.00054268 |
| AdaGrad-Norm, η = 3d* | 0.98494 [0.89518, 1.1368] | 1.3723 | -1.0097 | 1.0276 | 0.0016453 |
| AdaGrad-Norm, η = 10d* | 0.98663 [0.89722, 1.251] | 0.78501 | -1.122 | 1.0233 | 0.0054683 |

For a noise-dominated 126-update walk, the equivalent constant RMS step is η√(H₁₂₆/126) ≈ 0.21η; this compares accumulated squared step lengths, not the total arc length.

| Optimizer, seed 0 | First step length | Observed RMS step length | Equivalent constant RMS step prediction | Final distance |
|---|---:|---:|---:|---:|
| AdaGrad-Norm, η = d* | 0.0026396 | 0.00054268 | 0.00054732 | 0.0059581 |
| AdaGrad-Norm, η = 3d* | 0.0079186 | 0.0016453 | 0.001642 | 0.01894 |
| AdaGrad-Norm, η = 10d* | 0.026396 | 0.0054683 | 0.0054732 | 0.06287 |
| AdamW (lr 1e-4) | 0.058566 | 0.018208 | N/A | 0.79455 |
| SGD, 3η* (LOO reference) | 0.0063007 | 0.0072571 | N/A | 0.088039 |
| SGD, 10η* (LOO reference) | 0.032059 | 0.021951 | N/A | 0.25206 |

[Full outcomes and differences](an_table.md) · [Seed-0 curves](figs/an_curves_seed0.png) · [Three-seed means](figs/an_curves_3seed.png) · [Steps, KL and distance](figs/an_steps.png) · [Seed metrics](seed_metrics.csv) · [Mechanism values](mechanism_seed0.csv).

**Calibration.** KL* = 0.00058452; d* = 0.0026396; distance spread max/min = 2.5869; median gradient norm = 0.0032213; b₀ = 3.2213e-06. The five batch step lengths are 0.0037902, 0.0014652, 0.0026226, 0.003275, 0.0026395. Reported calibration warnings: [].

**Verification and provenance.** Gates: G1 passed; G2 passed; G3a passed; G3b passed; G4 passed; G5 passed. Source revision `2acb18e9629c577474ee7f1652c7cc6f5a3829ac`, branch `stage3-e1cg-an`; exact reference revision `03f82611746954839169a64305777c3d7f91b139`. Saved commands, job names and seeds are in the linked plan; no existing results were modified by this analysis.

[Methods and scope](../../methods.md) · [Gate evidence](../../runs/gates/amended_complete/gate_summary.json) · [Job file](../../jobs_e1cg_an.txt) · [Exact job plan](../../runs/plan_training.json) · [Calibration record](../../runs/calibration_derived_amended.json) · [Analysis inputs and omissions](analysis_record.json).

**Limitations/deviations.** Reference SGD uses LOO because critic-baseline SGD remains paused. All seed records remain visible. Across-seed differences smaller than twice pooled SD are unresolved. New and AdamW checkpoint evaluations have 160 episodes per sampler; LOO reference evaluations retain 80. AdaGrad-Norm's theorem assumes an unbiased estimator; this normalized, denoising-weighted estimator has not been shown to meet that assumption.

**Missing evidence:** none.

**Approved numerical-check amendment.** The original G3 measured the rounded float32 parameter displacement; its relative norm error was 2.3087e-06 against a 1e-6 limit, so training stopped. The user replaced that check with a float64 formula check and a float32 pre-add/update-rounding check; both must pass, followed by the unchanged 12-iteration CPU smoke.

The measured calibration remains d* = 0.0026395 and b₀ = 3.2213e-06. The exact approved run values are η = 0.00263955, 0.00791865 and 0.0263955, with b₀ = 0.00000322129; no calibration or passed gate was rerun. [Original G3](../../runs/gates/calibrated/gate_summary.json) · [Amended G3a/G3b evidence](../../runs/gates/amended_G3/gate_summary.json) · [Approval](../../provenance/g3_amendment.json).

G3a numerical evidence: relative error = 1.8073e-15.
G3b numerical evidence: implicit update relative error = 2.9625e-08; actual norm relative error diagnostic only = 2.2537e-06; expected relative RMS norm rounding = 2.0765e-06; previous failed relative norm error = 2.3087e-06.
