# Amended first-step gate

Both authorized precision checks passed on the saved gradient already certified in G2. The original failed G3 remains preserved; calibration, G1, G2 and G4 were not repeated. G5 has not run.

| Check | Result | Requirement |
|---|---:|---:|
| Float64 actual step: relative norm error | 1.80731275e-15 | ≤ 1e-10 |
| Native implicit update: relative norm error | 2.96249397e-08 | ≤ 1e-6 |
| Largest coordinate rounding residual | 0.499999853 ULP | ≤ 1 ULP |
| Coordinates exceeding the bound | 0 / 568392 | 0 |

The expected relative RMS norm-rounding error is 2.07646306e-06, compared with the original failed norm error 2.30865599e-06. This RMS estimate assumes independent uniform coordinate rounding. The native update uses the float32 alpha and gradient operands of `add_`, multiplied in float64 for inspection; no separately rounded multiply-then-add implementation replaces it. The requested test values are η = 0.00263955 and b₀ = 3.22129e-06. See [full evidence](gate_summary.json).
