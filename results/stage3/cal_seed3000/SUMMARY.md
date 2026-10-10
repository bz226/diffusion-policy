# SGD step calibration

Target sampled transition KL: 0.000584517336. Median eta: 0.000121072357; eta_theory: 4.84289427e-08; max/min spread: 3.39548.

| Batch | eta | eta_theory | Realized KL | KL exponent |
|---|---:|---:|---:|---:|
| 0 | 9.0451441e-05 | 3.61805764e-08 | 0.000583664632 | 2.0131067534603164 |
| 1 | 0.000244215711 | 9.76862845e-08 | 0.000585211876 | 1.9921272819622233 |
| 2 | 0.000151939793 | 6.07759171e-08 | 0.000585916409 | 1.9798722889100293 |
| 3 | 0.000121072357 | 4.84289427e-08 | 0.000583887227 | 2.008390340750348 |
| 4 | 7.19236923e-05 | 2.87694769e-08 | 0.000598960224 | 1.786353569670074 |

No actor or optimizer update is retained. Per-trial data and calibration warnings are in calibration.json.

Warnings: ["eta_b spread exceeds 3x"]
