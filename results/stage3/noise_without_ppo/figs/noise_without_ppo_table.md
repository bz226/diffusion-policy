# Paired gradient noise at without_PPO's parameters

Square brackets are 90% delete-one-group jackknife intervals with t critical values on groups − 1 degrees of freedom. Each independent group contains 50 fresh 80-episode batches. Norm intervals monotonically transform the signed squared-mean interval (CSV); variance and R use the joint jackknife. R is unitless and B = 80R is in episodes.

| Estimator | Updates | K / groups | Mean norm [90%] | tr Cov(g) [90%] | R [90%] | Mean resolved? | R bound | Expected cosine | B [90%] |
|---|---:|---:|---|---|---|---|---|---|---|
| without_PPO | 0 | 500 / 10 | 0.000538722 [0.000484837, 0.000587687] | 5.60504e-05 [5.57698e-05, 5.6331e-05] | 193.13 [156.448, 229.812] | yes | — | 0.0717718 | 15450.4 [12515.8, 18384.9] |
| without_PPO_and_GAE | 0 | 500 / 10 | 0.000159882 [4.90802e-05, 0.000220715] | 5.54409e-05 [5.52376e-05, 5.56441e-05] | 2168.87 [321.146, 4016.59] | yes | — | 0.0214676 | 173510 [25691.7, 321327] |
| without_PPO | 32 | 500 / 10 | 0.000241141 [0.000176811, 0.000291609] | 5.39112e-05 [5.36321e-05, 5.41903e-05] | 927.121 [487.602, 1366.64] | yes | — | 0.0328245 | 74169.7 [39008.1, 109331] |
| without_PPO_and_GAE | 32 | 500 / 10 | 0.000108303 [0, 0.000192805] | 5.40711e-05 [5.36276e-05, 5.45147e-05] | 4609.85 [0, ∞] | no | ≥ 1454.54 | ≤ 0.0262112 | 368788 [0, ∞] |
| without_PPO | 63 | 500 / 10 | 0.000235977 [0, 0.000341028] | 5.32063e-05 [5.28264e-05, 5.35862e-05] | 955.481 [0, ∞] | no | ≥ 457.49 | ≤ 0.0467019 | 76438.5 [0, ∞] |
| without_PPO_and_GAE | 63 | 500 / 10 | 0.000190131 [0, 0.00030514] | 5.29517e-05 [5.26735e-05, 5.323e-05] | 1464.79 [0, ∞] | no | ≥ 568.697 | ≤ 0.0418965 | 117183 [0, ∞] |
| without_PPO | 95 | 500 / 10 | 0.000205814 [0.0001352, 0.00025776] | 5.25734e-05 [5.22081e-05, 5.29387e-05] | 1241.12 [571.266, 1910.98] | yes | — | 0.0283738 | 99289.7 [45701.3, 152878] |
| without_PPO_and_GAE | 95 | 500 / 10 | 7.56679e-05 [0, 0.00015161] | 5.2722e-05 [5.22365e-05, 5.32076e-05] | 9208.08 [0, ∞] | no | ≥ 2293.7 | ≤ 0.0208755 | 736647 [0, ∞] |
| without_PPO | 126 | 500 / 10 | 0.000267986 [0.000170022, 0.000338712] | 5.21725e-05 [5.19495e-05, 5.23955e-05] | 726.469 [335.163, 1117.77] | yes | — | 0.037076 | 58117.5 [26813.1, 89422] |
| without_PPO_and_GAE | 126 | 500 / 10 | 0.000190725 [1.15101e-05, 0.00026948] | 5.21257e-05 [5.17712e-05, 5.24801e-05] | 1432.97 [-133.399, 2999.33] | yes | — | 0.0264077 | 114637 [-10671.9, 239947] |

## Paired Monte Carlo / GAE comparison

Both estimators use each batch's identical parameters, critic and native chunk normalization. The mean dot product excludes same-group pairs. Its cosine uses that same group U statistic for both denominator squared norms; it is unresolved if either full or deleted-group squared norm is nonpositive. R ratios and direction cosines are shown only when both individual means resolve.

| Updates | Trace ratio [90%] | R ratio [90%] | Mean dot product [90%] | Mean-direction cosine [90%] | Within-batch cosine [90%] |
|---:|---|---|---|---|---|
| 0 | 0.989125 [0.983667, 0.994583] | 11.2301 [2.20565, 20.2546] | 6.85195e-08 [1.94566e-08, 1.17582e-07] | 0.862807 [0.458583, 1.26703] | 0.699449 [0.696765, 0.702132] |
| 32 | 1.00297 [0.995628, 1.01031] | unresolved [unresolved, unresolved] | 3.27627e-08 [1.4485e-08, 5.10403e-08] | unresolved [unresolved, unresolved] | 0.752004 [0.750551, 0.753456] |
| 63 | 0.995216 [0.988946, 1.00149] | unresolved [unresolved, unresolved] | 3.89573e-08 [-1.61348e-08, 9.40495e-08] | unresolved [unresolved, unresolved] | 0.751288 [0.749142, 0.753434] |
| 95 | 1.00283 [0.99682, 1.00884] | unresolved [unresolved, unresolved] | 2.05484e-08 [-6.18413e-10, 4.17153e-08] | unresolved [unresolved, unresolved] | 0.75113 [0.749537, 0.752723] |
| 126 | 0.999102 [0.993692, 1.00451] | 1.97251 [-0.14841, 4.09343] | 5.15853e-08 [1.21878e-08, 9.09829e-08] | 0.97934 [0.863689, 1.09499] | 0.751029 [0.749174, 0.752883] |

## State and resolution limits

A mean is resolved only when its squared-mean interval is entirely positive. If it reaches zero, the requested bound R ≥ tr Cov(g) / upper(‖E g‖²) holds the variance at its point estimate; it is not a simultaneous confidence bound for numerator and denominator. The implied expected-cosine bound is an upper bound. The joint R interval is reported as [0, ∞) when the mean is unresolved or a deleted-group denominator is nonpositive; the separate plug-in bound is never its lower endpoint. Undefined nonpositive point estimates remain unresolved, never clipped to positive values.

| Updates | Reward divisor | Source | Approximation? |
|---:|---:|---|---|
| 0 | 1 | exact_initial_scaler_state_updated_per_measurement_batch | no |
| 32 | 440.462 | least_squares_five_fresh_batches_frozen_critic_through_origin | yes |
| 63 | 445.885 | least_squares_five_fresh_batches_frozen_critic_through_origin | yes |
| 95 | 443.26 | least_squares_five_fresh_batches_frozen_critic_through_origin | yes |
| 126 | 447.777 | least_squares_five_fresh_batches_frozen_critic_through_origin | yes |

Later fitted divisors do not reconstruct the historical running scaler. Their figures and intervals condition on this approximation. CSV estimator codes identify the formulas, not additional training runs.
