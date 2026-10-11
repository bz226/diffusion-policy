# Frozen rescale statistics

Converted 25 saved theta0 V4 batches from measurement seed1000; no new rollouts or updates. Each saved vector sum is divided by40 before the float64 coordinate RMS. The global midpoint median defines max(r, 0.001*median(r))+1e-8, then denominators are cast to actor dtype.

Coordinates: 568392; exact zeros: 47798; floored: 49850 (0.0877035567), counting r strictly below the floor before epsilon. The largest ceil(0.01*P)=5684 coordinates contribute 0.790165224 of the total sum of r squared.

| Percentile | r |
|---|---:|
| 0% | 0 |
| 1% | 0 |
| 10% | 1.45295162e-05 |
| 50% | 0.000626695415 |
| 90% | 0.0104395899 |
| 99% | 0.0936042468 |
| 100% | 2.20775439 |

Quantiles use linear interpolation; the 50th percentile is the global midpoint median.

Same E2 estimator/scaling; 40 float32 environment gradients summed in float64 need not equal four shuffled float32 training chunks bitwise. Deterministic conversion of the same cached input; processing seed4000 does not recreate measurement seed1000 rollouts.

See rescale_stats.json for source, ordered parameter geometry and formulas.
