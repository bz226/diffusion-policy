# Fixed-policy gradient noise

Twenty-five fresh batches are the bootstrap units; the 1,000 environment gradients are coupled within a batch by normalization and/or LOO. The specified n-based signal correction is consequently approximate for those variants.

| Variant | Signal norm squared [90% CI] | Noise trace | B_env | B_ep | Batch cosine |
|---|---:|---:|---:|---:|---:|
| nc4 | 2.70584e-07 [1.312794133596961e-06, 3.7103617522121862e-06] | 0.0021066 | 7785.37626603605 | 15570.7525320721 | 0.07149528163150881 |
| mc_critic | 1.41919e-07 [1.196587314156872e-06, 3.640082172062311e-06] | 0.00212682 | 14986.17755180817 | 29972.35510361634 | 0.051594776743189595 |
| mc_loo | 2.54788e-08 [1.3262036479619568e-06, 6.426281933920479e-06] | 0.00298694 | 117232.47014379216 | 234464.94028758432 | 0.018468514265462508 |
| theory_loo | -0.318811 [2.374269836011159, 38.99166883169697] | 11907 | None | None | None |
| theory_nobase | 48.3223 [364.7185043774556, 981.1971411172893] | 589445 | 12198.193565475925 | 24396.38713095185 | 0.057170413937217265 |

Null ratios indicate unresolved positive signal; signed signal estimates are retained. See noise_scale.json for lower bounds, all confidence intervals, the direction-cosine matrix and warmed-critic explained variance.

B_env lower bound uses max(0, variance 5th percentile) / signal 95th percentile, a Bonferroni combination of one-sided 95% percentile bounds. Signed point moments remain unchanged.

Signed debiased moments and unbounded cosine estimates are retained, never clipped. Derived values are null when signal is unresolved; an estimated cosine outside [-1,1] signals estimation uncertainty.
