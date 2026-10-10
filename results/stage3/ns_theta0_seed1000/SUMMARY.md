# Fixed-policy gradient noise

Twenty-five fresh batches are the bootstrap units; the 1,000 environment gradients are coupled within a batch by normalization and/or LOO. The specified n-based signal correction is consequently approximate for those variants.

| Variant | Signal norm squared [90% CI] | Noise trace | B_env | B_ep | Batch cosine |
|---|---:|---:|---:|---:|---:|
| nc4 | 1.05426e-07 [1.3295744177853914e-06, 3.5300624637576678e-06] | 0.00217456 | 20626.46876542398 | 41252.93753084796 | 0.04399434514890001 |
| mc_critic | -6.38305e-08 [1.1568558433991267e-06, 3.2443165206290594e-06] | 0.00217571 | None | None | None |
| mc_loo | 1.08836e-07 [1.3447328696178078e-06, 3.620076403895452e-06] | 0.00235428 | 21631.444028783477 | 43262.888057566954 | 0.04296215626240017 |
| theory_loo | 0.482817 [10.23789221245015, 29.237063416627386] | 18898.6 | 39142.4373681211 | 78284.8747362422 | 0.03195098649336166 |
| theory_nobase | 9.06636 [299.9886272099814, 780.2644835902587] | 518105 | 57145.84851195132 | 114291.69702390264 | 0.026447566081796527 |

Null ratios indicate unresolved positive signal; signed signal estimates are retained. See noise_scale.json for lower bounds, all confidence intervals, the direction-cosine matrix and warmed-critic explained variance.

B_env lower bound uses max(0, variance 5th percentile) / signal 95th percentile, a Bonferroni combination of one-sided 95% percentile bounds. Signed point moments remain unchanged.

Signed debiased moments and unbounded cosine estimates are retained, never clipped. Derived values are null when signal is unresolved; an estimated cosine outside [-1,1] signals estimation uncertainty.
