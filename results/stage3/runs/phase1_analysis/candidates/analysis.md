# Phase 1: fixed-policy gradient noise in HalfCheetah-v2

**Question.** At the pretrained policy and the final NC4 policy, how noisy are V1–V5 at the training batch of 80 episodes, and which estimator changes alter their mean directions?

**Answer.** The point estimates suggest noise-dominated gradients, but estimator rankings and cross-policy changes remain inconclusive. The saved percentile-bootstrap intervals are unreliable; they and confidence-based bounds are excluded. Two squared-signal estimates are negative and seven unique direction-cosine estimates exceed 1, showing that the corrected signal and direction estimates are not reliably resolved.

| Variant | θ₀: B_ep | θ₀: batch cosine | Checkpoint: B_ep | Checkpoint: batch cosine |
| --- | ---: | ---: | ---: | ---: |
| V1 NC4 | 41,253 | 0.044 | 15,571 | 0.071 |
| V2 MC critic | — | — | 29,972 | 0.052 |
| V3 MC LOO | 43,263 | 0.043 | 234,465 | 0.018 |
| V4 theory LOO | 78,285 | 0.032 | — | — |
| V5 theory no baseline | 114,292 | 0.026 | 24,396 | 0.057 |

B_ep is estimated noise trace / squared mean-gradient norm, expressed in episodes; batch cosine is the approximation (1+B_ep/80)^−1/2. Values are provisional points, not uncertainty-qualified conclusions. Dashes retain nonpositive-signal cases. [Noise-scale figure](noise_scale.png) · [Direction figure](direction_cosines.png) · [Captions](figure_captions.md).

**Noise at θ₀.** V5’s B_ep≈114,292 and batch cosine≈0.026 suggest a very weak directional signal at 80 episodes; its estimated SNR is 0.000700. V1/V3/V4 give cosines 0.044/0.043/0.032. V2’s signed squared-signal estimate is −6.38×10⁻⁸, so its noise ratio is unavailable. These measurements do not establish a required batch size.

**Estimator changes.** At θ₀, V3→V4 raises the point noise ratio 1.81×, and V4→V5 raises it 1.46×; dropping LOO raises the raw noise trace 27.4×, but also changes estimated signal. Their corrected direction cosines are 0.709 and −0.269, respectively, with no valid uncertainty. V1→V2 and V2→V3 are undefined at θ₀. A largest-effect ranking and claims of alignment or opposition are therefore unsupported. Raw gradient norms across V1–V5 also use different coefficient/reduction scales.

**Policy change.** From θ₀ to the checkpoint, point B_ep falls from 41,253 to 15,571 for V1 and 114,292 to 24,396 for V5, while V3 rises from 43,263 to 234,465. Checkpoint V4 has signed signal −0.319 and no defined noise ratio. These mixed point changes cannot establish how the underlying noise scale evolved. First-measurement critic explained variance is 0.644/0.273, with 25-batch averages 0.473/0.197; thus the two measurement points include different critic quality.

**Setup.** θ₀ used seed 1000 and 20 critic warm-up batches; NC4 1e-4 seed-0’s final checkpoint used measurement seed 2000 and five warm-ups. Each then measured 25 fresh batches of 40 environment rollouts (two episodes each), totaling 1,000 gradients per variant and 2,000 episodes. Actor, critic and reward scaler were fixed during measurement. All five variants share each batch; normalization and LOO introduce within-batch dependence, making the prescribed per-environment correction approximate. See [methods](../../../methods.md#fixed-policy-noise-measurements-and-uncertainty).

**Boundary.** Saved measurements and the original estimates are preserved. No training, fresh rollouts, bootstrap rerun or replacement uncertainty estimator was used. Resolving uncertainty requires a separate methodological decision, not interpreting missing estimates as zero. [Audit](../../noise_ci_review.json) · [All point moments](point_estimates.csv) · [Directions](direction_cosines.csv) · [Contrasts](contrasts.csv) · [Table definitions](table_caption.md) · [Verification and sources](../analysis_record.json).

**Decision.** Review these candidates for curation; the uncertainty limitation must accompany exports.
