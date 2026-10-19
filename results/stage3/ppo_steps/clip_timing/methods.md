# Methods: clip timing in DPPO's saved per-step data

## Inputs and scope

The input groups are `../runs/C35_j00`–`C35_j09`, `C70_j00`–`C70_j09`, and `C105_j00`–`C105_j09`. Each stores 50 batches of 80 episodes and 20 actor-loss gradients per batch as float64 group sums, plus per-batch scalar statistics. The source measurement revision is `838189eac4794ac921217c50e133534ce95dc6ba`. The primary norm and covariance estimates and intervals come unchanged from `../figs/noise_actual_table.csv`. These estimates describe one training seed and its saved optimizer state at each checkpoint.

No learning, sampling or gradient computation occurs. A new worktree isolates analysis source; it contains no new DPPO checkout or training path. New source files sit beside the existing Part C analysis code. All analysis outputs belong here. Only files that existed before this analysis are included in the preservation inventory; new files are not mistaken for modifications.

## Estimation

For step s and group g, let h(s,g) be its stored sum divided by 50. With G = 10 groups and T(s) = sum_g h(s,g), compute in float64:

`M(s,r) = [dot(T(s),T(r)) - sum_g dot(h(s,g),h(r,g))] / [G(G-1)]`.

Deleting one group recomputes the entire matrix and every derived ratio jointly. A 90% interval is the point estimate plus or minus `t(0.95,9)` times the jackknife standard error. Squared norms remain signed; a nonpositive squared-norm estimate has no real square root. Norm intervals transform signed squared-norm endpoints at the nonnegative boundary; this does not make an unresolved mean resolved. Undefined square roots or ratio replicas are explicitly reported rather than replaced with arbitrary positive numbers.

The extra mean is Δ(s) = E g(s) − E g(1). Its squared norm is `M(s,s)-2M(s,1)+M(1,1)`. Resolution requires its signed 90% interval to lie above zero. Fits through the origin use steps 5–8, with steps 5–6 as the specified sensitivity check. Fit coefficients, q ratios, directions and the on-policy component are recomputed on each delete-group replicate. M-derived cosines use M-derived denominators; both the saved-table and M-based step-1 resolution flags are retained.

For s in a given epoch, C(s) is the sum over earlier epochs of the average of `1 - 0.9**(s-t)` over that epoch's four steps t. C is zero for steps 1–4. The clip half-maximum step and the first post-step-8 q below 0.8 are discrete summaries. If a threshold is not reached, that fact is reported without extrapolation.

Clip scalar means and their intervals use the same ten independent groups. The variance ratio uses joint delete-group covariance estimates reconstructed from stored sums and per-batch squared norms; the primary variance shown remains the saved table value. The two variance calculations must agree numerically.

## Gates and interpretation

G1 verifies all required groups, steps, scalar rows, vector shapes and storage conventions. G2 requires exactly zero switched-off fraction and mean absolute ratio displacement below 1e-4 at step 1. G3 checks interval compatibility between the between-group and saved norm estimates, which need not be identical; an undefined required full-sample square root is reported as a failure. G4 checks every C(s) against the supplied three-decimal values. G5 verifies every pre-existing file's requested before/after hash.

The timing grid retains the existing norm limits [1e-5, 1e-2] and covariance limits [1e-5, 1e-3]. Shape and direction panels show resolved points only. The user approved a cosine range of [−1, 1] after the estimates revealed resolved negative alignments; the original zero lower limit would conceal them. Neither temporal agreement nor a stable direction measures the clip's removed gradient: that requires a separately authorized intervention.
