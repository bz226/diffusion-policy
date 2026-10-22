# Methods: DPPO batch averages and mean-gradient turning

## Question, data and boundaries

This diagnostic tests whether systematic parts of DPPO's step gradients cancel when averaged over a batch and whether the result retains epoch 1's mean direction. It is not a replay, a causal clipping intervention, or a reconstruction of the actual AdamW displacement. The user supplied the estimators and comparison design.

Use Part C groups `../runs/C35_j00..09`, `C70_j00..09` and `C105_j00..09`. Each contains 50 fresh batches from the same saved policy/optimizer state. The 20 step gradients within a batch are dependent; the ten jobs are the independent jackknife units. Results are conditional on one training seed and its three checkpoints. Frozen Adam states are `../runs/B_seed0/ppo_steps/full_state_{35,70,105}.pt`. The prior [clip-timing methods](../clip_timing/methods.md) define the reused between-group Gram matrix.

| Quantity | Meaning and unit | Fixed choice and source |
|---|---|---|
| K, groups, group size | measurement sample size | 500 batches, 10 groups, 50 batches/group; user constraint |
| PPO steps | raw actor-loss gradients in one batch | 20, four per epoch; stored Part C data |
| Actor coordinates | flattened actor_ft parameter order | 568,392; checked against stored parameter names/shapes |
| Epoch coefficients | four step gradients averaged | 1/4 within each epoch; user definition |
| Batch coefficients | all step gradients averaged | 1/20; user definition |
| Uncertainty | delete-one-group intervals | 90%, t quantile on 9 degrees of freedom; user constraint |
| Adam P | fixed checkpoint preconditioner | 1/(sqrt(v/(1−beta2^step))+eps), each parameter's stored state |
| Resource cap | local CPU computation | 30 minutes, four threads, 8 GiB; explicit approval |

## Joint estimates

Let h_g be group g's mean gradient at each of the 20 steps, and T = sum_g h_g. Reuse the clip-timing function for

`M = [T T^T − sum_g h_g h_g^T] / [G(G−1)]`.

For fixed coefficient vectors a,b, the mean inner product is a^T M b. The covariance Gram matrix of one batch's gradients is

`V = 50 [sum_g h_g h_g^T − T T^T/G] / (G−1)`.

Its quadratic form a^T V a estimates tr Cov(X). Recompute both matrices after deleting each group. Norms, noise-to-signal ratios, cosines, on-policy components and signal-kept ratios use the same full and deleted-group matrices jointly. Retain signed squared-norm and inner-product estimates. Resolution means the relevant signed 90% interval excludes zero; a squared norm must be resolved positively to define a resolved mean direction. Undefined square roots and denominator cases remain N/A, with valid-replicate counts; no clipping or positive-semidefinite repair is applied.

Epoch 1 uses coefficients (1/4,1/4,1/4,1/4,0,...). It is an approximation to the full gradient at the old policy, since parameters move between its minibatches and normalization is per minibatch. When this mean is unresolved, report the batch/epoch-1 inner product but leave directional components, signal kept and directional relative variance blank.

For the fixed Adam metric, compute separate M,V using P h_g. A Gram matrix using sqrt(P) h_g gives the mixed products a^T M_cross b = <E X,P E Y>. This includes the checkpoint second-moment scaling and epsilon, but excludes the trajectory of later moment changes and inherited first-moment effects on displacement.

## Turning and principal coordinates

Apply the differences g_s−g_1 for s=5..20 to M to form the uncentered extra-mean Gram matrix Γ. Keep all signed eigenvalues and report their shares of tr Γ, including negative shares from estimation noise. Principal coordinates use sqrt(lambda) times the eigenvector only for positive eigenvalues. Sort eigenvalues in descending order; anchor full-sample eigenvector signs and align deleted-group signs to the corresponding full-sample vector. Report coordinate instability if the decomposition is poorly separated. The origin remains no extra mean; do not center or repair Γ.

Cancellation is the square root of the all-entries sum of Γ divided by the sum of square roots of its diagonal entries. It and the cumulative eigenvalue shares are recomputed in every deletion. Nonpositive required squared norms can leave a point estimate or interval undefined; this is retained rather than resolved by regularization.

## Gates, outputs and reproducibility

G1 checks all group files, counts and vector schemas. G2 compares the sum of the 20 stored step sums with each stored G sum to relative L2 1e−8. G3 checks interval compatibility of group-based and all-batch variance and squared-mean estimators for all steps and G, reporting joint-jackknife ratios. G4 compares recomputed full and deleted-group M with the prior cache, allowing only documented float64 roundoff. G5 repeats the user's requested hashes of every file present before this analysis.

Pseudocode: verify the protected inventory; load one checkpoint's groups; compute raw gates; stop on failure; form fixed-Adam matrices; derive the requested joint statistics and coordinates; freeze the available reference table; render only after gates pass; verify protected files again. Use new Python 3.8 scripts beside the existing analysis and save a source snapshot plus the parent revision. Preserve inputs, source measurement revisions, exact commands and the approval in run records.
