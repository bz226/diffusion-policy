# Figure definitions

All E0/E1C/E1 curves show three-seed mean ± one sample standard deviation (ddof=1), with thin individual seeds; means require all three finite values. The x-axis counts actor updates applied before each rollout: training itr1 is x=0 and eval itr130 is x=117. A training diagnostic describes the update after that rollout, while returns and saturation concern its pre-update policy. Partial runs end where observed; no endpoint is extrapolated, and fixed-window summaries require every specified update in every seed.

**Returns:** undiscounted evaluation/training episode returns and raw discounted train-sampler return, gamma=0.99 per four-action decision, horizon250. Faint dashed DPPO references are its three-seed final scheduled eval and last-ten-training-batch means, not an invented single-step optimizer history. Stage1 did not record discounted return or the other Stage3 diagnostics, so no DPPO values are fabricated in those panels.

**Updates:** true summed-log-prob KL per action (log axis) and post-update actor L2 distance from pretrained parameters. **Saturation/direction:** fraction of coordinates outside the unchanged x0 clamp before the update and cosine with the previous training gradient. Undefined first-gradient cosines are gaps.

**Baseline quality:** MC+critic's unweighted residual-energy ratio (log axis), derived as sum_k(mean_k²+population_std_k²) separately for critic and LOO; the original weighted energy fields are not used. Explained variance is DPPO's scaled-target critic diagnostic, not variance explained in raw Monte Carlo returns. **Baseline bins:** ratio of time-averaged residual energies within each k-bin and specified update window; error bars are sample sd across three seeds, not episode uncertainty.

These plots condition on three training seeds; small differences are unresolved under the user's twice-pooled-sd rule. Fixed-policy V1–V3 noise measurements use full-batch normalization and warmed/frozen critics and are proxies for the training estimators. Nonpositive log-axis values and lower SD bounds are omitted, not clipped to arbitrary positive constants.
