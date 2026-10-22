# Gradient noise of without_PPO

Does replacing GAE by Monte Carlo change the noise of a one-step actor gradient, at identical parameters?

The approved collection is complete. The paired comparison uses native chunk normalization. Later checkpoints use fitted, fixed reward divisors, so those points approximate the historical training scaler; all intervals condition on the fits.

## Q1. Noise of the optimizer gradient

R = tr Cov(g) / ‖E g‖² is unitless. The two own-run references compare different parameters at equal updates.

| Updates | without_PPO R | without_PPO_and_GAE R, own run | DPPO step-1 R, own run |
|---:|---:|---:|---:|
| 0 | 193.13 | 1696.33 | — |
| 32 | 927.121 | 2452.26 | 736.109 |
| 63 | 955.481 | 1249.4 | 56790.5 |
| 95 | 1241.12 | 2579.54 | 6361.51 |
| 126 | 726.469 | 4456.45 | — |

[Own-run comparison](figs/one_step_noise_actual_compare.png). The older green reference uses full-batch normalization, previously checked approximately against its native chunks; the new paired estimates use the exact chunk path. This limits a strict panel-by-panel cross-study interpretation.

## Q2. Paired advantage comparison

| Updates | Monte Carlo / GAE covariance trace | Monte Carlo / GAE R | Within-batch cosine |
|---:|---:|---:|---:|
| 0 | 0.989125 | 11.2301 | 0.699449 |
| 32 | 1.00297 | unresolved | 0.752004 |
| 63 | 0.995216 | unresolved | 0.751288 |
| 95 | 1.00283 | unresolved | 0.75113 |
| 126 | 0.999102 | 1.97251 | 0.751029 |

[Paired figure](figs/gae_vs_mc_paired_noise.png) · [All intervals and mean-direction comparisons](figs/noise_without_ppo_table.md).

## Q3. Resolution

Means not resolved: without_PPO_and_GAE at 32 updates: R ≥ 1454.54; without_PPO at 63 updates: R ≥ 457.49; without_PPO_and_GAE at 63 updates: R ≥ 568.697; without_PPO_and_GAE at 95 updates: R ≥ 2293.7. Ratios drawn are point estimates; bounds condition on the variance estimate.

## Reproducibility and limits

Checkpoint iterations 0, 35, 70, 105 and 139 map to 0, 32, 63, 95 and 126 completed actor updates. Each point is capped at ten independent 50-batch jobs. The table records completed sample counts, scaler values and sources. Group sums and statistics are float64; 90% intervals use a delete-one-job jackknife and t quantiles (9 degrees of freedom for complete points). No gradients were recomputed for analysis.

Shared y-limits: norm [1e-05, 0.01]; covariance [1e-05, 0.001]; R [10.0, 1000000.0].

All new figures retain the existing per-gradient y-limits.

[Analysis record](runs/campaign/candidates/analysis_record.json) records exact input directories, revisions, seeds, command, sample counts and the existing-figure preservation check. See the study's methods and gate records for native gradient identity, state checks and independence verification. This analysis does not establish that GAE versus Monte Carlo alone explains the performance difference between trained policies.

## Verification

Native-gradient identity passed: GAE relative L2 0; Monte Carlo relative L2 0 (limit 1e−6). The critic/state check, independent-stream check and default-path CPU replay all passed. The final production audit checked 50 jobs, 2500 actual batches and 4000 first-batch episode returns; no seed or cross-job first-return duplicate was found. All 100 pre-existing figures retain their requested hashes.

[Exact gate evidence](runs/campaign/candidates/gate_summary.json) · [Analysis/source provenance](runs/campaign/candidates/analysis_record.json). No new gates were run during this analysis.
