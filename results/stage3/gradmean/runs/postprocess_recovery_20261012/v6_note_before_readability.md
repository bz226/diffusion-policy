# V6: the frozen critic baseline

**Question.** How does the warmed state-only critic change gradient variance relative to LOO along the fixed DPPO and without_PPO paths?

**Answer.** No resolved mean-direction inconsistency was detected at this scheduled analysis. All five policies have at least 500 V6 batches.

| Policy | V6 / all-V4 batches | Critic/LOO variance [90% interval] | V6 noise scale, episodes | Paired direction cosine |
|---|---:|---:|---:|---:|
| P0 | 500 / 3700 | 1.73 [1.68,1.78] | 9.23e+04 | 1.03 |
| D1 | 500 / 2800 | 3.72 [3.42,4.03] | 2.72e+05 | 1.13 |
| D2 | 500 / 3700 | 3.78 [3.5,4.05] | 3.16e+05 | 0.817 |
| N1 | 500 / 2800 | 3.43 [3.25,3.61] | 3.4e+05 | 2.86 |
| N2 | 500 / 3550 | 4.44 [4.05,4.83] | 3.87e+05 | 0.815 |

**Q1—critic versus LOO.** A variance ratio above 1 favors LOO; an interval crossing 1 is unresolved. P0: critic higher; D1: critic higher; D2: critic higher; N1: critic higher; N2: critic higher.

**Q2—decision bins.** The weighted residual-energy ratios are descriptive baseline-quality diagnostics, not measured gradient-variance ratios. P0: 0–49:2.4, 50–99:1.05, 100–149:0.448, 150–199:1.5, 200–249:25.9; D1: 0–49:5.73, 50–99:2.2, 100–149:0.576, 150–199:4.07, 200–249:79.9; D2: 0–49:6.08, 50–99:2.32, 100–149:0.562, 150–199:3.29, 200–249:73.7; N1: 0–49:5.03, 50–99:2.06, 100–149:0.589, 150–199:2.99, 200–249:60.6; N2: 0–49:7.2, 50–99:2.89, 100–149:0.705, 150–199:4.16, 200–249:86.2. [Bin table](runs/analyses/reportfix_20261012/final_v6_fixed/candidates/v6_baseline_bins.csv) includes raw signed critic residual means and their uncertainty; mean residual is baseline error, not policy-gradient bias.

**Q3—middle versus final.** DPPO critic/LOO variance: 3.72 [3.42,4.03] → 3.78 [3.5,4.05]; without_PPO critic/LOO variance: 3.43 [3.25,3.61] → 4.44 [4.05,4.83]. These are fixed seed-0 policy/critic snapshots; no independent training-seed uncertainty is measured.

**Units and uncertainty.** V6 trace and own squared signal use paper units (code gradients × 2500). Noise scale divides the V6 episode-equivalent trace by V4's squared signal using all batches. Its jackknife sums independent legacy/new-cohort variance contributions, deleting new groups jointly from numerator and denominator. Ratios and cosines use only paired new groups, exclude same-batch cross products for mean inner products, and delete each 50-batch group jointly. Bounds for unresolved all-V4 signal are plug-in lower bounds, not joint confidence limits.

**Consistency boundary.** Stop only when both paired signal intervals are positive and a valid 90% cosine interval lies below 1. N/A does not fail; estimates are never clipped. This repeated diagnostic is not a globally calibrated hypothesis test. The compute-driven plan adds exactly 500 V6 batches at each point, then stops; any unresolved directions remain unresolved. Earlier V4 closure records are preserved as historical evidence.

[Full estimates](runs/analyses/reportfix_20261012/final_v6_fixed/candidates/v6_main_results.csv) · [Paired variance ratios](runs/analyses/reportfix_20261012/final_v6_fixed/candidates/v6_variance_ratios.csv) · [Consistency](runs/analyses/reportfix_20261012/final_v6_fixed/candidates/v6_consistency.csv) · [Reproducible statistics](runs/analyses/reportfix_20261012/final_v6_fixed/candidates/v6_results.json). Candidates only; no automatic curation.

**Execution provenance.** H1 passed, H2 passed, H3 passed, H4 passed ([gate](runs/gate_v6/gate_summary.json)). Source switched at 2026-10-12T08:28:50.248219+00:00; old revision `44b2998228e1a93aaef544e8030c30640d17dc06`, new revision `ce909d3b356fd7754963d4ec3f6bf38619022853`. [Switch record](provenance/v6_source_switch.json). The [fixed compute decision](provenance/fixed_v6_plan_20261012.json) preceded V6 data; exact included counts appear above. Earlier batches and candidates are preserved.
