# Addendum G: fixed-policy gradient signal and noise

**Question.** At five pretrained, DPPO and NC4 checkpoints, how large and reliable is the mean discounted-return gradient, and how does it relate to the observed policy displacements?

**Answer.** This partial analysis contains verified measurements at 5 of five checkpoints; 0 meet the prescribed V4 split-correlation threshold of 0.68. Resolved V4 point estimates have noise scale above the 80-episode training batch at 5 of 5 measured checkpoints. No measured V4 signal/noise ratio is currently classified as unresolved.

**Main limitation.** These are measurements at fixed checkpoints from one training seed per path. They do not show that gradient noise caused a training outcome. Unresolved signals, missing data and failed groups remain visible; a sample-cap closure does not establish a reliable direction. This interim report cannot support a final five-checkpoint conclusion.

Numbers in square brackets are 90% delete-one-group normal intervals (±1.645 SE), unless stated otherwise. One independent group contains 50 whole batches of 80 episodes. V4/V5 use paper ascent-gradient units; V1–V3 publish noise scales and cosines only. A ≥ or ≤ plug-in bound uses the estimated trace and the upper signal interval endpoint and has **no joint confidence guarantee**. N/A is retained when a statistic or its uncertainty is undefined.

## Q1. Is the gradient noise-dominated at 80 episodes?

| Point | Batches K | V4 B (episodes) | Expected 80-episode batch cosine |
| --- | --- | --- | --- |
| P0 | 2300 | 5.375e+04 [4.896e+04, 5.854e+04] | 0.03855 [0.03684, 0.04026] |
| D1 | 2300 | 7.042e+04 [5.652e+04, 8.433e+04] | 0.03369 [0.03037, 0.037] |
| D2 | 2300 | 8.668e+04 [7.443e+04, 9.893e+04] | 0.03037 [0.02822, 0.03251] |
| N1 | 2300 | 9.527e+04 [7.774e+04, 1.128e+05] | 0.02897 [0.0263, 0.03164] |
| N2 | 2300 | 7.7e+04 [5.996e+04, 9.405e+04] | 0.03222 [0.02863, 0.0358] |

B above 80 means the estimated gradient-noise trace exceeds squared signal at this batch size; it is a descriptive scale comparison, not a hypothesis test. The expected cosine is the prescribed moment approximation, not a directly observed cosine. [main results](main_results.csv) includes all five estimators and their uncertainty.

## Q2. How do signal, noise and performance change along each path?

| Point / path position | Discounted J | V4 gradient norm | V4 episode trace | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| P0 | 1393 [1393, 1394] | 2025 [1934, 2116] | 2.205e+11 [2.174e+11, 2.236e+11] | 5.375e+04 [4.896e+04, 5.854e+04] |
| D1 | 1515 [1514, 1515] | 1187 [1075, 1299] | 9.917e+10 [9.509e+10, 1.033e+11] | 7.042e+04 [5.652e+04, 8.433e+04] |
| D2 | 1533 [1533, 1534] | 1105 [1027, 1182] | 1.058e+11 [1.02e+11, 1.095e+11] | 8.668e+04 [7.443e+04, 9.893e+04] |
| N1 | 1496 [1496, 1497] | 1090 [991.5, 1188] | 1.132e+11 [1.097e+11, 1.166e+11] | 9.527e+04 [7.774e+04, 1.128e+05] |
| N2 | 1518 [1517, 1518] | 1043 [931.8, 1155] | 8.383e+10 [7.941e+10, 8.825e+10] | 7.7e+04 [5.996e+04, 9.405e+04] |

Read the DPPO path as P0 → D1 → D2 and the NC4 path as P0 → N1 → N2. Unresolved norms and interval width limit descriptive ordering; marginal interval overlap is not used as a significance test. A higher return at these points does not by itself establish a general relation between policy quality and gradient norm.

## Q3. How do DPPO and NC4 compare at matched iterations?

| Saved iteration | DPPO / NC4 | Discounted J | V4 gradient norm | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| 70 | D1 | 1515 [1514, 1515] | 1187 [1075, 1299] | 7.042e+04 [5.652e+04, 8.433e+04] |
| 70 | N1 | 1496 [1496, 1497] | 1090 [991.5, 1188] | 9.527e+04 [7.774e+04, 1.128e+05] |
| 139 | D2 | 1533 [1533, 1534] | 1105 [1027, 1182] | 8.668e+04 [7.443e+04, 9.893e+04] |
| 139 | N2 | 1518 [1517, 1518] | 1043 [931.8, 1155] | 7.7e+04 [5.996e+04, 9.405e+04] |

The paired rows match saved iteration and training seed 0, not independently replicated training paths. No additional significance test or derived DPPO/NC4 ratio is introduced.

## Q4. How much does the baseline choice change variance or noise scale?

| Point | No baseline / LOO trace (V5/V4) | Critic / LOO B with MC (V2/V3) |
| --- | --- | --- |
| P0 | 29.34 [28.92, 29.75] | 3.426 [2.778, 4.075] |
| D1 | 79.65 [76.31, 82.99] | 13.21 [4.667, 21.76] |
| D2 | 79.77 [76.85, 82.69] | 11.88 [6.038, 17.72] |
| N1 | 64.15 [62.14, 66.15] | 9.072 [5.529, 12.61] |
| N2 | 88.17 [83.51, 92.82] | 10.06 [1.923, 18.2] |

A V5/V4 ratio above one gives the measured multiplicative variance reduction from LOO. V2/V3 compares noise scales, not pure variances, because their normalized estimators can have different mean signals. These contrasts delete the same groups jointly across variants; an unresolved denominator remains N/A. [contrasts](contrasts.csv).

## Q5. Are the DPPO-style and E1-style measurement estimators aligned with the discounted-return gradient?

| Point | cos(V1, V4) | cos(V3, V4) |
| --- | --- | --- |
| P0 | 0.9933 [0.964, 1.023] | 1 [0.9939, 1.006] |
| D1 | 0.9407 [0.8693, 1.012] | 0.9968 [0.9887, 1.005] |
| D2 | 1.023 [0.9366, 1.109] | 0.9635 [0.9352, 0.9917] |
| N1 | 0.9592 [0.8646, 1.054] | 0.9996 [0.9883, 1.011] |
| N2 | 1.03 [0.9618, 1.099] | 0.9897 [0.9799, 0.9996] |

V1 is the unclamped NC4/DPPO-style measurement proxy; V3 is the E1-style MC/LOO measurement variant. Both use full-batch advantage normalization, whereas training normalizes per chunk and original DPPO also retains coordinate log-probability clamps, so these are not exact training-update gradients. V4 estimates the raw discounted-return gradient. Same-batch products are excluded before forming the cosines; estimates outside [−1, 1] remain visible as estimation error, and N/A does not imply orthogonality. [same point cosines](same_point_cosines.csv).

## Q6. How much does the discounted-return gradient rotate?

| Checkpoint pair | V4 mean-gradient cosine |
| --- | --- |
| P0 / D1 | 0.4455 [0.3609, 0.5301] |
| P0 / D2 | 0.4497 [0.3759, 0.5234] |
| P0 / N1 | 0.7367 [0.6432, 0.8302] |
| P0 / N2 | 0.4366 [0.3509, 0.5223] |
| D1 / D2 | 0.7393 [0.6253, 0.8534] |
| D1 / N1 | 0.7224 [0.6474, 0.7973] |
| D1 / N2 | 0.7386 [0.6687, 0.8086] |
| D2 / N1 | 0.6069 [0.5015, 0.7123] |
| D2 / N2 | 0.5973 [0.4918, 0.7028] |
| N1 / N2 | 0.8738 [0.7905, 0.957] |

A lower cosine indicates a larger directional change only where signals and intervals are resolved. Different policies use independent measurement groups; this is not a paired-rollout comparison. The complete V1 and V4 matrices are in [cross point cosines](cross_point_cosines.csv).

## Q7. Do checkpoint displacements follow the starting gradient?

| Segment | Distance | Start cosine | First-order ΔJ | Trapezoid ΔJ | Measured ΔJ |
| --- | --- | --- | --- | --- | --- |
| P0 → D1 | 5.054 | 0.02429 [0.0218, 0.02678] | 248.6 [224.2, 273.1] | 97.09 [82.02, 112.2] | 121.3 [120.4, 122.2] |
| D1 → D2 | 4.659 | 0.01294 [0.009593, 0.01629] | 71.55 [55.22, 87.89] | 7.763 [-3.45, 18.98] | 18.52 [17.84, 19.2] |
| P0 → D2 | 6.858 | 0.02116 [0.01832, 0.02401] | 294 [255.4, 332.5] | 87.89 [63.34, 112.4] | 139.8 [139, 140.7] |
| P0 → N1 | 0.6353 | 0.1149 [0.1091, 0.1207] | 147.9 [140.2, 155.5] | 96.01 [91.11, 100.9] | 102.6 [101.7, 103.6] |
| N1 → N2 | 0.5521 | 0.05731 [0.04643, 0.06819] | 34.49 [29.92, 39.05] | 18.82 [15.47, 22.17] | 21.38 [20.66, 22.1] |
| P0 → N2 | 0.9097 | 0.113 [0.1076, 0.1185] | 208.3 [198.1, 218.4] | 113.2 [106.3, 120.1] | 124 [123.2, 124.9] |

The first-order prediction is the starting V4 gradient dotted with the checkpoint displacement; the trapezoid averages the two endpoint gradients. Numerical agreement or disagreement is shown directly without a new residual test. Curvature between endpoints and the difference between a checkpoint chord and the actual training path limit this approximation. [path alignment](path_alignment.csv).

## Q8. How loose is the empirical variance-lemma scale?

| Point | Mean score norm² | Max decision-decile mean | Score norm² p99 | Lemma / V5 episode trace |
| --- | --- | --- | --- | --- |
| P0 | 6.737e+04 | 7.344e+04 | 1.177e+05 | 9.619 |
| D1 | 6.914e+04 | 7.658e+04 | 1.305e+05 | 8.232 |
| D2 | 7.188e+04 | 7.824e+04 | 1.298e+05 | 7.958 |
| N1 | 6.415e+04 | 6.732e+04 | 1.074e+05 | 8.584 |
| N2 | 6.328e+04 | 6.988e+04 | 1.221e+05 | 8.098 |

The scale is observed r_max² × mean score norm² / (1 − 0.99)³. Its ratio to V5 trace reports empirical looseness, not a verified global bound: only 200 random decision scores from one group and an observed reward maximum are available. Group-jackknife uncertainty is N/A by the approved exception. Raw maxima, scale values and decile means are in [score variance lemma](score_variance_lemma.csv).

## Q9. Does training drift agree with the direct P0 signal measurement?

The direct P0 squared-gradient signal in paper units is **4.102e+06 [3.733e+06, 4.471e+06]**.

| Existing condition | J slope per actor update | Squared-gradient drift proxy | Proxy / P0 signal |
| --- | --- | --- | --- |
| SGD1 | 0.1795 [0.1633, 0.1956] | 3.706e+06 [3.372e+06, 4.039e+06] | 0.9034 [0.7884, 1.018] |
| SGD3 | 0.452 [0.3884, 0.5156] | 3.111e+06 [2.673e+06, 3.549e+06] | 0.7584 [0.6317, 0.8851] |
| SGD10 | 0.6417 [0.1486, 1.135] | 1.325e+06 [3.068e+05, 2.343e+06] | 0.323 [0.0731, 0.5729] |
| BATCH4 | 0.4576 [0.4455, 0.4697] | 3.15e+06 [3.066e+06, 3.233e+06] | 0.7679 [0.6958, 0.8399] |

Each seed uses OLS over all 126 pre-update training returns against completed actor updates 0–125; dividing by actor_lr/2500 forms the proxy. Intervals aggregate three independent training seeds, and the ratio combines that uncertainty with independent P0 groups. This is a path-average proxy, not a local θ0 gradient measurement; SGD10 is especially sensitive to its larger step. Agreement is descriptive and cannot validate a local drift identity. [drift per seed](drift_per_seed.csv) and [drift summary](drift_summary.csv).

## Q10. Do the old Phase-1 estimates fall inside the new intervals?

| Point / variant | Metric | Old point | New estimate / 90% interval | Old point inside? |
| --- | --- | --- | --- | --- |
| P0 / V1 | B_ep | 4.125e+04 | 3.264e+04 [2.982e+04, 3.545e+04] | No |
| P0 / V2 | B_ep | N/A | 1.247e+05 [1.015e+05, 1.479e+05] | N/A |
| P0 / V3 | B_ep | 4.326e+04 | 3.639e+04 [3.357e+04, 3.92e+04] | No |
| P0 / V4 | B_ep | 7.828e+04 | 5.375e+04 [4.896e+04, 5.854e+04] | No |
| P0 / V4 | u_sq_theory | 3.018e+06 | 4.102e+06 [3.733e+06, 4.471e+06] | No |
| P0 / V4 | trace_ep_theory | 2.362e+11 | 2.205e+11 [2.174e+11, 2.236e+11] | No |
| P0 / V5 | B_ep | 1.143e+05 | ≥ 1.195e+06 (plug-in; no point CI) | N/A |
| P0 / V5 | u_sq_theory | 5.666e+07 | 1.086e+06 [-3.241e+06, 5.413e+06] | No |
| P0 / V5 | trace_ep_theory | 6.476e+12 | 6.468e+12 [6.452e+12, 6.483e+12] | Yes |
| N2 / V1 | B_ep | 1.557e+04 | 1.55e+05 [9.441e+04, 2.156e+05] | No |
| N2 / V2 | B_ep | 2.997e+04 | 3.694e+05 [6.236e+04, 6.765e+05] | No |
| N2 / V3 | B_ep | 2.345e+05 | 3.671e+04 [3.118e+04, 4.224e+04] | No |
| N2 / V4 | B_ep | N/A | 7.7e+04 [5.996e+04, 9.405e+04] | N/A |
| N2 / V4 | u_sq_theory | -1.993e+06 | 1.089e+06 [8.554e+05, 1.322e+06] | No |
| N2 / V4 | trace_ep_theory | 1.488e+11 | 8.383e+10 [7.941e+10, 8.825e+10] | No |
| N2 / V5 | B_ep | 2.44e+04 | ≥ 1.204e+06 (plug-in; no point CI) | N/A |
| N2 / V5 | u_sq_theory | 3.02e+08 | 5.625e+05 [-5.012e+06, 6.137e+06] | No |
| N2 / V5 | trace_ep_theory | 7.368e+12 | 7.391e+12 [7.372e+12, 7.409e+12] | No |

Old intervals are not reused: within-batch dependence made their uncertainty unreliable, and critic/scaler preparation differs for V1/V2. Containment of one provisional point in a new interval is not a formal consistency test. Dimensioned comparisons are shown only for V4/V5 after the documented unit conversion. [phase1 comparison](phase1_comparison.csv).

## Measurement reliability and checkpoint inventory

| Point | Groups / batches | ρ | Mean-direction c | Moment c_U | Effective rank | Forecast batches |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 46 / 2300 | 0.6147 [0.588, 0.6415] | 0.8726 [0.8608, 0.8844] | 0.8797 [0.8707, 0.8887] | 418.8 [342.9, 494.8] | 2862 [2607, 3117] |
| D1 | 46 / 2300 | 0.5627 [0.5036, 0.6217] | 0.8486 [0.8199, 0.8773] | 0.8504 [0.8269, 0.8739] | 321.6 [239, 404.2] | 3750 [3010, 4491] |
| D2 | 46 / 2300 | 0.5106 [0.4641, 0.5571] | 0.8222 [0.7972, 0.8472] | 0.8245 [0.8056, 0.8434] | 247.2 [195.1, 299.3] | 4616 [3963, 5268] |
| N1 | 46 / 2300 | 0.5004 [0.4517, 0.5491] | 0.8167 [0.7899, 0.8436] | 0.8117 [0.7859, 0.8375] | 379 [270.6, 487.4] | 5073 [4140, 6006] |
| N2 | 46 / 2300 | 0.5519 [0.4925, 0.6113] | 0.8434 [0.8138, 0.873] | 0.8396 [0.8118, 0.8674] | 415.8 [348.7, 482.9] | 4101 [3193, 5008] |

ρ averages 20 balanced group half-splits; c describes the full measurement mean, unlike the 80-episode cosine in Q1. Effective rank uses the approved mean-free four-group difference statistic. Forecasts are descriptive: closure occurs only at approved round boundaries when ρ ≥ 0.68 or at 4,100 batches. Repeated convergence prefixes are dependent.

| Point | Original actor checkpoint | Distance from P0 | Frozen-preparation batches |
| --- | --- | --- | --- |
| P0 | [state_3000.pt](../../../../../../repro_halfcheetah/dppo/log/gym-pretrain/halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-04-42/checkpoint/state_3000.pt) | 0 | 20 |
| D1 | [state_70.pt](../../../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_20-01-32_0/checkpoint/state_70.pt) | 5.054 | 5 |
| D2 | [state_139.pt](../../../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_20-01-32_0/checkpoint/state_139.pt) | 6.858 | 5 |
| N1 | [state_70.pt](../../../../../../stage2_noclip/runs/NC4_lr1e-4_seed0/native/checkpoint/state_70.pt) | 0.6353 | 5 |
| N2 | [state_139.pt](../../../../../../stage2_noclip/runs/NC4_lr1e-4_seed0/native/checkpoint/state_139.pt) | 0.9097 | 5 |

Distances use the registered actor parameter order, float64 differences of saved float32 values, and no coordinate scaling. P0 is the released pretrained EMA policy; the four RL checkpoints are from training seed 0. [Checkpoint mapping](../../../../provenance/checkpoints.json) · [Distance and parameter-order record](../internal/checkpoint_metadata.json) · [per point](per_point.csv) (discounted and undiscounted returns, SEs, saturation, out-of-range actions and explained variance).

## Verification gates and retained failures

G1 fixed-batch gradient comparison: **passed**. [Offline gate record](../../../gate_offline_G1/gate_summary.json).

| Variant / check | Maximum absolute difference | Relative L2 difference |
| --- | --- | --- |
| V1 | 7.945e-10 | 5.703e-07 |
| V2 | 1.476e-09 | 6.291e-07 |
| V3 | 5.286e-10 | 5.959e-07 |
| V4 | 2.721e-06 | 6.34e-07 |
| V5 | 1.092e-05 | 6.596e-07 |
| V4 vs negative E2 loss gradient | 0 | 0 |

G5 default/NC4 fixed-batch replay compares losses, actor/critic gradients, updated parameters, optimizer states, sampling and RNG states. The table summarizes existing recorded comparisons without rerunning them.

| Replay | Status | All comparisons bitwise equal | Max abs | Max relative |
| --- | --- | --- | --- | --- |
| default_cpu | Passed | True | 0 | 0 |
| default_cuda0 | Passed | True | 0 | 0 |
| nc4_cpu | Passed | True | 0 | 0 |
| nc4_cuda0 | Passed | True | 0 | 0 |

Amended G2 uses 200 predeclared replicas in each regime, P = 20,000, K = 2,000, groups of 50, seed 8675309 and the same generator streams. Formal interval-coverage criteria apply to signal and trace only. [G2-G1 record](../../../gate_G2_G1/gate_summary.json).

| Regime | Status | Signal mean / truth; MC SE | Trace mean / truth; MC SE | Signal coverage | Trace coverage | Rank mean / truth |
| --- | --- | --- | --- | --- | --- | --- |
| L | Passed | 0.997 / 1; SE 0.03356 | 4000 / 4000; SE 1.161 | 0.965 | 0.9 | 61.62 / 61.69 |
| T | Passed | 1.003 / 1; SE 0.009053 | 470 / 470; SE 0.1364 | 0.92 | 0.9 | 61.62 / 61.69 |

Low-signal L has 1/200 nonpositive split correlations; its conditional signed cosine error is -0.02256 over 199 defined replicas and is descriptive. Target T has mean absolute cosine error 0.01493, with 200/200 replicas within absolute error 0.06 (required fraction ≥ 0.95). Pool shutdown success: True; surviving child PIDs: []; expected process exit: 0.

G2 scheduler job 17812422: state COMPLETED, exit 0:0, elapsed 44 seconds, 0 GPUs. [Terminal scheduler record](../../../gate_G2_G1/scheduler_completion.json).

G3 independent collection/cleanup: passed; seeds [100000, 100100]; different initial observations True and returns True; process exit codes [0, 0, 0]; surviving children []. G4 frozen actor: passed. [Live gate record](../../../gate_live_G1/gate_summary.json).

Automatic pilot timing review: **complete**. [Pilot forecast report](../../../pilot_review/pilot_review.md). The forecast is descriptive; the GPU-hour cap and admission policy remain unchanged.

The original G2 **failed** because one of 200 low-signal replicas had undefined direction reliability. Its saved summary and subsequent Python 3.8 cleanup traceback are preserved; the user approved Amendment G-1 to treat this case as unresolved and add the target regime. This is a protocol amendment after a retained failure, not an unchanged-gate success. [Original failed gate](../../../gate_G2/gate_summary.json).

## Execution, resources and deviations

Analysis parent commit: `fa8e70833275dcd27a3984e935b9618f20c62a21`. Training/collection source revisions are recorded separately per run; a later analysis commit does not replace the source snapshot used for collection. [Revision history](../../../../provenance/revisions.json).

Collection uses one A6000 GPU, 40 CPUs and 64 GiB per job, at most nine concurrent GPU jobs, and a 125 GPU-hour total cap. Each preparation or 50-batch collection job is capped at one hour. CPU analyses use four CPUs, 64 GiB and a 30-minute allocation; gate allocations are recorded separately.

Recorded used GPU hours: 47.36; outstanding reserved GPU hours: 0; cap: 125. These are scheduler allocation totals, including failed allocations; parallel job times must not be added to estimate campaign wall-clock duration.

Campaign record: 2026-10-11T20:55:24.992576+00:00 through 2026-10-12T04:47:23.158420+00:00, status `running`. Analysis started 2026-10-12T04:46:55.527053+00:00; elapsed wall time through report generation 92.56 seconds. [Campaign ledger](../../../campaign.json).

| Point | Collection attempts | Complete workers | Min job seconds | Max job seconds | Sum job seconds |
| --- | --- | --- | --- | --- | --- |
| P0 | 46 | 46 | 691.4 | 811.6 | 3.316e+04 |
| D1 | 46 | 46 | 691.5 | 799.2 | 3.321e+04 |
| D2 | 46 | 46 | 692.5 | 843 | 3.362e+04 |
| N1 | 46 | 46 | 691.7 | 812.9 | 3.354e+04 |
| N2 | 46 | 46 | 691.5 | 812.8 | 3.331e+04 |

Times above are observed worker wall times, not uncertainty estimates; the per-run inventory preserves scheduler allocation time, native computation time, job IDs, failures, source revisions and exact commands. [Run and timing inventory](run_inventory.csv) · [Metadata snapshot](report_provenance.json).

| Preparation / gate | Job | Recorded state | Allocation seconds | Allocated GPU seconds |
| --- | --- | --- | --- | --- |
| prepare_P0 | 17812544 | COMPLETED | 1191 | 1191 |
| prepare_D2 | 17812866 | COMPLETED | 319 | 319 |
| prepare_N2 | 17812867 | COMPLETED | 319 | 319 |
| prepare_D1 | 17812868 | COMPLETED | 333 | 333 |
| prepare_N1 | 17812869 | COMPLETED | 321 | 321 |
| G1 unchanged batch-gradient and G5 default replay after amended G2 | 17812451 | COMPLETED | 629 | 629 |

Preparation timing is kept separate from 50-batch measurement timing. Gate allocation times include process startup and verification overhead; they are not training iteration times.

Original collection seeds follow 100000 + 10000 × point_index + 100 × group_index; authorized replacement seeds start at 1000000. Preparation uses group_index 90. Point indices are P0 = 0, D1 = 1, D2 = 2, N1 = 3, N2 = 4; each run seeds its 40 environments with seed + 0…39. Exact seeds and native commands are in the inventory; no global RNG is used for the diagnostic splits.

Analysis command: `/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/analyze.py --output /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/analyses/round_2300 --partial`. Python: `3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:52:49) `; NumPy: `1.24.4`. The metadata snapshot preserves existing worker environment and GPU inventories.

**Exclusions and deviations.** 0 group records were excluded from the verified measurement set. No native-complete group is admitted unless the worker, cleanup and terminal scheduler records also passed. At most one approved independent-seed replacement is selected for a group; original failures remain stored.

Approved clarifications include the mean-free four-group effective-rank denominator; counting both P0 live-gate groups within round 1; public V1–V3 noise scales/cosines only; score uncertainty unavailable from one group; and the all-126-update drift proxy. Amendment G-1 additionally defines unresolved directions and upper-signal plug-in bounds, adds the target synthetic regime, and corrects process cleanup. [Approval record](../../../../provenance/approval.json) · [Amendment G-1](../../../../provenance/amendment_G1.json).

## Figures and complete numerical evidence

Each figure has an adjacent caption specifying units, uncertainty and limitations. The CSVs retain all approved per-point and per-variant estimates, bounds and missing-value reasons.

| Figure | PNG | PDF | Caption |
| --- | --- | --- | --- |
| v4_signal_noise_paths | [View](figures/v4_signal_noise_paths.png) | [PDF](figures/v4_signal_noise_paths.pdf) | [Caption](figures/v4_signal_noise_paths.caption.md) |
| v1_dimensionless_paths | [View](figures/v1_dimensionless_paths.png) | [PDF](figures/v1_dimensionless_paths.pdf) | [Caption](figures/v1_dimensionless_paths.caption.md) |
| v4_rho_convergence | [View](figures/v4_rho_convergence.png) | [PDF](figures/v4_rho_convergence.pdf) | [Caption](figures/v4_rho_convergence.caption.md) |
| variant_cosines_P0 | [View](figures/variant_cosines_P0.png) | [PDF](figures/variant_cosines_P0.pdf) | [Caption](figures/variant_cosines_P0.caption.md) |
| variant_cosines_D1 | [View](figures/variant_cosines_D1.png) | [PDF](figures/variant_cosines_D1.pdf) | [Caption](figures/variant_cosines_D1.caption.md) |
| variant_cosines_D2 | [View](figures/variant_cosines_D2.png) | [PDF](figures/variant_cosines_D2.pdf) | [Caption](figures/variant_cosines_D2.caption.md) |
| variant_cosines_N1 | [View](figures/variant_cosines_N1.png) | [PDF](figures/variant_cosines_N1.pdf) | [Caption](figures/variant_cosines_N1.caption.md) |
| variant_cosines_N2 | [View](figures/variant_cosines_N2.png) | [PDF](figures/variant_cosines_N2.pdf) | [Caption](figures/variant_cosines_N2.caption.md) |
| policy_cosines_V1 | [View](figures/policy_cosines_V1.png) | [PDF](figures/policy_cosines_V1.pdf) | [Caption](figures/policy_cosines_V1.caption.md) |
| policy_cosines_V4 | [View](figures/policy_cosines_V4.png) | [PDF](figures/policy_cosines_V4.pdf) | [Caption](figures/policy_cosines_V4.caption.md) |

[main results](main_results.csv) · [per point](per_point.csv) · [same point cosines](same_point_cosines.csv) · [cross point cosines](cross_point_cosines.csv) · [path alignment](path_alignment.csv) · [contrasts](contrasts.csv) · [score variance lemma](score_variance_lemma.csv) · [drift per seed](drift_per_seed.csv) · [drift summary](drift_summary.csv) · [phase1 comparison](phase1_comparison.csv) · [split convergence](split_convergence.csv)

[Table definitions](table_captions.md) · [Machine-readable results](results.json) · [Methods](../../../../methods.md) · [Analysis execution record](../analysis_status.json)

**Decision and boundary.** Review these uncurated candidates for keep, revise, or leave uncurated. No new trajectories, training updates, extrapolated stopping target or additional scientific analysis is authorized by this report. Earlier failures and excluded records remain in the run inventory.
