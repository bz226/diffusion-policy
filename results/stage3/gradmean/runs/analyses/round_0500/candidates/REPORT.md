# Addendum G: fixed-policy gradient signal and noise

**Question.** At five pretrained, DPPO and NC4 checkpoints, how large and reliable is the mean discounted-return gradient, and how does it relate to the observed policy displacements?

**Answer.** This partial analysis contains verified measurements at 5 of five checkpoints; 0 meet the prescribed V4 split-correlation threshold of 0.68. Resolved V4 point estimates have noise scale above the 80-episode training batch at 5 of 5 measured checkpoints. No measured V4 signal/noise ratio is currently classified as unresolved.

**Main limitation.** These are measurements at fixed checkpoints from one training seed per path. They do not show that gradient noise caused a training outcome. Unresolved signals, missing data and failed groups remain visible; a sample-cap closure does not establish a reliable direction. This interim report cannot support a final five-checkpoint conclusion.

Numbers in square brackets are 90% delete-one-group normal intervals (±1.645 SE), unless stated otherwise. One independent group contains 50 whole batches of 80 episodes. V4/V5 use paper ascent-gradient units; V1–V3 publish noise scales and cosines only. A ≥ or ≤ plug-in bound uses the estimated trace and the upper signal interval endpoint and has **no joint confidence guarantee**. N/A is retained when a statistic or its uncertainty is undefined.

## Q1. Is the gradient noise-dominated at 80 episodes?

| Point | Batches K | V4 B (episodes) | Expected 80-episode batch cosine |
| --- | --- | --- | --- |
| P0 | 500 | 5.856e+04 [4.199e+04, 7.513e+04] | 0.03694 [0.03168, 0.04219] |
| D1 | 500 | 5.496e+04 [1.885e+04, 9.108e+04] | 0.03812 [0.0259, 0.05035] |
| D2 | 500 | 7.326e+04 [3.377e+04, 1.128e+05] | 0.03303 [0.02378, 0.04227] |
| N1 | 500 | 8.247e+04 [4.952e+04, 1.154e+05] | 0.03113 [0.02464, 0.03762] |
| N2 | 500 | 8.95e+04 [5.744e+04, 1.216e+05] | 0.02988 [0.02435, 0.03542] |

B above 80 means the estimated gradient-noise trace exceeds squared signal at this batch size; it is a descriptive scale comparison, not a hypothesis test. The expected cosine is the prescribed moment approximation, not a directly observed cosine. [main results](main_results.csv) includes all five estimators and their uncertainty.

## Q2. How do signal, noise and performance change along each path?

| Point / path position | Discounted J | V4 gradient norm | V4 episode trace | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| P0 | 1394 [1393, 1394] | 1967 [1690, 2244] | 2.267e+11 [2.195e+11, 2.339e+11] | 5.856e+04 [4.199e+04, 7.513e+04] |
| D1 | 1515 [1514, 1516] | 1307 [889.6, 1724] | 9.389e+10 [8.874e+10, 9.903e+10] | 5.496e+04 [1.885e+04, 9.108e+04] |
| D2 | 1534 [1533, 1534] | 1178 [815.4, 1540] | 1.016e+11 [9.179e+10, 1.114e+11] | 7.326e+04 [3.377e+04, 1.128e+05] |
| N1 | 1496 [1496, 1497] | 1192 [927.5, 1457] | 1.173e+11 [1.081e+11, 1.264e+11] | 8.247e+04 [4.952e+04, 1.154e+05] |
| N2 | 1517 [1516, 1518] | 958.5 [774.7, 1142] | 8.223e+10 [7.644e+10, 8.802e+10] | 8.95e+04 [5.744e+04, 1.216e+05] |

Read the DPPO path as P0 → D1 → D2 and the NC4 path as P0 → N1 → N2. Unresolved norms and interval width limit descriptive ordering; marginal interval overlap is not used as a significance test. A higher return at these points does not by itself establish a general relation between policy quality and gradient norm.

## Q3. How do DPPO and NC4 compare at matched iterations?

| Saved iteration | DPPO / NC4 | Discounted J | V4 gradient norm | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| 70 | D1 | 1515 [1514, 1516] | 1307 [889.6, 1724] | 5.496e+04 [1.885e+04, 9.108e+04] |
| 70 | N1 | 1496 [1496, 1497] | 1192 [927.5, 1457] | 8.247e+04 [4.952e+04, 1.154e+05] |
| 139 | D2 | 1534 [1533, 1534] | 1178 [815.4, 1540] | 7.326e+04 [3.377e+04, 1.128e+05] |
| 139 | N2 | 1517 [1516, 1518] | 958.5 [774.7, 1142] | 8.95e+04 [5.744e+04, 1.216e+05] |

The paired rows match saved iteration and training seed 0, not independently replicated training paths. No additional significance test or derived DPPO/NC4 ratio is introduced.

## Q4. How much does the baseline choice change variance or noise scale?

| Point | No baseline / LOO trace (V5/V4) | Critic / LOO B with MC (V2/V3) |
| --- | --- | --- |
| P0 | 28.53 [27.65, 29.42] | 2.943 [1.149, 4.737] |
| D1 | 84.75 [80.01, 89.5] | N/A |
| D2 | 83.49 [75.16, 91.83] | N/A |
| N1 | 61.88 [56.75, 67] | N/A |
| N2 | 89.92 [83.47, 96.37] | N/A |

A V5/V4 ratio above one gives the measured multiplicative variance reduction from LOO. V2/V3 compares noise scales, not pure variances, because their normalized estimators can have different mean signals. These contrasts delete the same groups jointly across variants; an unresolved denominator remains N/A. [contrasts](contrasts.csv).

## Q5. Are the DPPO-style and E1-style measurement estimators aligned with the discounted-return gradient?

| Point | cos(V1, V4) | cos(V3, V4) |
| --- | --- | --- |
| P0 | 0.8892 [0.7328, 1.046] | 1 [0.9659, 1.035] |
| D1 | 0.9484 [0.6686, 1.228] | 1.007 [0.9745, 1.039] |
| D2 | 0.9942 [0.6565, 1.332] | 0.9426 [0.867, 1.018] |
| N1 | 0.8125 [0.3946, 1.23] | 1.011 [0.9631, 1.059] |
| N2 | 0.8146 [0.4589, 1.17] | 0.9956 [0.9202, 1.071] |

V1 is the unclamped NC4/DPPO-style measurement proxy; V3 is the E1-style MC/LOO measurement variant. Both use full-batch advantage normalization, whereas training normalizes per chunk and original DPPO also retains coordinate log-probability clamps, so these are not exact training-update gradients. V4 estimates the raw discounted-return gradient. Same-batch products are excluded before forming the cosines; estimates outside [−1, 1] remain visible as estimation error, and N/A does not imply orthogonality. [same point cosines](same_point_cosines.csv).

## Q6. How much does the discounted-return gradient rotate?

| Checkpoint pair | V4 mean-gradient cosine |
| --- | --- |
| P0 / D1 | 0.09001 [-0.128, 0.308] |
| P0 / D2 | 0.4143 [0.1953, 0.6333] |
| P0 / N1 | 0.5566 [0.296, 0.8173] |
| P0 / N2 | 0.3194 [0.05372, 0.5852] |
| D1 / D2 | 0.9374 [0.7114, 1.163] |
| D1 / N1 | 0.7025 [0.4328, 0.9722] |
| D1 / N2 | 0.7278 [0.4788, 0.9767] |
| D2 / N1 | 0.5781 [0.3122, 0.844] |
| D2 / N2 | 0.6812 [0.4188, 0.9437] |
| N1 / N2 | 0.7138 [0.5145, 0.913] |

A lower cosine indicates a larger directional change only where signals and intervals are resolved. Different policies use independent measurement groups; this is not a paired-rollout comparison. The complete V1 and V4 matrices are in [cross point cosines](cross_point_cosines.csv).

## Q7. Do checkpoint displacements follow the starting gradient?

| Segment | Distance | Start cosine | First-order ΔJ | Trapezoid ΔJ | Measured ΔJ |
| --- | --- | --- | --- | --- | --- |
| P0 → D1 | 5.054 | 0.02263 [0.01476, 0.03051] | 225.1 [163.3, 286.8] | 88.29 [54.49, 122.1] | 121.3 [120.1, 122.5] |
| D1 → D2 | 4.659 | 0.01349 [0.005405, 0.02158] | 82.16 [46.07, 118.3] | 8.329 [-15.86, 32.52] | 18.79 [17.63, 19.94] |
| P0 → D2 | 6.858 | 0.01865 [0.01075, 0.02656] | 251.7 [156, 347.4] | 55.96 [-2.817, 114.7] | 140 [139, 141.1] |
| P0 → N1 | 0.6353 | 0.1116 [0.09184, 0.1314] | 139.5 [120.2, 158.8] | 99.11 [86.95, 111.3] | 102.8 [101.6, 104] |
| N1 → N2 | 0.5521 | 0.06005 [0.04631, 0.07379] | 39.53 [33.13, 45.93] | 17.86 [11.34, 24.38] | 20.65 [19.48, 21.83] |
| P0 → N2 | 0.9097 | 0.1105 [0.09076, 0.1303] | 197.8 [173.6, 222.1] | 100.3 [85.09, 115.5] | 123.5 [122.4, 124.6] |

The first-order prediction is the starting V4 gradient dotted with the checkpoint displacement; the trapezoid averages the two endpoint gradients. Numerical agreement or disagreement is shown directly without a new residual test. Curvature between endpoints and the difference between a checkpoint chord and the actual training path limit this approximation. [path alignment](path_alignment.csv).

## Q8. How loose is the empirical variance-lemma scale?

| Point | Mean score norm² | Max decision-decile mean | Score norm² p99 | Lemma / V5 episode trace |
| --- | --- | --- | --- | --- |
| P0 | 6.737e+04 | 7.344e+04 | 1.177e+05 | 9.301 |
| D1 | 6.914e+04 | 7.658e+04 | 1.305e+05 | 7.866 |
| D2 | 7.188e+04 | 7.824e+04 | 1.298e+05 | 7.895 |
| N1 | 6.415e+04 | 6.732e+04 | 1.074e+05 | 8.264 |
| N2 | 6.328e+04 | 6.988e+04 | 1.221e+05 | 7.97 |

The scale is observed r_max² × mean score norm² / (1 − 0.99)³. Its ratio to V5 trace reports empirical looseness, not a verified global bound: only 200 random decision scores from one group and an observed reward maximum are available. Group-jackknife uncertainty is N/A by the approved exception. Raw maxima, scale values and decile means are in [score variance lemma](score_variance_lemma.csv).

## Q9. Does training drift agree with the direct P0 signal measurement?

The direct P0 squared-gradient signal in paper units is **3.871e+06 [2.774e+06, 4.967e+06]**.

| Existing condition | J slope per actor update | Squared-gradient drift proxy | Proxy / P0 signal |
| --- | --- | --- | --- |
| SGD1 | 0.1795 [0.1633, 0.1956] | 3.706e+06 [3.372e+06, 4.039e+06] | 0.9573 [0.6775, 1.237] |
| SGD3 | 0.452 [0.3884, 0.5156] | 3.111e+06 [2.673e+06, 3.549e+06] | 0.8037 [0.5532, 1.054] |
| SGD10 | 0.6417 [0.1486, 1.135] | 1.325e+06 [3.068e+05, 2.343e+06] | 0.3423 [0.06257, 0.622] |
| BATCH4 | 0.4576 [0.4455, 0.4697] | 3.15e+06 [3.066e+06, 3.233e+06] | 0.8137 [0.5864, 1.041] |

Each seed uses OLS over all 126 pre-update training returns against completed actor updates 0–125; dividing by actor_lr/2500 forms the proxy. Intervals aggregate three independent training seeds, and the ratio combines that uncertainty with independent P0 groups. This is a path-average proxy, not a local θ0 gradient measurement; SGD10 is especially sensitive to its larger step. Agreement is descriptive and cannot validate a local drift identity. [drift per seed](drift_per_seed.csv) and [drift summary](drift_summary.csv).

## Q10. Do the old Phase-1 estimates fall inside the new intervals?

| Point / variant | Metric | Old point | New estimate / 90% interval | Old point inside? |
| --- | --- | --- | --- | --- |
| P0 / V1 | B_ep | 4.125e+04 | 3.704e+04 [2.898e+04, 4.51e+04] | Yes |
| P0 / V2 | B_ep | N/A | 1.402e+05 [6.018e+04, 2.203e+05] | N/A |
| P0 / V3 | B_ep | 4.326e+04 | 4.765e+04 [3.645e+04, 5.886e+04] | Yes |
| P0 / V4 | B_ep | 7.828e+04 | 5.856e+04 [4.199e+04, 7.513e+04] | No |
| P0 / V4 | u_sq_theory | 3.018e+06 | 3.871e+06 [2.774e+06, 4.967e+06] | Yes |
| P0 / V4 | trace_ep_theory | 2.362e+11 | 2.267e+11 [2.195e+11, 2.339e+11] | No |
| P0 / V5 | B_ep | 1.143e+05 | ≥ 1.478e+05 (plug-in; no point CI) | N/A |
| P0 / V5 | u_sq_theory | 5.666e+07 | 7.938e+06 [-2.789e+07, 4.376e+07] | No |
| P0 / V5 | trace_ep_theory | 6.476e+12 | 6.468e+12 [6.447e+12, 6.489e+12] | Yes |
| N2 / V1 | B_ep | 1.557e+04 | ≥ 3.871e+04 (plug-in; no point CI) | N/A |
| N2 / V2 | B_ep | 2.997e+04 | ≥ 6.187e+04 (plug-in; no point CI) | N/A |
| N2 / V3 | B_ep | 2.345e+05 | 4.571e+04 [3.453e+04, 5.689e+04] | No |
| N2 / V4 | B_ep | N/A | 8.95e+04 [5.744e+04, 1.216e+05] | N/A |
| N2 / V4 | u_sq_theory | -1.993e+06 | 9.187e+05 [5.624e+05, 1.275e+06] | No |
| N2 / V4 | trace_ep_theory | 1.488e+11 | 8.223e+10 [7.644e+10, 8.802e+10] | No |
| N2 / V5 | B_ep | 2.44e+04 | ≥ 2.053e+05 (plug-in; no point CI) | N/A |
| N2 / V5 | u_sq_theory | 3.02e+08 | 7.145e+06 [-2.172e+07, 3.601e+07] | No |
| N2 / V5 | trace_ep_theory | 7.368e+12 | 7.394e+12 [7.357e+12, 7.43e+12] | Yes |

Old intervals are not reused: within-batch dependence made their uncertainty unreliable, and critic/scaler preparation differs for V1/V2. Containment of one provisional point in a new interval is not a formal consistency test. Dimensioned comparisons are shown only for V4/V5 after the documented unit conversion. [phase1 comparison](phase1_comparison.csv).

## Measurement reliability and checkpoint inventory

| Point | Groups / batches | ρ | Mean-direction c | Moment c_U | Effective rank | Forecast batches |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 10 / 500 | 0.2446 [0.1891, 0.3002] | 0.627 [0.5673, 0.6866] | 0.6371 [0.5828, 0.6913] | 460 [324.1, 595.9] | 3118 [2236, 4001] |
| D1 | 10 / 500 | 0.2777 [0.1707, 0.3847] | 0.6593 [0.5522, 0.7664] | 0.649 [0.5266, 0.7714] | 296.9 [-3.354, 597.1] | 2927 [1004, 4850] |
| D2 | 10 / 500 | 0.1933 [0.0785, 0.3081] | 0.5692 [0.4225, 0.7158] | 0.5943 [0.4884, 0.7002] | 143.6 [34.52, 252.8] | 3901 [1798, 6004] |
| N1 | 10 / 500 | 0.1773 [0.09597, 0.2587] | 0.5489 [0.4358, 0.6619] | 0.5715 [0.4926, 0.6503] | 420.4 [33.66, 807.1] | 4392 [2637, 6147] |
| N2 | 10 / 500 | 0.205 [0.1376, 0.2723] | 0.5833 [0.5004, 0.6662] | 0.5558 [0.4858, 0.6257] | 579 [201.8, 956.2] | 4766 [3059, 6473] |

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

Recorded used GPU hours: 10.97; outstanding reserved GPU hours: 0; cap: 125. These are scheduler allocation totals, including failed allocations; parallel job times must not be added to estimate campaign wall-clock duration.

Campaign record: 2026-10-11T20:55:24.992576+00:00 through 2026-10-11T23:06:33.906030+00:00, status `running`. Analysis started 2026-10-11T23:06:52.149833+00:00; elapsed wall time through report generation 28.05 seconds. [Campaign ledger](../../../campaign.json).

| Point | Collection attempts | Complete workers | Min job seconds | Max job seconds | Sum job seconds |
| --- | --- | --- | --- | --- | --- |
| P0 | 10 | 10 | 692.5 | 811.6 | 7194 |
| D1 | 10 | 10 | 691.5 | 797.4 | 7178 |
| D2 | 10 | 10 | 692.5 | 843 | 7344 |
| N1 | 10 | 10 | 692.2 | 811.7 | 7276 |
| N2 | 10 | 10 | 692.6 | 796.6 | 7253 |

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

Analysis command: `/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/analyze.py --output /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/analyses/round_0500 --partial`. Python: `3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:52:49) `; NumPy: `1.24.4`. The metadata snapshot preserves existing worker environment and GPU inventories.

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
