# Addendum G: fixed-policy gradient signal and noise

**Question.** At five pretrained, DPPO and NC4 checkpoints, how large and reliable is the mean discounted-return gradient, and how does it relate to the observed policy displacements?

**Answer.** This final analysis contains verified measurements at 5 of five checkpoints; 1 meet the prescribed V4 split-correlation threshold of 0.68. Resolved V4 point estimates have noise scale above the 80-episode training batch at 5 of 5 measured checkpoints. No measured V4 signal/noise ratio is currently classified as unresolved.

**Stopping plan.** Before any V6 collection, the user replaced adaptive continuation with a compute-driven fixed plan: all already admitted legacy jobs finish, then exactly ten new 50-batch six-variant groups are collected at each of five policies, followed by a stop. Thus V6 has 500 batches per policy; V1–V5 also retain every verified legacy group. Reaching the original 0.68 reliability threshold is descriptive, not required for completion. [V6 analysis](V6_NOTE.md)

**Main limitation.** These are measurements at fixed checkpoints from one training seed per path. They do not show that gradient noise caused a training outcome. Unresolved signals, missing data and failed groups remain visible; a sample-cap closure does not establish a reliable direction. The final scope is limited to the approved checkpoints and sample counts.

Numbers in square brackets are 90% delete-one-group normal intervals (±1.645 SE), unless stated otherwise. One independent group contains 50 whole batches of 80 episodes. V4/V5 use paper ascent-gradient units; V1–V3 publish noise scales and cosines only. A ≥ or ≤ plug-in bound uses the estimated trace and the upper signal interval endpoint and has **no joint confidence guarantee**. N/A is retained when a statistic or its uncertainty is undefined.

## Q1. Is the gradient noise-dominated at 80 episodes?

| Point | Batches K | V4 B (episodes) | Expected 80-episode batch cosine |
| --- | --- | --- | --- |
| P0 | 3700 | 5.305e+04 [4.94e+04, 5.671e+04] | 0.0388 [0.03747, 0.04014] |
| D1 | 2800 | 7.371e+04 [6.092e+04, 8.651e+04] | 0.03293 [0.03008, 0.03577] |
| D2 | 3700 | 7.959e+04 [6.984e+04, 8.935e+04] | 0.03169 [0.02975, 0.03362] |
| N1 | 2800 | 1.012e+05 [8.401e+04, 1.185e+05] | 0.0281 [0.0257, 0.03049] |
| N2 | 3550 | 8.771e+04 [7.25e+04, 1.029e+05] | 0.03019 [0.02756, 0.03281] |

B above 80 means the estimated gradient-noise trace exceeds squared signal at this batch size; it is a descriptive scale comparison, not a hypothesis test. The expected cosine is the prescribed moment approximation, not a directly observed cosine. [main results](main_results.csv) includes all five estimators and their uncertainty.

## Q2. How do signal, noise and performance change along each path?

| Point / path position | Discounted J | V4 gradient norm | V4 episode trace | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| P0 | 1393 [1393, 1394] | 2050 [1980, 2120] | 2.229e+11 [2.204e+11, 2.253e+11] | 5.305e+04 [4.94e+04, 5.671e+04] |
| D1 | 1515 [1514, 1515] | 1159 [1064, 1253] | 9.898e+10 [9.531e+10, 1.026e+11] | 7.371e+04 [6.092e+04, 8.651e+04] |
| D2 | 1533 [1533, 1534] | 1162 [1093, 1232] | 1.075e+11 [1.047e+11, 1.103e+11] | 7.959e+04 [6.984e+04, 8.935e+04] |
| N1 | 1496 [1496, 1497] | 1055 [964.6, 1145] | 1.126e+11 [1.096e+11, 1.156e+11] | 1.012e+05 [8.401e+04, 1.185e+05] |
| N2 | 1517 [1517, 1518] | 986.8 [904.6, 1069] | 8.541e+10 [8.187e+10, 8.896e+10] | 8.771e+04 [7.25e+04, 1.029e+05] |

Read the DPPO path as P0 → D1 → D2 and the NC4 path as P0 → N1 → N2. Unresolved norms and interval width limit descriptive ordering; marginal interval overlap is not used as a significance test. A higher return at these points does not by itself establish a general relation between policy quality and gradient norm.

## Q3. How do DPPO and NC4 compare at matched iterations?

| Saved iteration | DPPO / NC4 | Discounted J | V4 gradient norm | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| 70 | D1 | 1515 [1514, 1515] | 1159 [1064, 1253] | 7.371e+04 [6.092e+04, 8.651e+04] |
| 70 | N1 | 1496 [1496, 1497] | 1055 [964.6, 1145] | 1.012e+05 [8.401e+04, 1.185e+05] |
| 139 | D2 | 1533 [1533, 1534] | 1162 [1093, 1232] | 7.959e+04 [6.984e+04, 8.935e+04] |
| 139 | N2 | 1517 [1517, 1518] | 986.8 [904.6, 1069] | 8.771e+04 [7.25e+04, 1.029e+05] |

The paired rows match saved iteration and training seed 0, not independently replicated training paths. No additional significance test or derived DPPO/NC4 ratio is introduced.

## Q4. How much does the baseline choice change variance or noise scale?

| Point | No baseline / LOO trace (V5/V4) | Critic / LOO B with MC (V2/V3) |
| --- | --- | --- |
| P0 | 29.05 [28.72, 29.37] | 3.339 [2.972, 3.706] |
| D1 | 79.76 [76.75, 82.76] | 12.63 [6.307, 18.94] |
| D2 | 78.42 [76.31, 80.53] | 12.97 [8.105, 17.83] |
| N1 | 64.42 [62.65, 66.19] | 10.01 [6.142, 13.87] |
| N2 | 86.48 [82.89, 90.07] | 8.898 [4.032, 13.76] |

A V5/V4 ratio above one gives the measured multiplicative variance reduction from LOO. V2/V3 compares noise scales, not pure variances, because their normalized estimators can have different mean signals. These contrasts delete the same groups jointly across variants; an unresolved denominator remains N/A. [contrasts](contrasts.csv).

## Q5. Are the DPPO-style and E1-style measurement estimators aligned with the discounted-return gradient?

| Point | cos(V1, V4) | cos(V3, V4) |
| --- | --- | --- |
| P0 | 0.9915 [0.9766, 1.006] | 1 [0.9964, 1.004] |
| D1 | 0.9784 [0.9292, 1.028] | 1.001 [0.9942, 1.007] |
| D2 | 0.9974 [0.9402, 1.055] | 0.97 [0.9516, 0.9884] |
| N1 | 0.9836 [0.9072, 1.06] | 0.9994 [0.9902, 1.009] |
| N2 | 0.9849 [0.9343, 1.036] | 0.9908 [0.9825, 0.9991] |

V1 is the unclamped NC4/DPPO-style measurement proxy; V3 is the E1-style MC/LOO measurement variant. Both use full-batch advantage normalization, whereas training normalizes per chunk and original DPPO also retains coordinate log-probability clamps, so these are not exact training-update gradients. V4 estimates the raw discounted-return gradient. Same-batch products are excluded before forming the cosines; estimates outside [−1, 1] remain visible as estimation error, and N/A does not imply orthogonality. [same point cosines](same_point_cosines.csv).

## Q6. How much does the discounted-return gradient rotate?

| Checkpoint pair | V4 mean-gradient cosine |
| --- | --- |
| P0 / D1 | 0.4604 [0.3907, 0.5301] |
| P0 / D2 | 0.4146 [0.3648, 0.4643] |
| P0 / N1 | 0.7391 [0.6727, 0.8055] |
| P0 / N2 | 0.4642 [0.4014, 0.527] |
| D1 / D2 | 0.7532 [0.6673, 0.8392] |
| D1 / N1 | 0.721 [0.649, 0.7929] |
| D1 / N2 | 0.7619 [0.7064, 0.8174] |
| D2 / N1 | 0.6047 [0.5291, 0.6803] |
| D2 / N2 | 0.6407 [0.5669, 0.7145] |
| N1 / N2 | 0.8928 [0.829, 0.9565] |

A lower cosine indicates a larger directional change only where signals and intervals are resolved. Different policies use independent measurement groups; this is not a paired-rollout comparison. The complete V1 and V4 matrices are in [cross point cosines](cross_point_cosines.csv).

## Q7. Do checkpoint displacements follow the starting gradient?

| Segment | Distance | Start cosine | First-order ΔJ | Trapezoid ΔJ | Measured ΔJ |
| --- | --- | --- | --- | --- | --- |
| P0 → D1 | 5.054 | 0.02513 [0.02332, 0.02694] | 260.3 [240.5, 280.2] | 105.3 [92.31, 118.4] | 121.5 [120.7, 122.2] |
| D1 → D2 | 4.659 | 0.01489 [0.01159, 0.01818] | 80.36 [64.71, 96.01] | 9.644 [-1.125, 20.41] | 18.43 [17.86, 19] |
| P0 → D2 | 6.858 | 0.02216 [0.02008, 0.02424] | 311.5 [281.3, 341.7] | 97.78 [77.59, 118] | 139.9 [139.2, 140.6] |
| P0 → N1 | 0.6353 | 0.1181 [0.1136, 0.1226] | 153.7 [147.3, 160.2] | 99.16 [94.91, 103.4] | 103.1 [102.3, 103.9] |
| N1 → N2 | 0.5521 | 0.06011 [0.05048, 0.06973] | 35 [30.98, 39.02] | 18.57 [15.79, 21.35] | 20.92 [20.24, 21.59] |
| P0 → N2 | 0.9097 | 0.115 [0.1107, 0.1194] | 214.5 [206, 223] | 114.7 [109.1, 120.3] | 124 [123.3, 124.7] |

The first-order prediction is the starting V4 gradient dotted with the checkpoint displacement; the trapezoid averages the two endpoint gradients. Numerical agreement or disagreement is shown directly without a new residual test. Curvature between endpoints and the difference between a checkpoint chord and the actual training path limit this approximation. [path alignment](path_alignment.csv).

## Q8. How loose is the empirical variance-lemma scale?

| Point | Mean score norm² | Max decision-decile mean | Score norm² p99 | Lemma / V5 episode trace |
| --- | --- | --- | --- | --- |
| P0 | 6.737e+04 | 7.344e+04 | 1.177e+05 | 9.839 |
| D1 | 6.914e+04 | 7.658e+04 | 1.305e+05 | 8.237 |
| D2 | 7.188e+04 | 7.824e+04 | 1.298e+05 | 7.961 |
| N1 | 6.415e+04 | 6.732e+04 | 1.074e+05 | 8.588 |
| N2 | 6.328e+04 | 6.988e+04 | 1.221e+05 | 8.102 |

The scale is observed r_max² × mean score norm² / (1 − 0.99)³. Its ratio to V5 trace reports empirical looseness, not a verified global bound: only 200 random decision scores from one group and an observed reward maximum are available. Group-jackknife uncertainty is N/A by the approved exception. Raw maxima, scale values and decile means are in [score variance lemma](score_variance_lemma.csv).

## Q9. Does training drift agree with the direct P0 signal measurement?

The direct P0 squared-gradient signal in paper units is **4.201e+06 [3.915e+06, 4.488e+06]**.

| Existing condition | J slope per actor update | Squared-gradient drift proxy | Proxy / P0 signal |
| --- | --- | --- | --- |
| SGD1 | 0.1795 [0.1633, 0.1956] | 3.706e+06 [3.372e+06, 4.039e+06] | 0.882 [0.7824, 0.9816] |
| SGD3 | 0.452 [0.3884, 0.5156] | 3.111e+06 [2.673e+06, 3.549e+06] | 0.7405 [0.6247, 0.8563] |
| SGD10 | 0.6417 [0.1486, 1.135] | 1.325e+06 [3.068e+05, 2.343e+06] | 0.3154 [0.07208, 0.5587] |
| BATCH4 | 0.4576 [0.4455, 0.4697] | 3.15e+06 [3.066e+06, 3.233e+06] | 0.7497 [0.6948, 0.8046] |

Each seed uses OLS over all 126 pre-update training returns against completed actor updates 0–125; dividing by actor_lr/2500 forms the proxy. Intervals aggregate three independent training seeds, and the ratio combines that uncertainty with independent P0 groups. This is a path-average proxy, not a local θ0 gradient measurement; SGD10 is especially sensitive to its larger step. Agreement is descriptive and cannot validate a local drift identity. [drift per seed](drift_per_seed.csv) and [drift summary](drift_summary.csv).

## Q10. Do the old Phase-1 estimates fall inside the new intervals?

| Point / variant | Metric | Old point | New estimate / 90% interval | Old point inside? |
| --- | --- | --- | --- | --- |
| P0 / V1 | B_ep | 4.125e+04 | 3.162e+04 [2.97e+04, 3.355e+04] | No |
| P0 / V2 | B_ep | N/A | 1.198e+05 [1.064e+05, 1.332e+05] | N/A |
| P0 / V3 | B_ep | 4.326e+04 | 3.587e+04 [3.384e+04, 3.791e+04] | No |
| P0 / V4 | B_ep | 7.828e+04 | 5.305e+04 [4.94e+04, 5.671e+04] | No |
| P0 / V4 | u_sq_theory | 3.018e+06 | 4.201e+06 [3.915e+06, 4.488e+06] | No |
| P0 / V4 | trace_ep_theory | 2.362e+11 | 2.229e+11 [2.204e+11, 2.253e+11] | No |
| P0 / V5 | B_ep | 1.143e+05 | ≥ 1.201e+06 (plug-in; no point CI) | N/A |
| P0 / V5 | u_sq_theory | 5.666e+07 | 2.392e+06 [-6.055e+05, 5.389e+06] | No |
| P0 / V5 | trace_ep_theory | 6.476e+12 | 6.474e+12 [6.462e+12, 6.487e+12] | Yes |
| N2 / V1 | B_ep | 1.557e+04 | 1.479e+05 [1.096e+05, 1.862e+05] | No |
| N2 / V2 | B_ep | 2.997e+04 | 3.44e+05 [1.529e+05, 5.351e+05] | No |
| N2 / V3 | B_ep | 2.345e+05 | 3.866e+04 [3.413e+04, 4.32e+04] | No |
| N2 / V4 | B_ep | N/A | 8.771e+04 [7.25e+04, 1.029e+05] | N/A |
| N2 / V4 | u_sq_theory | -1.993e+06 | 9.738e+05 [8.113e+05, 1.136e+06] | No |
| N2 / V4 | trace_ep_theory | 1.488e+11 | 8.541e+10 [8.187e+10, 8.896e+10] | No |
| N2 / V5 | B_ep | 2.44e+04 | ≥ 3.58e+06 (plug-in; no point CI) | N/A |
| N2 / V5 | u_sq_theory | 3.02e+08 | -5.3e+05 [-3.123e+06, 2.063e+06] | No |
| N2 / V5 | trace_ep_theory | 7.368e+12 | 7.387e+12 [7.372e+12, 7.401e+12] | No |

Old intervals are not reused: within-batch dependence made their uncertainty unreliable, and critic/scaler preparation differs for V1/V2. Containment of one provisional point in a new interval is not a formal consistency test. Dimensioned comparisons are shown only for V4/V5 after the documented unit conversion. [phase1 comparison](phase1_comparison.csv).

## Measurement reliability and checkpoint inventory

| Point | Groups / batches | ρ | Mean-direction c | Moment c_U | Effective rank | Forecast batches |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 74 / 3700 | 0.7344 [0.7163, 0.7525] | 0.9203 [0.9137, 0.9268] | 0.9209 [0.916, 0.9257] | 413.3 [357.4, 469.3] | 2825 [2630, 3020] |
| D1 | 56 / 2800 | 0.6039 [0.554, 0.6539] | 0.8678 [0.8452, 0.8904] | 0.8674 [0.8486, 0.8863] | 307.8 [239, 376.6] | 3925 [3244, 4607] |
| D2 | 74 / 3700 | 0.6493 [0.6158, 0.6829] | 0.8873 [0.8734, 0.9013] | 0.8877 [0.8761, 0.8994] | 249.7 [212.2, 287.3] | 4238 [3719, 4758] |
| N1 | 56 / 2800 | 0.5313 [0.4843, 0.5784] | 0.833 [0.8088, 0.8573] | 0.8299 [0.8077, 0.8521] | 407.8 [316.2, 499.3] | 5391 [4473, 6309] |
| N2 | 71 / 3550 | 0.622 [0.579, 0.665] | 0.8757 [0.857, 0.8945] | 0.8741 [0.856, 0.8922] | 410.7 [355, 466.4] | 4671 [3861, 5481] |

ρ averages 20 balanced group half-splits; c describes the full measurement mean, unlike the 80-episode cosine in Q1. Effective rank uses the approved mean-free four-group difference statistic. The user replaced adaptive stopping with exactly 500 new V6 batches per point; forecasts and the original 0.68 threshold are descriptive and do not authorize more collection. Repeated convergence prefixes are dependent.

| Point | Original actor checkpoint | Distance from P0 | Frozen-preparation batches |
| --- | --- | --- | --- |
| P0 | [state_3000.pt](../../../../../../../repro_halfcheetah/dppo/log/gym-pretrain/halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-04-42/checkpoint/state_3000.pt) | 0 | 20 |
| D1 | [state_70.pt](../../../../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_20-01-32_0/checkpoint/state_70.pt) | 5.054 | 5 |
| D2 | [state_139.pt](../../../../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_20-01-32_0/checkpoint/state_139.pt) | 6.858 | 5 |
| N1 | [state_70.pt](../../../../../../../stage2_noclip/runs/NC4_lr1e-4_seed0/native/checkpoint/state_70.pt) | 0.6353 | 5 |
| N2 | [state_139.pt](../../../../../../../stage2_noclip/runs/NC4_lr1e-4_seed0/native/checkpoint/state_139.pt) | 0.9097 | 5 |

Distances use the registered actor parameter order, float64 differences of saved float32 values, and no coordinate scaling. P0 is the released pretrained EMA policy; the four RL checkpoints are from training seed 0. [Checkpoint mapping](../../../../../provenance/checkpoints.json) · [Distance and parameter-order record](../internal/checkpoint_metadata.json) · [per point](per_point.csv) (discounted and undiscounted returns, SEs, saturation, out-of-range actions and explained variance).

## Verification gates and retained failures

G1 fixed-batch gradient comparison: **passed**. [Offline gate record](../../../../gate_offline_G1/gate_summary.json).

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

Amended G2 uses 200 predeclared replicas in each regime, P = 20,000, K = 2,000, groups of 50, seed 8675309 and the same generator streams. Formal interval-coverage criteria apply to signal and trace only. [G2-G1 record](../../../../gate_G2_G1/gate_summary.json).

| Regime | Status | Signal mean / truth; MC SE | Trace mean / truth; MC SE | Signal coverage | Trace coverage | Rank mean / truth |
| --- | --- | --- | --- | --- | --- | --- |
| L | Passed | 0.997 / 1; SE 0.03356 | 4000 / 4000; SE 1.161 | 0.965 | 0.9 | 61.62 / 61.69 |
| T | Passed | 1.003 / 1; SE 0.009053 | 470 / 470; SE 0.1364 | 0.92 | 0.9 | 61.62 / 61.69 |

Low-signal L has 1/200 nonpositive split correlations; its conditional signed cosine error is -0.02256 over 199 defined replicas and is descriptive. Target T has mean absolute cosine error 0.01493, with 200/200 replicas within absolute error 0.06 (required fraction ≥ 0.95). Pool shutdown success: True; surviving child PIDs: []; expected process exit: 0.

G2 scheduler job 17812422: state COMPLETED, exit 0:0, elapsed 44 seconds, 0 GPUs. [Terminal scheduler record](../../../../gate_G2_G1/scheduler_completion.json).

G3 independent collection/cleanup: passed; seeds [100000, 100100]; different initial observations True and returns True; process exit codes [0, 0, 0]; surviving children []. G4 frozen actor: passed. [Live gate record](../../../../gate_live_G1/gate_summary.json).

Automatic pilot timing review: **complete**. [Pilot forecast report](../../../../pilot_review/pilot_review.md). The historical pilot forecast preceded the user’s fixed V6 compute stop.

The original G2 **failed** because one of 200 low-signal replicas had undefined direction reliability. Its saved summary and subsequent Python 3.8 cleanup traceback are preserved; the user approved Amendment G-1 to treat this case as unresolved and add the target regime. This is a protocol amendment after a retained failure, not an unchanged-gate success. [Original failed gate](../../../../gate_G2/gate_summary.json).

## Execution, resources and deviations

Analysis parent commit: `c7f6e58cf653d01fa4325bf8300569eafc404582`. Training/collection source revisions are recorded separately per run; a later analysis commit does not replace the source snapshot used for collection. [Revision history](../../../../../provenance/revisions.json).

Collection uses one A6000 GPU, 40 CPUs and 64 GiB per job, at most nine concurrent GPU jobs, and a 125 GPU-hour total cap. Each preparation or 50-batch collection job is capped at one hour. CPU analyses use four CPUs, 64 GiB and a 30-minute allocation; gate allocations are recorded separately.

Recorded used GPU hours: 68.05; outstanding reserved GPU hours: 0; cap: 125. These are scheduler allocation totals, including failed allocations; parallel job times must not be added to estimate campaign wall-clock duration.

Campaign record: 2026-10-11T20:55:24.992576+00:00 through 2026-10-12T10:51:22.450763+00:00, status `complete_with_analysis_failures`. Analysis started 2026-10-12T15:20:02.097712+00:00; elapsed wall time through report generation 114.4 seconds. [Campaign ledger](../../../../campaign.json).

| Point | Collection attempts | Complete workers | Min job seconds | Max job seconds | Sum job seconds |
| --- | --- | --- | --- | --- | --- |
| P0 | 74 | 74 | 691.4 | 811.6 | 5.35e+04 |
| D1 | 56 | 56 | 691.5 | 814.4 | 4.049e+04 |
| D2 | 74 | 74 | 691.5 | 843 | 5.423e+04 |
| N1 | 56 | 56 | 691.7 | 829 | 4.093e+04 |
| N2 | 73 | 71 | 691.1 | 846 | 5.191e+04 |

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

Analysis command: `/soalnas/share/data/zbao7/diffusion_policy/.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/analyze.py --root /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean --output /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/analyses/reportfix_20261012/final_v6_fixed`. Python: `3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:52:49) `; NumPy: `1.24.4`. The metadata snapshot preserves existing worker environment and GPU inventories.

**Exclusions and deviations.** 0 group records were excluded from the verified measurement set. Native, worker, cleanup and terminal scheduler success are normally required. The explicit administrative recovery admits seven fully completed measurements whose wrappers failed only at the obsolete branch-name check; their original FAILED scheduler states, failed manifests, zero native exit codes and successful cleanup records remain preserved. Other failed jobs remain excluded. At most one approved independent-seed replacement is selected for a group; original failures remain stored.

Approved clarifications include the mean-free four-group effective-rank denominator; counting both P0 live-gate groups within round 1; public V1–V3 noise scales/cosines only; score uncertainty unavailable from one group; and the all-126-update drift proxy. Amendment G-1 additionally defines unresolved directions and upper-signal plug-in bounds, adds the target synthetic regime, and corrects process cleanup. [Approval record](../../../../../provenance/approval.json) · [Amendment G-1](../../../../../provenance/amendment_G1.json).

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

[Table definitions](table_captions.md) · [Machine-readable results](results.json) · [Methods](../../../../../methods.md) · [Analysis execution record](../analysis_status.json)

**Decision and boundary.** Review these uncurated candidates for keep, revise, or leave uncurated. No new trajectories, training updates, extrapolated stopping target or additional scientific analysis is authorized by this report. Earlier failures and excluded records remain in the run inventory.
