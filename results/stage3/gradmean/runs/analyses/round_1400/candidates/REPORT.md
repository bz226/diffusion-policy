# Addendum G: fixed-policy gradient signal and noise

**Question.** At five pretrained, DPPO and NC4 checkpoints, how large and reliable is the mean discounted-return gradient, and how does it relate to the observed policy displacements?

**Answer.** This partial analysis contains verified measurements at 5 of five checkpoints; 0 meet the prescribed V4 split-correlation threshold of 0.68. Resolved V4 point estimates have noise scale above the 80-episode training batch at 5 of 5 measured checkpoints. No measured V4 signal/noise ratio is currently classified as unresolved.

**Main limitation.** These are measurements at fixed checkpoints from one training seed per path. They do not show that gradient noise caused a training outcome. Unresolved signals, missing data and failed groups remain visible; a sample-cap closure does not establish a reliable direction. This interim report cannot support a final five-checkpoint conclusion.

Numbers in square brackets are 90% delete-one-group normal intervals (±1.645 SE), unless stated otherwise. One independent group contains 50 whole batches of 80 episodes. V4/V5 use paper ascent-gradient units; V1–V3 publish noise scales and cosines only. A ≥ or ≤ plug-in bound uses the estimated trace and the upper signal interval endpoint and has **no joint confidence guarantee**. N/A is retained when a statistic or its uncertainty is undefined.

## Q1. Is the gradient noise-dominated at 80 episodes?

| Point | Batches K | V4 B (episodes) | Expected 80-episode batch cosine |
| --- | --- | --- | --- |
| P0 | 1400 | 5.823e+04 [5.032e+04, 6.613e+04] | 0.03704 [0.03455, 0.03954] |
| D1 | 1400 | 6.892e+04 [5.116e+04, 8.669e+04] | 0.03405 [0.0297, 0.0384] |
| D2 | 1400 | 8.573e+04 [6.645e+04, 1.05e+05] | 0.03053 [0.02713, 0.03393] |
| N1 | 1400 | 9.403e+04 [7.077e+04, 1.173e+05] | 0.02916 [0.02557, 0.03274] |
| N2 | 1400 | 7.349e+04 [6.058e+04, 8.64e+04] | 0.03298 [0.03007, 0.03589] |

B above 80 means the estimated gradient-noise trace exceeds squared signal at this batch size; it is a descriptive scale comparison, not a hypothesis test. The expected cosine is the prescribed moment approximation, not a directly observed cosine. [main results](main_results.csv) includes all five estimators and their uncertainty.

## Q2. How do signal, noise and performance change along each path?

| Point / path position | Discounted J | V4 gradient norm | V4 episode trace | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| P0 | 1394 [1393, 1395] | 1947 [1812, 2082] | 2.207e+11 [2.167e+11, 2.247e+11] | 5.823e+04 [5.032e+04, 6.613e+04] |
| D1 | 1514 [1514, 1515] | 1219 [1076, 1363] | 1.025e+11 [9.664e+10, 1.083e+11] | 6.892e+04 [5.116e+04, 8.669e+04] |
| D2 | 1533 [1533, 1534] | 1127 [1001, 1252] | 1.088e+11 [1.035e+11, 1.141e+11] | 8.573e+04 [6.645e+04, 1.05e+05] |
| N1 | 1496 [1496, 1497] | 1106 [977.8, 1234] | 1.15e+11 [1.102e+11, 1.198e+11] | 9.403e+04 [7.077e+04, 1.173e+05] |
| N2 | 1517 [1517, 1518] | 1061 [960.1, 1161] | 8.265e+10 [7.745e+10, 8.785e+10] | 7.349e+04 [6.058e+04, 8.64e+04] |

Read the DPPO path as P0 → D1 → D2 and the NC4 path as P0 → N1 → N2. Unresolved norms and interval width limit descriptive ordering; marginal interval overlap is not used as a significance test. A higher return at these points does not by itself establish a general relation between policy quality and gradient norm.

## Q3. How do DPPO and NC4 compare at matched iterations?

| Saved iteration | DPPO / NC4 | Discounted J | V4 gradient norm | V4 B (episodes) |
| --- | --- | --- | --- | --- |
| 70 | D1 | 1514 [1514, 1515] | 1219 [1076, 1363] | 6.892e+04 [5.116e+04, 8.669e+04] |
| 70 | N1 | 1496 [1496, 1497] | 1106 [977.8, 1234] | 9.403e+04 [7.077e+04, 1.173e+05] |
| 139 | D2 | 1533 [1533, 1534] | 1127 [1001, 1252] | 8.573e+04 [6.645e+04, 1.05e+05] |
| 139 | N2 | 1517 [1517, 1518] | 1061 [960.1, 1161] | 7.349e+04 [6.058e+04, 8.64e+04] |

The paired rows match saved iteration and training seed 0, not independently replicated training paths. No additional significance test or derived DPPO/NC4 ratio is introduced.

## Q4. How much does the baseline choice change variance or noise scale?

| Point | No baseline / LOO trace (V5/V4) | Critic / LOO B with MC (V2/V3) |
| --- | --- | --- |
| P0 | 29.37 [28.83, 29.9] | 3.054 [2.278, 3.83] |
| D1 | 77.05 [72.55, 81.55] | 13.96 [1.228, 26.7] |
| D2 | 77.41 [73.55, 81.27] | 10.37 [0.3123, 20.42] |
| N1 | 63.32 [60.58, 66.05] | 11.44 [4.208, 18.68] |
| N2 | 89.36 [83.62, 95.1] | 6.1 [1.429, 10.77] |

A V5/V4 ratio above one gives the measured multiplicative variance reduction from LOO. V2/V3 compares noise scales, not pure variances, because their normalized estimators can have different mean signals. These contrasts delete the same groups jointly across variants; an unresolved denominator remains N/A. [contrasts](contrasts.csv).

## Q5. Are the DPPO-style and E1-style measurement estimators aligned with the discounted-return gradient?

| Point | cos(V1, V4) | cos(V3, V4) |
| --- | --- | --- |
| P0 | 0.9806 [0.9199, 1.041] | 1.004 [0.9963, 1.012] |
| D1 | 0.8902 [0.7762, 1.004] | 1.001 [0.9863, 1.015] |
| D2 | 1.022 [0.9015, 1.143] | 0.9463 [0.8977, 0.9949] |
| N1 | 0.9762 [0.8493, 1.103] | 1.008 [0.9931, 1.023] |
| N2 | 0.9439 [0.8399, 1.048] | 0.982 [0.9579, 1.006] |

V1 is the unclamped NC4/DPPO-style measurement proxy; V3 is the E1-style MC/LOO measurement variant. Both use full-batch advantage normalization, whereas training normalizes per chunk and original DPPO also retains coordinate log-probability clamps, so these are not exact training-update gradients. V4 estimates the raw discounted-return gradient. Same-batch products are excluded before forming the cosines; estimates outside [−1, 1] remain visible as estimation error, and N/A does not imply orthogonality. [same point cosines](same_point_cosines.csv).

## Q6. How much does the discounted-return gradient rotate?

| Checkpoint pair | V4 mean-gradient cosine |
| --- | --- |
| P0 / D1 | 0.403 [0.2735, 0.5326] |
| P0 / D2 | 0.4968 [0.3838, 0.6097] |
| P0 / N1 | 0.6809 [0.5323, 0.8294] |
| P0 / N2 | 0.3828 [0.2649, 0.5007] |
| D1 / D2 | 0.774 [0.6348, 0.9133] |
| D1 / N1 | 0.7383 [0.6308, 0.8458] |
| D1 / N2 | 0.7208 [0.6357, 0.8059] |
| D2 / N1 | 0.6458 [0.5062, 0.7854] |
| D2 / N2 | 0.6544 [0.528, 0.7807] |
| N1 / N2 | 0.8725 [0.7701, 0.975] |

A lower cosine indicates a larger directional change only where signals and intervals are resolved. Different policies use independent measurement groups; this is not a paired-rollout comparison. The complete V1 and V4 matrices are in [cross point cosines](cross_point_cosines.csv).

## Q7. Do checkpoint displacements follow the starting gradient?

| Segment | Distance | Start cosine | First-order ΔJ | Trapezoid ΔJ | Measured ΔJ |
| --- | --- | --- | --- | --- | --- |
| P0 → D1 | 5.054 | 0.02523 [0.02164, 0.02882] | 248.3 [216.1, 280.5] | 95.32 [75.69, 115] | 120.6 [119.6, 121.5] |
| D1 → D2 | 4.659 | 0.01286 [0.008528, 0.01719] | 73.05 [51.97, 94.12] | 6.034 [-8.925, 20.99] | 18.76 [17.91, 19.6] |
| P0 → D2 | 6.858 | 0.02107 [0.01747, 0.02466] | 281.3 [236.6, 326.1] | 77.87 [48.09, 107.7] | 139.3 [138.3, 140.3] |
| P0 → N1 | 0.6353 | 0.1178 [0.1095, 0.1261] | 145.7 [135.3, 156.1] | 95.43 [88.78, 102.1] | 102.6 [101.6, 103.6] |
| N1 → N2 | 0.5521 | 0.05405 [0.04085, 0.06726] | 33.01 [27.25, 38.77] | 17.79 [13.34, 22.23] | 21 [20.2, 21.79] |
| P0 → N2 | 0.9097 | 0.1138 [0.1058, 0.1219] | 201.6 [187.8, 215.5] | 107.5 [98.41, 116.5] | 123.6 [122.7, 124.5] |

The first-order prediction is the starting V4 gradient dotted with the checkpoint displacement; the trapezoid averages the two endpoint gradients. Numerical agreement or disagreement is shown directly without a new residual test. Curvature between endpoints and the difference between a checkpoint chord and the actual training path limit this approximation. [path alignment](path_alignment.csv).

## Q8. How loose is the empirical variance-lemma scale?

| Point | Mean score norm² | Max decision-decile mean | Score norm² p99 | Lemma / V5 episode trace |
| --- | --- | --- | --- | --- |
| P0 | 6.737e+04 | 7.344e+04 | 1.177e+05 | 9.598 |
| D1 | 6.914e+04 | 7.658e+04 | 1.305e+05 | 8.237 |
| D2 | 7.188e+04 | 7.824e+04 | 1.298e+05 | 7.97 |
| N1 | 6.415e+04 | 6.732e+04 | 1.074e+05 | 8.556 |
| N2 | 6.328e+04 | 6.988e+04 | 1.221e+05 | 8.103 |

The scale is observed r_max² × mean score norm² / (1 − 0.99)³. Its ratio to V5 trace reports empirical looseness, not a verified global bound: only 200 random decision scores from one group and an observed reward maximum are available. Group-jackknife uncertainty is N/A by the approved exception. Raw maxima, scale values and decile means are in [score variance lemma](score_variance_lemma.csv).

## Q9. Does training drift agree with the direct P0 signal measurement?

The direct P0 squared-gradient signal in paper units is **3.791e+06 [3.266e+06, 4.316e+06]**.

| Existing condition | J slope per actor update | Squared-gradient drift proxy | Proxy / P0 signal |
| --- | --- | --- | --- |
| SGD1 | 0.1795 [0.1633, 0.1956] | 3.706e+06 [3.372e+06, 4.039e+06] | 0.9775 [0.815, 1.14] |
| SGD3 | 0.452 [0.3884, 0.5156] | 3.111e+06 [2.673e+06, 3.549e+06] | 0.8206 [0.6579, 0.9834] |
| SGD10 | 0.6417 [0.1486, 1.135] | 1.325e+06 [3.068e+05, 2.343e+06] | 0.3495 [0.07653, 0.6225] |
| BATCH4 | 0.4576 [0.4455, 0.4697] | 3.15e+06 [3.066e+06, 3.233e+06] | 0.8309 [0.7127, 0.9491] |

Each seed uses OLS over all 126 pre-update training returns against completed actor updates 0–125; dividing by actor_lr/2500 forms the proxy. Intervals aggregate three independent training seeds, and the ratio combines that uncertainty with independent P0 groups. This is a path-average proxy, not a local θ0 gradient measurement; SGD10 is especially sensitive to its larger step. Agreement is descriptive and cannot validate a local drift identity. [drift per seed](drift_per_seed.csv) and [drift summary](drift_summary.csv).

## Q10. Do the old Phase-1 estimates fall inside the new intervals?

| Point / variant | Metric | Old point | New estimate / 90% interval | Old point inside? |
| --- | --- | --- | --- | --- |
| P0 / V1 | B_ep | 4.125e+04 | 3.173e+04 [2.803e+04, 3.543e+04] | No |
| P0 / V2 | B_ep | N/A | 1.182e+05 [8.657e+04, 1.499e+05] | N/A |
| P0 / V3 | B_ep | 4.326e+04 | 3.871e+04 [3.411e+04, 4.331e+04] | Yes |
| P0 / V4 | B_ep | 7.828e+04 | 5.823e+04 [5.032e+04, 6.613e+04] | No |
| P0 / V4 | u_sq_theory | 3.018e+06 | 3.791e+06 [3.266e+06, 4.316e+06] | No |
| P0 / V4 | trace_ep_theory | 2.362e+11 | 2.207e+11 [2.167e+11, 2.247e+11] | No |
| P0 / V5 | B_ep | 1.143e+05 | ≥ 7.948e+05 (plug-in; no point CI) | N/A |
| P0 / V5 | u_sq_theory | 5.666e+07 | 1.887e+06 [-4.382e+06, 8.155e+06] | No |
| P0 / V5 | trace_ep_theory | 6.476e+12 | 6.482e+12 [6.463e+12, 6.502e+12] | Yes |
| N2 / V1 | B_ep | 1.557e+04 | 1.096e+05 [6.515e+04, 1.54e+05] | No |
| N2 / V2 | B_ep | 2.997e+04 | 2.134e+05 [4.502e+04, 3.818e+05] | No |
| N2 / V3 | B_ep | 2.345e+05 | 3.498e+04 [2.979e+04, 4.018e+04] | No |
| N2 / V4 | B_ep | N/A | 7.349e+04 [6.058e+04, 8.64e+04] | N/A |
| N2 / V4 | u_sq_theory | -1.993e+06 | 1.125e+06 [9.112e+05, 1.338e+06] | No |
| N2 / V4 | trace_ep_theory | 1.488e+11 | 8.265e+10 [7.745e+10, 8.785e+10] | No |
| N2 / V5 | B_ep | 2.44e+04 | ≥ 6.94e+05 (plug-in; no point CI) | N/A |
| N2 / V5 | u_sq_theory | 3.02e+08 | 1.351e+06 [-7.94e+06, 1.064e+07] | No |
| N2 / V5 | trace_ep_theory | 7.368e+12 | 7.386e+12 [7.364e+12, 7.408e+12] | Yes |

Old intervals are not reused: within-batch dependence made their uncertainty unreliable, and critic/scaler preparation differs for V1/V2. Containment of one provisional point in a new interval is not a formal consistency test. Dimensioned comparisons are shown only for V4/V5 after the documented unit conversion. [phase1 comparison](phase1_comparison.csv).

## Measurement reliability and checkpoint inventory

| Point | Groups / batches | ρ | Mean-direction c | Moment c_U | Effective rank | Forecast batches |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 28 / 1400 | 0.4882 [0.444, 0.5324] | 0.81 [0.7848, 0.8351] | 0.8111 [0.792, 0.8303] | 419.1 [307, 531.3] | 3100 [2680, 3521] |
| D1 | 28 / 1400 | 0.4356 [0.3598, 0.5113] | 0.779 [0.7308, 0.8271] | 0.7868 [0.7477, 0.8259] | 309.2 [177.3, 441] | 3670 [2724, 4616] |
| D2 | 28 / 1400 | 0.3989 [0.344, 0.4539] | 0.7552 [0.7173, 0.7931] | 0.7526 [0.7156, 0.7896] | 259.4 [193, 325.7] | 4565 [3538, 5592] |
| N1 | 28 / 1400 | 0.3928 [0.3344, 0.4513] | 0.751 [0.7098, 0.7922] | 0.7373 [0.6953, 0.7793] | 442.6 [323, 562.3] | 5007 [3768, 6246] |
| N2 | 28 / 1400 | 0.4431 [0.3917, 0.4945] | 0.7836 [0.7516, 0.8157] | 0.7771 [0.7496, 0.8045] | 425 [341.8, 508.2] | 3913 [3226, 4601] |

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

Recorded used GPU hours: 29.24; outstanding reserved GPU hours: 0; cap: 125. These are scheduler allocation totals, including failed allocations; parallel job times must not be added to estimate campaign wall-clock duration.

Campaign record: 2026-10-11T20:55:24.992576+00:00 through 2026-10-12T01:56:58.359808+00:00, status `running`. Analysis started 2026-10-12T01:56:53.585680+00:00; elapsed wall time through report generation 59.42 seconds. [Campaign ledger](../../../campaign.json).

| Point | Collection attempts | Complete workers | Min job seconds | Max job seconds | Sum job seconds |
| --- | --- | --- | --- | --- | --- |
| P0 | 28 | 28 | 691.4 | 811.6 | 2.017e+04 |
| D1 | 28 | 28 | 691.5 | 799.2 | 2.027e+04 |
| D2 | 28 | 28 | 692.5 | 843 | 2.059e+04 |
| N1 | 28 | 28 | 691.7 | 811.7 | 2.046e+04 |
| N2 | 28 | 28 | 691.5 | 812.8 | 2.033e+04 |

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

Analysis command: `/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/analyze.py --output /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/analyses/round_1400 --partial`. Python: `3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:52:49) `; NumPy: `1.24.4`. The metadata snapshot preserves existing worker environment and GPU inventories.

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
