# Checkpoint measurements: three seed-0 paths

All main estimates use the same 500 V6-enabled batches at each completed point (40,000 episodes, ten independent jobs). Brackets are 90% delete-one-group jackknife intervals with Student-t quantiles; B jointly recomputes its numerator and denominator. The shared gradient norm and every main B denominator use V4 on those same batches. Unresolved B is a plug-in lower bound from the upper signal interval, not a two-sided interval; an asterisk flags other unresolved intervals. Units: gradients in paper units, trace in squared paper units, B in episodes, and returns in raw reward units.

| Point | Method | Checkpoint | Updates | Estimator | K | m | Groups | J_disc | Return | ‖u‖ | Trace/episode | B | B/80 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | shared pretrained policy | state_3000.pt | 0 | V4 | 500 | 40000 | 10 | 1.39e+03 [1.39e+03, 1.39e+03] | 3.91e+03 [3.91e+03, 3.92e+03] | 2.12e+03 [1.91e+03, 2.33e+03] | 2.24e+11 [2.16e+11, 2.31e+11] | 4.99e+04 [4.02e+04, 5.97e+04] | 624 [502, 747] |
| P0 | shared pretrained policy | state_3000.pt | 0 | V5 | 500 | 40000 | 10 | 1.39e+03 [1.39e+03, 1.39e+03] | 3.91e+03 [3.91e+03, 3.92e+03] | 2.12e+03 [1.91e+03, 2.33e+03] | 6.47e+12 [6.42e+12, 6.52e+12] | 1.44e+06 [1.15e+06, 1.74e+06] | 1.81e+04 [1.44e+04, 2.17e+04] |
| P0 | shared pretrained policy | state_3000.pt | 0 | V6 | 500 | 40000 | 10 | 1.39e+03 [1.39e+03, 1.39e+03] | 3.91e+03 [3.91e+03, 3.92e+03] | 2.12e+03 [1.91e+03, 2.33e+03] | 3.88e+11 [3.84e+11, 3.91e+11] | 8.66e+04 [6.92e+04, 1.04e+05] | 1.08e+03 [865, 1.3e+03] |
| D1 | DPPO | state_70.pt | 63 | V4 | 500 | 40000 | 10 | 1.51e+03 [1.51e+03, 1.52e+03] | 4.32e+03 [4.32e+03, 4.33e+03] | 1.24e+03 [853, 1.62e+03] | 9.8e+10 [8.84e+10, 1.08e+11] | 6.41e+04 [2.24e+04, 1.06e+05] | 801 [281, 1.32e+03] |
| D1 | DPPO | state_70.pt | 63 | V5 | 500 | 40000 | 10 | 1.51e+03 [1.51e+03, 1.52e+03] | 4.32e+03 [4.32e+03, 4.33e+03] | 1.24e+03 [853, 1.62e+03] | 7.87e+12 [7.82e+12, 7.93e+12] | 5.15e+06 [1.76e+06, 8.53e+06] | 6.43e+04 [2.2e+04, 1.07e+05] |
| D1 | DPPO | state_70.pt | 63 | V6 | 500 | 40000 | 10 | 1.51e+03 [1.51e+03, 1.52e+03] | 4.32e+03 [4.32e+03, 4.33e+03] | 1.24e+03 [853, 1.62e+03] | 3.65e+11 [3.61e+11, 3.69e+11] | 2.39e+05 [8.21e+04, 3.95e+05] | 2.98e+03 [1.03e+03, 4.94e+03] |
| D2 | DPPO | state_139.pt | 126 | V4 | 500 | 40000 | 10 | 1.53e+03 [1.53e+03, 1.53e+03] | 4.39e+03 [4.38e+03, 4.39e+03] | 1.17e+03 [930, 1.41e+03] | 1.13e+11 [1.04e+11, 1.22e+11] | 8.26e+04 [4.38e+04, 1.21e+05] | 1.03e+03 [548, 1.52e+03] |
| D2 | DPPO | state_139.pt | 126 | V5 | 500 | 40000 | 10 | 1.53e+03 [1.53e+03, 1.53e+03] | 4.39e+03 [4.38e+03, 4.39e+03] | 1.17e+03 [930, 1.41e+03] | 8.41e+12 [8.35e+12, 8.47e+12] | 6.16e+06 [3.59e+06, 8.73e+06] | 7.7e+04 [4.49e+04, 1.09e+05] |
| D2 | DPPO | state_139.pt | 126 | V6 | 500 | 40000 | 10 | 1.53e+03 [1.53e+03, 1.53e+03] | 4.39e+03 [4.38e+03, 4.39e+03] | 1.17e+03 [930, 1.41e+03] | 4.26e+11 [4.23e+11, 4.3e+11] | 3.12e+05 [1.83e+05, 4.42e+05] | 3.9e+03 [2.29e+03, 5.52e+03] |
| N1 | without_PPO | state_70.pt | 63 | V4 | 500 | 40000 | 10 | 1.5e+03 [1.5e+03, 1.5e+03] | 4.26e+03 [4.25e+03, 4.27e+03] | 889 [442, 1.34e+03] | 1.1e+11 [1.03e+11, 1.17e+11] | 1.4e+05 [-2.53e+04, 3.04e+05] | 1.74e+03 [-317, 3.8e+03] |
| N1 | without_PPO | state_70.pt | 63 | V5 | 500 | 40000 | 10 | 1.5e+03 [1.5e+03, 1.5e+03] | 4.26e+03 [4.25e+03, 4.27e+03] | 889 [442, 1.34e+03] | 7.24e+12 [7.2e+12, 7.28e+12] | 9.17e+06 [-2.03e+06, 2.04e+07] | 1.15e+05 [-2.53e+04, 2.54e+05] |
| N1 | without_PPO | state_70.pt | 63 | V6 | 500 | 40000 | 10 | 1.5e+03 [1.5e+03, 1.5e+03] | 4.26e+03 [4.25e+03, 4.27e+03] | 889 [442, 1.34e+03] | 3.78e+11 [3.74e+11, 3.82e+11] | 4.79e+05 [-1.03e+05, 1.06e+06] | 5.99e+03 [-1.29e+03, 1.33e+04] |
| N2 | without_PPO | state_139.pt | 126 | V4 | 500 | 40000 | 10 | 1.52e+03 [1.52e+03, 1.52e+03] | 4.33e+03 [4.33e+03, 4.34e+03] | 842 [461, 1.22e+03] | 8.48e+10 [7.6e+10, 9.36e+10] | 1.2e+05 [1.21e+04, 2.27e+05] | 1.49e+03 [152, 2.84e+03] |
| N2 | without_PPO | state_139.pt | 126 | V5 | 500 | 40000 | 10 | 1.52e+03 [1.52e+03, 1.52e+03] | 4.33e+03 [4.33e+03, 4.34e+03] | 842 [461, 1.22e+03] | 7.36e+12 [7.32e+12, 7.41e+12] | 1.04e+07 [1.28e+06, 1.95e+07] | 1.3e+05 [1.59e+04, 2.43e+05] |
| N2 | without_PPO | state_139.pt | 126 | V6 | 500 | 40000 | 10 | 1.52e+03 [1.52e+03, 1.52e+03] | 4.33e+03 [4.33e+03, 4.34e+03] | 842 [461, 1.22e+03] | 3.77e+11 [3.74e+11, 3.8e+11] | 5.31e+05 [6.39e+04, 9.99e+05] | 6.64e+03 [799, 1.25e+04] |
| C1 | without_PPO_and_GAE | state_70.pt | 63 | V4 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| C1 | without_PPO_and_GAE | state_70.pt | 63 | V5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| C1 | without_PPO_and_GAE | state_70.pt | 63 | V6 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| C2 | without_PPO_and_GAE | state_139.pt | 126 | V4 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| C2 | without_PPO_and_GAE | state_139.pt | 126 | V5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| C2 | without_PPO_and_GAE | state_139.pt | 126 | V6 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |

Same-cohort baseline comparisons and the trained estimator's B (relative to its own mean). Both pretrained V1 and V2 values are retained because they cannot be pooled.

| Point | Critic/LOO variance | No-baseline/critic variance | Trained estimator | B | cos(trained,V4) |
|---|---:|---:|---|---:|---:|
| P0 | 1.73 [1.68, 1.79] | 16.7 [16.5, 16.8] | V1 | 2.75e+04 [2.21e+04, 3.29e+04] | 1 [0.882, 1.12] |
| P0 | 1.73 [1.68, 1.79] | 16.7 [16.5, 16.8] | V2 | 1.04e+05 [4.68e+04, 1.62e+05] | 1.1 [0.843, 1.35] |
| D1 | 3.72 [3.39, 4.06] | 21.6 [21.3, 21.8] | V1 | 1.02e+05 [2.72e+04, 1.77e+05] | 0.913 [0.617, 1.21] |
| D2 | 3.78 [3.47, 4.09] | 19.7 [19.6, 19.9] | V1 | 1.09e+05 [7.85e+03, 2.09e+05] | 0.921 [0.497, 1.35] |
| N1 | 3.43 [3.23, 3.63] | 19.1 [19, 19.3] | V1 | ≥1.15e+05 (signal unresolved) | 1.76 [-0.117, 3.64] |
| N2 | 4.44 [4.01, 4.88] | 19.5 [19.4, 19.7] | V1 | ≥1.25e+05 (signal unresolved) | 0.884 [0.273, 1.5] |

**Reference only: all available V4 batches.** These values never enter the main figures or matched-cohort B denominators.

| Point | K | m | Groups | ‖u‖ | B |
|---|---:|---:|---:|---:|---:|
| P0 | 3700 | 296000 | 74 | 2.05e+03 [1.98e+03, 2.12e+03] | 5.31e+04 [4.94e+04, 5.68e+04] |
| D1 | 2800 | 224000 | 56 | 1.16e+03 [1.06e+03, 1.26e+03] | 7.37e+04 [6.07e+04, 8.67e+04] |
| D2 | 3700 | 296000 | 74 | 1.16e+03 [1.09e+03, 1.23e+03] | 7.96e+04 [6.97e+04, 8.95e+04] |
| N1 | 2800 | 224000 | 56 | 1.05e+03 [963, 1.15e+03] | 1.01e+05 [8.37e+04, 1.19e+05] |
| N2 | 3550 | 284000 | 71 | 987 [903, 1.07e+03] | 8.77e+04 [7.23e+04, 1.03e+05] |

The 80× whole-batch trace is the stipulated episode-equivalent variance for the coupled leave-one-out baseline. Measurement critics are warmed and frozen; V1/V2 normalization uses full-batch statistics, whereas the training references normalize per chunk. Intervals are conditional on these checkpoints and do not measure training-seed variability. Exact selected jobs appear in [render_record.json](render_record.json).
