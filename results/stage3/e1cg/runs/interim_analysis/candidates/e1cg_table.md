# Training and checkpoint outcomes

Each cell is mean ± sample SD across seeds 0–2, followed by per-seed values. All three seeds are required for a cell; missing/failed seeds remain explicit. KL and consecutive-gradient cosine are means over training updates (only the first undefined cosine is omitted).

| Method | Eval at 9.36M | Train last 10 | J_disc gain last 10 | Final train sampler return | Final train sampler J_disc | Final eval sampler return | Final eval sampler J_disc | Mean KL/update | Mean consecutive-gradient cosine | Final distance | Final pre-update saturation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| without_PPO_and_GAE | 4503.9 ± 13.363 (4490.7, 4517.4, 4503.6) | 4207.5 ± 48.415 (4217.9, 4249.9, 4154.8) | 112.56 ± 14.063 (114.41, 125.61, 97.666) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | 0.00059102 ± 3.1039e-05 (0.00061528, 0.00055604, 0.00060175) | -0.001788 ± 0.0035271 (-0.005769, -0.000542, 0.00094696) | 0.84458 ± 0.009946 (0.85464, 0.84435, 0.83476) | 0.11723 ± 0.030773 (0.1302, 0.1394, 0.082096) |
| without_PPO_and_GAE + γ^k | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) |
| E1: MC + LOO | 4669.6 ± 28.089 (4683.4, 4688.2, 4637.3) | 4429.4 ± 17.052 (4422.4, 4417, 4448.9) | 184.13 ± 3.9671 (183.71, 180.39, 188.29) | 4420.4 ± 63.987 (4361.4, 4411.5, 4488.4) | 1551.1 ± 20.269 (1534.1, 1545.7, 1573.5) | 4705.2 ± 15.941 (4714.3, 4714.5, 4686.8) | 1642.6 ± 5.4594 (1639.7, 1648.9, 1639.3) | 0.00062349 ± 4.5782e-05 (0.00057715, 0.00062464, 0.00066869) | 0.0074549 ± 0.0041384 (0.0027365, 0.010469, 0.0091595) | 0.94871 ± 0.020338 (0.92662, 0.95285, 0.96666) | 0.1385 ± 0.0296 (0.11549, 0.12812, 0.17189) |
| E2: discounted MC + LOO | 4542 ± 81.448 (4538.8, 4625, 4462.2) | 4306.6 ± 14.378 (4296.3, 4323, 4300.4) | 141.04 ± 6.6624 (137.81, 148.7, 136.6) | 4356.7 ± 18.384 (4368.3, 4366.2, 4335.5) | 1529.5 ± 4.68 (1532.1, 1532.3, 1524.1) | 4562.1 ± 24.133 (4578.1, 4573.8, 4534.3) | 1594.7 ± 12.555 (1608.4, 1591.8, 1583.8) | 0.00051159 ± 2.8713e-05 (0.00051021, 0.00054097, 0.0004836) | -0.0019453 ± 0.0043249 (-0.002773, 0.0027336, -0.0057965) | 0.87634 ± 0.011819 (0.86291, 0.88514, 0.88097) | 0.12771 ± 0.019699 (0.13446, 0.14314, 0.10552) |

J_disc gain subtracts the saved pretrained train-sampler mean 1369.0240467260296; its uncertainty is not included in training-seed SD. Distance is after the final update; saturation is from the final training rollout before that update. New final-checkpoint measurements use 160 episodes per sampler; existing LOO references retain their original 80-episode measurements.

| Difference | Metric | Mean difference ± standard error | Twice pooled SD | Reading |
|---|---|---:|---:|---|
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Train last 10 | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | J_disc gain last 10 | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Final train sampler return | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Final train sampler J_disc | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Final eval sampler return | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Final eval sampler J_disc | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Mean KL/update | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Mean consecutive-gradient cosine | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Final distance | N/A ± N/A | N/A | unavailable |
| without_PPO_and_GAE + γ^k − without_PPO_and_GAE | Final pre-update saturation | N/A ± N/A | N/A | unavailable |
| E2: discounted MC + LOO − E1: MC + LOO | Eval at 9.36M | -127.62 ± 49.742 | 121.84 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Train last 10 | -122.83 ± 12.878 | 31.544 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | J_disc gain last 10 | -43.095 ± 4.4768 | 10.966 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Final train sampler return | -63.772 ± 38.437 | 94.152 | unresolved |
| E2: discounted MC + LOO − E1: MC + LOO | Final train sampler J_disc | -21.627 ± 12.01 | 29.42 | unresolved |
| E2: discounted MC + LOO − E1: MC + LOO | Final eval sampler return | -143.18 ± 16.698 | 40.903 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Final eval sampler J_disc | -47.948 ± 7.9045 | 19.362 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Mean KL/update | -0.0001119 ± 3.1201e-05 | 7.6425e-05 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Mean consecutive-gradient cosine | -0.0094002 ± 0.0034559 | 0.0084653 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Final distance | -0.07237 ± 0.013581 | 0.033266 | exceeds reading-rule threshold |
| E2: discounted MC + LOO − E1: MC + LOO | Final pre-update saturation | -0.010793 ± 0.020528 | 0.050283 | unresolved |

SE = pooled sample SD × √(2/3); differences smaller than twice pooled seed SD are unresolved. No additional significance tests are applied.
