# Optimizer outcomes

Each metric is mean ± sample SD over seeds 0, 1 and 2, followed by all three seed values. A mean requires three finite values; missing or failed runs are retained. Main rows share the estimator without_PPO_and_GAE + γ^k. The reference rows change the estimator as stated; they do not isolate optimizer geometry.

| Optimizer / reference | Eval at 9.36M | Train last 10 updates | J_disc gain, last 10 updates | Final train-sampler return | Final train-sampler J_disc | Final eval-sampler return | Final eval-sampler J_disc | First-update KL | Mean KL/update | Final distance from θ₀ | Final pre-update x̂₀ saturation | Collapse by seed | Completion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdaGrad-Norm, η = d* | N/A ± N/A (4287.9, N/A, N/A) | N/A ± N/A (3900.8, N/A, N/A) | N/A ± N/A (22.095, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (0.00097446, 0.000262, N/A) | N/A ± N/A (2.9218e-05, N/A, N/A) | N/A ± N/A (0.0059581, N/A, N/A) | N/A ± N/A (0.066678, N/A, N/A) | [false, false, null] | 1/3 training; 0/3 final evaluations |
| AdaGrad-Norm, η = 3d* | N/A ± N/A (4339.3, N/A, N/A) | N/A ± N/A (3984.7, N/A, N/A) | N/A ± N/A (42.046, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (0.0072191, 0.0085218, N/A) | N/A ± N/A (0.00024023, N/A, N/A) | N/A ± N/A (0.01894, N/A, N/A) | N/A ± N/A (0.072783, N/A, N/A) | [false, false, null] | 1/3 training; 0/3 final evaluations |
| AdaGrad-Norm, η = 10d* | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (0.045885, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | N/A ± N/A (N/A, N/A, N/A) | [false, null, null] | 0/3 training; 0/3 final evaluations |
| AdamW (lr 1e-4) | 4496.8 ± 48.44 (4552.7, 4467.6, 4470.1) | 4185.5 ± 62.675 (4257.3, 4157.7, 4141.5) | 105.87 ± 21.303 (130.43, 92.374, 94.802) | 4212.2 ± 40.424 (4247.2, 4168, 4221.4) | 1479 ± 16.463 (1491.5, 1460.4, 1485.2) | 4505.6 ± 13.276 (4520.8, 4496.3, 4499.8) | 1583 ± 11.31 (1595.5, 1573.5, 1579.9) | 0.0044816 ± 0.00042858 (0.00442, 0.0049377, 0.0040872) | 0.00052192 ± 2.8804e-05 (0.00050228, 0.00055499, 0.0005085) | 0.78364 ± 0.01005 (0.79455, 0.7816, 0.77476) | 0.096956 ± 0.023362 (0.1151, 0.10517, 0.070596) | [false, false, false] | 3/3 training; 3/3 final evaluations |
| AdamW, no γ^k (reference) | 4503.9 ± 13.363 (4490.7, 4517.4, 4503.6) | 4207.5 ± 48.415 (4217.9, 4249.9, 4154.8) | 112.56 ± 14.063 (114.41, 125.61, 97.666) | 4218.7 ± 42.424 (4267.3, 4189.1, 4199.7) | 1486.4 ± 15.263 (1502.8, 1472.6, 1483.8) | 4481.1 ± 53.627 (4534.1, 4482.2, 4426.9) | 1572.1 ± 19.501 (1590.3, 1574.6, 1551.5) | 0.0042991 ± 0.00051146 (0.0046674, 0.0037151, 0.0045147) | 0.00059102 ± 3.1039e-05 (0.00061528, 0.00055604, 0.00060175) | 0.84458 ± 0.009946 (0.85464, 0.84435, 0.83476) | 0.11723 ± 0.030773 (0.1302, 0.1394, 0.082096) | [false, false, false] | 3/3 training; 3/3 final evaluations |
| SGD, 3η* (LOO reference) | 4438 ± 84.222 (4526.8, 4359.3, 4428) | 4092.5 ± 38.677 (4101.1, 4050.2, 4126) | 81.393 ± 14.012 (88.149, 65.283, 90.745) | 4099.1 ± 14.694 (4107.5, 4082.1, 4107.6) | 1453.6 ± 16.532 (1460.4, 1434.8, 1465.7) | 4402.7 ± 41.256 (4403.9, 4360.9, 4443.4) | 1555.2 ± 17.468 (1562.6, 1535.2, 1567.6) | 0.0051487 ± 0.0033451 (0.0013198, 0.0066228, 0.0075036) | 0.0036863 ± 0.00082159 (0.0033707, 0.0046189, 0.0030694) | 0.087553 ± 0.001547 (0.088039, 0.085822, 0.088799) | 0.089083 ± 0.0045657 (0.086841, 0.086072, 0.094336) | [false, false, false] | 3/3 training; 3/3 final evaluations |
| SGD, 10η* (LOO reference) | 4458.3 ± 105.92 (4533, 4504.9, 4337.1) | 4202.1 ± 111.11 (4261.5, 4270.9, 4074) | 110.67 ± 44.253 (130.78, 141.3, 59.935) | 4222.1 ± 120.03 (4274.7, 4306.8, 4084.7) | 1482.2 ± 49.224 (1503.5, 1517.1, 1425.9) | 4495.9 ± 98.65 (4519, 4581, 4387.8) | 1581.3 ± 42.261 (1599, 1611.7, 1533) | 0.047016 ± 0.042646 (0.095845, 0.017084, 0.02812) | 0.031953 ± 0.0088409 (0.038482, 0.021892, 0.035486) | 0.24169 ± 0.022423 (0.25206, 0.25705, 0.21596) | 0.1467 ± 0.014221 (0.131, 0.15873, 0.15036) | [false, false, false] | 3/3 training; 3/3 final evaluations |
| AdamW, paper-form LOO (reference) | 4542 ± 81.448 (4538.8, 4625, 4462.2) | 4306.6 ± 14.378 (4296.3, 4323, 4300.4) | 141.04 ± 6.6624 (137.81, 148.7, 136.6) | 4356.7 ± 18.384 (4368.3, 4366.2, 4335.5) | 1529.5 ± 4.68 (1532.1, 1532.3, 1524.1) | 4562.1 ± 24.133 (4578.1, 4573.8, 4534.3) | 1594.7 ± 12.555 (1608.4, 1591.8, 1583.8) | 0.004482 ± 0.00039585 (0.0047182, 0.0047028, 0.004025) | 0.00051159 ± 2.8713e-05 (0.00051021, 0.00054097, 0.0004836) | 0.87634 ± 0.011819 (0.86291, 0.88514, 0.88097) | 0.12771 ± 0.019699 (0.13446, 0.14314, 0.10552) | [false, false, false] | 3/3 training; 3/3 final evaluations |

J_disc gain subtracts 1369.0240467260296, the existing pretrained train-sampler mean; its uncertainty is not included in seed SD. A run collapses when any ten consecutive training-update returns average below the pretrained train-sampler mean minus 300. The threshold and first crossing are in seed_metrics.csv; non-finite values and crashes are recorded separately. Final saturation is measured on the last pre-update rollout, while distance is measured after the final update. New and AdamW final checkpoint evaluations use 160 episodes per sampler; LOO references retain their saved 80-episode evaluations.

## Differences

Each difference is condition minus reference, with SE = pooled seed SD × √(2/3). Differences smaller than twice pooled SD are called unresolved.

| Difference | Metric | Difference ± SE | Twice pooled SD | Reading |
|---|---|---:|---:|---|
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Train last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | J_disc gain, last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final train-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final train-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final eval-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final eval-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | First-update KL | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Mean KL/update | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final distance from θ₀ | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final pre-update x̂₀ saturation | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Train last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | J_disc gain, last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final train-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final train-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final eval-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final eval-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | First-update KL | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Mean KL/update | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final distance from θ₀ | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final pre-update x̂₀ saturation | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Train last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | J_disc gain, last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final train-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final train-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final eval-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final eval-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | First-update KL | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Mean KL/update | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final distance from θ₀ | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final pre-update x̂₀ saturation | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Train last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | J_disc gain, last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final train-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final train-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final eval-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final eval-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | First-update KL | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Mean KL/update | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final distance from θ₀ | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final pre-update x̂₀ saturation | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Train last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | J_disc gain, last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final train-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final train-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final eval-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final eval-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | First-update KL | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Mean KL/update | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final distance from θ₀ | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final pre-update x̂₀ saturation | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Eval at 9.36M | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Train last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | J_disc gain, last 10 updates | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final train-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final train-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final eval-sampler return | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final eval-sampler J_disc | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | First-update KL | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Mean KL/update | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final distance from θ₀ | N/A ± N/A | N/A | unavailable |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final pre-update x̂₀ saturation | N/A ± N/A | N/A | unavailable |
