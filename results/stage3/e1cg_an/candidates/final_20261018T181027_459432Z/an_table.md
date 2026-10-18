# Optimizer outcomes

Each metric is mean ± sample SD over seeds 0, 1 and 2, followed by all three seed values. A mean requires three finite values; missing or failed runs are retained. Main rows share the estimator without_PPO_and_GAE + γ^k. The reference rows change the estimator as stated; they do not isolate optimizer geometry.

| Optimizer / reference | Eval at 9.36M | Train last 10 updates | J_disc gain, last 10 updates | Final train-sampler return | Final train-sampler J_disc | Final eval-sampler return | Final eval-sampler J_disc | First-update KL | Mean KL/update | Final distance from θ₀ | Final pre-update x̂₀ saturation | Collapse by seed | Completion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdaGrad-Norm, η = d* | 4277.9 ± 25.156 (4287.9, 4249.3, 4296.5) | 3896.5 ± 19.769 (3900.8, 3875, 3913.8) | 18.302 ± 6.3185 (22.095, 11.008, 21.804) | 3945.6 ± 39.82 (3953.6, 3980.8, 3902.4) | 1404.2 ± 9.1218 (1402.5, 1414, 1396) | 4292.8 ± 46.444 (4258.8, 4345.7, 4273.9) | 1513.1 ± 11.198 (1504.8, 1525.8, 1508.7) | 0.00050542 ± 0.0004063 (0.00097446, 0.000262, 0.0002798) | 2.4528e-05 ± 4.4685e-06 (2.9218e-05, 2.0321e-05, 2.4044e-05) | 0.0062261 ± 0.00023945 (0.0059581, 0.0063011, 0.006419) | 0.065342 ± 0.001648 (0.066678, 0.065849, 0.063501) | [false, false, false] | 3/3 training; 3/3 final evaluations |
| AdaGrad-Norm, η = 3d* | 4329.2 ± 9.2357 (4339.3, 4326.9, 4321.3) | 3966 ± 32.866 (3984.7, 3928, 3985.2) | 37.211 ± 8.0505 (42.046, 27.917, 41.669) | 3955.7 ± 39.942 (3912.4, 3963.5, 3991.1) | 1403.9 ± 12.207 (1390, 1412.8, 1408.9) | 4276.8 ± 45.532 (4293.2, 4311.8, 4225.3) | 1508 ± 11.961 (1517.3, 1512.3, 1494.5) | 0.0059802 ± 0.0033382 (0.0072191, 0.0085218, 0.0021996) | 0.00024969 ± 3.3888e-05 (0.00024023, 0.00022153, 0.0002873) | 0.018663 ± 0.00034097 (0.01894, 0.018765, 0.018282) | 0.06813 ± 0.0040984 (0.072783, 0.065055, 0.066552) | [false, false, false] | 3/3 training; 3/3 final evaluations |
| AdaGrad-Norm, η = 10d* | 4304.6 ± 23.487 (4331.3, 4287.5, 4294.9) | 4001.4 ± 38.373 (4019.1, 4027.7, 3957.4) | 51.779 ± 12.395 (53.109, 63.456, 38.773) | 3995 ± 98.662 (4085.7, 3890, 4009.2) | 1422.1 ± 17.556 (1438.8, 1403.8, 1423.6) | 4328.7 ± 33.635 (4342.3, 4353.5, 4290.4) | 1533.9 ± 9.0667 (1538.5, 1539.7, 1523.4) | 0.044978 ± 0.018474 (0.045885, 0.026067, 0.062982) | 0.0023537 ± 0.00066453 (0.0024322, 0.0016534, 0.0029755) | 0.062197 ± 0.0027934 (0.06287, 0.064593, 0.059129) | 0.083943 ± 0.016535 (0.090523, 0.096175, 0.065132) | [false, false, false] | 3/3 training; 3/3 final evaluations |
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
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Eval at 9.36M | -218.96 ± 31.513 | 77.191 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Train last 10 updates | -288.94 ± 37.943 | 92.94 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | J_disc gain, last 10 updates | -87.565 ± 12.829 | 31.425 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final train-sampler return | -266.61 ± 32.76 | 80.246 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final train-sampler J_disc | -74.832 ± 10.866 | 26.617 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final eval-sampler return | -212.8 ± 27.889 | 68.313 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final eval-sampler J_disc | -69.875 ± 9.189 | 22.508 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | First-update KL | -0.0039762 ± 0.00034096 | 0.00083518 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Mean KL/update | -0.00049739 ± 1.6829e-05 | 4.1222e-05 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final distance from θ₀ | -0.77741 ± 0.0058039 | 0.014217 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − AdamW (lr 1e-4) | Final pre-update x̂₀ saturation | -0.031614 ± 0.013521 | 0.03312 | unresolved |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Eval at 9.36M | -167.68 ± 28.471 | 69.739 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Train last 10 updates | -219.5 ± 40.859 | 100.08 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | J_disc gain, last 10 updates | -68.657 ± 13.148 | 32.207 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final train-sampler return | -256.54 ± 32.81 | 80.367 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final train-sampler J_disc | -75.101 ± 11.833 | 28.984 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final eval-sampler return | -228.82 ± 27.382 | 67.073 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final eval-sampler J_disc | -74.977 ± 9.5041 | 23.28 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | First-update KL | 0.0014986 ± 0.0019431 | 0.0047597 | unresolved |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Mean KL/update | -0.00027224 ± 2.5678e-05 | 6.2898e-05 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final distance from θ₀ | -0.76497 ± 0.0058056 | 0.014221 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − AdamW (lr 1e-4) | Final pre-update x̂₀ saturation | -0.028827 ± 0.013694 | 0.033543 | unresolved |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Eval at 9.36M | -192.28 ± 31.081 | 76.133 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Train last 10 updates | -184.09 ± 42.429 | 103.93 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | J_disc gain, last 10 updates | -54.088 ± 14.23 | 34.856 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final train-sampler return | -217.22 ± 61.558 | 150.79 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final train-sampler J_disc | -56.954 ± 13.895 | 34.037 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final eval-sampler return | -176.88 ± 20.877 | 51.138 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final eval-sampler J_disc | -49.124 ± 8.3689 | 20.5 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | First-update KL | 0.040496 ± 0.010669 | 0.026133 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Mean KL/update | 0.0018318 ± 0.00038403 | 0.00094068 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final distance from θ₀ | -0.72144 ± 0.0060222 | 0.014751 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − AdamW (lr 1e-4) | Final pre-update x̂₀ saturation | -0.013013 ± 0.016524 | 0.040476 | unresolved |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Eval at 9.36M | -160.16 ± 50.748 | 124.31 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Train last 10 updates | -195.9 ± 25.078 | 61.428 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | J_disc gain, last 10 updates | -63.09 ± 8.8741 | 21.737 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final train-sampler return | -153.49 ± 24.505 | 60.025 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final train-sampler J_disc | -49.421 ± 10.901 | 26.703 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final eval-sampler return | -109.94 ± 35.866 | 87.854 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final eval-sampler J_disc | -42.038 ± 11.98 | 29.344 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | First-update KL | -0.0046433 ± 0.0019455 | 0.0047654 | unresolved |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Mean KL/update | -0.0036618 ± 0.00047435 | 0.0011619 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final distance from θ₀ | -0.081327 ± 0.00090378 | 0.0022138 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = d* − SGD, 3η* (LOO reference) | Final pre-update x̂₀ saturation | -0.023741 ± 0.0028025 | 0.0068646 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Eval at 9.36M | -108.88 ± 48.917 | 119.82 | unresolved |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Train last 10 updates | -126.46 ± 29.303 | 71.778 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | J_disc gain, last 10 updates | -44.182 ± 9.3298 | 22.853 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final train-sampler return | -143.43 ± 24.572 | 60.188 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final train-sampler J_disc | -49.69 ± 11.865 | 29.062 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final eval-sampler return | -125.96 ± 35.474 | 86.893 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final eval-sampler J_disc | -47.14 ± 12.223 | 29.94 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | First-update KL | 0.00083144 ± 0.0027284 | 0.0066833 | unresolved |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Mean KL/update | -0.0034366 ± 0.00047475 | 0.0011629 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final distance from θ₀ | -0.068891 ± 0.00091459 | 0.0022403 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 3d* − SGD, 3η* (LOO reference) | Final pre-update x̂₀ saturation | -0.020953 ± 0.0035422 | 0.0086766 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Eval at 9.36M | -133.48 ± 50.481 | 123.65 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Train last 10 updates | -91.053 ± 31.456 | 77.05 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | J_disc gain, last 10 updates | -29.613 ± 10.801 | 26.456 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final train-sampler return | -104.11 ± 57.591 | 141.07 | unresolved |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final train-sampler J_disc | -31.542 ± 13.923 | 34.104 | unresolved |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final eval-sampler return | -74.018 ± 30.732 | 75.278 | unresolved |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final eval-sampler J_disc | -21.287 ± 11.363 | 27.833 | unresolved |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | First-update KL | 0.039829 ± 0.01084 | 0.026551 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Mean KL/update | -0.0013326 ± 0.00061009 | 0.0014944 | unresolved |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final distance from θ₀ | -0.025356 ± 0.0018436 | 0.0045158 | exceeds reading-rule threshold |
| AdaGrad-Norm, η = 10d* − SGD, 3η* (LOO reference) | Final pre-update x̂₀ saturation | -0.0051399 ± 0.0099035 | 0.024258 | unresolved |
