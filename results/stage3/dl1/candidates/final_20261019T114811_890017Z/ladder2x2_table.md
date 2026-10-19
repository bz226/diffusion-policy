# Two-by-two reward comparison

Each cell is the mean ± sample SD (ddof=1) over three training seeds, followed by seeds 0, 1, 2. Missing measurements are not replaced by other runs. Evaluation at 9.36M is iteration 130; the last five evals are iterations 90–130, and the last ten training updates are iterations 129 and 131–139. Each discounted-return gain subtracts that seed's first training batch (iteration 1, before any actor update).

## Eval at 9.36M

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | 4758.4 ± 108.29; 4636.3 / 4796 / 4842.8 |
| without_PPO | 4603.5 ± 19.143; 4624.1 / 4586.2 / 4600.3 |
| without_PPO_and_GAE | 4503.9 ± 13.363; 4490.7 / 4517.4 / 4503.6 |
| without_GAE | 4621 ± 18.559; 4620.9 / 4639.7 / 4602.6 |

## Last five evals

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | 4714.8 ± 67.918; 4636.7 / 4760.3 / 4747.3 |
| without_PPO | 4567.6 ± 16.958; 4558.1 / 4557.5 / 4587.2 |
| without_PPO_and_GAE | 4484.2 ± 41.913; 4514.5 / 4501.6 / 4436.3 |
| without_GAE | 4586.5 ± 8.3347; 4594.5 / 4587.2 / 4577.8 |

## Train last 10 updates

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | 4476.4 ± 87.714; 4378.5 / 4502.9 / 4547.8 |
| without_PPO | 4311.3 ± 10.772; 4303.5 / 4323.6 / 4306.7 |
| without_PPO_and_GAE | 4207.5 ± 48.415; 4217.9 / 4249.9 / 4154.8 |
| without_GAE | 4295.7 ± 81.068; 4366.2 / 4313.6 / 4207.1 |

## J_disc gain, last 10

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | 76.004 ± 4.1746; 80.34 / 75.66 / 72.012 |
| without_GAE | 136.91 ± 16.648; 156.1 / 126.34 / 128.29 |

## Final distance from own start

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | 6.9144 ± 0.048869; 6.8585 / 6.9359 / 6.9488 |
| without_PPO | 0.89446 ± 0.01566; 0.90975 / 0.87845 / 0.89519 |
| without_PPO_and_GAE | 0.84458 ± 0.009946; 0.85464 / 0.84435 / 0.83476 |
| without_GAE | 6.9372 ± 0.050504; 6.902 / 6.9146 / 6.9951 |

## Mean true-KL diagnostic per update

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | 0.00069527 ± 4.1368e-05; 0.00066847 / 0.00067444 / 0.00074292 |
| without_PPO_and_GAE | 0.00059102 ± 3.1039e-05; 0.00061528 / 0.00055604 / 0.00060175 |
| without_GAE | 0.086073 ± 0.0090711; 0.079451 / 0.096412 / 0.082355 |

## Final rollout saturation

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | 0.11723 ± 0.030773; 0.1302 / 0.1394 / 0.082096 |
| without_GAE | 0.20005 ± 0.015721; 0.21315 / 0.20437 / 0.18262 |

## Mean clipfrac

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | 0.023921 ± 0.00051116; 0.024488 / 0.023781 / 0.023494 |

## Mean switched-off fraction

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | 0.014471 ± 0.0002672; 0.014757 / 0.014427 / 0.014228 |

## Mean PPO approximate KL per update

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | 7.8145e-06 ± 4.8973e-07; 7.4393e-06 / 7.6357e-06 / 8.3685e-06 |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | 1.0164e-05 ± 4.7275e-06; 7.3889e-06 / 1.5623e-05 / 7.4811e-06 |

True-KL diagnostics are sampled coordinate-summed, unclamped post-update transition KL estimates; PPO approximate KL is the within-update, clamped coordinate-mean statistic. Their columns are intentionally separate. Final saturation comes from the last training rollout (before update 126), while final parameter distance is after update 126. Checkpoint distances use float64 differences from each run's own state_0.
