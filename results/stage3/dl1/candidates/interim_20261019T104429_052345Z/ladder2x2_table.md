# Two-by-two reward comparison

Each cell is the mean ± sample SD (ddof=1) over three training seeds, followed by seeds 0, 1, 2. Missing measurements are not replaced by other runs. Evaluation at 9.36M is iteration 130; the last five evals are iterations 90–130, and the last ten training updates are iterations 129 and 131–139. Each discounted-return gain subtracts that seed's first training batch (iteration 1, before any actor update).

## Eval at 9.36M

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Last five evals

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Train last 10 updates

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## J_disc gain, last 10

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Final distance from own start

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Mean true-KL diagnostic per update

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Final rollout saturation

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Mean clipfrac

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Mean switched-off fraction

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

## Mean PPO approximate KL per update

| Method | Mean ± SD; seed 0 / 1 / 2 |
| --- | --- |
| DPPO | not logged or pending completion |
| without_PPO | not logged or pending completion |
| without_PPO_and_GAE | not logged or pending completion |
| without_GAE | not logged or pending completion |

True-KL diagnostics are sampled coordinate-summed, unclamped post-update transition KL estimates; PPO approximate KL is the within-update, clamped coordinate-mean statistic. Their columns are intentionally separate. Final saturation comes from the last training rollout (before update 126), while final parameter distance is after update 126. Checkpoint distances use float64 differences from each run's own state_0.
