# Actor-epoch comparison, seed 0

One fresh-data update consumes 80 episodes. Missing values are not logged; no historical run was repeated.

## Setup

| Method | Actor epochs | Actor steps/batch | Actor lr | Movement / DPPO |
| --- | --- | --- | --- | --- |
| DPPO | 5 | 20 | 0.0001 | 1 |
| without_PPO | single full-batch step | 1 | 0.0001 | 0.1 |
| without_PPO, lr 1e-3 | single full-batch step | 1 | 0.001 | 1 |
| DPPO, 1 epoch | 1 | 4 | 0.0001 | 0.2 |
| DPPO, 1 epoch, lr 5e-4 | 1 | 4 | 0.0005 | 1 |
| DPPO, 2 epochs | 2 | 8 | 0.0001 | 0.4 |
| DPPO, 2 epochs, lr 2.5e-4 | 2 | 8 | 0.00025 | 1 |

Movement for without_PPO assumes quarter-batch √v is twice full-batch √v. The remaining matches use k·lr/(5·1e-4).

## Returns

| Method | Eval at 9.36M | Mean last five evals | Train last ten updates | J_disc gain last ten updates |
| --- | --- | --- | --- | --- |
| DPPO | 4636.3 | 4636.7 | 4378.5 | not logged |
| without_PPO | 4624.1 | 4558.1 | 4303.5 | not logged |
| without_PPO, lr 1e-3 | 3279 | 3419.1 | 3069.8 | not logged |
| DPPO, 1 epoch | 4517.2 | 4463.1 | 4266.2 | 75.59 |
| DPPO, 1 epoch, lr 5e-4 | 4608.8 | 4601.4 | 4325.1 | 106.96 |
| DPPO, 2 epochs | 4666.5 | 4661 | 4404 | 144.46 |
| DPPO, 2 epochs, lr 2.5e-4 | 4711.7 | 4742.2 | 4452.5 | 182.11 |

## Update diagnostics

| Method | Final parameter distance | Mean last-step approx_kl | Mean last-step switched-off fraction | Mean last-step |ratio − 1| | Final saturation |
| --- | --- | --- | --- | --- | --- |
| DPPO | 6.8585 | 1.4003e-05 | not logged | not logged | not logged |
| without_PPO | 0.90975 | not logged | not logged | not logged | not logged |
| without_PPO, lr 1e-3 | 8.0028 | not logged | not logged | not logged | not logged |
| DPPO, 1 epoch | 1.7526 | 1.189e-06 | 0.00020206 | 0.0010686 | 0.22467 |
| DPPO, 1 epoch, lr 5e-4 | 7.1581 | 1.2643e-05 | 0.023798 | 0.0035497 | 0.32974 |
| DPPO, 2 epochs | 3.256 | 5.6875e-06 | 0.0073217 | 0.0024169 | 0.22631 |
| DPPO, 2 epochs, lr 2.5e-4 | 6.9837 | 1.2025e-05 | 0.024969 | 0.003635 | 0.28466 |

J_disc gain subtracts each run’s own first training-batch value. Final saturation uses the last training rollout, before update 126; final distance uses the checkpoint after update 126.

Stage-1 three-seed sample SD (the reading-rule threshold is twice this): Eval at 9.36M: 108.29; Mean last five evals: 67.918; Train last ten updates: 87.714; J_disc gain last ten updates: not logged; Final parameter distance: 0.048869; Mean last-step approx_kl: 1.9614e-06; Mean last-step switched-off fraction: not logged; Mean last-step |ratio − 1|: not logged; Final saturation: not logged.
