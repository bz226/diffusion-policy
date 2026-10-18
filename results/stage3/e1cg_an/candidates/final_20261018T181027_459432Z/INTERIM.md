# Matched-iteration interim comparison

Seed 0 per optimizer, estimator without_PPO_and_GAE + γ^k. Evaluation is at iteration 70; training metrics are from iteration 69, the immediately preceding update. Values are from 80 episodes; missing values remain N/A.

| Optimizer | Evaluation | Training return | J_disc | KL/update | Status |
|---|---:|---:|---:|---:|---|
| AdaGrad-Norm, η = d* | 4319.2 | 3926.9 | 1394.2 | 2.4143e-05 | complete |
| AdaGrad-Norm, η = 3d* | 4312.5 | 4049.4 | 1444.6 | 4.0415e-05 | complete |
| AdaGrad-Norm, η = 10d* | 4191.5 | 3932.2 | 1392.4 | 0.0005113 | complete |
| AdamW (lr 1e-4) | 4457.5 | 4143.8 | 1447.7 | 0.0002735 | complete |
