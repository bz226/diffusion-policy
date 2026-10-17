# Matched-iteration interim comparison

Evaluation is at iteration 60; training return and discounted return are from iteration 59. Each row is one seed and 80 rollout episodes. Missing values remain N/A.

| Method | Seed | Evaluation | Training return | J_disc_train | Run status |
|---|---:|---:|---:|---:|---|
| without_PPO_and_GAE | 0 | 4512.4 | 4113 | 1447.8 | complete |
| without_PPO_and_GAE | 1 | 4266.2 | 4115.2 | 1464.7 | complete |
| without_PPO_and_GAE | 2 | 4461.6 | 4048.7 | 1431.4 | complete |
| without_PPO_and_GAE + γ^k | 0 | 4342.1 | 4087.1 | 1461.8 | partial |
| without_PPO_and_GAE + γ^k | 1 | 4467.3 | 4009.7 | 1416.3 | partial |
| without_PPO_and_GAE + γ^k | 2 | N/A | N/A | N/A | missing |
