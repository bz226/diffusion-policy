# Matched-iteration interim comparison

Evaluation is at iteration 70; training return and discounted return are from iteration 69. Each row is one seed and 80 rollout episodes. Missing values remain N/A.

| Method | Seed | Evaluation | Training return | J_disc_train | Run status |
|---|---:|---:|---:|---:|---|
| without_PPO_and_GAE | 0 | 4494.7 | 4178.9 | 1449.8 | complete |
| without_PPO_and_GAE | 1 | 4379.2 | 4057.5 | 1427.4 | complete |
| without_PPO_and_GAE | 2 | 4443.9 | 4077.3 | 1463.4 | complete |
| without_PPO_and_GAE + γ^k | 0 | 4457.5 | 4143.8 | 1447.7 | complete |
| without_PPO_and_GAE + γ^k | 1 | 4459.7 | 4151.2 | 1459.6 | complete |
| without_PPO_and_GAE + γ^k | 2 | 4355.2 | 4086 | 1428.5 | complete |
