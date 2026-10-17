# Gains by episode segment

Δ is the checkpoint's mean raw reward per decision minus the shared pretrained mean. T is mean Δ in the first two bins minus mean Δ in the last two. Each checkpoint has 160 training-sampler episodes; the common reference has 480. No seed or episode uncertainty is implied by the point estimates below.

| Method | Seed | Checkpoint iteration | Updates | Δ 0–49 | Δ 50–99 | Δ 100–149 | Δ 150–199 | Δ 200–249 | T | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| without_PPO_and_GAE | 0 | 35 | 32 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 0 | 70 | 63 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 0 | 105 | 95 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 0 | 139 | 126 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 1 | 35 | 32 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 1 | 70 | 63 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 1 | 105 | 95 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 1 | 139 | 126 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 2 | 35 | 32 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 2 | 70 | 63 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 2 | 105 | 95 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE | 2 | 139 | 126 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 0 | 35 | 32 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 0 | 70 | 63 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 0 | 105 | 95 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 0 | 139 | 126 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 1 | 35 | 32 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 1 | 70 | 63 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 1 | 105 | 95 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 1 | 139 | 126 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 2 | 35 | 32 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 2 | 70 | 63 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 2 | 105 | 95 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
| without_PPO_and_GAE + γ^k | 2 | 139 | 126 | N/A | N/A | N/A | N/A | N/A | N/A | missing |
