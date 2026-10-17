# Gains by episode segment

Δ is the checkpoint's mean raw reward per decision minus the shared pretrained mean. T is mean Δ in the first two bins minus mean Δ in the last two. Each checkpoint has 160 training-sampler episodes; the common reference has 480. No seed or episode uncertainty is implied by the point estimates below.

| Method | Seed | Checkpoint iteration | Updates | Δ 0–49 | Δ 50–99 | Δ 100–149 | Δ 150–199 | Δ 200–249 | T | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| without_PPO_and_GAE | 0 | 35 | 32 | 0.6772 | 0.40761 | 0.52515 | -0.39738 | 0.20095 | 0.64062 | complete |
| without_PPO_and_GAE | 0 | 70 | 63 | 0.48706 | 0.70875 | 0.70407 | 0.53263 | 1.4542 | -0.3955 | complete |
| without_PPO_and_GAE | 0 | 105 | 95 | 0.7835 | 0.89561 | 0.81542 | 1.1035 | 1.1442 | -0.28431 | complete |
| without_PPO_and_GAE | 0 | 139 | 126 | 0.94261 | 1.2732 | 1.3529 | 1.1163 | 1.7594 | -0.32995 | complete |
| without_PPO_and_GAE | 1 | 35 | 32 | -0.071595 | -0.23153 | -0.16804 | -0.28223 | 0.40308 | -0.21199 | complete |
| without_PPO_and_GAE | 1 | 70 | 63 | 0.38473 | 0.66803 | 0.96765 | 0.6845 | 0.90596 | -0.26885 | complete |
| without_PPO_and_GAE | 1 | 105 | 95 | 0.69899 | 0.51973 | 0.75042 | 0.84228 | 1.4391 | -0.53132 | complete |
| without_PPO_and_GAE | 1 | 139 | 126 | 0.81403 | 0.61055 | 0.71501 | 1.1299 | 1.6107 | -0.658 | complete |
| without_PPO_and_GAE | 2 | 35 | 32 | 0.27636 | 0.26787 | 0.52098 | -0.12765 | 0.72543 | -0.026771 | complete |
| without_PPO_and_GAE | 2 | 70 | 63 | 0.46128 | 0.37477 | 0.49528 | 0.22528 | 0.17193 | 0.21942 | complete |
| without_PPO_and_GAE | 2 | 105 | 95 | 0.6078 | 0.74831 | 0.67647 | 0.14578 | 0.66526 | 0.27253 | complete |
| without_PPO_and_GAE | 2 | 139 | 126 | 0.92298 | 0.89826 | 1.0569 | 0.88433 | 1.3299 | -0.19647 | complete |
| without_PPO_and_GAE + γ^k | 0 | 35 | 32 | 0.5358 | 0.45225 | 0.72782 | 0.96086 | 0.60226 | -0.28753 | complete |
| without_PPO_and_GAE + γ^k | 0 | 70 | 63 | 0.72755 | 0.76918 | 0.82479 | 1.0257 | 1.0032 | -0.26612 | complete |
| without_PPO_and_GAE + γ^k | 0 | 105 | 95 | 0.89388 | 1.0436 | 1.248 | 1.127 | 1.242 | -0.21575 | complete |
| without_PPO_and_GAE + γ^k | 0 | 139 | 126 | 0.70944 | 1.2909 | 1.1763 | 1.0965 | 1.7703 | -0.43324 | complete |
| without_PPO_and_GAE + γ^k | 1 | 35 | 32 | 0.36291 | -0.20542 | -0.32994 | 0.28368 | 0.73348 | -0.42983 | complete |
| without_PPO_and_GAE + γ^k | 1 | 70 | 63 | 0.15705 | 0.41538 | 0.019364 | 0.22831 | 0.63446 | -0.14517 | complete |
| without_PPO_and_GAE + γ^k | 1 | 105 | 95 | 0.72703 | 0.38497 | 0.39243 | 0.52123 | 1.3467 | -0.37794 | complete |
| without_PPO_and_GAE + γ^k | 1 | 139 | 126 | 0.50398 | 0.57582 | 1.0817 | 0.81452 | 1.482 | -0.60834 | complete |
| without_PPO_and_GAE + γ^k | 2 | 35 | 32 | 0.23461 | 0.012593 | 0.28619 | 0.1001 | 0.71183 | -0.28236 | complete |
| without_PPO_and_GAE + γ^k | 2 | 70 | 63 | 0.74738 | 0.51855 | 0.72758 | 0.24581 | 1.0809 | -0.030385 | complete |
| without_PPO_and_GAE + γ^k | 2 | 105 | 95 | 0.4392 | 0.74999 | 0.52641 | 0.77065 | 1.4204 | -0.50093 | complete |
| without_PPO_and_GAE + γ^k | 2 | 139 | 126 | 0.80498 | 1.0461 | 0.96587 | 0.99997 | 1.709 | -0.42896 | complete |
