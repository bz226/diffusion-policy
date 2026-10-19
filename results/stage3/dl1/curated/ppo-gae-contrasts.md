# Contrasts

For each ordinary contrast, pooled seed SD is sqrt((s_A²+s_B²)/2). For the interaction it is sqrt((s_DPPO²+s_without_PPO²+s_without_GAE²+s_without_PPO_and_GAE²)/2), the pooled SD of two independent reuse contrasts; no paired-seed covariance is assumed. A difference smaller in magnitude than twice this SD is called unresolved. SE is also recorded in CSV as pooled SD × sqrt(2/3), without an additional significance test.

| Contrast | Metric | Difference | Pooled seed SD | Reading |
| --- | --- | --- | --- | --- |
| GAE under reuse | Last five evals | 128.28 | 48.386 | exceeds reading-rule threshold |
| GAE under reuse | Train last 10 updates | 180.72 | 84.457 | exceeds reading-rule threshold |
| GAE under reuse | J_disc gain, last 10 | not logged | not logged | not logged |
| GAE without reuse | Last five evals | 83.434 | 31.971 | exceeds reading-rule threshold |
| GAE without reuse | Train last 10 updates | 103.73 | 35.072 | exceeds reading-rule threshold |
| GAE without reuse | J_disc gain, last 10 | not logged | not logged | not logged |
| reuse under GAE | Last five evals | 147.2 | 49.5 | exceeds reading-rule threshold |
| reuse under GAE | Train last 10 updates | 165.11 | 62.49 | exceeds reading-rule threshold |
| reuse under GAE | J_disc gain, last 10 | not logged | not logged | not logged |
| reuse under Monte Carlo | Last five evals | 102.36 | 30.217 | exceeds reading-rule threshold |
| reuse under Monte Carlo | Train last 10 updates | 88.113 | 66.768 | unresolved |
| reuse under Monte Carlo | J_disc gain, last 10 | 60.906 | 12.136 | exceeds reading-rule threshold |
| interaction | Last five evals | 44.842 | 57.994 | unresolved |
| interaction | Train last 10 updates | 76.993 | 91.449 | unresolved |
| interaction | J_disc gain, last 10 | not logged | not logged | not logged |
