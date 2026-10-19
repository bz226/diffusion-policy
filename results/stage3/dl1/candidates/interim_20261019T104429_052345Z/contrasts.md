# Contrasts

For each ordinary contrast, pooled seed SD is sqrt((s_A²+s_B²)/2). For the interaction it is sqrt((s_DPPO²+s_without_PPO²+s_without_GAE²+s_without_PPO_and_GAE²)/2), the pooled SD of two independent reuse contrasts; no paired-seed covariance is assumed. A difference smaller in magnitude than twice this SD is called unresolved. SE is also recorded in CSV as pooled SD × sqrt(2/3), without an additional significance test.

| Contrast | Metric | Difference | Pooled seed SD | Reading |
| --- | --- | --- | --- | --- |
| GAE under reuse | Last five evals | not logged | not logged | not logged |
| GAE under reuse | Train last 10 updates | not logged | not logged | not logged |
| GAE under reuse | J_disc gain, last 10 | not logged | not logged | not logged |
| GAE without reuse | Last five evals | not logged | not logged | not logged |
| GAE without reuse | Train last 10 updates | not logged | not logged | not logged |
| GAE without reuse | J_disc gain, last 10 | not logged | not logged | not logged |
| reuse under GAE | Last five evals | not logged | not logged | not logged |
| reuse under GAE | Train last 10 updates | not logged | not logged | not logged |
| reuse under GAE | J_disc gain, last 10 | not logged | not logged | not logged |
| reuse under Monte Carlo | Last five evals | not logged | not logged | not logged |
| reuse under Monte Carlo | Train last 10 updates | not logged | not logged | not logged |
| reuse under Monte Carlo | J_disc gain, last 10 | not logged | not logged | not logged |
| interaction | Last five evals | not logged | not logged | not logged |
| interaction | Train last 10 updates | not logged | not logged | not logged |
| interaction | J_disc gain, last 10 | not logged | not logged | not logged |
