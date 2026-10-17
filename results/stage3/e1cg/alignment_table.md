# Existing-gradient alignment

Cosines compare Euclidean mean ascent directions. Brackets are 90% delete-one-group intervals: ten independent 50-batch groups and t₉, except the preserved nine-group warmed checkpoint with t₈. Numerators and both squared-norm denominators use the same paired-batch U-statistic, excluding same-batch products. A nonpositive denominator in any full/deleted sample makes the cosine unresolved; values above 1 are retained as noisy estimates.

| Point | Path | Update | K | Critic | c(V2,V6) | c(V2,V4) | c(V2,V5) | c(V4,V6) | c(V5,V6) | Common-mean checks |
|---|---|---:|---:|---|---|---|---|---|---|---|
| W0_000 | without_PPO_and_GAE | 0 | 500 | own checkpoint | 0.783 [0.2787, 1.287] | 0.9887 [0.8352, 1.142] | 0.783 [0.2787, 1.287] | 0.6833 [0.2042, 1.162] | 1 [1, 1] | passed |
| W0_035 | without_PPO_and_GAE | 32 | 500 | own checkpoint | 1.013 [0.8543, 1.171] | 1.162 [0.7116, 1.612] | unresolved | 1.114 [0.858, 1.37] | unresolved | unresolved |
| W0_070 | without_PPO_and_GAE | 63 | 500 | own checkpoint | 1.015 [0.9241, 1.105] | 0.9119 [0.5399, 1.284] | unresolved | 0.9712 [0.6893, 1.253] | unresolved | unresolved |
| W0_105 | without_PPO_and_GAE | 95 | 500 | own checkpoint | 1.14 [0.801, 1.479] | 0.9331 [0.5714, 1.295] | 0.847 [0.3287, 1.365] | 0.9019 [0.6532, 1.151] | 0.725 [0.1385, 1.311] | passed |
| W0_139 | without_PPO_and_GAE | 126 | 500 | own checkpoint | 1.065 [0.7578, 1.373] | 1.173 [-0.104, 2.451] | unresolved | 1.085 [0.4064, 1.764] | unresolved | unresolved |
| P0 | pretrained policy | 0 | 500 | warmed 20 batches | 1.022 [0.9494, 1.095] | 1.095 [0.8432, 1.347] | unresolved | 1.028 [0.8967, 1.159] | unresolved | unresolved |
| D1 | DPPO | 63 | 500 | warmed 5 batches | 1.076 [0.6344, 1.519] | 0.7833 [0.1653, 1.401] | unresolved | 1.133 [0.02065, 2.245] | unresolved | unresolved |
| D2 | DPPO | 126 | 500 | warmed 5 batches | unresolved | unresolved | unresolved | 0.8168 [0.1063, 1.527] | 0.8605 [-6.726, 8.447] | passed |
| N1 | without_PPO | 63 | 500 | warmed 5 batches | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved |
| N2 | without_PPO | 126 | 500 | warmed 5 batches | unresolved | unresolved | unresolved | 0.8152 [-0.6032, 2.234] | unresolved | unresolved |
| C1 | without_PPO_and_GAE | 63 | 450 | warmed 5 batches | 1.033 [0.7371, 1.329] | 1.189 [0.518, 1.861] | 0.6276 [-8.066, 9.321] | 0.8723 [0.4956, 1.249] | 0.9479 [-9.375, 11.27] | passed |

**Limits.** V2 at the five own-critic points is raw MC return minus cV, normalized over the full batch. The older warmed cohorts use MC of scaled/clipped rewards minus V; those are distinct proxies and are not pooled. Training normalized each 50,000-pair chunk; the authorized two-batch proxy check had cosines 0.999991 and 0.999994, and quarter/full norm ratios 1.000176 and 0.999930. The final own-critic point uses a five-batch fitted c = 438.775862; its interval conditions on that scale. AdamW preconditions gradients, so Euclidean alignment is an indicator, not an exact trajectory prediction.

The CSV lists checkpoint paths, source revisions, every selected group and invalid deletion. A failed common-mean interval excludes 1 and blocks interpretation of the main cosine; an unresolved check does not establish agreement.
