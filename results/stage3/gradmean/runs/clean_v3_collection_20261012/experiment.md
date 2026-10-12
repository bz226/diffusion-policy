# C1/C2 collection for clean figures version3

Status: **blocked**. Updated 2026-10-12T16:31:14.885784+00:00.

Twenty50-batch V1-V6 collect jobs and two5-batch critic preparations; no actor updates. The collector uses internal labels D1/D2 for external C1/C2; all paths, checkpoints and seeds are separate.

Monitor interval:30minutes. Estimate:60–70minutes plus scheduling, based on completed V6 jobs. MAX_PAR=5; each worker1A6000/40CPUs/64GiB with a1-hour cap; inherited aggregate cap125GPU-hours.

| Run | Scheduler job | State | Progress |
|---|---|---|---|
| prepare_C1 | 17825416 | complete | 5 |
| prepare_C2 | 17825417 | complete | 5 |
| C1_j00 | 17825441 | complete | 50 |
| C1_j01 | 17825442 | complete | 50 |
| C1_j02 | 17825443 | complete | 50 |
| C1_j03 | 17825444 | submitted | 47 |
| C1_j04 | 17825446 | complete | 50 |
| C1_j05 | 17825571 | submitted | 4 |
| C1_j06 | 17825578 | submitted | — |
| C1_j07 | 17825582 | submitted | — |
| C1_j08 | 17825583 | submitted | — |

Next step: finish only the fixed C1/C2 jobs; no adaptive rounds or automatic replacement jobs.
