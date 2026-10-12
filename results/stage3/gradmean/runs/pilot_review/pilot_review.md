# Pilot timing readout

Verified P0_j00 completed fifty batches, exited zero, and reaped all forty environment workers. Its mean rollout + five-gradient + diagnostic work was **13.43 seconds per batch**; the recurring mean after removing the one-time score sample was **13.42 seconds**.

At the maximum 20,500 batches, this projects **76.41 GPU-hours** of batch work, or **79.18 GPU-hours** including the pilot allocation overhead repeated for 410 collection jobs. Ideal wall time with nine A6000s is **8.80 hours** for the latter estimate. Preparations, gates, CPU analysis barriers, queue delays and any retries are additional; point-specific timings may differ and early closing reduces work.

The pilot spent 0.76 seconds on the 200-score diagnostic. Estimated startup before collection was 11.37 seconds; recorded cleanup was 0.06 seconds. The allocation took 696.00 seconds total. These components are reported separately; startup is an estimate from saved timestamps.

**The approved 125 GPU-hour cap and MAX_PAR=9 remain unchanged.** Actual allocated time plus outstanding job reservations controls admission.

Available preparation checks:

- P0: 20 warmup batches recorded; frozen-critic EV 0.147103 (completed).

[Full timing definitions and source paths](pilot_review.json). Explained variance uses each preparation’s separate frozen-critic evaluation batch.
