# Gradient mean along DPPO and NC4 paths

The amended synthetic gate **passed in both regimes**. Live-policy measurements have not started; G1/G5 replay and G3/G4 live checks remain before production. Unresolved directions now report N/A and unresolved signal produces a noise-scale lower bound, without stopping collection.

| G2 regime (200 replicates each) | Mean signal (true 1) | Mean trace (true) | Signal / trace 90% coverage | Mean effective rank (true 61.6922) |
|---|---:|---:|---:|---:|
| L | 0.996961 | 4000.1117 (4000) | 96.5% / 90% | 61.6249 |
| T | 1.003092 | 470.0131 (470) | 92% / 90% | 61.6249 |

Covariance variance splits90% low-rank and10% isotropic in both regimes. L had1/200 nonpositive split cosines, correctly marked N/A; signed cosine error averaged−0.022559 over the other199. T had mean absolute cosine error0.014930;200/200 errors were≤0.06. Every formal criterion passed; complete estimates, Monte-Carlo standard errors and jackknife evidence are in the [gate record](runs/gate_G2_G1/gate_summary.json).

The Python3.8 cleanup error was an unsupported `cancel_futures` keyword in `ProcessPoolExecutor.shutdown`. The source now calls `shutdown(wait=True)`, explicitly joins/closes workers, and checks for surviving children. Slurm job17812422 completed with exit0:0 in44seconds; all eight workers exited0 and no children remained. New grad_mean jobs also close the vector environment and verify all forty workers at exit. Their live shutdown check is still pending.

The [original failed gate and traceback](provenance/report_before_G1_amendment.md) remain preserved. See the [methods](methods.md), [amendment authorization](provenance/amendment_G1.json), and [current status](experiment.md). The approved125GPU-hour cap and all scientific definitions outside G-1 remain unchanged. The next action is the saved-batch gate, followed automatically by the two P0 live checks and the authorized collection rounds if every gate passes.
