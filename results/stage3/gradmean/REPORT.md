# Gradient mean along DPPO and NC4 paths

The amended synthetic gate **passed in both regimes**. G1/G5 passed; P0 preparation is now running before the G3/G4 live checks. Production collection still awaits those live gates. Unresolved directions now report N/A and unresolved signal produces a noise-scale lower bound, without stopping collection.

| G2 regime (200 replicates each) | Mean signal (true 1) | Mean trace (true) | Signal / trace 90% coverage | Mean effective rank (true 61.6922) |
|---|---:|---:|---:|---:|
| L | 0.996961 | 4000.1117 (4000) | 96.5% / 90% | 61.6249 |
| T | 1.003092 | 470.0131 (470) | 92% / 90% | 61.6249 |

Covariance variance splits 90% low-rank and 10% isotropic in both regimes. L had 1/200 nonpositive split cosines, correctly marked N/A; signed cosine error averaged −0.022559 over the other 199. T had mean absolute cosine error 0.014930 ; 200/200 errors were ≤ 0.06. Every formal criterion passed; complete estimates, Monte-Carlo standard errors and jackknife evidence are in the [gate record](runs/gate_G2_G1/gate_summary.json).

The Python 3.8 cleanup error was an unsupported `cancel_futures` keyword in `ProcessPoolExecutor.shutdown`. The source now calls `shutdown(wait=True)`, explicitly joins/closes workers, and checks for surviving children. Slurm job 17812422 completed with exit 0:0 in 44 seconds; all eight workers exited 0 and no children remained. New grad_mean jobs also close the vector environment and verify all forty workers at exit. Their live shutdown check is still pending. G1 matched the five Phase-1 gradient definitions at maximum relative L2 difference 6.60e-7, and V4 matched the negative E2 loss gradient exactly. Default and NC4 replay passed on both CPU and GPU with maximum absolute difference zero; [G1/G5 evidence](runs/gate_offline_G1/gate_summary.json).

The [original failed gate and traceback](provenance/report_before_G1_amendment.md) remain preserved. See the [methods](methods.md), [amendment authorization](provenance/amendment_G1.json), and [current status](experiment.md). The approved 125 GPU-hour cap and all scientific definitions outside G-1 remain unchanged. The controller has automatically advanced to P0 preparation and will run the two P0 live checks, then the authorized collection rounds if every gate passes.

<!-- gradmean-runtime-status-start -->
**Current runtime status (2026-10-12T05:56:36.884956+00:00): blocked.** Parent branch must be stage3-gradmean. No further collection is being admitted. The revised G2 gate passed. [Bootstrap failure record](runs/bootstrap.json); [Offline gate](runs/gate_offline_G1/gate_summary.json); [Live gate](runs/gate_live_G1/gate_summary.json); [Live gate log](runs/gate_live_G1.out); [Revised G2](runs/gate_G2_G1/gate_summary.json).
<!-- gradmean-runtime-status-end -->
