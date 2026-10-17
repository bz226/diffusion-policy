# CPU verification

All requested checks passed before training. [Detailed results](gate_summary.json) record the commands and comparisons; the reference scientific revision is `b459412cfc272a7b8830c77925271843c27222b2`.

| Check | Result |
|---|---|
| G1: post-normalization weights | Worst relative L2 error: update **2.31e−7**, split diagnostic **1.51e−7**; tolerance 1e−6. Configured calibration and training gradients matched bitwise. |
| G2: episode index | Exact `k = t mod 250`; a misplaced boundary raised the alignment error. |
| G3: unit time weights | Actor gradients, actor/critic parameters, optimizer states and RNG matched bitwise; maximum difference **0**. Only a test-local weight tensor was replaced; γ was unchanged. |
| G4: V6 factor | Explicit float64 definition: relative error **6.56e−16**; tolerance 1e−12. |
| G5: unchanged defaults | Native full updates for DPPO defaults, without_PPO defaults and actual E1C flags matched bitwise, including losses, gradients, optimizer states, RNG and fixed-noise sampling. Maximum difference **0**. |
| G6: inert segments | Existing checkpoint-evaluation outputs and RNG matched bitwise, excluding timings; maximum difference **0**. |
| Optional E2C C1 | Raw residual relative error **5.88e−16**; actual E2C coefficient flags matched the explicit float32 expression bitwise. |

Training retains normalization within each shuffled 50,000-pair chunk; split diagnostics retain full-batch statistics. Both use sample SD and epsilon 1e−8. [Dispatch checks](ckpt_dispatch/checks.json) additionally verify 2/6 repeats, sampler selection, exact episode counts and invalid flag combinations.

**Scope and limitations.** Checks reuse the [saved NC4 fixture](../../../runs/gate_capture/nc4/training_batch.pt) already used to verify E1C, with Monte Carlo critic coefficients. It is not an original completed-E1C rollout. G6 replays saved rollout inputs through the checkpoint-evaluation pipeline, including actual saturation calculations; it generates no new rollouts. V6 is recomputed explicitly because the reference source predates V6. These checks ran on CPU.

The first startup failed before scientific checks because D4RL tried its read-only home-directory cache. [Preserved attempt](../gate_attempts/startup_d4rl/classification.json). Redirecting cache/temp paths fixed this environment issue without an algorithm change. Successful gates took 589.69 seconds; the failed startup took 31.02 seconds, within the one-hour cap.

[Final cleanup verification](final_cleanup_check.json) found no active gate processes. All 11 gate constructor records confirm that their environment workers exited and no children remained.
