# Stage 2: DPPO clipping ablations

Which clipping mechanisms support stable HalfCheetah-v2 fine-tuning, and does their effect depend on reusing a batch for multiple actor updates?

The agent’s scheduler handoff interrupted [R0](runs/R0_rerun_seed0/manifest.json) and [NC1 seed 1](runs/NC1_seed1/manifest.json) at 138/140 iterations, leaving their final two training rows and checkpoint missing. These orchestration failures remain [preserved](provenance/parallel_handoff_failure.json). The authorized fresh replacements `R0_recovery_seed0` and `NC1_recovery_seed1` now occupy those two analysis slots regardless of outcome; original trajectories are excluded, not spliced or averaged together.

**Executed results:** 19 of 19 new full runs completed; 12 met the prescribed observed-collapse criterion. R0 ended with status `complete`. The fixed-batch differential gate passed before launch; trajectories were not used to test code equivalence.

| Condition | Completed | Final eval mean ± std | Observed collapses |
|---|---:|---:|---:|
| baseline | 3/3 | 4758.4 ± 88.4 | 0/3 |
| R0 | 1/1 | 4621.8 (one run) | 0/1 |
| NC1 | 3/3 | -757.5 ± 593.5 | 3/3 |
| NC2 | 3/3 | -13.1 ± 833.1 | 3/3 |
| NC3 | 3/3 | 3043.0 ± 496.8 | 3/3 |
| NC4_lr1e-4 | 3/3 | 4603.5 ± 15.6 | 0/3 |
| NC4_lr1e-3 | 3/3 | 3726.4 ± 403.6 | 0/3 |
| NC4_lr3e-3 | 3/3 | -97.6 ± 476.8 | 3/3 |

Final evaluation means iteration 130 at 9.36 million training steps; completed training ends at 10.08 million. Bands use population standard deviation over all three requested seeds, without survivor averaging or endpoint substitution. Collapse means an observed evaluation below half the initial return or any recorded scientific NaN/inf. All 19 selected runs have finite saved metrics.

NC1 widens the ratio-clip bound to 1e6 and removes KL stopping; NC2 additionally removes log-probability and practical noise clamps; NC3 additionally sums log-probabilities. NC4 uses NC2 with one actor step per batch at the three specified constant learning rates. [Per-seed results](curated/condition_seed_results.csv) and [captions](curated/stage2-captions.md) cover all selected runs. NC3 seed 1, iteration 31, has clip fraction approximately 1e-6: the prescribed finite bound was exceeded rarely, as retained in the [curated diagnostics](curated/halfcheetah-nc2-nc3-captions.md).

[CPU/GPU gate results](runs/verification_full_update/verification.json) include each minibatch loss, actor/critic gradients, final parameters, fixed-noise sampling and RNG isolation. Maximum absolute / relative differences: CPU 0 / 0; GPU 0 / 0. CPU requires bitwise equality; GPU tolerance was fixed at `atol=1e-7, rtol=1e-5`. [Flag verification](runs/verification/verification.json) additionally checks the NC4 ratio-one/clip-zero invariant.

R0 produced 0 non-stopping evaluation red flags against the three Stage 1 seeds' same-step mean ± five sample standard deviations; see its [manifest](runs/R0_recovery_seed0/manifest.json). P1/P2 and the previously stopped R0 remain preserved outside condition averages.

Curated artifacts: [evaluation returns](curated/eval_return_vs_env_steps.png), [training returns](curated/train_return_vs_env_steps.png), [diagnostics](curated/diagnostics_vs_iteration.png), and [condition × seed table](curated/condition_seed_results.csv). [Analysis records](runs/final_analysis/analysis_record.json) provide native log/result/checkpoint paths, failures, last diagnostics, per-run and per-condition times, and observed GPU sharing.

Source remains `ab46b150fa34b5a5b457cd4062cd4c5ad830d964` on `stage2-noclip`; [diff](provenance/dppo_changes.patch), [methods](methods.md), [dependencies](provenance/pip_freeze.txt). Scientific settings and Stage 1 outputs are unchanged. The KL diagnostic is a sampled transition estimate, not the exact final-action KL.
NC1 seed 2 used H200 NVL after a passing [saved-batch H200 check](runs/verification_h200/verification.json); other runs used A6000. This approved hardware difference is a comparison limitation. [Authorization](provenance/h200_seed2_authorization.json) and run manifests retain resource details.


**Completion:** the last training process finished October 10 at 12:18 a.m. PDT. All 19 selected runs have 140 rows and all 95 expected checkpoints; monitors are paused. [Completion verification](runs/monitor_checks/20261010T084908_completion.json).

Actual Stage 2 allocation: **45.95 GPU-hours** of the approved 58.5 (78.6%), including probes, verification and interrupted attempts; [accounting](provenance/final_compute_usage.json).

**Curation:** the user approved all full-analysis plots and the table. [Verified copies and captions](curated/INDEX.md#full-stage-2-comparison) retain their sources; earlier curated subsets remain preserved. No further experiments are authorized.
