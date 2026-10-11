# FRS addendum: gate passed, training awaiting source commit

**Question.** Does frozen per-coordinate gradient scaling recover E2_beta0's progress and stability?

**Answer.** The implementation passed all CPU/A6000 gates. Training has not launched: the parent `.git` directory is read-only in this session, preventing the required `stage3-theory` branch and commit. Thus comparisons of eval return, training return, discounted J, KL and parameter distance, and the assessment of training stability, are not yet available.

**Verification.** NC4 AdamW, E2 AdamW and E2 SGD fixed-batch replays matched the existing Stage-3 source bitwise on CPU and GPU: maximum absolute and relative differences were zero. This covers minibatch losses, actor/critic gradients, updated parameters and optimizer state, logged values, fixed-noise sampling and RNG states. FRS diagnostics on/off, denominator-one equivalence to SGD, and repeated cached scale conversion also matched bitwise. Denominators stayed fixed; no deterministic-algorithm settings were enabled.

The independent float64 formula comparison uses relative L2 error of the updated parameter vector, with a 1e-7 threshold. Coordinatewise relative errors and float32 displacement roundoff are reported separately, not tested against that threshold.

| Device | Maximum absolute error | Maximum coordinatewise relative error | Relative L2 error |
|---|---:|---:|---:|
| CPU | 3.00436e-6 | 7.57413e-4 | 2.19636e-8 |
| A6000 | 3.00401e-6 | 1.96656e-3 | 2.19629e-8 |

[Full gate evidence](runs/gate/gate_summary.json) includes displacement errors and every comparison. The full training update was bitwise identical to the directly applied optimizer step.

**Frozen scales.** These use the permitted reuse route: 25 saved pretrained-policy V4 batch sums divided by 40, measurement seed 1000, processing seed 4000. Repeatability refers to cached conversion, not new same-seed rollouts. Per-environment float32 gradients summed in float64 have the E2 estimator and units but a different accumulation order from shuffled training chunks.

| Raw r quantile | Value |
|---|---:|
| Minimum | 0 |
| 1% | 0 |
| 10% | 1.45295162e-5 |
| 50% | 6.26695415e-4 |
| 90% | 1.04395899e-2 |
| 99% | 9.36042468e-2 |
| Maximum | 2.20775439 |

Of 568,392 coordinates, **47,798 (8.41%) are exactly zero** and **49,850 (8.77%) are raised to the floor** before epsilon. The largest 5,684 coordinates (top 1%, rounded up) carry **79.0165% of total r²**. The pre-epsilon floor is 6.26695415e-7; epsilon is 1e-8. [Statistics and definitions](runs/gate/rescale_seed4000_A/native/rescale_stats.json).

**Setup and next step.** FRS1 uses the unchanged E2 estimator and constant LR 1e-4, seeds 0–2, 140 iterations, followed by final-checkpoint evaluations. FRS3 is excluded. The approved cap is 14 GPU-hours; gate job 17806033 completed with exit 0 after 1,123 allocated GPU-seconds (0.31194 GPU-hours). All 279 YAML files are unchanged; all 403 source files are ordinary, versionable files; generated tensors/checkpoints remain ignored.

The imported parent revision is `c4f3c465e56af756e162dece7656fac9b8e97efa`; the original Stage-3 scientific revision was `1515be0411a78df94e940de22df45834edadd858`. No new commit exists yet. [Source diff](runs/gate/source_changes.diff), [methods](methods.md), [approval](provenance/approval.json), [exact launch sequence](jobs.txt).

From the project root in a normal terminal:

```bash
git switch -c stage3-theory
python3 scripts/git_add_all.py
git commit -m "stage3: add frozen-rescaled SGD condition"
```

Source binding will verify that the committed files equal the passed gate snapshot before launching. No training results or curated artifacts are claimed.
