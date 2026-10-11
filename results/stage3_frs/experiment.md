# FRS: SGD with frozen coordinate scales

**Question.** Does a fixed diagonal rescaling recover E2_beta0's progress and stability without adapting the scales during training?

**Status.** All CPU/GPU gates passed; Slurm job 17806033 completed successfully in 18m43s. Source commit `b7ac3a1` is on `stage3-theory`. The Git executable-permission comparison is corrected and its 15 mechanical tests pass; final source binding is pending. All scientific source contents still match the gate. No FRS training jobs have started. [Gate results and scale measurements](REPORT.md).

**Design.** FRS1 uses the unchanged E2 estimator, one SGD step per fresh batch, learning rate 1e-4 and fixed denominators `max(r, 1e-3 × median(r)) + 1e-8`. The allowed reuse route estimates `r` from the 25 saved pretrained-policy V4 batch sums divided by 40; there is no new sampling for scales. Seeds 0–2 run in parallel for 140 iterations, followed by final-checkpoint evaluations. References are the completed E2, E2_beta0 and SGD10 runs. [Methods](methods.md), [approval](provenance/approval.json), [source snapshot](provenance/dppo_source_snapshot.json).

**Boundary.** Cached-input reproducibility verifies scale computation, not fresh same-seed trajectories. The saved V4 sums have the E2 estimator and units, with a different floating-point accumulation order from training. Prior Phase-1 confidence intervals are not used. New artifacts will be candidates until separately approved for curation.

<!-- frs-status-start -->
Gate monitoring finished after successful terminal verification on 2026-10-11. Training has not started; next action is verified source binding and launch. Used: 0.31194 GPU-hours of 14. Training cap: four hours/seed; final evaluation cap: 15 minutes/seed; no automatic retries. Estimated training duration is 2h15m from E2_beta0/SGD10 runs taking 2.00–2.07 hours on A6000/40 CPUs; the prepared scheduled monitor runs every 30 minutes, `max(30m, 135m/6)`.
<!-- frs-status-end -->
