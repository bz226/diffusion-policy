# Discounting decisions after advantage normalization

Does γ^k change what without_PPO_and_GAE learns, and move improvement toward earlier decisions?

**Status:** All required training and checkpoint evaluations are verified complete.

**Q1 — Discounted objective.** J_disc gain last 10: -6.6956 ± 14.738 (SE), unresolved.

**Q2 — Benchmark returns.** Eval at 9.36M: -7.0478 ± 29.011 (SE), unresolved; Train last 10: -22.058 ± 45.724 (SE), unresolved.

**Q3 — Prediction.** Own-critic c(V2,V6) point estimates (update: cosine) are 0: 0.783; 32: 1.013; 63: 1.015; 95: 1.14; 126: 1.065. The ≥0.9-at-every-checkpoint criterion is not met, so Part A does not give the stated retracing prediction. None of the three primary learning contrasts exceeds the reading-rule threshold. These noisy Euclidean cosines are indicators; normalization and AdamW preconditioning prevent an exact trajectory prediction.

**Q4 — Episode tilt.**
- Checkpoint 35: T without γ^k 0.13395 ± 0.44845; with γ^k -0.33324 ± 0.08369 (seed SD); difference -0.4672 ± 0.26338 (SE), unresolved.
- Checkpoint 70: T without γ^k -0.14831 ± 0.3247; with γ^k -0.14723 ± 0.11788 (seed SD); difference 0.0010842 ± 0.19944 (SE), unresolved.
- Checkpoint 105: T without γ^k -0.18103 ± 0.41176; with γ^k -0.36487 ± 0.14304 (seed SD); difference -0.18384 ± 0.25166 (SE), unresolved.
- Checkpoint 139: T without γ^k -0.39481 ± 0.23751; with γ^k -0.49018 ± 0.10235 (seed SD); difference -0.095376 ± 0.14931 (SE), unresolved.

**Q5 — LOO comparison.** J_disc gain last 10: -43.095 ± 4.4768 (SE), exceeds reading-rule threshold; Eval at 9.36M: -127.62 ± 49.742 (SE), exceeds reading-rule threshold; Train last 10: -122.83 ± 12.878 (SE), exceeds reading-rule threshold. J_disc gain last 10: same nominal direction; at least one contrast is unresolved; Eval at 9.36M: same nominal direction; at least one contrast is unresolved; Train last 10: same nominal direction; at least one contrast is unresolved. The LOO comparison also changes denoising weights, log-prob reduction and normalization, so it is not the same single intervention.

**Other behavior.** Mean KL/update: -6.9101e-05 ± 2.4448e-05 (SE), exceeds reading-rule threshold; Mean consecutive-gradient cosine: 0.0033338 ± 0.002141 (SE), unresolved; Final distance: -0.060949 ± 0.0081634 (SE), exceeds reading-rule threshold; Final pre-update saturation: -0.020275 ± 0.022307 (SE), unresolved.

[Outcome and difference table](e1cg_table.md) · [Episode segments](segments_table.md) · [Seed-0 curves](figs/e1cg_curves_seed0.png) · [Three-seed means](figs/e1cg_curves_3seed.png) · [Final segment gains](figs/e1cg_gain_by_segment.png).

**Verification and provenance.** G1 passed; G2 passed; G3 passed; G4 passed; G5 passed; G6 passed. [Source records](analysis_record.json) preserve every input, source revision, job and gate record. All three seeds are retained. Final checkpoint sampling is separate from the last logged evaluation at 9.36M. Part A records retain full-batch-proxy, fitted-scale and older scaled-return caveats. No new sampling, tuning or modifications to saved results were performed by this analysis.

**Missing evidence:** none.

**Part A.** [Alignment table](../../../alignment_table.md) · [Mean-gradient alignment figure](../../../figs/e1cg_alignment.png) · [Interim interpretation](../../part_a/candidates/INTERIM.md).

**Execution.** Source branch `stage3-e1cg`, revision `03f82611746954839169a64305777c3d7f91b139`; reference revision `b459412cfc272a7b8830c77925271843c27222b2`. Each training allocation requests one A6000 and 40 CPUs; observed nodes: soal-11. Completed allocation time is 7.5294 GPU-hours of the approved 32; the optional condition decision is `skipped_no_three_extra_slots_while_all_primary_running`. Existing jobs were left running.

[Exact commands, seeds and job records](../../plan.json) · [Primary job file](../../../jobs_e1cg.txt) · [Optional job file](../../../jobs_e2c.txt) · [Methods and deviations](../../../methods.md) · [Gate evidence](../../gates/gate_summary.json).

The first gate attempt failed before scientific checks because D4RL's default cache was read-only. The retry used a study-local cache; this was an installation-path fix, with no algorithm change.
