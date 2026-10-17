# Discounting decisions after advantage normalization

Does γ^k change what without_PPO_and_GAE learns, and move improvement toward earlier decisions?

**Status:** Partial analysis; incomplete evidence is listed explicitly and no final conclusion is made.

**Q1 — Discounted objective.** J_disc gain last 10: N/A ± N/A (SE), unavailable.

**Q2 — Benchmark returns.** Eval at 9.36M: N/A ± N/A (SE), unavailable; Train last 10: N/A ± N/A (SE), unavailable.

**Q3 — Prediction.** Own-critic c(V2,V6) point estimates (update: cosine) are 0: 0.783; 32: 1.013; 63: 1.015; 95: 1.14; 126: 1.065. The ≥0.9-at-every-checkpoint criterion is not met, so Part A does not give the stated retracing prediction. These noisy Euclidean cosines are indicators; normalization and AdamW preconditioning prevent an exact trajectory prediction.

**Q4 — Episode tilt.**
- Checkpoint 35: pending/incomplete.
- Checkpoint 70: pending/incomplete.
- Checkpoint 105: pending/incomplete.
- Checkpoint 139: pending/incomplete.

**Q5 — LOO comparison.** J_disc gain last 10: -43.095 ± 4.4768 (SE), exceeds reading-rule threshold; Eval at 9.36M: -127.62 ± 49.742 (SE), exceeds reading-rule threshold; Train last 10: -122.83 ± 12.878 (SE), exceeds reading-rule threshold. J_disc gain last 10: primary comparison pending; Eval at 9.36M: primary comparison pending; Train last 10: primary comparison pending. The LOO comparison also changes denoising weights, log-prob reduction and normalization, so it is not the same single intervention.

**Other behavior.** Mean KL/update: N/A ± N/A (SE), unavailable; Mean consecutive-gradient cosine: N/A ± N/A (SE), unavailable; Final distance: N/A ± N/A (SE), unavailable; Final pre-update saturation: N/A ± N/A (SE), unavailable.

[Outcome and difference table](e1cg_table.md) · [Episode segments](segments_table.md) · [Seed-0 curves](figs/e1cg_curves_seed0.png) · [Three-seed means](figs/e1cg_curves_3seed.png) · [Final segment gains](figs/e1cg_gain_by_segment.png).

**Verification and provenance.** G1 passed; G2 passed; G3 passed; G4 passed; G5 passed; G6 passed. [Source records](analysis_record.json) preserve every input, source revision, job and gate record. All three seeds are retained. Final checkpoint sampling is separate from the last logged evaluation at 9.36M. Part A records retain full-batch-proxy, fitted-scale and older scaled-return caveats. No new sampling, tuning or modifications to saved results were performed by this analysis.

**Missing evidence:** E1C seed0 final checkpoint evaluation; E1C seed1 final checkpoint evaluation; E1C seed2 final checkpoint evaluation; E1CG seed0 training partial; E1CG seed0 final checkpoint evaluation; E1CG seed1 training partial; E1CG seed1 final checkpoint evaluation; E1CG seed2 training missing; E1CG seed2 final checkpoint evaluation; Part C segment evaluations incomplete (0/24).

**Part A.** [Alignment table](../../../alignment_table.md) · [Mean-gradient alignment figure](../../../figs/e1cg_alignment.png) · [Interim interpretation](../../part_a/candidates/INTERIM.md).

**Execution.** Source branch `stage3-e1cg`, revision `03f82611746954839169a64305777c3d7f91b139`; reference revision `b459412cfc272a7b8830c77925271843c27222b2`. Each training allocation requests one A6000 and 40 CPUs; observed nodes: soal-11. Completed allocation time is 0 GPU-hours of the approved 32; the optional condition decision is `waiting_for_primary_running_and_three_free_slots`. Existing jobs were left running.

[Exact commands, seeds and job records](../../plan.json) · [Primary job file](../../../jobs_e1cg.txt) · [Optional job file](../../../jobs_e2c.txt) · [Methods and deviations](../../../methods.md) · [Gate evidence](../../gates/gate_summary.json).

The first gate attempt failed before scientific checks because D4RL's default cache was read-only. The retry used a study-local cache; this was an installation-path fix, with no algorithm change.
