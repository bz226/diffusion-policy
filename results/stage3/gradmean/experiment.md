# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** Stopped at G2 before environment measurements: one of200 synthetic draws gave a negative split-half correlation, so its direction-cosine estimate was undefined. The signal, noise trace, interval coverage and effective-rank checks passed. [Failure report](REPORT.md), [complete gate results](runs/gate_G2/gate_summary.json).

**Scope and evidence.** All five required policy checkpoints exist. The new collector and statistical helpers are implemented in an ordinary source snapshot; full gate verification, scheduling and final analysis remain incomplete. Only the fixed CPU synthetic gate ran. No actor updates, policy preparation or collection occurred; Q1–Q10 remain unmeasured. [Methods](methods.md), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json).

**Resources.** CPU job17812088 is terminal after20 seconds on40 CPUs;0 of125 approved GPU-hours used. No collection jobs or monitors were started. Existing jobs and Stage-1/2/3 results were not modified.

**Next decision.** The scientific gate and its undefined-value policy remain unchanged. Any revised gate or resumed collection requires explicit authorization. All failed evidence is retained, without automatic curation.
