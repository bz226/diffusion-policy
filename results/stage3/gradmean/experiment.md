# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** Resumed under [Amendment G-1](provenance/amendment_G1.json). The two-regime G2 and explicit Python3.8 cleanup are being implemented. The prior failed G2 is preserved in [its report](provenance/report_before_G1_amendment.md) and [saved results](runs/gate_G2/gate_summary.json); no environment measurements have yet run.

**Scope and evidence.** All five required policy checkpoints exist. The new collector and statistical helpers are implemented in an ordinary source snapshot; full gate verification, scheduling and final analysis remain incomplete. Only the fixed CPU synthetic gate ran. No actor updates, policy preparation or collection occurred; Q1–Q10 remain unmeasured. [Methods](methods.md), [commits](provenance/revisions.json), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json).

**Resources.** CPU job17812088 is terminal after20 seconds on40 CPUs;0 of125 approved GPU-hours used. No collection jobs or monitors were started. Existing jobs and Stage-1/2/3 results were not modified.

**Next step.** Run amended G2, unchanged formal G1/default-equivalence checks, then the extended live cleanup gate. On success, continue the approved collection automatically within125 GPU-hours. Unresolved directions stay open; no automatic curation.

Amended G2 completed: both L/T regimes passed200replicates each; Slurm17812422 exited0:0 with no surviving workers. The saved-batch and live gates are next. See [REPORT.md](REPORT.md).
