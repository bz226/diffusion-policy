# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** Amendment G-1 is implemented and committed at `f425082`. Both amended G2 regimes passed 200 replicates, including weak-signal N/A handling and clean Python 3.8 worker shutdown. [Results](REPORT.md) and [gate evidence](runs/gate_G2_G1/gate_summary.json). The saved-batch G1/G5 gate is running as Slurm 17812451; G3/G4 live checks follow only if it passes. No production measurements have begun.

**Scope.** All five required checkpoints exist. The scientific source is an ordinary snapshot, with no changes to earlier studies. This study does not update the actor. Q1–Q10 remain unmeasured. [Methods](methods.md), [amendment](provenance/amendment_G1.json), [revisions](provenance/revisions.json), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json). The [original failed G2 evidence](provenance/report_before_G1_amendment.md) is preserved.

**Resources and monitoring.** The hard cap is 125 GPU-hours, counting actual usage and outstanding reservations. G1/G5 has one A6000 reserved for at most 2 hours; collection has not yet consumed GPU time. Gate progress is checked during execution; subsequent one-hour jobs have 30-minute monitors and the 12–16 hour campaign has a 2 hour 40 minute monitor, following `max(30 minutes, estimate/6)`. Each job has its own enforced cap and explicit child cleanup.

**Next step.** Finish the unchanged G1/default-equivalence check, then prepare P0 and run two independent 50-batch groups for G3/G4 and the pilot. If all gates pass, continue the approved collection automatically. Unresolved directions stay open. No automatic curation.
