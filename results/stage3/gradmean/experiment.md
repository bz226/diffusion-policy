# Mean gradient and noise along DPPO and NC4 paths

**Question.** How do the discounted-objective gradient, its noise and alignment with training movement change along the DPPO and NC4 paths?

**Status.** Amendment G-1 is implemented and committed at `f425082`. Both amended G2 regimes passed 200 replicates, including weak-signal N/A handling and clean Python 3.8 worker shutdown. [Results](REPORT.md) and [gate evidence](runs/gate_G2_G1/gate_summary.json). The saved-batch G1/G5 gate passed on CPU and GPU with zero replay differences (Slurm 17812451, 629 GPU-seconds). P0 preparation is now admitted; G3/G4 live checks follow. No production measurements have begun.

**Scope.** All five required checkpoints exist. The scientific source is an ordinary snapshot, with no changes to earlier studies. This study does not update the actor. Q1–Q10 remain unmeasured. [Methods](methods.md), [amendment](provenance/amendment_G1.json), [revisions](provenance/revisions.json), [source provenance](../../stage3_gradmean/provenance/dppo_source_snapshot.json). The [original failed G2 evidence](provenance/report_before_G1_amendment.md) is preserved.

**Resources and monitoring.** The hard cap is 125 GPU-hours, counting actual usage and outstanding reservations. G1/G5 used 629 GPU-seconds. P0 preparation has one A6000 reserved for at most one hour. Gate progress is checked during execution; subsequent one-hour jobs have 30-minute monitors and the 12–16 hour campaign has a 2 hour 40 minute monitor, following `max(30 minutes, estimate/6)`. Each job has its own enforced cap and explicit child cleanup.

**Next step.** Finish P0 preparation and run two independent 50-batch groups for G3/G4 and the pilot. If all gates pass, continue the approved collection automatically. Unresolved directions stay open. No automatic curation.

<!-- gradmean-bootstrap-start -->
Gate transition monitor: 2026-10-11T20:50:51.460516+00:00. **running**; phase: await_offline_G1_G2_G5. Monitor active.
<!-- gradmean-bootstrap-end -->

<!-- gradmean-status-start -->
Last monitor: 2026-10-11T20:55:25.696536+00:00. **live_gate_running**. P0: open, 0 batches; D1: open, 0 batches; D2: open, 0 batches; N1: open, 0 batches; N2: open, 0 batches. GPU hours used: 0.175; outstanding reserved: 1.000; cap:125. Monitor active. Next: Complete admitted groups and evaluate the next approved round boundary.
<!-- gradmean-status-end -->
