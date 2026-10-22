# DPPO actor epochs at fixed and matched learning rates

**Question.** Does DPPO's reuse improve sample efficiency beyond the first-order movement obtained by increasing the learning rate with fewer actor epochs?

**Status: complete; all four seed-0 runs, final analysis and saved outputs are verified.** The monitor is paused. Training used 8.524 GPU-hours including the live gate, within the 20-hour cap. CPU default and critic/RNG comparisons were bitwise identical; all live cases took the required four/eight actor steps and twenty critic steps. The actor takes four or eight AdamW steps per fresh batch; all twenty critic steps, losses, backward passes and shuffles remain. The baseline's KL stopping rule remains enabled.

| Actor epochs | Actor lr | Actor steps per batch | First-order movement / DPPO |
| --- | --- | --- | --- |
| 1 | 1e-4 | 4 | 0.2 |
| 1 | 5e-4 | 4 | 1 |
| 2 | 1e-4 | 8 | 0.4 |
| 2 | 2.5e-4 | 8 | 1 |

The approved cap is 20 GPU-hours, with at most four concurrent A6000 allocations, 40 CPUs and 64 GiB each. A training attempt is capped at four hours; live gates at one GPU-hour; CPU gates at two hours wall time. Only diagnosed same-seed retries that fit the remaining cap are permitted. Matched-lr conditions launch first. No reference reruns, additional seeds or checkpoint evaluations are authorized.

The four runs took 2h05m38s–2h08m03s each. Monitoring checked every 30 minutes, based on the approved 2h30m planning estimate. Training ended at 21:09 PDT on October 21 and final analysis at 21:12 PDT. No external job or source tree was changed.

Historical DPPO and without_PPO records lack J_disc trajectories and some clipping diagnostics. Missing quantities remain “not logged.” Thus their distance-versus-J_disc and discounted-gain contrasts cannot be recovered from checkpoints alone. The reading threshold is twice Stage-1 DPPO's three-seed sample SD for the same metric, wherever that metric exists.

[Methods](methods.md) · [Approved scope](provenance/approval.json) · [Imported source](provenance/dppo_source_snapshot.json). The user-selected report, figures, tables and methods are saved in the [curated index](curated/INDEX.md).

<!-- EP gate monitor -->
**Gates passed.** The original controller stopped before training when its report-combining subprocess exceeded 120 seconds during startup. All gate outputs remained passed. A lightweight finalizer combined those records without repeating scientific work; source contents and resource caps were reverified. The replacement training controller, job **18011538**, completed successfully; its 30-minute monitor is now paused.
[Gate results](runs/gates/complete/gate_summary.json) · [Recovery record](provenance/launch_recovery.json) · [Original failure](runs/gate_continuation/state.json).
<!-- end EP gate monitor -->

<!-- EP runtime monitor -->
**Execution status (2026-10-22T04:12:03.518833+00:00):** complete; 4/4 jobs verified complete in training.

GPU time: 8.524 hours used; 0.000 hours reserved; 20-hour combined cap. Active or queued: none.

[Live status](runs/status.json) · [Execution record](runs/campaign_training/campaign.json).
<!-- end EP runtime monitor -->

<!-- EP manual monitor -->
**Completion verified (2026-10-22T04:30:26.721870+00:00).** All four runs saved 140 iterations and checkpoints 0, 35, 70, 105 and 139, with successful exits. No nonfinite measurements, KL early stops or evaluation-collapse flags were recorded. The final report, tables and all three PNG/PDF figure sets are available.

| Method | Evaluation at 9.36M |
| --- | ---: |
| DPPO, 1 epoch | 4517.20 |
| DPPO, 1 epoch, lr 5e-4 | 4608.76 |
| DPPO, 2 epochs | 4666.50 |
| DPPO, 2 epochs, lr 2.5e-4 | 4711.69 |

The matched 1-epoch run is 27.51 below DPPO’s seed-0 final evaluation; that difference is unresolved under the specified seed-SD yardstick. This single-seed comparison is compatible with the movement-only explanation but does not establish equivalence. Historical missing J_disc prevents the full discounted-objective comparison.

[Final report](REPORT.md) · [Table](epochs_table.md) · [Figures](figs/) · [Monitoring record](runs/manual_monitor_history.jsonl). The selected outputs are saved in the [curated index](curated/INDEX.md).
<!-- end EP manual monitor -->
