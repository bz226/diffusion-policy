# DPPO with Monte Carlo critic advantages

**Question.** Does replacing GAE with raw Monte Carlo return minus the critic improve DPPO when its reuse and clipping mechanisms remain enabled?

**Status: completed.** Three training seeds finished all 140 iterations and 126 fresh-data updates; the final read-only comparison is complete. without_GAE evaluation at 9.36M is 4621 ± 18.559 and last-ten-update training return is 4295.7 ± 81.068 (mean ± sample SD over seeds). [Report](REPORT.md) · [Results table](ladder2x2_table.md).

The approved budget is 16 GPU-hours, at most four hours per training attempt, one GPU-hour for gates and two hours aggregate CPU-gate wall time. Seeds 0–2 started in parallel at 02:41 PDT on October 19, on soal-10 GPUs 0, 1 and 2 (jobs 17959932, 17959933 and 17959934). Each has one A6000, 40 CPUs and 64GiB. Recent DPPO runs took about 2h03m; the planning estimate is 2h15–2h30 per seed. Monitoring is every 30 minutes: max(30 minutes, 2h30m/6). Other jobs retain their allocations and queue priority.

Historical DPPO and without_PPO records lack discounted-return trajectories and saturation. Those entries remain “not logged”; no extra reference sampling is authorized. Checkpoint parameters permit distance measurements, but cannot recover missing rewards. Contrasts measure the entire PPO reuse/clipping bundle, not reuse alone.

**Verification.** The resolved config differs from DPPO only by the two approved flags. MC advantages match the explicit formula to relative L2 error 6.01×10⁻¹⁶. Default replay, unchanged critic targets/step and diagnostics on/off comparisons passed bitwise on CPU. The live gate executed 20 actor and critic steps of 50,000 pairs, prepared advantages once, and showed ratio movement on steps 2–20. All environment workers exited cleanly. [Gate results](runs/gates/complete/gate_summary.json).

**Curation.** The user approved all five figure sets, the report, results table and contrasts. [Curated selection and captions](curated/INDEX.md). Original results and checkpoints are preserved; no additional sampling is authorized.

[Methods](methods.md) · [Approval and caps](provenance/approval.json) · [Imported source](provenance/dppo_source_snapshot.json).

<!-- DL1 gate monitor -->
**Gate status (2026-10-19T11:48:24.312427+00:00):** complete. Training and its requested final analysis finished; see campaign status.
[Continuation record](runs/gate_continuation/state.json).
<!-- end DL1 gate monitor -->

<!-- DL1 runtime monitor -->
**Execution status (2026-10-19T11:48:24.257936+00:00):** complete; 3/3 jobs verified complete in training.

GPU time: 6.294 hours used; 0.000 hours reserved; 16-hour combined cap. Active or queued: none.

[Live status](runs/status.json) · [Execution record](runs/campaign_training/campaign.json).
<!-- end DL1 runtime monitor -->
