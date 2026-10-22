# Gradient noise of without_PPO

**Status: all five gates passed; collection is running.** The GAE and Monte Carlo gradient checks and the full default-update replay are bitwise identical to their references. All checkpoint critics are unchanged; the live seed-independence and cleanup checks passed. Initial production batches have ratio 1 and clip fraction 0. See the [CPU gate record](runs/gates/cpu/gate_summary.json) and [live gate record](runs/gates/live/G3_G4_live/worker.json).

Does replacing GAE with a Monte Carlo advantage change the mean direction or noise of the gradient at without_PPO's own parameters? This measurement compares the two gradients on identical fresh batches, and compares the native gradient with existing measurements at other runs' parameters.

The original seed-0 run has all five checkpoints: iterations 0, 35, 70, 105 and 139 contain 0, 32, 63, 95 and 126 actor updates. Each point will use ten independent jobs of 50 batches, with 80 episodes per batch. The actor and critic remain fixed. Results are conditional on one training seed and the supplied critic and reward scale.

**Approved corrections.** Both gradients use the run's actual normalization within each shuffled 50,000-pair chunk. The initial reward-scaler state is known and is updated with each fresh batch as in training. Later checkpoints lack historical scaler state and logged divisors: their scale will be fitted on five separate batches and then frozen. Those measurements are explicitly scaler approximations. Exact saved-batch gradient gates do not establish that the missing historical state has been recovered. The older green comparison used an approved full-batch normalization approximation.

The approved budget is 50 collection jobs, at most five concurrent GPUs, 30 minutes per collection job and **30 GPU-hours overall**, including gates, preparation and bounded retries after a documented fix. The live probes and initial production batches take about 31–34 seconds each: allow **4.4–5 hours of collection with five running GPUs**, plus scheduler delays. This supersedes the preliminary 2–3-hour estimate; the compute and per-job caps are unchanged. The explicitly approved monitor checks every 30 minutes and pauses after verified completion or a terminal failure. Gate failure stops admissions; existing unrelated jobs are never changed. Timing evidence is in [runtime estimates](runs/runtime_estimates.jsonl).

Next: complete the 50 approved collection jobs, verify their saved statistics, and generate the requested analysis. The [methods](methods.md) specify statistics, state limitations and files. Deliverables are three candidate figure sets, a table and the requested Q1–Q3 report. Curation requires a separate decision.

<!-- noise monitor -->
**Execution status (2026-10-20T00:30:32.573993+00:00):** complete; 50/50 collection jobs verified.

Completed allocations: 22.198 GPU-hours; reserved: 0.000; cap:30. Monitor:paused.

[Current execution record](runs/campaign/state.json). All50groups and candidate analysis verified. Curation awaits a separate decision.
<!-- end noise monitor -->
