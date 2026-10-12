# Monte Carlo returns with a critic under AdamW

**Question.** Does the improvement from without_PPO (GAE + critic) to MC + LOO come from changing the return estimate or the baseline?

**Status.** Completed: all three E1C seeds, saved outputs and the requested analysis are verified. [Candidate report](runs/final_analysis/candidates/REPORT.md), [methods](methods.md), [gate evidence](runs/gate/gate_summary.json). No artifacts were curated automatically.

**Design.** Three seeds (0–2), one 140-iteration run each, using the E1 single-actor-step AdamW configuration at 1e-4. Actor advantages are raw Monte Carlo returns minus the pre-update critic times the reward-scaler divisor saved before the batch. The critic remains untrained initially and learns its usual GAE targets. The user approved matching E1's actual normalization within shuffled 50,000-pair chunks, replacing the addendum's full-batch wording. Existing E0, E1 and DPPO results are references; they will not be rerun.

**Resources and monitoring.** The total cap is 16 GPU-hours, including gates and retries. Each seed gets one A6000, 40 CPU cores and 64 GiB for at most four hours. E0/E1 took 2h20–2h32 per seed, approximately 60–65 seconds per iteration; the estimate is 2h30 per E1C seed before queue delays. A 30-minute monitor follows `max(30 minutes, estimated runtime/6)`. The three E1C seeds take priority; gradient-mean resumes with at most five concurrent workers while they run. The critic-addendum campaign remains paused.

**Next step.** Review the completed candidate figures and tables; select artifacts and names for curation.

<!-- E1C runtime monitor -->
Last monitor: 2026-10-12T10:14:34.001866+00:00. Campaign: **completed**. Seeds complete: 3/3. GPU-hours used: 7.234; reserved: 0.000; cap16. [Current counters](runs/campaign_progress.json). Next: review candidate analysis; curation needs selection.
<!-- end E1C runtime monitor -->
