# Stage 3: score estimators and projected SGD

**Status: campaign running.** At 11:10 PDT on October 10, 17 training runs are complete, six are progressing and one is pending a GPU allocation. Calibration and both fixed-policy noise measurements are complete. No crash, non-finite result or evaluation-collapse threshold crossing has been observed. [Manual check](runs/manual_monitor_20261010T181035Z.json).

Does replacing NC4's estimator and AdamW with the specified finite-horizon score estimator, SGD and a Euclidean projection improve the discounted objective, and is progress limited by gradient noise?

**Completed-five analysis.** E1 has the highest final scheduled eval among E0/E1/E2/SGD1/SGD3: **4,670 ± 28** across three seeds. E2 lowers both discounted and undiscounted training return relative to E1; SGD3 improves training return over SGD1, while their eval difference is unresolved. [Approved curated analysis, figures and tables](curated/INDEX.md). Final-checkpoint evaluations remain pending.

The approved campaign includes E0, E1, E2, E2_beta0, E2_nobase, SGD1, SGD3, SGD10 and SGD03, three independent seeds each. PROJ3 and PROJ10 follow the specified skip rule. BATCH4 is included. All normal training runs use 140 iterations and the existing 40 environments; BATCH4 changes only the requested rollout length. Final checkpoints are evaluated with both samplers. Two fixed-policy noise measurements and five calibration batches precede training.

The measured anchors are **KL* = 0.000584517336086918** and **R = 8.686025494700294**. Calibration gives **η* = 0.00012107235674859004**. Its five batch estimates span **3.40×**, exceeding the specified 3× warning threshold; execution continues with the median as prescribed. [Calibration](cal_seed3000/SUMMARY.md), [anchors](provenance/anchors.json). No Stage-2 snapshot commit was needed; Stage-3 uses branch `stage3-theory`.

Fresh DPPO and NC4 batches were captured at the unchanged Stage-2 tip. All requested gates passed: default updates and sampling matched bitwise on CPU and A6000 GPU (maximum absolute and relative differences 0); diagnostics preserved update state and global RNGs bitwise. MC, LOO, discount, gradient, SGD, projection and synthetic noise checks passed. The tested code is committed as `1515be0411a78df94e940de22df45834edadd858`. [Gate evidence](runs/gate/gate_summary.json), [source provenance](provenance/source_commit.json).

The user approved a **200 allocated GPU-hour maximum**. Per-job caps are 4 hours for normal training, 16 hours for BATCH4, 3 hours for each noise measurement, 1 hour for calibration, 15 minutes for each final checkpoint evaluation and 30 minutes for the pretrained evaluation. Capture and gates together have a one-hour cap. There will be at most nine separate A6000 GPU allocations, each with 40 CPUs and 64 GiB RAM; actual concurrency depends on Slurm availability.

The central duration estimates are 3 hours for normal training (range 2.8–3.4) and 12 hours for BATCH4 (range 11–14), based on NC4's observed 2h21m–2h25m plus new diagnostics and BATCH4's fourfold sample count. Their scheduled monitor intervals are 30 minutes and 2 hours, respectively, using the central estimate divided by six. Measurements use 30-minute checks. Monitoring records progress and stops after verified terminal status; it never authorizes retries or configuration changes.

**Phase-1 limitation.** Review found an upward bias in the percentile bootstrap distribution of squared signal near zero. Its confidence intervals and derived uncertainty claims are unreliable; raw measurements remain intact. [Audit](runs/noise_ci_review.json). Training comparisons use sample SD across seeds and are unaffected. No replacement uncertainty estimator has been run.

Next: finish remaining wave-1 seeds, apply the projection skip rule, run BATCH4 and final checkpoint evaluations. Full setup: [methods](methods.md). Final deliverables will be in `REPORT.md`, `all_runs_long.csv` and `figs/`. The five-condition interim artifacts are curated; future artifacts require a separate decision.

<!-- stage3-status-start -->
Last monitor: 2026-10-10T18:19:47.620403+00:00; campaign **running**. Run states: complete 20, planned 4, running 6. [Live record](runs/campaign_progress.json); [command journal](runs/campaign_commands.jsonl).
<!-- stage3-status-end -->
