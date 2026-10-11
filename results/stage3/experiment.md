# Stage 3: score estimators and projected SGD

**Status: campaign running.** All 27 Wave-1 training runs are complete and analyzed. BATCH4 seeds 0–2 remain active, with final checkpoint evaluations still pending. No completed Wave-1 run collapsed, produced non-finite values or had a zero actor update. The live monitor record below tracks the remaining jobs.

Does replacing NC4's estimator and AdamW with the specified finite-horizon score estimator, SGD and a Euclidean projection improve the discounted objective, and is progress limited by gradient noise?

**Completed Wave-1 analysis.** E1 has the highest mean final scheduled eval, **4,670 ± 28** across three seeds. SGD10 reaches **4,458 ± 106**; its differences from SGD3 and E2 remain unresolved under the prescribed pooled-SD rule. Removing LOO lowers returns, while removing Adam momentum leaves returns unresolved despite 14.7× larger median KL. [Approved expanded analysis, figures and tables](curated/INDEX.md#completed-wave-1-comparison). Split-noise estimates are undefined in 207/378 windows, preventing a reliable noise ranking.

The approved campaign includes E0, E1, E2, E2_beta0, E2_nobase, SGD1, SGD3, SGD10 and SGD03, three independent seeds each. PROJ3 and PROJ10 follow the specified skip rule. BATCH4 is included. All normal training runs use 140 iterations and the existing 40 environments; BATCH4 changes only the requested rollout length. Final checkpoints are evaluated with both samplers. Two fixed-policy noise measurements and five calibration batches precede training.

The measured anchors are **KL* = 0.000584517336086918** and **R = 8.686025494700294**. Calibration gives **η* = 0.00012107235674859004**. Its five batch estimates span **3.40×**, exceeding the specified 3× warning threshold; execution continues with the median as prescribed. [Calibration](cal_seed3000/SUMMARY.md), [anchors](provenance/anchors.json). No Stage-2 snapshot commit was needed; Stage-3 uses branch `stage3-theory`.

Fresh DPPO and NC4 batches were captured at the unchanged Stage-2 tip. All requested gates passed: default updates and sampling matched bitwise on CPU and A6000 GPU (maximum absolute and relative differences 0); diagnostics preserved update state and global RNGs bitwise. MC, LOO, discount, gradient, SGD, projection and synthetic noise checks passed. The tested code is committed as `1515be0411a78df94e940de22df45834edadd858`. [Gate evidence](runs/gate/gate_summary.json), [source provenance](provenance/source_commit.json).

The user approved a **200 allocated GPU-hour maximum**. Per-job caps are 4 hours for normal training, 16 hours for BATCH4, 3 hours for each noise measurement, 1 hour for calibration, 15 minutes for each final checkpoint evaluation and 30 minutes for the pretrained evaluation. Capture and gates together have a one-hour cap. There will be at most nine separate A6000 GPU allocations, each with 40 CPUs and 64 GiB RAM; actual concurrency depends on Slurm availability.

The central duration estimates are 3 hours for normal training (range 2.8–3.4) and 12 hours for BATCH4 (range 11–14), based on NC4's observed 2h21m–2h25m plus new diagnostics and BATCH4's fourfold sample count. Their scheduled monitor intervals are 30 minutes and 2 hours, respectively, using the central estimate divided by six. Measurements use 30-minute checks. Monitoring records progress and stops after verified terminal status; it never authorizes retries or configuration changes.

**Phase 1 completed.** Both policies have 25 measured batches. [Curated analysis and plots](curated/INDEX.md#phase-1-fixed-policy-gradient-noise) show provisional point estimates only: V5’s B_ep is 114,292 at θ₀ and 24,396 at the NC4 checkpoint. Rankings and policy changes remain inconclusive because percentile-bootstrap intervals are unreliable, two signal estimates are negative, and seven direction estimates exceed cosine bounds. [Audit](runs/noise_ci_review.json). Training-seed SDs are unaffected; no replacement uncertainty method was run.

Next: finish BATCH4, then final checkpoint evaluations and analysis. PROJ3/PROJ10 were skipped as prescribed: maximum SGD distances 0.0888/0.2571 are below 6.9488. [Methods](methods.md). Wave-1 artifacts A–D are curated under the approved names; earlier curated artifacts remain unchanged.

<!-- stage3-status-start -->
Last monitor: 2026-10-11T00:51:37.010663+00:00; campaign **running**. Run states: complete 30, running 3, skipped 6. [Live record](runs/campaign_progress.json); [command journal](runs/campaign_commands.jsonl).
<!-- stage3-status-end -->
