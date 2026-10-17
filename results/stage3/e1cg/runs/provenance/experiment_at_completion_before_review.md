# Discounting decisions after advantage normalization

**Question.** Does weighting decision k by 0.99^k, after the existing advantage normalization, change the discounted objective or rewards learned by without_PPO_and_GAE?

**Status.** Approved; implementation, Part A and all six CPU gates are complete. The campaign was submitted at 2026-10-17 09:20 UTC as Slurm controller job 17922557, using tested revision `03f82611746954839169a64305777c3d7f91b139`. Seeds 0 and 1 are running on soal-11 GPUs 3 and 4 (jobs 17922559/17922560); seed 2 is queued for resources (17922561). Their first training iterations completed in 65.1 and 68.5 seconds with finite diagnostics. The 30-minute monitor is active. [Methods and approved scope](methods.md).

**Existing-data finding.** Own-critic alignment estimates at updates 0/32/63/95/126 are 0.783/1.013/1.015/1.140/1.065. All five 90% intervals include 1; the premise of at least 0.9 alignment at every checkpoint is not established. No resolved common-mean sanity interval excludes 1, while several checks remain unresolved. [Part A table](alignment_table.md) · [figure](figs/e1cg_alignment.png) · [interim interpretation](runs/part_a/candidates/INTERIM.md).

**Design.** Compare three existing seeds with three new seeds (0–2), each for 140 iterations. The sole primary change is the post-normalization decision weight. Raw Monte Carlo returns, the pre-update critic and reward divisor, per-50,000-pair normalization, denoising weights, and one AdamW step at 1e-4 remain fixed. E2C is approved as an optional three-seed condition only if capacity allows after the primary seeds start. Existing E1/E2 results are reference rows.

**Resources.** The user approved 32 GPU-hours on 2026-10-17, including optional E2C, checkpoint evaluation and bounded same-seed retries. Training has a four-hour limit per seed on one A6000 with 40 CPUs. The analogous completed runs took 8,756 / 8,513 / 8,774 seconds, so the estimate is 2h22–2h26 per seed, plus queue delays. Monitoring starts with the campaign and checks every 30 minutes, following max(30 minutes, estimated run time / 6). CPU gates are capped at one hour. No tuning or additional samples are authorized.

**Provenance and isolation.** Branch `stage3-e1cg` starts at `b459412cfc272a7b8830c77925271843c27222b2`, the exact source revision used by the reference runs. Work is isolated in `.codex-runtime/worktrees/stage3-e1cg`; this study's source is an ordinary snapshot, with no nested Git repository. Existing source trees, results and running jobs remain untouched.

**Interpretation.** Part A reports Euclidean mean-gradient alignment, not AdamW's preconditioned direction. Older warmed-critic cohorts use a different scaled-return proxy; they are reported separately. Unresolved signals or failed common-mean checks prevent interpretation at the affected point. The approved plot may extend above 1 to show noisy ratio estimates without clipping; the nine-group cohort uses t₈.

**Verification.** G1–G6 and optional E2C C1 passed. Default replay, unit time weights and segment-output checks were bitwise identical; the largest coefficient relative error was 2.31e−7. The saved NC4 fixture used for earlier E1C gates was reused, rather than a completed E1C rollout. [Gate evidence](runs/gates/gate_summary.json).

**Next step.** Monitor the three admitted seeds, then perform the specified checkpoint evaluations and analysis. The optional condition is admitted only if the approved free-capacity rule is met. Artifacts remain candidates until selected for curation.

<!-- E1CG runtime monitor -->
**Execution status (2026-10-17T14:25:06.404817+00:00):** complete. 28 jobs verified complete.

GPU time: 7.529 hours completed; 0.000 hours reserved; 32-hour cap. Active or queued: none.

Next: review completed measurements and generate the approved final analysis.

[Live status](runs/status.json) · [Execution record](runs/campaign/campaign.json).
<!-- end E1CG runtime monitor -->
