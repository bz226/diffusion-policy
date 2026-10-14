# DPPO per-step gradient noise

All measurements are complete. The original-run first-step signal at update 95 is unresolved, so its requested log-scale figure cannot be drawn. The remaining figures and full table were generated; the controller reported this plotting limitation and paused monitoring. No scientific run crashed.

This experiment measures the gradient noise DPPO passes to AdamW across 20 reused-batch steps, at policies containing 32, 63 and 95 updates. Each of the six measurement points (three original-run points and three full-state rerun points) has 10 independent jobs ×50 fresh batches, with 80 episodes per batch.

At original-run update 95, the unbiased squared-mean-gradient estimate is −3.33×10⁻⁸, with 90% interval [−8.48×10⁻⁸,1.83×10⁻⁸]. A negative estimate is a sampling outcome: the true squared norm is nonnegative, but these 500 batches do not resolve it. Its square root and relative-noise ratio are therefore unavailable. Independent recomputation from the saved group sums confirmed the result. At update 63 the point estimate is positive, but its interval also crosses zero.

All 30 Part-A and 30 Part-C jobs completed. Every Part-C batch reached all 20 PPO steps; KL early stopping never fired. The rerun completed 140 iterations with final evaluation 4,708, within the prescribed sanity range. R1 passed; R2 reproduced post-update actor parameters, optimizer states and RNG bitwise, and S1 matched first-step gradients exactly on both checked batches. All five full training-state saves are retained.

| Output | Status and limitation |
|---|---|
| [Per-step gradient noise](curated/dppo_per_step_noise.png) ([PDF](curated/dppo_per_step_noise.pdf)) | Ready; point estimates from one training seed. Some early-step signal intervals include zero. |
| [Pairs switched off by clipping](curated/dppo_per_step_inactive.png) ([PDF](curated/dppo_per_step_inactive.pdf)) | Ready; all 20 steps observed in every batch. |
| [Full table](curated/ppo_steps_table.md) ([CSV](curated/ppo_steps_table.csv)) | Ready; 63 rows, 500 batches per row, 90% delete-one-group t₉ intervals. |
| Original-run first-step figure | Omitted because update 95 has a nonpositive squared-signal estimate. |

See the [analysis note](curated/NOTE.md), [methods](methods.md), [exact job plan](plan.json), and [verification record](runs/final_review.json). Original checkpoints lack reward-scaler state, so Part A uses the authorized five-batch scale fit and is approximate; Part C restores the rerun's full scaler and AdamW states. All plots are conditional on the measured checkpoints. The per-step noise figure needs a shared 3.5-decade span to include its data.

Total GPU use was 57.38 hours of the approved 80. Collection and analysis stopped before the controller's 12-hour deadline. The initial 6–8-hour estimate underestimated per-step measurement cost; the controller ran approximately 11h 51m. One initial CPU-only launcher failed before scientific execution because Slurm spooled its relative path; the corrected absolute-path launcher and its revision are recorded in provenance.

The approved plots, table and note are saved in the [curated index](curated/INDEX.md). No additional sampling is scheduled. The remaining decision is how to present the unresolved first-step point; the saved signed estimate and interval are preserved.
