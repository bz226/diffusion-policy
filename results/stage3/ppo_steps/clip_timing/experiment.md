# DPPO: clipping and the reuse-induced mean gradient

**Status: completed CPU-only analysis; G1–G5 passed.** Does the clip engage when the extra mean gradient stops following the proposed reuse model? Clip activity rises during epoch 2, but the extra mean rotates: its cosine with the final direction reaches −0.37, −0.51 and −0.51, with intervals below zero. This is against the proposed fixed-direction cap mechanism, while leaving clipping's causal contribution unresolved.

Use the completed Part C measurements at 32, 63 and 95 updates: ten independent groups of 50 batches per checkpoint, with 20 PPO steps per batch. The saved per-gradient table supplies the primary mean norms and covariance traces; between-group products of the stored float64 sums supply the extra mean and direction. All intervals use delete-one-group jackknife standard errors and the 90% Student-t multiplier with nine degrees of freedom.

The approved bound is 30 minutes of local computation, four CPU threads and 8 GiB RAM. No scheduler jobs, GPU use, rollouts or gradients are permitted. All existing files under the parent measurement folder are covered by the requested before/after hash check. The new code lives on `analysis-dppo-clip-timing`, isolated from running studies.

Both requested figures, the per-step and summary table, and Q1–Q5 with checkpoint verdicts are in the [report](REPORT.md). All 2,186 pre-existing files were unchanged in the [preservation check](provenance/existing_files_after.json). The user approved a cosine axis of −1 to 1 so negative values remain visible; all other analysis choices are unchanged. The user approved all A–C artifacts; the [curated collection](curated/INDEX.md) contains both figures, captions, the table and report.

[Methods](methods.md) · [Preserved-file inventory](provenance/existing_files_before.json)
