# FRS: SGD with frozen coordinate scales

**Question.** Does a fixed diagonal rescaling recover E2_beta0's progress and stability without adapting the scales during training?

**Answer and status.** Completed and verified: all three FRS1 seeds **collapsed in performance**, despite finishing 140 iterations without NaN/inf. Mean eval return at itr130 was **−447 ± 221**, compared with **4,588 ± 42** for E2_beta0. Final evaluations and analysis are complete; approved outputs are curated; monitors are paused. Total usage was **6.664 of 14 GPU-hours**. [Final report](REPORT.md), [verified analysis](runs/final_analysis/review_record.json).

**Design.** FRS1 uses the unchanged E2 estimator, one SGD step per fresh batch, learning rate 1e-4 and fixed denominators `max(r, 1e-3 × median(r)) + 1e-8`. The allowed reuse route estimates `r` from the 25 saved pretrained-policy V4 batch sums divided by 40; there is no new sampling for scales. Seeds 0–2 run in parallel for 140 iterations, followed by final-checkpoint evaluations. References are the completed E2, E2_beta0 and SGD10 runs. [Methods](methods.md), [approval](provenance/approval.json), [source snapshot](provenance/dppo_source_snapshot.json).

**Outputs.** The approved [training comparison](curated/frs-training-comparison.png), [ratio/KL diagnostics](curated/frs-update-instability.png), and [report and tables](curated/frs-results/REPORT.md) are retained in the [curated index](curated/INDEX.md), with captions and provenance links. All original candidates, raw logs, result.pkl files and checkpoints remain in the experiment folder; the [per-seed table](curated/frs-results/seed_results.csv) records their paths.

**Boundary.** Cached-input reproducibility verifies scale computation, not fresh same-seed trajectories. The saved V4 sums have the E2 estimator and units, with a different floating-point accumulation order from training. Prior Phase-1 confidence intervals are not used. The approved curation retains every FRS1 seed, including all reward collapses.

<!-- frs-status-start -->
Last monitor: 2026-10-11T11:40:12.326840+00:00; **complete**. FRS1_seed0: complete; FRS1_seed1: complete; FRS1_seed2: complete; ckpt_FRS1_seed0_seed5000: complete; ckpt_FRS1_seed1_seed5000: complete; ckpt_FRS1_seed2_seed5000: complete. Monitor paused.
<!-- frs-status-end -->
