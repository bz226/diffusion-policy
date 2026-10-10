# Stage 2: DPPO clipping ablations

Which clipping mechanisms support stable HalfCheetah-v2 fine-tuning, and does their effect depend on reusing a batch for multiple actor updates?

**Status: execution is incomplete. My scheduler handoff interrupted R0 and NC1 seed 1 after 138 of 140 iterations.** Both retain all scheduled evaluations through iteration 130 and checkpoints through iteration 105; their final two training iterations and final checkpoints are missing. This was an orchestration error, not a numerical training failure. NC1 seeds 0 and 2 completed all 140 iterations. [Failure record](provenance/parallel_handoff_failure.json) preserves the cause, stopping points and last diagnostics.

The remaining 15 approved runs continue under the instruction to record individual failures and proceed; nine have been submitted concurrently. Interrupted runs remain marked failed and will not be restarted automatically. Full condition comparisons and candidate outputs remain pending; all prior logs, results and checkpoints are preserved.

The fixed-batch gate passed on the same saved real training batch, model, optimizer and RNG states: 20,000 observations and 200,000 denoising pairs, five epochs and 20 actor/critic minibatch steps. Every compared loss, gradient, final parameter/optimizer state and fixed-noise sampling value was bitwise equal on CPU and A6000, with maximum absolute/relative differences zero. Diagnostics preserved Python, NumPy, torch CPU and CUDA RNG states. [Gate evidence](runs/verification_full_update/verification.json) records the predeclared GPU tolerance and raw comparisons. Trajectory differences never decide equivalence or stop runs; R0's same-step evaluations outside the Stage 1 mean ± five sample standard deviations remain informational red flags.

The user approved parallel conditions within nine training allocations (360 CPUs). Each run retains one GPU, 40 CPUs, 64 GiB and a three-hour cap; each condition's three seeds are submitted together. [SOAL limits](provenance/parallel_allocation_limits.json) allow nine, but six A6000s were occupied by other users at the availability check, leaving about four for this study. Slurm determines actual concurrency. [Parallel authorization](provenance/parallel_dispatch_authorization.json) retains the original 58.5 GPU-hour budget and 24-hour controller deadline.

Only NC1 seed 2 used H200 NVL. Its [saved-batch H200 check](runs/verification_h200/verification.json) passed with zero absolute/relative differences across 78 comparisons and unchanged RNG states. All future jobs request A6000. The hardware difference remains a comparison limitation.

Scientific source remains clean at `ab46b150fa34b5a5b457cd4062cd4c5ad830d964`; [methods](methods.md), [diff](provenance/dppo_changes.patch) and [dependencies](provenance/pip_freeze.txt) define the unchanged experiment. Active runs are monitored every 30 minutes, with terminal monitors paused. Stage 1 is read-only. Final analysis will retain failures and generate only the approved figures and condition-by-seed table; curation still requires review.

<!-- campaign-status-start -->
Current execution (2026-10-09T21:28:17+00:00): running_parallel_conditions; condition parallel; 12/18 ablations submitted; R0 failed. [Live run record](runs/campaign_revised.json).
<!-- campaign-status-end -->
