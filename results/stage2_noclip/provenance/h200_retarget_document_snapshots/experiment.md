# Stage 2: DPPO clipping ablations

Which clipping mechanisms support stable HalfCheetah-v2 fine-tuning, and does their effect depend on reusing a batch for multiple actor updates?

**Status: the revised fixed-batch gate passed; the authorized training campaign is launching.** Clipping effects remain unmeasured. Trajectory differences no longer decide code equivalence or stop R0.

Pristine and edited code were compared on the same saved real training batch: 20,000 observations and 200,000 denoising pairs, with identical initial model, optimizer and RNG states. Both actual update loops executed all five epochs and 20 actor/critic minibatch steps. **Every compared loss component, actor/critic gradient, final parameter/optimizer state and sampling-chain value was bitwise identical on both CPU and GPU: maximum absolute difference 0; maximum relative difference 0.** Per-minibatch results and raw evidence are linked in the [gate record](runs/verification_full_update/verification.json).

Sampling used the same 21 fixed noise tensors and compared the returned chain, final action and all 20 denoising transition states. Diagnostics preserved Python, NumPy, torch CPU and all visible CUDA generator states. No threading, deterministic-algorithm or other numerical setting changed. The GPU tolerance was declared in advance as `atol=1e-7, rtol=1e-5`; observed equality was exact. The full gate took 210.43 seconds on one A6000, including real-batch capture and both CPU/GPU comparisons. [Independent review](provenance/fixed_batch_gate_independent_review.json) documents coverage.

The fresh R0 uses `R0_rerun_seed0`, default flags, seed 0 and 140 iterations. It starts from the released pretrained policy. Its same-step evaluations outside the Stage 1 mean ± five sample standard deviations are recorded as non-stopping red flags. The old probes and stopped R0 remain preserved and are excluded from condition averages.

The 18 ablations execute as six sequential condition groups, each with all three seeds submitted together for parallel execution. Actual overlap depends on Slurm availability: initially R0 and two NC1 seeds started, while NC1 seed 2 waited for a free A6000. Each training process receives one RTX A6000, 40 CPUs and 64 GiB; R0 can overlap the first group for four concurrent GPUs. All scientific settings and commit `ab46b150fa34b5a5b457cd4062cd4c5ad830d964` remain unchanged. [Methods](methods.md), [source diff](provenance/dppo_changes.patch), and [revised authorization](provenance/revised_regression_authorization.json) define scope and budgets.

Runs are estimated at about 2h15m, capped at three hours each, and monitored every 30 minutes with immediate terminal checks. The original 58.5 GPU-hour ceiling remains in force. Individual crashes/nonfinite failures are recorded and remaining runs continue; trajectories never trigger a regression stop. After all runs terminate, the controller generates only the approved three figures and condition-by-seed table in candidates, with captions and paths to every log, result and checkpoint. Stage 1 is read-only; nothing is curated automatically.

<!-- campaign-status-start -->
Current execution (2026-10-09T17:49:34+00:00): running_condition; condition NC1; 0/18 ablations submitted; R0 starting. [Live run record](runs/campaign_revised.json).
<!-- campaign-status-end -->
