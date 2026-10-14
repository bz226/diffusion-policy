# DPPO per-step measurement methods

## Definition and fixed setup

This implements the user-approved per-step gradient diagnostic. DPPO source is the Stage-3 gradient-mean revision `44b2998228e1a93aaef544e8030c30640d17dc06`, imported as ordinary files. Development is isolated from existing collection source. The finalized source revision is recorded in the job plan and each output. No deterministic-algorithm settings are enabled.

Each batch has 40 environments × 500 decisions = 80 complete 250-decision episodes and 200,000 decision/denoising pairs. The actor has 568,392 parameters. The native configuration uses five epochs × four 50,000-pair minibatches, ratio epsilon 0.01, coordinate log-prob clamp [-5,2], mean log-prob reduction, per-minibatch advantage normalization, identity quantile clipping, denoising discount 0.99, GAE lambda 0.95, reward scaling/clipping and noise clip ±3 standard deviations. Sampling and log-prob noise floors are 0.1. `max_grad_norm` is None and `reset_at_iteration` is false. Full batches finish after autoreset at decisions 250 and 500, so successive batches start at episode boundaries.

## Parts and state

Original checkpoints come from Stage-1 seed 0 (`2026-10-08_20-01-32_0`), not a later rerun. Saves at iterations 35, 70 and 105 contain 32, 63 and 95 actor-update batches respectively. Iteration 130 evaluates after 117 updates (9.36M training steps). Evaluation iterations never update the actor.

A: restore each original checkpoint's critic and actor, with no warm-up. Original files/logs lack the scaler divisor and statistics. Fit c = sum(VG)/sum(V²) on exactly five new batches at each point, with raw terminal MC returns, gamma0.99 and the frozen critic. Freeze that divisor during measurements. This is the authorized approximation; it does not recreate the original running scaler's mean/count or current-batch update. The checkpoint critic is compared bitwise and by the explicitly requested hash. Shuffle 200,000 pairs once per batch and compute four gradients at theta_old without optimizer steps. Save group sums for g1 and the mean of all four gradients.

B: rerun seed0 from the released pretrained policy,140 iterations,savefreq35,wandb=null. Only guarded observation/save hooks are added to the native training loop. Save full model/optimizers/schedulers/scaler/global RNG/counters at iterations0,35,70,105,139; log the start and post-scaling c each iteration. At iteration36 capture the update input and RNG, then post-update state. R2 restores state35 in a fresh process and replays that saved update; maximum coordinate relative actor deviation must be <=1e-5. Final evaluation outside the original three-seed range by more than150 is flagged.

C: each batch restores model, AdamW moments/counts, schedulers and full reward scaler from the rerun checkpoint. Keep the measurement job's fresh RNG streams rather than restoring checkpoint RNG. Collect one native train-sampler batch, update the restored reward scaler, compute native pre-update GAE, then execute the literal native actor loop. Critic optimizer steps are skipped, as authorized; critic backward is retained and does not change actor gradients. Observe gradients after backward and before optimizer step; use a no-grad forward for pair-level clip scalars and verify it changes neither the actor gradients nor global RNG. S1 compares the native loop's first gradient against A's calculation on two identical batches/permutations (maximum relative deviation <=1e-6).

## Records and statistics

For each step store a float64 group sum and per-batch squared norms, plus native clipfrac/KL, switched-off fraction, mean absolute ratio displacement, gradient norm and within-batch cosine to step1. Zero gradients remain valid observations with undefined cosine. No individual gradient arrays are written. C additionally stores group sums and squared norms of the sum of gradients and net parameter displacement. Early stopping is recorded; missing later steps are never imputed.

With K batches, group-summed S and sum of squared norms Q, signal squared=(S·S−Q)/(K(K−1)); batch variance=(Q−S·S/K)/(K−1); reported per-episode variance=80×batch variance; B=per-episode variance/signal squared. For reused-batch steps the factor80 is a plotting convention. Uncertainty uses delete-one-job jackknife and two-sided90% Student-t intervals (9 degrees of freedom for10 groups), recomputing ratios jointly. Nonpositive signal estimates remain in the table; no invented log-axis values are plotted. Every figure is conditional on one training seed and measured checkpoints.

## Seeds, resources and outputs

Preparation seeds:2009000,2019000,2029000. Part-A group seeds:2000000+10000p+100j; Part-C:2100000+10000p+100j, p=0,1,2 and j=0..9. S1 seed2190000; rerun seed0. Native DPPO environment seeding is preserved: job seeds initialize Python/NumPy/Torch, but the inherited MuJoCo private reset RNG is not fully seeded, so trajectories are not bitwise reproducible.

Use the original Python3.8/torch2.4.0 CUDA12.1 environment and MuJoCo2.1.0, with OMP/MKL/OpenBLAS thread overrides unset as in Stage1. Every GPU allocation requests40 CPUs and64GiB. Approved hard budget80 GPU-hours, maximum9 GPUs, explicit per-job wall caps, no automatic retry. The controller records scheduler identities, exact commands, effective configs, source revision, progress and clean exits. Existing studies and interactive allocations are left alone.

The approved figures, table and note are generated first into candidate directories; none is automatically curated. Part-A analysis runs when its30 groups finish; final analysis follows all requested jobs.
