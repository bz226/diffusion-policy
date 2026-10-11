# Frozen-rescaling addendum methods

This is the approved FRS1 addendum to the [Stage-3 protocol](../stage3/methods.md). The scientific design comes from the user's addendum; no external theoretical guarantee is inferred from this test. Source and all new outputs are ordinary files under `results/stage3_frs/`. The completed Stage-3 source, results and curated artifacts are read-only references.

## Estimator and frozen scales

Let `g` be the E2 code ascent gradient: the average of the score times raw MC–LOO return and within-episode discount over 200,000 decision/denoising pairs. The implementation differentiates its negative loss. All E2 choices remain fixed: gamma=0.99, H=250 decisions, 80 episodes from 40 environments × 500 decisions, ten fine-tuned transitions, 24-coordinate sum, no advantage normalization, no coordinate-logprob clamp, denoising weight one, noise bound100, and the retained x0 clamp/noise floors.

The source is `../stage3/ns_theta0_seed1000/noise_batch_vector_sums.npy`: float64 shape(25,5,568392), V4/theory_loo at index3. Each saved vector sums40 per-environment gradients; division by40 is the full-batch gradient in code units, with no additional factor2500. Sign does not affect RMS. The actor was frozen at theta0; critic warmup and frozen reward-scaler statistics do not enter V4's raw MC–LOO coefficients.

Compute `r_i = sqrt(mean_b[(sum_b,V4,i / 40)^2])` in float64, then `rtilde_i = max(r_i, f*median(r)) + 1e-8`. The median is global over all568392 coordinates; for an even coordinate count it is the midpoint of the two middle values. Save ordered actor parameter names, shapes and dtypes with the corresponding raw r tensors; cast each fixed denominator once to that actor parameter's dtype/device before use. Parameter order is registration order, never sorted checkpoint keys. Statistics must be finite, nonnegative and shape-compatible.

The update is `theta <- theta - eta * loss_gradient / rtilde`, implemented by dividing `.grad` in place and calling momentum-free, zero-weight-decay SGD once. No scales adapt. Existing raw E2 gradient norms/cosines and split-gradient diagnostics are retained before the optimizer transformation, as for AdamW references. New max_ratio and median_ratio summarize all coordinates of `abs(loss_gradient)/rtilde` at each update, including zero-gradient coordinates. Update norms and theta distances describe the actual parameter movement.

Saved V4 gradients were differentiated in float32 per environment and summed in float64. Their estimator and scaling equal E2, but this does not reconstruct the bit pattern of four shuffled50,000-pair training chunks. The reuse route is expressly allowed by the addendum; no fresh scale-measurement trajectories are drawn.

## Fixed choices and execution

| Setting | Value | Basis |
|---|---|---|
| Condition / training seeds | FRS1 / 0,1,2 | Approved minimum addendum |
| Scale source | 25 saved theta0 V4 batches, measurement seed1000 | Allowed reuse option |
| Processing seed | 4000 for both scale-conversion replays | Requested reproducibility gate |
| Global floor multiplier f / epsilon | 1e-3 / 1e-8 | User-defined update |
| Actor learning rate / scheduler min | 1e-4 / 1e-4 | User-defined FRS1 |
| Actor optimizer | SGD, momentum0, dampening0, weight decay0, no Nesterov | User-defined FRS |
| Iterations / save frequency | 140 /35 | Stage-3 constraint |
| Critic, rollouts and all other settings | Unchanged E2 | Controlled comparison |
| Final evaluation | seed5000;80 episodes per train/eval sampler | Inherited Stage-3 protocol |
| Hardware per training seed | One A6000,40 CPUs,64 GiB | Same setup as references |
| Parallel training / limits | Three seeds;4h each;14 GPU-hours total | Approved budget |
| Gate / final-eval allocation caps | 1h /15min each | Total reservation13.75 GPU-hours |
| Estimated runtime / monitoring | 2h15m per training seed /30min | Completed E2_beta0/SGD10 took2.00–2.07h |

Use the unchanged Stage-1 environment, MuJoCo and DPPO data exports; OMP/MKL/OPENBLAS thread variables remain unset. Pass every new key with `+`, `wandb=null`, and absolute log/data/scale paths. Print all effective flags at startup and validate them in the worker. New defaults preserve Stage-3 computation; YAML files are not edited. FRS3 is not approved in this campaign.

## Gate and stop conditions

Before training, compare Stage-3 defaults and E2 SGD paths against the new snapshot on the same saved batch/model/optimizer/RNG states: CPU bitwise and GPU float tolerance, including losses, gradients, parameters, optimizer state, fixed-noise sampling and RNG state. Test the new optimizer with denominator1 against SGD bitwise on CPU. Compare a rescaled step with independently formed theta-minus-eta-gradient-over-denominator to relative L2 error1e-7 on updated parameters; report absolute/relative errors and displacement rounding separately.

Run `rescale_stats` twice with processing seed4000 and identical saved V4 input, requiring identical numeric outputs. This is cached-input conversion reproducibility; the original Gym private RNG is not seeded by the wrapper, so it is not a claim of reproducible fresh trajectories. Verify diagnostics on/off leave parameter/optimizer/RNG states identical and scale tensors remain fixed. Do not enable deterministic-algorithm settings. Any formal gate failure stops before training; preserve the failure evidence and report it.

After the gate, commit the checked source on the parent repository's `stage3-theory` branch before launch. Per-run provenance uses that explicit parent revision and saved source bytes, not a nested repository. Each process owns one independent seed. A run stops on alignment failure, crash or non-finite quantities and records the iteration and last diagnostics; other seeds continue. No automatic retries, hyperparameter changes or budget extensions occur. Stop admissions at the approved cap, verify terminal outputs and pause monitoring after completion.

## Analysis and deliverables

Report raw r quantiles at0,1,10,50,90,99,100 percent, exact zero coordinates, coordinates strictly below the floor before epsilon, and the share of total r-squared carried by the largest ceil(0.01*P) coordinates. These describe gradient RMS heterogeneity, not the unreliable Phase-1 signal-norm confidence intervals.

Compare FRS1 with existing E2, E2_beta0 and SGD10 using the Stage-3 plots and condition/per-seed tables, plus max_ratio/median_ratio and KL over time. Include eval/train return, raw discounted J, distance from theta0, final-checkpoint evaluations, KL, stability/collapse and timing. Use means and sample SD(ddof1) across all three seeds; mark incomplete/non-finite runs explicitly. Differences below twice the pooled seed SD remain unresolved, with no additional significance tests. These runs test frozen scales, not a convergence theorem or guaranteed equivalence to adaptive Adam.

Keep plots and tables with captions in a run's `candidates/` directory. Present one artifact/name decision after the approved outputs are ready; do not curate automatically. Preserve every run, negative outcome and prior reference.
