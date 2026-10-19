# DPPO with the Monte Carlo advantage

**Question.** Does replacing GAE with the Monte Carlo return minus the critic change performance when the PPO pieces are retained?

**Status.** All three new training seeds are complete and verified. The original nine reference runs were reused without new sampling.

The reward comparisons fill the missing two-by-two cell. Historical logging limits the mechanistic conclusions: DPPO and without_PPO have no training J_disc, and original DPPO has neither saved clipfrac nor the switched-off fraction. No missing quantity is inferred from an unrequested rerun or from an unmatched sampler.

[Full per-seed table](ladder2x2_table.md) · [Contrasts](contrasts.md) · [Data and commands](analysis_record.json)

![Seed-0 reward curves](figs/ladder2x2_curves_seed0.png)

## Q1. GAE under reuse

DPPO minus without_GAE: last-five-evaluation difference 128.28; pooled seed SD 48.386; exceeds reading-rule threshold; training return difference 180.72; pooled seed SD 84.457; exceeds reading-rule threshold. Discounted-return gain cannot be compared because DPPO did not log it.

## Q2. Reuse and the interaction

Reuse under Monte Carlo: last-five-evaluation difference 102.36; pooled seed SD 30.217; exceeds reading-rule threshold; training return difference 88.113; pooled seed SD 66.768; unresolved.

Reuse under GAE: last-five-evaluation difference 147.2; pooled seed SD 49.5; exceeds reading-rule threshold; training return difference 165.11; pooled seed SD 62.49; exceeds reading-rule threshold.

The interaction is reuse under GAE minus reuse under Monte Carlo: last-five-evaluation difference 44.842; pooled seed SD 57.994; unresolved; training return difference 76.993; pooled seed SD 91.449; unresolved. Its J_disc counterpart is not logged. “Reuse” here changes the full PPO bundle (step reuse, ratio clip, KL stop, log-prob clamp and sampling-noise truncation); the contrast does not isolate reuse alone.

## Q3. Direction versus movement

The complete four-method question is not identifiable from the existing records: original DPPO and without_PPO did not save J_disc. Their parameter distances can be recovered from checkpoints, but distance alone cannot supply their missing vertical coordinate. The distance plots show only genuinely logged J_disc paired with the pre-rollout parameter distance, without interpolating or substituting measurements. Mean final distances are DPPO 6.9144; without_PPO 0.89446; without_PPO_and_GAE 0.84458; without_GAE 6.9372.

## Q4. How hard the PPO clip works

For without_GAE, mean clipfrac is 0.023921 and mean switched-off fraction is 0.014471. Both historical DPPO quantities are not logged, preventing that direct comparison. Mean within-update PPO approximate KL is 1.0164e-05 for without_GAE versus 7.8145e-06 for DPPO; the separate true-KL column must not be compared to this PPO statistic.

## Setup and validity

Three seeds per method, 140 iterations, 126 fresh-data updates, 80 episodes per batch. without_GAE changes only the actor advantage to raw Monte Carlo return minus the pre-update critic converted to raw units; critic targets remain GAE λ=0.95. The PPO loop, noise clip, log-prob clamp, ratio clip and KL early stop retain the DPPO settings. All averages and sample SDs use independent training seeds.

Checkpoint state_0/35/70/105/139 contains 0/32/63/95/126 updates. Evaluation iteration 130 is at update 117 (9.36M environment steps). Training rows log their post-batch step counter but the plotted rollout uses the pre-update count. θ₀ for gain is each seed's own iteration-1 training J_disc, rather than the earlier shared pretrained anchor.

[Methods and commands](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/methods.md) · [Gate records](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/runs/gates) · [Training job file](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/jobs_dl1.txt). Exact source revisions, per-run commands, completion evidence, resource sharing and filenames are retained in [analysis_record.json](analysis_record.json). No final-checkpoint sampling was requested or performed by this analysis.

## Gates, revision and allocation

| Gate | Result | Max absolute difference | Max relative difference |
| --- | --- | --- | --- |
| G1 | passed | not logged | not logged |
| G2 | passed | not logged | not logged |
| G3 | passed | 0 | 0 |
| G4 | passed | not logged | not logged |
| G5 | passed | 0 | 0 |
| G6 | passed | 0 | 0 |

Full numerical evidence: [/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/runs/gates/complete/gate_summary.json](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1/runs/gates/complete/gate_summary.json). Execution branch `stage3-dl1`, revision `9c51cd209aba253fcd611e169f4c21615e37ef95`, based on `35cb89ba46236860cc01738942d7e5d2dbad74e3`.

| Seed | Scheduler job | Node | GPU allocation | Wall seconds |
| --- | --- | --- | --- | --- |
| 0 | 17959932 | soal-10 | NVIDIA RTX A6000, GPU-3682bd5c-3f3a-46c3-f3aa-303045ccf55d, 580.82.07, 49140 MiB | 7568.6 |
| 1 | 17959933 | soal-10 | NVIDIA RTX A6000, GPU-592bcec5-b313-adef-6efd-48801e6391c0, 580.82.07, 49140 MiB | 7420.6 |
| 2 | 17959934 | soal-10 | NVIDIA RTX A6000, GPU-700af187-4695-9790-fafb-9a86d62a3011, 580.82.07, 49140 MiB | 7587.9 |

Each training process receives one allocated GPU and 40 CPUs. Allocations do not terminate or alter other jobs. Submission times and GPU identities in the saved worker manifests document resource sharing.

## Figures

[ladder2x2 curves seed0](figs/ladder2x2_curves_seed0.png) · [PDF](figs/ladder2x2_curves_seed0.pdf)

Seed 0 of each method; filled circles are point estimates, with no uncertainty shown. The horizontal axis counts actor updates before the rollout, each consuming 80 episodes; evaluation keeps the DDPM noise except at the last denoising step, while training uses a final-step noise floor of 0.1. J_disc is the raw-reward sum discounted by 0.99 per four-environment-step decision. DPPO and without_PPO did not log J_disc, so they have no discounted-return curve; all reward curves use the original runs. No smoothing or replacement runs are used.

[ladder2x2 distance seed0](figs/ladder2x2_distance_seed0.png) · [PDF](figs/ladder2x2_distance_seed0.pdf)

Seed 0 of each method; no uncertainty shown. Each point pairs the rollout's discounted training return with the actor distance before that rollout; the same iteration's post-update distance is not used. For a checkpoint-only distance record, only an exactly matching rollout update is eligible. DPPO and without_PPO have saved checkpoints but no training J_disc, so neither can be placed on this axis. Consequently this figure cannot establish the direction-versus-movement comparison across all four methods. For three-seed curves both coordinates are averaged at matching update indices, rather than interpolating returns at matched distances.

[ladder2x2 curves 3seed](figs/ladder2x2_curves_3seed.png) · [PDF](figs/ladder2x2_curves_3seed.pdf)

Three-seed means; filled circles are point estimates, with no uncertainty shown. The horizontal axis counts actor updates before the rollout, each consuming 80 episodes; evaluation keeps the DDPM noise except at the last denoising step, while training uses a final-step noise floor of 0.1. J_disc is the raw-reward sum discounted by 0.99 per four-environment-step decision. DPPO and without_PPO did not log J_disc, so they have no discounted-return curve; all reward curves use the original runs. No smoothing or replacement runs are used.

[ladder2x2 distance 3seed](figs/ladder2x2_distance_3seed.png) · [PDF](figs/ladder2x2_distance_3seed.pdf)

Three-seed means; no uncertainty shown. Each point pairs the rollout's discounted training return with the actor distance before that rollout; the same iteration's post-update distance is not used. For a checkpoint-only distance record, only an exactly matching rollout update is eligible. DPPO and without_PPO have saved checkpoints but no training J_disc, so neither can be placed on this axis. Consequently this figure cannot establish the direction-versus-movement comparison across all four methods. For three-seed curves both coordinates are averaged at matching update indices, rather than interpolating returns at matched distances.

[without gae ppo steps](figs/without_gae_ppo_steps.png) · [PDF](figs/without_gae_ppo_steps.pdf)

Seed 0; each point averages the logged minibatch quantities within one update, normally 20 steps. A pair is switched off when its normalized, denoising-weighted advantage is positive and its ratio exceeds 1+ε, or its advantage is negative and its ratio falls below 1−ε. DPPO retained per-step approximate KL in its original console log, but neither clipfrac nor switched-off fraction was saved with W&B disabled; those two reference curves are therefore absent. Approximate KL uses PPO's clamped, coordinate-mean log-prob ratio and is distinct from the post-update true-KL diagnostic. Points only; no smoothing or uncertainty shown.

Canonical study folder: [/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1](/soalnas/share/data/zbao7/diffusion_policy/results/stage3/dl1).
