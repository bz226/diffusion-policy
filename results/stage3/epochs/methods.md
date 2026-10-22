# DPPO actor-epoch comparison

## Design

The four conditions follow the user's EP protocol. Seed 0 is the only new training seed. All use the original HalfCheetah DPPO YAML at base cc7234a through the gated DL1 source revision 9c51cd209aba253fcd611e169f4c21615e37ef95. The actor advantage remains GAE λ=0.95; no Monte Carlo override or COMMON skeleton is applied.

Each iteration collects 500 decisions in each of 40 environments: 80 complete 250-decision episodes. A decision executes four environment actions. Every tenth iteration is evaluation only. The 140 iterations give 126 fresh-data actor updates; the last evaluation, iteration 130, measures the policy after 117 updates (9.36M training environment steps). Checkpoints at iterations 0, 35, 70, 105 and 139 contain 0, 32, 63, 95 and 126 updates.

| Setting | Value | Source |
| --- | --- | --- |
| Actor epochs k | 1 or 2; absent flag defaults to 5 | User EP intervention |
| Actor lr, same arm | 1e-4 | Baseline |
| Actor lr, matched arm | 5e-4 at k=1; 2.5e-4 at k=2 | User's first-order k·lr match |
| Scheduler min lr | Equal to actor lr | User EP constant-lr requirement |
| Update epochs / minibatch pairs | 5 / 50,000 | Baseline; unchanged for critic and loop |
| PPO ratio clip / KL stop | 0.01 / 1 | Baseline |
| Log-prob clamp / sampling-noise clip | [−5,2] / ±3 standard deviations | Baseline |
| Decision discount / denoising discount | No γ^k actor factor / 0.99 | Baseline |
| Training / save / evaluation iterations | 140 / every 35 plus final / every 10 | User EP protocol |

For each epoch, draw the original torch permutation and execute all four native minibatch loss/zero_grad/backward/critic-step sequences. Apply the actor optimizer step only while epoch < k. Preserve the original KL-stop branch; if it fires, record the actual counts without changing the algorithm. The live verification requires 4k actor steps and twenty critic steps on its checked batch. No forward or backward computation is skipped.

## Verification and diagnostics

G1 compares all four resolved configurations against Stage-1 DPPO, allowing only the actor epoch flag, matched learning rates and output naming. G2 directly replays an original saved batch with default flags on CPU and requires identical parameters, optimizer and global RNG states. G3 checks actual live AdamW/critic step increments for all four conditions. G4 requires the k=1 saved-batch critic and optimizer state, and all global RNG states, to equal the native baseline bitwise. G5 checks constant intended learning rates through the scheduler and throughout every saved run.

Inert observers record means over actor steps actually applied, last-actor-step switched-off fraction, approximate KL and mean absolute ratio displacement, cumulative AdamW step count, actor lr and parameter distance. Discounted returns use raw decision rewards, and any saturation subsample uses a dedicated generator. Existing baseline behavior is unchanged when the flag is absent. No additional diagnostic flag is passed in the resolved configurations.

## Analysis

Use the original Stage-1 seed-0 DPPO and Stage-2 seed-0 without_PPO at lr 1e-4 and 1e-3. One x-axis update means one fresh batch, regardless of optimizer step count. Rollout curves use the number of updates *before* that rollout. Discounted training return is the mean episode sum Σ_k0.99^kR_k. Its gain subtracts each run's own first training batch (iteration 1), as in the completed two-by-two analysis. Last-ten training updates are iterations 129 and 131–139; last-five evaluations are iterations 90–130.

Parameter distance is relative to each run's own state_0. A rollout's return is paired with its pre-rollout distance, never its same-row post-update distance. Historical checkpoint-only distances are used only at matching rollout update counts. Missing J_disc, saturation or clip records remain missing; no reruns or proxy measurements are substituted. Original DPPO's log can supply last-minibatch approximate KL where recorded, but does not supply the other two last-step quantities.

Figures follow the user's same-lr/matched-lr two-column layout, seed-0 point estimates only. The table records setup, terminal and late-window returns, discounted gain, distance, clipping and saturation. For each metric with three Stage-1 seeds, differences smaller than twice their sample SD are unresolved. A single-seed contrast cannot establish a population effect. Matching k·lr is a first-order premise, not a verified equality of total movement under AdamW momentum.

## Scope and provenance

Development uses a separate parent-repository worktree on stage3-epochs, from the DL1 tip. Source is ordinary files under results/stage3/epochs/dppo, exported with the repository snapshot helper before creating the study. Main output paths are absolute and lie under results/stage3/epochs. Per-run records retain source revision and relevant source contents, overrides, seed, environment, scheduler/GPU identity, counts, timing and exit status. Gates stop on failure. Training failures preserve all outputs and allow only a diagnosed same-seed retry within the approved remaining budget. Curation is a separate user decision.
