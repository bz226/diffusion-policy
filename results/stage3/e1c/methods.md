# E1C methods and execution protocol

## Question and comparison

E1C isolates raw Monte Carlo returns with a learned critic, between E0 (without_PPO, GAE + critic) and E1 (MC + LOO). This is the user's Addendum E1C. Reuse the three completed E0/E1 seeds and the Stage-1 DPPO curves; no reference training is repeated. The user explicitly approved per-chunk normalization to match the actual E1 runs. Full-batch normalization is not introduced.

## Algorithm

For each fresh batch of 40 environments × 500 decisions, retain raw four-step decision rewards, episode boundaries, the pre-update critic values, and the reward-scaler divisor c copied before the rollout changes scaler statistics. Compute raw reward-to-go G in float64 with gamma=0.99, stopping at every 250-decision episode end. The actor residual is G−cV(s); convert to the existing actor dtype in the unchanged preparation path. The critic still uses its usual running-scaled GAE(lambda=0.95) targets and 20 minibatch steps.

Accumulate one actor gradient across the existing final critic permutation's four 50,000-pair chunks. Each chunk normalizes its own advantages with PyTorch's sample standard deviation plus 1e-8, then applies denoising weights 0.99^(9−i). Score loss averages the unclamped coordinate log probabilities. Weight each chunk by its share of the 200,000-pair batch and take exactly one AdamW step. There is no decision discount gamma^k. Sampling/log-prob noise floors remain 0.1, and the existing x0 clamp remains unchanged.

| Setting | Value and reason |
|---|---|
| Seeds | 0, 1, 2; requested independent training runs |
| Iterations / checkpoint interval | 140 / 35; match Stage 3 |
| Episodes per batch / horizon | 80 / 250 decisions; existing HalfCheetah configuration |
| Actor optimizer / learning rate | AdamW, 1e-4; unchanged E1 control |
| Advantage normalization | Per shuffled 50,000-pair chunk; approved correction to match E1 |
| Actor loss / log-prob reduction | Score / mean of 24 coordinates; E1 configuration |
| Denoising discount / decision discount | 0.99 / false; E1 configuration |
| Clipping flags | Ratio coefficients 1e6, target_kl=null, clamp_logprob=false, randn_clip_value=100; Stage-3 COMMON |
| Diagnostics | stage3_diag=true, baseline_diag=true; inert |

## Gates and provenance

Case A uses the passed critic source at branch tip `0f5118492ca41abed24896176225ebcfc813b1f5`, exported with the repository snapshot helper into an ordinary source directory in the isolated E1C worktree. Runtime outputs live at the main repository's absolute `results/stage3/e1c/` path. Record subsequent source/runtime commits and retain gate source snapshots. The main gradient-mean source stays unchanged.

Run A1 on the saved batch: native float64 residuals versus explicit G−cV, relative error ≤1e-12; verify normalization in each actual chunk. Check A3 using identical E1C flags and permutation: replace critic outputs by zero and compare the accumulated actor gradient with mc_none, relative L2 ≤1e-6 on CPU. Reuse the passed C2, C4 and C5 evidence for A2, A4 and A5 only after verifying source equality. Gate failure stops launch. No deterministic-algorithm settings are enabled.

## Diagnostics and units

Existing critic diagnostics retain per-decision-index residual means and population standard deviations across episodes. Their original energy keys use gamma^(2k), appropriate to the critic addendum. E1C's requested **unweighted** energy is recovered exactly from those moments: E=sum_k(mean_k²+std_k²). Use the same sum over each specified 50-decision bin and divide critic energy by LOO energy. Keep original weighted keys distinct. This analysis-only identity does not change training, gradients or RNG states.

Gradient-mean V1–V3 are fixed-policy, warmed/frozen-critic proxies. They normalize across the full batch. V2 uses MC returns of clipped scaled rewards minus the frozen critic; E1C uses raw MC returns minus c_start times its learning critic. Equivalence up to scaling additionally requires inactive reward clipping and matching normalization/scaler. Report these limitations with Q4 instead of claiming exact training-estimator identity. Pre-normalization baseline-independence arguments do not make the normalized update an unbiased objective gradient.

## Execution and failure handling

The approved 16 GPU-hour cap includes measured allocation time and full outstanding reservations. Each training attempt has a four-hour cap; at most one same-seed retry is allowed after identifying and fixing the cause, within the remaining aggregate cap. Preserve every attempt, failure iteration and last diagnostics. No tuning or numerical rescue is permitted. Close vector environments and workers explicitly; successful jobs must exit zero. CPU gates use two threads and a 20-minute timeout; any GPU gate would be bounded by one hour.

Start three E1C seeds in parallel before admitting new gradient-mean jobs. Existing jobs must never be cancelled. Resume the administrative G controller after reconciling its obsolete branch-check failures, keeping scientific data and actual scheduler statuses. While E1C runs, G admits at most five workers, subject to available GPUs/CPUs and its own approved cap. C remains paused. Monitor each campaign every 30 minutes; record an iteration-70 comparison and estimated completion from observed iterations.

## Analysis

Define x as the number of actor updates applied **before** the rollout: at itr 1, x=0; at evaluation itr 130, x=117. A training rollout after x prior updates produces update x+1, making the final ten training records updates 117–126. Plot three-seed means with sample sd (ddof=1) and thin individual lines; DPPO is a faint dashed reference. Use labels `without_PPO (GAE + critic)`, `MC + critic`, and `MC + LOO`.

Produce the requested returns/J_disc, KL/distance, saturation/gradient-cosine, unweighted residual-energy, and explained-variance figures. Tabulate per-seed quantities before mean±sd: eval itr130; last-ten-update train return and J_disc gain; median training KL; final theta distance and saturation. J_disc gain subtracts the existing pretrained train-sampler reference from all conditions, using the same saved reference and recording its value. No new checkpoint evaluation is added. For early/late bin comparisons use the specified update numbers 1–10, 60–70 and 117–126, not iteration numbers.

For each metric, differences smaller than twice sqrt((sd_A²+sd_B²)/2) are unresolved. Apply this rule to Q1–Q3 without extra significance tests. Q4 uses the latest fully completed gradient-mean round available at analysis time, with round/point/source recorded. Preserve partial and failed runs explicitly. Write the requested REPORT, tables, figure captions and Q1–Q4 after completion; place artifacts in candidates before user-selected curation.
