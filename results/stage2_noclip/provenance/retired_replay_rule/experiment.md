# Stage 2: DPPO clipping ablations

Which clipping mechanisms support stable HalfCheetah-v2 fine-tuning, and does their effect depend on reusing a batch for multiple actor updates?

**Executed result: inconclusive. R0 failed the required replay check at iteration 6; the campaign stopped before any ablation launched.** The pristine probes already differed before training. This gate failure therefore does not, by itself, establish a change in default computation.

The first mismatch was the original log's four-decimal PG loss:

| Run | Iteration-6 PG loss | Initial eval return | First train return |
|---|---:|---:|---:|
| P1, pristine | −0.0004 | 4351.0564 | 3898.7218 |
| P2, pristine | −0.0004 | 4294.7541 | 3866.3807 |
| R0, edited defaults | **−0.0005** | 4367.2946 | 3924.8082 |

These are three seed-0 runs, not independent condition averages; no uncertainty band is shown. Both probes completed 20 iterations on pristine `cc7234a` before source edits. Their differing initial returns demonstrate unavailable exact replay in this setup. Under the approved fallback, R0 must match every original value where P1=P2, including rounded tokens; their PG-loss equality could be coincidental. The [independent audit](provenance/R0_failure_context.json) reconstructed 1,475 enforced comparisons through iteration 6: all earlier comparisons passed, and this was the sole mismatch. R0's initial evaluation was inside the Stage 1 mean ± five sample-standard-deviation interval, 4092.31–4591.65; no gross-deviation flag occurred before stopping.

[Fixed-batch verification](runs/verification/verification.json) passed on CPU and A6000: pinned/default losses, gradients and AdamW updates matched exactly on the tested inputs. Clamp/reduction flags and invalid-value rejection worked; diagnostics preserved global RNG states. The single-actor-step checks covered 200,000 pairs, retained 20 critic steps, and produced ratio **1.0** and clip fraction **0.0**. R0 printed `clamp_logprob=True logprob_reduce=mean actor_single_step=False`. These checks support the implementation but do not override the failed replay gate.

R0 saved seven finite records through 480,000 training steps. Its last diagnostics were `kl_true_per_action=0.0769467`, `logratio_p99=0.3791184`, `clamp_hit_frac=0`, `approx_kl=1.28212e-5`, and `clipfrac=0.021622`. The supervisor intentionally terminated it during iteration-7 collection after detecting the iteration-6 mismatch; no NaN/inf or spontaneous crash occurred.

The two requested source files are committed on `stage2-noclip` as `ab46b150fa34b5a5b457cd4062cd4c5ad830d964`; [diff against cc7234a](provenance/dppo_changes.patch). No YAML, dependencies, or Stage 1 outputs changed. [Methods](methods.md) retain the approved design. Read-only [seeding inspection](provenance/seeding_inspection.json) found that DPPO's wrapper does not forward the seed to Gym HalfCheetah's separate reset RNG, consistent with probe divergence; underlying RNG states were not captured, so this is not proven to be the sole cause. No fix was made.

P1/P2 ran concurrently on separate RTX A6000s; each job had 40 CPUs and 64 GiB. R0 ran alone. Process times were 17m40s, 17m50s, and 5m57s respectively; mean training iterations were 57.18s, 57.63s, and 56.51s. GPU verification took 36.64s. [Terminal records](provenance/terminal_status.json) link every native log directory, `result.pkl`, checkpoint and runtime; [environment versions](provenance/pip_freeze.txt) and commands are preserved. All jobs are terminal and monitors paused.

**Decision boundary:** stopped under the required R0-failure rule. No clipping-effect conclusion, ablation figures, or condition comparison table is available. Logs and checkpoints remain intact; nothing was curated. Further scientific work requires a revised, approved instruction.
