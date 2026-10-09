# DPPO HalfCheetah: curated artifacts

Question: Does unmodified DPPO reproduce the approximate HalfCheetah-v2 Fig. 5 fine-tuning curve under the requested three-seed protocol?

User approval: “keep in curated” for artifacts A–C and their proposed filenames. Curation date: 2026-10-09. All three proposed artifacts were retained; the original candidates and every run remain preserved.

Analysis run: [three_seed_analysis](../runs/three_seed_analysis/). Training records: [seed 0](../runs/seed0_handoff/manifest.json), [seed 1](../runs/seed1/manifest.json), [seed 2](../runs/seed2/manifest.json). Reproduce and interpret these outputs using the [methods](../methods.md), [report](../experiment.md), [analysis code](../code/analyze_results.py), [independent audit](../runs/three_seed_analysis/independent_audit.json), and [visual review](../runs/three_seed_analysis/review.json). DPPO commit: cc7234ad7ff39a8f32de3af903606723a16f0648.

| ID | Curated artifact | Question answered / limitation | Original source |
|---|---|---|---|
| A | [halfcheetah-dppo-eval-return.png](halfcheetah-dppo-eval-return.png) | Evaluation improvement; last evaluation is at 9.36 million steps. | [Candidate](../runs/three_seed_analysis/candidates/eval_return_vs_env_steps.png) |
| B | [halfcheetah-dppo-train-return.png](halfcheetah-dppo-train-return.png) | Training return with exploration noise; differs from evaluation. | [Candidate](../runs/three_seed_analysis/candidates/train_return_vs_env_steps.png) |
| C | [halfcheetah-dppo-return-checkpoints.csv](halfcheetah-dppo-return-checkpoints.csv) | Per-seed and aggregate returns; nearest saved evaluation steps. | [Candidate](../runs/three_seed_analysis/candidates/returns.csv) |

**Caption A**

Evaluation return during unmodified DPPO fine-tuning: each point is a native within-iteration mean of undiscounted 1000-step episode returns. The x-axis counts training environment steps in millions; colored lines show seeds 0, 1, and 2, and the black line and shaded band show their mean ± population standard deviation (ddof=0), not a confidence interval. The last observed across-seed mean is 4758.37 at 9.36 million steps. Evaluation ends before the 10 million-step target; no smoothing or extrapolation is applied.

**Caption B**

Training return during unmodified DPPO fine-tuning: each point is a native within-iteration mean of undiscounted 1000-step episode returns, collected using training exploration noise before that iteration's update. The x-axis counts training environment steps in millions; colored lines show seeds 0, 1, and 2, and the black line and shaded band show their mean ± population standard deviation (ddof=0), not a confidence interval. The final training mean is 4439.12 at 10.08 million steps. Training and evaluation returns use different sampling-noise settings and are not interchangeable; neither curve is smoothed or extrapolated.

**Caption C**

Raw undiscounted episode returns are shown for each of three independent runs and their mean and population standard deviation (ddof=0), not a confidence interval. Evaluation rows use the nearest observed iteration to each requested step; both requested and actual steps are shown, without interpolation, and the final row lists training iteration 1 at 80,000 steps. The last observed evaluation mean is 4758.37 at 9.36 million steps, leaving the exact 10 million-step evaluation unmeasured.

Boundary: three independent seeds were requested, versus five in the paper. Exact evaluation at 10 million training steps is unmeasured; no smoothing, interpolation, extrapolation or additional evaluation was introduced.
