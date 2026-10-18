# AdaGrad-Norm: stopped at the first-step verification gate

**Question.** Does a single-scalar adaptive step train stably on `without_PPO_and_GAE + γ^k`, compared with AdamW and the completed SGD references?

**Answer: inconclusive; training was not launched.** G3 failed the prescribed relative tolerance of 1e-6 for the actual float32 first-step displacement. The discrepancy was 2.30866e-6 relative (6.09381e-9 absolute). Float32 rounding is a plausible explanation, but the saved check does not establish the cause. The optimizer, parameter precision and tolerance were not changed.

## Verification evidence

| Gate | Observed result | Outcome |
|---|---|---|
| G1: optimizer arithmetic | Maximum relative parameter error 1.10437e-7; accumulator relative error below 8.59e-16 | Pass |
| G2: incoming gradient | All 568,392 coordinates bitwise identical to AdamW; maximum absolute/relative difference 0 | Pass |
| G3: calibrated first step | Actual displacement 0.0026395528628152345; formula 0.0026395467690097864; relative difference 2.308655985808447e-6 | **Fail** |
| G4: unchanged paths | Default DPPO, single-step and discounted-estimator CPU replays bitwise identical, including actor/critic parameters, optimizer state and RNG | Pass |
| G5: 12 live CPU iterations | Not launched after G3 failed | Not run |

G3 used η=0.002639548322140887, b₀=3.2212902344925517e-6 and the G2-certified saved-batch gradient, whose norm is 0.0029694458966360083. It subtracted actual float32 parameter values and measured the displacement in float64. RNG states remained bitwise unchanged. [Preliminary evidence](runs/gates/preliminary_retry/gate_summary.json) · [G3 evidence](runs/gates/calibrated/gate_summary.json).

## Calibration and execution

All five calibration batches completed with the existing estimator and fixed KL*=0.000584517336086918. They gave d*=0.002639548322140887, a max/min distance spread of 2.58688, median gradient norm 0.0032212902344925516, and b₀=3.2212902344925517e-6. Realized KLs were 0.000583445–0.000591748; scaling exponents were 1.9093–2.0079. No calibration warning fired. [Calibration data](cal_seed3000/calibration.json).

Source revision: `2acb18e9629c577474ee7f1652c7cc6f5a3829ac`, branch `stage3-e1cg-an`, based on the reference revision `03f82611746954839169a64305777c3d7f91b139`. Source was developed in an isolated worktree and exported as ordinary study files. No existing experiment or result was changed. Four scientific source files implement the optimizer, factory, checkpoint state and logs; the estimator/update loop is unchanged. [Source diff](provenance/source_changes.patch) · [Methods](methods.md).

Calibration job 17940151 (seed 3000, A6000) completed in 114 seconds: **0.03167 GPU-hours**. Its controller was 17940149. CPU continuation 17940191 exited on G3 failure, left no child processes, paused its monitor and submitted no training plan or job. Aggregate CPU gate time was 740.064 seconds. The initial replay startup failed on a read-only D4RL cache; using the study-local cache fixed that setup issue, and its 11.313 seconds remain charged. [Commands and job records](runs/continuation/state.json) · [Calibration plan](runs/plan_calibration.json).

**Decision and boundary.** Execution is stopped as requested. Q1–Q4, training comparisons and figures cannot be answered without the unlaunched runs. Further investigation or a change to the verification criterion requires approval; no artifacts were curated and the unfinished study was not added to the completed-experiment checklist.
