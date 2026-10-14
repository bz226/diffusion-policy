# DPPO per-step gradient noise

This measurement asks how much noise DPPO's AdamW receives at each minibatch step. The table and figures use the gradient of the actual PPO actor loss in training-loss units; later steps depend on the saved optimizer moments and within-batch updates.

Checkpoint mapping recorded in the collection metadata: state_35 → 32 updates, state_70 → 63, state_105 → 95. Part A uses the original checkpoint critic and θ_old; Part C restores the rerun's full state before each batch. No gradient-norm clipping is configured, and reset_at_iteration is false. The sample is one training seed per path, not independent training seeds.

| Updates | Part A B₁ | Part A B_full | Part C B₁ | without_PPO_and_GAE as-run B |
|---:|---:|---:|---:|---:|
| 32 | 96895 [17194, 1.766e+05] | 55798 [29078, 82517] | 58889 [0, ∞) | 1.9618e+05 [0, ∞) |
| 63 | 7.4552e+05 [0, ∞) | 1.0848e+05 [42456, 1.7451e+05] | 4.5432e+06 [0, ∞) | 99952 [10950, 1.8895e+05] |
| 95 | N/A [0, ∞) | 1.6128e+05 [44394, 2.7816e+05] | 5.0892e+05 [0, ∞) | 2.0636e+05 [0, ∞) |

B is in episodes; one 80-episode update has relative variance B/80. The reference uses Monte Carlo-minus-critic advantages and a full-batch actor step, so its difference from DPPO is not solely a minibatch-size effect. Within Part A, B₁ versus B_full isolates first-minibatch sampling at the same θ_old.

| Part | Updates | Groups / batches | c (range if changing) | Source | B interval reaches zero signal? |
|---|---:|---|---|---|---|
| A | 32 | 10 / 500 | 445.86 [445.86, 445.86] | least_squares_five_batches_fixed | steps: none |
| A | 63 | 10 / 500 | 448.53 [448.53, 448.53] | least_squares_five_batches_fixed | steps: 1 |
| A | 95 | 10 / 500 | 450.33 [450.33, 450.33] | least_squares_five_batches_fixed | steps: 1 |
| C | 32 | 10 / 500 | 485.59 [484.97, 487.3] | restored_full_scaler_updated_on_each_batch | steps: 1, 2 |
| C | 63 | 10 / 500 | 491.25 [490.91, 492.02] | restored_full_scaler_updated_on_each_batch | steps: 1, 2, 3, 4, 5 |
| C | 95 | 10 / 500 | 493.59 [493.45, 494.11] | restored_full_scaler_updated_on_each_batch | steps: 1, 2, 3, 4 |

KL early-stop batches by checkpoint: 32 updates: 0, 63 updates: 0, 95 updates: 0. Missing later steps are never filled with zeros.
Part C skips critic optimizer updates; actor advantages were already computed before the update loop. The tabulated c is the batch's used divisor (with its observed range); restored starting divisors are retained separately in the CSV.

R3 rerun sanity: within requested tolerance; final rerun = 4708.1; original final evaluations = 4636.3, 4796, 4842.8.

Gate records (the analysis does not rerun or infer gates):
- [capture_gate.json](../runs/gates/R1/capture_gate.json)
- [replay_result.json](../runs/gates/R2/replay_result.json)
- [S1.json](../runs/gates/S1/S1.json)

Figures not yet drawable:
- dppo_step1_noise: Requires all three points, 500 observed gradients per step, and positive finite point estimates; invalid points [(95, 1)]; partial points [].

Data require more than two decades; all panels still share the same span: {'dppo_per_step_noise': 3.5}.

Full precision, signed squared-mean intervals and per-step counts are in [ppo_steps_table.csv](ppo_steps_table.csv); a readable table is in [ppo_steps_table.md](ppo_steps_table.md). Group paths, seeds and source commits are in [analysis_record.json](../candidates/final/analysis_record.json). No scientific jobs were launched by this analysis.
