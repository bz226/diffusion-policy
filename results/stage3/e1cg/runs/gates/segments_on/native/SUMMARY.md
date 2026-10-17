# Final-checkpoint evaluation

| Sampler | Undiscounted return (mean ± s.e.) | Discounted return (mean ± s.e.) | x0 saturation |
|---|---:|---:|---:|
| train | 4054.91 ± 71.3885 | 1432.45 ± 19.8394 | 0.0673171 |
| eval | 4054.91 ± 71.3885 | 1432.45 ± 19.8394 | 0.0671782 |

Seed initialized once; each repeat collects train then eval, with continuous policy RNG across repeats. Pretrained repeat 0 has the same policy-RNG prefix as each one-repeat checkpoint job. DDPM deterministic=True still adds noise at t>0, and zero noise at t=0. Seed 5000 aligns policy RNG streams across checkpoints, but the existing wrapper does not seed Gym's underlying np_random generator, so initial environment states are not fully paired; environment seeding remains unchanged.

Episode standard error uses sample SD (ddof=1)/sqrt(n); pretrained pooled summaries contain 240 episodes per sampler, each final checkpoint 80.

action_oor_frac uses physical actions sent to Gym after the wrapper's float32 affine unnormalization; policy_action_oor_frac separately uses normalized policy coordinates.
