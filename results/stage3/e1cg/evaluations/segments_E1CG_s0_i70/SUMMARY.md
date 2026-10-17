# Final-checkpoint evaluation

| Sampler | Undiscounted return (mean ± s.e.) | Discounted return (mean ± s.e.) | x0 saturation |
|---|---:|---:|---:|
| train | 4162.59 ± 43.6407 | 1470.78 ± 12.8612 | 0.0929965 |

Seed initialized once; train sampler only, with continuous policy RNG across repeats. Seed 5000 aligns policy RNG streams across checkpoints, but the existing wrapper does not seed Gym np_random, so initial environment states are not fully paired; environment seeding remains unchanged.

Episode standard error uses sample SD (ddof=1)/sqrt(n); 160 episodes per selected sampler.

action_oor_frac uses physical actions sent to Gym after the wrapper's float32 affine unnormalization; policy_action_oor_frac separately uses normalized policy coordinates.
