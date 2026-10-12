# Three-seed reward comparison

Each cell is mean ± sample sd (ddof = 1); brackets give seed 0, 1, 2. Each method retains its own mean at x = 0. Training is averaged within each seed over its last ten training updates (iterations 129 and 131–139), then summarized across seeds. No smoothing is applied.

| Method | Evaluation at 9.36M | Last-ten-update training | Initial evaluation | Initial training |
|---|---:|---:|---:|---:|
| DPPO | 4758.37 ± 108.29<br>[4636.27, 4796.04, 4842.80] | 4476.37 ± 87.71<br>[4378.46, 4502.90, 4547.76] | 4341.98 ± 49.93<br>[4286.90, 4384.28, 4354.75] | 3877.93 ± 63.49<br>[3841.50, 3841.06, 3951.24] |
| without_PPO | 4603.52 ± 19.14<br>[4624.09, 4586.22, 4600.26] | 4311.27 ± 10.77<br>[4303.55, 4323.58, 4306.69] | 4319.81 ± 47.28<br>[4316.40, 4274.33, 4368.70] | 3789.53 ± 153.34<br>[3614.22, 3855.66, 3898.72] |
| without_PPO_and_GAE | 4503.88 ± 13.36<br>[4490.69, 4517.41, 4503.55] | 4207.54 ± 48.41<br>[4217.93, 4249.92, 4154.78] | 4312.55 ± 18.42<br>[4293.06, 4314.92, 4329.67] | 3971.66 ± 27.61<br>[3964.48, 4002.15, 3948.34] |

DPPO matches the required 4,758 ± 108 check. All nine original runs finished 140 iterations, and all 1,260 reward means match their native logs. The final evaluation is iteration 130, after 117 updates = 9.36M training steps.

| Difference (second minus first) | Evaluation difference ± s.e. | Last-ten-training difference ± s.e. |
|---|---:|---:|
| without_PPO − DPPO | -154.85 ± 63.49 | -165.11 ± 51.02 |
| without_PPO_and_GAE − DPPO | -254.49 ± 63.00 | -268.83 ± 57.84 |
| without_PPO_and_GAE − without_PPO | -99.64 ± 13.48 | -103.73 ± 28.64 |

The difference s.e. is sd_pooled × √(2/3), where sd_pooled² = (sd_first² + sd_second²)/2. These are the requested unpaired seed-level standard errors, not episode-level errors or confidence intervals; only three seeds are available.

| Method | Seed 0 run record | Seed 1 run record | Seed 2 run record |
|---|---|---|---|
| DPPO | [seed 0](../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_20-01-32_0/result.pkl) | [seed 1](../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_22-06-13_1/result.pkl) | [seed 2](../../../../repro_halfcheetah/dppo/log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-10-08_22-48-33_2/result.pkl) |
| without_PPO | [seed 0](../../../../stage2_noclip/runs/NC4_lr1e-4_seed0/native/result.pkl) | [seed 1](../../../../stage2_noclip/runs/NC4_lr1e-4_seed1/native/result.pkl) | [seed 2](../../../../stage2_noclip/runs/NC4_lr1e-4_seed2/native/result.pkl) |
| without_PPO_and_GAE | [seed 0](../../../e1c/E1C_seed0/result.pkl) | [seed 1](../../../e1c/E1C_seed1/result.pkl) | [seed 2](../../../e1c/E1C_seed2/result.pkl) |

Original Stage-1 runs are used, including seed 0's successfully completed watchdog handoff; it did not restart training. No requested run was missing or replaced. DPPO retains its original ±3σ truncation of nominal Gaussian noise. The single-seed PNGs, PDFs and captions passed before/after SHA-256 equality checks. No training, gradient analysis or running-job change was performed.

[Evaluation PNG](evaluation-reward-three-seeds.png) · [PDF](evaluation-reward-three-seeds.pdf) · [Training PNG](training-reward-three-seeds.png) · [PDF](training-reward-three-seeds.pdf) · [Exact inputs, values and checks](../../runs/clean_rewards_3seed_20261012T193300.153408Z/candidates/reward_3seed_record.json).
