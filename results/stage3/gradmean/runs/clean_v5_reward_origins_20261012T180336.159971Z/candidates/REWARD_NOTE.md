# Reward figures: each run starts from its own value

Each curve uses its own saved mean at zero training steps: iteration 0 for evaluation and iteration 1 for training. There is one seed per method (seed 0), 80 episodes per point, no pooling, no bands and no smoothing.

| Method | Initial evaluation | Evaluation s.e. | Initial training | Training s.e. | Final evaluation at 9.36M | Matches own log |
|---|---:|---:|---:|---:|---:|---|
| DPPO | 4286.896957 | unavailable | 3841.496354 | unavailable | 4636.273298 | yes |
| without_PPO | 4316.396299 | unavailable | 3614.221929 | unavailable | 4624.087352 | yes |
| without_PPO_and_GAE | 4293.057226 | unavailable | 3964.482818 | unavailable | 4490.688112 | yes |

The requested standard errors (episode sd / √80) cannot be recovered: the original undiscounted episode returns, SDs and sums of squares were not saved. Discounted-return errors and dispersion across run means are not substitutes. This is the only unavailable requested quantity.

The final evaluation is iteration 130, after 117 updates × 80,000 training environment steps = 9.36M. DPPO uses the original Stage-1 seed-0 run. DPPO retains its original ±3σ truncation of nominal Gaussian sampling noise.

This reward-only note supersedes earlier pooled-start reward descriptions. Gradient measurements, gradient figures, the checkpoint table and existing curated copies are unchanged. A separate reward-only refresh follows the existing finalizer so its archived renderer cannot leave the old reward figures in place.

[Evaluation PNG](eval_reward.png) · [PDF](eval_reward.pdf) · [Training PNG](train_reward.png) · [PDF](train_reward.pdf) · [Saved inputs and provenance](reward_render_record.json).
