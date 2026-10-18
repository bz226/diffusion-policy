Seed 0 per optimizer; 80 episodes per point. Estimator: without_PPO_and_GAE + γ^k; only the actor optimizer changes. J_disc is the training batch's mean of Σ_k 0.99^k R_k in raw rewards. Evaluation keeps the DDPM's noise at every denoising step except the last. Points show estimates without uncertainty bands; seed variation is in the [results table](adagrad-norm-results-table.md).

[Source caption](../figs/an_curves_seed0.caption.md) · [Methods](../methods.md) · [Analysis inputs](../analysis_record.json).
