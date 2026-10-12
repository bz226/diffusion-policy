# Clean figures, version 3

**Scope.** Seed 0 only: DPPO, without_PPO and without_PPO_and_GAE. All three training runs completed; DPPO uses the original measured Stage-1 run, not a rerun. Gradient measurements pending complete 500-batch cohorts: C1, C2.

**Mapping.** P0: state_3000.pt → 0 updates; D1: state_70.pt → 63 updates; D2: state_139.pt → 126 updates; N1: state_70.pt → 63 updates; N2: state_139.pt → 126 updates; C1: state_70.pt → 63 updates; C2: state_139.pt → 126 updates. Each run independently verifies evaluation iteration 130 → 117 updates → 9.36M training steps; checkpoints save after the update phase.

**Reward checks.** DPPO: final seed-0 evaluation 4636.2733; without_PPO: final seed-0 evaluation 4624.0874; without_PPO_and_GAE: final seed-0 evaluation 4490.6881; every value matches its native log. Pooled pretrained evaluation/training means: 4298.783494 / 3806.733700 over 240 original episodes each. Original episode SDs were not saved, so their requested standard-error bars are unavailable and omitted; neither three-run SDs nor fresh rollouts were substituted.

**Measurement.** Main figures and table use the same earliest-launched ten V6-enabled jobs per completed point: K=500, m=40,000. All-available V4 estimates appear only as labelled table references. With ten groups, the 90% jackknife critical value is t(0.95,9)=1.833113. Unresolved squared signals remain bounds/N/A; greater precision from unmatched cohorts is not substituted.

**Display limits.** Main P0 appears once. V1 and V2 have different pretrained B values, so their overlay origins are omitted and both values remain in the table. V1/V2 use warmed, frozen critics and full-batch normalization in collection; they are proxies for the evolving, per-chunk-normalized training estimators. DPPO's nominal Gaussian sampler retains its ±3σ truncation. All artifacts remain uncurated candidates.

**Collection provenance.** The added points use their own checkpoint critics, five warm-up batches and frozen reward scalers, with unchanged V6 scientific source. The collector internally accepts only the original point labels, so C1/C2 use explicit D1/D2 aliases in separate output folders; original D data are never mixed. [Collection record](../runs/clean_v3_collection_20261012/plan.json) records source revisions, seeds, commands and the unchanged aggregate compute cap.

| Point | K | m | Groups | Selected jobs (launch order) |
|---|---:|---:|---:|---|
| P0 | 500 | 40000 | 10 | P0_j82;P0_j84;P0_j83;P0_j85;P0_j86;P0_j87;P0_j89;P0_j91;P0_j88;P0_j92 |
| D1 | 500 | 40000 | 10 | D1_j82;D1_j84;D1_j83;D1_j85;D1_j86;D1_j88;D1_j89;D1_j87;D1_j91;D1_j92 |
| D2 | 500 | 40000 | 10 | D2_j83;D2_j82;D2_j84;D2_j85;D2_j86;D2_j87;D2_j89;D2_j88;D2_j91;D2_j92 |
| N1 | 500 | 40000 | 10 | N1_j84;N1_j82;N1_j83;N1_j85;N1_j86;N1_j87;N1_j88;N1_j89;N1_j91;N1_j92 |
| N2 | 500 | 40000 | 10 | N2_j83;N2_j84;N2_j82;N2_j85;N2_j86;N2_j87;N2_j88;N2_j89;N2_j91;N2_j92 |
| C1 | pending | pending | pending | — |
| C2 | pending | pending | pending | — |

[Evaluation](eval_reward.png) · [Training](train_reward.png) · [Critic gradient noise](gradient_noise_V6.png) · [LOO](gradient_noise_V4.png) · [No baseline](gradient_noise_V5.png) · [Table](points_table.md). Captions accompany every figure; [render record](render_record.json) preserves exact inputs, intervals and checks.
