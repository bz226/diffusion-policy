# Completed experiments

## Experiments

### DPPO HalfCheetah-v2 baseline reproduction

- Experiment report: [repro_halfcheetah](./results/repro_halfcheetah/experiment.md)
- Experiment design: [Methods and execution protocol](./results/repro_halfcheetah/methods.md)
- Curated results: [Artifact index](./results/repro_halfcheetah/curated/INDEX.md)

#### Analysis: three-seed evaluation/training curves and return table

- Analysis record: [three_seed_analysis](./results/repro_halfcheetah/runs/three_seed_analysis/candidates/summary.json)
- Analysis design: [Measurements and aggregation](./results/repro_halfcheetah/methods.md#measurements-and-aggregation)
- Curated results: [Plots, table and captions](./results/repro_halfcheetah/curated/INDEX.md)

### DPPO clipping ablations (HalfCheetah-v2)

- Experiment report: [stage2_noclip](./results/stage2_noclip/experiment.md)
- Experiment design: [Methods](./results/stage2_noclip/methods.md)
- Curated results: [Artifact index](./results/stage2_noclip/curated/INDEX.md)

### Stage 3 estimator, SGD and batch-size experiments

- Experiment report: [stage3](./results/stage3/experiment.md)
- Experiment design: [Methods and execution protocol](./results/stage3/methods.md)
- Curated results: [Final results and earlier snapshots](./results/stage3/curated/INDEX.md)

### Stage 3 frozen per-coordinate rescaling (FRS1)

- Experiment report: [stage3_frs](./results/stage3_frs/experiment.md)
- Experiment design: [Frozen-rescaling methods](./results/stage3_frs/methods.md)
- Curated results: [Plots, reviewed report and tables](./results/stage3_frs/curated/INDEX.md)

### Stage 3 Phase 1: fixed-policy gradient-noise measurements

- Experiment records: [Pretrained policy](./results/stage3/runs/ns_theta0_seed1000/manifest.json), [NC4 checkpoint](./results/stage3/runs/ns_ckpt_seed2000/manifest.json)
- Experiment design: [Fixed-policy noise measurements and uncertainty](./results/stage3/methods.md#fixed-policy-noise-measurements-and-uncertainty)
- Curated results: [Phase-1 plots, tables and limitations](./results/stage3/curated/INDEX.md#phase-1-fixed-policy-gradient-noise)

#### Analysis: point noise estimates and direction diagnostics

- Analysis record: [phase1_analysis](./results/stage3/runs/phase1_analysis/analysis_record.json)
- Analysis design: [Fixed-policy methods and reporting limits](./results/stage3/methods.md#fixed-policy-noise-measurements-and-uncertainty)
- Curated results: [Noise-scale and direction diagnostics](./results/stage3/curated/INDEX.md#phase-1-fixed-policy-gradient-noise)

### Stage 3 Monte Carlo returns with a critic under AdamW (E1C)

- Experiment: [E1C](./results/stage3/e1c/experiment.md) ([folder](./results/stage3/e1c/)).
- Experiment design: [Methods and execution protocol](./results/stage3/e1c/methods.md).
- Curated: —

## Comparison analyses

### FRS1 versus Stage 3 E2, E2_beta0 and SGD10

- Analysis record: [Verified final comparison](./results/stage3_frs/runs/final_analysis/review_record.json)
- Analysis design: [Comparisons and reporting rules](./results/stage3_frs/methods.md#analysis-and-deliverables)
- Curated results: [Plots, reviewed report and tables](./results/stage3_frs/curated/INDEX.md)

### Stage 3 final comparison, including BATCH4 and checkpoint evaluations

- Analysis record: [final_analysis_review](./results/stage3/runs/final_analysis_review/review_record.json)
- Analysis design: [Training comparisons](./results/stage3/methods.md#training-comparisons)
- Curated results: [Final plots, report and tables](./results/stage3/curated/INDEX.md#final-comparison)

### Stage 3 completed Wave-1 comparison against Stage 1 and NC4

- Analysis record: [completed_wave1_analysis](./results/stage3/runs/completed_wave1_analysis/analysis_record.json)
- Analysis design: [Training comparisons](./results/stage3/methods.md#training-comparisons)
- Curated results: [Returns, diagnostics, noise and results tables](./results/stage3/curated/INDEX.md#completed-wave-1-comparison)

### Stage 3 E0/E1/E2/SGD1/SGD3 comparison against Stage 1 and NC4

- Analysis record: [completed_five_analysis](./results/stage3/runs/completed_five_analysis/analysis_record.json)
- Analysis design: [Training comparisons](./results/stage3/methods.md#training-comparisons)
- Curated results: [Returns, update diagnostics and results tables](./results/stage3/curated/INDEX.md)

### Full Stage 2 comparison against the Stage 1 baseline

- Analysis record: [final_analysis](./results/stage2_noclip/runs/final_analysis/analysis_record.json)
- Analysis design: [Measurements and aggregation](./results/stage2_noclip/methods.md#analysis-and-deliverables)
- Curated results: [Full comparison plots, table and captions](./results/stage2_noclip/curated/INDEX.md#full-stage-2-comparison)

### NC4 learning-rate comparison against the Stage 1 baseline

- Analysis captions: [nc4_analysis](./results/stage2_noclip/runs/nc4_analysis/candidates/captions.md)
- Analysis design: [NC4 subset methods](./results/stage2_noclip/methods.md#nc4-subset-analysis-requested-before-campaign-completion)
- Curated results: [Artifact index and captions](./results/stage2_noclip/curated/INDEX.md#nc4-learning-rates)

### NC2/NC3 clipping comparison against the Stage 1 baseline

- Analysis captions: [nc23_analysis](./results/stage2_noclip/runs/nc23_analysis/candidates/captions.md)
- Analysis design: [NC2/NC3 subset methods](./results/stage2_noclip/methods.md#nc2-and-nc3-plots-before-full-stage-2-analysis)
- Curated results: [Artifact index and captions](./results/stage2_noclip/curated/INDEX.md#nc2-and-nc3)
