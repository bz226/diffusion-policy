# Exact-critic gradient measurements

Status: **complete**. All five preparations, the two-batch normalization check and fifty 50-batch collections passed saved-output, scheduler and cleanup checks. Every point has 500 fresh batches (40,000 episodes); no actor or critic updates occurred. The final divisor used the approved five-batch fit.

[Gradient figures, table and interpretation](../../figs_clean/NOTE.md) · [Independent checks](final_analysis/artifact_check.json) · [Methods](../../methods.md#clean-figures-version-5).

Normalization was **full**, frozen before collection. Aggregate GPU time is 80.412 hours against the 125-hour cap. Existing single-seed and three-seed reward artifacts and all curated files were preserved byte for byte. No outputs were automatically curated.
