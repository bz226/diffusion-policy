# Addendum G stopped at the synthetic gate

**Question.** Can the proposed estimators support reliable gradient-direction measurements before collecting along the DPPO and NC4 paths?

**Answer.** G2 failed: one of200 synthetic replicates had negative split-half correlation, making the prescribed cosine transform undefined. No policy preparation, environment collection or GPU gate was launched. Q1–Q10 about the real training paths remain unanswered by this addendum.

## Gate evidence

The predeclared setup used20,000 coordinates,2,000 batches in40 groups,200 replicates, squared mean1 and covariance trace4000. [Exact setup](g2_predeclared.json), [saved results](runs/gate_G2/gate_summary.json).

| Check | Observed | Requirement | Result |
| --- | ---: | --- | --- |
| Mean squared-signal estimate |0.996961; Monte Carlo SE0.033564|Truth1 within3 SE|Pass|
| Mean noise trace |4000.11165; Monte Carlo SE1.16112|Truth4000 within3 SE|Pass|
| Signal90% interval coverage |193/200 =96.5%|80–97%|Pass|
| Trace90% interval coverage |180/200 =90.0%|80–97%|Pass|
| Mean effective rank |61.62489 vs truth61.69222|Relative error≤20%|Pass (−0.109%)|
| Mean direction-cosine check |199/200 estimates defined|All-replicate mean error≤0.03|Fail: mean unavailable|

Replicate125 (zero-based) had rho=−0.00706345 and squared signal=−0.05412045, with90% jackknife interval[−1.00596,0.89772]. Its realized cosine was0.505914. The transform sqrt(2*rho/(1+rho)) has no real value there; no clipping, replacement or exclusion was applied.

Among the199 defined cases only, mean estimated/realized cosines were0.559505/0.582064; mean absolute per-replicate error was0.090965. This conditional comparison does **not** replace the failed all200 gate. [Independent saved-record review](runs/gate_G2/failure_review.json).

A secondary Python3.8 cleanup error (`shutdown(cancel_futures=True)`) occurred after all200 results and the scientific failure were saved. It did not cause the negative correlation or alter the statistics. The [log](setup/G2-17812088.out) preserves both failures; neither was retried.

## Execution and remaining work

Only CPU Slurm job17812088 ran:20 allocation seconds,40 CPUs, no GPU, against the125 GPU-hour cap. Synthetic computation took18.9s. The job is terminal; no collection jobs or monitors remain active. G1/G3/G4/G5 were not executed. Deterministic algebra checks and small collector software checks passed, but these do not establish the scientific gate or default-path equivalence.

All five checkpoints are available in the [checkpoint registry](provenance/checkpoints.json), so no DPPO rerun was needed. Distances and path tables/figures were not computed after the stop. Source is on `stage3-gradmean`, starting at `23f57ef4acee74846e35af68fa43076d5c2afcb6`; implementation/evidence commit `0ae03b5db30198b873ee82d5e991e28aa51fc817` is recorded in [revisions](provenance/revisions.json). The executed G2 code is preserved in [its source snapshot](setup/g2_source/gate_synthetic.py). The collector/statistics implementation is prepared; scheduling and full analysis are unfinished and unvalidated. [Methods](methods.md), [approval](provenance/approval.json), [exact submission command](provenance/G2_submission.json).

**Decision.** Collection is blocked by the prescribed gate. Changing the handling of undefined direction estimates or the gate requires a new explicit decision; no threshold, setup or stopping rule was changed. Existing results remain untouched, and no new artifacts were curated.
