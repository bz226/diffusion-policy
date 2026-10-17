# Part A: mean-gradient alignment

The existing data do not establish the ≥0.9 alignment premise at every own-critic checkpoint. The update-0 point is below that threshold, and all five intervals are wide enough to include 1. Thus this measurement gives no firm prediction that the two learning curves will retrace each other.

update 0: 0.783 [0.2787, 1.287]; update 32: 1.013 [0.8543, 1.171]; update 63: 1.015 [0.9241, 1.105]; update 95: 1.14 [0.801, 1.479]; update 126: 1.065 [0.7578, 1.373].

**Checks:** unresolved main cosines: D2, N1, N2. Common-mean intervals excluding 1: none. Unresolved common-mean checks: W0_035, W0_070, W0_139, P0, D1, N1, N2. A failed check blocks interpretation at that point; unresolved checks leave the sanity test inconclusive.

The [table](alignment_table.md) separates all eleven paired cohorts and records estimator differences, normalization and fitted-scale limitations. [Figure and caption](e1cg_alignment.caption.md). No new sampling or training was performed. The approved display retains estimates above 1; the nine-group cohort uses t₈.
