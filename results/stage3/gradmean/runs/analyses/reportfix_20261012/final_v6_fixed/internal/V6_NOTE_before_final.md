# V6: the paper estimator with a frozen critic baseline

**Question.** How does the frozen critic change gradient variance relative to the leave-one-out and no-baseline estimators along the DPPO and without_PPO paths?

**Status.** Implementation and CPU H1–H4 checks passed in the isolated `stage3-gradmean-v6` worktree. Controller `17821491` admitted the first five V6 groups after the verified source switch at 08:28:50 UTC on October 12. E1C continues on its original source. Previously admitted collection jobs finished without interruption.

**Design.** V6 uses raw Monte Carlo returns minus the frozen critic converted to raw reward units, weighted by the within-episode discount. Each policy will have exactly 500 new V6 batches. Comparisons involving V6 use its actual cohort; its critical-batch-size denominator uses all V4 batches. See [methods](methods.md#addendum-g2-paper-estimator-with-the-frozen-critic), [scope](provenance/amendment_G2.json), and [initial revision](provenance/v6/scope.json).

**Next step.** Finish the fixed collection from commit `ce909d3`, then publish the cohort-matched analysis. The user replaced the original adaptive stopping plan for compute reasons before new V6 data: admitted jobs finish, then exactly ten new groups per point run and collection stops. The 125 GPU-hour aggregate cap and E1C priority remain. H1 relative error was 2.58×10⁻¹⁵; H2 gradient identities were exact; H3 preserved V1–V5 and RNG states bitwise; H4 preserved the frozen state. [Switch provenance](provenance/v6_source_switch.json) records both commits; completed measurements and Q1–Q3 remain pending.
