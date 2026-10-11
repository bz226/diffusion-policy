# Candidate captions

## Eight metrics

FRS1 versus completed E2, E2_beta0 and SGD10: the eight panels use raw saved metrics against training environment steps. Thin curves are seeds 0/1/2; bold means and bands use sample SD (ddof=1) only where all three finite values exist, without smoothing or extrapolation. B_env is 10 times summed split-gradient squared difference divided by summed split dot over complete nine-training-iteration windows; undefined windows are gaps, crosses denote nonpositive denominators, and B_env=40 is a descriptive guide under the approximate independent-environment interpretation. Log axes omit nonpositive points/lower SD bounds without epsilon substitution; saturation and action violation use training rows only. Final in-training evaluation is itr130 at9.36M steps, whereas training ends at10.08M; split estimates are batch-coupled and three runs do not establish a general ordering.

## Ratios and KL

FRS1 coordinate-gradient ratios and sampled transition KL versus training iteration, with individual seeds and mean ± sample SD across all three seeds. max_ratio and median_ratio are the maximum and global midpoint median of |loss gradient|/frozen denominator before SGD; the median is over coordinates, not seeds. The aligned panels expose temporal coincidence with KL excursions without defining a new spike threshold or a causal test; raw per-seed maxima, temporal medians and the ratios at each seed's maximum-KL iteration are in ratio_extrema.csv. Nonpositive log values and lower SD bounds are omitted without replacement, failed prefixes remain visible, and missing values never receive survivor-only means.

## Tables

Per-seed and condition tables mirror Stage3 definitions; all means/sample SD require three finite seed summaries, and failed runs retain finite prefixes with unavailable endpoints. Last10 uses the final ten training iterations only for completed140-row runs; final checkpoint evaluations use both samplers and delta-J subtracts saved theta0 means. Collapse is an observed evaluation below half that seed's initial evaluation or a numerical nonfinite failure; undefined diagnostics alone are not collapse. Timing includes native mean training-iteration time and process wall time; repeated diagnostic observations are not independent seeds. Comparisons use twice sqrt((s_FRS1²+s_E2_beta0²)/2), not an SE or significance test; old theta0 has three evaluation repeats, while FRS final evaluation has one, and Gym initial states are not paired.

## Frozen RMS statistics

Statistics are copied from the passed gate's deterministic conversion of25 saved V4 batches, without new sampling. Quantiles are raw coordinate RMS; floor counts use strict pre-epsilon comparison and the top1% share uses ceil(.01P) coordinates. They do not repair or replace the prior Phase1 signal-norm confidence intervals.
