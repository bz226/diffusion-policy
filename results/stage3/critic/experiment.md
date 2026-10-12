# Plain SGD with a critic baseline

**Question.** At fixed SGD step sizes, how does replacing the LOO baseline with the learned critic change progress and noise, and can DPPO's estimator work with SGD?

**Status.** Paused at the user’s request. C1–C5 and calibration passed; all 15 training runs remain unstarted. Controller 17813817 has stopped and the critic monitor is paused. Gradient-mean collection and its monitor continue at the existing capacity. The proposed four-plus-three split was not activated; the unused scheduling patch was preserved and reverted. [Pause record](runs/user_pause_20261012/confirmation.json), [calibration evidence](calG_seed3001/calibration.json).

**Design.** SGDC3/SGDC10 use raw Monte Carlo returns minus the pre-update critic in raw units, with decision discounting and the E2 score estimator. SGDC3w additionally restores the existing 20-batch warmed critic/scaler. SGDG3/SGDG10 use E0's GAE and per-minibatch normalization, calibrated to the existing KL target. Each condition uses seeds 0–2, 140 iterations, 40 environments and the unchanged Stage-3 single-step skeleton. Existing SGD3/SGD10/E0/E2 runs are references and will not be rerun. [Methods](methods.md), [approval and exact scope](provenance/approval.json).

**Resources.** The aggregate cap is 60 GPU-hours, counting actual allocation time plus outstanding reservations. Training has a four-hour cap per run, gates two hours combined, calibration one hour, and checkpoint evaluation 15 minutes per run. Comparable runs project 35–40 GPU-hours and 11–14 hours at three concurrent A6000s, excluding waiting for G and the scheduler. G keeps priority; three-seed cohorts wait if they cannot fit without reducing its capacity. Every long job will have a 30-minute monitor; campaign checks are at most 2 hours 20 minutes apart.

**Next step.** Await an explicit user request to resume the critic campaign. Focus remains on gradient-mean. [Results and requested analysis](REPORT.md); curation requires a separate decision.

The stopped critic controller used no GPU. Gradient-mean retains its existing nine-allocation admission cap; no collection job was interrupted.

<!-- C runtime monitor -->
User pause confirmed: 2026-10-12T05:09:31.178509+00:00. Critic controller stopped; monitor paused; 0/15 training runs started. GPU-hours used: 0.2525; reserved: 0.000; hard cap: 60. Resume requires user instruction.
