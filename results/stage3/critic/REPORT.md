# Addendum C: critic baseline with plain SGD

**Status.** Paused at the user’s request. C1–C5 and calibration passed; all 15 training runs remain unstarted. Controller 17813817 and its monitor have stopped. Gradient-mean continues unchanged, and the four-plus-three split was never activated. Resume requires an explicit user request. [Pause confirmation](runs/user_pause_20261012/confirmation.json).

Five calibration batches gave eta*_G = 0.37436859, with max/min spread 1.581 and KL exponents 1.891–2.007. The realized trial KL range was 0.00058382–0.00059284 against target 0.00058452. No calibration warning was raised. Job 17813978 exited 0:0 after 126 allocated seconds, with all environment workers closed. These are calibration results, not evidence about training performance.

The queue will use the prescribed SGDG rates 3 eta*_G = 1.12310578 and 10 eta*_G = 3.74368593. SGDC uses the unchanged Stage-3 E2 rates. Gates and calibration have used 0.2525 GPU-hours against the 60 GPU-hour cap; no experiment has been retried.

[Calibration](calG_seed3001/calibration.json) · [Gate evidence](runs/gate/gate_summary.json) · [Current status](experiment.md) · [Methods](methods.md) · [Dated monitoring evidence](runs/manual_monitor_history.jsonl). Training tables, figures and Q1–Q5 remain pending; outputs remain uncurated.
