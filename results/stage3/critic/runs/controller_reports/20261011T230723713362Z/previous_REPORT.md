# Addendum C: critic baseline with plain SGD

Scientific source is committed at `052c461`, with the reviewed queue and analysis at `f45eb5f`, on the isolated `stage3-critic` branch. Small Python 3.8 checks passed: 14 scientific helper checks, 46 runtime checks and 33 analysis checks. These do not substitute for the formal C1–C5 gate, running as Slurm job 17813670 on an A6000; no C training runs have started.

C1 (raw-unit critic advantages), C2 (iteration-start baseline state) and C3 (warmed initialization) passed. Diagnostic-inertness and unchanged-default replay checks are still running. A [bounded monitor](runs/gate_continuation.json) will advance automatically only after the gate passes and its allocation exits with code 0. The approved queue uses the existing KL*=0.000584517336086918 and E2 eta*=0.00012107235674859004, while calG determines eta_G from five fresh batches. G retains priority.

[Gate evidence](runs/gate/gate_summary.json) · [Methods](methods.md) · [Current status](experiment.md) · [Budget and scope](provenance/approval.json). Results, tables and Q1–Q5 remain pending. The cap is 60 GPU-hours, and no curation is authorized.

Queue-test bookkeeping issue: a synthetic fixture was written to the live campaign-state path before any production was admitted. It is preserved under [unit-test evidence](runs/unit_runtime_C/accidental_campaign_fixture.json); test isolation is corrected and verified before enabling continuation. The scientific gate and source are unaffected.
