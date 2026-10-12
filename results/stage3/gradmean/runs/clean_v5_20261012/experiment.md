# Version 5 own-critic gradient collection

Status: **running**; 34/56 jobs verified. The five points use their exact checkpoint critic without warm-up; only the final divisor uses the authorized five-batch fit. The full/quarter normalization choice is frozen after the two-batch comparison, before collection.

MAX_PAR=5; 30-minute job caps and 125 aggregate GPU-hours. Scheduled monitor: every 30 minutes. Failures stop admissions; already-running jobs are left unchanged. No automatic retries.

[Plan](plan.json) · [Current ledger](campaign.json) · [Checkpoint/scaler audit](log_audit.json).
