#!/usr/bin/env bash
set -euo pipefail
source /soalnas/share/data/zbao7/diffusion_policy/results/repro_halfcheetah/code/environment.sh
estimate=$(python - <<'PY'
import datetime, json, os, time
from pathlib import Path
record = json.loads((Path(os.environ['REPRO_ROOT']) / 'runs/seed0/manifest.json').read_text())
elapsed = time.time() - datetime.datetime.fromisoformat(record['start_utc']).timestamp()
print(max(1, int(record['estimate_seconds'] - elapsed)))
PY
)
set -x
exec python "$REPRO_ROOT/code/adopt_running_run.py" --perform-handoff --watchdog-pid 1598871 --dppo-pid 1598874 --expected-start-ticks 56792860 --pipe-fd 3 --remaining-estimate-seconds "$estimate"
