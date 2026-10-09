"""Estimate the next approved seed's runtime from completed native timings."""
import json
from pathlib import Path
import pickle
import statistics

root = Path(__file__).resolve().parents[1]
paths = []
for run_id in ('smoke', 'seed0_handoff', 'seed1', 'seed2'):
    manifest_path = root / 'runs' / run_id / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') == 'complete':
            paths.append(Path(manifest['result_path']))
completed = []
for path in paths:
    with path.open('rb') as f:
        rows = pickle.load(f)
    if len(rows) == 140:
        completed.append((path, rows))
if completed:
    estimate = statistics.mean(sum(float(r['time']) for r in rows) for _, rows in completed)
    basis = 'mean sum of native iteration times in completed full seeds'
    sources = [str(path) for path, _ in completed]
else:
    smoke = [(path, pickle.load(path.open('rb'))) for path in paths]
    smoke = [(path, rows) for path, rows in smoke if len(rows) == 2]
    if len(smoke) != 1:
        raise SystemExit('Expected exactly one verified two-iteration smoke result')
    path, rows = smoke[0]
    estimate = 14 * float(rows[0]['time']) + 126 * float(rows[1]['time'])
    basis = '14 evaluation and 126 training iterations, using measured smoke timings'
    sources = [str(path)]
record = dict(estimate_seconds=estimate, basis=basis, sources=sources,
              uncertainty='Smoke includes startup/warmup effects; later iterations may differ.')
with (root / 'provenance' / 'runtime_estimates.jsonl').open('a') as f:
    f.write(json.dumps(record) + '\n')
print(max(1, round(estimate)))
