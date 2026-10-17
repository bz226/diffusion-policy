#!/usr/bin/env python
"""Re-render only approved outcome figures after reserving legend space."""
import ast
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[2]
project = root.parents[2]
os.environ['MPLCONFIGDIR'] = str(root / 'runs/cache/matplotlib')
module_path = root / 'code/analyze.py'
ast.parse(module_path.read_text())
spec = importlib.util.spec_from_file_location('e1cg_plot_layout', str(module_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output = root / 'runs/final_analysis/candidates'
names = ('e1cg_curves_seed0', 'e1cg_curves_3seed', 'e1cg_gain_by_segment')
protected = []
for directory in (root, output):
    protected += list(directory.glob('*.csv'))
    protected += [directory / name for name in ('REPORT.md', 'e1cg_table.md', 'segments_table.md', 'alignment_table.md', 'analysis_record.json') if (directory / name).is_file()]
    protected += list((directory / 'figs').glob('e1cg_alignment.*'))
    protected += list((directory / 'figs').glob('*.caption.md'))
protected += [root / ('E1CG_seed%d/result.pkl' % seed) for seed in range(3)]
protected += [project / ('results/stage3/e1c/E1C_seed%d/result.pkl' % seed) for seed in range(3)]
def contents(paths):
    return {str(path): path.read_bytes() for path in paths}
before = contents(protected)
runs = []
for condition in ('E1C', 'E1CG'):
    for seed in range(3):
        directory = (project / 'results/stage3/e1c' if condition == 'E1C' else root) / ('%s_seed%d' % (condition, seed))
        runs.append(module.load_run(condition, seed, directory))
with (root / 'segments_table.csv').open() as stream:
    segments = list(csv.DictReader(stream))
for row in segments:
    row['seed'] = int(row['seed'])
    row['checkpoint_itr'] = int(row['checkpoint_itr'])
    for index in range(5):
        row['delta_bin%d' % index] = float(row['delta_bin%d' % index])
module.figures(runs, segments, output)
for name in names:
    for suffix in ('.png', '.pdf'):
        shutil.copy2(output / 'figs' / (name + suffix), root / 'figs' / (name + suffix))
after = contents(protected)
assert before == after, 'Protected saved evidence or report changed'
record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
              command=[sys.executable, '-B', str(Path(__file__).resolve())],
              change='Plot-only layout: reserve upper nine percent for the figure legend; no data or scientific calculation changed.',
              figures=list(names), python=sys.version, protected_files_unchanged=len(before),
              protected_files=list(before),
              training_source_changed=False, jobs_launched=False)
(root / 'runs/plot_layout_fix/record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: record[key] for key in ('figures', 'python', 'protected_files_unchanged')}))
