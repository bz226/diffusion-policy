#!/usr/bin/env python
"""Read-only E1C launch gate validation; no scientific or scheduler actions."""
import argparse
import json
from pathlib import Path


def check(path):
    result = json.loads(path.read_text())
    if result.get('status') != 'passed' or result.get('passed') is not True:
        raise RuntimeError('E1C gate has not passed')
    for name in ('A1', 'A2', 'A3', 'A4', 'A5'):
        if result['checks'][name].get('passed') is not True:
            raise RuntimeError(name + ' did not pass')
    if result.get('cleanup', {}).get('all_children_exited') is not True:
        raise RuntimeError('E1C gate cleanup not verified')
    if not result['source_comparison_after']['passed']:
        raise RuntimeError('E1C gate source mismatch')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path, nargs='?', default=Path('/soalnas/share/data/zbao7/diffusion_policy/results/stage3/e1c/runs/gate/gate_summary.json'))
    args = parser.parse_args()
    result = check(args.path)
    print('E1C A1-A5 passed; CPU gate %.1f seconds; A2/A4/A5 reused from unchanged critic source.' % result['elapsed_seconds'])
