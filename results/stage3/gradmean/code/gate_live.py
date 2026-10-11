#!/usr/bin/env python
"""Verify G3/G4 and extended clean-exit gate from two finished P0 groups."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from campaign import ROOT, validate_gate, verify


def require(value, message):
    if not value:
        raise AssertionError(message)


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(ROOT in args.output.resolve().parents and not args.output.exists(),
            'Preserve evidence: use a fresh gate directory within Addendum G')
    args.output.mkdir(parents=True)
    offline = load(args.offline)
    require(offline.get('passed') is True and offline.get('stage') == 'offline',
            'Offline G1/G2/G5 must pass first')
    report = dict(stage='full', passed=False, live_gate_passed=False, status='running',
                  offline_gate=str(args.offline.resolve()), parent_commit=offline['parent_commit'],
                  source_snapshot=offline['source_snapshot'], G1=offline['G1'], G2=offline['G2'],
                  G5=offline['G5'], command=sys.argv, started_unix=time.time())
    try:
        validate_gate(args.offline, offline['parent_commit'])
        entries = []
        for run_id in ('prepare_P0', 'P0_j00', 'P0_j01'):
            manifest_path = ROOT / 'runs' / run_id / 'manifest.json'
            manifest = load(manifest_path)
            require(manifest.get('status') == 'complete' and manifest.get('subprocess_exitcode') == 0,
                    run_id + ' did not complete with process exit0')
            require(manifest.get('remaining_process_group_members') == [],
                    run_id + ' left subprocess-group members')
            verification = verify(manifest)
            job_id = manifest['scheduler_job_id']
            command = ['sacct', '--allocations', '--noheader', '--parsable2', '--jobs=' + job_id,
                       '--format=JobIDRaw,State,ExitCode,ElapsedRaw']
            accounting = subprocess.check_output(command, text=True, timeout=120)
            rows = [line.split('|') for line in accounting.splitlines() if line.startswith(job_id + '|')]
            require(len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0'],
                    run_id + ' is not Slurm COMPLETED exit0:0: ' + accounting)
            directory = Path(manifest['logdir'])
            meta = load(directory / ('grad_mean.json' if run_id == 'prepare_P0' else 'meta.json'))
            cleanup = load(directory / 'cleanup.json')
            require(len(cleanup['workers']) == 40, run_id + ' must explicitly close forty env workers')
            entries.append(dict(run_id=run_id, manifest=str(manifest_path), verification=verification,
                                scheduler_command=command, scheduler_output=accounting, meta=meta,
                                cleanup_path=str(directory / 'cleanup.json'),
                                closed_workers=len(cleanup['workers']), seconds=manifest['elapsed_seconds']))
        left, right = entries[1]['meta'], entries[2]['meta']
        hashes = ('frozen_sha256', 'critic_before_sha256', 'scaler_before_sha256', 'actor_before_sha256')
        require(all(left[key] == right[key] and left[key] for key in hashes),
                'Collect jobs loaded different frozen actor/critic/scaler states')
        require(left['seed'] != right['seed'], 'Live collect jobs must have distinct seeds')
        observations_differ = not np.array_equal(left['first_observations'], right['first_observations'])
        returns_differ = not np.array_equal(left['first_episode_returns'], right['first_episode_returns'])
        require(observations_differ and returns_differ, 'Independent jobs reused initial observations or returns')
        actor_inert = all(e['meta']['actor_before_sha256'] == e['meta']['actor_after_sha256'] for e in entries)
        require(actor_inert, 'Actor changed during fixed-policy measurement')
        report['G3'] = dict(passed=True, identical_frozen_states={key: left[key] for key in hashes},
                            seeds=[left['seed'], right['seed']], first_observations_differ=observations_differ,
                            first_returns_differ=returns_differ, process_exit_codes=[0, 0, 0],
                            surviving_children=[], records=[{k: v for k, v in e.items() if k != 'meta'} for e in entries])
        report['G4'] = dict(passed=True, actor_parameter_hash_unchanged=True,
                            actor_hash=left['actor_before_sha256'])
        report.update(passed=True, live_gate_passed=True, status='passed')
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        report['ended_unix'] = time.time()
        (args.output / 'gate_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, gate=str(args.output / 'gate_summary.json'))), flush=True)


if __name__ == '__main__':
    main()
