"""Finalize already-passed EP gate records without importing training packages.

This is bookkeeping only: no gate execution, sampling, gradients or submission.
Python 3.8 compatible. The original failed controller and gate records stay intact.
"""
import json
from pathlib import Path

import campaign
from worker import atomic_json, read_json, utc, verify_source


def combine(cpu, live, source, accounting):
    if not (cpu.get('passed') and live.get('passed')):
        raise ValueError('Cannot finalize failed or unfinished gates')
    if cpu.get('phase') != 'cpu' or live.get('phase') != 'live':
        raise ValueError('Wrong gate phase')
    if cpu['source'] != live['source'] or cpu['source'] != source:
        raise ValueError('Gate source paths differ')
    checks = dict(cpu['checks'], **live['checks'])
    if set(checks) != {'G1', 'G2', 'G3', 'G4', 'G5'}:
        raise ValueError('Missing gate')
    if not all(value.get('passed') for value in checks.values()):
        raise ValueError('One gate did not pass')
    for phase, limit, gpu_count in (('cpu', 7200, 0), ('gpu', 3600, 1)):
        info = accounting[phase]
        if info['state'] != 'COMPLETED' or info['exit_code'] != '0:0':
            raise ValueError('Unsuccessful gate allocation: ' + phase)
        if info['elapsed_seconds'] > limit or info['gpu_count'] != gpu_count:
            raise ValueError('Gate allocation exceeded approved resources: ' + phase)
    return dict(passed=True, status='passed', phase='merge', source=source,
                original_source=cpu['original_source'],
                original_upstream_commit=cpu['original_upstream_commit'],
                fixture=cpu['fixture'], checks=checks, commands=[],
                cpu_gate_seconds=max(accounting['cpu']['elapsed_seconds'], cpu['cpu_gate_seconds']),
                gpu_gate_seconds=max(accounting['gpu']['elapsed_seconds'], live['gpu_gate_seconds']),
                phase_wall_seconds=dict(cpu=cpu['cpu_gate_seconds'], gpu=live['gpu_gate_seconds']),
                scheduler_accounting=accounting,
                phase_numeric_settings=dict(cpu=cpu['numeric_settings'], live=live['numeric_settings']),
                finalized_utc=utc(), finalizer='code/finalize_passed_gates.py',
                scientific_gates_rerun=False)


def main():
    root = campaign.ROOT
    target = root / 'runs/gates/complete/gate_summary.json'
    if target.parent.exists():
        raise RuntimeError('Refuse to overwrite gate output')
    if (root / 'runs/plan_training.json').exists():
        raise RuntimeError('Training plan already exists; inspect before proceeding')
    source = read_json(root / 'provenance/launch_source.json')
    verified = verify_source(source)
    cpu_path = root / 'runs/gates/cpu/gate_summary.json'
    live_path = root / 'runs/gates/live/gate_summary.json'
    cpu, live = read_json(cpu_path), read_json(live_path)
    ids = [cpu['slurm_job_id'], live['slurm_job_id']]
    allocations = campaign.scheduler_rows(ids)
    accounting = dict(cpu=allocations[ids[0]], gpu=allocations[ids[1]])
    report = combine(cpu, live, source['source'], accounting)
    report['phase_records'] = [str(cpu_path), str(live_path)]
    report['source_verification'] = verified
    # Confirm that every recorded worker, including explicit cleanup, succeeded.
    workers = {}
    for phase, record in (('cpu', cpu), ('live', live)):
        for worker in record['workers']:
            item = read_json(root / 'runs/gates' / phase / worker['name'] / 'worker.json')
            if worker['exit_code'] != 0 or not item.get('passed') or not item.get('all_children_exited'):
                raise ValueError('Worker or cleanup failed: ' + worker['name'])
            workers[worker['name']] = dict(passed=True, all_children_exited=True)
    report['worker_verification'] = workers
    # A failed phase or check cannot be converted into a pass by this helper.
    bad = dict(cpu, passed=False)
    try:
        combine(bad, live, source['source'], accounting)
    except ValueError:
        pass
    else:
        raise AssertionError('Failed-phase rejection check failed')
    atomic_json(target, report)
    print(json.dumps(dict(passed=True, output=str(target), source_verification=verified,
                          cpu_gate_seconds=report['cpu_gate_seconds'],
                          gpu_gate_seconds=report['gpu_gate_seconds']), indent=2))


if __name__ == '__main__':
    main()
