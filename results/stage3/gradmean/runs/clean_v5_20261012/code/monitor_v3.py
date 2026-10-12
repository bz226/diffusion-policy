"""Read-only completion/accounting monitor for already-admitted v3 workers.

This script cannot submit, cancel, resume, modify or restart scientific jobs.
It writes only new v5 administrative evidence; old v3 results remain intact.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path('/soalnas/share/data/zbao7/diffusion_policy')
HERE = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/stage3/gradmean/runs/clean_v3_collection_20261012'
module_spec = importlib.util.spec_from_file_location('v3_runtime_readonly', str(OLD / 'code/collection.py'))
v3 = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(v3)
v3.old.JOURNAL = HERE / 'v3_winddown_commands.jsonl'
INITIAL_BASE_GPU_SECONDS = 244995
CAP_GPU_SECONDS = 450000
INTERVAL_SECONDS = 1800
MONITOR_CAP_SECONDS = 4500


def audit():
    campaign = v3.old.read(OLD / 'campaign.json')
    if campaign.get('status') != 'blocked' or campaign.get('controller_job_id') != '17825414':
        raise RuntimeError('Unexpected original campaign state; read-only audit stops')
    control = v3.old.accounting('17825414')
    cancelled_final_plot = v3.old.accounting('17825457')
    entries = []
    for original in campaign['entries']:
        entry = {key:original.get(key) for key in ('run_id', 'job_id', 'submitted_utc')}
        status = v3.old.accounting(entry['job_id'])
        entry.update(status)
        manifest = v3.old.read(OLD / 'workers' / entry['run_id'] / 'manifest.json', {})
        spec = v3.old.read(OLD / 'specs' / (entry['run_id'] + '.json'))
        observed = v3.old.observe(spec)
        entry.update(worker_status=manifest.get('status'), completed_batches=observed['completed_batches'],
                     native_status=observed['status'], native_error=observed['error'])
        if status.get('scheduler_state') in v3.old.TERMINAL:
            if status['scheduler_state'] == 'COMPLETED' and status.get('exit_code') == '0:0' and manifest.get('status') == 'complete':
                entry['verification'] = v3.verify(spec)
                entry['verified_complete'] = True
            else:
                entry['verified_complete'] = False
                entry['failure_reason'] = manifest.get('failure_reason', 'Scheduler or native worker did not complete successfully')
        entries.append(entry)
    live = [e for e in entries if e.get('scheduler_state') not in v3.old.TERMINAL]
    actual = sum(e.get('allocated_gpu_seconds', 0) for e in entries)
    complete_actual = sum(e.get('allocated_gpu_seconds', 0) for e in entries if e.get('scheduler_state') in v3.old.TERMINAL)
    reserve = len(live) * 3600
    result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), controller=control,
                  cancelled_final_plot=cancelled_final_plot, entries=entries,
                  running_job_ids=[e['job_id'] for e in live], all_workers_terminal=not live,
                  all_workers_verified=not live and all(e.get('verified_complete') for e in entries),
                  not_launched_run_ids=[name for name in v3.RUN_IDS if name not in [e['run_id'] for e in entries]],
                  collection_admissions_stopped=True, old_results_modified=False,
                  ledger=dict(base_gpu_seconds=INITIAL_BASE_GPU_SECONDS,
                    v3_actual_gpu_seconds_as_observed=actual,
                    aggregate_actual_gpu_seconds_as_observed=INITIAL_BASE_GPU_SECONDS + actual,
                    aggregate_completed_plus_full_running_reservations=INITIAL_BASE_GPU_SECONDS + complete_actual + reserve,
                    aggregate_gpu_seconds_cap=CAP_GPU_SECONDS,
                    remaining_gpu_seconds_after_observed_actual=CAP_GPU_SECONDS - INITIAL_BASE_GPU_SECONDS - actual,
                    is_final=not live))
    v3.old.atomic(HERE / 'v3_winddown_latest.json', result)
    v3.old.append(HERE / 'v3_winddown_monitor_history.jsonl', result)
    if not live:
        previous = v3.old.read(HERE / 'v3_final_ledger.json')
        if previous is None:
            v3.old.atomic(HERE / 'v3_final_ledger.json', result)
        elif previous['ledger'] != result['ledger'] or previous['all_workers_verified'] != result['all_workers_verified']:
            raise RuntimeError('Previously final accounting changed; preserve both audit records for review')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    while True:
        result = audit()
        print(json.dumps({key:result[key] for key in ('recorded_utc', 'running_job_ids', 'all_workers_terminal', 'all_workers_verified', 'ledger')}, indent=2), flush=True)
        if result['all_workers_terminal']:
            return 0 if result['all_workers_verified'] else 1
        if not args.monitor:
            return 0
        if time.monotonic() - started + INTERVAL_SECONDS > MONITOR_CAP_SECONDS:
            raise RuntimeError('Read-only monitor cap reached; no jobs changed')
        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    sys.exit(main())
