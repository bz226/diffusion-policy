"""Read-only thirty-minute monitor for the explicitly approved CPU gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/clean_v5_20261012')
JOB = '17825755'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def check(previous=None):
    command = ['sacct', '--allocations', '--noheader', '--parsable2', '--jobs=' + JOB,
               '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%180,ExitCode']
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    if process.returncode:
        raise RuntimeError(process.stderr)
    fields = next((row.split('|') for row in process.stdout.splitlines() if row.startswith(JOB + '|')), None)
    if fields is None:
        status = dict(scheduler_state='UNKNOWN')
    else:
        tres = dict(part.split('=', 1) for part in fields[3].split(',') if '=' in part)
        status = dict(scheduler_state=fields[1].split()[0], elapsed_seconds=int(fields[2]),
                      allocated_tres=fields[3], gpu_count=int(tres.get('gres/gpu', 0)), exit_code=fields[4])
    gate = read(ROOT / 'gate_cpu/gate_summary.json')
    workers = {p.parent.name:read(p) for p in (ROOT / 'gate_cpu').glob('*/worker.json')}
    progress = {p.name:p.stat().st_size for p in (ROOT / 'gate_cpu').glob('*.out')}
    counter = (sorted(workers), sorted(progress.items()), gate.get('passed'), gate.get('status'))
    stalled = (previous or {}).get('unchanged_checks', 0) + 1 if (previous or {}).get('counter') == list(counter) else 0
    result = dict(utc=datetime.now(timezone.utc).isoformat(), gate_job_id=JOB,
                  **status, gate_passed=gate.get('passed'), gate_status=gate.get('status'),
                  gate_error=gate.get('error'), completed_worker_reports=sorted(workers),
                  log_bytes=progress, counter=list(counter), unchanged_checks=stalled,
                  suspected_stall_confirmed=stalled >= 2, monitor_interval_seconds=1800,
                  jobs_modified=False, new_science_launched=False)
    terminal = status['scheduler_state'] in TERMINAL
    result['monitor_status'] = 'paused' if terminal else 'active'
    result['verified_success'] = (terminal and status['scheduler_state'] == 'COMPLETED' and
                                  status.get('exit_code') == '0:0' and status.get('gpu_count') == 0 and gate.get('passed') is True)
    (ROOT / 'gate_monitor.json').write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'gate_monitor_history.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    previous = None
    for _ in range(5):
        current = check(previous)
        if current['monitor_status'] == 'paused':
            return 0 if current['verified_success'] else 1
        previous = current
        time.sleep(1800)
    raise RuntimeError('Read-only gate monitor cap reached; no jobs changed')


if __name__ == '__main__':
    sys.exit(main())
