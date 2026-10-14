"""Bounded Slurm admissions for the approved DPPO per-step study (Python 3.8).

The plan supplies every scientific command and readiness gate. This module
never alters scientific configuration, retries submissions, or resumes a run.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from worker import atomic_json, append_json, json_check, read_json, utc


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
HARD_CAPS = {'A': 1800, 'C': 5400, 'B': 14400, 'prepare': 900,
             'gate': 3600, 'analysis': 3600}
STATUS_START = '<!-- ppo_steps_campaign_status:start -->'
STATUS_END = '<!-- ppo_steps_campaign_status:end -->'


def validate(plan):
    if not 1 <= int(plan.get('max_parallel', 9)) <= 9:
        raise ValueError('max_parallel must be between 1 and 9')
    if not 0 < float(plan.get('budget_gpu_hours', 80)) <= 80:
        raise ValueError('GPU-hour cap must be positive and at most 80')
    if not Path(plan['run_root']).is_absolute() or not Path(plan['python']).is_absolute():
        raise ValueError('run_root and python must be absolute')
    if plan.get('experiment_path') and not Path(plan['experiment_path']).is_absolute():
        raise ValueError('experiment_path must be absolute')
    ids = [job['id'] for job in plan['jobs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job IDs')
    for job in plan['jobs']:
        if not job['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in job['id']):
            raise ValueError('Unsafe job ID: ' + job['id'])
        kind = job['kind']
        if kind not in HARD_CAPS or not 0 < int(job['cap_seconds']) <= HARD_CAPS[kind]:
            raise ValueError('Invalid time cap for ' + job['id'])
        if int(job.get('gpu_count', 0 if kind == 'analysis' else 1)) not in (0, 1):
            raise ValueError('Only zero or one GPU per job is permitted')
        if int(job.get('gpu_count', 0 if kind == 'analysis' else 1)):
            if int(job.get('cpus', 40)) != 40 or int(job.get('memory_gib', 64)) != 64:
                raise ValueError('GPU jobs require the approved 40 CPUs / 64 GiB allocation')
        if not isinstance(job['command'], list) or not job['command'] or not all(isinstance(v, str) for v in job['command']):
            raise ValueError('A nonempty argv list is required for ' + job['id'])
        if not Path(job['cwd']).is_absolute():
            raise ValueError('cwd must be absolute')
        if not set(job.get('depends_on', [])).issubset(ids):
            raise ValueError('Unknown dependency for ' + job['id'])
        if job['id'] in job.get('depends_on', []):
            raise ValueError('Self dependency')
    # Detect dependency cycles before any scheduling mutation.
    resolved = set()
    while len(resolved) < len(ids):
        ready = {job['id'] for job in plan['jobs']
                 if set(job.get('depends_on', [])).issubset(resolved)}
        if ready.issubset(resolved):
            raise ValueError('Dependency cycle')
        resolved.update(ready)
    return plan


def command(argv, timeout=30, readonly=False):
    attempts = 3 if readonly else 1
    last = None
    for attempt in range(attempts):
        try:
            result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout
            last = RuntimeError('%r exited %d: %s' % (argv, result.returncode, result.stderr[-2000:]))
        except subprocess.TimeoutExpired as error:
            last = error
        if attempt + 1 < attempts:
            time.sleep((2, 5)[attempt])
    raise last


def scheduler_rows(ids):
    if not ids:
        return {}
    output = command(['sacct', '-n', '-P', '-j', ','.join(ids),
                      '--format=JobIDRaw,State,ExitCode,ElapsedRaw'], readonly=True)
    rows = {}
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) >= 4 and fields[0] in ids:
            rows[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'),
                                  exit_code=fields[2], elapsed_seconds=int(fields[3] or 0))
    missing = [job_id for job_id in ids if job_id not in rows]
    if missing:
        output = command(['squeue', '-h', '-j', ','.join(missing), '-o', '%i|%T'], readonly=True)
        for line in output.splitlines():
            fields = line.split('|')
            if len(fields) == 2 and fields[0] in missing:
                rows[fields[0]] = dict(state=fields[1], exit_code=None, elapsed_seconds=0)
    return rows


def ready(job, state):
    for dep in job.get('depends_on', []):
        if state['jobs'].get(dep, {}).get('status') != 'complete':
            return False, 'dependency ' + dep
    for path in job.get('ready_files', []):
        if not Path(path).is_file():
            return False, 'missing ' + path
    for check in job.get('ready_checks', []):
        okay, detail = json_check(check)
        if not okay:
            return False, detail
    return True, None


def costs(plan, state):
    actual = reserved = 0.0
    for job in plan['jobs']:
        saved = state['jobs'].get(job['id'], {})
        count = int(job.get('gpu_count', 0 if job['kind'] == 'analysis' else 1))
        if saved.get('status') in ('complete', 'failed'):
            actual += float(saved.get('elapsed_seconds', job['cap_seconds'])) * count
        elif saved.get('scheduler_job_id') or saved.get('status') == 'submission_unknown':
            reserved += job['cap_seconds'] * count
    return actual, reserved


def update_experiment(plan, state):
    """Replace only the explicitly delimited current-status block."""
    if not plan.get('experiment_path'):
        return
    path = Path(plan['experiment_path'])
    text = path.read_text()
    actual, reserved = costs(plan, state)
    records = state['jobs']
    completed = sum(record.get('status') == 'complete' for record in records.values())
    active = [name for name, record in records.items()
              if record.get('status') not in ('complete', 'failed')]
    status = state['status']
    if status == 'complete':
        next_step = 'All approved jobs and required saved outputs are verified. Review the candidate artifacts; curation needs a separate decision.'
    elif status in ('blocked', 'draining_after_failure'):
        next_step = 'New admissions are stopped. Inspect the recorded failure before deciding on recovery; any admitted workers retain their own limits.'
    else:
        next_step = 'Continue the approved queue as its saved-state and gate prerequisites become available. Part C remains gated by replay and step-1 checks.'
    active_text = ', '.join('`%s`' % name for name in active[:9]) or 'none'
    if len(active) > 9:
        active_text += ' and %d more' % (len(active) - 9)
    detail = state.get('error') or '; '.join(state.get('failures', [])[-1:])
    if not detail and state.get('warnings'):
        detail = state['warnings'][-1]
    relative = os.path.relpath(str(Path(plan['run_root']) / 'campaign.json'), str(path.parent))
    block = [STATUS_START, '**Current execution status (%s).** %s: %d/%d jobs verified complete.' %
             (utc(), status.replace('_', ' '), completed, len(plan['jobs'])), '',
             'GPU use: %.3f hours completed; %.3f hours reserved for admitted jobs; %.1f-hour cap. Active or pending: %s.' %
             (actual / 3600, reserved / 3600, float(plan.get('budget_gpu_hours', 80)), active_text), '',
             '**Next step.** ' + next_step]
    if detail:
        block += ['', '**Attention.** ' + str(detail).replace('\n', ' ')[:600]]
    block += ['', '[Live execution record](%s).' % relative, STATUS_END]
    block = '\n'.join(block)
    if STATUS_START in text or STATUS_END in text:
        if text.count(STATUS_START) != 1 or text.count(STATUS_END) != 1:
            raise ValueError('Experiment status markers are incomplete or duplicated')
        first, last = text.index(STATUS_START), text.index(STATUS_END)
        if last < first:
            raise ValueError('Experiment status markers are reversed')
        updated = text[:first] + block + text[last + len(STATUS_END):]
    else:
        updated = text + ('' if text.endswith('\n\n') else '\n' if text.endswith('\n') else '\n\n') + block + '\n'
    temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
    temporary.write_text(updated)
    os.chmod(str(temporary), path.stat().st_mode & 0o777)
    os.replace(str(temporary), str(path))


def status_record(plan, root, state, update_document=False):
    if update_document:
        try:
            update_experiment(plan, state)
            state.pop('experiment_update_error', None)
        except (OSError, ValueError) as error:
            state['experiment_update_error'] = '%s: %s' % (type(error).__name__, error)
            print('Experiment status update failed: ' + state['experiment_update_error'], file=sys.stderr, flush=True)
    atomic_json(root / 'campaign.json', state)


def submit(plan, job, root, state):
    spec = dict(job, run_root=str(root), monitor_seconds=plan.get('monitor_seconds', 3600))
    spec['startup_checks'] = plan.get('global_checks', []) + job.get('ready_checks', [])
    spec_path = root / 'workers' / job['id'] / 'spec.json'
    if spec_path.exists():
        raise RuntimeError('Worker spec already exists; no automatic retry: ' + job['id'])
    atomic_json(spec_path, spec)
    gpu = int(job.get('gpu_count', 0 if job['kind'] == 'analysis' else 1))
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    argv = ['sbatch', '--parsable', '--account=' + plan.get('account', 'soal'),
            '--partition=' + plan.get('partition', 'soal'), '--no-requeue',
            '--job-name=ppo-' + job['id'], '--cpus-per-task=' + str(job.get('cpus', 40 if gpu else 4)),
            '--mem=%dG' % job.get('memory_gib', 64),
            '--time=%d' % int(math.ceil(job['cap_seconds'] / 60.0)),
            '--output=' + str(logs / (job['id'] + '.scheduler.out')),
            '--error=' + str(logs / (job['id'] + '.scheduler.err'))]
    if gpu:
        argv.append('--gres=gpu:a6000:1')
    argv += [str(Path(__file__).with_name('run_worker.sh')), plan['python'], str(spec_path), str(Path(__file__).with_name('worker.py'))]
    state['jobs'][job['id']] = dict(status='submission_unknown', submission_started_utc=utc(),
                                  command=argv, cap_seconds=job['cap_seconds'], gpu_count=gpu)
    atomic_json(root / 'campaign.json', state)
    append_json(root / 'commands.jsonl', dict(utc=utc(), action='submit', job=job['id'], argv=argv))
    # A timeout leaves submission_unknown. Never repeat sbatch automatically.
    response = command(argv, timeout=30, readonly=False).strip()
    scheduler_id = response.split(';')[0]
    if not scheduler_id.isdigit():
        raise RuntimeError('Unrecognized submission response: ' + response)
    state['jobs'][job['id']].update(status='pending', scheduler_job_id=scheduler_id,
                                  submitted_utc=utc(), spec=str(spec_path))
    atomic_json(root / 'campaign.json', state)


def refresh(plan, root, state):
    ids = [record['scheduler_job_id'] for record in state['jobs'].values()
           if record.get('scheduler_job_id') and record.get('status') not in ('complete', 'failed')]
    rows = scheduler_rows(ids)
    failures = []
    for job in plan['jobs']:
        record = state['jobs'].get(job['id'])
        if not record or record.get('status') in ('complete', 'failed'):
            continue
        scheduler_id = record.get('scheduler_job_id')
        if scheduler_id not in rows:
            record['missing_scheduler_checks'] = record.get('missing_scheduler_checks', 0) + 1
            if record['missing_scheduler_checks'] >= 3:
                failures.append('Scheduler cannot locate admitted job ' + job['id'])
            continue
        row = rows[scheduler_id]
        record.update(scheduler=row, observed_utc=utc(), missing_scheduler_checks=0)
        manifest = read_json(root / 'workers' / job['id'] / 'manifest.json', {})
        record['worker_status'] = manifest.get('status')
        progress = read_json(job['progress_file'], {}) if job.get('progress_file') else manifest.get('progress', {})
        record['progress'] = progress
        if manifest.get('status') == 'failed':
            failures.append(job['id'] + ': ' + manifest.get('error', 'worker failed'))
        if row['state'] in TERMINAL:
            record['elapsed_seconds'] = row['elapsed_seconds']
            if row['state'] == 'COMPLETED' and row['exit_code'] == '0:0' and manifest.get('status') == 'complete':
                record.update(status='complete', ended_utc=utc())
            else:
                record.update(status='failed', ended_utc=utc())
                failures.append('%s terminal %s/%s, worker %s' %
                                (job['id'], row['state'], row['exit_code'], manifest.get('status')))
        else:
            record['status'] = 'running' if row['state'] == 'RUNNING' else 'pending'
    return failures


def run(plan_path):
    plan = validate(read_json(plan_path))
    root = Path(plan['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'campaign.json').exists():
        raise RuntimeError('Existing campaign requires an explicit reviewed recovery; refusing resume')
    for check in plan.get('global_checks', []):
        okay, detail = json_check(check)
        if not okay:
            raise RuntimeError('Global gate not passed: ' + detail)
    state = dict(campaign_id=plan['campaign_id'], status='running', started_utc=utc(),
                 plan=plan, plan_path=str(Path(plan_path).resolve()), jobs={},
                 controller_job_id=os.environ.get('SLURM_JOB_ID'), monitor_status='active')
    atomic_json(root / 'plan_snapshot.json', plan)
    status_record(plan, root, state, update_document=True)
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    start = time.monotonic()
    next_monitor = 0.0
    try:
        while True:
            if interrupted:
                raise RuntimeError('Controller interrupted; worker allocations retain independent caps')
            if time.monotonic() - start >= float(plan.get('controller_cap_seconds', 43200)) - 60:
                raise RuntimeError('Controller wall cap reached; workers retain independent caps')
            failures = refresh(plan, root, state)
            if failures:
                state.update(status='draining_after_failure', failures=state.get('failures', []) + failures)
            active = [r for r in state['jobs'].values() if r.get('status') not in ('complete', 'failed')]
            if state['status'] == 'running':
                for job in plan['jobs']:
                    if len(active) >= int(plan.get('max_parallel', 9)):
                        break
                    if job['id'] in state['jobs']:
                        continue
                    okay, why = ready(job, state)
                    if not okay:
                        continue
                    actual, reserved = costs(plan, state)
                    reserve = job['cap_seconds'] * int(job.get('gpu_count', 0 if job['kind'] == 'analysis' else 1))
                    if actual + reserved + reserve > float(plan.get('budget_gpu_hours', 80)) * 3600:
                        if not active:
                            raise RuntimeError('Remaining GPU budget cannot cover next complete job reservation')
                        continue
                    submit(plan, job, root, state)
                    active.append(state['jobs'][job['id']])
            actual, reserved = costs(plan, state)
            state.update(updated_utc=utc(), actual_gpu_hours=actual / 3600,
                         reserved_gpu_hours=reserved / 3600,
                         completed=sum(r.get('status') == 'complete' for r in state['jobs'].values()),
                         total=len(plan['jobs']))
            monitor_due = time.monotonic() >= next_monitor
            if monitor_due:
                for job in plan['jobs']:
                    record = state['jobs'].get(job['id'], {})
                    if record.get('status') != 'running' or not job.get('progress_file'):
                        continue
                    progress = record.get('progress', {})
                    # Explicit counter keys exclude constantly changing timestamps.
                    keys = job.get('progress_keys', ['completed', 'done', 'itr', 'iteration',
                                                     'batch', 'batches', 'phase', 'substage'])
                    counters = {key: progress[key] for key in keys if key in progress}
                    if not counters:
                        continue
                    if counters == record.get('last_monitor_counters'):
                        record['unchanged_monitor_checks'] = record.get('unchanged_monitor_checks', 0) + 1
                    else:
                        record['unchanged_monitor_checks'] = 0
                    record['last_monitor_counters'] = counters
                    if record['unchanged_monitor_checks'] >= 2:
                        warning = 'Possible stall: %s counters unchanged at two monitor checks' % job['id']
                        if warning not in state.get('warnings', []):
                            state.setdefault('warnings', []).append(warning)
                            print(warning, file=sys.stderr, flush=True)
                append_json(root / 'monitor_history.jsonl', dict(utc=utc(), status=state['status'],
                            completed=state['completed'], total=state['total'],
                            actual_gpu_hours=state['actual_gpu_hours'], reserved_gpu_hours=state['reserved_gpu_hours'],
                            jobs={key: dict(status=value['status'], scheduler_job_id=value.get('scheduler_job_id'),
                                            progress=value.get('progress')) for key, value in state['jobs'].items()}))
                next_monitor = time.monotonic() + float(plan.get('monitor_seconds', 3600))
            status_record(plan, root, state, update_document=monitor_due or bool(failures))
            if state['completed'] == state['total']:
                state.update(status='complete', ended_utc=utc(), monitor_status='paused')
                status_record(plan, root, state, update_document=True)
                return 0
            if state['status'] == 'draining_after_failure' and not active:
                raise RuntimeError('Admissions stopped after failure; all other admitted workers have finished')
            if not active and state['completed'] < state['total']:
                waiting = {job['id']: ready(job, state)[1] for job in plan['jobs']
                           if job['id'] not in state['jobs']}
                raise RuntimeError('No active producer can satisfy remaining readiness gates: %r' % waiting)
            time.sleep(float(plan.get('poll_seconds', 30)))
    except BaseException as error:
        state.update(status='blocked', error='%s: %s' % (type(error).__name__, error),
                     ended_utc=utc(), monitor_status='paused')
        status_record(plan, root, state, update_document=True)
        print(state['error'], file=sys.stderr, flush=True)
        return 1
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'run'))
    parser.add_argument('plan')
    args = parser.parse_args()
    if args.action == 'validate':
        plan = validate(read_json(args.plan))
        print(json.dumps(dict(valid=True, jobs=len(plan['jobs']), max_parallel=plan.get('max_parallel', 9),
                              budget_gpu_hours=plan.get('budget_gpu_hours', 80))))
    else:
        sys.exit(run(args.plan))
