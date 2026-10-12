"""One explicit administrative recovery; never restart a scientific job.

The controller failed during a read-only Slurm query. Retry only transient
read failures, never submissions. Preserve the earlier bounded ESTALE retry.
Original workers, scientific source, configurations and results are untouched.
"""
import argparse
import copy
from datetime import datetime, timezone
import errno
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
RUN = HERE.parent
sys.path.insert(0, str(RUN / 'code'))
import queue_v5 as q

FAILED_CONTROLLER = '17826164'
COMMIT = '30b5956e113ae0b392daf6056f03d8a894a99f73'
ADMITTED = q.PREPARES + [q.NORM] + q.COLLECTS[:-2]
REMAINING = q.COLLECTS[-2:]
BASE_COMMAND = q.old.command



def query_command(argv, check=True, runner=None, sleeper=time.sleep):
    """Bounded transient-read retries; sbatch and all mutations use original code."""
    readonly = argv[0] in ('squeue', 'sacct') or argv[:2] == ['scontrol', 'show']
    if not readonly:
        return BASE_COMMAND(argv, check=check)
    runner = runner or subprocess.run
    record = None
    for attempt in range(3):
        try:
            result = runner(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=30)
            record = dict(utc=q.old.utc(), command=argv, returncode=result.returncode,
                          stdout=result.stdout, stderr=result.stderr, attempt=attempt + 1)
            transient = any(token in result.stderr.lower() for token in
                            ('unable to contact', 'connect failure', 'socket timed out', 'communication connection failure'))
        except subprocess.TimeoutExpired:
            record = dict(utc=q.old.utc(), command=argv, returncode=124,
                          stdout='', stderr='Read-only Slurm query timed out after 30 seconds', attempt=attempt + 1)
            transient = True
        q.old.append(HERE / 'commands.jsonl', record)
        if not record['returncode'] or not transient or attempt == 2:
            break
        sleeper((2, 5)[attempt])
    if check and record['returncode']:
        raise RuntimeError('Read-only scheduler query failed after bounded attempts: ' + repr(record))
    return record


def read_progress(path, reader=None, sleeper=time.sleep):
    reader = reader or q.old.read
    for attempt in range(3):
        try:
            return reader(path, {})
        except OSError as error:
            if error.errno != errno.ESTALE or attempt == 2:
                raise
            sleeper((0.1, 0.5)[attempt])


def observe(spec):
    progress = read_progress(Path(spec['logdir']) / 'stage3_progress.json')
    meta = q.old.read(q.meta_path(spec), {})
    return dict(utc=q.old.utc(), progress=q.old.safe(progress), status=meta.get('status'), error=meta.get('error'),
                completed_batches=meta.get('completed_batches', meta.get('batch_count', meta.get('completed_warmup_batches', 0))),
                meta_nonfinite=not q.old.finite_json(meta))


def validate_original(state):
    if (state.get('status') != 'blocked' or state.get('controller_job_id') != FAILED_CONTROLLER or
            'Unable to contact slurm controller (connect failure)' not in state.get('failure_reason', '')):
        raise ValueError('Only the identified administrative Slurm-query failure can be recovered')
    if state.get('pending_submission_intent') is not None:
        raise ValueError('Unresolved submission intent forbids recovery')
    recoveries = state.get('controller_recoveries', [])
    if (len(recoveries) != 1 or recoveries[0].get('failed_controller') != '17825855' or
            recoveries[0].get('replacement_controller') != FAILED_CONTROLLER):
        raise ValueError('Only the known prior administrative recovery is permitted')
    if state.get('source_provenance', {}).get('source_commit') != COMMIT:
        raise ValueError('The approved scientific source revision changed')
    entries = state.get('entries', [])
    if len(entries) != len(ADMITTED) or [e.get('run_id') for e in entries] != ADMITTED:
        raise ValueError('All fifty-four admitted jobs must be preserved exactly')
    for entry in entries:
        if entry.get('status') == 'failed' or (entry.get('scheduler_state') in q.old.TERMINAL and
                                             entry.get('scheduler_state') != 'COMPLETED'):
            raise ValueError('A scientific failure requires a decision, never a retry')
    if state.get('plan') != q.plan():
        raise ValueError('The fixed plan or budget changed')
    return copy.deepcopy(state)


def reconcile():
    original = q.old.read(q.STATE, {})
    state = validate_original(original)
    scheduler = q.old.accounting(FAILED_CONTROLLER)
    if scheduler.get('scheduler_state') != 'FAILED' or scheduler.get('exit_code') != '1:0':
        raise ValueError('The original controller is not terminal in the expected state')
    trace = (RUN / 'logs/recovery-controller-17826164.out').read_text()
    if 'Unable to contact slurm controller (connect failure)' not in trace:
        raise ValueError('The preserved traceback does not identify the authorized polling failure')
    source = q.validate_source()
    if q.normalization() != 'full':
        raise ValueError('The already-frozen normalization choice changed')
    for entry in state['entries']:
        spec = q.old.read(RUN / 'specs' / (entry['run_id'] + '.json'))
        if spec != q.make_spec(entry['run_id']):
            raise ValueError('An admitted configuration changed: ' + entry['run_id'])
        entry.update(q.old.accounting(entry['job_id']))
        worker = q.old.read(RUN / 'workers' / entry['run_id'] / 'manifest.json', {})
        if entry['scheduler_state'] == 'COMPLETED':
            if (entry.get('exit_code') != '0:0' or entry.get('gpu_count') != 1 or worker.get('status') != 'complete' or
                    worker.get('subprocess_exitcode') != 0 or worker.get('remaining_process_group_members') != []):
                raise ValueError('Completed worker has not passed cleanup: ' + entry['run_id'])
            entry.update(status='complete', verification=q.verify(spec))
        else:
            raise ValueError('All fifty-four original jobs must finish successfully before this recovery: ' + entry['run_id'])
    remaining = [name for name in q.RUN_IDS if name not in {e['run_id'] for e in state['entries']}]
    if remaining != REMAINING or len(remaining) != 2:
        raise ValueError('Recovery must admit exactly the remaining two jobs')
    if any((RUN / 'specs' / (name + '.json')).exists() or
           (RUN / 'workers' / name).exists() or (RUN / 'collect' / name).exists() for name in remaining):
        raise ValueError('An unrecorded admission would make resubmission ambiguous')
    ledger = q.ledger(state)
    if ledger['aggregate_actual_plus_reserved'] + len(remaining) * q.CAP > q.TOTAL_CAP:
        raise ValueError('All remaining reservations must fit the original aggregate budget')
    deadline = datetime.fromisoformat(original['started_utc']).timestamp() + q.plan()['controller_cap_seconds']
    if time.time() >= deadline - 30:
        raise ValueError('The original controller wall-clock budget has expired')
    audit = dict(utc=q.old.utc(), failed_controller=FAILED_CONTROLLER, source_provenance=source,
                 preserved_run_ids=ADMITTED, remaining_run_ids=remaining, reconciled_entries=state['entries'],
                 ledger=ledger, original_wall_clock_deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                 scientific_jobs_restarted=False, scientific_source_changed=False,
                 progress_read_retry=dict(errno='ESTALE', attempts=3, delays_seconds=[0.1, 0.5]),
                 scheduler_query_retry=dict(commands=['squeue', 'sacct', 'scontrol show'], attempts=3,
                                            timeout_seconds=30, delays_seconds=[2, 5], submissions_retried=False),
                 remaining_specs=[q.make_spec(name) for name in remaining],
                 approval='User approved only the remaining two jobs and final analysis within the existing cap')
    return original, state, audit, deadline


def run():
    lock = (RUN / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    original, state, audit, deadline = reconcile()
    q.old.once(HERE / 'failed_campaign.json', original)
    q.old.once(HERE / 'recovery_admission_audit.json', audit)
    snapshot = HERE / 'runtime_snapshot'
    snapshot.mkdir(exist_ok=False)
    for path in (HERE / 'code').glob('*'):
        if path.is_file():
            shutil.copy2(str(path), str(snapshot / path.name))
    state.update(status='running', monitor_status='active', controller_job_id=os.environ['SLURM_JOB_ID'],
                 controller_recoveries=state.get('controller_recoveries', []) + [dict(failed_controller=FAILED_CONTROLLER, replacement_controller=os.environ['SLURM_JOB_ID'],
                     utc=q.old.utc(), audit=str(HERE / 'recovery_admission_audit.json'), source_snapshot=str(snapshot))])
    state.pop('ended_utc', None)
    state.pop('failure_reason', None)
    stop, next_monitor = [], 0
    signal.signal(signal.SIGTERM, lambda sig, frame: stop.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stop.append(sig))
    try:
        while True:
            q.update(state)
            due = time.monotonic() >= next_monitor
            q.record(state, monitor=due)
            if due:
                next_monitor = time.monotonic() + 1800
            failures = [e['run_id'] for e in state['entries'] if e.get('status') == 'failed']
            if failures:
                raise RuntimeError('Scientific failure; no retries or further admissions: ' + ', '.join(failures))
            if stop or time.time() >= deadline - 30:
                raise RuntimeError('Recovery controller stopped or reached the original deadline; workers untouched')
            if len(state['entries']) == len(q.RUN_IDS) and all(e.get('status') == 'complete' for e in state['entries']):
                state.update(status='complete', ended_utc=q.old.utc(), monitor_status='paused')
                q.record(state, monitor=True)
                return 0
            jobs = q.old.scheduler_jobs()
            nodes = q.old.command(['scontrol', 'show', 'nodes', '-o'])['stdout']
            slots = q.old.queue_slots(state['entries'], jobs, max_parallel=2, free_slots=q.old.free_a6000_slots(nodes))
            for name in list(q.ready(state))[:slots]:
                if name not in REMAINING or q.ledger(state)['aggregate_actual_plus_reserved'] + q.CAP > q.TOTAL_CAP:
                    raise ValueError('Admission would exceed the remaining fixed plan or budget')
                q.submit(state, name)
            time.sleep(15)
    except BaseException as error:
        state.update(status='blocked', ended_utc=q.old.utc(), monitor_status='paused',
                     failure_reason=type(error).__name__ + ': ' + str(error))
        q.record(state, monitor=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('validate', 'run'))
    args = parser.parse_args()
    q.COMMIT = COMMIT
    q.old.JOURNAL = HERE / 'commands.jsonl'
    q.observe = observe
    q.old.command = query_command
    q.bind_supervision()
    if args.mode == 'validate':
        original, unused_state, audit, unused_deadline = reconcile()
        q.old.once(HERE / 'failed_campaign.json', original)
        q.old.atomic(HERE / 'preflight.json', audit)
        print('Recovery preflight passed; all fifty-four admissions preserved, exactly two remain.', flush=True)
        return 0
    return run()


if __name__ == '__main__':
    sys.exit(main())
