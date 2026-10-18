"""Approved one-shot calibration -> CPU gates -> training handoff.

No scientific retries, GPU work or changes to frozen source are performed by
this CPU supervisor. A failed/incomplete gate cannot reach training submission.
"""
import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import campaign
from worker import atomic_json, append_json, descendants, read_json, reap, stop_owned, utc, verify_source

ROOT = campaign.ROOT
REVISION = '2acb18e9629c577474ee7f1652c7cc6f5a3829ac'
PRELIMINARY = ROOT / 'runs/gates/preliminary_retry'
CALIBRATED = ROOT / 'runs/gates/calibrated'
CALIBRATION = ROOT / 'cal_seed3000/calibration.json'
PRIOR_GATE_SECONDS = 731.9507804671302
TOTAL_GATE_SECONDS = 7200.0
SUPERVISOR_SECONDS = 7200.0


def require(value, message):
    if not value:
        raise RuntimeError(message)


def calibration_ready(record, completion):
    if record.get('status') in ('failed', 'blocked', 'draining_after_failure'):
        raise RuntimeError('Calibration controller stopped: ' + str(record.get('error', record.get('failures'))))
    if record.get('status') != 'complete':
        return False
    require(completion.get('verified') is True, 'Calibration completion evidence missing')
    require(record.get('reserved_gpu_hours') == 0, 'Calibration still has outstanding allocations')
    require(record.get('source_provenance', {}).get('revision') == REVISION,
            'Calibration source revision differs from the approved frozen source')
    jobs = record.get('jobs', {})
    require(set(jobs) == {'cal_seed3000'} and jobs['cal_seed3000'].get('status') == 'complete',
            'The approved five-batch calibration did not complete')
    return True


def guarded_handoff(gate, build, submit):
    """Keep the gate adjacent to both side effects; used by the mocked check."""
    require(gate.get('passed') is True and gate.get('status') == 'passed',
            'Full CPU gate did not pass; training is forbidden')
    require(all(gate.get('checks', {}).get(key, {}).get('passed') is True
                for key in ('G1', 'G2', 'G3', 'G4', 'G5')),
            'A required CPU check is absent or failed; training is forbidden')
    return submit(build())


def self_check():
    called = []
    build = lambda: called.append('build') or 'plan'
    submit = lambda plan: called.append('submit') or plan
    cases = [{'status': 'failed', 'passed': False},
             {'status': 'running', 'passed': False},
             {'status': 'passed', 'passed': True, 'checks': {'G3': {'passed': True}}}]
    for gate in cases:
        try:
            guarded_handoff(gate, build, submit)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Failed/incomplete gate reached the handoff')
        assert called == []
    good = dict(status='passed', passed=True,
                checks={key: {'passed': True} for key in ('G1', 'G2', 'G3', 'G4', 'G5')})
    assert guarded_handoff(good, build, submit) == 'plan'
    assert called == ['build', 'submit']
    return dict(passed=True, failed_or_incomplete_gate_blocks_both_side_effects=True,
                passed_gate_order=['build', 'submit'], subprocesses_started=0, scientific_tests_run=0)


def document(state, terminal=False):
    path = ROOT / 'experiment.md'
    if not path.exists():
        return
    start, end = '<!-- AN continuation -->', '<!-- end AN continuation -->'
    details = state.get('error') or state.get('detail', '')
    block = '\n'.join([start, '**Calibrated gate / training handoff (%s):** %s. %s' %
                       (state['updated_utc'], state['status'], details), '',
                       '[Continuation record](runs/continuation/state.json).', end])
    text = path.read_text()
    if start not in text and end not in text:
        text = text.rstrip() + '\n\n' + block + '\n'
    else:
        require(text.count(start) == text.count(end) == 1 and text.index(start) < text.index(end),
                'Malformed continuation document markers')
        text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    temporary = path.with_name(path.name + '.continuation.tmp.%d' % os.getpid())
    temporary.write_text(text)
    os.replace(str(temporary), str(path))
    if terminal and state.get('status') == 'blocked':
        report = ROOT / 'REPORT.md'
        submission = ('Training submission was attempted; its scheduler state requires review before any retry. '
                      if state.get('training_submission_attempted') else
                      'No training allocation was submitted by this continuation. ')
        note = ('# AdaGrad-Norm on the discounted Monte Carlo critic estimator\n\n'
                '**Status: blocked before training.** %s\n\n'
                'The required gate was not bypassed. %s'
                'Existing calibration evidence and all gate diagnostics are retained. The estimator and tolerances '
                'were not changed.\n\n'
                '[Gate details](runs/gates/calibrated/gate_summary.json) · '
                '[Continuation record](runs/continuation/state.json) · [Methods](methods.md).\n') % (state['error'], submission)
        if report.exists():
            (ROOT / 'runs/continuation/REPORT_before_block.md').write_bytes(report.read_bytes())
        report.write_text(note)


def main():
    directory = ROOT / 'runs/continuation'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / 'state.json'
    require(not state_path.exists(), 'Continuation already has a record; no automatic restart')
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'Cannot enable owned-child cleanup supervision')
    state = dict(status='waiting_calibration', started_utc=utc(), updated_utc=utc(),
                 scheduler_job_id=os.environ.get('SLURM_JOB_ID'), pid=os.getpid(),
                 revision=REVISION, cpu_gate_cap_seconds=TOTAL_GATE_SECONDS,
                 prior_gate_seconds=PRIOR_GATE_SECONDS, supervisor_cap_seconds=SUPERVISOR_SECONDS,
                 monitor_seconds=1800, monitor_status='active', commands=[], detail='No training launched.')
    started = time.monotonic()
    child = None
    def save(update_document=False):
        state.update(updated_utc=utc(), supervisor_elapsed_seconds=time.monotonic() - started)
        atomic_json(state_path, state)
        if update_document:
            document(state)
    def interrupted(signum, frame):
        raise RuntimeError('CPU continuation interrupted by signal %d' % signum)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    for name in ('continue_after_calibration.py', 'run_continuation.sh', 'gate_complete.py', 'gate_adagrad_norm.py'):
        source = ROOT / 'code' / name
        target = directory / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    save(True)
    try:
        plan = campaign.validate(read_json(ROOT / 'runs/plan_calibration.json'))
        verify_source(plan['source_provenance'])
        next_monitor = 0.0
        while True:
            require(time.monotonic() - started < SUPERVISOR_SECONDS - 90,
                    'Bounded continuation waiting deadline reached; no training launched')
            record = read_json(ROOT / 'runs/campaign_calibration/campaign.json', {})
            completion = read_json(ROOT / 'runs/completion_calibration.json', {})
            if calibration_ready(record, completion):
                break
            if time.monotonic() >= next_monitor:
                append_json(directory / 'monitor_history.jsonl',
                            dict(utc=utc(), phase='waiting_calibration', calibration_status=record.get('status')))
                next_monitor = time.monotonic() + 1800
                save(True)
            time.sleep(15)
        derived = campaign.calibration_summary(CALIBRATION)
        atomic_json(ROOT / 'runs/calibration_derived.json', derived)
        state.update(calibration=derived, status='calibrated_gate',
                     detail='Calibration verified; running strict G3 and the complete 12-iteration CPU G5.')
        prior = read_json(PRELIMINARY / 'gate_summary.json')
        require(prior.get('preliminary_passed') is True, 'Preliminary CPU checks did not pass')
        require(math.isclose(prior['aggregate_gate_seconds'], PRIOR_GATE_SECONDS, rel_tol=0, abs_tol=1e-6),
                'Prior CPU gate accounting changed')
        require(not CALIBRATED.exists(), 'Calibrated gate directory already exists; do not rerun checks')
        # gate_complete subtracts the prior 731.95 seconds from the approved cap.
        argv = [str(campaign.PYTHON), '-B', str(ROOT / 'code/gate_complete.py'),
                '--source', str(campaign.SOURCE), '--preliminary', str(PRELIMINARY),
                '--output', str(CALIBRATED), '--eta', repr(derived['d_star']), '--b0', repr(derived['b0']),
                '--cap-seconds', '7200', '--threads', '2']
        state['commands'].append(argv)
        append_json(directory / 'commands.jsonl', dict(utc=utc(), action='calibrated_gate', argv=argv))
        save(True)
        deadline = min(started + SUPERVISOR_SECONDS - 60,
                       time.monotonic() + TOTAL_GATE_SECONDS - PRIOR_GATE_SECONDS + 45)
        previous_counter, unchanged, next_monitor = None, 0, 0.0
        with (directory / 'calibrated_gate.out').open('x') as output:
            gate_env = os.environ.copy()
            gate_env['CUDA_VISIBLE_DEVICES'] = ''
            child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                     env=gate_env, start_new_session=True)
            state['gate_pid'] = child.pid
            save()
            while child.poll() is None:
                require(time.monotonic() < deadline, 'CPU gate or continuation wall-clock cap exhausted')
                if time.monotonic() >= next_monitor:
                    progress = read_json(CALIBRATED / 'live_smoke_12itr/smoke_progress.json', {})
                    counter = len(progress.get('iterations', []))
                    unchanged = unchanged + 1 if counter == previous_counter else 0
                    previous_counter = counter
                    append_json(directory / 'monitor_history.jsonl',
                                dict(utc=utc(), phase='calibrated_gate', completed_iterations=counter,
                                     suspected_stall=unchanged >= 2))
                    state['completed_cpu_smoke_iterations'] = counter
                    next_monitor = time.monotonic() + 1800
                    save(True)
                time.sleep(2)
            state['gate_returncode'] = child.returncode
            if child.returncode:
                failed = read_json(CALIBRATED / 'gate_summary.json', {})
                detail = failed.get('error', 'inspect gate_summary.json and calibrated_gate.out')
                g3 = failed.get('checks', {}).get('G3', {})
                if 'relative_error' in g3:
                    detail += ' (G3 relative error %.9g; required <= 1e-6).' % g3['relative_error']
                raise RuntimeError('Calibrated CPU gate failed: ' + detail)
        gate = read_json(CALIBRATED / 'gate_summary.json', {})
        require(gate.get('source') == str(campaign.SOURCE) and gate.get('eta') == derived['d_star']
                and gate.get('b0') == derived['b0'], 'Calibrated gate used different source or calibrated values')
        require(gate.get('aggregate_gate_seconds', float('inf')) <= TOTAL_GATE_SECONDS,
                'CPU gate exceeded its aggregate two-hour allowance')
        verify_source(plan['source_provenance'])
        reap()
        require(not descendants(os.getpid()), 'Calibrated gate left child processes behind')

        def build():
            state.update(status='building_training_plan', detail='All five CPU gates passed; preparing bounded training.')
            save(True)
            argv = [str(campaign.PYTHON), '-B', str(ROOT / 'code/campaign.py'), 'build-training',
                    '--revision', REVISION, '--gate', str(CALIBRATED / 'gate_summary.json'),
                    '--calibration', str(CALIBRATION)]
            state['commands'].append(argv)
            append_json(directory / 'commands.jsonl', dict(utc=utc(), action='build_training_plan', argv=argv))
            with (directory / 'build_training.out').open('x') as output:
                result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=120)
            require(result.returncode == 0, 'Training plan build failed; no controller submitted')
            target = ROOT / 'runs/plan_training.json'
            campaign.validate(read_json(target))
            return target

        def submit(target):
            state.update(status='training_submission_pending', detail='Full gate passed; submitting the approved training controller once.')
            save(True)
            argv = ['sbatch', '--parsable',
                    '--output=' + str(ROOT / 'runs/logs/controller_training.out'),
                    '--error=' + str(ROOT / 'runs/logs/controller_training.err'),
                    str(ROOT / 'code/run_controller.sh'), str(campaign.PYTHON),
                    str(ROOT / 'code/campaign.py'), str(target)]
            state['commands'].append(argv)
            append_json(directory / 'commands.jsonl', dict(utc=utc(), action='submit_training_controller', argv=argv))
            # Persist intent before the scheduler call; never retry ambiguous submission.
            state['training_submission_attempted'] = True
            save()
            submit_env = os.environ.copy()
            submit_env.pop('CUDA_VISIBLE_DEVICES', None)
            response = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      universal_newlines=True, timeout=30, env=submit_env)
            state['training_submission_response'] = dict(returncode=response.returncode,
                                                         stdout=response.stdout, stderr=response.stderr)
            save()
            require(response.returncode == 0, 'Training-controller scheduler call failed: ' + response.stderr[-1000:])
            answer = response.stdout.strip().split(';')[0]
            require(answer.isdigit(), 'Ambiguous training-controller submission; manual review required: ' + answer)
            return answer

        state['training_controller_job_id'] = guarded_handoff(gate, build, submit)
        state.update(status='handed_off', monitor_status='paused', ended_utc=utc(),
                     detail='Training controller %s owns the nine runs, final evaluations and 30-minute monitors.' % state['training_controller_job_id'])
        save(True)
        return 0
    except BaseException as error:
        remaining = stop_owned(child)
        state.update(status='blocked', error=str(error), traceback=traceback.format_exc(),
                     child_processes_after=remaining, ended_utc=utc(), monitor_status='paused')
        if state.get('training_submission_attempted'):
            state['error'] += ' Training submission was attempted; inspect the recorded scheduler response before any retry.'
        save()
        document(state, terminal=True)
        print(state['error'], file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
    else:
        sys.exit(main())
