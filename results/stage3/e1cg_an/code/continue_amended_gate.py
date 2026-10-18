"""One-shot approved amended-G3 -> G5 -> training handoff; Python 3.8."""
import time
STARTED = time.monotonic()

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import campaign
from worker import atomic_json, append_json, descendants, read_json, reap, stop_owned, utc, verify_source

ROOT = campaign.ROOT
REVISION = '2acb18e9629c577474ee7f1652c7cc6f5a3829ac'
PRIOR = ROOT / 'runs/gates/preliminary_retry/gate_summary.json'
G3 = ROOT / 'runs/gates/amended_G3/gate_summary.json'
G3_TIMING = ROOT / 'runs/gates/amended_G3/process_timing.json'
OUTPUT = ROOT / 'runs/gates/amended_complete'
DIRECTORY = ROOT / 'runs/continuation_amended'
CALIBRATION = ROOT / 'cal_seed3000/calibration.json'
CPU_CAP = 7200.0


def require(value, message):
    if not value:
        raise RuntimeError(message)


def gate_allows_training(gate):
    return (gate.get('passed') is True and gate.get('status') == 'passed'
            and all(gate.get('checks', {}).get(key, {}).get('passed') is True
                    for key in ('G1', 'G2', 'G3a', 'G3b', 'G4', 'G5')))


def self_check():
    good = dict(passed=True, status='passed', checks={key: {'passed': True}
                for key in ('G1', 'G2', 'G3a', 'G3b', 'G4', 'G5')})
    assert gate_allows_training(good)
    for key in good['checks']:
        changed = dict(good, checks=dict(good['checks']))
        changed['checks'][key] = {'passed': False}
        assert not gate_allows_training(changed)
    assert not gate_allows_training({'passed': True, 'status': 'passed'})
    assert list(campaign.AMENDED_ETAS.values()) == [0.00263955, 0.00791865, 0.0263955]
    assert campaign.AMENDED_B0 == 0.00000322129
    return dict(passed=True, every_required_check_blocks_training_when_false=True,
                exact_values=True, scientific_tests_run=0, subprocesses_started=0)


def update_document(state, terminal=False):
    path = ROOT / 'experiment.md'
    if path.exists():
        start, end = '<!-- AN amended continuation -->', '<!-- end AN amended continuation -->'
        block = '\n'.join([start, '**Amended gate / training handoff (%s):** %s. %s' %
                           (state['updated_utc'], state['status'], state.get('error', state.get('detail', ''))), '',
                           '[Continuation record](runs/continuation_amended/state.json).', end])
        text = path.read_text()
        if start not in text and end not in text:
            text = text.rstrip() + '\n\n' + block + '\n'
        else:
            require(text.count(start) == text.count(end) == 1, 'Malformed amended-continuation markers')
            text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
        temp = path.with_name(path.name + '.amended.tmp.%d' % os.getpid())
        temp.write_text(text)
        os.replace(str(temp), str(path))
    if terminal:
        report = ROOT / 'REPORT.md'
        if report.exists():
            (DIRECTORY / 'REPORT_before_block.md').write_bytes(report.read_bytes())
        submission = ('A controller submission was attempted; verify its scheduler record before retrying.'
                      if state.get('training_submission_attempted') else 'No training was submitted by this continuation.')
        report.write_text('# AdaGrad-Norm on the discounted Monte Carlo critic estimator\n\n'
                          '**Status: blocked.** %s %s\n\n'
                          'No gate was bypassed. The original failure, measured calibration and amended checks are retained. '
                          'No scientific run was retried automatically.\n\n'
                          '[Amended gate](runs/gates/amended_complete/gate_summary.json) · '
                          '[Continuation](runs/continuation_amended/state.json) · [Methods](methods.md).\n'
                          % (state['error'], submission))


def execute(amendment_path):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock = (DIRECTORY / 'continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = DIRECTORY / 'state.json'
    require(not state_path.exists(), 'Amended continuation already recorded; no automatic restart')
    require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'Cannot supervise owned-child cleanup')
    state = dict(status='checking_amended_gate', started_utc=utc(), updated_utc=utc(),
                 scheduler_job_id=os.environ.get('SLURM_JOB_ID'), pid=os.getpid(),
                 revision=REVISION, amendment_path=str(amendment_path),
                 commands=[], monitor_seconds=1800, monitor_status='active')
    child = None
    def save(document=False):
        state.update(updated_utc=utc(), supervisor_elapsed_seconds=time.monotonic() - STARTED)
        atomic_json(state_path, state)
        if document:
            update_document(state)
    def interrupted(signum, frame):
        raise RuntimeError('Amended continuation received signal %d' % signum)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        for name in ('continue_amended_gate.py', 'run_amended_continuation.sh', 'campaign.py', 'gate_adagrad_norm.py'):
            source = ROOT / 'code' / name
            destination = DIRECTORY / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        plan = campaign.validate(read_json(ROOT / 'runs/plan_calibration.json'))
        verify_source(plan['source_provenance'])
        require(plan['source_provenance']['revision'] == REVISION, 'Frozen source revision differs')
        calibration_campaign = read_json(ROOT / 'runs/campaign_calibration/campaign.json', {})
        require(calibration_campaign.get('status') == 'complete'
                and calibration_campaign.get('reserved_gpu_hours') == 0
                and read_json(ROOT / 'runs/completion_calibration.json', {}).get('verified'),
                'Calibration completion was not verified; it will not be rerun')
        derived = campaign.amended_calibration_summary(CALIBRATION, amendment_path)
        # Save separately: neither measured calibration nor the prior derived file is overwritten.
        atomic_json(ROOT / 'runs/calibration_derived_amended.json', derived)
        old = read_json(PRIOR)
        revised = read_json(G3)
        require(old.get('preliminary_passed') is True, 'Reused G1/G2/G4 did not pass')
        require(revised.get('status') == 'passed' and revised.get('passed') is True
                and all(revised.get('checks', {}).get(k, {}).get('passed') is True for k in ('G3a', 'G3b')),
                'Amended G3a/G3b have not both passed')
        require(revised.get('eta') == campaign.AMENDED_ETAS['AN1']
                and revised.get('b0') == campaign.AMENDED_B0, 'Amended G3 used different exact values')
        internal_prior = float(revised['aggregate_gate_seconds'])
        require(internal_prior >= 740.0639064861462, 'Amended gate prior-time accounting is incomplete')
        timing = read_json(G3_TIMING)
        require(isinstance(timing, dict) and timing.get('returncode') == 0,
                'Amended G3 full-process timing/success record is missing')
        process_seconds = float(timing['elapsed_seconds'])
        require(process_seconds >= 0, 'Invalid amended G3 process timing')
        prior_seconds = 759.9507804671302 + max(process_seconds, float(revised['elapsed_seconds']))
        state.update(calibration=derived, prior_gate_seconds=prior_seconds,
                     internal_amended_gate_aggregate_seconds=internal_prior,
                     conservative_previous_cpu_wall_seconds=759.9507804671302,
                     amended_G3_full_process_seconds=process_seconds,
                     conservative_overhead_charge_seconds=prior_seconds - internal_prior,
                     cpu_accounting_basis='Previous 731.9507804671302s gates + 28s failed-gate CPU allocation; amended G3 full process; this supervisor from module entry.',
                     cpu_gate_cap_seconds=CPU_CAP, status='G5_live_smoke',
                     detail='G3a/G3b passed; running only the remaining full 12-iteration CPU smoke.')
        require(not OUTPUT.exists(), 'Amended full-gate output already exists; do not rerun G5')
        OUTPUT.mkdir(parents=True)
        checks = {key: old['checks'][key] for key in ('G1', 'G2', 'G4')}
        checks.update({key: revised['checks'][key] for key in ('G3a', 'G3b')})
        checks['G3'] = dict(passed=True, amendment=str(G3), definition='Approved G3a and G3b both passed')
        gate = dict(status='running', passed=False, checks=checks, source=str(campaign.SOURCE),
                    source_revision=REVISION, eta=derived['d_star'], b0=derived['b0'],
                    prior_gate_seconds=prior_seconds, cap_seconds=CPU_CAP,
                    preliminary=str(PRIOR), amended_G3=str(G3), amendment_path=str(amendment_path),
                    conservative_prior_cpu_wall_seconds=prior_seconds,
                    raw_internal_amended_gate_aggregate_seconds=internal_prior,
                    amended_G3_process_timing=str(G3_TIMING))
        atomic_json(OUTPUT / 'gate_summary.json', gate)
        save(True)
        argv = [str(campaign.PYTHON), '-B', str(ROOT / 'code/gate_adagrad_norm.py'), '--worker', 'smoke',
                '--source', str(campaign.SOURCE), '--output', str(OUTPUT / 'live_smoke_12itr'),
                '--eta', repr(derived['d_star']), '--b0', repr(derived['b0']), '--threads', '2']
        state['commands'].append(argv)
        append_json(DIRECTORY / 'commands.jsonl', dict(utc=utc(), action='G5_only', argv=argv))
        gate_env = os.environ.copy()
        gate_env['CUDA_VISIBLE_DEVICES'] = ''
        # All supervisor startup since module entry counts against the remaining CPU gate budget.
        deadline = STARTED + CPU_CAP - prior_seconds
        require(time.monotonic() < deadline, 'Aggregate CPU gate budget already exhausted')
        next_monitor, previous_counter, unchanged = 0.0, None, 0
        with (DIRECTORY / 'G5_live_smoke.out').open('x') as output:
            child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, env=gate_env,
                                     start_new_session=True)
            state['gate_pid'] = child.pid
            save()
            while child.poll() is None:
                require(time.monotonic() < deadline, 'Aggregate two-hour CPU gate cap exhausted')
                if time.monotonic() >= next_monitor:
                    try:
                        progress = read_json(OUTPUT / 'live_smoke_12itr/smoke_progress.json', {})
                    except json.JSONDecodeError:
                        progress = {}
                    counter = len(progress.get('iterations', []))
                    unchanged = unchanged + 1 if counter == previous_counter else 0
                    previous_counter = counter
                    append_json(DIRECTORY / 'monitor_history.jsonl', dict(utc=utc(), phase='G5',
                                completed_iterations=counter, suspected_stall=unchanged >= 2))
                    state['completed_cpu_smoke_iterations'] = counter
                    next_monitor = time.monotonic() + 1800
                    save(True)
                time.sleep(2)
            state['gate_returncode'] = child.returncode
            require(child.returncode == 0, 'G5 CPU smoke failed; see G5_live_smoke.out')
        gate['checks']['G5'] = read_json(OUTPUT / 'live_smoke_12itr/smoke.json', {})
        gate.update(elapsed_seconds=time.monotonic() - STARTED,
                    aggregate_gate_seconds=prior_seconds + time.monotonic() - STARTED)
        require(gate['aggregate_gate_seconds'] <= CPU_CAP, 'Aggregate CPU gate budget exceeded')
        gate['passed'] = all(gate['checks'].get(key, {}).get('passed') is True
                             for key in ('G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'))
        gate['status'] = 'passed' if gate['passed'] else 'failed'
        require(gate_allows_training(gate), 'A required amended gate failed; training is forbidden')
        reap()
        require(not descendants(os.getpid()), 'G5 left child processes behind')
        verify_source(plan['source_provenance'])
        atomic_json(OUTPUT / 'gate_summary.json', gate)
        state['full_gate_passed'] = True
        argv = [str(campaign.PYTHON), '-B', str(ROOT / 'code/campaign.py'), 'build-training',
                '--revision', REVISION, '--gate', str(OUTPUT / 'gate_summary.json'),
                '--calibration', str(CALIBRATION), '--amendment', str(amendment_path)]
        state.update(status='building_training_plan', detail='All amended gates passed; exact approved values are fixed.')
        state['commands'].append(argv)
        save(True)
        with (DIRECTORY / 'build_training.out').open('x') as output:
            result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=120)
        require(result.returncode == 0, 'Training plan build failed; no training controller submitted')
        target = ROOT / 'runs/plan_training.json'
        campaign.validate(read_json(target))
        # Read the complete pass marker again immediately before scheduler submission.
        require(gate_allows_training(read_json(OUTPUT / 'gate_summary.json')), 'Full gate no longer passes')
        argv = ['sbatch', '--parsable', '--output=' + str(ROOT / 'runs/logs/controller_training.out'),
                '--error=' + str(ROOT / 'runs/logs/controller_training.err'), str(ROOT / 'code/run_controller.sh'),
                str(campaign.PYTHON), str(ROOT / 'code/campaign.py'), str(target)]
        state.update(status='training_submission_pending', training_submission_attempted=True)
        state['commands'].append(argv)
        save()
        append_json(DIRECTORY / 'commands.jsonl', dict(utc=utc(), action='submit_training_controller', argv=argv))
        submit_env = os.environ.copy()
        submit_env.pop('CUDA_VISIBLE_DEVICES', None)
        response = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=30, env=submit_env)
        state['training_submission_response'] = dict(returncode=response.returncode, stdout=response.stdout,
                                                     stderr=response.stderr)
        save()
        require(response.returncode == 0, 'Training scheduler submission failed; inspect its recorded response')
        job_id = response.stdout.strip().split(';')[0]
        require(job_id.isdigit(), 'Ambiguous training-controller response; never retry automatically')
        state.update(status='handed_off', training_controller_job_id=job_id, ended_utc=utc(),
                     monitor_status='paused', detail='Training controller %s owns nine runs, evaluations and monitors.' % job_id)
        save(True)
        return 0
    except BaseException as error:
        remaining = stop_owned(child)
        state.update(status='blocked', error=str(error), traceback=traceback.format_exc(),
                     child_processes_after=remaining, monitor_status='paused', ended_utc=utc())
        gate_path = OUTPUT / 'gate_summary.json'
        if gate_path.exists():
            failed = read_json(gate_path, {})
            if failed.get('status') != 'passed':
                failed.update(status='failed', passed=False, error=str(error),
                              elapsed_seconds=time.monotonic() - STARTED,
                              aggregate_gate_seconds=state.get('prior_gate_seconds', 0) + time.monotonic() - STARTED)
                atomic_json(gate_path, failed)
        save()
        update_document(state, terminal=True)
        print(state['error'], file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--self-check', action='store_true')
    parser.add_argument('--amendment', type=Path, default=ROOT / 'provenance/g3_amendment.json')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
    else:
        sys.exit(execute(args.amendment.resolve()))
