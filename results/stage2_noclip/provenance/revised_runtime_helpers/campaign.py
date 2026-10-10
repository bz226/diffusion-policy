"""Dispatch approved ablations after the fixed-batch gate; never retry a run."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from fixed_batch_gate import validate_gate

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('NC1', 'NC2', 'NC3', 'NC4_lr1e-4', 'NC4_lr1e-3', 'NC4_lr3e-3')
SLURM_TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED',
                  'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT', 'TIMEOUT'}
SCHEDULER_UNKNOWN_LIMIT_SECONDS = 300


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def save(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def scheduler_state(job_id):
    """Observe the exact parent allocation, never infer a state from its absence."""
    evidence = []
    commands = [
        ['squeue', '--noheader', '--jobs=' + str(job_id), '--format=%i|%T'],
        ['sacct', '--noheader', '--parsable2', '--allocations', '--jobs=' + str(job_id),
         '--format=JobIDRaw,State,ExitCode,Start,End'],
    ]
    for command in commands:
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            evidence.append({'command': command, 'error': str(error)})
            continue
        evidence.append({'command': command, 'returncode': response.returncode,
                         'stdout': response.stdout, 'stderr': response.stderr})
        if response.returncode:
            continue
        records = [line.strip().split('|') for line in response.stdout.splitlines() if line.strip()]
        matching = [fields for fields in records if fields[0].strip() == str(job_id)]
        if len(matching) == 1 and len(matching[0]) >= 2:
            fields = matching[0]
            raw_state = fields[1].strip()
            state = raw_state.split()[0].rstrip('+') if raw_state else None
            return {'job_id': str(job_id), 'state': state, 'raw_state': raw_state,
                    'exit_code': fields[2].strip() if len(fields) > 2 else None,
                    'start': fields[3].strip() if len(fields) > 3 else None,
                    'end': fields[4].strip() if len(fields) > 4 else None,
                    'source': command[0], 'observed_utc': utc(), 'evidence': evidence}
    return {'job_id': str(job_id), 'state': None, 'observed_utc': utc(), 'evidence': evidence}


def native_command(run):
    name, seed = run['condition'], run['seed']
    args = ['python', 'script/run.py', '--config-dir=cfg/gym/finetune/halfcheetah-v2',
            '--config-name=ft_ppo_diffusion_mlp', 'train.n_train_itr=140',
            'train.save_model_freq=35', 'seed=' + str(seed), 'wandb=null']
    if name != 'R0':
        args += ['model.clip_ploss_coef=1e6', 'model.clip_ploss_coef_base=1e6', 'train.target_kl=null']
    if name in ('NC2', 'NC3') or name.startswith('NC4'):
        args += ['+model.clamp_logprob=false', 'model.randn_clip_value=100']
    if name == 'NC3':
        args += ['+model.logprob_reduce=sum']
    if name.startswith('NC4'):
        lr = name.split('_lr')[1]
        args += ['+train.actor_single_step=true', 'train.actor_lr=' + lr,
                 'train.actor_lr_scheduler.min_lr=' + lr]
    args += ['logdir=' + str(ROOT / 'runs' / run['run_id'] / 'native')]
    return args


class Campaign:
    def __init__(self, commit, r0_job, gate, matrix_path=None, timeout_seconds=86400):
        self.commit, self.r0_job, self.interrupted = commit, r0_job, False
        self.gate_path = Path(gate).resolve()
        gate_record = validate_gate(self.gate_path, commit)
        self.matrix_path = Path(matrix_path or ROOT / 'provenance/run_matrix_revised.json').resolve()
        self.matrix = read(self.matrix_path)
        if not 0 < timeout_seconds <= 86400:
            raise ValueError('Controller cap exceeds the approved 24 hours')
        self.started_clock = time.monotonic()
        self.deadline = self.started_clock + timeout_seconds
        self.next_report = self.started_clock
        expected = {('R0', 0)} | {(condition, seed) for condition in GROUPS for seed in (0, 1, 2)}
        if not isinstance(self.matrix, list) or len(self.matrix) != 19:
            raise ValueError('Campaign matrix must contain exactly 19 full runs')
        if {(run['condition'], run['seed']) for run in self.matrix} != expected:
            raise ValueError('Campaign matrix differs from the approved conditions/seeds')
        for run in self.matrix:
            expected_id = ('R0_rerun' if run['condition'] == 'R0' else run['condition']) + '_seed' + str(run['seed'])
            if (run['run_id'] != expected_id or
                    Path(run['manifest']).resolve() != (ROOT / 'runs' / expected_id / 'manifest.json').resolve() or
                    run['runtime_cap_seconds'] != 10800 or run['gpu_type'] != 'NVIDIA RTX A6000'):
                raise ValueError('Campaign identity, path or resource cap differs from approved matrix')
        self.r0_run = next(run for run in self.matrix if run['condition'] == 'R0')
        self.scheduler_unknown_since = {}
        self.path = ROOT / 'runs/campaign_revised.json'
        self.state = {'status': 'fixed_batch_gate_passed', 'started_utc': utc(), 'commit': commit,
                      'r0_job_id': r0_job, 'submitted_jobs': [], 'condition': None,
                      'fixed_batch_gate': gate_record, 'matrix_path': str(self.matrix_path),
                      'timeout_seconds': timeout_seconds, 'trajectory_stopping': False,
                      'monitor_interval_seconds': 1800, 'monitor_status': 'running',
                      'retries': 0, 'controller_slurm_job': os.environ.get('SLURM_JOB_ID')}
        with self.path.open('x') as handle:
            json.dump(self.state, handle, indent=2)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'interrupted', True))

    def record(self, terminal=False):
        self.state['updated_utc'] = utc()
        save(self.path, self.state)
        if terminal or time.monotonic() >= self.next_report:
            self.next_report = time.monotonic() + 1800
            event = {'utc': utc(), 'status': self.state['status'], 'condition': self.state['condition'],
                     'submitted_ablations': len(self.state['submitted_jobs']),
                     'condition_progress': self.state.get('condition_progress', {}),
                     'r0_status': self.state.get('r0_status'),
                     'monitor_status': self.state['monitor_status']}
            with (ROOT / 'runs/campaign_revised_monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            report_path = ROOT / 'experiment.md'
            if report_path.exists():
                prose = report_path.read_text()
                start, end = '<!-- campaign-status-start -->', '<!-- campaign-status-end -->'
                if prose.count(start) == 1 and prose.count(end) == 1:
                    message = ('Current execution (%s): %s; condition %s; %d/18 ablations submitted; '
                               'R0 %s. [Live run record](runs/campaign_revised.json).') % (
                                   utc(), self.state['status'], self.state['condition'] or 'not started',
                                   len(self.state['submitted_jobs']), self.state.get('r0_status', 'starting'))
                    before, rest = prose.split(start, 1)
                    _, after = rest.split(end, 1)
                    temporary = report_path.with_name('experiment.md.campaign_pending')
                    temporary.write_text(before + start + '\n' + message + '\n' + end + after)
                    temporary.replace(report_path)

    def check_control(self):
        if self.interrupted:
            raise RuntimeError('Campaign controller interrupted; no continuation authorized')
        if time.monotonic() >= self.deadline:
            raise RuntimeError('Approved 24-hour campaign-controller cap reached')
        # This is the only code-equivalence gate. Trajectory values never enter it.
        return validate_gate(self.gate_path, self.commit)

    def check_r0(self):
        self.check_control()
        manifest = self.observe_run(self.r0_run, self.r0_job)
        self.state['r0_status'] = manifest['status'] if manifest else 'queued_or_starting'
        observations = read(Path(self.r0_run['manifest']).parent / 'trajectory_observations.json')
        self.state['r0_trajectory_red_flags'] = observations.get('red_flags', []) if observations else []
        return manifest, observations

    def check_source_failure(self, manifest):
        if manifest and manifest.get('status') == 'failed':
            if (manifest.get('execution_phase') == 'startup_validation' or manifest.get('source_after_error')
                    or manifest.get('startup_flag_validation_error')):
                raise RuntimeError('Source/setup validation failed; no further dispatch: ' +
                                   manifest.get('failure_reason', manifest.get('source_after_error', 'see manifest')))

    def observe_run(self, run, job_id):
        path = Path(run['manifest'])
        manifest = read(path)
        if manifest and manifest.get('status') in ('complete', 'failed'):
            self.check_source_failure(manifest)
            return manifest
        observation = scheduler_state(job_id)
        self.state.setdefault('scheduler_observations', {})[run['run_id']] = observation
        if observation['state'] is None:
            since = self.scheduler_unknown_since.setdefault(str(job_id), time.monotonic())
            if time.monotonic() - since >= SCHEDULER_UNKNOWN_LIMIT_SECONDS:
                raise RuntimeError('Slurm state remained unobservable for 300 seconds for job %s; no outcome is inferred' % job_id)
            return manifest
        self.scheduler_unknown_since.pop(str(job_id), None)
        if observation['state'] not in SLURM_TERMINAL:
            return manifest
        # Re-read after querying Slurm so a normally completed worker's terminal
        # record wins a race with this external observer.
        manifest = read(path)
        if manifest and manifest.get('status') in ('complete', 'failed'):
            self.check_source_failure(manifest)
            return manifest
        run_dir = path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        if manifest is not None:
            with (run_dir / 'manifest_before_external_failure.json').open('x') as handle:
                json.dump(manifest, handle, indent=2, allow_nan=False)
                handle.write('\n')
        native = run_dir / 'native'
        reason = ('Slurm job %s reached %s (ExitCode %s) without a terminal supervisor manifest; '
                  'scientific completion is unverified') % (job_id, observation['raw_state'], observation['exit_code'])
        updated = dict(manifest or {})
        updated.update({'run_id': run['run_id'], 'condition': run['condition'], 'seed': run['seed'],
                        'status': 'failed', 'failure_reason': reason,
                        'failure_origin': 'external_slurm_terminal_observation',
                        'external_slurm_observation': observation,
                        'external_terminal_observed_utc': utc(), 'slurm_job_id': str(job_id),
                        'result_path': str(native / 'result.pkl'), 'logdir': str(native),
                        'verification_complete': False, 'monitor_status': 'paused',
                        'timeout_seconds': run['runtime_cap_seconds']})
        updated.setdefault('nonfinite_failure', False)  # No nonfinite event was observed by this controller.
        updated.setdefault('failing_iteration', None)
        updated['checkpoints_found'] = [str(item) for item in sorted((native / 'checkpoint').glob('state_*.pt')) if item.is_file()]
        progress = read(run_dir / 'progress.json')
        if progress is not None:
            updated['progress_at_external_failure'] = progress
        with (run_dir / 'external_failure.json').open('x') as handle:
            json.dump({'reason': reason, 'scheduler': observation, 'observed_utc': utc()}, handle, indent=2, allow_nan=False)
            handle.write('\n')
        save(path, updated)
        self.state.setdefault('external_failures', []).append({'run_id': run['run_id'], 'manifest': str(path), 'scheduler_state': observation['state']})
        self.record()
        return updated

    def submit(self, run):
        self.check_control()
        if run['condition'] == 'R0' or run not in self.matrix:
            raise RuntimeError('Only the 18 approved ablation runs may be submitted')
        if len(self.state['submitted_jobs']) >= 18 or any(entry['run_id'] == run['run_id'] for entry in self.state['submitted_jobs']):
            raise RuntimeError('Refusing extra or repeated submission ' + run['run_id'])
        run_dir = ROOT / 'runs' / run['run_id']
        if (run_dir / 'manifest.json').exists() or (ROOT / (run['run_id'] + '.out')).exists():
            raise RuntimeError('Refusing duplicate run ' + run['run_id'])
        # One GPU per allocation; Slurm assigns distinct available devices.
        command = ['sbatch', '--parsable', '--account=soal', '--partition=soal',
                   '--nodes=1', '--gres=gpu:a6000:1',
                   '--cpus-per-task=40', '--mem=64G', '--time=03:00:00',
                   '--signal=B:TERM@30', '--no-requeue', '--export=PATH,HOME,USER,LOGNAME,LANG',
                   '--job-name=stage2-' + run['run_id'], '--chdir=' + str(ROOT / 'dppo'),
                   '--output=' + str(ROOT / 'setup' / ('slurm-' + run['run_id'] + '-%j.log')),
                   str(ROOT / 'code/run_worker.sh'), '--run-id', run['run_id'],
                   '--expected-commit', self.commit, '--timeout-seconds', '10800',
                   '--fixed-batch-gate', str(self.gate_path),
                   '--estimate-seconds', '8100', '--'] + native_command(run)
        entry = {'run_id': run['run_id'], 'command': command, 'submission_started_utc': utc()}
        self.state['submitted_jobs'].append(entry)
        self.record()  # A interrupted submission is visible and is never retried blindly.
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            entry['submission_error'] = str(error)
            self.record()
            raise RuntimeError('Slurm submission outcome is unknown; it will not be retried: ' + run['run_id']) from error
        entry.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
        if response.returncode or not re.fullmatch(r'\d+(?:;[^\n]+)?\n?', response.stdout):
            self.record()
            raise RuntimeError('Slurm submission failed or was ambiguous: ' + run['run_id'])
        entry['job_id'] = response.stdout.strip().split(';')[0]
        entry['submitted_utc'] = utc()
        self.record()

    def cancel_owned(self):
        jobs = [str(self.r0_job)] + [j['job_id'] for j in self.state['submitted_jobs'] if 'job_id' in j]
        response = subprocess.run(['scancel'] + jobs, capture_output=True, text=True, timeout=60)
        self.state['cancellation'] = {'job_ids': jobs, 'returncode': response.returncode,
                                      'stdout': response.stdout, 'stderr': response.stderr}

    def run(self):
        try:
            self.check_control()
            self.state['fixed_batch_gate_confirmed_utc'] = utc()
            for name in GROUPS:
                self.check_r0()
                self.state.update(status='running_condition', condition=name)
                self.record()
                runs = [run for run in self.matrix if run['condition'] == name]
                assert len(runs) == 3 and sorted(run['seed'] for run in runs) == [0, 1, 2]
                for run in runs:
                    self.check_r0()
                    self.submit(run)
                while True:
                    self.check_r0()
                    jobs = {entry['run_id']: entry['job_id'] for entry in self.state['submitted_jobs'] if 'job_id' in entry}
                    manifests = [self.observe_run(run, jobs[run['run_id']]) for run in runs]
                    self.state['condition_progress'] = {
                        run['run_id']: m['status'] if m else 'queued_or_starting'
                        for run, m in zip(runs, manifests)}
                    self.record()
                    if all(m and m['status'] in ('complete', 'failed') for m in manifests):
                        break
                    time.sleep(15)
            while True:
                manifest, _ = self.check_r0()
                if manifest and manifest['status'] in ('complete', 'failed'):
                    break
                time.sleep(15)
            self.state.update(status='runs_terminal_analysis_pending', ended_utc=utc())
        except BaseException as error:
            self.state.update(status='stopped', error=str(error), ended_utc=utc())
            try:
                self.cancel_owned()
            except Exception as cancel_error:
                self.state['cancellation_error'] = str(cancel_error)
            raise
        finally:
            self.state['monitor_status'] = 'paused'
            self.record(terminal=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--r0-job', required=True)
    parser.add_argument('--gate', required=True)
    parser.add_argument('--matrix', default=str(ROOT / 'provenance/run_matrix_revised.json'))
    parser.add_argument('--timeout-seconds', type=int, default=86400)
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}', args.commit)
    assert args.r0_job.isdigit()
    Campaign(args.commit, args.r0_job, args.gate, args.matrix, args.timeout_seconds).run()
