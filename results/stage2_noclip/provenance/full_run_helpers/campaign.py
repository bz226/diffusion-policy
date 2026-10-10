"""Dispatch the approved ablation groups after R0 passes; never retry a run."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

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
    def __init__(self, commit, r0_job):
        self.commit, self.r0_job, self.interrupted = commit, r0_job, False
        self.matrix = read(ROOT / 'provenance/run_matrix.json')
        expected = {('R0', 0)} | {(condition, seed) for condition in GROUPS for seed in (0, 1, 2)}
        if not isinstance(self.matrix, list) or len(self.matrix) != 19:
            raise ValueError('Campaign matrix must contain exactly 19 full runs')
        if {(run['condition'], run['seed']) for run in self.matrix} != expected:
            raise ValueError('Campaign matrix differs from the approved conditions/seeds')
        for run in self.matrix:
            expected_id = run['condition'] + '_seed' + str(run['seed'])
            if (run['run_id'] != expected_id or
                    Path(run['manifest']).resolve() != (ROOT / 'runs' / expected_id / 'manifest.json').resolve() or
                    run['runtime_cap_seconds'] != 10800 or run['gpu_type'] != 'NVIDIA RTX A6000'):
                raise ValueError('Campaign identity, path or resource cap differs from approved matrix')
        self.r0_run = next(run for run in self.matrix if run['condition'] == 'R0')
        self.scheduler_unknown_since = {}
        self.path = ROOT / 'runs/campaign.json'
        self.state = {'status': 'waiting_for_r0_replay', 'started_utc': utc(), 'commit': commit,
                      'r0_job_id': r0_job, 'submitted_jobs': [], 'condition': None,
                      'retries': 0, 'controller_slurm_job': os.environ.get('SLURM_JOB_ID')}
        with self.path.open('x') as handle:
            json.dump(self.state, handle, indent=2)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'interrupted', True))

    def record(self):
        self.state['updated_utc'] = utc()
        save(self.path, self.state)

    def check_r0(self):
        if self.interrupted:
            raise RuntimeError('Campaign controller interrupted; no continuation authorized')
        manifest = self.observe_run(self.r0_run, self.r0_job)
        if manifest and manifest['status'] == 'failed':
            raise RuntimeError('R0 failed: ' + manifest.get('failure_reason', 'see manifest'))
        replay = read(ROOT / 'runs/R0_seed0/replay_status.json')
        if replay and replay.get('r0_passed') is False:
            raise RuntimeError('R0 replay failed; see replay_status.json')
        if manifest and manifest['status'] == 'complete' and not (replay and replay.get('r0_passed') is True):
            raise RuntimeError('R0 completed without a passed exact-replay gate; no ablation may proceed')
        return manifest, replay

    def observe_run(self, run, job_id):
        path = Path(run['manifest'])
        manifest = read(path)
        if manifest and manifest.get('status') in ('complete', 'failed'):
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
            while True:
                _, replay = self.check_r0()
                if replay and replay.get('r0_passed') is True:
                    break
                time.sleep(10)
            self.state['r0_replay_passed_utc'] = utc()
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
                if manifest and manifest['status'] == 'complete':
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
            self.record()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--r0-job', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}', args.commit)
    assert args.r0_job.isdigit()
    Campaign(args.commit, args.r0_job).run()
