#!/usr/bin/env python3
"""Wait for the original campaign, then dispatch exactly two bounded replacements.

This controller never changes or signals the original controller or its workers.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from campaign import (Campaign, GROUPS, ROOT, SLURM_TERMINAL, read, save, scheduler_state,
                      utc, validate_gate)
from recovery_scope import (CAP, COMMIT, ESTIMATE, PRIMARY_JOB, RUNS,
                            time_remaining, validate_authorization)

TERMINAL = ('complete', 'failed')


def submission_command(run, authorization_path):
    return ['sbatch', '--parsable', '--account=soal', '--partition=soal',
            '--nodes=1', '--gres=gpu:a6000:1', '--cpus-per-task=40', '--mem=64G',
            '--time=02:15:00', '--signal=B:TERM@5', '--no-requeue',
            '--export=PATH,HOME,USER,LOGNAME,LANG',
            '--job-name=stage2-' + run['run_id'], '--chdir=' + str(ROOT / 'dppo'),
            '--output=' + str(ROOT / 'setup' / ('slurm-' + run['run_id'] + '-%j.log')),
            str(ROOT / 'code/run_recovery_worker.sh'), '--run-id', run['run_id'],
            '--authorization', str(authorization_path)]


def original_jobs(state):
    """Return the complete original campaign's 19 distinct GPU allocations."""
    entries = state.get('submitted_jobs', [])
    expected = {condition + '_seed' + str(seed) for condition in GROUPS for seed in (0, 1, 2)}
    if len(entries) != 18 or {entry.get('run_id') for entry in entries} != expected:
        raise ValueError('Original campaign has not submitted exactly 18 unique ablations')
    jobs = {'R0_rerun_seed0': str(state.get('r0_job_id', ''))}
    jobs.update({entry['run_id']: str(entry.get('job_id', '')) for entry in entries})
    if (len(jobs) != 19 or len(set(jobs.values())) != 19 or
            any(not job.isdigit() for job in jobs.values())):
        raise ValueError('Original training allocation identities are ambiguous')
    return jobs


class RecoveryCampaign(Campaign):
    def __init__(self, authorization_path):
        # Never invoke the original constructor, which owns primary campaign state.
        self.authorization_path = Path(authorization_path).resolve()
        self.auth = validate_authorization(self.authorization_path)
        self.commit, self.gate_path = COMMIT, Path(self.auth['fixed_batch_gate'])
        self.path = Path(self.auth['recovery_record_path'])
        self.interrupted, self.scheduler_unknown_since = False, {}
        self.next_report = time.monotonic()
        gate = validate_gate(self.gate_path, COMMIT)
        job_id = os.environ.get('SLURM_JOB_ID', '')
        if not job_id.isdigit() or job_id == PRIMARY_JOB:
            raise ValueError('Recovery requires a separate CPU-only Slurm allocation')
        if time_remaining(self.auth) < CAP:
            raise ValueError('Too little original campaign time remains to launch replacements')
        self.state = {'status': 'waiting_for_original_campaign', 'started_utc': utc(),
                      'commit': COMMIT, 'controller_slurm_job': job_id,
                      'authorization_path': str(self.authorization_path),
                      'primary_controller_job_id': PRIMARY_JOB,
                      'primary_campaign_path': self.auth['primary_campaign_path'],
                      'matrix_path': self.auth['matrix_path'],
                      'original_deadline_utc': self.auth['original_deadline_utc'],
                      'worker_runtime_cap_seconds': CAP, 'estimate_seconds': ESTIMATE,
                      'max_recovery_runs': 2, 'max_simultaneous_recovery_gpus': 2,
                      'additional_gpu_seconds_cap': CAP * 2,
                      'fixed_batch_gate': gate, 'submitted_jobs': [], 'retries': 0,
                      'monitor_interval_seconds': 1800, 'monitor_status': 'running',
                      'trajectory_stopping': False}
        with self.path.open('x') as stream:
            json.dump(self.state, stream, indent=2)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'interrupted', True))
        self.record()

    def record(self, terminal=False):
        self.state['updated_utc'] = utc()
        save(self.path, self.state)
        if terminal or time.monotonic() >= self.next_report:
            self.next_report = time.monotonic() + 1800
            event = {key: self.state.get(key) for key in
                     ('status', 'updated_utc', 'controller_slurm_job', 'primary_controller_state',
                      'primary_campaign_status', 'condition_progress', 'monitor_status', 'error')}
            with (ROOT / 'runs/recovery_campaign_monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            print('Recovery monitor: ' + json.dumps(event), flush=True)
            report = ROOT / 'experiment.md'
            if report.exists():
                prose = report.read_text()
                start, end = '<!-- recovery-status-start -->', '<!-- recovery-status-end -->'
                if prose.count(start) == prose.count(end) == 1:
                    progress = ', '.join('%s: %s' % item for item in self.state.get('condition_progress', {}).items())
                    next_step = ('await original campaign completion' if self.state['status'] == 'waiting_for_original_campaign'
                                 else 'check replacement outputs' if self.state['status'] == 'running_replacements'
                                 else 'review terminal records')
                    message = ('Recovery (%s): %s%s; next: %s. [Run record](runs/recovery_campaign.json).') % (
                        utc(), self.state['status'], '; ' + progress if progress else '', next_step)
                    before, rest = prose.split(start, 1)
                    _, after = rest.split(end, 1)
                    temporary = report.with_name('experiment.md.recovery_pending')
                    temporary.write_text(before + start + '\n' + message + '\n' + end + after)
                    temporary.replace(report)

    def check_control(self):
        if self.interrupted:
            raise RuntimeError('Recovery controller interrupted; no automatic retry')
        if time_remaining(self.auth) <= 0:
            raise RuntimeError('Original campaign deadline reached')
        return validate_gate(self.gate_path, self.commit)

    def observe_allocation(self, job_id):
        observation = scheduler_state(job_id)
        if observation['state'] is None:
            since = self.scheduler_unknown_since.setdefault(str(job_id), time.monotonic())
            if time.monotonic() - since >= 300:
                raise RuntimeError('Slurm state unobservable for 300 seconds: ' + str(job_id))
        else:
            self.scheduler_unknown_since.pop(str(job_id), None)
        return observation

    def original_ready(self):
        """Require original controller completion and all 19 terminal GPU jobs."""
        self.check_control()
        state = read(Path(self.auth['primary_campaign_path']))
        if (not isinstance(state, dict) or state.get('commit') != COMMIT or
                str(state.get('controller_slurm_job')) != PRIMARY_JOB):
            raise ValueError('Primary campaign identity changed; recovery will not launch')
        observation = self.observe_allocation(PRIMARY_JOB)
        self.state.update(primary_controller_state=observation['state'],
                          primary_campaign_status=state.get('status'),
                          primary_controller_observation=observation)
        if observation['state'] not in SLURM_TERMINAL:
            return False
        if observation['state'] != 'COMPLETED':
            raise RuntimeError('Primary controller did not complete normally; recovery will not launch')
        if state.get('status') != 'runs_terminal_recovery_pending':
            raise RuntimeError('Original finalizer has not deferred analysis for authorized recovery')
        jobs = original_jobs(state)
        observations = {}
        for run_id, job in jobs.items():
            manifest = read(ROOT / 'runs' / run_id / 'manifest.json')
            if (not manifest or manifest.get('status') not in TERMINAL or
                    str(manifest.get('slurm_job_id')) != job or
                    manifest.get('monitor_status') != 'paused'):
                raise RuntimeError('Original run lacks its terminal record: ' + run_id)
            observations[run_id] = self.observe_allocation(job)
        self.state['original_allocation_observations'] = observations
        ready = all(item['state'] in SLURM_TERMINAL for item in observations.values())
        if ready:
            self.state['original_terminal_verified_utc'] = utc()
        return ready

    def submit(self, run):
        self.check_control()
        if not self.state.get('original_terminal_verified_utc'):
            raise RuntimeError('Cannot launch before original campaign allocations are terminal')
        if time_remaining(self.auth) < CAP:
            raise RuntimeError('Insufficient original deadline allowance for a full replacement')
        if run not in self.auth['runs'] or run['run_id'] not in RUNS:
            raise RuntimeError('Unapproved recovery run')
        if (len(self.state['submitted_jobs']) >= 2 or
                any(entry['run_id'] == run['run_id'] for entry in self.state['submitted_jobs'])):
            raise RuntimeError('Recovery submission already attempted; no retry')
        run_dir = ROOT / 'runs' / run['run_id']
        if (run_dir / 'manifest.json').exists() or (ROOT / (run['run_id'] + '.out')).exists():
            raise RuntimeError('Recovery output already exists; no overwrite or retry')
        command = submission_command(run, self.authorization_path)
        entry = {'run_id': run['run_id'], 'condition': run['condition'], 'seed': run['seed'],
                 'replaced_run_id': run['replaced_run_id'], 'command': command,
                 'submission_started_utc': utc()}
        self.state['submitted_jobs'].append(entry)
        self.record()  # Reserve the attempt before the external scheduler call.
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            entry['submission_error'] = str(error)
            self.record()
            raise RuntimeError('Recovery submission outcome unknown; no retry') from error
        entry.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
        if response.returncode or not re.fullmatch(r'\d+(?:;[^\n]+)?\n?', response.stdout):
            self.record()
            raise RuntimeError('Recovery submission failed or ambiguous; no retry')
        entry['job_id'] = response.stdout.strip().split(';')[0]
        self.record()

    def cancel_owned(self):
        """Cancel only jobs returned by these two successful sbatch submissions."""
        jobs = [entry['job_id'] for entry in self.state['submitted_jobs'] if entry.get('job_id')]
        if not jobs:
            return
        if (len(jobs) > 2 or len(set(jobs)) != len(jobs) or PRIMARY_JOB in jobs or
                any(not str(job).isdigit() for job in jobs)):
            raise RuntimeError('Recovery cancellation refused: invalid owned job identities')
        original = read(Path(self.auth['primary_campaign_path']))
        primary_jobs = {str(original.get('r0_job_id'))} | {
            str(entry.get('job_id')) for entry in original.get('submitted_jobs', [])}
        if primary_jobs.intersection(jobs):
            raise RuntimeError('Recovery cancellation refused: an original allocation was listed')
        command = ['scancel'] + jobs
        response = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.state['cancellation'] = {'command': command, 'returncode': response.returncode,
                                      'stdout': response.stdout, 'stderr': response.stderr}

    def recovery_done(self):
        self.check_control()
        states, allocation_states = {}, {}
        jobs = {entry['run_id']: entry['job_id'] for entry in self.state['submitted_jobs']}
        for run in self.auth['runs']:
            # Campaign.observe_run writes only this replacement's terminal evidence.
            decorated = dict(run, runtime_cap_seconds=CAP)
            manifest = self.observe_run(decorated, jobs[run['run_id']])
            states[run['run_id']] = manifest['status'] if manifest else 'queued_or_starting'
            allocation_states[run['run_id']] = self.observe_allocation(jobs[run['run_id']])
        self.state.update(condition_progress=states, recovery_allocation_observations=allocation_states)
        self.record()
        return (all(status in TERMINAL for status in states.values()) and
                all(item['state'] in SLURM_TERMINAL for item in allocation_states.values()))

    def run(self):
        try:
            while not self.original_ready():
                if time_remaining(self.auth) < CAP:
                    raise RuntimeError('Original campaign left too little time for replacements; none launched')
                self.record()
                time.sleep(15)
            self.state['status'] = 'running_replacements'
            self.record()
            for run in self.auth['runs']:
                self.submit(run)
            while not self.recovery_done():
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
        command = [sys.executable, str(ROOT / 'code/finalize_campaign.py'),
                   '--matrix', self.auth['matrix_path'], '--campaign', self.auth['primary_campaign_path'],
                   '--recovery-record', str(self.path)]
        self.state['finalizer_command'] = command
        self.record()
        # CPU-only analysis has no permission to launch or retry a training run.
        response = subprocess.run(command, check=False)
        if response.returncode:
            self.state.update(status='analysis_failed', analysis_returncode=response.returncode)
            self.record(terminal=True)
            raise RuntimeError('Recovery runs are terminal but final analysis failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    RecoveryCampaign(args.authorization).run()
