"""Adopt the approved Stage 2 campaign and admit independent seed groups in parallel.

This changes orchestration only. The original worker, command construction, scientific
gate, observations, failure handling and finalizer are reused unchanged.
"""
import argparse
import copy
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import time

from campaign import (Campaign, GROUPS, ROOT, SCHEDULER_UNKNOWN_LIMIT_SECONDS,
                      SLURM_TERMINAL, read, scheduler_state, utc, validate_gate)

TERMINAL = ('complete', 'failed')


def validate_documented_handoff_failure(state, handoff, r0_job):
    """Allow only this recorded controller handoff failure; never retry its runs."""
    failure_path = Path(handoff.get('failure_record', '')).resolve()
    if (handoff.get('allow_documented_handoff_failure') is not True or
            failure_path != (ROOT / 'provenance/parallel_handoff_failure.json').resolve() or
            str(handoff.get('old_controller_job_id')) != '17777246' or
            state.get('error') != 'Campaign controller interrupted; no continuation authorized'):
        raise ValueError('Stopped campaign lacks the exact documented handoff-failure exception')
    failure = read(failure_path)
    if (not isinstance(failure, dict) or failure.get('status') != 'handoff_failed_training_interrupted' or
            str(failure.get('controller_job_id')) != '17777246' or
            failure.get('replacement_controller_submitted') is not False or
            failure.get('training_runs_restarted') is not False or
            failure.get('scientific_source_unchanged') is not True or
            failure.get('old_controller_state') != state):
        raise ValueError('Handoff-failure record does not match the exact stopped controller state')
    expected_jobs = {str(r0_job)} | {str(entry['job_id']) for entry in state['submitted_jobs']}
    cancellation = state.get('cancellation', {})
    cancelled = cancellation.get('job_ids', [])
    if (not isinstance(cancelled, list) or len(cancelled) != len(expected_jobs) or
            set(map(str, cancelled)) != expected_jobs or cancellation.get('returncode') != 0):
        raise ValueError('Handoff cancellation record differs from the exact owned training jobs')
    run_jobs = {'R0_rerun_seed0': str(r0_job)}
    run_jobs.update({entry['run_id']: str(entry['job_id']) for entry in state['submitted_jobs']})
    records = failure.get('runs', [])
    if (not isinstance(records, list) or len(records) != len(run_jobs) or
            {item.get('run_id') for item in records} != set(run_jobs)):
        raise ValueError('Handoff failure must document every adopted run')
    for item in records:
        run_id = item['run_id']
        manifest = read(ROOT / 'runs' / run_id / 'manifest.json')
        if (not manifest or manifest.get('run_id') != run_id or
                str(manifest.get('slurm_job_id')) != run_jobs[run_id] or
                manifest.get('status') not in TERMINAL or manifest.get('status') != item.get('status') or
                manifest.get('monitor_status') != 'paused' or
                manifest.get('execution_phase') == 'startup_validation' or
                manifest.get('source_after_error') or manifest.get('startup_flag_validation_error') or
                manifest.get('nonfinite_failure')):
            raise ValueError('Handoff adoption requires terminal runs without scientific/source/setup failures: ' + run_id)
        if manifest['status'] == 'failed':
            if (manifest.get('failure_reason') != 'Received SIGTERM' or
                    item.get('failure_reason') != 'Received SIGTERM' or
                    manifest.get('execution_phase') != 'training' or not manifest.get('end_utc') or
                    not 0 < manifest.get('subprocess_wall_seconds', 0) <= 10800):
                raise ValueError('Only documented training SIGTERM interruptions may be adopted: ' + run_id)
    return failure_path


def validate_matrix(matrix, matrix_path):
    expected = {('R0', 0)} | {(name, seed) for name in GROUPS for seed in (0, 1, 2)}
    if not isinstance(matrix, list) or len(matrix) != 19:
        raise ValueError('Campaign matrix must contain exactly 19 full runs')
    if {(run['condition'], run['seed']) for run in matrix} != expected:
        raise ValueError('Campaign matrix differs from the approved conditions/seeds')
    for run in matrix:
        expected_id = ('R0_rerun' if run['condition'] == 'R0' else run['condition']) + '_seed' + str(run['seed'])
        if (run['run_id'] != expected_id or
                Path(run['manifest']).resolve() != (ROOT / 'runs' / expected_id / 'manifest.json').resolve() or
                run['runtime_cap_seconds'] != 10800 or run['gpu_type'] != 'NVIDIA RTX A6000'):
            raise ValueError('Campaign identity, path or resource cap differs from approved matrix')
    if Path(matrix_path).resolve() != (ROOT / 'provenance/run_matrix_revised.json').resolve():
        raise ValueError('Parallel continuation must use the original matrix')


def validate_resume(state, handoff, commit, r0_job, matrix_path, path, now):
    if (handoff.get('approved') is not True or handoff.get('max_training_jobs') != 9 or
            handoff.get('preserve_original_deadline') is not True or
            handoff.get('expected_commit') != commit or
            Path(handoff.get('state_path', '')).resolve() != path.resolve() or
            Path(handoff.get('matrix_path', '')).resolve() != matrix_path.resolve()):
        raise ValueError('Parallel handoff authorization is missing or differs from approved scope')
    if (state.get('commit') != commit or str(state.get('r0_job_id')) != str(r0_job) or
            state.get('started_utc') != handoff.get('expected_started_utc') or
            str(state.get('controller_slurm_job')) != str(handoff.get('old_controller_job_id')) or
            Path(state.get('matrix_path', '')).resolve() != matrix_path.resolve() or
            state.get('timeout_seconds') != 86400 or state.get('trajectory_stopping') is not False or
            state.get('retries') != 0 or state.get('monitor_interval_seconds') != 1800 or
            state.get('status') not in ('fixed_batch_gate_passed', 'running_condition', 'stopped')):
        raise ValueError('Existing campaign state is not an eligible unchanged continuation')
    old_job = str(handoff.get('old_controller_job_id', ''))
    if not old_job.isdigit():
        raise ValueError('Old controller job ID is required')
    entries = state.get('submitted_jobs')
    if not isinstance(entries, list) or len(entries) > 18:
        raise ValueError('Invalid previously submitted jobs')
    permitted = {name + '_seed' + str(seed) for name in GROUPS for seed in (0, 1, 2)}
    ids, jobs = set(), {str(r0_job), old_job}
    for entry in entries:
        run_id, job_id = entry.get('run_id'), str(entry.get('job_id', ''))
        if (run_id not in permitted or run_id in ids or not job_id.isdigit() or job_id in jobs or
                entry.get('returncode') != 0 or entry.get('submission_error')):
            raise ValueError('Duplicate, unknown, failed or ambiguous submission cannot be resumed')
        ids.add(run_id)
        jobs.add(job_id)
    if state['status'] == 'stopped':
        validate_documented_handoff_failure(state, handoff, r0_job)
    started = dt.datetime.fromisoformat(state['started_utc'])
    if started.tzinfo is None:
        raise ValueError('Original campaign start must include a timezone')
    remaining = (started + dt.timedelta(seconds=state['timeout_seconds']) - now).total_seconds()
    if not 0 < remaining <= 86400:
        raise ValueError('Original 24-hour deadline has expired or start lies in the future')
    return remaining


def pending_groups(matrix, entries, occupied, maximum=9):
    """Return whole missing seed groups that fit; adopted partial groups stay unique."""
    seen = {entry['run_id'] for entry in entries}
    groups, available = [], maximum - occupied
    for name in GROUPS:
        missing = [run for run in matrix if run['condition'] == name and run['run_id'] not in seen]
        if missing and len(missing) <= available:
            groups.append(missing)
            available -= len(missing)
    return groups


class ParallelCampaign(Campaign):
    def __init__(self, commit, r0_job, gate, matrix_path, handoff_path):
        # Deliberately do not call Campaign.__init__: it creates a new campaign.
        self.commit, self.r0_job, self.interrupted = commit, str(r0_job), False
        self.gate_path, self.matrix_path = Path(gate).resolve(), Path(matrix_path).resolve()
        self.handoff_path = Path(handoff_path).resolve()
        self.path = ROOT / 'runs/campaign_revised.json'
        self.matrix, self.state = read(self.matrix_path), read(self.path)
        if not self.state:
            raise ValueError('An existing campaign is required; this controller never starts a new one')
        validate_matrix(self.matrix, self.matrix_path)
        handoff = read(self.handoff_path)
        if not isinstance(handoff, dict):
            raise ValueError('Explicit handoff authorization is required')
        remaining = validate_resume(self.state, handoff, commit, self.r0_job, self.matrix_path,
                                    self.path, dt.datetime.now(dt.timezone.utc))
        gate_record = validate_gate(self.gate_path, commit)
        if self.state.get('fixed_batch_gate') != gate_record:
            raise ValueError('Fixed-batch gate differs from the existing campaign')
        old_observation = scheduler_state(handoff['old_controller_job_id'])
        if old_observation['state'] not in SLURM_TERMINAL:
            raise ValueError('Old controller must be confirmed terminal before takeover')
        new_job = os.environ.get('SLURM_JOB_ID', '')
        if not new_job.isdigit() or new_job in (str(handoff['old_controller_job_id']), self.r0_job):
            raise ValueError('Parallel continuation requires its own CPU Slurm controller allocation')
        self.lock = (ROOT / 'runs/parallel_controller.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Ensure the state observed above was not changed during validation.
        if read(self.path) != self.state:
            raise ValueError('Campaign state changed during handoff; no state was overwritten')
        self.r0_run = next(run for run in self.matrix if run['condition'] == 'R0')
        self.scheduler_unknown_since = {}
        self.started_clock = time.monotonic()
        original_deadline = (dt.datetime.fromisoformat(self.state['started_utc']) +
                             dt.timedelta(seconds=self.state['timeout_seconds']))
        remaining = (original_deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining <= 0:
            raise ValueError('Original campaign deadline elapsed during handoff validation')
        self.deadline = self.started_clock + remaining
        self.next_report = self.started_clock
        self.max_training_jobs = 9
        self.allocation_states = {}
        previous_stop = ({key: copy.deepcopy(self.state[key]) for key in
                          ('status', 'error', 'ended_utc', 'cancellation', 'monitor_status')
                          if key in self.state} if self.state['status'] == 'stopped' else None)
        handoff_record = {
            'utc': utc(), 'authorization': str(self.handoff_path),
            'from_job': str(handoff['old_controller_job_id']), 'to_job': new_job,
            'old_controller_observation': old_observation,
            'remaining_original_cap_seconds_at_start': remaining}
        if previous_stop is not None:
            handoff_record.update(previous_stop=previous_stop, failure_record=handoff['failure_record'],
                                  interrupted_runs_retried=False)
        self.state.setdefault('controller_handoffs', []).append(handoff_record)
        self.state.update(controller_slurm_job=new_job, scheduling_mode='parallel_conditions',
                          max_training_jobs=9, max_training_cpus=360,
                          status='running_parallel_conditions', condition='parallel',
                          monitor_status='running')
        self.record()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'interrupted', True))

    def refresh_allocations(self):
        """Count queued and draining allocations too, never just running manifests."""
        jobs = [(self.r0_run['run_id'], self.r0_job)] + [
            (entry['run_id'], entry['job_id']) for entry in self.state['submitted_jobs']]
        self.allocation_states = {}
        unknown = False
        for run_id, job_id in jobs:
            observed = scheduler_state(job_id)
            self.state.setdefault('parallel_allocation_observations', {})[run_id] = observed
            self.allocation_states[run_id] = observed['state']
            if observed['state'] is None:
                unknown = True
                since = self.scheduler_unknown_since.setdefault('allocation:' + str(job_id), time.monotonic())
                if time.monotonic() - since >= SCHEDULER_UNKNOWN_LIMIT_SECONDS:
                    raise RuntimeError('Allocation remained unobservable for 300 seconds: ' + str(job_id))
            else:
                self.scheduler_unknown_since.pop('allocation:' + str(job_id), None)
        occupied = sum(state not in SLURM_TERMINAL for state in self.allocation_states.values())
        self.state['active_or_pending_training_allocations'] = occupied
        self.state['dispatch_paused_for_unknown_allocation'] = unknown
        if occupied > self.max_training_jobs:
            raise RuntimeError('Existing training allocations exceed the approved nine-job cap')
        return occupied, unknown

    def submit(self, run):
        occupied = sum(state not in SLURM_TERMINAL for state in self.allocation_states.values())
        if self.state.get('dispatch_paused_for_unknown_allocation') or occupied >= self.max_training_jobs:
            raise RuntimeError('No confirmed capacity for another approved training allocation')
        super().submit(run)
        self.allocation_states[run['run_id']] = 'PENDING'
        self.state['active_or_pending_training_allocations'] = occupied + 1

    def cycle(self):
        r0, _ = self.check_r0()
        jobs = {entry['run_id']: entry['job_id'] for entry in self.state['submitted_jobs']}
        manifests = {self.r0_run['run_id']: r0}
        for run in self.matrix:
            if run['run_id'] in jobs:
                manifests[run['run_id']] = self.observe_run(run, jobs[run['run_id']])
        self.state['condition_progress'] = {
            run['run_id']: (manifests[run['run_id']]['status'] if manifests.get(run['run_id'])
                            else ('queued_or_starting' if run['run_id'] in jobs else 'not_submitted'))
            for run in self.matrix if run['condition'] != 'R0'}
        occupied, unknown = self.refresh_allocations()
        if not unknown:
            for group in pending_groups(self.matrix, self.state['submitted_jobs'], occupied,
                                        self.max_training_jobs):
                for run in group:
                    self.check_r0()
                    self.submit(run)
                    self.state['condition_progress'][run['run_id']] = 'queued_or_starting'
        self.record()
        return (len(manifests) == 19 and
                all(item and item['status'] in TERMINAL for item in manifests.values()) and
                not unknown and occupied == 0)

    def run(self):
        try:
            self.check_control()
            while not self.cycle():
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
    parser.add_argument('--handoff', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.commit) or not args.r0_job.isdigit():
        parser.error('A full source revision and numeric R0 job ID are required')
    ParallelCampaign(args.commit, args.r0_job, args.gate, args.matrix, args.handoff).run()
