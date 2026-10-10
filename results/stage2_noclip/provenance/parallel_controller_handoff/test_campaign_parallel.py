"""CPU-only orchestration checks; never contact Slurm or run DPPO."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import campaign
import campaign_parallel as cp

COMMIT = 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964'


class ParallelCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'runs').mkdir()
        (self.root / 'provenance').mkdir()
        self.matrix_path = self.root / 'provenance/run_matrix_revised.json'
        self.state_path = self.root / 'runs/campaign_revised.json'
        self.gate_path = self.root / 'runs/gate.json'
        self.handoff_path = self.root / 'provenance/handoff.json'
        self.matrix = []
        for condition in ['R0'] + list(cp.GROUPS):
            for seed in ([0] if condition == 'R0' else [0, 1, 2]):
                run_id = ('R0_rerun' if condition == 'R0' else condition) + '_seed' + str(seed)
                self.matrix.append({'condition': condition, 'seed': seed, 'run_id': run_id,
                    'manifest': str(self.root / 'runs' / run_id / 'manifest.json'),
                    'runtime_cap_seconds': 10800, 'gpu_type': 'NVIDIA RTX A6000'})
        self.started = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
        self.gate = {'path': str(self.gate_path), 'source_commit': COMMIT, 'passed': True}
        self.state = {'status': 'running_condition', 'started_utc': self.started.isoformat(),
            'commit': COMMIT, 'r0_job_id': '100', 'controller_slurm_job': '101',
            'submitted_jobs': [self.entry('NC1_seed' + str(seed), str(102 + seed)) for seed in range(3)],
            'condition': 'NC1', 'fixed_batch_gate': self.gate,
            'matrix_path': str(self.matrix_path), 'timeout_seconds': 86400,
            'trajectory_stopping': False, 'monitor_interval_seconds': 1800,
            'monitor_status': 'running', 'retries': 0, 'r0_status': 'running'}
        self.handoff = {'approved': True, 'old_controller_job_id': '101',
            'max_training_jobs': 9, 'preserve_original_deadline': True,
            'expected_started_utc': self.started.isoformat(), 'state_path': str(self.state_path),
            'matrix_path': str(self.matrix_path), 'expected_commit': COMMIT}
        for module in (cp, campaign):
            patcher = patch.object(module, 'ROOT', self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write_inputs()

    @staticmethod
    def entry(run_id, job_id):
        return {'run_id': run_id, 'job_id': job_id, 'returncode': 0, 'stdout': job_id + '\n'}

    def write_inputs(self):
        for path, value in [(self.matrix_path, self.matrix), (self.state_path, self.state),
                            (self.handoff_path, self.handoff)]:
            path.write_text(json.dumps(value))

    def construct(self, old_state='CANCELLED'):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '200'}), \
                patch.object(cp, 'validate_gate', return_value=self.gate), \
                patch.object(cp, 'scheduler_state', return_value={'state': old_state}), \
                patch.object(cp.signal, 'signal'):
            obj = cp.ParallelCampaign(COMMIT, '100', self.gate_path, self.matrix_path, self.handoff_path)
        self.addCleanup(obj.lock.close)
        return obj

    def test_resume_preserves_jobs_original_start_and_deadline(self):
        obj = self.construct()
        self.assertEqual(obj.state['submitted_jobs'], self.state['submitted_jobs'])
        self.assertEqual(obj.state['started_utc'], self.state['started_utc'])
        self.assertEqual(obj.state['timeout_seconds'], 86400)
        self.assertAlmostEqual(obj.deadline - time.monotonic(), 22 * 3600, delta=3)
        self.assertEqual(obj.state['controller_slurm_job'], '200')
        self.assertEqual(obj.state['max_training_jobs'], 9)

    def test_live_old_controller_is_rejected_without_state_write(self):
        before = self.state_path.read_text()
        with self.assertRaisesRegex(ValueError, 'confirmed terminal'):
            self.construct('RUNNING')
        self.assertEqual(self.state_path.read_text(), before)

    def test_unknown_old_controller_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'confirmed terminal'):
            self.construct(None)

    def test_no_ambiguous_retry(self):
        del self.state['submitted_jobs'][0]['job_id']
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.construct()

    def test_no_duplicate_run_or_job(self):
        for key in ('run_id', 'job_id'):
            with self.subTest(key=key):
                original = copy.deepcopy(self.state)
                self.state['submitted_jobs'][1][key] = self.state['submitted_jobs'][0][key]
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    self.construct()
                self.state = original

    def test_failed_submission_is_rejected(self):
        self.state['submitted_jobs'][0]['returncode'] = 1
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.construct()

    def test_expired_original_budget_is_rejected(self):
        expired = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25)).isoformat()
        self.state['started_utc'] = self.handoff['expected_started_utc'] = expired
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'deadline'):
            self.construct()

    def test_scientific_or_resource_matrix_change_is_rejected(self):
        self.matrix[-1]['runtime_cap_seconds'] = 10801
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'resource cap'):
            self.construct()

    def test_handoff_without_authorization_is_rejected(self):
        self.handoff['approved'] = False
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'authorization'):
            self.construct()

    def test_whole_group_admission_and_no_duplicates(self):
        groups = cp.pending_groups(self.matrix, self.state['submitted_jobs'], occupied=3)
        self.assertEqual([[run['condition'] for run in group] for group in groups],
                         [['NC2'] * 3, ['NC3'] * 3])
        self.assertEqual(cp.pending_groups(self.matrix, self.state['submitted_jobs'], occupied=7), [])

    def test_partial_existing_group_admits_only_missing_seeds(self):
        entries = self.state['submitted_jobs'] + [self.entry('NC2_seed0', '105')]
        groups = cp.pending_groups(self.matrix, entries, occupied=7)
        self.assertEqual([[run['run_id'] for run in group] for group in groups],
                         [['NC2_seed1', 'NC2_seed2']])

    def test_pending_and_completing_count_even_terminal_manifest(self):
        obj = self.construct()
        states = ['COMPLETED', 'RUNNING', 'PENDING', 'COMPLETING']
        with patch.object(cp, 'scheduler_state', side_effect=[{'state': s} for s in states]):
            count, unknown = obj.refresh_allocations()
        self.assertEqual(count, 3)
        self.assertFalse(unknown)

    def test_unknown_allocation_occupies_slot_and_pauses_dispatch(self):
        obj = self.construct()
        with patch.object(cp, 'scheduler_state', side_effect=[{'state': s} for s in ['COMPLETED', 'RUNNING', None, 'COMPLETED']]):
            self.assertEqual(obj.refresh_allocations(), (2, True))
        with self.assertRaisesRegex(RuntimeError, 'confirmed capacity'):
            obj.submit(self.matrix[4])

    def test_unknown_allocation_timeout_is_not_silently_terminal(self):
        obj = self.construct()
        obj.scheduler_unknown_since['allocation:100'] = time.monotonic() - 301
        with patch.object(cp, 'scheduler_state', return_value={'state': None}):
            with self.assertRaisesRegex(RuntimeError, 'unobservable'):
                obj.refresh_allocations()

    def test_submit_refuses_tenth_allocation(self):
        obj = self.construct()
        obj.allocation_states = {str(i): 'PENDING' for i in range(9)}
        with self.assertRaisesRegex(RuntimeError, 'confirmed capacity'):
            obj.submit(self.matrix[4])

    def simulate_cycle(self, statuses, scheduler_states):
        obj = self.construct()
        submissions = []
        def fake_submit(run):
            self.assertNotIn(run['run_id'], [entry['run_id'] for entry in obj.state['submitted_jobs']])
            self.assertLess(sum(s not in cp.SLURM_TERMINAL for s in obj.allocation_states.values()), 9)
            obj.state['submitted_jobs'].append(self.entry(run['run_id'], str(300 + len(submissions))))
            obj.allocation_states[run['run_id']] = 'PENDING'
            submissions.append(run['run_id'])
        def observe(run, _job):
            return {'status': statuses.get(run['run_id'], 'running')}
        with patch.object(obj, 'check_r0', return_value=({'status': statuses.get('R0_rerun_seed0', 'running')}, None)), \
                patch.object(obj, 'observe_run', side_effect=observe), \
                patch.object(cp, 'scheduler_state', side_effect=lambda job: {'state': scheduler_states.get(str(job), 'RUNNING')}), \
                patch.object(obj, 'submit', side_effect=fake_submit):
            finished = obj.cycle()
        return obj, submissions, finished

    def test_cycle_admits_parallel_groups_up_to_nine(self):
        obj, submitted, finished = self.simulate_cycle({}, {'100': 'COMPLETED'})
        self.assertEqual(submitted, ['NC2_seed0', 'NC2_seed1', 'NC2_seed2', 'NC3_seed0', 'NC3_seed1', 'NC3_seed2'])
        self.assertFalse(finished)
        self.assertEqual(len(obj.state['submitted_jobs']), 9)

    def test_r0_occupies_slot_and_caps_group_admission(self):
        _, submitted, _ = self.simulate_cycle({}, {})
        self.assertEqual(submitted, ['NC2_seed0', 'NC2_seed1', 'NC2_seed2'])

    def test_numerical_terminal_failure_does_not_retry_or_block_new_group(self):
        _, submitted, finished = self.simulate_cycle({'NC1_seed0': 'failed'}, {'102': 'FAILED'})
        self.assertNotIn('NC1_seed0', submitted)
        self.assertEqual(len(submitted), 6)
        self.assertFalse(finished)

    def test_infrastructure_failure_uses_existing_abort_semantics(self):
        obj = self.construct()
        with self.assertRaisesRegex(RuntimeError, 'Source/setup validation'):
            obj.check_source_failure({'status': 'failed', 'execution_phase': 'startup_validation'})

    def test_scientific_failure_can_remain_terminal(self):
        obj = self.construct()
        obj.check_source_failure({'status': 'failed', 'execution_phase': 'training', 'nonfinite_failure': True})

    def test_complete_campaign_reaches_finalization_only_allocation_terminal(self):
        self.state['submitted_jobs'] = [self.entry(run['run_id'], str(300 + i))
                                       for i, run in enumerate(self.matrix) if run['condition'] != 'R0']
        self.write_inputs()
        statuses = {run['run_id']: 'complete' for run in self.matrix}
        scheduler = {entry['job_id']: 'COMPLETED' for entry in self.state['submitted_jobs']}
        scheduler['100'] = 'COMPLETED'
        obj, submitted, finished = self.simulate_cycle(statuses, scheduler)
        self.assertTrue(finished)
        self.assertEqual(submitted, [])

    def test_worker_command_unchanged(self):
        obj = self.construct()
        obj.allocation_states = {'R0_rerun_seed0': 'COMPLETED'}
        run = next(run for run in self.matrix if run['run_id'] == 'NC3_seed0')
        response = type('Response', (), {'returncode': 0, 'stdout': '900\n', 'stderr': ''})()
        with patch.object(obj, 'check_control'), patch.object(campaign.subprocess, 'run', return_value=response) as sub:
            obj.submit(run)
        command = sub.call_args.args[0]
        for argument in ('--gres=gpu:a6000:1', '--cpus-per-task=40', '--mem=64G', '--time=03:00:00'):
            self.assertIn(argument, command)
        self.assertEqual(command[command.index('--') + 1:], campaign.native_command(run))
        self.assertEqual(obj.allocation_states['NC3_seed0'], 'PENDING')


if __name__ == '__main__':
    unittest.main(verbosity=2)
