"""Fake scheduler tests: no training, scheduler mutation, or external processes."""
import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import campaign
import recovery_campaign as recovery
import recovery_scope as scope


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(scope.ROOT / 'tmp'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'runs').mkdir()
        self.auth_path = scope.ROOT / 'provenance/interrupted_recovery_authorization.json'
        self.auth = scope.validate_authorization(self.auth_path)
        self.controller = recovery.RecoveryCampaign.__new__(recovery.RecoveryCampaign)
        obj = self.controller
        obj.auth = dict(self.auth, primary_campaign_path=str(self.root / 'primary.json'))
        obj.authorization_path = self.auth_path
        obj.commit, obj.gate_path = scope.COMMIT, Path(self.auth['fixed_batch_gate'])
        obj.path = self.root / 'runs/recovery_campaign.json'
        obj.interrupted, obj.scheduler_unknown_since = False, {}
        obj.next_report = 0
        obj.state = {'submitted_jobs': [], 'original_terminal_verified_utc': 'confirmed',
                     'monitor_status': 'running'}
        obj.record = lambda terminal=False: None
        obj.check_control = lambda: None
        self.root_patch = patch.object(recovery, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.remaining = patch.object(recovery, 'time_remaining', return_value=20000)
        self.remaining_mock = self.remaining.start()
        self.addCleanup(self.remaining.stop)
        self.primary = {
            'commit': scope.COMMIT, 'controller_slurm_job': scope.PRIMARY_JOB,
            'status': 'runs_terminal_recovery_pending', 'r0_job_id': '100',
            'submitted_jobs': [{'run_id': condition + '_seed' + str(seed),
                                'job_id': str(101 + 3 * index + seed)}
                               for index, condition in enumerate(campaign.GROUPS)
                               for seed in range(3)]}
        self.write_primary()
        for run_id, job_id in recovery.original_jobs(self.primary).items():
            self.write_manifest(run_id, job_id)

    def write_primary(self):
        Path(self.controller.auth['primary_campaign_path']).write_text(json.dumps(self.primary))

    def write_manifest(self, run_id, job_id, status='complete', **extra):
        target = self.root / 'runs' / run_id / 'manifest.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(dict(run_id=run_id, slurm_job_id=job_id,
                                         status=status, monitor_status='paused', **extra)))
        return target

    def observed(self, job, state='COMPLETED'):
        return dict(job_id=str(job), state=state)

    def test_authorization_scope_rejects_changed_choices(self):
        for key, bad in [('worker_runtime_cap_seconds', 10800), ('max_recovery_runs', 3),
                         ('primary_controller_job_id', '11'), ('gpu_type', 'NVIDIA H200'),
                         ('cpus_per_run', 20), ('no_automatic_retries', False),
                         ('worst_case_total_gpu_hours', 59)]:
            with self.subTest(key=key):
                changed = dict(self.auth, **{key: bad})
                with patch.object(Path, 'read_text', return_value=json.dumps(changed)):
                    with self.assertRaises(ValueError):
                        scope.validate_authorization(self.auth_path)

    def test_worker_commands_differ_only_in_log_directory(self):
        for run_id, (condition, seed, previous) in scope.RUNS.items():
            args = scope.worker_args(run_id, self.auth)
            original = campaign.native_command(dict(run_id=previous, condition=condition, seed=seed))
            self.assertEqual(args.command[:-1], original[:-1])
            self.assertEqual(args.command[-1], 'logdir=' + str(scope.ROOT / 'runs' / run_id / 'native'))
            self.assertEqual((args.rows, args.timeout_seconds, args.estimate_seconds), (140, 8100, 7500))
            self.assertEqual((args.condition, args.seed, args.probe), (condition, seed, False))
        with self.assertRaises(ValueError):
            scope.worker_args('NC1_recovery_seed0', self.auth)

    def test_original_running_does_not_query_or_modify_workers(self):
        self.controller.observe_allocation = lambda job: self.observed(job, 'RUNNING')
        before = sorted((path, path.read_bytes()) for path in self.root.rglob('*.json'))
        self.assertFalse(self.controller.original_ready())
        self.assertEqual(before, sorted((path, path.read_bytes()) for path in self.root.rglob('*.json')))

    def test_primary_must_complete_and_analysis_must_be_deferred(self):
        self.controller.observe_allocation = lambda job: self.observed(job, 'FAILED')
        with self.assertRaisesRegex(RuntimeError, 'complete normally'):
            self.controller.original_ready()
        self.controller.observe_allocation = lambda job: self.observed(job)
        self.primary['status'] = 'candidate_artifacts_ready_for_review'
        self.write_primary()
        with self.assertRaisesRegex(RuntimeError, 'deferred'):
            self.controller.original_ready()

    def test_original_terminal_failed_runs_remain_preserved(self):
        self.write_manifest('R0_rerun_seed0', '100', status='failed', failure_reason='Received SIGTERM')
        self.controller.observe_allocation = lambda job: self.observed(job)
        before = sorted((path, path.read_bytes()) for path in self.root.rglob('*.json'))
        self.assertTrue(self.controller.original_ready())
        self.assertEqual(before, sorted((path, path.read_bytes()) for path in self.root.rglob('*.json')))
        self.assertEqual(len(self.controller.state['original_allocation_observations']), 19)

    def test_terminal_manifest_is_not_enough_while_allocation_drains(self):
        self.controller.observe_allocation = lambda job: self.observed(job, 'COMPLETING' if job == '101' else 'COMPLETED')
        self.assertFalse(self.controller.original_ready())

    def test_original_missing_or_wrong_terminal_record_blocks_launch(self):
        self.write_manifest('R0_rerun_seed0', '999')
        self.controller.observe_allocation = lambda job: self.observed(job)
        with self.assertRaisesRegex(RuntimeError, 'terminal record'):
            self.controller.original_ready()

    def test_original_unknown_slot_cannot_replace_an_expected_allocation(self):
        self.primary['submitted_jobs'][0]['run_id'] = 'unrelated_seed0'
        with self.assertRaisesRegex(ValueError, '18 unique'):
            recovery.original_jobs(self.primary)

    def test_submission_before_ready_or_too_late_is_refused(self):
        run = self.auth['runs'][0]
        del self.controller.state['original_terminal_verified_utc']
        with patch.object(recovery.subprocess, 'run') as external:
            with self.assertRaisesRegex(RuntimeError, 'before original'):
                self.controller.submit(run)
            external.assert_not_called()
            self.controller.state['original_terminal_verified_utc'] = 'confirmed'
            self.remaining_mock.return_value = 8099
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                self.controller.submit(run)
            external.assert_not_called()

    def test_two_exact_resource_submissions_no_duplicates(self):
        responses = [SimpleNamespace(returncode=0, stdout='900\n', stderr=''),
                     SimpleNamespace(returncode=0, stdout='901\n', stderr='')]
        with patch.object(recovery.subprocess, 'run', side_effect=responses) as external:
            for run in self.auth['runs']:
                self.controller.submit(run)
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                self.controller.submit(self.auth['runs'][0])
        self.assertEqual(external.call_count, 2)
        for call in external.call_args_list:
            command = call.args[0]
            for resource in ('--gres=gpu:a6000:1', '--cpus-per-task=40', '--mem=64G',
                             '--time=02:15:00', '--no-requeue'):
                self.assertIn(resource, command)
            self.assertEqual(command[0], 'sbatch')
        self.assertEqual([entry['job_id'] for entry in self.controller.state['submitted_jobs']], ['900', '901'])

    def test_ambiguous_submission_is_never_retried(self):
        response = SimpleNamespace(returncode=0, stdout='not-a-job-id', stderr='')
        with patch.object(recovery.subprocess, 'run', return_value=response) as external:
            with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                self.controller.submit(self.auth['runs'][0])
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                self.controller.submit(self.auth['runs'][0])
            external.assert_called_once()
        self.assertEqual(len(self.controller.state['submitted_jobs']), 1)

    def test_cancellation_only_owns_two_new_jobs(self):
        self.controller.state['submitted_jobs'] = [{'job_id': '900'}, {'job_id': '901'}]
        response = SimpleNamespace(returncode=0, stdout='', stderr='')
        with patch.object(recovery.subprocess, 'run', return_value=response) as external:
            self.controller.cancel_owned()
            self.assertEqual(external.call_args.args[0], ['scancel', '900', '901'])
            external.reset_mock()
            self.controller.state['submitted_jobs'] = [{'job_id': '100'}]
            with self.assertRaisesRegex(RuntimeError, 'original allocation'):
                self.controller.cancel_owned()
            external.assert_not_called()

    def test_nonfinite_replacement_failure_does_not_drop_other_run(self):
        self.controller.auth['runs'] = []
        self.controller.state['submitted_jobs'] = []
        for index, run in enumerate(self.auth['runs']):
            job_id = str(900 + index)
            path = self.write_manifest(run['run_id'], job_id, status='failed' if index == 0 else 'complete',
                                       nonfinite_failure=index == 0, execution_phase='training')
            self.controller.auth['runs'].append(dict(run, manifest=str(path)))
            self.controller.state['submitted_jobs'].append(dict(run_id=run['run_id'], job_id=job_id))
        self.controller.observe_allocation = lambda job: self.observed(job)
        self.assertTrue(self.controller.recovery_done())
        self.assertEqual(set(self.controller.state['condition_progress'].values()), {'failed', 'complete'})

    def test_startup_failure_is_fatal_to_recovery(self):
        run = self.auth['runs'][0]
        path = self.write_manifest(run['run_id'], '900', status='failed', execution_phase='startup_validation')
        self.controller.auth['runs'] = [dict(run, manifest=str(path))]
        self.controller.state['submitted_jobs'] = [dict(run_id=run['run_id'], job_id='900')]
        with self.assertRaisesRegex(RuntimeError, 'Source/setup'):
            self.controller.recovery_done()

    def test_deadline_worker_admission_boundary(self):
        deadline = dt.datetime.fromisoformat(self.auth['original_deadline_utc'])
        self.assertEqual(scope.time_remaining(self.auth, deadline - dt.timedelta(seconds=8100)), 8100)
        self.assertLess(scope.time_remaining(self.auth, deadline - dt.timedelta(seconds=8099)), 8100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
