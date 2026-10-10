"""Controller tests with mocked Slurm; never submit/cancel real jobs."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import campaign
from test_revised_regression import gate_fixture


def response(stdout='', returncode=0, stderr=''):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='campaign_fixture_', dir=str(Path(__file__).resolve().parent))
        self.root = Path(self.temp.name)
        self.patch = mock.patch.object(campaign, 'ROOT', self.root)
        self.patch.start()
        for name in ('provenance', 'runs', 'setup'):
            (self.root / name).mkdir()
        self.matrix = []
        for condition in ('R0',) + campaign.GROUPS:
            for seed in ([0] if condition == 'R0' else [0, 1, 2]):
                run_id = ('R0_rerun' if condition == 'R0' else condition) + '_seed' + str(seed)
                self.matrix.append({'condition': condition, 'seed': seed, 'run_id': run_id,
                                    'manifest': str(self.root / 'runs' / run_id / 'manifest.json'),
                                    'runtime_cap_seconds': 10800, 'gpu_type': 'NVIDIA RTX A6000'})
        (self.root / 'provenance/run_matrix_revised.json').write_text(json.dumps(self.matrix))
        self.gate = self.root / 'gate.json'
        self.gate.write_text(json.dumps(gate_fixture('a' * 40, self.root)))
        with mock.patch.object(campaign.signal, 'signal'):
            self.controller = campaign.Campaign('a' * 40, '100', self.gate)
        self.run = self.matrix[1]

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def write_manifest(self, run, data):
        path = Path(run['manifest'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_sacct_terminal_without_squeue_record(self):
        with mock.patch.object(campaign.subprocess, 'run', side_effect=[response(), response('101|CANCELLED by 7|0:15|Unknown|2026-10-09T01:00:00\n')]):
            observed = campaign.scheduler_state('101')
        self.assertEqual(observed['state'], 'CANCELLED')
        self.assertEqual(observed['exit_code'], '0:15')
        self.assertEqual(observed['source'], 'sacct')

    def test_active_queue_state_does_not_consult_stale_accounting(self):
        with mock.patch.object(campaign.subprocess, 'run', return_value=response('101|PENDING\n')) as called:
            self.assertEqual(campaign.scheduler_state('101')['state'], 'PENDING')
        self.assertEqual(called.call_count, 1)

    def test_queued_failure_becomes_explicit_external_failure(self):
        observed = {'state': 'CANCELLED', 'raw_state': 'CANCELLED by 7', 'exit_code': '0:15'}
        with mock.patch.object(campaign, 'scheduler_state', return_value=observed):
            manifest = self.controller.observe_run(self.run, '101')
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['external_slurm_observation'], observed)
        self.assertIsNone(manifest['failing_iteration'])
        self.assertFalse(manifest['nonfinite_failure'])
        self.assertFalse(Path(manifest['result_path']).exists())
        self.assertNotIn('subprocess_start_utc', manifest)
        self.assertTrue((Path(self.run['manifest']).parent / 'external_failure.json').exists())

    def test_killed_worker_preserves_previous_manifest_and_snapshot_pointer(self):
        original = {'status': 'running', 'last_valid_result_path': '/observed/itr_12.pkl', 'last_diagnostics': {'kl_true_per_action': 0.01}}
        self.write_manifest(self.run, original)
        observed = {'state': 'TIMEOUT', 'raw_state': 'TIMEOUT', 'exit_code': '0:15'}
        with mock.patch.object(campaign, 'scheduler_state', return_value=observed):
            result = self.controller.observe_run(self.run, '101')
            repeated = self.controller.observe_run(self.run, '101')
        saved = json.loads((Path(self.run['manifest']).parent / 'manifest_before_external_failure.json').read_text())
        self.assertEqual(saved, original)
        self.assertEqual(result['last_valid_result_path'], original['last_valid_result_path'])
        self.assertEqual(result, repeated)

    def test_completed_verified_manifest_is_not_rewritten(self):
        original = {'status': 'complete', 'verification_complete': True}
        self.write_manifest(self.run, original)
        with mock.patch.object(campaign, 'scheduler_state') as scheduler:
            self.assertEqual(self.controller.observe_run(self.run, '101'), original)
        scheduler.assert_not_called()

    def test_pending_job_is_not_falsely_failed(self):
        with mock.patch.object(campaign, 'scheduler_state', return_value={'state': 'PENDING'}):
            self.assertIsNone(self.controller.observe_run(self.run, '101'))
        self.assertFalse(Path(self.run['manifest']).exists())

    def test_unknown_scheduler_state_has_bounded_wait_without_fake_outcome(self):
        self.controller.scheduler_unknown_since['101'] = 0
        with mock.patch.object(campaign, 'scheduler_state', return_value={'state': None}), mock.patch.object(campaign.time, 'monotonic', return_value=301):
            with self.assertRaisesRegex(RuntimeError, 'unobservable'):
                self.controller.observe_run(self.run, '101')
        self.assertFalse(Path(self.run['manifest']).exists())

    def test_r0_external_failure_does_not_stop_remaining_runs(self):
        with mock.patch.object(campaign, 'scheduler_state', return_value={'state': 'FAILED', 'raw_state': 'FAILED', 'exit_code': '1:0'}):
            manifest, observation = self.controller.check_r0()
        self.assertEqual(manifest['status'], 'failed')
        self.assertIsNone(observation)

    def test_r0_complete_needs_no_trajectory_gate(self):
        self.write_manifest(self.matrix[0], {'status': 'complete'})
        manifest, observation = self.controller.check_r0()
        self.assertEqual(manifest['status'], 'complete')
        self.assertIsNone(observation)

    def test_submit_one_gpu_without_invalid_node_list_and_no_retry(self):
        with mock.patch.object(campaign.subprocess, 'run', return_value=response('101\n')) as called:
            self.controller.submit(self.run)
            command = called.call_args[0][0]
            self.assertIn('--nodes=1', command)
            self.assertIn('--gres=gpu:a6000:1', command)
            self.assertFalse(any(item.startswith('--nodelist') for item in command))
            with self.assertRaisesRegex(RuntimeError, 'repeated submission'):
                self.controller.submit(self.run)
        self.assertEqual(called.call_count, 1)

    def test_ambiguous_submission_is_recorded_and_never_retried(self):
        with mock.patch.object(campaign.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 60)) as called:
            with self.assertRaisesRegex(RuntimeError, 'outcome is unknown'):
                self.controller.submit(self.run)
            with self.assertRaisesRegex(RuntimeError, 'repeated submission'):
                self.controller.submit(self.run)
        self.assertEqual(called.call_count, 1)
        self.assertIn('submission_error', self.controller.state['submitted_jobs'][0])
        self.assertNotIn('job_id', self.controller.state['submitted_jobs'][0])

    def test_matrix_cap_and_condition_guard(self):
        self.matrix[-1]['runtime_cap_seconds'] = 10801
        (self.root / 'provenance/run_matrix_revised.json').write_text(json.dumps(self.matrix))
        with self.assertRaisesRegex(ValueError, 'resource cap'):
            campaign.Campaign('a' * 40, '100', self.gate)

    def test_failed_ablations_do_not_prevent_remaining_groups(self):
        submitted = []
        def submit(run):
            submitted.append(run['run_id'])
            self.controller.state['submitted_jobs'].append({'run_id': run['run_id'], 'job_id': str(200 + len(submitted))})
            self.write_manifest(run, {'status': 'failed' if run['seed'] == 0 else 'complete'})
        with mock.patch.object(self.controller, 'check_r0', return_value=({'status': 'failed', 'failure_reason': 'NaN'}, {'red_flags': [{'eval_return': -1e6}]})), mock.patch.object(self.controller, 'submit', side_effect=submit), mock.patch.object(campaign.subprocess, 'run') as external:
            self.controller.run()
        self.assertEqual(len(submitted), 18)
        self.assertEqual(len(set(submitted)), 18)
        self.assertEqual(self.controller.state['status'], 'runs_terminal_analysis_pending')
        external.assert_not_called()

    def test_interruption_cancels_owned_work_and_stops_new_groups(self):
        def gate():
            if len(self.controller.state['submitted_jobs']) >= 3:
                raise RuntimeError('Campaign controller interrupted')
            return {'status': 'running'}, None
        def submit(run):
            self.controller.state['submitted_jobs'].append({'run_id': run['run_id'], 'job_id': str(200 + len(self.controller.state['submitted_jobs']))})
        with mock.patch.object(self.controller, 'check_r0', side_effect=gate), mock.patch.object(self.controller, 'submit', side_effect=submit), mock.patch.object(self.controller, 'cancel_owned') as cancel:
            with self.assertRaisesRegex(RuntimeError, 'controller interrupted'):
                self.controller.run()
        self.assertEqual(len(self.controller.state['submitted_jobs']), 3)
        self.assertEqual(self.controller.state['status'], 'stopped')
        cancel.assert_called_once_with()

    def test_old_failed_replay_record_is_never_read(self):
        old = self.root / 'runs/R0_seed0'
        old.mkdir()
        (old / 'replay_status.json').write_text(json.dumps({'r0_passed': False}))
        self.write_manifest(self.matrix[0], {'status': 'running'})
        with mock.patch.object(campaign, 'scheduler_state', return_value={'state': 'RUNNING'}):
            manifest, observation = self.controller.check_r0()
        self.assertEqual(manifest['status'], 'running')
        self.assertIsNone(observation)

    def test_fixed_batch_gate_is_required_before_submitting(self):
        report = json.loads(self.gate.read_text())
        report['cpu']['full_update']['comparisons']['actor_gradients']['passed'] = False
        self.gate.write_text(json.dumps(report))
        with mock.patch.object(campaign.subprocess, 'run') as external:
            with self.assertRaisesRegex(ValueError, 'actor_gradients'):
                self.controller.submit(self.run)
        external.assert_not_called()
        self.assertEqual(self.controller.state['submitted_jobs'], [])

    def test_true_source_failure_stops_dispatch(self):
        self.write_manifest(self.matrix[0], {'status': 'failed', 'execution_phase': 'startup_validation',
                                          'failure_reason': 'Unexpected source commit'})
        with self.assertRaisesRegex(RuntimeError, 'Source/setup validation'):
            self.controller.check_r0()

    def test_experiment_updates_only_status_marker_and_pauses_monitor(self):
        report = self.root / 'experiment.md'
        report.write_text('Evidence before\n<!-- campaign-status-start -->\nOld\n<!-- campaign-status-end -->\nMethods after\n')
        self.controller.state.update(status='runs_terminal_analysis_pending', monitor_status='paused')
        self.controller.record(terminal=True)
        text = report.read_text()
        self.assertTrue(text.startswith('Evidence before\n<!-- campaign-status-start -->\n'))
        self.assertTrue(text.endswith('<!-- campaign-status-end -->\nMethods after\n'))
        self.assertIn('runs_terminal_analysis_pending', text)
        event = json.loads((self.root / 'runs/campaign_revised_monitor.jsonl').read_text().splitlines()[-1])
        self.assertEqual(event['monitor_status'], 'paused')


if __name__ == '__main__':
    unittest.main()
