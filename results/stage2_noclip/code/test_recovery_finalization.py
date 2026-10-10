"""Recovery selection and deferred finalization checks; synthetic CPU fixtures only."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_stage2 as analysis
import finalize_campaign as finalizer


class RecoveryFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='recovery_finalizer_', dir=str(Path(__file__).resolve().parent))
        self.root = Path(self.temp.name)
        self.root_patch = mock.patch.object(finalizer, 'ROOT', self.root)
        self.root_patch.start()
        (self.root / 'provenance').mkdir()
        (self.root / 'runs').mkdir()
        self.original_path = self.root / 'provenance/run_matrix_revised.json'
        self.matrix_path = self.root / 'provenance/run_matrix_recovery.json'
        self.campaign_path = self.root / 'runs/campaign_revised.json'
        self.recovery_path = self.root / 'runs/recovery_campaign.json'
        self.original = []
        for condition in ['R0'] + analysis.CONDITIONS:
            for seed in ([0] if condition == 'R0' else range(3)):
                run_id = 'R0_rerun_seed0' if condition == 'R0' else condition + '_seed' + str(seed)
                self.original.append({'condition': condition, 'seed': seed, 'run_id': run_id,
                    'manifest': str(self.root / 'runs' / run_id / 'manifest.json'),
                    'runtime_cap_seconds': 10800, 'gpu_type': 'NVIDIA RTX A6000'})
        self.matrix = copy.deepcopy(self.original)
        replacement_runs = []
        for row in self.matrix:
            key = (row['condition'], row['seed'])
            if key in finalizer.REPLACEMENTS:
                old_id, new_id = finalizer.REPLACEMENTS[key]
                row.update(run_id=new_id, manifest=str(self.root / 'runs' / new_id / 'manifest.json'), runtime_cap_seconds=8100)
                replacement_runs.append({key: row[key] for key in ('condition', 'seed', 'run_id', 'manifest')})
                replacement_runs[-1]['replaced_run_id'] = old_id
        self.original_path.write_text(json.dumps(self.original))
        self.matrix_path.write_text(json.dumps(self.matrix))
        self.authorization = {'approved': True, 'expected_commit': finalizer.COMMIT,
            'worker_runtime_cap_seconds': 8100, 'max_recovery_runs': 2,
            'original_matrix_path': str(self.original_path), 'matrix_path': str(self.matrix_path),
            'recovery_record_path': str(self.recovery_path), 'primary_campaign_path': str(self.campaign_path),
            'runs': replacement_runs}
        self.authorization_path = self.root / 'provenance/interrupted_recovery_authorization.json'
        self.authorization_path.write_text(json.dumps(self.authorization))
        self.campaign = {'status': 'runs_terminal_analysis_pending', 'submitted_jobs': [{} for _ in range(18)]}
        self.campaign_path.write_text(json.dumps(self.campaign))
        self.recovery = {'status': 'runs_terminal_analysis_pending', 'monitor_status': 'paused',
            'commit': finalizer.COMMIT, 'worker_runtime_cap_seconds': 8100, 'retries': 0,
            'submitted_jobs': [{'run_id': row['run_id'], 'job_id': str(index + 100)}
                               for index, row in enumerate(replacement_runs)]}

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_default_finalizer_defers_without_gate_read_or_outputs(self):
        with mock.patch('sys.argv', ['finalize', '--matrix', str(self.original_path), '--campaign', str(self.campaign_path)]), \
                mock.patch.object(finalizer, 'validate_gate') as gate, \
                mock.patch.object(finalizer, 'generate') as generate, \
                mock.patch.object(finalizer, 'load_campaign') as load:
            finalizer.main()
        self.assertEqual(json.loads(self.campaign_path.read_text())['status'], 'runs_terminal_recovery_pending')
        gate.assert_not_called()
        generate.assert_not_called()
        load.assert_not_called()
        self.assertFalse((self.root / 'runs/final_analysis').exists())
        self.assertFalse((self.root / 'experiment.md').exists())

    def test_default_finalizer_defers_even_when_recovery_record_exists(self):
        self.recovery_path.write_text(json.dumps(self.recovery))
        self.assertTrue(finalizer.defer_for_recovery(self.campaign_path, self.campaign,
                                                    self.authorization, None))
        self.assertEqual(json.loads(self.recovery_path.read_text()), self.recovery)

    def test_recovery_invocation_does_not_defer(self):
        self.assertFalse(finalizer.defer_for_recovery(self.campaign_path, self.campaign,
                                                     self.authorization, self.recovery_path))

    def test_exact_matrix_accepted_without_mutating_original(self):
        before = self.original_path.read_text()
        self.assertEqual(finalizer.validate_recovery_matrix(self.matrix_path, self.original_path), self.matrix)
        self.assertEqual(self.original_path.read_text(), before)

    def test_wrong_seed_extra_run_or_third_replacement_rejected(self):
        variants = []
        wrong_seed = copy.deepcopy(self.matrix)
        wrong_seed[0]['seed'] = 1
        variants.append(wrong_seed)
        variants.append(self.matrix + [self.matrix[0]])
        third = copy.deepcopy(self.matrix)
        third[4]['run_id'] = 'NC2_recovery_seed0'
        variants.append(third)
        for matrix in variants:
            with self.subTest(matrix=matrix):
                self.matrix_path.write_text(json.dumps(matrix))
                with self.assertRaises(ValueError):
                    finalizer.validate_recovery_matrix(self.matrix_path, self.original_path)

    def test_wrong_cap_and_manifest_rejected(self):
        for key, value in [('runtime_cap_seconds', 10800), ('manifest', 'other.json')]:
            matrix = copy.deepcopy(self.matrix)
            matrix[0][key] = value
            self.matrix_path.write_text(json.dumps(matrix))
            with self.assertRaises(ValueError):
                finalizer.validate_recovery_matrix(self.matrix_path, self.original_path)

    def test_authorization_exact_scope(self):
        self.assertEqual(finalizer.recovery_authorization(), self.authorization)
        for key, value in [('approved', False), ('expected_commit', 'wrong'), ('max_recovery_runs', 3),
                           ('worker_runtime_cap_seconds', 10800)]:
            bad = dict(self.authorization, **{key: value})
            self.authorization_path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                finalizer.recovery_authorization()

    def test_pending_recovery_rejected_before_loading_any_measurements(self):
        self.recovery['status'] = 'running_replacements'
        self.recovery_path.write_text(json.dumps(self.recovery))
        with mock.patch.object(finalizer, 'load_campaign') as load:
            with self.assertRaisesRegex(ValueError, 'must be terminal'):
                finalizer.recovery_selection(self.matrix_path, self.recovery_path, self.authorization)
        load.assert_not_called()

    def test_extra_attempt_or_active_monitor_rejected(self):
        for key, value in [('monitor_status', 'active'), ('retries', 1),
                           ('submitted_jobs', self.recovery['submitted_jobs'] + [{'run_id': 'extra', 'job_id': '103'}])]:
            bad = dict(self.recovery, **{key: value})
            self.recovery_path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                finalizer.recovery_selection(self.matrix_path, self.recovery_path, self.authorization)

    def test_historical_attempts_retained_without_inclusion_in_selected_matrix(self):
        self.recovery_path.write_text(json.dumps(self.recovery))
        old_runs = [{'condition': row['condition'], 'seed': row['seed'], 'run_id': row['run_id'],
                     'manifest_path': row['manifest'], 'result_path': row['manifest'] + '.pkl',
                     'manifest': {'logdir': 'preserved', 'checkpoints_found': ['state_105.pt']}}
                    for row in self.original]
        with mock.patch.object(finalizer, 'load_campaign', return_value=old_runs), \
                mock.patch.object(finalizer, 'verify_paused'), \
                mock.patch.object(finalizer, 'summarize_run', side_effect=lambda run: {'run_id': run['run_id'], 'status': 'stopped'}), \
                mock.patch.object(finalizer, 'resources', return_value={'runs': [{'run_id': row['run_id']} for row in self.original]}):
            _, selection = finalizer.recovery_selection(self.matrix_path, self.recovery_path, self.authorization)
        self.assertEqual({row['run_id'] for row in selection['excluded_interrupted_runs']},
                         {'R0_rerun_seed0', 'NC1_seed1'})
        self.assertEqual({row['run_id'] for row in selection['excluded_interrupted_resources']},
                         {'R0_rerun_seed0', 'NC1_seed1'})
        self.assertEqual(len(self.matrix), 19)
        self.assertNotIn('R0_rerun_seed0', {row['run_id'] for row in self.matrix})
        self.assertNotIn('NC1_seed1', {row['run_id'] for row in self.matrix})

    def test_report_uses_selected_r0_redflags_and_preserves_history(self):
        sources, summaries = [], []
        for row in self.matrix:
            path = Path(row['manifest'])
            path.parent.mkdir(exist_ok=True)
            manifest = {'trajectory_red_flags': [1, 2]}
            if row['run_id'] == 'NC1_seed2':
                manifest['hardware_exception'] = {'approved': True}
            path.write_text(json.dumps(manifest))
            sources.append({'condition': row['condition'], 'manifest': row['manifest']})
            summaries.append({'condition': row['condition'], 'status': 'complete', 'collapse': False,
                              'eval_return_itr130': 4700.0})
        for seed in range(3):
            summaries.append({'condition': 'baseline', 'status': 'complete', 'collapse': False,
                              'eval_return_itr130': 4700.0})
        (self.root / 'provenance/parallel_handoff_failure.json').write_text('{}')
        old_path = self.root / 'runs/R0_rerun_seed0/manifest.json'
        old_path.parent.mkdir(exist_ok=True)
        old_path.write_text(json.dumps({'trajectory_red_flags': [1] * 99}))
        gate_path = self.root / 'gate.json'
        gate_path.write_text(json.dumps({device: {stage: {'comparisons': {'test': {'max_abs': 0, 'max_rel': 0}}}
                                                 for stage in ['full_update', 'sampling']}
                                        for device in ['cpu', 'gpu']}))
        record = {'summaries': summaries, 'sources': sources, 'recovery_selection': {'approved': True}}
        text = finalizer.current_report(record, {'path': str(gate_path)})
        self.assertIn('R0 produced 2 non-stopping', text)
        self.assertNotIn('R0 produced 99', text)
        self.assertIn('runs/R0_recovery_seed0/manifest.json', text)
        self.assertIn('runs/R0_rerun_seed0/manifest.json', text)
        self.assertIn('138/140', text)
        self.assertIn('regardless of outcome', text)
        self.assertIn('H200 NVL', text)
        self.assertLessEqual(len(text.split()), 600)

    def test_generated_candidates_expose_selected_and_excluded_attempts(self):
        rows = [{'itr': 0, 'step': 0, 'eval_episode_reward': 4200.0, 'time': 1.0},
                {'itr': 1, 'step': 80000, 'train_episode_reward': 4000.0, 'time': 1.0,
                 'kl_true_per_action': 0.1, 'logratio_p99': 0.2, 'clamp_hit_frac': 0.01,
                 'approx_kl': 0.02, 'clipfrac': 0.0}]
        entries = [{'condition': 'baseline', 'seed': seed, 'run_id': 'baseline_seed%d' % seed,
                    'manifest': str(self.root / ('baseline_seed%d.json' % seed))} for seed in range(3)] + self.matrix
        runs = [{'condition': row['condition'], 'seed': row['seed'], 'run_id': row['run_id'],
                 'manifest_path': row['manifest'], 'manifest': {}, 'status': 'stopped',
                 'result_path': row['manifest'] + '.pkl', 'effective_result_path': row['manifest'] + '.pkl',
                 'result_source': 'native', 'fallback_used': False, 'native_read_error': None,
                 'fallback_read_error': None, 'rows': copy.deepcopy(rows)} for row in entries]
        selection = {'excluded_interrupted_runs': [{'run_id': 'R0_rerun_seed0'}, {'run_id': 'NC1_seed1'}]}
        output = self.root / 'generated/candidates'
        with mock.patch.object(analysis, 'load_campaign', return_value=runs), \
                mock.patch.dict(os.environ, {'MPLCONFIGDIR': str(self.root / 'matplotlib_cache')}):
            record_path = analysis.generate(self.matrix_path, [], output, 'R0_recovery_seed0', selection)
        record = json.loads(record_path.read_text())
        self.assertEqual(record['selected_r0_run_id'], 'R0_recovery_seed0')
        self.assertEqual(record['recovery_selection'], selection)
        selected_ids = {row['run_id'] for row in record['sources']}
        self.assertIn('R0_recovery_seed0', selected_ids)
        self.assertIn('NC1_recovery_seed1', selected_ids)
        self.assertNotIn('R0_rerun_seed0', selected_ids)
        self.assertNotIn('NC1_seed1', selected_ids)
        self.assertEqual(sum(row['condition'] != 'baseline' for row in record['summaries']), 19)
        captions = (output / 'captions.txt').read_text()
        self.assertEqual(captions.count('irrespective of their outcomes'), 4)
        self.assertEqual(captions.count('excluded provenance'), 4)
        self.assertIn('R0_recovery_seed0', (output / 'condition_seed_results.csv').read_text())


if __name__ == '__main__':
    unittest.main()
