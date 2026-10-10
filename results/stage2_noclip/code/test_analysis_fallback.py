"""Failed-run result recovery tests using only synthetic local fixtures."""
import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest

import analyze_stage2 as analysis


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='analysis_fallback_fixture_', dir=str(Path(__file__).resolve().parent))
        self.directory = Path(self.temp.name)
        self.native = self.directory / 'native.pkl'
        self.fallback = self.directory / 'observed.pkl'
        self.manifest_path = self.directory / 'manifest.json'
        self.rows = [{'itr': 0, 'step': 0, 'time': 13.0, 'eval_episode_reward': 4200.0},
                     {'itr': 1, 'step': 80000, 'time': 57.0, 'train_episode_reward': 3800.0,
                      'kl_true_per_action': 0.01, 'logratio_p99': 0.2, 'clamp_hit_frac': 0.03,
                      'approx_kl': 0.001, 'clipfrac': 0.01}]
        self.native.write_bytes(pickle.dumps(self.rows))
        self.fallback.write_bytes(pickle.dumps(self.rows))
        self.manifest = {'status': 'failed', 'run_id': 'NC1_seed0', 'seed': 0,
                         'result_path': str(self.native), 'last_valid_result_path': str(self.fallback),
                         'last_valid_result_rows': 2}
        self.save_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def load(self):
        return analysis.load_run(self.manifest_path, 'NC1', 0, 'NC1_seed0')

    def test_corrupt_failed_native_uses_only_declared_prefix(self):
        self.native.write_bytes(b'\x80')
        run = self.load()
        self.assertTrue(run['fallback_used'])
        self.assertEqual(run['effective_result_path'], str(self.fallback))
        self.assertEqual(run['rows'], self.rows)
        self.assertIsNotNone(run['native_read_error'])
        summary = analysis.summarize_run(run)
        self.assertIsNone(summary['eval_return_itr70'])
        self.assertIsNone(summary['eval_return_itr130'])
        self.assertFalse(summary['collapse'])

    def test_failed_corrupt_native_without_fallback_is_explicitly_unavailable(self):
        self.native.write_bytes(b'not pickle')
        del self.manifest['last_valid_result_path']
        self.save_manifest()
        run = self.load()
        self.assertEqual(run['rows'], [])
        self.assertEqual(run['result_source'], 'unavailable')
        self.assertIsNotNone(run['native_read_error'])
        self.assertIsNone(analysis.summarize_run(run)['collapse'])

    def test_completed_native_corruption_always_raises(self):
        self.native.write_bytes(b'\x80')
        self.manifest['status'] = 'complete'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Completed native results cannot be read'):
            self.load()

    def test_readable_nonfinite_native_is_never_replaced(self):
        self.rows[1]['kl_true_per_action'] = math.inf
        self.native.write_bytes(pickle.dumps(self.rows))
        run = self.load()
        self.assertFalse(run['fallback_used'])
        self.assertEqual(run['result_source'], 'native')
        self.assertEqual(run['rows'][1]['kl_true_per_action'], math.inf)
        self.assertTrue(analysis.summarize_run(run)['collapse'])

    def test_two_corrupt_sources_do_not_fabricate_rows(self):
        self.native.write_bytes(b'\x80')
        self.fallback.write_bytes(b'\x80')
        run = self.load()
        self.assertEqual(run['rows'], [])
        self.assertIsNotNone(run['native_read_error'])
        self.assertIsNotNone(run['fallback_read_error'])
        self.assertIsNone(run['effective_result_path'])

    def test_missing_native_may_use_declared_snapshot(self):
        self.manifest['result_path'] = str(self.directory / 'missing.pkl')
        self.save_manifest()
        run = self.load()
        self.assertEqual(run['result_source'], 'declared_validated_prefix')
        self.assertIn('FileNotFoundError', run['native_read_error'])

    def test_undeclared_snapshot_is_not_discovered_or_used(self):
        self.manifest['result_path'] = str(self.directory / 'missing.pkl')
        del self.manifest['last_valid_result_path']
        self.save_manifest()
        run = self.load()
        self.assertEqual(run['rows'], [])
        self.assertTrue(self.fallback.exists())

    def test_declared_snapshot_count_must_match(self):
        self.native.write_bytes(b'\x80')
        self.manifest['last_valid_result_rows'] = 3
        self.save_manifest()
        run = self.load()
        self.assertEqual(run['rows'], [])
        self.assertIn('row count', run['fallback_read_error'])

    def test_minimum_preserves_infinities_and_rejects_nan(self):
        for value, expected, expected_status in [(math.inf, 4200.0, 'defined'),
                                                 (-math.inf, -math.inf, 'defined'),
                                                 (math.nan, None, 'undefined_nan_present')]:
            with self.subTest(value=value):
                run = self.load()
                run['rows'].append({'itr': 10, 'step': 720000, 'eval_episode_reward': value})
                summary = analysis.summarize_run(run)
                self.assertEqual(summary['minimum_eval_return'], expected)
                self.assertEqual(summary['minimum_eval_status'], expected_status)
                self.assertEqual(summary['saved_nonfinite_count'], 1)
                self.assertTrue(summary['collapse'])
        run = self.load()
        run['rows'] = [{'itr': 0, 'step': 0, 'eval_episode_reward': math.inf}]
        summary = analysis.summarize_run(run)
        self.assertEqual(summary['minimum_eval_return'], math.inf)
        self.assertEqual(summary['minimum_eval_status'], 'defined')
        run['rows'] = []
        summary = analysis.summarize_run(run)
        self.assertIsNone(summary['minimum_eval_return'])
        self.assertEqual(summary['minimum_eval_status'], 'no_evaluations')


if __name__ == '__main__':
    unittest.main()
