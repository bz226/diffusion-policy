"""Synthetic helper checks; no Slurm submission, scientific rollout or GPU use."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest import mock

from fixed_batch_gate import COMPARISONS, validate_gate
from trajectory_observer import TrajectoryObserver
import supervise_run
import analyze_stage2


def gate_fixture(commit='a' * 40, root=None):
    root = Path(root)
    comparison = {'passed': True, 'max_abs': 0.0, 'max_rel': 0.0, 'bitwise_equal': True}
    device = {'passed': True, 'required_bitwise': True,
              'full_update': {'passed': True, 'epochs': 5, 'minibatches': 20,
                              'actor_steps': 20, 'critic_steps': 20,
                              'same_saved_batch': True, 'same_initial_model_state': True,
                              'same_initial_optimizer_states': True, 'same_initial_rng_state': True,
                              'initial_comparisons': {key: copy.deepcopy(comparison) for key in ('batch', 'model_state', 'actor_optimizer_state', 'critic_optimizer_state', 'rng')},
                              'minibatch_comparisons': [{'minibatch': index, 'epoch': index // 4,
                                  **{key: copy.deepcopy(comparison) for key in ('loss_components', 'actor_gradients', 'critic_gradients')}} for index in range(20)],
                              'comparisons': {key: copy.deepcopy(comparison) for key in COMPARISONS}},
              'sampling': {'passed': True, 'returned_chain_states': 11, 'observed_transition_states': 20,
                           'comparisons': {key: copy.deepcopy(comparison) for key in ('chain', 'trajectories', 'all_20_transition_states')}},
              'rng_isolation': {'passed': True, 'cuda_generator_count': 1,
                                'states': {key: True for key in ('python', 'numpy', 'torch_cpu', 'torch_cuda')}}}
    gpu = copy.deepcopy(device)
    gpu['required_bitwise'] = False
    gpu['tolerance'] = {'atol': 1e-7, 'rtol': 1e-5}
    for name in ('cpu', 'gpu'):
        (root / name).mkdir(exist_ok=True)
        for filename in ('pristine_update_evidence.pt', 'edited_update_evidence.pt',
                         'sampling_evidence.pt', 'diagnostic_rng_evidence.pt'):
            (root / name / filename).write_bytes(b'synthetic fixture, no scientific tensors')
    batch = root / 'training_batch.pt'
    batch.write_bytes(b'synthetic fixture, no scientific batch')
    return {'status': 'passed', 'passed': True, 'source_commit': commit,
            'reference_commit': 'cc7234ad7ff39a8f32de3af903606723a16f0648',
            'deterministic_algorithms_enabled': False, 'deterministic_settings_unchanged': True,
            'numeric_settings_before': {'deterministic_algorithms': False},
            'numeric_settings_after': {'deterministic_algorithms': False},
            'capture': {'training_iteration': 1, 'training_environment_transitions': 80000,
                        'evaluation_environment_transitions': 80000,
                        'env_step_denoising_pairs': 200000, 'optimizer_steps_performed': 0,
                        'saved_batch_path': str(batch), 'batch_shapes': {key: [1] for key in ('obs_k', 'chains_k', 'returns_k', 'values_k', 'advantages_k', 'logprobs_k')}},
            'cpu': device, 'gpu': gpu}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='regression_fixture_', dir=str(Path(__file__).resolve().parent))
        self.root = Path(self.temp.name)
        self.path = self.root / 'gate.json'
        self.report = gate_fixture(root=self.root)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        self.path.write_text(json.dumps(self.report))
        return validate_gate(self.path, 'a' * 40)

    def test_passed_report(self):
        self.assertTrue(self.validate()['passed'])

    def test_missing_comparison_rejected(self):
        for key in COMPARISONS:
            self.report = gate_fixture(root=self.root)
            del self.report['gpu']['full_update']['comparisons'][key]
            with self.assertRaisesRegex(ValueError, key):
                self.validate()

    def test_cpu_tolerance_match_is_not_bitwise_match(self):
        c = self.report['cpu']['full_update']['comparisons']['actor_parameters']
        c.update(bitwise_equal=False, max_abs=1e-9, max_rel=1e-9)
        with self.assertRaisesRegex(ValueError, 'bitwise'):
            self.validate()

    def test_wrong_source_or_settings_rejected(self):
        for key, value in [('source_commit', 'b' * 40), ('deterministic_algorithms_enabled', True)]:
            self.report = gate_fixture(root=self.root)
            self.report[key] = value
            with self.assertRaises(ValueError):
                self.validate()

    def test_missing_sampling_rng_or_full_update_rejected(self):
        for section in ('sampling', 'rng_isolation', 'full_update'):
            self.report = gate_fixture(root=self.root)
            del self.report['cpu'][section]
            with self.assertRaises(ValueError):
                self.validate()

    def test_sampling_subcomparison_cannot_hide_under_passed_summary(self):
        self.report['gpu']['sampling']['comparisons']['chain']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'sampling chain'):
            self.validate()

    def test_each_rng_state_must_be_unchanged(self):
        for state in ('python', 'numpy', 'torch_cpu', 'torch_cuda'):
            self.report = gate_fixture(root=self.root)
            self.report['gpu']['rng_isolation']['states'][state] = False
            with self.assertRaisesRegex(ValueError, 'RNG states'):
                self.validate()

    def test_all_minibatch_losses_and_gradients_must_be_present(self):
        for key in ('loss_components', 'actor_gradients', 'critic_gradients'):
            self.report = gate_fixture(root=self.root)
            del self.report['cpu']['full_update']['minibatch_comparisons'][19][key]
            with self.assertRaisesRegex(ValueError, key):
                self.validate()

    def test_initial_inputs_must_be_bitwise_equal_even_on_gpu(self):
        self.report['gpu']['full_update']['initial_comparisons']['rng']['bitwise_equal'] = False
        with self.assertRaisesRegex(ValueError, 'initial rng'):
            self.validate()

    def test_permutation_must_be_exact_even_on_gpu(self):
        self.report['gpu']['full_update']['comparisons']['permutations']['bitwise_equal'] = False
        with self.assertRaisesRegex(ValueError, 'permutations'):
            self.validate()

    def test_preserved_evidence_files_are_required(self):
        (self.root / 'gpu/sampling_evidence.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'saved gate evidence'):
            self.validate()

    def test_old_r0_id_refused_new_id_requires_gate(self):
        base = ['test', '--expected-commit', 'a' * 40, '--timeout-seconds', '10800', '--estimate-seconds', '8100']
        with mock.patch('sys.argv', base + ['--run-id', 'R0_seed0']), mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                supervise_run.parse_args()
        with mock.patch('sys.argv', base + ['--run-id', 'R0_rerun_seed0']), mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                supervise_run.parse_args()
        with mock.patch('sys.argv', base + ['--run-id', 'R0_rerun_seed0', '--fixed-batch-gate', str(self.path)]):
            self.assertEqual(supervise_run.parse_args().condition, 'R0')

    def test_analysis_refuses_archived_r0_by_default(self):
        matrix = [{'condition': condition, 'seed': seed, 'run_id': condition + '_seed' + str(seed),
                   'manifest': 'unused.json'} for condition in ['R0'] + analyze_stage2.CONDITIONS
                  for seed in ([0] if condition == 'R0' else [0, 1, 2])]
        matrix_path = self.root / 'matrix.json'
        matrix_path.write_text(json.dumps(matrix))
        with mock.patch.object(analyze_stage2, 'load_run') as load:
            with self.assertRaisesRegex(ValueError, 'explicitly selected diagnostics baseline'):
                analyze_stage2.load_campaign(matrix_path, [])
        load.assert_not_called()


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='trajectory_fixture_', dir=str(Path(__file__).resolve().parent))
        self.parent = Path(self.temp.name)
        self.study = self.parent / 'stage2_noclip'
        self.run = self.study / 'runs/R0_rerun_seed0'
        self.run.mkdir(parents=True)
        for seed, name in enumerate(('seed0_handoff', 'seed1', 'seed2')):
            directory = self.parent / 'repro_halfcheetah/runs' / name
            directory.mkdir(parents=True)
            rows = []
            for itr in range(140):
                row = {'itr': itr, 'step': (itr - itr // 10) * 80000}
                if itr % 10 == 0:
                    row['eval_episode_reward'] = 100.0 + seed * 10 + itr
                rows.append(row)
            result = directory / 'result.pkl'
            with result.open('wb') as stream:
                pickle.dump(rows, stream)
            (directory / 'manifest.json').write_text(json.dumps({'status': 'complete', 'verified_rows': 140,
                                                               'seed': seed, 'result_path': str(result)}))
        self.observer = TrajectoryObserver(self.study, self.run)

    def tearDown(self):
        self.temp.cleanup()

    def test_huge_trajectory_difference_is_record_only(self):
        rows = [{'itr': 0, 'step': 0, 'eval_episode_reward': -1e9}]
        flags = self.observer.check(rows)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]['action'], 'record_only_continue')
        self.assertEqual(flags[0]['sample_std'], 10.0)
        self.assertEqual(flags[0]['lower'], 60.0)
        self.assertEqual(flags[0]['upper'], 160.0)
        self.assertEqual(self.observer.report['status'], 'observing')
        self.assertFalse(self.observer.report['training_or_log_token_comparisons'])
        self.assertEqual(self.observer.check(rows), [])

    def test_boundary_inclusive_and_training_values_ignored(self):
        rows = [{'itr': 0, 'step': 0, 'eval_episode_reward': 60.0},
                {'itr': 1, 'step': 80000, 'train_episode_reward': -1e100, 'pg_loss': 'anything'}]
        self.assertEqual(self.observer.check(rows), [])
        self.assertEqual(len(self.observer.report['evaluations']), 1)

    def test_terminal_partial_observation_preserves_red_flags(self):
        self.observer.check([{'itr': 0, 'step': 0, 'eval_episode_reward': 1e9}], final=True)
        self.assertEqual(self.observer.report['status'], 'partial_run_terminal')
        self.assertEqual(len(self.observer.report['red_flags']), 1)

    def test_full_trajectory_is_not_compared_with_probe_or_baseline_tokens(self):
        rows = [{'itr': itr, 'step': (itr - itr // 10) * 80000,
                 ('eval_episode_reward' if itr % 10 == 0 else 'train_episode_reward'): -1e10}
                for itr in range(140)]
        self.assertEqual(len(self.observer.check(rows, final=True)), 14)
        self.assertEqual(self.observer.report['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
