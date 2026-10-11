"""Mechanical admission/provenance tests; no environments, GPU jobs, or sampling."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('campaign.py')
SPEC = importlib.util.spec_from_file_location('gradmean_runtime', str(PATH))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
import sys
sys.modules['campaign'] = c
B_SPEC = importlib.util.spec_from_file_location('gradmean_bootstrap', str(PATH.with_name('bootstrap.py')))
b = importlib.util.module_from_spec(B_SPEC)
B_SPEC.loader.exec_module(b)


class RuntimeTests(unittest.TestCase):
    def test_disjoint_all_environment_seed_ranges(self):
        seen = set()
        for point in c.POINTS:
            seeds = [c.seed_for(point)] + [c.seed_for(point, j, retry) for j in range(82) for retry in (False, True)]
            for seed in seeds:
                workers = set(range(seed, seed + 40))
                self.assertFalse(workers & seen)
                seen.update(workers)

    def test_round_group_counts_and_priority(self):
        self.assertEqual([target // 50 for target in c.TARGETS], [10, 28, 46, 64, 82])
        self.assertEqual(c.PRIORITY, ('P0', 'D2', 'N2', 'D1', 'N1'))

    def test_no_undefined_or_negative_rho_closure(self):
        for rho in (None, -.2, 0, .679999):
            self.assertEqual(c.close_status(rho, 500), 'open')
            self.assertEqual(c.close_status(rho, 4100), 'closed_cap')
        self.assertEqual(c.close_status(.68, 500), 'closed_rho')
        with self.assertRaises(ValueError):
            c.close_status(.9, 50)

    def test_ledger_charges_terminal_actual_and_full_outstanding_caps(self):
        value = c.ledger([dict(scheduler_state='COMPLETED', allocated_gpu_seconds=900, cap_seconds=3600),
                          dict(scheduler_state='FAILED', allocated_gpu_seconds=300, cap_seconds=3600),
                          dict(scheduler_state='RUNNING', allocated_gpu_seconds=200, cap_seconds=3600),
                          dict(scheduler_state='PENDING', allocated_gpu_seconds=0, cap_seconds=3600)])
        self.assertEqual(value['actual_gpu_seconds'], 1200)
        self.assertEqual(value['outstanding_reserved_gpu_seconds'], 7200)
        self.assertEqual(value['actual_plus_reserved_gpu_seconds'], 8400)

    def test_slot_limits_count_other_jobs_and_gpus(self):
        jobs = [dict(job_id=str(i), gpus=1) for i in range(3)] + [dict(job_id='controller', gpus=0)]
        self.assertEqual(c.queue_slots([], jobs), 8)
        own = [dict(job_id=str(i), scheduler_state='RUNNING') for i in range(8)]
        self.assertEqual(c.queue_slots(own, [dict(job_id=str(i), gpus=1) for i in range(8)]), 1)
        self.assertEqual(c.queue_slots([], [dict(job_id=str(i), gpus=0) for i in range(12)]), 0)

    def test_only_one_named_replacement_slot(self):
        self.assertEqual(c.parse_run('N2_j81_retry1'), ('N2', 81, True))
        for name in ('N2_j82', 'P0_j00_retry2', 'prepare_P0_retry1', 'P3_j00'):
            with self.assertRaises(ValueError):
                c.parse_run(name)

    def test_unknown_accounting_keeps_full_reservation(self):
        result = c.ledger([dict(scheduler_state=None, cap_seconds=3600, allocated_gpu_seconds=50)])
        self.assertEqual(result['actual_plus_reserved_gpu_seconds'], 3600)

    def test_specs_preserve_science_and_score_first_group_each_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'mechanical fixture')
            registry = {p: dict(checkpoint=str(checkpoint), warmup_batches=20 if p == 'P0' else 5) for p in c.POINTS}
            c.atomic(root / 'provenance/checkpoints.json', registry)
            for point in c.POINTS:
                c.atomic(root / 'runs' / ('prepare_' + point) / 'manifest.json', dict(status='complete'))
                c.atomic(root / 'prepare' / point / 'grad_mean.json', dict(status='completed', frozen_path=str(root / (point + '.pt')), frozen_sha256='fixture'))
            with patch.object(c, 'ROOT', root):
                for point in c.POINTS:
                    for group in (0, 1):
                        spec = c.run_spec('%s_j%02d' % (point, group), 'reference')
                        self.assertEqual(spec['expected_settings']['gm_score_sample'], 200 if group == 0 else 0)
                        self.assertTrue(all(key in spec['overrides'] for key in c.COMMON))
                        self.assertFalse(any('actor_single_step' in key or 'target_kl' in key or 'clip_ploss' in key for key in spec['overrides']))
                        self.assertEqual(spec['cap_seconds'], 3600)
                        self.assertEqual((spec['cpus'], spec['memory_gib'], spec['gpu_count']), (40, 64, 1))
                retry = c.run_spec('D1_j00_retry1', 'reference')
                self.assertEqual(retry['expected_settings']['gm_score_sample'], 200)

    def test_offline_gate_prevents_production_before_any_scheduler_call(self):
        campaign = c.Campaign.__new__(c.Campaign)
        campaign.args = argparse.Namespace(gate=Path('offline'), commit='reference', action='submit')
        campaign.state = dict(submitted=[])
        with patch.object(c, 'validate_gate', return_value={'stage': 'offline'}), patch.object(c, 'scheduler_jobs', side_effect=AssertionError('must not query or launch')):
            with self.assertRaises(ValueError):
                campaign.submit('prepare_D1')

    def test_duplicate_admission_rejected_before_external_actions(self):
        campaign = c.Campaign.__new__(c.Campaign)
        campaign.state = dict(submitted=[dict(run_id='P0_j00')])
        with self.assertRaises(ValueError):
            campaign.submit('P0_j00')

    def test_budget_guard_precedes_spec_or_sbatch(self):
        campaign = c.Campaign.__new__(c.Campaign)
        campaign.args = argparse.Namespace(gate=Path('full'), commit='reference', action='run')
        campaign.state = dict(submitted=[], external=[dict(scheduler_state='COMPLETED', allocated_gpu_seconds=c.BUDGET - 100, cap_seconds=c.BUDGET)])
        campaign.dispatch_slots = 9
        campaign.record = lambda: None
        with patch.object(c, 'validate_gate', return_value={'stage': 'full'}), patch.object(c, 'run_spec', side_effect=AssertionError('budget must reject first')):
            self.assertFalse(campaign.submit('P0_j00'))

    def test_successful_retry_is_one_logical_group_and_failed_original_preserved(self):
        campaign = c.Campaign.__new__(c.Campaign)
        campaign.state = dict(submitted=[dict(run_id='P0_j00', terminal_recorded=True, run_status='failed'),
                                         dict(run_id='P0_j00_retry1', terminal_recorded=True, run_status='complete')])
        self.assertEqual(campaign.successful_group('P0', 0), 'P0_j00_retry1')
        self.assertEqual(len(campaign.state['submitted']), 2)

    def test_source_proof_allows_docs_descendant_and_git_mode_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root, source = parent / 'output', parent / 'source'
            snapshot = root / 'gate/source'
            source.mkdir()
            snapshot.mkdir(parents=True)
            for path in (source / 'script.sh', snapshot / 'script.sh'):
                path.write_bytes(b'echo fixture\n')
                path.chmod(0o770)
            gate = root / 'gate.json'
            c.atomic(gate, dict(passed=True, stage='offline', parent_commit='reference', source_snapshot=str(snapshot), slurm_job_id='123'))
            def fake_command(argv, check=True):
                if argv[0] == 'sacct':
                    output = '123|COMPLETED|0:0|\n'
                elif 'branch' in argv:
                    output = 'stage3-gradmean\n'
                elif 'rev-parse' in argv:
                    output = 'later-docs-commit\n'
                elif 'ls-tree' in argv:
                    output = '100755 blob old-blob\tsource/script.sh\0'
                else:
                    output = ''
                return dict(stdout=output, returncode=0)
            payload = b'echo fixture\n'
            binary = b'old-blob blob ' + str(len(payload)).encode() + b'\n' + payload + b'\n'
            with patch.object(c, 'PARENT', parent), patch.object(c, 'ROOT', root), patch.object(c, 'SOURCE', source), \
                    patch.object(c, 'command', side_effect=fake_command), \
                    patch.object(c.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=binary)):
                self.assertEqual(c.validate_gate(gate, 'reference')['current_parent_commit'], 'later-docs-commit')
                (source / 'script.sh').write_bytes(b'changed\n')
                with self.assertRaises(ValueError):
                    c.validate_gate(gate, 'reference')

    def test_full_gate_is_required_even_when_offline_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'offline.json'
            c.atomic(gate, dict(passed=True, stage='offline', parent_commit='reference'))
            with self.assertRaises(ValueError):
                c.validate_gate(gate, 'reference', full=True)

    def test_live_gate_failure_prevents_additional_admission(self):
        campaign = c.Campaign.__new__(c.Campaign)
        campaign.args = argparse.Namespace(gate=Path('offline'), commit='reference', action='submit')
        campaign.state = dict(submitted=[dict(run_id='P0_j00', run_status='failed')])
        with patch.object(c, 'validate_gate', return_value={'stage': 'offline'}):
            with self.assertRaises(ValueError):
                campaign.submit('P0_j01')

    def test_global_analysis_barrier_does_not_add_unreached_rounds(self):
        campaign = c.Campaign.__new__(c.Campaign)
        campaign.state = dict(points={point: dict(status='closed_rho', rounds=[{'target_batches': 500}]) for point in c.POINTS})
        labels = []
        def schedule(label, partial):
            labels.append(label)
            return dict(terminal_recorded=False)
        campaign.schedule_analysis = schedule
        self.assertTrue(campaign.round_analysis_barrier())
        self.assertEqual(labels, ['round_0500'])

    def test_formal_nonzero_exit_blocks_live_transition(self):
        bootstrap = b.Bootstrap.__new__(b.Bootstrap)
        bootstrap.args = argparse.Namespace(offline=Path('gate'), formal_job='123')
        bootstrap.check_cap = lambda: None
        bootstrap.monitor = lambda *args, **kwargs: None
        bootstrap.formal_status = lambda: dict(state='COMPLETED', exit_code='1:0')
        with patch.object(b, 'read', return_value={'passed': True, 'status': 'passed'}):
            with self.assertRaises(RuntimeError):
                bootstrap.await_offline()

    def test_zero_exit_without_gate_pass_does_not_admit_live_work(self):
        bootstrap = b.Bootstrap.__new__(b.Bootstrap)
        bootstrap.args = argparse.Namespace(offline=Path('gate'), formal_job='123')
        bootstrap.check_cap = lambda: None
        bootstrap.monitor = lambda *args, **kwargs: None
        bootstrap.formal_status = lambda: dict(state='COMPLETED', exit_code='0:0')
        with patch.object(b, 'read', return_value={'passed': False, 'stage': 'offline'}):
            with self.assertRaises(RuntimeError):
                bootstrap.await_offline()

    def test_common_gate_validation_rejects_running_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'gate.json'
            c.atomic(gate, dict(passed=True, stage='offline', parent_commit='reference', slurm_job_id='123'))
            with patch.object(c, 'command', return_value={'stdout': '123|RUNNING|0:0|\n'}):
                with self.assertRaises(ValueError):
                    c.validate_gate(gate, 'reference')

    def test_failure_cleanup_handles_already_exited_leader(self):
        child = SimpleNamespace(pid=123, poll=lambda: 1)
        with patch.object(c, 'process_group_members', side_effect=[[456], [], [], []]), patch.object(c.os, 'killpg') as kill:
            self.assertEqual(c.terminate_owned_group(child), [])
            kill.assert_called_once_with(123, c.signal.SIGTERM)


if __name__ == '__main__':
    unittest.main(verbosity=2)
