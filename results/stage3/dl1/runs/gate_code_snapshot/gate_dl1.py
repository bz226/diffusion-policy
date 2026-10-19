#!/usr/bin/env python
"""DL1 G1--G6 gate controller. Executes workers, never submits Slurm jobs."""
import argparse
import difflib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch
from omegaconf import OmegaConf

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
STUDY = PROJECT / 'results/stage3/dl1'
SOURCE = PROJECT / '.codex-runtime/worktrees/stage3-dl1/results/stage3/dl1/dppo'
ORIGINAL = PROJECT / 'results/repro_halfcheetah/dppo'
E1C = PROJECT / '.codex-runtime/worktrees/stage3-e1c/results/stage3_e1c/dppo'
WORKER = STUDY / 'code/gate_dl1_worker.py'
sys.path.append(str(PROJECT / 'results/stage3/critic/code'))
from gate_common import compare, numeric_settings, require
from gate_dl1_worker import FIXTURE, write

COMPARE_FIELDS = ('initial_rng', 'initial_model', 'initial_actor_optimizer', 'initial_critic_optimizer',
                  'input_batch', 'loss_components', 'actor_gradients', 'critic_gradients', 'permutations',
                  'actor_parameters', 'critic_parameters', 'model_state', 'actor_optimizer_state',
                  'critic_optimizer_state', 'actor_scheduler_state', 'critic_scheduler_state',
                  'rng_after_update', 'rng_after_iteration', 'logged_values', 'sampling', 'input_batch_after')


def config_gate(source, output):
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    fixture = torch.load(FIXTURE, map_location='cpu', weights_only=False)
    old = OmegaConf.load(ORIGINAL / 'cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp.yaml')
    new = OmegaConf.load(source / 'cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp.yaml')
    for cfg in (old, new):
        cfg.seed = 0
        cfg.train.n_train_itr = 140
        cfg.train.save_model_freq = 35
        cfg.wandb = None
        # Approved output/assets are absolute. Identical paths avoid disguising
        # an algorithmic difference under path interpolation.
        cfg.logdir = str(STUDY / 'DL1_seed0')
        cfg.base_policy_path = fixture['config']['base_policy_path']
        cfg.normalization_path = fixture['config']['normalization_path']
    OmegaConf.update(new, 'train.adv_estimator', 'mc_critic', force_add=True)
    OmegaConf.update(new, 'train.baseline_diag', True, force_add=True)
    a, b = OmegaConf.to_container(old, resolve=True), OmegaConf.to_container(new, resolve=True)
    differences = []
    def walk(left, right, path=''):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = path + '.' + key if path else key
                if key not in left:
                    differences.append({'key': child, 'old': '<absent>', 'new': right[key]})
                elif key not in right:
                    differences.append({'key': child, 'old': left[key], 'new': '<absent>'})
                else:
                    walk(left[key], right[key], child)
        elif left != right:
            differences.append({'key': path, 'old': left, 'new': right})
    walk(a, b)
    expected = [{'key': 'train.adv_estimator', 'old': '<absent>', 'new': 'mc_critic'},
                {'key': 'train.baseline_diag', 'old': '<absent>', 'new': True}]
    require(differences == expected, 'G1 resolved config differs beyond the two authorized flags: ' + repr(differences))
    old_text = OmegaConf.to_yaml(OmegaConf.create(a), resolve=True)
    new_text = OmegaConf.to_yaml(OmegaConf.create(b), resolve=True)
    (output / 'stage1_resolved.yaml').write_text(old_text)
    (output / 'dl1_resolved.yaml').write_text(new_text)
    diff = ''.join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                     fromfile='Stage-1 DPPO', tofile='DL1'))
    (output / 'resolved_config.diff').write_text(diff)
    return {'passed': True, 'differences': differences, 'resolved_diff': str(output / 'resolved_config.diff'),
            'common_overrides': ['train.n_train_itr=140', 'train.save_model_freq=35', 'seed=0', 'wandb=null'],
            'paths_normalized_to_same_absolute_assets_and_output': True,
            'stage1_source': str(ORIGINAL), 'source': str(source), 'reference_upstream_commit': 'cc7234a'}


def loaded(path, name='evidence.pt'):
    return torch.load(path / name, map_location='cpu', weights_only=False)


def compare_evidence(before, after, fields=COMPARE_FIELDS):
    checks = {key: compare(before[key], after[key], True, key) for key in fields}
    return {'passed': all(c['passed'] for c in checks.values()), 'comparisons': checks,
            'max_abs': max(c['max_abs'] for c in checks.values()),
            'max_rel': max((c['max_rel'] for c in checks.values() if isinstance(c['max_rel'], (int, float))), default=0),
            'bitwise_required': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('cpu', 'live', 'merge'), required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-result', type=Path)
    parser.add_argument('--live-result', type=Path)
    parser.add_argument('--timeout-seconds', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Refuse to overwrite gate output')
    output.mkdir(parents=True)
    default_timeout = {'cpu': 7140, 'live': 3540, 'merge': 120}[args.phase]
    timeout = args.timeout_seconds or default_timeout
    require(0 < timeout <= default_timeout, 'Gate timeout exceeds approved phase cap')
    started = time.monotonic()
    report = {'passed': False, 'status': 'running', 'phase': args.phase, 'source': str(args.source),
              'original_source': str(ORIGINAL), 'original_upstream_commit': 'cc7234ad7ff39a8f32de3af903606723a16f0648',
              'e1c_reference': str(E1C), 'fixture': str(FIXTURE), 'checks': {}, 'commands': [],
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'numeric_settings': numeric_settings(),
              'cpu_gate_seconds': 0., 'gpu_gate_seconds': 0.}
    def save():
        write(output / 'gate_summary.json', report)
    def run(name, source, kind, profile):
        report['active_worker'] = name
        target = output / name
        command = [sys.executable, '-B', str(WORKER), '--source', str(source), '--kind', kind,
                   '--profile', profile, '--output', str(target)]
        report['commands'].append(command)
        save()
        remaining = timeout - (time.monotonic() - started)
        require(remaining > 0, 'Gate exhausted its time cap')
        env = os.environ.copy()
        if kind != 'live':
            env['CUDA_VISIBLE_DEVICES'] = ''
        print('GATE START ' + name, flush=True)
        with (output / (name + '.out')).open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, cwd=str(source), start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                raise
        require(code == 0, 'Gate worker failed: ' + name + '; read ' + str(output / (name + '.out')))
        metadata = json.loads((target / 'worker.json').read_text())
        require(metadata.get('passed') and metadata.get('all_children_exited'), 'Gate worker/cleanup failed: ' + name)
        report.setdefault('workers', []).append({'name': name, 'seconds': metadata['elapsed_seconds'], 'exit_code': code})
        save()
        return target
    save()
    try:
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        if args.phase == 'cpu':
            require(not torch.cuda.is_available(), 'CPU phase must hide GPUs')
            report['checks']['G1'] = config_gate(args.source, output)
            save()
            old_features = run('G2_e1c', E1C, 'features', 'e1c')
            new_features = run('G2_dl1', args.source, 'features', 'dl1_on')
            feature_comparison = compare(loaded(old_features, 'features.pt'), loaded(new_features, 'features.pt'), True)
            report['checks']['G2'] = {'passed': feature_comparison['passed'], 'e1c_comparison': feature_comparison,
                'e1c_formula': json.loads((old_features / 'features.json').read_text()),
                'dl1_formula': json.loads((new_features / 'features.json').read_text())}
            save()
            require(report['checks']['G2']['passed'], 'G2 E1C/DL1 advantages differ')
            reference_path = run('G5_stage1', ORIGINAL, 'replay', 'default')
            default_path = run('G5_dl1_defaults', args.source, 'replay', 'default')
            reference, default = loaded(reference_path), loaded(default_path)
            report['checks']['G5'] = compare_evidence(reference, default)
            report['checks']['G5']['comparison_scope'] = 'direct cc7234a native update, end-of-iteration schedulers/model step, fixed-noise sampling, parameters/optimizer/global RNG'
            save()
            require(report['checks']['G5']['passed'], 'G5 direct Stage-1 default replay failed')
            del default
            off_path = run('G6_diag_off', args.source, 'replay', 'dl1_off')
            on_path = run('G6_diag_on', args.source, 'replay', 'dl1_on')
            off, on = loaded(off_path), loaded(on_path)
            report['checks']['G6'] = compare_evidence(off, on)
            report['checks']['G6']['prepared_batch'] = compare(off['prepared_batch'], on['prepared_batch'], True)
            report['checks']['G6']['passed'] &= report['checks']['G6']['prepared_batch']['passed']
            save()
            require(report['checks']['G6']['passed'], 'G6 baseline diagnostics changed the full CPU PPO update')
            critic_fields = ('critic_first_step_parameters', 'critic_first_step_optimizer')
            critic = compare_evidence(reference, on, critic_fields)
            critic['targets'] = compare(reference['native_gae']['returns_k'], on['native_gae']['returns_k'], True)
            critic['returns_trajs'] = compare(reference['native_gae']['returns_trajs'], on['native_gae']['returns_trajs'], True)
            critic['same_native_gae_statements'] = reference['native_gae']['source_ast'] == on['native_gae']['source_ast']
            critic['first_minibatch_gradient'] = compare(reference['critic_gradients'][0], on['critic_gradients'][0], True)
            critic['first_permutation'] = compare(reference['permutations'][0], on['permutations'][0], True)
            critic['passed'] &= all(critic[k]['passed'] for k in ('targets', 'returns_trajs', 'first_minibatch_gradient', 'first_permutation'))
            critic['passed'] &= critic['same_native_gae_statements']
            critic['gae_lambda'] = .95
            report['checks']['G3'] = critic
            save()
            require(critic['passed'], 'G3 native critic targets/first critic step differ from Stage-1 DPPO')
            report['cpu_gate_seconds'] = time.monotonic() - started
        elif args.phase == 'live':
            require(args.cpu_result is not None, 'Live G4 must require passed CPU gates')
            previous = json.loads(args.cpu_result.read_text())
            require(previous.get('passed') and previous.get('phase') == 'cpu', 'CPU gate not passed')
            path = run('G4_live', args.source, 'live', 'dl1_on')
            report['checks']['G4'] = json.loads((path / 'G4.json').read_text())
            report['cpu_gate_seconds'] = previous['cpu_gate_seconds']
            report['gpu_gate_seconds'] = time.monotonic() - started
        else:
            require(args.cpu_result is not None and args.live_result is not None, 'Merge requires both gate summaries')
            cpu, live_result = json.loads(args.cpu_result.read_text()), json.loads(args.live_result.read_text())
            require(cpu.get('passed') and live_result.get('passed'), 'Cannot merge failed gate phases')
            require(cpu['source'] == live_result['source'] == str(args.source), 'Gate source paths differ')
            report['checks'] = dict(cpu['checks'], **live_result['checks'])
            require(set(report['checks']) == {'G1', 'G2', 'G3', 'G4', 'G5', 'G6'}, 'Missing gate')
            require(all(x['passed'] for x in report['checks'].values()), 'One gate did not pass')
            report['cpu_gate_seconds'] = cpu['cpu_gate_seconds']
            report['gpu_gate_seconds'] = live_result['gpu_gate_seconds']
            report['phase_records'] = [str(args.cpu_result), str(args.live_result)]
        report.pop('active_worker', None)
        report.update(passed=True, status='passed')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if args.phase == 'cpu':
            report['cpu_gate_seconds'] = report['elapsed_seconds']
        elif args.phase == 'live':
            report['gpu_gate_seconds'] = report['elapsed_seconds']
        save()


if __name__ == '__main__':
    main()
