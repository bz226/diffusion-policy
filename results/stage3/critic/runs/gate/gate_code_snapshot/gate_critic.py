#!/usr/bin/env python
"""Bounded Addendum C gate. A failure stops before any scientific campaign."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from gate_common import compare, numeric_settings, require

PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
WORKTREE = PARENT / '.codex-runtime/worktrees/stage3-critic'
ROOT = PARENT / 'results/stage3/critic'
CODE = ROOT / 'code'
REFERENCE = PARENT / 'results/stage3_gradmean/dppo'
EDITED = WORKTREE / 'results/stage3_critic/dppo'
FROZEN = PARENT / 'results/stage3/gradmean/prepare/P0/frozen_P0.pt'
FIELDS = ('initial_rng', 'initial_model', 'initial_actor_optimizer', 'initial_critic_optimizer',
          'constructor_actor_optimizer', 'constructor_critic_optimizer', 'constructor_actor_scheduler',
          'constructor_critic_scheduler', 'input_batch', 'loss_components', 'actor_gradients',
          'critic_gradients', 'permutations', 'actor_parameters', 'critic_parameters', 'model_state',
          'actor_optimizer_state', 'critic_optimizer_state', 'actor_scheduler_state', 'critic_scheduler_state',
          'rng_after_update', 'diagnostic_generator_state', 'logged_values', 'sampling', 'input_batch_after')
STRICT = {'initial_rng', 'permutations', 'rng_after_update', 'diagnostic_generator_state'}


def source_files(root):
    result = {}
    for path in root.rglob('*'):
        name = path.relative_to(root)
        if any(part in ('__pycache__', '.pytest_cache') for part in name.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        require('.git' not in name.parts and not path.is_symlink(), 'Source must be ordinary files')
        if path.is_file():
            result[str(name)] = path
    return result


def load(path):
    return torch.load(path / 'evidence.pt', map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/gate')
    parser.add_argument('--timeout-seconds', type=int, default=7140)
    args = parser.parse_args()
    output = args.output.resolve()
    require(ROOT in output.parents and not output.exists(), 'Gate needs a fresh output directory inside Addendum C')
    require(0 < args.timeout_seconds <= 7140, 'Gate must fit the approved two-hour allocation')
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {'passed': False, 'status': 'running', 'stage': 'full', 'C1': {}, 'C2': {}, 'C3': {},
              'C4': {'passed': False, 'cases': {}}, 'C5': {'passed': False, 'cases': {}},
              'commands': [], 'workers': [], 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'reference': str(REFERENCE), 'edited': str(EDITED), 'parent_repository': str(WORKTREE),
              'parent_commit': subprocess.check_output(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True).strip(),
              'reference_parent_commit': subprocess.check_output(['git', '-C', str(PARENT), 'rev-parse', 'stage3-gradmean'], text=True).strip(),
              'numeric_settings': numeric_settings(), 'gate_wall_limit_seconds': 7200}
    def save():
        (output / 'gate_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def run(script, name, flags):
        report['active_phase'] = name
        argv = [sys.executable, '-B', str(CODE / script), '--output', str(output / name)] + list(flags)
        report['commands'].append(argv)
        save()
        remaining = args.timeout_seconds - (time.monotonic() - started)
        require(remaining > 0, 'Gate reached its execution cap')
        print('GATE START ' + name, flush=True)
        with (output / (name + '.out')).open('x') as stream:
            process = subprocess.Popen(argv, cwd=str(EDITED), stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                returncode = process.wait(timeout=remaining)
            except BaseException:
                # Give the worker's Python finally block time to close its env.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                raise
        report['workers'].append({'name': name, 'returncode': returncode})
        save()
        require(returncode == 0, name + ' failed; inspect its output')
        return output / name
    try:
        require(torch.cuda.is_available(), 'C gate requires an allocated GPU')
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        require(FROZEN.is_file(), 'Reuse the completed gradient-mean frozen_P0.pt')
        require(subprocess.check_output(['git', '-C', str(WORKTREE), 'branch', '--show-current'], text=True).strip()
                == 'stage3-critic', 'C source parent must be stage3-critic')
        report['gpu_name'] = torch.cuda.get_device_name(0)
        initial = source_files(EDITED)
        contents = {name: (path.read_bytes(), path.stat().st_mode & 0o111) for name, path in initial.items()}
        snapshot = output / 'source_snapshot'
        for name, path in initial.items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        report.update(source_snapshot=str(snapshot), source_files=len(initial))
        code_snapshot = output / 'gate_code_snapshot'
        code_snapshot.mkdir()
        for path in CODE.glob('gate*.py'):
            shutil.copy2(path, code_snapshot / path.name)
        fixture = PARENT / 'results/stage3/runs/gate_capture/nc4/training_batch.pt'
        for feature in ('C1', 'C3', 'C2'):
            path = run('gate_features.py', feature, ['--repo', str(EDITED), '--fixture', str(fixture),
                       '--frozen', str(FROZEN), '--feature', feature])
            report[feature] = json.loads((path / 'feature.json').read_text())
            save()
            require(report[feature]['passed'], feature + ' failed')
        prepared = {}
        for condition in ('default', 'nc4'):
            fixture = PARENT / 'results/stage3/runs/gate_capture' / condition / 'training_batch.pt'
            prepared[condition] = run('gate_replay.py', 'prepare_cpu_' + condition,
                                     ['--repo', str(REFERENCE), '--fixture', str(fixture), '--device', 'cpu', '--prepare-cpu'])
            for device in ('cpu', 'cuda:0'):
                name = condition + '_' + device.replace(':', '')
                flags = ['--fixture', str(fixture), '--device', device]
                if device == 'cpu':
                    flags += ['--cpu-logprobs', str(prepared[condition] / 'cpu_old_logprobs.pt')]
                before = run('gate_replay.py', 'C5_' + name + '_reference', ['--repo', str(REFERENCE)] + flags)
                after = run('gate_replay.py', 'C5_' + name + '_edited', ['--repo', str(EDITED)] + flags)
                old, new = load(before), load(after)
                checks = {key: compare(old[key], new[key], device == 'cpu' or key in STRICT, key) for key in FIELDS}
                actor_steps = 20 if condition == 'default' else 1
                require(len(old['actor_gradients']) == len(new['actor_gradients']) == actor_steps, 'C5 actor step count changed')
                require(len(old['critic_gradients']) == len(new['critic_gradients']) == 20, 'C5 critic step count changed')
                report['C5']['cases'][name] = {'passed': all(item['passed'] for item in checks.values()),
                                              'comparisons': checks, 'actor_steps': actor_steps, 'critic_steps': 20}
                save()
                require(report['C5']['cases'][name]['passed'], 'C5 default differential failed: ' + name)
                del old, new
        report['C5']['passed'] = True
        fixture = PARENT / 'results/stage3/runs/gate_capture/nc4/training_batch.pt'
        for estimator in ('mc_critic', 'gae'):
            overrides = {'model.actor_loss': 'score', 'train.stage3_diag': True,
                         'train.actor_optimizer': 'sgd', 'train.actor_lr': 1e-4,
                         'train.actor_lr_scheduler.min_lr': 1e-4}
            if estimator == 'mc_critic':
                overrides.update({'train.adv_estimator': 'mc_critic', 'train.decision_discount': True,
                                  'model.gamma_denoising': 1., 'model.logprob_reduce': 'sum', 'model.norm_adv': False})
            paths = []
            for enabled in (False, True):
                overrides['train.baseline_diag'] = enabled
                paths.append(run('gate_replay.py', 'C4_' + estimator + ('_on' if enabled else '_off'),
                                 ['--repo', str(EDITED), '--fixture', str(fixture), '--device', 'cpu',
                                  '--cpu-logprobs', str(prepared['nc4'] / 'cpu_old_logprobs.pt'),
                                  '--overrides', json.dumps(overrides, sort_keys=True)]))
            old, new = load(paths[0]), load(paths[1])
            checks = {key: compare(old[key], new[key], True, key) for key in FIELDS}
            checks['prepared_batch'] = compare(old['prepared_batch'], new['prepared_batch'], True)
            checks['rng_before_prepare'] = compare(old['rng_before_prepare'], new['rng_before_prepare'], True)
            checks['rng_after_prepare'] = compare(old['rng_after_prepare'], new['rng_after_prepare'], True)
            for variant, evidence in (('off', old), ('on', new)):
                checks['prepare_rng_inert_' + variant] = compare(evidence['rng_before_prepare'], evidence['rng_after_prepare'], True)
                require(evidence['measurement_state_inert']['passed'], 'C4 diagnostics changed training state')
            report['C4']['cases'][estimator] = {'passed': all(item['passed'] for item in checks.values()),
                                               'comparisons': checks, 'actor_steps': len(new['actor_gradients']),
                                               'critic_steps': len(new['critic_gradients'])}
            require(len(old['actor_gradients']) == len(new['actor_gradients']) == 1 and
                    len(old['critic_gradients']) == len(new['critic_gradients']) == 20, 'C4 wrong optimizer step count')
            save()
            require(report['C4']['cases'][estimator]['passed'], 'C4 baseline diagnostics changed training: ' + estimator)
            del old, new
        report['C4']['passed'] = True
        current = source_files(EDITED)
        require(current.keys() == initial.keys() and all((path.read_bytes(), path.stat().st_mode & 0o111) == contents[name]
                                                       for name, path in current.items()), 'C source changed during gate')
        cleanup = []
        for path in sorted(output.glob('**/cleanup.json')):
            row = json.loads(path.read_text())
            require(row.get('status') == 'passed' and row.get('all_children_exited') is True,
                    'A gate worker did not close and reap its vector env: ' + str(path))
            cleanup.append({'path': str(path), 'workers': len(row.get('workers', [])),
                            'all_children_exited': True, 'status': 'passed'})
        require(len(cleanup) == len(report['workers']) + 1, 'Missing cleanup evidence (C3 constructs two envs)')
        require(not mp.active_children(), 'Gate controller has active children')
        report.update(passed=True, status='passed', scientific_source_unchanged=True,
                      cleanup=cleanup, all_children_exited=True)
        report.pop('active_phase', None)
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        save()


if __name__ == '__main__':
    main()
