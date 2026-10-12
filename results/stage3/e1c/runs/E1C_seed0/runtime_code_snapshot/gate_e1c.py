#!/usr/bin/env python
"""Saved-batch E1C A1/A3 gate; reuse unchanged Addendum C A2/A4/A5 evidence."""
import argparse
import copy
import difflib
import filecmp
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PARENT / 'results/stage3/e1c'
SOURCE = PARENT / '.codex-runtime/worktrees/stage3-e1c/results/stage3_e1c/dppo'
WORKTREE = PARENT / '.codex-runtime/worktrees/stage3-e1c'
CRITIC_GATE = PARENT / 'results/stage3/critic/runs/gate'
FIXTURE = PARENT / 'results/stage3/runs/gate_capture/nc4/training_batch.pt'
CPU_LOGPROBS = CRITIC_GATE / 'prepare_cpu_nc4/cpu_old_logprobs.pt'
sys.path.insert(0, str(PARENT / 'results/stage3/critic/code'))
from gate_common import compare, freeze, numeric_settings, require, rng_state, transfer
from gate_features import brute_returns, error
from gate_replay import make_agent


def ordinary_source(directory):
    result = {}
    for path in directory.rglob('*'):
        name = path.relative_to(directory)
        if '__pycache__' in name.parts or '.pytest_cache' in name.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        require('.git' not in name.parts and not path.is_symlink(), 'Source must contain ordinary files only')
        if path.is_file():
            result[str(name)] = path
    return result


def check_source():
    old, new = ordinary_source(CRITIC_GATE / 'source_snapshot'), ordinary_source(SOURCE)
    require(old.keys() == new.keys(), 'E1C scientific source file set differs from the passed critic gate')
    differences = [name for name in old if not filecmp.cmp(str(old[name]), str(new[name]), shallow=False)
                   or (old[name].stat().st_mode & 0o111) != (new[name].stat().st_mode & 0o111)]
    administrative = []
    if '.gitignore' in differences:
        prior = old['.gitignore'].read_text()
        replacement = ('# Keep source, measurements, result.pkl, and Hydra configurations versionable.\n'
                       '# Runtime logs and command/monitor transcripts stay local.\n'
                       '*.log\n*.log.*\n*.out\n*.err\ncommands.jsonl\n*_commands.jsonl\n'
                       'monitor_history.jsonl\n*_monitor_history.jsonl\n*_monitor.jsonl\nevents.jsonl\n')
        old_line = '# Keep source, result.pkl, run.log, and Hydra configurations versionable.\n'
        require(prior.count(old_line) == 1, 'Unexpected original source-ignore policy')
        require(new['.gitignore'].read_text() == prior.replace(old_line, replacement),
                'Source-ignore policy differs beyond the required helper-generated log exclusions')
        require((old['.gitignore'].stat().st_mode & 0o111) == (new['.gitignore'].stat().st_mode & 0o111),
                'Source-ignore executable bits changed')
        administrative.append({'path': '.gitignore', 'reason': 'Current snapshot helper implements required runtime-log exclusions.',
                               'diff': ''.join(difflib.unified_diff(prior.splitlines(True), new['.gitignore'].read_text().splitlines(True),
                                                                 fromfile='critic-gate/.gitignore', tofile='e1c/.gitignore'))})
        differences.remove('.gitignore')
    require(not differences, 'Scientific source differs from passed critic gate: ' + repr(differences))
    return {'passed': True, 'comparison': 'all ordinary file contents and executable bits',
            'files': len(old), 'scientific_files_identical': len(old) - len(administrative),
            'administrative_ignore_policy_difference': administrative,
            'reference_snapshot': str(CRITIC_GATE / 'source_snapshot'), 'differences': differences}


def overrides():
    return {'model.clip_ploss_coef': 1e6, 'model.clip_ploss_coef_base': 1e6,
            'train.target_kl': None, 'model.clamp_logprob': False, 'model.randn_clip_value': 100,
            'train.actor_single_step': True, 'model.actor_loss': 'score', 'train.stage3_diag': False,
            'train.adv_estimator': 'mc_critic', 'train.baseline_diag': True,
            'train.decision_discount': False, 'model.gamma_denoising': 0.99,
            'model.logprob_reduce': 'mean', 'model.norm_adv': True,
            'train.actor_optimizer': 'adamw', 'train.actor_lr': 1e-4,
            'train.actor_lr_scheduler.min_lr': 1e-4, 'train.critic_init_path': None}


def prepare(agent, saved, batch, estimator):
    agent.adv_estimator = estimator
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_before']))
    agent._stage3_iteration_start()
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_after']))
    agent._stage3_actions = saved['action_trajs'].copy()
    agent._stage3_rollout_finished(saved['raw_reward_trajs'], saved['firsts_trajs'], False)
    return agent._stage3_prepare_batch(
        saved['raw_reward_trajs'], saved['firsts_trajs'], saved['terminated_trajs'], saved['action_trajs'],
        batch['obs_k'], batch['chains_k'], batch['returns_k'], batch['values_k'], batch['advantages_k'],
        batch['logprobs_k'], saved['scaled_reward_trajs'])


def loss_arguments(agent, batch, coefficients, pairs):
    decisions, steps = torch.unravel_index(pairs, (agent.n_steps * agent.n_envs, agent.model.ft_denoising_steps))
    return ({key: value[decisions] for key, value in batch['obs_k'].items()},
            batch['chains_k'][decisions, steps], batch['chains_k'][decisions, steps + 1], steps,
            batch['returns_k'][decisions], batch['values_k'][decisions], coefficients[decisions],
            batch['logprobs_k'][decisions, steps])


def a1(agent, saved, batch, permutation):
    from agent.finetune.critic_baseline import mc_critic_advantages
    coefficients = prepare(agent, saved, batch, 'mc_critic')
    divisor = np.sqrt(np.asarray(saved['reward_scaler_before']['ret_rms'].var, dtype=np.float64)
                      + saved['reward_scaler_before']['epsilon'])
    returns = brute_returns(saved['raw_reward_trajs'], saved['firsts_trajs'], agent.gamma)
    values = batch['values_k'].double().numpy().reshape(returns.shape)
    expected = returns - divisor * values
    actual = agent._critic_baseline_raw_adv
    raw_check = error(expected, actual)
    require(actual.dtype == np.float64 and raw_check['relative_l2'] <= 1e-12,
            'A1 native float64 advantages differ from explicit G-cV')
    require(np.array_equal(agent._critic_baseline_divisor, divisor), 'A1 baseline divisor differs from start state')
    cast_check = compare(torch.as_tensor(actual, dtype=torch.float32).reshape(-1), coefficients, True)
    require(cast_check['passed'] and not agent.decision_discount, 'A1 must cast raw residual without decision discount')
    nonunit = {}
    for label, candidate in (('scalar', np.asarray(3.75)), ('per_environment', np.linspace(2., 6., values.shape[1]))):
        result = mc_critic_advantages(saved['raw_reward_trajs'], saved['firsts_trajs'], agent.gamma, values, candidate)
        nonunit[label] = error(returns - values * candidate, result)
        require(nonunit[label]['relative_l2'] <= 1e-12, 'A1 nonunit divisor check failed')

    # Capture normalization inside the unchanged native loss, before its inert
    # 0/1 quantile bounds and denoising weights. No source is patched.
    lines, first_line = inspect.getsourcelines(agent.model.loss)
    capture_line = first_line + next(i for i, line in enumerate(lines) if 'advantage_min = torch.quantile' in line)
    loss_code = agent.model.loss.__func__.__code__
    captures = []
    def trace(frame, event, arg):
        if frame.f_code is not loss_code:
            return None
        if event == 'line' and frame.f_lineno == capture_line:
            captures.append(frame.f_locals['advantages'].detach().clone())
        return trace
    chunks = []
    with torch.no_grad():
        for start in range(0, len(permutation), agent.batch_size):
            pairs = permutation[start:start + agent.batch_size]
            arguments = loss_arguments(agent, batch, coefficients, pairs)
            incoming = arguments[6].clone()
            target = (incoming - incoming.mean()) / (incoming.std() + 1e-8)
            before = len(captures)
            previous_trace = sys.gettrace()
            try:
                sys.settrace(trace)
                agent.model.loss(*arguments, use_bc_loss=agent.use_bc_loss, reward_horizon=agent.reward_horizon)
            finally:
                sys.settrace(previous_trace)
            require(len(captures) == before + 1, 'A1 did not observe exactly one normalization per chunk')
            normalized = captures[-1]
            comparison = compare(target, normalized, True)
            mean, std = normalized.double().mean().item(), normalized.double().std().item()
            require(comparison['passed'], 'A1 normalization differs from E1 native formula')
            require(abs(mean) <= 2e-6 and abs(std - 1.) <= 2e-6, 'A1 normalized chunk moments are not zero/unit')
            chunks.append({'index': len(chunks), 'pairs': len(pairs), 'mean': mean, 'std_ddof1': std,
                           'expected_std_epsilon_adjusted': float(incoming.std() / (incoming.std() + 1e-8)),
                           'formula_comparison': comparison})
    require(len(chunks) == 4 and sum(row['pairs'] for row in chunks) == 200000, 'Wrong complete-batch partition')
    return {'passed': True, 'raw_advantages': raw_check, 'nonunit_divisors': nonunit,
            'native_float32_cast': cast_check, 'divisor': float(divisor), 'normalization': 'per shuffled 50000-pair chunk, matching E1',
            'chunk_moment_tolerance': 2e-6, 'std_ddof': 1, 'epsilon': 1e-8,
            'normalization_capture_source': str(SOURCE / 'model/diffusion/diffusion_ppo.py'),
            'normalization_capture_line': capture_line, 'chunks': chunks}


def a3(agent, saved, batch, permutation, directory):
    # Replace the critic's output with zero by zeroing its test-local weights.
    # The source and saved fixture stay unchanged.
    original_critic = freeze(agent.model.critic.state_dict())
    try:
        with torch.no_grad():
            for parameter in agent.model.critic.parameters():
                parameter.zero_()
            zero_values = agent.model.critic(batch['obs_k']).reshape(-1).detach()
        require(torch.count_nonzero(zero_values).item() == 0, 'A3 critic output is not exactly zero')
        zero_batch = dict(batch, values_k=zero_values)
        critic_coeff = prepare(agent, saved, zero_batch, 'mc_critic')
        none_coeff = prepare(agent, saved, zero_batch, 'mc_none')
        coeff_check = compare(none_coeff, critic_coeff, True)
        require(coeff_check['passed'], 'A3 zero-critic and no-baseline coefficients differ')
        agent.adv_estimator = 'mc_critic'
        from agent.finetune.critic_baseline import configured_actor_gradient
        configured_batch = {'chains_k': zero_batch['chains_k'], 'obs_k': zero_batch['obs_k'],
                            'returns': zero_batch['returns_k'], 'values': zero_batch['values_k'],
                            'oldlogprobs': zero_batch['logprobs_k']}
        before = rng_state()
        gradient_critic = configured_actor_gradient(agent, configured_batch, critic_coeff, permutation)
        agent.adv_estimator = 'mc_none'
        gradient_none = configured_actor_gradient(agent, configured_batch, none_coeff, permutation)
        rng_check = compare(before, rng_state(), True)
        comparison = error(gradient_none.double().numpy(), gradient_critic.double().numpy())
        exact = compare(gradient_none, gradient_critic, True)
        require(comparison['relative_l2'] <= 1e-6 and rng_check['passed'], 'A3 gradient/RNG equality failed')
        torch.save({'critic_gradient': gradient_critic.detach(), 'no_baseline_gradient': gradient_none.detach(),
                    'permutation': permutation, 'critic_coefficients': critic_coeff,
                    'no_baseline_coefficients': none_coeff}, directory / 'A3_evidence.pt')
        return {'passed': True, 'relative_l2_tolerance': 1e-6, 'gradient_comparison': comparison,
                'bitwise_gradient_comparison': exact, 'coefficient_comparison': coeff_check,
                'global_rng': rng_check, 'zero_critic_values': int(zero_values.numel()),
                'gradient_norm': float(gradient_none.double().norm()), 'pairs': len(permutation),
                'chunk_size': agent.batch_size, 'chunk_weights': [0.25] * 4,
                'loss_path': 'configured_actor_gradient calls unchanged model.loss for every chunk; same final-epoch permutation as training',
                'optimizer_step_applied': False}
    finally:
        agent.model.critic.load_state_dict(original_critic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/gate')
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / 'gate_summary.json'
    require(not summary.exists(), 'Refuse to overwrite existing E1C gate evidence')
    started = time.monotonic()
    report = {'status': 'running', 'passed': False, 'command': [sys.executable] + sys.argv,
              'source': str(SOURCE), 'parent_worktree': str(WORKTREE), 'fixture': str(FIXTURE),
              'slurm_job_id': None, 'cpu_only': True,
              'cpu_logprobs': str(CPU_LOGPROBS), 'device': 'cpu', 'python_version': platform.python_version(),
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'configured_training_stage3_diag': True, 'gate_stage3_diag': False,
              'gate_scope': 'Saved-batch A1/A3 only. Training diagnostics are separately covered by reused C4; no rollout or optimizer update.',
              'normalization_amendment': 'User approved per-chunk normalization matching the existing E1 runs.',
              'gate_wall_limit_seconds': 1200, 'checks': {}}
    agent = None
    try:
        torch.set_num_threads(2)
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        require(not torch.cuda.is_available(), 'This gate is CPU-only; hide GPUs')
        report['numeric_settings'] = numeric_settings()
        report['source_comparison'] = check_source()
        report['source_parent_commit'] = subprocess.check_output(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True).strip()
        report['parent_commit'] = report['source_parent_commit']
        snapshot = directory / 'source_snapshot'
        require(not snapshot.exists(), 'Refuse to replace an existing gate source snapshot')
        for name, path in ordinary_source(SOURCE).items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        report['source_snapshot'] = str(snapshot)
        previous = json.loads((CRITIC_GATE / 'gate_summary.json').read_text())
        require(previous['passed'] and previous['scientific_source_unchanged'] and previous['all_children_exited'], 'Prior critic gate incomplete')
        for target, original in (('A2', 'C2'), ('A4', 'C4'), ('A5', 'C5')):
            evidence = previous[original]
            require(evidence['passed'], 'Reused gate did not pass: ' + original)
            with (directory / ('reused_' + original + '.json')).open('x') as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
                stream.write('\n')
            report['checks'][target] = {'passed': True, 'reused': True, 'source_gate': original,
                'source_path': str(CRITIC_GATE / 'gate_summary.json'), 'copied_evidence': 'reused_' + original + '.json',
                'source_parent_commit': previous['parent_commit'], 'reference_parent_commit': previous['reference_parent_commit']}
        agent, saved, cfg = make_agent(SOURCE, FIXTURE, 'cpu', directory, overrides())
        require(not mp.active_children(), 'Native constructor left workers after cleanup')
        report['effective_overrides'] = overrides()
        batch = transfer(saved['batch'], 'cpu')
        batch['logprobs_k'] = torch.load(CPU_LOGPROBS, map_location='cpu', weights_only=True)
        generator = torch.Generator(device='cpu')
        generator.set_state(saved['rng']['torch_cpu'])
        total = agent.n_steps * agent.n_envs * agent.model.ft_denoising_steps
        for _ in range(agent.update_epochs):
            permutation = torch.randperm(total, generator=generator)
        before_state = freeze({'model': agent.model.state_dict(), 'actor_optimizer': agent.actor_optimizer.state_dict(),
                               'critic_optimizer': agent.critic_optimizer.state_dict()})
        report['checks']['A1'] = a1(agent, saved, batch, permutation)
        print('A1 passed: G-cV and all four native chunk-normalization moments.', flush=True)
        report['checks']['A3'] = a3(agent, saved, batch, permutation, directory)
        print('A3 passed: zero-critic gradient equals mc_none.', flush=True)
        after_state = freeze({'model': agent.model.state_dict(), 'actor_optimizer': agent.actor_optimizer.state_dict(),
                              'critic_optimizer': agent.critic_optimizer.state_dict()})
        report['model_optimizer_unchanged'] = compare(before_state, after_state, True)
        require(report['model_optimizer_unchanged']['passed'], 'Saved-batch gate changed model/optimizer state')
        report['source_comparison_after'] = check_source()
        cleanup = json.loads((directory / 'native/cleanup.json').read_text())
        require(cleanup['status'] == 'passed' and cleanup['all_children_exited'], 'Constructor cleanup failed')
        require(not mp.active_children(), 'Gate leaves active child processes')
        report['cleanup'] = {'status': 'passed', 'all_children_exited': True, 'workers': len(cleanup['workers']),
                             'evidence': str(directory / 'native/cleanup.json')}
        report.update(status='passed', passed=all(check['passed'] for check in report['checks'].values()))
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(report['checks'])
        report['elapsed_seconds'] = time.monotonic() - started
        summary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
