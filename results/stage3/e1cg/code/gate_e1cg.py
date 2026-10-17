#!/usr/bin/env python
"""CPU-only E1CG gates on the same saved fixture used for E1C verification."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
STUDY = PARENT / 'results/stage3/e1cg'
EDITED = PARENT / '.codex-runtime/worktrees/stage3-e1cg/results/stage3/e1cg/dppo'
REFERENCE = PARENT / '.codex-runtime/worktrees/stage3-e1c/results/stage3_e1c/dppo'
FIXTURES = PARENT / 'results/stage3/runs/gate_capture'
CRITIC = PARENT / 'results/stage3/critic'
sys.path.insert(0, str(CRITIC / 'code'))
from gate_common import compare, freeze, require, restore_rng, rng_state, transfer
from gate_replay import make_agent, replay
from gate_features import brute_returns, error


def flags(post=False, diagnostic=True):
    out = {'model.actor_loss': 'score', 'train.adv_estimator': 'mc_critic',
           'train.baseline_diag': True, 'train.stage3_diag': diagnostic}
    if post:
        out.update({'train.decision_discount': True, 'train.decision_discount_post_norm': True})
    return out


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepared(agent, saved, batch):
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_before']))
    agent._stage3_iteration_start()
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_after']))
    agent._stage3_actions = saved['action_trajs'].copy()
    agent._stage3_rollout_finished(saved['raw_reward_trajs'], saved['firsts_trajs'], False)
    return agent._stage3_prepare_batch(saved['raw_reward_trajs'], saved['firsts_trajs'],
        saved['terminated_trajs'], saved['action_trajs'], batch['obs_k'], batch['chains_k'],
        batch['returns_k'], batch['values_k'], batch['advantages_k'], batch['logprobs_k'],
        saved['scaled_reward_trajs'])


def loss_args(agent, batch, coeff, pairs):
    d, j = torch.unravel_index(pairs, (agent.n_steps * agent.n_envs, agent.model.ft_denoising_steps))
    return ({key: value[d] for key, value in batch['obs_k'].items()},
            batch['chains_k'][d, j], batch['chains_k'][d, j + 1], j,
            batch['returns_k'][d], batch['values_k'][d], coeff[d], batch['logprobs_k'][d, j]), d, j


def features(agent, saved, batch, out):
    from agent.finetune.stage3_math import actor_gradient, episode_indices
    from agent.finetune.critic_baseline import configured_actor_coefficients, configured_actor_gradient
    coefficient = prepared(agent, saved, batch)
    k = episode_indices(saved['firsts_trajs'])
    exact_k = np.broadcast_to((np.arange(500) % 250)[:, None], (500, 40))
    require(np.array_equal(k, exact_k), 'G2 time indices differ')
    broken = saved['firsts_trajs'].copy()
    broken[250, 0], broken[249, 0] = 0, 1
    caught = False
    try:
        episode_indices(broken)
    except ValueError as exc:
        caught = 'episode alignment failed' in str(exc)
    require(caught, 'G2 misalignment assertion did not fire')
    raw = brute_returns(saved['raw_reward_trajs'], saved['firsts_trajs'], agent.gamma)
    divisor = np.sqrt(np.asarray(saved['reward_scaler_before']['ret_rms'].var, dtype=np.float64)
                      + saved['reward_scaler_before']['epsilon'])
    values = batch['values_k'].double().numpy().reshape(raw.shape)
    residual = raw - divisor * values
    actual_raw = agent._critic_baseline_raw_adv
    c1 = error(residual, actual_raw)
    require(c1['relative_l2'] <= 1e-12, 'C1 explicit raw MC critic advantages differ')
    v6 = np.power(agent.gamma, exact_k) * residual
    g4 = error(v6, np.power(agent.gamma, k) * actual_raw)
    require(g4['relative_l2'] <= 1e-12, 'G4 V6 definition differs')
    gen = torch.Generator(device='cpu')
    gen.set_state(saved['rng']['torch_cpu'])
    for unused in range(agent.update_epochs):
        permutation = torch.randperm(200000, generator=gen)

    source, first_line = inspect.getsourcelines(agent.model.loss)
    capture = first_line + next(i for i, line in enumerate(source) if '# get ratio' in line) + 1
    code = agent.model.loss.__func__.__code__
    seen = []
    def trace(frame, event, arg):
        if frame.f_code is code:
            if event == 'line' and frame.f_lineno == capture:
                seen.append(frame.f_locals['advantages'].detach().clone())
            return trace
        return None
    update_errors = []
    with torch.no_grad():
        for start in range(0, 200000, 50000):
            arguments, decisions, steps = loss_args(agent, batch, coefficient, permutation[start:start + 50000])
            incoming = arguments[6].double().numpy()
            normalized = (incoming - incoming.mean()) / (incoming.std(ddof=1) + 1e-8)
            expected = normalized * np.power(agent.gamma, k.reshape(-1)[decisions.numpy()])
            expected *= np.power(agent.model.gamma_denoising, 9 - steps.numpy())
            previous = sys.gettrace()
            try:
                sys.settrace(trace)
                agent.model.loss(*arguments, reward_horizon=agent.reward_horizon,
                                 decision_weights=agent._stage3_decision_weights[decisions])
            finally:
                sys.settrace(previous)
            require(len(seen) == len(update_errors) + 1, 'G1 loss coefficient capture missing')
            check = error(expected, seen[-1].double().numpy())
            require(check['relative_l2'] <= 1e-6, 'G1 update coefficient mismatch')
            update_errors.append(check)

    lines, beginning = inspect.getsourcelines(actor_gradient)
    line = beginning + next(i for i, text in enumerate(lines) if 'loss = -(coefficients * logp)' in text)
    split_errors = []
    raw_pairs = coefficient.double().numpy()
    mean = raw_pairs.mean()
    std = np.sqrt(np.sum((raw_pairs - mean) ** 2) * 10 / (200000 - 1))
    normalized = (raw_pairs - mean) / (std + 1e-8)
    def split_trace(frame, event, arg):
        if frame.f_code is actor_gradient.__code__:
            if event == 'line' and frame.f_lineno == line:
                d = frame.f_locals['decisions_b'].cpu().numpy()
                j = frame.f_locals['steps_b'].cpu().numpy()
                expected = normalized[d] * np.power(agent.gamma, k.reshape(-1)[d]) * np.power(agent.model.gamma_denoising, 9 - j)
                check = error(expected, frame.f_locals['coefficients'].detach().double().numpy())
                require(check['relative_l2'] <= 1e-6, 'G1 split coefficient mismatch')
                split_errors.append(check)
            return split_trace
        return None
    before = rng_state()
    for begin in (0, 20):
        previous = sys.gettrace()
        try:
            sys.settrace(split_trace)
            actor_gradient(agent, agent._stage3_batch, coefficient, reduce='mean', gamma_denoising=.99,
                           normalize=True, env_indices=range(begin, begin + 20),
                           decision_weights=agent._stage3_decision_weights)
        finally:
            sys.settrace(previous)
    require(len(split_errors) == 4, 'G1 split did not cover all pairs')
    require(compare(before, rng_state(), True)['passed'], 'split diagnostics changed RNG')

    # General configured-gradient path must equal the actual loss pass with the same partition.
    configured_batch = {'raw_rewards': saved['raw_reward_trajs'], 'firsts': saved['firsts_trajs'],
        'values': batch['values_k'].numpy().reshape(500, 40), 'returns': batch['returns_k'],
        'chains_k': batch['chains_k'], 'obs_k': batch['obs_k'], 'oldlogprobs': batch['logprobs_k']}
    configured = configured_actor_coefficients(agent, configured_batch)
    require(compare(coefficient, configured, True)['passed'], 'configured coefficients differ from update')
    configured_grad = configured_actor_gradient(agent, configured_batch, configured, permutation)
    agent.actor_optimizer.zero_grad()
    for start in range(0, 200000, 50000):
        arguments, d, j = loss_args(agent, batch, coefficient, permutation[start:start + 50000])
        loss = agent.model.loss(*arguments, reward_horizon=agent.reward_horizon,
                               decision_weights=agent._stage3_decision_weights[d])[0]
        (loss * .25).backward()
    native_grad = torch.cat([p.grad.detach().flatten() for p in agent.model.actor_ft.parameters()])
    gradient_check = compare(native_grad, configured_grad, True)
    require(gradient_check['passed'], 'configured loss gradient differs from training accumulation')
    # Optional E2C follows the existing pre-normalization (norm off) path.
    old_setting = (agent.decision_discount_post_norm, agent.model.norm_adv,
                   agent.model.gamma_denoising, agent.model.logprob_reduce)
    try:
        agent.decision_discount_post_norm = False
        agent.model.norm_adv = False
        agent.model.gamma_denoising = 1.
        agent.model.logprob_reduce = 'sum'
        e2c_coefficients = configured_actor_coefficients(agent, configured_batch)
        explicit_float32 = torch.as_tensor(actual_raw, dtype=torch.float32).reshape(-1)
        explicit_float32 *= torch.as_tensor(np.power(agent.gamma, k), dtype=torch.float32).reshape(-1)
        e2c_check = compare(explicit_float32, e2c_coefficients, True)
        require(e2c_check['passed'], 'C1 actual E2C coefficient path differs')
    finally:
        (agent.decision_discount_post_norm, agent.model.norm_adv,
         agent.model.gamma_denoising, agent.model.logprob_reduce) = old_setting
    report = {'G1': {'passed': True, 'update_chunks': update_errors, 'split_chunks': split_errors,
              'update_normalization': 'per shuffled 50000-pair chunk; ddof=1; epsilon=1e-8',
              'split_normalization': 'full 200000-pair statistics before environment selection, unchanged from E1C',
              'configured_gradient': gradient_check},
              'G2': {'passed': True, 'time_indices_exact': True, 'misalignment_raises': True},
              'G4': {'passed': True, 'comparison': g4, 'definition': 'explicit float64 V6; exact E1C source predates V6'},
              'C1': {'passed': True, 'comparison': c1, 'optional_E2C': True, 'configured_float32': e2c_check, 'settings': {'decision_discount': True, 'decision_discount_post_norm': False, 'norm_adv': False, 'gamma_denoising': 1., 'logprob_reduce': 'sum'}}}
    write(out / 'features.json', report)


def strip_times(value):
    if isinstance(value, dict):
        return {k: strip_times(v) for k, v in value.items()
                if k not in ('seconds', 'elapsed_seconds') and not k.endswith('_seconds')}
    if isinstance(value, list):
        return [strip_times(v) for v in value]
    return value


def segments(agent, saved, batch, out, enabled):
    import agent.finetune.stage3_modes as modes
    from omegaconf import OmegaConf
    agent.seed = 5000
    OmegaConf.update(agent.cfg, 'train.ckpt_segments', enabled, force_add=True)
    OmegaConf.update(agent.cfg, 'train.ckpt_eval_repeats', 1, force_add=True)
    supplied = {'raw_rewards': saved['raw_reward_trajs'], 'firsts': saved['firsts_trajs'],
                'actions': saved['action_trajs'], 'chains_k': batch['chains_k'], 'obs_k': batch['obs_k']}
    calls = []
    def collect(**kw):
        calls.append(dict(kw))
        # Saved-rollout replay: explicitly preserve and test the caller's RNG ordering.
        random.random(); np.random.uniform(); torch.rand(3)
        agent.model.eval() if kw['deterministic'] else agent.model.train()
        return supplied
    agent._stage3_collect = collect
    restore_rng(saved['rng'])
    initial = rng_state()
    result = modes.run_ckpt_eval(agent)
    payload = {'result': result, 'calls': calls, 'initial_rng': initial, 'final_rng': rng_state()}
    torch.save(payload, out / 'segments.pt')
    if enabled:
        ep = saved['raw_reward_trajs'].T.reshape(80, 250).reshape(80, 5, 50).mean(axis=2)
        for sampler in ('train', 'eval'):
            rows = result['samplers'][sampler]['reward_segments']
            for i, row in enumerate(rows):
                require(row['mean'] == ep[:, i].mean(), 'G6 segment mean formula differs')
                require(row['se'] == ep[:, i].std(ddof=1) / np.sqrt(80), 'G6 segment SE formula differs')


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    override = json.loads(args.overrides)
    agent, saved, cfg = make_agent(args.source, FIXTURES / args.fixture / 'training_batch.pt', 'cpu', args.output, override)
    batch = transfer(saved['batch'], 'cpu')
    path = CRITIC / ('runs/gate/prepare_cpu_' + args.fixture + '/cpu_old_logprobs.pt')
    batch['logprobs_k'] = torch.load(path, map_location='cpu', weights_only=True)
    if args.worker == 'features':
        features(agent, saved, batch, args.output)
    elif args.worker == 'segments':
        segments(agent, saved, batch, args.output, args.segments)
    else:
        if args.ones:
            # Test-only injection changes time weights, never gamma or MC returns.
            original_prepare = agent._stage3_prepare_batch
            original_before = agent._stage3_before_update
            def before(batch_data):
                agent._stage3_decision_weights.fill_(1.)
                batch_data['decision_weights'] = agent._stage3_decision_weights
                return original_before(batch_data)
            def prepare(*a, **kw):
                result = original_prepare(*a, **kw)
                agent._stage3_decision_weights.fill_(1.)
                return result
            agent._stage3_before_update = before
            agent._stage3_prepare_batch = prepare
        replay(agent, saved, batch, args.source, args.output, override.get('train.stage3_diag', False))
    write(args.output / 'worker.json', {'status': 'passed', 'source': str(args.source), 'cpu_only': True})


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    clock_path = STUDY / 'runs/gate_clock.json'
    if clock_path.exists():
        deadline = json.loads(clock_path.read_text())['deadline_unix']
    else:
        deadline = time.time() + 3600 - 31.02
        write(clock_path, {'deadline_unix': deadline, 'cap_seconds': 3600, 'prior_attempt_seconds': 31.02})
    report = {'status': 'running', 'passed': False, 'reference_commit': 'b459412cfc272a7b8830c77925271843c27222b2',
              'fixture_note': 'Reuses the saved NC4 batch previously used for E1C A1/A3, evaluating mc_critic coefficients; not an original completed-E1C rollout.',
              'cpu_only': True, 'commands': [], 'checks': {}}
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        write(args.output / 'gate_summary.json', report)
    def run(name, mode, source, fixture='nc4', overrides=None, extra=None):
        report['active'] = name
        target = args.output / name
        cmd = [sys.executable, '-B', __file__, '--worker', mode, '--source', str(source),
               '--output', str(target), '--fixture', fixture, '--overrides', json.dumps(overrides or {})] + (extra or [])
        report['commands'].append(cmd)
        save()
        print('START ' + name, flush=True)
        with (args.output / (name + '.out')).open('w') as stream:
            remaining = deadline - time.time()
            require(remaining > 0, 'one-hour CPU gate cap exhausted')
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=min(1800, remaining))
        require(result.returncode == 0, 'Gate worker failed: ' + name)
        return target
    def evidence(path, name='evidence.pt'):
        return torch.load(path / name, map_location='cpu', weights_only=False)
    keys = ('initial_rng', 'initial_model', 'initial_actor_optimizer', 'initial_critic_optimizer',
            'constructor_actor_optimizer', 'constructor_critic_optimizer', 'constructor_actor_scheduler',
            'constructor_critic_scheduler', 'input_batch', 'loss_components', 'actor_gradients', 'critic_gradients',
            'permutations', 'actor_parameters', 'critic_parameters', 'model_state', 'actor_optimizer_state',
            'critic_optimizer_state', 'actor_scheduler_state', 'critic_scheduler_state', 'rng_after_update',
            'diagnostic_generator_state', 'logged_values', 'sampling', 'input_batch_after')
    try:
        require(not torch.cuda.is_available(), 'Hide GPUs for CPU gates')
        feature = run('features', 'features', EDITED, overrides=flags(True, False))
        report['checks'].update(json.loads((feature / 'features.json').read_text()))
        save()
        comparisons = {}
        for name, fixture, overrides in [('default', 'default', {}), ('nc4', 'nc4', {}), ('e1c', 'nc4', flags())]:
            old = run(name + '_old', 'replay', REFERENCE, fixture, overrides)
            new = run(name + '_new', 'replay', EDITED, fixture, overrides)
            a, b = evidence(old), evidence(new)
            checks = {key: compare(a[key], b[key], True, key) for key in keys}
            checks['scientific_stage3_metrics'] = compare(strip_times(a['stage3_metrics']), strip_times(b['stage3_metrics']), True)
            passed = all(c['passed'] for c in checks.values())
            comparisons[name] = {'passed': passed, 'comparisons': checks}
            report['checks']['G5'] = {'passed': all(v['passed'] for v in comparisons.values()), 'cases': comparisons}
            save()
            require(passed, 'G5 bitwise replay failed: ' + name)
            del a, b
        unit = run('unit_time_weights', 'replay', EDITED, overrides=flags(True), extra=['--ones'])
        a, b = evidence(args.output / 'e1c_old'), evidence(unit)
        selected = ('actor_gradients', 'actor_parameters', 'critic_parameters', 'actor_optimizer_state',
                    'critic_optimizer_state', 'rng_after_update', 'permutations')
        checks = {key: compare(a[key], b[key], True, key) for key in selected}
        report['checks']['G3'] = {'passed': all(x['passed'] for x in checks.values()), 'comparisons': checks,
                                'hook': 'test-local decision_weights.fill_(1), gamma and raw MC returns unchanged'}
        save()
        require(report['checks']['G3']['passed'], 'G3 unit time-weight replay differs')
        del a, b
        paths = [run('segments_old', 'segments', REFERENCE, overrides=flags(False, False)),
                 run('segments_off', 'segments', EDITED, overrides=flags(False, False)),
                 run('segments_on', 'segments', EDITED, overrides=flags(False, False), extra=['--segments'])]
        data = [evidence(p, 'segments.pt') for p in paths]
        def existing_only(value):
            if isinstance(value, dict):
                return {k: existing_only(v) for k, v in value.items() if k not in (
                    'reward_segments', 'episode_segment_rewards', 'ckpt_segments', 'ckpt_samplers', 'segment_units')}
            if isinstance(value, list):
                return [existing_only(v) for v in value]
            return value
        checks = {label: compare(strip_times(data[0]), existing_only(strip_times(value)), True)
                  for label, value in zip(('old_to_off', 'old_to_on'), data[1:])}
        report['checks']['G6'] = {'passed': all(x['passed'] for x in checks.values()), 'comparisons': checks,
                'scope': 'CPU checkpoint-evaluation pipeline on identical saved rollout inputs, actual saturation diagnostics, no new rollouts; timings excluded'}
        save()
        require(report['checks']['G6']['passed'], 'G6 existing outputs or RNG changed')
        report['passed'] = all(report['checks'][k]['passed'] for k in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'C1'))
        report['status'] = 'passed' if report['passed'] else 'failed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['features', 'replay', 'segments'])
    parser.add_argument('--source', type=Path, default=EDITED)
    parser.add_argument('--output', type=Path, default=STUDY / 'runs/gates')
    parser.add_argument('--fixture', default='nc4')
    parser.add_argument('--overrides', default='{}')
    parser.add_argument('--ones', action='store_true')
    parser.add_argument('--segments', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.worker:
        worker(args)
    else:
        main(args)
