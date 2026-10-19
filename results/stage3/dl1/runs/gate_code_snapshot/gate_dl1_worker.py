#!/usr/bin/env python
"""Isolated DL1 CPU replay/features and live G4; Python 3.8.

Each worker imports exactly one scientific source tree. The native update and
GAE blocks are extracted without modifying their statements. Prior fixture and
gate helpers are read only. No scheduler jobs are submitted by this module.
"""
import argparse
import ast
import copy
import importlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback

import numpy as np
import torch

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
STUDY = PROJECT / 'results/stage3/dl1'
FIXTURE = PROJECT / 'results/stage3/runs/gate_capture/default/training_batch.pt'
sys.path.append(str(PROJECT / 'results/stage3/critic/code'))
from gate_common import assignment_to, compare, freeze, numeric_settings, require, restore_rng, rng_state, transfer
from gate_features import brute_returns, error
from gate_replay import sampling

FIELDS = ('obs_k', 'chains_k', 'returns_k', 'values_k', 'advantages_k', 'logprobs_k')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def close(agent, output, primary_error=None):
    """Explicit native vector-env close and child joins, also for cc7234a."""
    if agent is None or not hasattr(agent, 'venv'):
        return {'status': 'passed', 'all_children_exited': True, 'workers': []}
    workers = list(getattr(agent.venv, 'processes', []))
    failure = None
    try:
        agent.venv.close()
    except BaseException as exc:
        failure = exc
    for process in workers:
        process.join(timeout=5)
    remaining = [p.pid for p in workers if p.is_alive()]
    remaining += [p.pid for p in mp.active_children() if p.pid not in remaining]
    record = {'status': 'passed' if failure is None and not remaining else 'failed',
              'workers': [p.pid for p in workers], 'all_children_exited': not remaining,
              'remaining_children': remaining, 'primary_error': repr(primary_error) if primary_error else None,
              'close_error': repr(failure) if failure else None}
    write(output / 'cleanup.json', record)
    require(record['status'] == 'passed', 'Explicit environment cleanup failed: ' + repr(record))
    return record


def overrides(profile):
    if profile == 'default':
        return {}
    if profile in ('dl1_off', 'dl1_on'):
        return {'train.adv_estimator': 'mc_critic', 'train.baseline_diag': profile == 'dl1_on'}
    if profile == 'e1c':
        return {'train.adv_estimator': 'mc_critic', 'train.baseline_diag': True,
                'train.actor_single_step': True, 'model.actor_loss': 'score',
                'model.clip_ploss_coef': 1e6, 'model.clip_ploss_coef_base': 1e6,
                'train.target_kl': None, 'model.clamp_logprob': False,
                'model.randn_clip_value': 100, 'train.stage3_diag': False}
    raise ValueError(profile)


def construct(source, output, profile, device='cpu', live=False):
    sys.path.insert(0, str(source))
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    saved = torch.load(FIXTURE, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(saved['config'])
    cfg.device = device
    cfg.model.device = device
    cfg.logdir = str(output / 'native')
    for key, value in overrides(profile).items():
        OmegaConf.update(cfg, key, value, force_add=True)
    if live:
        cfg.train.n_train_itr = 2
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    cls = get_class(cfg._target_)
    agent = cls.__new__(cls)
    try:
        cls.__init__(agent, cfg)
    except BaseException as exc:
        close(agent, output, exc)
        raise
    if live:
        return agent, saved, cfg
    close(agent, output)
    agent.model.load_state_dict(saved['model_state'])
    for name, module in agent.model.named_modules():
        module.training = saved['module_training'][name]
    agent.actor_optimizer.load_state_dict(copy.deepcopy(saved['actor_optimizer_state']))
    agent.critic_optimizer.load_state_dict(copy.deepcopy(saved['critic_optimizer_state']))
    agent.actor_lr_scheduler.load_state_dict(copy.deepcopy(saved['actor_scheduler_state']))
    agent.critic_lr_scheduler.load_state_dict(copy.deepcopy(saved['critic_scheduler_state']))
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_after']))
    for key, value in saved['attrs'].items():
        if 'train.' + key not in overrides(profile):
            setattr(agent, key, value)
    if hasattr(agent, 'diagnostic_generator'):
        agent.diagnostic_generator.set_state(saved['diagnostic_generator_state'])
    return agent, saved, cfg


def native_nodes(source):
    filename = source / 'agent/finetune/train_ppo_diffusion_agent.py'
    tree = ast.parse(filename.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in run.body if isinstance(n, ast.While))
    update = next(n for n in loop.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.operand, ast.Name)
                  and n.test.operand.id == 'eval_mode')
    return filename, loop, update


def compile_block(source, name, arguments, body):
    module = importlib.import_module('agent.finetune.train_ppo_diffusion_agent')
    function = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=x) for x in arguments], vararg=None, kwonlyargs=[],
        kw_defaults=[], kwarg=None, defaults=[]), body=copy.deepcopy(body), decorator_list=[])
    function.body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])))
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(module.__dict__)
    exec(compile(program, '<dl1-native-' + name + '>', 'exec'), namespace)
    return namespace[name]


def native_gae(source, agent, saved):
    _, _, update = native_nodes(source)
    no_grad = next(n for n in update.body if isinstance(n, ast.With))
    start = next(i for i, n in enumerate(no_grad.body) if assignment_to(n, 'obs_venv_ts'))
    end = next(i for i, n in enumerate(no_grad.body) if assignment_to(n, 'returns_trajs'))
    block = compile_block(source, 'gae', ('self', 'obs_venv', 'reward_trajs', 'terminated_trajs', 'values_trajs'),
                          no_grad.body[start:end + 1])
    with torch.no_grad():
        local = block(agent, copy.deepcopy(saved['final_observation']), saved['scaled_reward_trajs'].copy(),
                      saved['terminated_trajs'].copy(), saved['values_trajs'].copy())
    returns = torch.as_tensor(local['returns_trajs'], dtype=torch.float32).reshape(-1)
    require(agent.gae_lambda == .95, 'Critic GAE lambda changed')
    check = compare(saved['batch']['returns_k'], returns, True)
    require(check['passed'], 'Native GAE targets differ from saved DPPO targets')
    return {'returns_trajs': local['returns_trajs'], 'advantages_trajs': local['advantages_trajs'],
            'returns_k': returns, 'saved_targets_comparison': check,
            'source_ast': ast.dump(ast.Module(body=no_grad.body[start:end + 1], type_ignores=[]), include_attributes=False)}


def prepare(agent, saved, batch):
    if not getattr(agent, 'stage3_active', False):
        return {}
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_before']))
    agent._stage3_iteration_start()
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_after']))
    agent._stage3_actions = saved['action_trajs'].copy()
    agent._stage3_rollout_finished(saved['raw_reward_trajs'], saved['firsts_trajs'], False)
    before = rng_state()
    values_before, returns_before = batch['values_k'].clone(), batch['returns_k'].clone()
    batch['advantages_k'] = agent._stage3_prepare_batch(saved['raw_reward_trajs'], saved['firsts_trajs'],
        saved['terminated_trajs'], saved['action_trajs'], batch['obs_k'], batch['chains_k'],
        batch['returns_k'], batch['values_k'], batch['advantages_k'], batch['logprobs_k'], saved['scaled_reward_trajs'])
    checks = {'rng': compare(before, rng_state(), True), 'values': compare(values_before, batch['values_k'], True),
              'returns': compare(returns_before, batch['returns_k'], True)}
    require(all(v['passed'] for v in checks.values()), 'Actor preparation altered targets, values or global RNG')
    return {'raw_advantages': freeze(agent._critic_baseline_raw_adv),
            'divisor': freeze(agent._critic_baseline_divisor),
            'actor_advantages': batch['advantages_k'].clone(), 'inert_checks': checks}


def features(source, output, profile):
    agent, saved, _ = construct(source, output, profile)
    batch = transfer(saved['batch'], 'cpu')
    restore_rng(saved['rng'])
    actual = prepare(agent, saved, batch)
    returns = brute_returns(saved['raw_reward_trajs'], saved['firsts_trajs'], agent.gamma)
    divisor = np.sqrt(np.asarray(saved['reward_scaler_before']['ret_rms'].var, dtype=np.float64)
                      + saved['reward_scaler_before']['epsilon'])
    expected = returns - divisor * saved['values_trajs'].astype(np.float64)
    metrics = error(expected, actual['raw_advantages'])
    cast = compare(torch.as_tensor(expected, dtype=torch.float32).reshape(-1), actual['actor_advantages'], True)
    # Brute forward sum and native backward recursion can round at float64.
    # Native cast must match its own float64 residual, not a different summation order.
    native_cast = compare(torch.as_tensor(actual['raw_advantages'], dtype=torch.float32).reshape(-1),
                          actual['actor_advantages'], True)
    require(metrics['relative_l2'] <= 1e-12 and native_cast['passed'], 'G2 advantage formula/cast failed')
    require(not agent.decision_discount and agent.model.norm_adv, 'Wrong DL1 advantage normalization/discount')
    torch.save(actual, output / 'features.pt')
    write(output / 'features.json', {'passed': True, 'raw_comparison': metrics,
          'native_float32_cast': native_cast, 'bruteforce_float32_cast_diagnostic': cast,
          'normalization': 'unchanged native per-minibatch normalization', 'divisor': float(divisor),
          'profile': profile, 'pairs': 200000})


def replay(source, output, profile):
    agent, saved, _ = construct(source, output, profile)
    batch = transfer(saved['batch'], 'cpu')
    evidence = {'loss_components': [], 'actor_gradients': [], 'critic_gradients': [], 'permutations': []}
    evidence['native_gae'] = native_gae(source, agent, saved)
    restore_rng(saved['rng'])
    evidence['initial_rng'] = rng_state()
    evidence['initial_model'] = freeze(agent.model.state_dict())
    evidence['initial_actor_optimizer'] = freeze(agent.actor_optimizer.state_dict())
    evidence['initial_critic_optimizer'] = freeze(agent.critic_optimizer.state_dict())
    evidence['input_batch'] = freeze(batch)
    evidence['preparation'] = prepare(agent, saved, batch)
    evidence['prepared_batch'] = freeze(batch)
    before_losses, before_actor, before_critic = agent.model.loss, agent.actor_optimizer.step, agent.critic_optimizer.step
    def loss(*args, **kwargs):
        result = before_losses(*args, **kwargs)
        evidence['loss_components'].append(freeze(result))
        return result
    def actor(*args, **kwargs):
        evidence['actor_gradients'].append(freeze({name: p.grad for name, p in agent.model.actor_ft.named_parameters()}))
        return before_actor(*args, **kwargs)
    def critic(*args, **kwargs):
        evidence['critic_gradients'].append(freeze({name: p.grad for name, p in agent.model.critic.named_parameters()}))
        result = before_critic(*args, **kwargs)
        if len(evidence['critic_gradients']) == 1:
            evidence['critic_first_step_parameters'] = freeze(agent.model.critic.state_dict())
            evidence['critic_first_step_optimizer'] = freeze(agent.critic_optimizer.state_dict())
        return result
    agent.model.loss, agent.actor_optimizer.step, agent.critic_optimizer.step = loss, actor, critic
    native_perm = torch.randperm
    def permutation(*args, **kwargs):
        result = native_perm(*args, **kwargs)
        if kwargs.get('generator') is None:
            evidence['permutations'].append(freeze(result))
        return result
    _, loop, update = native_nodes(source)
    start = next(i for i, node in enumerate(update.body) if assignment_to(node, 'total_steps'))
    block = compile_block(source, 'update', ('self',) + FIELDS, update.body[start:])
    try:
        torch.randperm = permutation
        local = block(agent, **batch)
    finally:
        torch.randperm = native_perm
    evidence['rng_after_update'] = rng_state()
    # Include the original end-of-iteration scheduler/model transition, before save/log.
    end_start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute)
                     and n.test.left.attr == 'itr' and any(isinstance(x, ast.Attribute) and x.attr == 'n_critic_warmup_itr' for x in n.test.comparators))
    end_stop = next(i for i in range(end_start, len(loop.body)) if assignment_to(loop.body[i], 'diffusion_min_sampling_std'))
    finish = compile_block(source, 'finish_iteration', ('self',), loop.body[end_start:end_stop + 1])
    finish(agent)
    evidence.update(actor_parameters=freeze(dict(agent.model.actor_ft.named_parameters())),
        critic_parameters=freeze(dict(agent.model.critic.named_parameters())), model_state=freeze(agent.model.state_dict()),
        actor_optimizer_state=freeze(agent.actor_optimizer.state_dict()), critic_optimizer_state=freeze(agent.critic_optimizer.state_dict()),
        actor_scheduler_state=freeze(agent.actor_lr_scheduler.state_dict()), critic_scheduler_state=freeze(agent.critic_lr_scheduler.state_dict()),
        rng_after_iteration=rng_state(), input_batch_after=freeze(batch))
    common = ('loss', 'pg_loss', 'entropy_loss', 'v_loss', 'clipfracs', 'approx_kl', 'ratio', 'bc_loss', 'eta', 'explained_var')
    evidence['logged_values'] = {k: freeze(local[k]) for k in common}
    evidence['stage3_metrics'] = freeze(getattr(agent, '_stage3_metrics', {}))
    evidence['sampling'] = sampling(agent, saved, 'cpu')
    require(len(evidence['actor_gradients']) == len(evidence['critic_gradients']) == 20, 'CPU replay did not execute 20 actor/critic steps')
    require(len(evidence['permutations']) == 5, 'CPU replay did not shuffle five times')
    torch.save(evidence, output / 'evidence.pt')
    return {'passed': True, 'actor_steps': 20, 'critic_steps': 20, 'epochs': 5, 'pairs_per_minibatch': agent.batch_size,
            'profile': profile, 'fixed_inputs': str(FIXTURE), 'native_loop': str(source / 'agent/finetune/train_ppo_diffusion_agent.py')}


def live(source, output):
    require(torch.cuda.is_available(), 'G4 requires an allocated GPU')
    agent = None
    primary = None
    observed = {'actor_steps': [], 'critic_steps': 0, 'loss_sizes': [], 'baseline_preparations': 0}
    try:
        agent, _, cfg = construct(source, output, 'dl1_on', 'cuda:0', live=True)
        require(agent.max_grad_norm is None and not agent.actor_single_step and agent.target_kl == 1,
                'G4 DPPO loop/clipping configuration changed')
        require(agent.model.actor_loss == 'ratio' and agent.model.clamp_logprob and agent.model.randn_clip_value == 3,
                'G4 must use the DPPO skeleton')
        old_loss, old_actor, old_critic, old_prepare = agent.model.loss, agent.actor_optimizer.step, agent.critic_optimizer.step, agent._stage3_prepare_batch
        def loss(*args, **kwargs):
            observed['loss_sizes'].append(int(len(args[6])))
            return old_loss(*args, **kwargs)
        def actor(*args, **kwargs):
            observed['actor_steps'].append(int(agent.itr))
            return old_actor(*args, **kwargs)
        def critic(*args, **kwargs):
            observed['critic_steps'] += 1
            return old_critic(*args, **kwargs)
        def prepare_once(*args, **kwargs):
            observed['baseline_preparations'] += 1
            return old_prepare(*args, **kwargs)
        agent.model.loss, agent.actor_optimizer.step, agent.critic_optimizer.step, agent._stage3_prepare_batch = loss, actor, critic, prepare_once
        agent.run()
        with (Path(cfg.logdir) / 'result.pkl').open('rb') as stream:
            rows = pickle.load(stream)
        require([r['itr'] for r in rows] == [0, 1], 'G4 live run is not eval0 plus train1')
        row = rows[1]
        steps = row['ppo_steps']
        require(len(steps) == row['ppo_step_count'] == len(observed['actor_steps']) == observed['critic_steps'] == 20,
                'G4 live iteration must execute exactly 20 actor/critic steps')
        require(observed['baseline_preparations'] == 1, 'G4 actor advantage was recomputed inside the update loop')
        require(observed['loss_sizes'] == [50000] * 20, 'G4 minibatch sizes differ')
        require(all(s['minibatch_size'] == 50000 for s in steps), 'G4 recorded minibatch sizes differ')
        require(all(s['mean_abs_ratio_minus_one'] > 0 for s in steps[1:]), 'G4 reused steps show no ratio movement')
        for index, step in enumerate(steps):
            require(step['step'] == index + 1 and step['epoch'] == index // 4 and step['minibatch'] == index % 4,
                    'G4 observer step/epoch/minibatch ordering differs')
            for key in ('clipfrac', 'approx_kl', 'switched_off_frac', 'mean_abs_ratio_minus_one'):
                require(np.isfinite(step[key]), 'G4 observer has nonfinite ' + key)
        record = {'passed': True, 'observed': observed, 'steps': steps, 'reset_at_iteration': bool(agent.reset_at_iteration),
                  'max_grad_norm': agent.max_grad_norm, 'train_result': row, 'eval_result': rows[0],
                  'gpu': torch.cuda.get_device_name(0), 'config': str(output / 'config.yaml'),
                  'sampler_noise_clip': agent.model.randn_clip_value, 'target_kl': agent.target_kl}
        write(output / 'G4.json', record)
        return {'passed': True, 'actor_steps': 20, 'critic_steps': 20, 'minibatch_size': 50000,
                'baseline_preparations': 1, 'record': str(output / 'G4.json')}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if agent is not None:
            close(agent, output, primary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('replay', 'features', 'live'), required=True)
    parser.add_argument('--profile', default='default', choices=('default', 'dl1_off', 'dl1_on', 'e1c'))
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(not args.output.exists(), 'Refuse to overwrite gate evidence')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(int(os.environ.get('DL1_GATE_THREADS', '2')))
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(InterruptedError('Gate terminated')))
    started = time.monotonic()
    report = {'passed': False, 'status': 'running', 'kind': args.kind, 'profile': args.profile,
              'source': str(args.source), 'command': [sys.executable] + sys.argv,
              'numeric_settings_before': numeric_settings(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    write(args.output / 'worker.json', report)
    try:
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        if args.kind != 'live':
            require(not torch.cuda.is_available(), 'CPU gates require CUDA_VISIBLE_DEVICES empty')
        if args.kind == 'features':
            features(args.source.resolve(), args.output, args.profile)
            result = {'passed': True}
        elif args.kind == 'replay':
            result = replay(args.source.resolve(), args.output, args.profile)
        else:
            result = live(args.source.resolve(), args.output)
        require(not mp.active_children(), 'Worker left child processes alive')
        report.update(result, status='passed', all_children_exited=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, numeric_settings_after=numeric_settings())
        write(args.output / 'worker.json', report)


if __name__ == '__main__':
    main()
