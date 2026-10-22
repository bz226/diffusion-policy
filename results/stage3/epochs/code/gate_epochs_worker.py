#!/usr/bin/env python
"""Isolated EP CPU replay, learning-rate checks and live G3; Python 3.8.

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
STUDY = PROJECT / 'results/stage3/epochs'
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


PROFILES = {
    "EP1": {"train.actor_update_epochs": 1},
    "EP1M": {"train.actor_update_epochs": 1, "train.actor_lr": 5e-4,
             "train.actor_lr_scheduler.min_lr": 5e-4},
    "EP2": {"train.actor_update_epochs": 2},
    "EP2M": {"train.actor_update_epochs": 2, "train.actor_lr": 2.5e-4,
             "train.actor_lr_scheduler.min_lr": 2.5e-4},
}


def overrides(profile):
    return {} if profile == "default" else dict(PROFILES[profile])


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
    exec(compile(program, '<epochs-native-' + name + '>', 'exec'), namespace)
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
    return {"actor_advantages": batch["advantages_k"].clone(), "inert_checks": checks}


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
    expected_actor_steps = 4 * overrides(profile).get('train.actor_update_epochs', 5)
    require(len(evidence['actor_gradients']) == expected_actor_steps, 'CPU replay actor-step count differs')
    require(len(evidence['critic_gradients']) == 20, 'CPU replay critic-step count differs')
    require(len(evidence['permutations']) == 5, 'CPU replay did not shuffle five times')
    torch.save(evidence, output / 'evidence.pt')
    return {'passed': True, 'actor_steps': expected_actor_steps, 'critic_steps': 20, 'epochs': 5, 'pairs_per_minibatch': agent.batch_size,
            'profile': profile, 'fixed_inputs': str(FIXTURE), 'native_loop': str(source / 'agent/finetune/train_ppo_diffusion_agent.py')}


def adam_counts(optimizer):
    return [int(state["step"].item()) if torch.is_tensor(state["step"]) else int(state["step"])
            for state in optimizer.state.values() if "step" in state]


def learning_rates(source, output, profile):
    """Exercise the native scheduler and LR assertion at all 140 positions."""
    agent, _, cfg = construct(source, output, profile, "cpu", live=True)
    close(agent, output)
    expected = float(cfg.train.actor_lr)
    rows = []
    before = rng_state()
    for itr in range(140):
        agent.itr = itr
        agent._stage3_assert_lr()
        before_lr = [float(g["lr"]) for g in agent.actor_optimizer.param_groups]
        if itr >= agent.n_critic_warmup_itr:
            agent.actor_lr_scheduler.step()
        agent._stage3_assert_lr()
        after_lr = [float(g["lr"]) for g in agent.actor_optimizer.param_groups]
        require(all(x == expected for x in before_lr + after_lr),
                "G5 lr differs at iteration {}: {} -> {}".format(itr, before_lr, after_lr))
        rows.append({"itr": itr, "before": before_lr, "after": after_lr})
    rng_check = compare(before, rng_state(), True)
    require(rng_check["passed"], "Scheduler check unexpectedly changed RNG")
    record = {"passed": True, "profile": profile, "actor_lr": expected,
              "iterations": rows, "rng": rng_check,
              "scope": "native scheduler and assertion at each of 140 positions, no rollouts or gradients"}
    write(output / "G5.json", record)
    return {"passed": True, "record": str(output / "G5.json"), "actor_lr": expected}


def live(source, output, profile):
    require(torch.cuda.is_available(), 'G3 requires an allocated GPU')
    agent = None
    primary = None
    observed = {'actor_steps': [], 'critic_steps': 0, 'loss_sizes': [], 'permutations': 0}
    original_randperm = torch.randperm
    try:
        agent, _, cfg = construct(source, output, profile, 'cuda:0', live=True)
        k = int(cfg.train.actor_update_epochs)
        expected_actor_steps = 4 * k
        expected_lr = float(cfg.train.actor_lr)
        require(agent.max_grad_norm is None and not agent.actor_single_step and agent.target_kl == 1,
                'G3 DPPO loop/clipping configuration changed')
        require(agent.model.actor_loss == 'ratio' and agent.model.clamp_logprob and agent.model.randn_clip_value == 3,
                'G3 must use the DPPO skeleton')
        require(agent.adv_estimator == 'gae' and not agent.baseline_diag,
                'G3 must keep Stage-1 advantages and only the approved epoch flag')
        old_loss, old_actor, old_critic = agent.model.loss, agent.actor_optimizer.step, agent.critic_optimizer.step
        initial_actor_counts = adam_counts(agent.actor_optimizer)
        initial_critic_counts = adam_counts(agent.critic_optimizer)
        def loss(*args, **kwargs):
            observed['loss_sizes'].append(int(len(args[6])))
            return old_loss(*args, **kwargs)
        def actor(*args, **kwargs):
            observed['actor_steps'].append(int(agent.itr))
            return old_actor(*args, **kwargs)
        def critic(*args, **kwargs):
            observed['critic_steps'] += 1
            return old_critic(*args, **kwargs)
        def permutation(*args, **kwargs):
            if kwargs.get('generator') is None and args and int(args[0]) == 200000:
                observed['permutations'] += 1
            return original_randperm(*args, **kwargs)
        agent.model.loss, agent.actor_optimizer.step, agent.critic_optimizer.step = loss, actor, critic
        torch.randperm = permutation
        agent.run()
        with (Path(cfg.logdir) / 'result.pkl').open('rb') as stream:
            rows = pickle.load(stream)
        require([r['itr'] for r in rows] == [0, 1], 'G3 live run must be eval0 plus train1')
        row = rows[1]
        steps = row['ppo_steps']
        require(len(steps) == row['ppo_step_count'] == len(observed['actor_steps']) == expected_actor_steps,
                'G3 actor must execute exactly 4k steps')
        require(observed['critic_steps'] == row['critic_optimizer_steps'] == 20,
                'G3 critic must execute exactly 20 steps')
        require(observed['loss_sizes'] == [50000] * 20 and observed['permutations'] == 5,
                'G3 must retain every forward/backward and all five shuffles')
        actor_counts, critic_counts = adam_counts(agent.actor_optimizer), adam_counts(agent.critic_optimizer)
        require(bool(actor_counts) and all(n == expected_actor_steps for n in actor_counts),
                'G3 actor AdamW state counters differ from 4k')
        require(bool(critic_counts) and all(n == 20 for n in critic_counts),
                'G3 critic AdamW state counters differ from20')
        require(not initial_actor_counts and not initial_critic_counts,
                'G3 live initialization unexpectedly has optimizer moments')
        require(row['actor_optimizer_steps'] == expected_actor_steps,
                'G3 logged actor steps differ from observer')
        require(row['actor_adam_step_count'] == expected_actor_steps,
                'G3 logged AdamW state count differs')
        for index, step in enumerate(steps):
            require(step['step'] == index + 1 and step['epoch'] == index // 4 and step['minibatch'] == index % 4,
                    'G3 recorded actor-step ordering differs')
            for key in ('clipfrac', 'approx_kl', 'switched_off_frac', 'mean_abs_ratio_minus_one'):
                require(np.isfinite(step[key]), 'G3 observer has nonfinite ' + key)
                require(row['ppo_' + key + '_mean'] == float(np.mean([s[key] for s in steps])),
                        'G3 actor-only mean differs: ' + key)
                require(row['ppo_' + key + '_last'] == steps[-1][key],
                        'G3 last actor-step metric differs: ' + key)
        require(all(r['actor_lr'] == expected_lr for r in rows), 'G5 live logged learning rate differs')
        record = {'passed': True, 'profile': profile, 'observed': observed, 'steps': steps,
                  'actor_adam_counts': actor_counts, 'critic_adam_counts': critic_counts,
                  'reset_at_iteration': bool(agent.reset_at_iteration), 'max_grad_norm': agent.max_grad_norm,
                  'train_result': row, 'eval_result': rows[0], 'gpu': torch.cuda.get_device_name(0),
                  'config': str(output / 'config.yaml'), 'sampler_noise_clip': agent.model.randn_clip_value,
                  'target_kl': agent.target_kl, 'actor_lr': expected_lr,
                  'G5_live_logged_lr_passed': True}
        write(output / 'G3.json', record)
        return {'passed': True, 'actor_steps': expected_actor_steps, 'critic_steps': 20,
                'minibatch_size': 50000, 'record': str(output / 'G3.json')}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        torch.randperm = original_randperm
        if agent is not None:
            close(agent, output, primary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('replay', 'learning_rates', 'live'), required=True)
    parser.add_argument('--profile', default='default', choices=('default', 'EP1', 'EP1M', 'EP2', 'EP2M'))
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(not args.output.exists(), 'Refuse to overwrite gate evidence')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(int(os.environ.get('EP_GATE_THREADS', '2')))
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
        if args.kind == 'learning_rates':
            result = learning_rates(args.source.resolve(), args.output, args.profile)
        elif args.kind == 'replay':
            result = replay(args.source.resolve(), args.output, args.profile)
        else:
            result = live(args.source.resolve(), args.output, args.profile)
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
