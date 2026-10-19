#!/usr/bin/env python
"""Isolated full-batch CPU identity checks; never submit jobs (Python 3.8)."""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
FIXTURE = PROJECT / 'results/stage3/runs/gate_capture/nc4/training_batch.pt'
sys.path.append(str(PROJECT / 'results/stage3/dl1/code'))
import gate_dl1_worker as native
from gate_common import compare, freeze, numeric_settings, require, restore_rng, rng_state, transfer
from gate_replay import prepare_cpu


class CapturedActorGradient(Exception):
    """Test-only stop immediately before the native actor optimizer step."""


def flat_gradient(agent):
    parameters = list(agent.model.actor_ft.parameters())
    require(all(p.grad is not None for p in parameters), 'An actor parameter has no gradient')
    return torch.cat([p.grad.detach().reshape(-1) for p in parameters]).cpu().clone()


def relative_error(reference, candidate):
    a, b = reference.double(), candidate.double()
    delta = b - a
    denominator = torch.linalg.vector_norm(a).item()
    require(denominator > 0, 'Identity reference gradient has zero norm')
    return {'relative_l2': torch.linalg.vector_norm(delta).item() / denominator,
            'max_abs': delta.abs().max().item(),
            'reference_norm': denominator,
            'measured_norm': torch.linalg.vector_norm(b).item(),
            'bitwise_equal': bool(torch.equal(reference, candidate))}


def construct(args):
    native.FIXTURE = args.fixture
    # The isolated measurement source intentionally keeps its historical
    # training defaults. The new MC implementation is a separate entrypoint,
    # so only the native MC reference enables the E1C training flags.
    profile = 'e1c' if args.estimator == 'V2' and args.kind == 'reference' else 'default'
    agent, saved, cfg = native.construct(args.source, args.output, profile)
    require(agent.actor_single_step and agent.batch_size == 50000 and agent.update_epochs == 5,
            'Identity requires one native actor step, five epochs and 50000-pair chunks')
    require(agent.max_grad_norm is None and agent.ent_coef == 0 and not agent.use_bc_loss,
            'Unexpected extra actor-gradient terms or gradient clipping')
    require(agent.model.ft_denoising_steps == 10 and agent.n_steps == 500 and agent.n_envs == 40,
            'The native saved batch must contain 200000 pairs')
    return agent, saved, cfg


def reference(args):
    agent, saved, cfg = construct(args)
    if args.cpu_logprobs is None:
        # GPU-saved probabilities cannot enforce the native CPU ratio=1 check.
        # Recompute theta_old probabilities once with the native CPU chunk
        # shapes. Every comparison consumes this exact same saved tensor.
        prepare_cpu(agent, saved, args.output)
        old_path = args.output / 'cpu_old_logprobs.pt'
    else:
        old_path = args.cpu_logprobs
    batch = transfer(saved['batch'], 'cpu')
    batch['logprobs_k'] = torch.load(old_path, map_location='cpu', weights_only=False)
    prepared = native.prepare(agent, saved, batch)
    observed = {'permutations': [], 'critic_steps': 0, 'loss_sizes': [], 'actor_steps': 0}
    original_loss = agent.model.loss
    original_critic_step = agent.critic_optimizer.step
    original_randperm = torch.randperm
    def loss(*positional, **keyword):
        observed['loss_sizes'].append(int(len(positional[6])))
        return original_loss(*positional, **keyword)
    def critic_step(*positional, **keyword):
        observed['critic_steps'] += 1
        return original_critic_step(*positional, **keyword)
    def actor_step(*positional, **keyword):
        observed['gradient'] = flat_gradient(agent)
        observed['actor_steps'] += 1
        raise CapturedActorGradient()
    def randperm(*positional, **keyword):
        value = original_randperm(*positional, **keyword)
        if keyword.get('generator') is None:
            observed['permutations'].append(value.detach().cpu().clone())
        return value
    agent.model.loss = loss
    agent.critic_optimizer.step = critic_step
    agent.actor_optimizer.step = actor_step
    _, _, update = native.native_nodes(args.source)
    start = next(i for i, node in enumerate(update.body) if native.assignment_to(node, 'total_steps'))
    block = native.compile_block(args.source, 'capture_before_optimizer', ('self',) + native.FIELDS,
                                 update.body[start:])
    restore_rng(saved['rng'])
    before_actor = freeze(agent.model.actor_ft.state_dict())
    try:
        torch.randperm = randperm
        block(agent, **batch)
        raise AssertionError('Native actor optimizer step was never reached')
    except CapturedActorGradient:
        pass
    finally:
        torch.randperm = original_randperm
    require(observed['critic_steps'] == 20 and observed['actor_steps'] == 1,
            'Did not execute the full native 20-step critic loop and single actor step')
    require(len(observed['permutations']) == 5 and observed['loss_sizes'] == [50000] * 24,
            'Native shuffles or minibatch shapes differ from the intended run')
    require(compare(before_actor, agent.model.actor_ft.state_dict(), True)['passed'],
            'Reference actor changed before its optimizer step')
    # Only the captured gradient and compact identity input are retained. There
    # is no new rollout or training output in this CPU verification.
    evidence = {'gradient': observed.pop('gradient'), 'permutation': observed['permutations'][-1],
                'advantages': batch['advantages_k'].clone(), 'preparation': prepared,
                'input_old_logprobs': str(old_path), 'observed': observed,
                'native_source': str(args.source), 'fixture': str(args.fixture),
                'normalization': 'native independent 50000-pair chunk statistics'}
    torch.save(evidence, args.output / 'identity.pt')
    return {'passed': True, 'native_actor_gradient_captured_before_step': True,
            'critic_steps': observed['critic_steps'], 'loss_calls': len(observed['loss_sizes']),
            'pairs': 200000, 'old_logprobs': str(old_path), 'estimator': args.estimator}


def measured(args):
    require(args.reference is not None, 'Measured worker needs native reference evidence')
    agent, saved, cfg = construct(args)
    from agent.finetune.noise_without_ppo import native_gradient, mc_critic_advantages, gae_advantages
    reference = torch.load(args.reference, map_location='cpu', weights_only=False)
    batch = transfer(saved['batch'], 'cpu')
    batch['logprobs_k'] = torch.load(reference['input_old_logprobs'], map_location='cpu', weights_only=False)
    batch.update(raw_rewards=saved['raw_reward_trajs'].copy(), firsts=saved['firsts_trajs'].copy(),
                 values=batch['values_k'].double().numpy().reshape(500, 40),
                 gae_adv=saved['advantages_trajs'].copy())
    if args.estimator == 'V1':
        from util.reward_scaling import RunningRewardScaler
        scaler = RunningRewardScaler(agent.n_envs)
        scaler.__dict__.update(copy.deepcopy(saved['reward_scaler_before']))
        scaled = scaler(reward=saved['raw_reward_trajs'].T.copy(),
                        first=saved['firsts_trajs'][:-1].T.copy()).T
        def scaler_dict(state):
            result = copy.deepcopy(state)
            result['ret_rms'] = vars(result['ret_rms'])
            return result
        scaler_checks = {
            'scaled_rewards': compare(saved['scaled_reward_trajs'], scaled, True),
            'state_after_current_batch': compare(scaler_dict(saved['reward_scaler_after']),
                                               scaler_dict(vars(scaler)), True)}
        require(all(v['passed'] for v in scaler_checks.values()),
                'Native restored running scaler does not reproduce the saved batch')
        raw_adv = gae_advantages(agent, saved['scaled_reward_trajs'].copy(),
            saved['values_trajs'].copy(), saved['terminated_trajs'].copy(),
            copy.deepcopy(saved['final_observation']))
        advantages = torch.as_tensor(raw_adv, dtype=torch.float32).reshape(-1)
        require(compare(advantages, saved['batch']['advantages_k'], True)['passed'],
                'Measured GAE preprocessing differs from the native saved targets')
    else:
        scale = saved['reward_scaler_before']
        divisor = np.sqrt(np.asarray(scale['ret_rms'].var, dtype=np.float64) + scale['epsilon'])
        raw_adv = mc_critic_advantages(batch, agent.gamma, divisor)
        advantages = torch.as_tensor(raw_adv, dtype=torch.float32).reshape(-1)
    advantage_check = compare(reference['advantages'], advantages, True)
    require(advantage_check['passed'], 'Measured/native advantages differ before normalization')
    restore_rng(saved['rng'])
    before_rng = rng_state()
    before_model = freeze(agent.model.state_dict())
    before_optim = freeze({'actor': agent.actor_optimizer.state_dict(), 'critic': agent.critic_optimizer.state_dict()})
    gradient, metrics = native_gradient(agent, batch, advantages, reference['permutation'],
                                       actor_loss='score' if args.estimator == 'V2' else 'ratio')
    gradient = gradient.detach().cpu()
    error = relative_error(reference['gradient'], gradient)
    require(error['relative_l2'] <= 1e-6,
            'G%s measured actor gradient fails relative L2 <= 1e-6: %r' %
            ('1' if args.estimator == 'V1' else '2', error))
    inert = {'model': compare(before_model, agent.model.state_dict(), True),
             'optimizer': compare(before_optim, {'actor': agent.actor_optimizer.state_dict(),
                                               'critic': agent.critic_optimizer.state_dict()}, True),
             'global_rng': compare(before_rng, rng_state(), True)}
    require(all(x['passed'] for x in inert.values()), 'Measurement changed model, optimizer or global RNG')
    record = {'passed': True, 'gradient_comparison': error, 'advantages_bitwise': advantage_check,
              'inert_checks': inert, 'metrics': metrics, 'estimator': args.estimator,
              'source': str(args.source), 'reference': str(args.reference), 'pairs': 200000,
              'normalization': 'four independently normalized native 50000-pair chunks'}
    if args.estimator == 'V1':
        record['running_scaler_saved_batch_checks'] = scaler_checks
    native.write(args.output / 'identity.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=FIXTURE)
    parser.add_argument('--kind', choices=('reference', 'measured'), required=True)
    parser.add_argument('--estimator', choices=('V1', 'V2'), required=True)
    parser.add_argument('--cpu-logprobs', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    require(not args.output.exists(), 'Refuse to overwrite gate output')
    args.output.mkdir(parents=True)
    torch.set_num_threads(int(os.environ.get('NOISE_GATE_THREADS', '8')))
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(InterruptedError('Gate terminated')))
    start = time.monotonic()
    record = {'passed': False, 'status': 'running', 'command': [sys.executable] + sys.argv,
              'numeric_settings': numeric_settings(), 'source': str(args.source), 'kind': args.kind,
              'estimator': args.estimator, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    native.write(args.output / 'worker.json', record)
    try:
        require(not torch.cuda.is_available(), 'CPU identity gate must hide GPUs')
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        result = reference(args) if args.kind == 'reference' else measured(args)
        require(not mp.active_children(), 'CPU worker left environment processes alive')
        record.update(result, passed=True, status='passed', all_children_exited=True)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        native.write(args.output / 'worker.json', record)


if __name__ == '__main__':
    main()
