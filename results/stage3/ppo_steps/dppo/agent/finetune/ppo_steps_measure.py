"""DPPO gradient measurements; training code and historical results are untouched."""
import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from agent.finetune.grad_mean import state_sha256
from agent.finetune.stage3_math import mc_returns, episode_indices
from agent.finetune.ppo_steps_state import (_copy, snapshot, restore, rng_state,
                                           restore_rng, scaler_divisor, native_update)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def validate(agent):
    model = agent.model
    if (agent.n_steps, agent.n_envs, agent.batch_size, agent.update_epochs,
            model.ft_denoising_steps) != (500, 40, 50000, 5, 10):
        raise ValueError('DPPO batch geometry changed')
    if (agent.max_grad_norm is not None or agent.reset_at_iteration or agent.actor_single_step
            or model.actor_loss != 'ratio' or not model.clamp_logprob or not model.norm_adv
            or model.logprob_reduce != 'mean' or model.randn_clip_value != 3
            or model.gamma_denoising != .99 or model.clip_ploss_coef != .01
            or model.clip_ploss_coef_base != .01 or agent.target_kl != 1
            or agent.gae_lambda != .95 or agent.gamma != .99 or agent.learn_eta
            or agent.use_bc_loss or agent.ent_coef != 0 or agent.stage3_active
            or not agent.reward_scale_running or agent.reward_scale_const != 1.0
            or model.get_min_sampling_denoising_std() != .1
            or model.min_logprob_denoising_std != .1):
        raise ValueError('Measurement must use the native DPPO default estimator and sampler')
    if (model.clip_advantage_lower_quantile, model.clip_advantage_upper_quantile) != (0, 1):
        raise ValueError('Expected identity advantage quantile clip')
    if sum(p.numel() for p in model.actor_ft.parameters()) != 568392:
        raise ValueError('Actor parameter count changed')


def prepared_batch(agent, collected):
    episode_indices(collected['firsts'])
    output = {key: collected[key] for key in ('obs_k', 'chains_k')}
    for key, source in (('returns_k', 'returns'), ('values_k', 'values'),
                        ('advantages_k', 'gae_adv')):
        output[key] = torch.tensor(collected[source], device=agent.device).float().reshape(-1)
    output['logprobs_k'] = collected['oldlogprobs']
    return output


def minibatch_args(agent, batch, indices):
    decisions, denoising = torch.unravel_index(indices, (agent.n_steps * agent.n_envs,
                                                       agent.model.ft_denoising_steps))
    return ({'state': batch['obs_k']['state'][decisions]},
            batch['chains_k'][decisions, denoising], batch['chains_k'][decisions, denoising + 1],
            denoising, batch['returns_k'][decisions], batch['values_k'][decisions],
            batch['advantages_k'][decisions], batch['logprobs_k'][decisions, denoising])


def flat_gradient(agent):
    params = list(agent.model.actor_ft.parameters())
    if any(p.grad is None for p in params):
        raise ValueError('Missing actor gradient coordinate')
    value = torch.cat([p.grad.detach().reshape(-1) for p in params]).cpu().double().numpy()
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite DPPO actor gradient')
    return value


@torch.no_grad()
def step_scalars(agent, args, losses):
    """Observational forward only; same normalization, clamp and ratio as loss."""
    obs, previous, following, denoising, _, _, advantages, old = args
    model = agent.model
    new = model.get_logprobs_subsample(obs, previous, following, denoising)
    new = new.clamp(-5, 2)[:, :agent.reward_horizon].mean(dim=(-1, -2)).view(-1)
    old = old.clamp(-5, 2)[:, :agent.reward_horizon].mean(dim=(-1, -2)).view(-1)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    adv = adv.clamp(torch.quantile(adv, 0.0), torch.quantile(adv, 1.0))
    discount = torch.tensor([model.gamma_denoising ** (model.ft_denoising_steps - i - 1)
                             for i in denoising]).to(agent.device)
    adv *= discount
    logratio = new - old
    ratio = logratio.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Nonfinite DPPO importance ratio')
    native_kl = ((ratio - 1) - logratio).mean().item()
    if not math.isclose(native_kl, float(losses[4]), rel_tol=1e-5, abs_tol=1e-9):
        raise AssertionError('Scalar diagnostic differs from the native loss KL')
    if not math.isclose(ratio.mean().item(), float(losses[5]), rel_tol=1e-6, abs_tol=1e-7):
        raise AssertionError('Scalar diagnostic differs from native mean ratio')
    off = ((adv > 0) & (ratio > 1.01)) | ((adv < 0) & (ratio < .99))
    return {'clipfrac': float(losses[3]), 'approx_kl': float(losses[4]),
            'switched_off_frac': off.float().mean().item(),
            'mean_abs_ratio_minus_one': (ratio - 1).abs().mean().item()}


def gradient_at_old(agent, batch, indices):
    args = minibatch_args(agent, batch, indices)
    losses = agent.model.loss(*args, use_bc_loss=False, reward_horizon=agent.reward_horizon)
    agent.actor_optimizer.zero_grad()
    losses[0].backward()
    gradient = flat_gradient(agent)
    return gradient, step_scalars(agent, args, losses)


def part_a_batch(agent, batch, indices=None):
    indices = (torch.randperm(200000, device=agent.device) if indices is None else indices)
    gradients, scalars = [], []
    for start in range(0, 200000, 50000):
        gradient, metrics = gradient_at_old(agent, batch, indices[start:start + 50000])
        gradients.append(gradient)
        scalars.append(metrics)
    # Only one batch's four temporary vectors are held; none is written to disk.
    full = np.stack(gradients).mean(axis=0, dtype=np.float64)
    full_scalars = {key: float(np.mean([row[key] for row in scalars])) for key in scalars[0]}
    return [('1', gradients[0], scalars[0]), ('full', full, full_scalars)]


def part_c_batch(agent, batch, callback):
    """Observe the literal native update loop; skip only critic optimizer steps."""
    native_loss = agent.model.loss
    native_step = agent.actor_optimizer.step
    critic_step = agent.critic_optimizer.step
    current = {}
    count = [0]
    def loss(*args, **kwargs):
        values = native_loss(*args, **kwargs)
        current.update(args=args, losses=values)
        return values
    def step(*args, **kwargs):
        count[0] += 1
        gradient = flat_gradient(agent)
        before = rng_state()
        metrics = step_scalars(agent, current['args'], current['losses'])
        if not same(before, rng_state()) or not np.array_equal(gradient, flat_gradient(agent)):
            raise AssertionError('Per-step diagnostics changed RNG or optimizer gradients')
        callback(count[0], gradient, metrics)
        return native_step(*args, **kwargs)
    agent.model.loss = loss
    agent.actor_optimizer.step = step
    agent.critic_optimizer.step = lambda *args, **kwargs: None
    try:
        result = native_update(agent, batch)
    finally:
        agent.model.loss = native_loss
        agent.actor_optimizer.step = native_step
        agent.critic_optimizer.step = critic_step
    if count[0] != result['actor_optimizer_steps']:
        raise AssertionError('Actor step observer missed a native optimizer step')
    return result


def critic_check(agent, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    expected = {key[len('critic.'):]: value for key, value in state.items() if key.startswith('critic.')}
    actual = agent.model.critic.state_dict()
    if not expected or not same(expected, actual):
        raise AssertionError('Checkpoint critic was not restored bitwise')
    # Explicitly requested F1-style critic hash, not general artifact hashing.
    expected_hash, actual_hash = state_sha256(expected), state_sha256(actual)
    if expected_hash != actual_hash:
        raise AssertionError('Critic hash mismatch')
    return {'passed': True, 'checkpoint_critic_sha256': expected_hash,
            'frozen_critic_sha256': actual_hash, 'warmup_batches': 0}


def prepare(agent, checkpoint, commit):
    directory = Path(agent.logdir)
    gate = critic_check(agent, checkpoint)
    before = _copy(agent.model.state_dict())
    rows = []
    for index in range(5):
        batch = agent._stage3_collect(deterministic=False, update_scaler=False)
        raw_return = mc_returns(batch['raw_rewards'], batch['firsts'], agent.gamma)
        values = np.asarray(batch['values'], dtype=np.float64)
        rows.append({'batch': index, 'sum_VG': float(np.sum(values * raw_return, dtype=np.float64)),
                     'sum_V2': float(np.sum(values * values, dtype=np.float64))})
        atomic_json(directory / 'progress.json', {'status': 'running', 'completed': index + 1, 'total': 5})
    c = math.fsum(row['sum_VG'] for row in rows) / math.fsum(row['sum_V2'] for row in rows)
    if not math.isfinite(c) or c <= 0 or not same(before, agent.model.state_dict()):
        raise AssertionError('Scale fit changed parameters or produced invalid c')
    frozen = {'model': before, 'c': c, 'c_source': 'least_squares_five_batches_fixed',
              'source_checkpoint': str(checkpoint), 'commit': commit, 'seed': agent.seed,
              'fit_batches': rows, 'critic_gate': gate}
    with (directory / 'frozen.pt').open('xb') as stream:
        torch.save(frozen, stream)
    result = {'status': 'completed', 'critic_gate': gate, 'c': c, 'c_source': frozen['c_source'],
              'fit_batches': rows, 'seed': agent.seed, 'commit': commit, 'checkpoint': str(checkpoint),
              'limitation': 'Original scaler mean/count were not saved; fixed fitted scaling cannot replay the native current-batch running-scaler update.'}
    atomic_json(directory / 'prepare.json', result)
    atomic_json(directory / 'progress.json', {'status': 'complete', 'completed': 5, 'total': 5})
    return result


def scalar_row(index, step, gradient, first, metrics, c, c_source):
    squared = float(np.dot(gradient, gradient))
    first_sq = float(np.dot(first, first))
    cosine = (float(np.dot(gradient, first) / math.sqrt(squared * first_sq))
              if squared > 0 and first_sq > 0 else None)
    return dict(metrics, batch=index, step=step, grad_sq=squared, grad_norm=math.sqrt(squared),
                cos_g1=cosine,
                c=c, c_source=c_source)


def save_rows(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + '.pending')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect(agent, part, checkpoint_itr, state_path, commit):
    directory = Path(agent.logdir)
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    before = _copy(agent.model.state_dict())
    if part == 'A':
        agent.model.load_state_dict(state['model'])
        agent.running_reward_scaler.ret_rms.var = np.asarray(state['c'] ** 2 - agent.running_reward_scaler.epsilon)
        if not same(agent.model.critic.state_dict(), {key[7:]: value for key, value in state['model'].items() if key.startswith('critic.')}):
            raise AssertionError('Frozen critic load mismatch')
        before = _copy(agent.model.state_dict())
    rows, totals, sums = [], [], {}
    meta = {'status': 'running', 'part': part, 'checkpoint_itr': checkpoint_itr,
            'updates': checkpoint_itr - checkpoint_itr // 10, 'seed': agent.seed, 'commit': commit,
            'K': 50, 'completed_batches': 0, 'parameter_count': 568392,
            'parameter_specs': [{'name': n, 'shape': list(p.shape)} for n, p in agent.model.actor_ft.named_parameters()],
            'source_state': str(state_path), 'max_grad_norm': None, 'reset_at_iteration': False,
            'group_vector_kind': 'sum', 'stored_dtype': 'float64',
            'c_source': state['c_source'] if part == 'A' else 'restored_full_scaler_updated_on_each_batch',
            'critic_optimizer_steps_skipped': part == 'C',
            'seed_note': 'Job seed initializes Python, NumPy and Torch; inherited DPPO environment seeding is unchanged and does not reach the private MuJoCo reset RNG.'}
    started = time.monotonic()
    for index in range(50):
        if part == 'C':
            restore(agent, state, restore_rng=False)
            agent.itr = checkpoint_itr + 1
        agent.model.train()
        c_start = scaler_divisor(agent)
        tick = time.monotonic()
        fresh = agent._stage3_collect(deterministic=False, update_scaler=part == 'C')
        rollout_seconds = time.monotonic() - tick
        batch = prepared_batch(agent, fresh)
        c = scaler_divisor(agent)
        tick = time.monotonic()
        first = [None]
        gradient_sum = np.zeros(568392, dtype=np.float64)
        theta_before = torch.cat([p.detach().reshape(-1) for p in agent.model.actor_ft.parameters()]).cpu().double().numpy()
        def observe(step, gradient, metrics):
            key = 'step%02d' % int(step) if str(step) != 'full' else 'full'
            if first[0] is None:
                first[0] = gradient.copy()
            sums.setdefault(key, np.zeros_like(gradient))
            sums[key] += gradient
            if step != 'full':
                gradient_sum[:] += gradient
            rows.append(scalar_row(index, step, gradient, first[0], metrics, c, meta['c_source']))
        if part == 'A':
            for step, gradient, metrics in part_a_batch(agent, batch):
                observe(step, gradient, metrics)
            result = {'actor_optimizer_steps': 0, 'target_kl_fired': False}
        else:
            result = part_c_batch(agent, batch, observe)
            theta_after = torch.cat([p.detach().reshape(-1) for p in agent.model.actor_ft.parameters()]).cpu().double().numpy()
            delta = theta_after - theta_before
            for key, vector in (('G', gradient_sum), ('delta', delta)):
                sums.setdefault(key, np.zeros_like(vector))
                sums[key] += vector
            totals.append({'batch': index, 'G_sq': float(gradient_sum @ gradient_sum),
                           'delta_sq': float(delta @ delta)})
        record = {'batch': index, 'c_start': c_start, 'c_used': c,
                  'rollout_seconds': rollout_seconds, 'gradient_seconds': time.monotonic() - tick,
                  'actor_steps': result['actor_optimizer_steps'],
                  'target_kl_fired': bool(result.get('target_kl_fired', result.get('approx_kl', 0) > agent.target_kl))}
        meta.setdefault('batches', []).append(record)
        meta['completed_batches'] = index + 1
        meta['elapsed_seconds'] = time.monotonic() - started
        save_rows(directory / 'batches.csv', rows)
        save_rows(directory / 'batch_totals.csv', totals)
        atomic_json(directory / 'meta.json', meta)
        atomic_json(directory / 'progress.json', {'status': 'running', 'completed': index + 1, 'total': 50,
                                                'elapsed_seconds': meta['elapsed_seconds']})
    for name, vector in sums.items():
        with (directory / ('sums_' + name + '.npy')).open('xb') as stream:
            np.save(stream, vector, allow_pickle=False)
    if part == 'A' and not same(before, agent.model.state_dict()):
        raise AssertionError('Part A modified checkpoint parameters')
    meta['status'] = 'completed'
    atomic_json(directory / 'meta.json', meta)
    atomic_json(directory / 'progress.json', {'status': 'complete', 'completed': 50, 'total': 50})
    return meta


def gate_s1(agent, state_path):
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    comparisons = []
    for index in range(2):
        restore(agent, state, restore_rng=False)
        agent.itr = int(state['itr']) + 1
        agent.model.train()
        batch = prepared_batch(agent, agent._stage3_collect(deterministic=False, update_scaler=True))
        rng = rng_state()
        indices = torch.randperm(200000, device=agent.device)
        reference, _ = gradient_at_old(agent, batch, indices[:50000])
        restore_rng(rng)
        got = []
        def observe(step, gradient, metrics):
            if step == 1:
                got.append(gradient.copy())
        part_c_batch(agent, batch, observe)
        actual = got[0]
        difference = np.abs(actual - reference)
        relative = float(np.max(difference / np.maximum(np.abs(reference), 1e-12)))
        comparisons.append({'batch': index, 'max_abs': float(difference.max()),
                            'max_relative': relative, 'passed': relative <= 1e-6})
    result = {'status': 'passed' if all(r['passed'] for r in comparisons) else 'failed',
              'passed': all(r['passed'] for r in comparisons), 'comparisons': comparisons}
    atomic_json(Path(agent.logdir) / 'S1.json', result)
    if not result['passed']:
        raise AssertionError('S1 first-step gradient equivalence failed')
    return result
