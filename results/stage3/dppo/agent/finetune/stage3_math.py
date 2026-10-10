"""Stage-3 estimator mathematics and side-effect-free gradient measurements."""

import math
import numpy as np
import torch


def mc_returns(rewards, firsts, gamma):
    """Float64 reward-to-go; boundary t+1 stops the recursion after reward t."""
    rewards = np.asarray(rewards, dtype=np.float64)
    firsts = np.asarray(firsts)
    if rewards.ndim != 2 or firsts.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError("rewards/firsts must have shapes (T,E)/(T+1,E)")
    if not np.isfinite(rewards).all() or not np.isfinite(gamma):
        raise ValueError("non-finite MC input")
    if not np.isin(firsts, (0, 1)).all():
        raise ValueError("episode boundaries must be binary")
    result = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running * (1.0 - firsts[t + 1])
        result[t] = running
    return result


def episode_indices(firsts, horizon=250):
    """Assert complete aligned episodes and return the within-episode index."""
    firsts = np.asarray(firsts)
    if firsts.ndim != 2 or len(firsts) < 2 or horizon < 1:
        raise ValueError("invalid episode-boundary shape or horizon")
    steps, envs = len(firsts) - 1, firsts.shape[1]
    expected = np.zeros_like(firsts)
    expected[::horizon] = 1
    if steps % horizon or not np.array_equal(firsts, expected):
        observed = {str(e): np.flatnonzero(firsts[:, e]).tolist() for e in range(envs)}
        raise ValueError("episode alignment failed: horizon=%d, steps=%d, boundaries=%r" %
                         (horizon, steps, observed))
    return np.broadcast_to((np.arange(steps) % horizon)[:, None], (steps, envs)).copy()


def as_episodes(values, firsts, horizon=250):
    values = np.asarray(values, dtype=np.float64)
    indices = episode_indices(firsts, horizon)
    if values.shape != indices.shape:
        raise ValueError("values do not match episode boundaries")
    return values.T.reshape(-1, horizon)


def loo_advantages(returns, firsts, horizon=250):
    """LOO baseline across all episodes at equal within-episode decision k."""
    returns = np.asarray(returns, dtype=np.float64)
    episodes = as_episodes(returns, firsts, horizon)
    if len(episodes) < 2:
        raise ValueError("LOO needs at least two episodes")
    baseline = (episodes.sum(axis=0, keepdims=True) - episodes) / (len(episodes) - 1)
    shape = (returns.shape[1], returns.shape[0])
    baseline = baseline.reshape(shape).T
    return returns - baseline, baseline


def discounted_episode_returns(rewards, firsts, gamma, horizon=250):
    episodes = as_episodes(rewards, firsts, horizon)
    return episodes @ np.power(float(gamma), np.arange(horizon, dtype=np.float64))


def mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "n": int(len(values))}


def _noise_from_gram(mean_dot, scalar_cross_sums, n):
    if n < 2:
        raise ValueError("at least two gradient vectors are required")
    covariance = (np.asarray(scalar_cross_sums, dtype=np.float64) - n * mean_dot) / (n - 1)
    signal = mean_dot - covariance / n
    return {"mean_dot": mean_dot, "covariance": covariance,
            "cross_signal": signal, "u_sq": np.diag(signal).copy(),
            "tr_cov": np.diag(covariance).copy(), "n": int(n)}


def noise_moments(sum_vectors, scalar_cross_sums, n):
    """Prescribed per-env moment estimator; no clipping of signed estimates."""
    sums = np.asarray(sum_vectors, dtype=np.float64)
    if sums.ndim == 1:
        sums = sums[None]
    if sums.ndim != 2:
        raise ValueError("sum_vectors must have shape (variants, parameters)")
    scalar_cross_sums = np.asarray(scalar_cross_sums, dtype=np.float64)
    if scalar_cross_sums.shape != (len(sums), len(sums)):
        raise ValueError("scalar_cross_sums shape mismatch")
    mean_dot = (sums @ sums.T) / float(n * n)
    return _noise_from_gram(mean_dot, scalar_cross_sums, n)


def batch_noise_summary(batch_sums, batch_cross_sums, envs_per_batch=40,
                        episodes_per_env=2, bootstrap_reps=2000, seed=0):
    """Cluster bootstrap over batches using only their vector-sum Gram matrix.

    This follows the requested moment correction, which is approximate for
    variants coupled through full-batch normalization or LOO coefficients.
    """
    sums = np.asarray(batch_sums, dtype=np.float64)
    scalar = np.asarray(batch_cross_sums, dtype=np.float64)
    if sums.ndim != 3 or scalar.shape != (sums.shape[0], sums.shape[1], sums.shape[1]):
        raise ValueError("expected batch_sums(B,V,P), batch_cross_sums(B,V,V)")
    batches, variants, parameters = sums.shape
    n = batches * envs_per_batch
    flattened = sums.reshape(batches * variants, parameters)
    gram = (flattened @ flattened.T).reshape(batches, variants, batches, variants)
    point = noise_moments(sums.sum(axis=0), scalar.sum(axis=0), n)
    # Dedicated generator: never draws from NumPy's global generator.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 314159]))
    weights = rng.multinomial(batches, np.full(batches, 1.0 / batches),
                              size=bootstrap_reps).astype(np.float64)
    boot_dot = np.einsum("rb,bvcw,rc->rvw", weights, gram, weights, optimize=True) / (n * n)
    boot_scalar = np.einsum("rb,bvw->rvw", weights, scalar, optimize=True)
    boot_cov = (boot_scalar - n * boot_dot) / (n - 1)
    boot_signal = boot_dot - boot_cov / n
    boot_u = np.diagonal(boot_signal, axis1=1, axis2=2)
    boot_var = np.diagonal(boot_cov, axis1=1, axis2=2)
    ci_u = np.quantile(boot_u, [0.05, 0.95], axis=0)
    ci_var = np.quantile(boot_var, [0.05, 0.95], axis=0)
    # Lower bound uses simultaneous Bonferroni one-sided 95% percentile bounds:
    # variance >= its 5th percentile and signal <= its 95th percentile.
    rows = []
    for v in range(variants):
        u, variance = float(point["u_sq"][v]), float(point["tr_cov"][v])
        resolved = bool(ci_u[0, v] > 0 and u > 0 and variance >= 0)
        row = {"u_sq": u, "u_sq_ci90": ci_u[:, v].tolist(),
               "tr_sigma": variance, "tr_sigma_ci90": ci_var[:, v].tolist(),
               "signal_ci_includes_zero": bool(ci_u[0, v] <= 0 <= ci_u[1, v]),
               "signal_resolved_positive": resolved,
               "B_env": variance / u if resolved else None,
               "B_ep": episodes_per_env * variance / u if resolved else None,
               "snr_training_batch": envs_per_batch * u / variance if resolved and variance > 0 else None,
               "expected_batch_cosine": (1 + variance / (envs_per_batch * u)) ** -0.5 if resolved else None,
               "B_env_lower_bound": (max(0.0, float(ci_var[0, v])) / float(ci_u[1, v]))
                   if ci_u[1, v] > 0 else None}
        if resolved and np.all(boot_u[:, v] > 0) and np.all(boot_var[:, v] >= 0):
            ratios = boot_var[:, v] / boot_u[:, v]
            row["B_env_ci90"] = np.quantile(ratios, [0.05, 0.95]).tolist()
            row["B_ep_ci90"] = (episodes_per_env * np.quantile(ratios, [0.05, 0.95])).tolist()
            row["snr_training_batch_ci90"] = (np.quantile(envs_per_batch / ratios, [0.05, 0.95]).tolist()
                                                   if np.all(ratios > 0) else None)
            row["expected_batch_cosine_ci90"] = np.quantile((1 + ratios / envs_per_batch) ** -0.5, [0.05, 0.95]).tolist()
        else:
            row.update({key: None for key in ("B_env_ci90", "B_ep_ci90", "snr_training_batch_ci90", "expected_batch_cosine_ci90")})
        rows.append(row)
    cosines = []
    cosine_ci = []
    for v in range(variants):
        row, row_ci = [], []
        for w in range(variants):
            if rows[v]["signal_resolved_positive"] and rows[w]["signal_resolved_positive"]:
                row.append(float(point["cross_signal"][v, w] /
                                 np.sqrt(point["u_sq"][v] * point["u_sq"][w])))
                if np.all(boot_u[:, v] > 0) and np.all(boot_u[:, w] > 0):
                    cos = boot_signal[:, v, w] / np.sqrt(boot_u[:, v] * boot_u[:, w])
                    row_ci.append(np.quantile(cos, [0.05, 0.95]).tolist())
                else:
                    row_ci.append(None)
            else:
                row.append(None)
                row_ci.append(None)
        cosines.append(row)
        cosine_ci.append(row_ci)
    return {"variants": rows, "cosine_matrix": cosines, "cosine_ci90": cosine_ci,
            "cross_signal": point["cross_signal"].tolist(),
            "n_env_rollouts": n, "n_batches": batches, "bootstrap_reps": bootstrap_reps,
            "gram": gram, "uncertainty_unit": "whole fresh batch",
            "dependence_note": "The prescribed n-based moment correction is approximate for full-batch-normalized and LOO-coupled variants; batch bootstrap preserves within-batch dependence but does not eliminate centering bias.",
            "bound_note": "B_env lower bound uses max(0, variance 5th percentile) / signal 95th percentile, a Bonferroni combination of one-sided 95% percentile bounds. Signed point moments remain unchanged.",
            "cosine_note": "Signed debiased moments and unbounded cosine estimates are retained, never clipped. Derived values are null when signal is unresolved; an estimated cosine outside [-1,1] signals estimation uncertainty."}


def actor_gradient(agent, batch, advantages, reduce="sum", gamma_denoising=1.0,
                   normalize=False, env_indices=None, clamp=False):
    """Return a loss gradient without touching .grad, RNG, or model state.

    Normalization uses the entire pair batch before selecting environments.
    Chunk weighting is each chunk's pair count divided by selected pair count.
    """
    if reduce not in ("mean", "sum"):
        raise ValueError("invalid coordinate reduction")
    model = agent.model
    parameters = list(model.actor_ft.parameters())
    device = parameters[0].device
    dtype = parameters[0].dtype
    total_decisions = batch["chains_k"].shape[0]
    denoising = model.ft_denoising_steps
    adv = torch.as_tensor(advantages, device=device, dtype=dtype).detach().reshape(-1).clone()
    if adv.numel() != total_decisions:
        raise ValueError("one advantage per decision is required")
    if normalize:
        # Equivalent to std() on every decision repeated once per denoising step.
        mean = adv.mean()
        count = adv.numel() * denoising
        std = torch.sqrt(((adv - mean) ** 2).sum() * denoising / (count - 1))
        adv = (adv - mean) / (std + 1e-8)
    decisions = torch.arange(total_decisions, device=device)
    if env_indices is not None:
        environments = torch.as_tensor(env_indices, device=device, dtype=torch.long)
        decisions = decisions.reshape(-1, agent.n_envs)[:, environments].reshape(-1)
    # Ordered decision-major pair indices; no random permutation or RNG use.
    pairs = (decisions[:, None] * denoising + torch.arange(denoising, device=device)).reshape(-1)
    gradient = torch.zeros(sum(p.numel() for p in parameters), dtype=dtype, device=device)
    discounts = torch.tensor([gamma_denoising ** (denoising - j - 1) for j in range(denoising)],
                             device=device, dtype=dtype)
    for start in range(0, len(pairs), agent.batch_size):
        selected = pairs[start:start + agent.batch_size]
        decisions_b, steps_b = torch.div(selected, denoising, rounding_mode="floor"), selected % denoising
        logp = model.get_logprobs_subsample(
            {key: value[decisions_b] for key, value in batch["obs_k"].items()},
            batch["chains_k"][decisions_b, steps_b],
            batch["chains_k"][decisions_b, steps_b + 1], steps_b)
        if clamp:
            logp = logp.clamp(min=-5, max=2)
        logp = logp[:, :agent.reward_horizon, :]
        logp = logp.mean(dim=(-1, -2)) if reduce == "mean" else logp.sum(dim=(-1, -2))
        coefficients = adv[decisions_b] * discounts[steps_b]
        loss = -(coefficients * logp).sum() / len(pairs)
        chunk = torch.autograd.grad(loss, parameters, allow_unused=True)
        offset = 0
        for parameter, value in zip(parameters, chunk):
            if value is not None:
                gradient[offset:offset + parameter.numel()].add_(value.detach().reshape(-1))
            offset += parameter.numel()
    return gradient


@torch.no_grad()
def project_ball_(parameters, origin, radius):
    """In-place Euclidean projection; inside-ball parameters are untouched."""
    parameters = list(parameters)
    theta = torch.cat([p.detach().reshape(-1) for p in parameters])
    origin = torch.as_tensor(origin, device=theta.device, dtype=theta.dtype)
    if theta.shape != origin.shape or radius is None or radius < 0 or not math.isfinite(radius):
        raise ValueError("invalid projection origin or radius")
    displacement = theta - origin
    norm = torch.linalg.vector_norm(displacement.double())
    active = bool(norm.item() > radius)
    if active:
        projected = origin + displacement * (radius / norm).to(theta.dtype)
        offset = 0
        for parameter in parameters:
            parameter.copy_(projected[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()
    return {"theta_dist_preproj": norm.item(), "proj_active": active}
