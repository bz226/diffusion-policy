"""Flagged critic baselines; calculations use only pre-update saved values."""
import json
from pathlib import Path

import numpy as np
import torch

from agent.finetune.stage3_math import as_episodes, episode_indices, loo_advantages, mc_returns


def reward_scaler_divisor(agent):
    """Raw-reward divisor, copied before the current batch changes the scaler."""
    if not agent.reward_scale_running:
        return np.asarray(1.0, dtype=np.float64)
    scaler = agent.running_reward_scaler
    return np.sqrt(np.asarray(scaler.ret_rms.var, dtype=np.float64) + scaler.epsilon).copy()


def critic_residual(raw_returns, values, divisor):
    returns = np.asarray(raw_returns, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(returns.shape)
    divisor = np.asarray(divisor, dtype=np.float64)
    if divisor.ndim > 1 or (divisor.ndim == 1 and divisor.shape != (returns.shape[1],)):
        raise ValueError("critic divisor must be scalar or one value per environment")
    if not (np.isfinite(returns).all() and np.isfinite(values).all()
            and np.isfinite(divisor).all() and np.all(divisor > 0)):
        raise FloatingPointError("non-finite critic baseline input or nonpositive divisor")
    return returns - values * divisor


def mc_critic_advantages(raw_rewards, firsts, gamma, values, divisor):
    """Float64 G-cV, before decision discount and the actor's float32 cast."""
    return critic_residual(mc_returns(raw_rewards, firsts, gamma), values, divisor)


def baseline_diagnostics(raw_returns, values, divisor, firsts, gamma):
    """Per-k episode moments and pooled bin moments, with raw residual energies.

    Standard deviations use ddof=0. Each bin mean/std pools its 80 x 50
    episode/decision residuals. Full per-k episode moments are retained so the
    time-to-go variation and episode variation remain distinguishable.
    """
    critic = as_episodes(critic_residual(raw_returns, values, divisor), firsts)
    loo, _ = loo_advantages(raw_returns, firsts)
    loo = as_episodes(loo, firsts)
    weights = np.power(float(gamma), 2 * np.arange(critic.shape[1], dtype=np.float64))
    metrics = {"baseline_episode_count": int(len(critic)),
               "baseline_residual_std_ddof": 0,
               "baseline_bin_edges": [0, 50, 100, 150, 200, 250],
               "baseline_bin_moment_definition": "pooled episodes and decisions; per-k episode moments also retained"}
    for name, residual in (("critic", critic), ("loo", loo)):
        metrics["baseline_residual_mean_" + name] = residual.mean(axis=0).tolist()
        metrics["baseline_residual_std_" + name] = residual.std(axis=0, ddof=0).tolist()
        metrics["baseline_energy_" + name] = float(np.mean(np.sum(residual ** 2 * weights, axis=1)))
        metrics["baseline_bin_residual_mean_" + name] = []
        metrics["baseline_bin_residual_std_" + name] = []
        metrics["baseline_bin_energy_" + name] = []
        for lo in range(0, 250, 50):
            part = residual[:, lo:lo + 50]
            metrics["baseline_bin_residual_mean_" + name].append(float(part.mean()))
            metrics["baseline_bin_residual_std_" + name].append(float(part.std(ddof=0)))
            metrics["baseline_bin_energy_" + name].append(float(np.mean(np.sum(part ** 2 * weights[lo:lo + 50], axis=1))))
    denominator = metrics["baseline_energy_loo"]
    metrics["baseline_energy_ratio"] = metrics["baseline_energy_critic"] / denominator if denominator > 0 else None
    metrics["baseline_energy_ratio_reason"] = "zero_loo_energy" if denominator == 0 else None
    metrics["baseline_bin_energy_ratio"] = [a / b if b > 0 else None for a, b in zip(
        metrics["baseline_bin_energy_critic"], metrics["baseline_bin_energy_loo"])]
    return metrics


def load_critic_initialization(agent, path):
    """Load only the frozen critic and scaler; optimizer state stays fresh."""
    from agent.finetune.grad_mean import _restore_scaler, actor_state, scaler_state, state_sha256
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("point") != "P0" or not all(key in payload for key in ("critic", "reward_scaler")):
        raise ValueError("critic_init_path must contain the gradient-mean frozen P0 critic/scaler")
    if not agent.reward_scale_running:
        raise ValueError("critic_init_path requires the existing running reward scaler")
    before = {"actor": state_sha256(actor_state(agent))}
    agent.model.critic.load_state_dict(payload["critic"], strict=True)
    _restore_scaler(agent, payload["reward_scaler"])
    critic_hash = state_sha256(agent.model.critic.state_dict())
    scaler_hash = state_sha256(scaler_state(agent))
    if critic_hash != payload.get("critic_sha256") or scaler_hash != payload.get("scaler_sha256"):
        raise ValueError("loaded critic/scaler differs from the saved frozen P0 state")
    after_actor = state_sha256(actor_state(agent))
    if after_actor != before["actor"]:
        raise RuntimeError("critic initialization changed actor or actor_ft")
    # This function does not call either optimizer. State evidence is retained for C3.
    audit = {"path": str(path), "critic_sha256": critic_hash, "scaler_sha256": scaler_hash,
             "actor_before_sha256": before["actor"], "actor_after_sha256": after_actor,
             "optimizer_state_loaded": False}
    (Path(agent.logdir) / "critic_initialization.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def configured_actor_coefficients(agent, batch):
    """Use training's estimator/cast/discount order for calibration."""
    if agent.adv_estimator == "gae":
        advantages = np.asarray(batch["gae_adv"], dtype=np.float64)
    else:
        advantages = mc_returns(batch["raw_rewards"], batch["firsts"], agent.gamma)
        if agent.adv_estimator == "mc_loo":
            advantages, _ = loo_advantages(advantages, batch["firsts"])
        elif agent.adv_estimator == "mc_critic":
            advantages = critic_residual(advantages, batch["values"], agent._critic_baseline_divisor)
        elif agent.adv_estimator != "mc_none":
            raise ValueError("unsupported calibration advantage estimator")
    coefficients = torch.as_tensor(advantages, device=agent.device, dtype=torch.float32).reshape(-1)
    if agent.decision_discount:
        weights = torch.as_tensor(agent.gamma ** episode_indices(batch["firsts"]),
                                  device=agent.device, dtype=coefficients.dtype).reshape(-1)
        coefficients = coefficients * weights
    return coefficients


def configured_actor_gradient(agent, batch, coefficients, permutation):
    """Actual per-chunk actor loss, without changing .grad or optimizer state.

    The supplied full pair permutation represents the final critic epoch used
    by the training single-step pass. model.loss therefore retains exactly its
    per-minibatch normalization and configured clamp/reduction/score semantics.
    """
    parameters = list(agent.model.actor_ft.parameters())
    total = batch["chains_k"].shape[0] * agent.model.ft_denoising_steps
    permutation = permutation.to(device=agent.device, dtype=torch.long)
    if len(permutation) != total or not torch.equal(
            torch.sort(permutation).values, torch.arange(total, device=agent.device)):
        raise ValueError("calibration requires a complete pair permutation")
    returns = torch.as_tensor(batch["returns"], device=agent.device, dtype=torch.float32).reshape(-1)
    values = torch.as_tensor(batch["values"], device=agent.device, dtype=torch.float32).reshape(-1)
    gradient = torch.zeros(sum(p.numel() for p in parameters), device=agent.device,
                           dtype=parameters[0].dtype)
    for start in range(0, total, agent.batch_size):
        pairs = permutation[start:start + agent.batch_size]
        decisions, steps = torch.unravel_index(pairs, (batch["chains_k"].shape[0],
                                                      agent.model.ft_denoising_steps))
        losses = agent.model.loss(
            {key: value[decisions] for key, value in batch["obs_k"].items()},
            batch["chains_k"][decisions, steps], batch["chains_k"][decisions, steps + 1],
            steps, returns[decisions], values[decisions], coefficients[decisions],
            batch["oldlogprobs"][decisions, steps], use_bc_loss=agent.use_bc_loss,
            reward_horizon=agent.reward_horizon)
        loss = (losses[0] + losses[1] * agent.ent_coef + losses[6] * agent.bc_loss_coeff)
        chunks = torch.autograd.grad(loss * (len(pairs) / total), parameters, allow_unused=True)
        offset = 0
        for parameter, chunk in zip(parameters, chunks):
            if chunk is not None:
                gradient[offset:offset + parameter.numel()].add_(chunk.detach().reshape(-1))
            offset += parameter.numel()
    return gradient
