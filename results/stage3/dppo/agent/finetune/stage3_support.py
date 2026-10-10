"""Flagged Stage-3 bookkeeping and nontraining rollout support.

Default training never calls these diagnostic/estimator paths. Measurements use
autograd.grad and owned tensors, leaving optimizer gradients and global RNG alone.
"""
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from agent.finetune.stage3_math import (
    actor_gradient, discounted_episode_returns, episode_indices, loo_advantages,
    mc_returns, mean_se, project_ball_,
)

log = logging.getLogger(__name__)


def flat_actor(model):
    return torch.cat([p.detach().reshape(-1) for p in model.actor_ft.parameters()])


class Stage3Support:
    def _stage3_init(self, cfg):
        self.stage3_cfg = cfg
        self.adv_estimator = cfg.train.get("adv_estimator", "gae")
        self.decision_discount = cfg.train.get("decision_discount", False)
        self.proj_radius = cfg.train.get("proj_radius", None)
        self.stage3_diag = cfg.train.get("stage3_diag", False)
        self.stage3_mode = cfg.train.get("mode", "train")
        if self.adv_estimator not in ("gae", "mc_loo", "mc_none"):
            raise ValueError("adv_estimator must be gae, mc_loo or mc_none")
        if self.stage3_mode not in ("train", "noise_scale", "calibrate", "ckpt_eval"):
            raise ValueError("invalid Stage-3 mode")
        if self.decision_discount and self.model.norm_adv:
            raise ValueError("decision_discount is incompatible with norm_adv=true")
        if self.proj_radius is not None and (
            not math.isfinite(float(self.proj_radius)) or float(self.proj_radius) < 0
        ):
            raise ValueError("proj_radius must be finite and nonnegative")
        self.stage3_active = bool(self.stage3_diag or self.adv_estimator != "gae"
                                 or self.decision_discount or self.proj_radius is not None
                                 or self.stage3_mode != "train")
        self.stage3_theta0 = flat_actor(self.model).clone() if self.stage3_active else None
        if self.stage3_active:
            with np.load(cfg.normalization_path) as normalization:
                self._stage3_action_min = normalization["action_min"].copy()
                self._stage3_action_max = normalization["action_max"].copy()
        self._stage3_prev_gradient = None
        self._stage3_metrics = {}
        self._stage3_diag_seconds = 0.0
        self._stage3_update_started = None
        self._stage3_expected_lr = float(cfg.train.actor_lr)
        self._stage3_constant_lr = float(cfg.train.actor_lr_scheduler.min_lr) == self._stage3_expected_lr
        group = self.actor_optimizer.param_groups[0]
        settings = {
            "actor_loss": self.model.actor_loss, "adv_estimator": self.adv_estimator,
            "decision_discount": self.decision_discount,
            "gamma_denoising": self.model.gamma_denoising,
            "logprob_reduce": self.model.logprob_reduce, "norm_adv": self.model.norm_adv,
            "clamp_logprob": self.model.clamp_logprob,
            "randn_clip_value": self.model.randn_clip_value,
            "actor_single_step": self.actor_single_step,
            "actor_optimizer": cfg.train.get("actor_optimizer", "adamw"),
            "optimizer_class": type(self.actor_optimizer).__name__,
            "lr": group["lr"], "actor_beta1": cfg.train.get("actor_beta1", 0.9),
            "betas": group.get("betas"), "momentum": group.get("momentum"),
            "scheduler_min_lr": cfg.train.actor_lr_scheduler.min_lr,
            "scheduler_max_lr": cfg.train.actor_lr, "proj_radius": self.proj_radius,
            "mode": self.stage3_mode, "stage3_diag": self.stage3_diag,
            "ns_warmup_batches": cfg.train.get("ns_warmup_batches", 20),
            "ns_batches": cfg.train.get("ns_batches", 25),
            "calib_kl_target": cfg.train.get("calib_kl_target", None),
            "ckpt_eval_repeats": cfg.train.get("ckpt_eval_repeats", 1),
            "denoised_clip_value": self.model.denoised_clip_value,
            "min_sampling_denoising_std": self.model.get_min_sampling_denoising_std(),
            "min_logprob_denoising_std": self.model.min_logprob_denoising_std,
            "max_grad_norm": self.max_grad_norm,
        }
        log.info("STAGE3_SETTINGS " + json.dumps(settings, sort_keys=True))

    def _stage3_assert_lr(self):
        if self._stage3_constant_lr:
            for group in self.actor_optimizer.param_groups:
                if group["lr"] != self._stage3_expected_lr:
                    raise AssertionError("actor learning rate changed despite equal min/max: %r != %r" %
                                         (group["lr"], self._stage3_expected_lr))

    def _stage3_iteration_start(self):
        self._stage3_started = time.perf_counter()
        self._stage3_diag_seconds = 0.0
        self._stage3_metrics = {}
        self._stage3_update_started = None
        self._stage3_actions = np.empty((self.n_steps, self.n_envs, self.act_steps, self.action_dim))
        self._stage3_iteration_theta = flat_actor(self.model).clone() if self.stage3_diag else None

    def _stage3_action_metrics(self, actions):
        # Repeat the unchanged wrapper's float32 affine map to the actual Gym input.
        actions = np.asarray(actions, dtype=np.float32)
        raw_actions = (actions + 1) / 2
        raw_actions = raw_actions * (self._stage3_action_max - self._stage3_action_min) + self._stage3_action_min
        return {"action_oor_frac": float(np.mean(np.abs(raw_actions) > 1)),
                "policy_action_oor_frac": float(np.mean(np.abs(actions) > 1))}

    def _stage3_rollout_finished(self, rewards, firsts, eval_mode):
        self._stage3_metrics["rollout_seconds"] = time.perf_counter() - self._stage3_started
        before = time.perf_counter()
        episode_indices(firsts)
        self._stage3_raw_rewards = rewards.copy()
        self._stage3_firsts = firsts.copy()
        if self.stage3_diag:
            summary = mean_se(discounted_episode_returns(rewards, firsts, self.gamma))
            prefix = "J_disc_eval" if eval_mode else "J_disc_train"
            self._stage3_metrics.update({prefix: summary["mean"], prefix + "_se": summary["se"],
                                        "episode_count": summary["n"]})
            self._stage3_metrics.update(self._stage3_action_metrics(self._stage3_actions))
        self._stage3_diag_seconds += time.perf_counter() - before
        self._stage3_update_started = time.perf_counter()
        self._stage3_update_diag_start = self._stage3_diag_seconds

    def _stage3_prepare_batch(self, raw_reward_trajs, firsts_trajs, terminated_trajs,
                             action_trajs, obs_k, chains_k, returns_k, values_k,
                             advantages_k, logprobs_k, scaled_rewards):
        """Change actor advantages only; critic targets and values are untouched."""
        before = time.perf_counter()
        indices = episode_indices(firsts_trajs)
        raw_adv = advantages_k.detach().cpu().double().numpy().reshape(raw_reward_trajs.shape)
        if self.adv_estimator != "gae":
            raw_adv = mc_returns(raw_reward_trajs, firsts_trajs, self.gamma)
            if self.adv_estimator == "mc_loo":
                raw_adv, _ = loo_advantages(raw_adv, firsts_trajs)
            advantages_k = torch.as_tensor(raw_adv, device=self.device, dtype=torch.float32).reshape(-1)
        if self.stage3_diag:
            self._stage3_metrics.update(adv_raw_mean=float(np.mean(raw_adv)),
                                        adv_raw_std=float(np.std(raw_adv, ddof=0)),
                                        reward_clip_frac=float(np.mean(np.abs(scaled_rewards) >= 10)))
        if self.decision_discount:
            # Form the finite-horizon decision coefficient before the loss pass.
            weights = torch.as_tensor(self.gamma ** indices, device=self.device,
                                      dtype=advantages_k.dtype).reshape(-1)
            advantages_k = advantages_k * weights
        batch = {"raw_rewards": raw_reward_trajs, "firsts": firsts_trajs,
                 "terminated": terminated_trajs, "actions": action_trajs,
                 "obs_k": obs_k, "chains_k": chains_k, "oldlogprobs": logprobs_k,
                 "returns_k": returns_k, "values_k": values_k,
                 "actor_advantages": advantages_k, "scaled_rewards": scaled_rewards}
        self._stage3_batch = batch
        self._stage3_diag_seconds += time.perf_counter() - before
        if self.stage3_diag:
            self._stage3_before_update(batch)
        return advantages_k

    def _stage3_before_update(self, batch):
        before = time.perf_counter()
        half = self.n_envs // 2
        if 2 * half != self.n_envs:
            raise ValueError("split gradients require an even number of environments")
        kwargs = dict(reduce=self.model.logprob_reduce,
                      gamma_denoising=self.model.gamma_denoising,
                      normalize=self.model.norm_adv, clamp=self.model.clamp_logprob)
        gradients = [actor_gradient(self, batch, batch["actor_advantages"],
                                    env_indices=range(start, start + half), **kwargs).double()
                     for start in (0, half)]
        self._stage3_metrics.update(split_dot=torch.dot(*gradients).item(),
                                    split_diff_sq=(gradients[0] - gradients[1]).square().sum().item())
        self._stage3_metrics.update(self._stage3_saturation(batch, self.itr))
        self._stage3_diag_seconds += time.perf_counter() - before

    @torch.no_grad()
    def _stage3_saturation(self, batch, iteration):
        # Separate from the unchanged Stage-2 advancing diagnostic generator.
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(self.seed) * 1000003 + int(iteration) * 9176 + 7321) % (2**63 - 1))
        denoising = self.model.ft_denoising_steps
        total = batch["chains_k"].shape[0] * denoising
        pairs = torch.randperm(total, generator=generator)[:min(total, 20000)].to(self.device)
        decisions, steps = torch.div(pairs, denoising, rounding_mode="floor"), pairs % denoising
        x = batch["chains_k"][decisions, steps]
        timesteps = denoising - 1 - steps
        noise = self.model.actor_ft(x, timesteps,
                                    cond={key: value[decisions] for key, value in batch["obs_k"].items()})
        scale1 = self.model.sqrt_recip_alphas_cumprod[timesteps].reshape(-1, 1, 1)
        scale2 = self.model.sqrt_recipm1_alphas_cumprod[timesteps].reshape(-1, 1, 1)
        raw = scale1 * x - scale2 * noise
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite pre-update raw x0 prediction")
        hit = raw.abs() > self.model.denoised_clip_value
        # Equal-step average (rather than changing the weight with subsample counts).
        per_step = [hit[steps == j].double().mean() for j in range(denoising)]
        return {"x0_sat_frac_mean": torch.stack(per_step).mean().item(),
                "x0_sat_frac_last": hit[timesteps == 0].double().mean().item()}

    def _stage3_before_step(self):
        before = time.perf_counter()
        if self.stage3_diag:
            gradient = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None
                                  else torch.zeros_like(p).reshape(-1)
                                  for p in self.model.actor_ft.parameters()]).double()
            norm = torch.linalg.vector_norm(gradient).item()
            previous = self._stage3_prev_gradient
            cosine, reason = None, "first_training_iteration"
            if previous is not None:
                denominator = norm * torch.linalg.vector_norm(previous).item()
                cosine = torch.dot(gradient, previous).item() / denominator if denominator > 0 else None
                reason = None if denominator > 0 else "zero_gradient"
            self._stage3_metrics.update(grad_norm=norm, grad_cos_prev=cosine,
                                        grad_cos_prev_reason=reason)
            self._stage3_current_gradient = gradient.clone()
        self._stage3_diag_seconds += time.perf_counter() - before

    def _stage3_after_step(self):
        before = time.perf_counter()
        if self.proj_radius is not None:
            projection = project_ball_(self.model.actor_ft.parameters(), self.stage3_theta0, self.proj_radius)
        else:
            projection = {"theta_dist_preproj": torch.linalg.vector_norm(
                flat_actor(self.model).double() - self.stage3_theta0.double()).item(), "proj_active": False}
        if self.stage3_diag:
            self._stage3_metrics.update(projection)
            theta = flat_actor(self.model).double()
            self._stage3_metrics.update(theta_dist=torch.linalg.vector_norm(theta - self.stage3_theta0.double()).item(),
                                        update_norm=torch.linalg.vector_norm(theta - self._stage3_iteration_theta.double()).item())
        self._stage3_diag_seconds += time.perf_counter() - before

    def _stage3_finish_update(self, explained_var, return_variance):
        if self.stage3_diag:
            self._stage3_prev_gradient = self._stage3_current_gradient
            before = time.perf_counter()
            self._stage3_metrics["explained_var"] = None if return_variance == 0 else float(explained_var)
            self._stage3_metrics["explained_var_reason"] = "zero_return_variance" if return_variance == 0 else None
            self._stage3_metrics["update_seconds"] = (
                before - self._stage3_update_started
                - (self._stage3_diag_seconds - self._stage3_update_diag_start))
            self._stage3_metrics["diagnostics_seconds"] = self._stage3_diag_seconds

    def _stage3_check_update_numeric(self, losses=None, actor=False, critic=False):
        bad = {}
        for name, value in (losses or {}).items():
            scalar = value.detach().item() if torch.is_tensor(value) else float(value)
            if not math.isfinite(scalar):
                bad[name] = repr(scalar)
        for name, module, check in (("actor", self.model.actor_ft, actor),
                                     ("critic", self.model.critic, critic)):
            if check and any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                             for p in module.parameters()):
                bad[name + "_gradient"] = "nonfinite"
        if bad:
            failure = {"itr": self.itr, "reason": "nonfinite", "values": bad,
                       "last_diagnostics": {key: repr(value) for key, value in self._stage3_metrics.items()}}
            (Path(self.logdir) / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise FloatingPointError("nonfinite Stage-3 update: " + repr(bad))

    def _stage3_record(self, record, eval_mode):
        if self.stage3_diag:
            if eval_mode:
                self._stage3_metrics.update(diagnostics_seconds=self._stage3_diag_seconds, update_seconds=0.0)
            record.update(self._stage3_metrics)
            record["actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]
            log.info("STAGE3_METRICS " + json.dumps(self._stage3_metrics, sort_keys=True))

    def _stage3_check_finite(self, record):
        if not self.stage3_active:
            return
        bad = {key: repr(value) for key, value in record.items()
               if isinstance(value, (float, np.floating)) and not math.isfinite(value)}
        if bad:
            failure = {"itr": self.itr, "reason": "nonfinite", "values": bad}
            (Path(self.logdir) / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise FloatingPointError("nonfinite Stage-3 metrics: " + repr(bad))

    def _stage3_collect(self, deterministic=False, n_steps=None, update_scaler=True):
        """Fresh complete batches for measurement modes; actor is never stepped."""
        started = time.perf_counter()
        steps = self.n_steps if n_steps is None else int(n_steps)
        self.model.eval() if deterministic else self.model.train()
        if deterministic or not hasattr(self, "_stage3_mode_obs"):
            self._stage3_mode_obs = self.reset_env_all()
            self._stage3_mode_done = np.ones(self.n_envs)
        firsts = np.zeros((steps + 1, self.n_envs))
        firsts[0] = self._stage3_mode_done
        observations = np.zeros((steps, self.n_envs, self.n_cond_step, self.obs_dim))
        chains = np.zeros((steps, self.n_envs, self.model.ft_denoising_steps + 1,
                           self.horizon_steps, self.action_dim))
        rewards = np.zeros((steps, self.n_envs))
        terminated = np.zeros_like(rewards)
        truncated = np.zeros_like(rewards)
        actions = np.zeros((steps, self.n_envs, self.act_steps, self.action_dim))
        with torch.no_grad():
            for t in range(steps):
                if t % 100 == 0:
                    log.info("STAGE3_ROLLOUT decision=%d/%d", t, steps)
                obs = self._stage3_mode_obs
                cond = {"state": torch.from_numpy(obs["state"]).float().to(self.device)}
                sample = self.model(cond=cond, deterministic=deterministic, return_chain=True)
                action = sample.trajectories.cpu().numpy()[:, :self.act_steps]
                next_obs, reward, term, trunc, _ = self.venv.step(action)
                observations[t] = obs["state"]
                chains[t] = sample.chains.cpu().numpy()
                rewards[t], terminated[t], truncated[t], actions[t] = reward, term, trunc, action
                self._stage3_mode_done = term | trunc
                firsts[t + 1] = self._stage3_mode_done
                self._stage3_mode_obs = next_obs
            episode_indices(firsts)
            obs_k = {"state": torch.from_numpy(observations).float().to(self.device).reshape(
                -1, self.n_cond_step, self.obs_dim)}
            chains_k = torch.from_numpy(chains).float().to(self.device).reshape(
                -1, self.model.ft_denoising_steps + 1, self.horizon_steps, self.action_dim)
            values_parts, logprob_parts = [], []
            for start in range(0, steps * self.n_envs, self.logprob_batch_size):
                end = start + self.logprob_batch_size
                cond = {"state": obs_k["state"][start:end]}
                values_parts.append(self.model.critic(cond).cpu().numpy().reshape(-1))
                logprob_parts.append(self.model.get_logprobs(cond, chains_k[start:end]).reshape(
                    -1, self.model.ft_denoising_steps, self.horizon_steps, self.action_dim))
            values = np.concatenate(values_parts).reshape(steps, self.n_envs).astype(np.float64)
            oldlogprobs = torch.cat(logprob_parts)
            if self.reward_scale_running:
                if update_scaler:
                    scaled = self.running_reward_scaler(rewards.T, firsts[:-1].T).T
                else:
                    scaled = self.running_reward_scaler.transform(rewards.T).T
            else:
                scaled = rewards.copy()
            advantages = np.zeros_like(rewards)
            last = 0
            final_values = self.model.critic({"state": torch.from_numpy(
                self._stage3_mode_obs["state"]).float().to(self.device)}).reshape(1, -1).cpu().numpy()
            for t in reversed(range(steps)):
                next_values = final_values if t == steps - 1 else values[t + 1]
                nonterminal = 1 - terminated[t]
                delta = scaled[t] * self.reward_scale_const + self.gamma * next_values * nonterminal - values[t]
                advantages[t] = last = delta + self.gamma * self.gae_lambda * nonterminal * last
            returns = advantages + values
        return {"raw_rewards": rewards, "firsts": firsts, "terminated": terminated,
                "truncated": truncated, "obs_k": obs_k, "chains_k": chains_k,
                "oldlogprobs": oldlogprobs, "values": values, "gae_adv": advantages,
                "returns": returns, "scaled_rewards": scaled, "actions": actions,
                "timing": {"rollout_seconds": time.perf_counter() - started}}

    def _stage3_critic_update(self, batch):
        started = time.perf_counter()
        total_decisions = batch["chains_k"].shape[0]
        denoising = self.model.ft_denoising_steps
        returns = torch.as_tensor(batch["returns"], device=self.device, dtype=torch.float32).reshape(-1)
        values = torch.as_tensor(batch["values"], device=self.device, dtype=torch.float32).reshape(-1)
        advantages = torch.as_tensor(batch["gae_adv"], device=self.device, dtype=torch.float32).reshape(-1)
        steps = 0
        for epoch in range(self.update_epochs):
            indices = torch.randperm(total_decisions * denoising, device=self.device)
            for start in range(0, len(indices), self.batch_size):
                pairs = indices[start:start + self.batch_size]
                decisions, denoise = torch.div(pairs, denoising, rounding_mode="floor"), pairs % denoising
                losses = self.model.loss(
                    {key: value[decisions] for key, value in batch["obs_k"].items()},
                    batch["chains_k"][decisions, denoise], batch["chains_k"][decisions, denoise + 1],
                    denoise, returns[decisions], values[decisions], advantages[decisions],
                    batch["oldlogprobs"][decisions, denoise], use_bc_loss=False,
                    reward_horizon=self.reward_horizon)
                self.critic_optimizer.zero_grad()
                (losses[2] * self.vf_coef).backward()
                self.critic_optimizer.step()
                steps += 1
        self.critic_lr_scheduler.step()
        return {"critic_steps": steps, "update_seconds": time.perf_counter() - started}
