"""Non-training Stage-3 protocols; sampling remains the DPPO sampler."""

import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from agent.finetune.stage3_math import (
    actor_gradient, as_episodes, batch_noise_summary, discounted_episode_returns,
    episode_indices, loo_advantages, mc_returns, mean_se,
)

log = logging.getLogger(__name__)
VARIANTS = ("nc4", "mc_critic", "mc_loo", "theory_loo", "theory_nobase")


def _strict_json(value):
    if isinstance(value, dict):
        return {str(k): _strict_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number cannot be written as scientific output")
    return value


def _write_json(path, data):
    Path(path).write_text(json.dumps(_strict_json(data), indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def _flat_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.actor_ft.parameters()]).clone()


@torch.no_grad()
def _set_parameters(model, vector):
    offset = 0
    for parameter in model.actor_ft.parameters():
        parameter.copy_(vector[offset:offset + parameter.numel()].view_as(parameter))
        offset += parameter.numel()
    if offset != len(vector):
        raise ValueError("actor parameter-vector length mismatch")


def _generator(seed, iteration, purpose):
    # SeedSequence does not use any global Python/NumPy/torch generator.
    state = np.random.SeedSequence([int(seed), int(iteration), int(purpose)]).generate_state(2)
    local_seed = (int(state[0]) | (int(state[1]) << 32)) % (2 ** 63 - 1)
    return torch.Generator(device="cpu").manual_seed(local_seed)


def diagnostic_pairs(agent, batch, iteration, purpose=37, count=20000):
    count_available = batch["chains_k"].shape[0] * agent.model.ft_denoising_steps
    if count_available < count:
        raise ValueError("insufficient chain pairs for prescribed diagnostic subsample")
    return torch.randperm(count_available, generator=_generator(agent.seed, iteration, purpose))[:count].to(agent.device)


@torch.no_grad()
def sampled_kl(agent, batch, pairs):
    denoising = agent.model.ft_denoising_steps
    decisions = torch.div(pairs, denoising, rounding_mode="floor")
    steps = pairs % denoising
    new = agent.model.get_logprobs_subsample(
        {key: value[decisions] for key, value in batch["obs_k"].items()},
        batch["chains_k"][decisions, steps], batch["chains_k"][decisions, steps + 1], steps)
    old = batch["oldlogprobs"][decisions, steps]
    difference = new.double().sum(dim=(-1, -2)) - old.double().sum(dim=(-1, -2))
    value = (denoising * (torch.expm1(difference) - difference).mean()).item()
    if not math.isfinite(value):
        raise FloatingPointError("non-finite calibration KL")
    return value


@torch.no_grad()
def saturation_metrics(agent, batch, pairs):
    """Raw predicted x0 before clamping, on actual stored fine-tuned chains."""
    model = agent.model
    if model.use_ddim or not model.predict_epsilon:
        raise ValueError("Stage-3 saturation formula requires the prescribed DDPM epsilon model")
    steps = pairs % model.ft_denoising_steps
    decisions = torch.div(pairs, model.ft_denoising_steps, rounding_mode="floor")
    timesteps = model.ft_denoising_steps - 1 - steps
    x = batch["chains_k"][decisions, steps]
    noise = model.actor_ft(x, timesteps, cond={key: value[decisions]
                                             for key, value in batch["obs_k"].items()})
    sqrt_recip = model.sqrt_recip_alphas_cumprod[timesteps].reshape(-1, 1, 1)
    sqrt_recipm1 = model.sqrt_recipm1_alphas_cumprod[timesteps].reshape(-1, 1, 1)
    raw = sqrt_recip * x - sqrt_recipm1 * noise
    if not torch.isfinite(raw).all():
        raise FloatingPointError("non-finite raw x0 prediction")
    saturated = raw.abs() > model.denoised_clip_value
    last = timesteps == 0
    return {"x0_sat_frac_mean": saturated.double().mean().item(),
            "x0_sat_frac_last": saturated[last].double().mean().item() if last.any() else None,
            "x0_sat_pairs": int(len(pairs)), "x0_sat_last_pairs": int(last.sum().item())}


def _explained_variance(batch):
    targets = np.asarray(batch["returns"], dtype=np.float64).reshape(-1)
    values = np.asarray(batch["values"], dtype=np.float64).reshape(-1)
    variance = targets.var()
    return float(1 - (targets - values).var() / variance) if variance > 0 else None


def _record_progress(agent, mode, phase, completed, target, started, **values):
    state = {"mode": mode, "phase": phase, "completed": completed, "target": target,
             "elapsed_seconds": time.monotonic() - started, **values}
    _write_json(Path(agent.logdir) / "mode_progress.json", state)
    log.info("stage3_mode_progress %s", json.dumps(_strict_json(state), sort_keys=True))


def _variant_coefficients(agent, batch):
    firsts = batch["firsts"]
    k = episode_indices(firsts)
    raw_returns = mc_returns(batch["raw_rewards"], firsts, agent.gamma)
    loo, _ = loo_advantages(raw_returns, firsts)
    scaled_returns = mc_returns(np.asarray(batch["scaled_rewards"]) * agent.reward_scale_const,
                                firsts, agent.gamma)
    gamma_k = agent.gamma ** k
    return (
        (batch["gae_adv"], "mean", 0.99, True),
        (scaled_returns - batch["values"], "mean", 0.99, True),
        (loo, "mean", 0.99, True),
        (loo * gamma_k, "sum", 1.0, False),
        (raw_returns * gamma_k, "sum", 1.0, False),
    )


def run_noise_scale(agent):
    cfg = agent.cfg.train
    warmup = int(cfg.get("ns_warmup_batches", 20))
    batches = int(cfg.get("ns_batches", 25))
    if warmup < 0 or batches < 2:
        raise ValueError("invalid noise-scale protocol length")
    started = time.monotonic()
    actor_before = _flat_parameters(agent.model)
    directory = Path(agent.logdir)
    warmup_rows = []
    for iteration in range(warmup):
        batch_started = time.monotonic()
        batch = agent._stage3_collect(deterministic=False, update_scaler=True)
        episode_indices(batch["firsts"])
        updates = agent._stage3_critic_update(batch)
        row = {"batch": iteration, "seconds": time.monotonic() - batch_started,
               "explained_var_preupdate": _explained_variance(batch)}
        if isinstance(updates, dict):
            row["critic_optimizer_steps"] = updates.get("critic_optimizer_steps")
        warmup_rows.append(row)
        _record_progress(agent, "noise_scale", "critic_warmup", iteration + 1, warmup, started)
    if not torch.equal(actor_before, _flat_parameters(agent.model)):
        raise RuntimeError("noise-scale warmup changed the frozen actor")
    # Freeze by not stepping parameters/scaler. Autograd still needs actor parameters.
    critic_before = {key: value.detach().clone() for key, value in agent.model.critic.state_dict().items()}
    scaler_before = (agent.running_reward_scaler.ret_rms.var.copy(),
                     agent.running_reward_scaler.ret_rms.mean.copy(),
                     agent.running_reward_scaler.ret_rms.count,
                     agent.running_reward_scaler.ret.copy())
    parameter_count = actor_before.numel()
    batch_sums = np.lib.format.open_memmap(directory / "noise_batch_vector_sums.npy", mode="w+",
                                         dtype=np.float64, shape=(batches, len(VARIANTS), parameter_count))
    cross_sums = np.zeros((batches, len(VARIANTS), len(VARIANTS)), dtype=np.float64)
    rows = []
    for iteration in range(batches):
        batch_started = time.monotonic()
        batch = agent._stage3_collect(deterministic=False, update_scaler=False)
        coefficients = _variant_coefficients(agent, batch)
        batch_sums[iteration] = 0
        for environment in range(agent.n_envs):
            vectors = []
            for advantages, reduction, discount, normalize in coefficients:
                gradient = actor_gradient(agent, batch, advantages, reduce=reduction,
                                          gamma_denoising=discount, normalize=normalize,
                                          env_indices=[environment])
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("non-finite noise-scale gradient at batch %d env %d" %
                                             (iteration, environment))
                vectors.append(gradient.double().cpu().numpy())
            vectors = np.stack(vectors)
            batch_sums[iteration] += vectors
            cross_sums[iteration] += vectors @ vectors.T
            if (environment + 1) % 10 == 0:
                _record_progress(agent, "noise_scale", "measure", iteration, batches, started,
                                 environments_completed=environment + 1)
        batch_sums.flush()
        row = {"batch": iteration, "seconds": time.monotonic() - batch_started,
               "explained_var": _explained_variance(batch),
               "J_disc_train": mean_se(discounted_episode_returns(batch["raw_rewards"], batch["firsts"], agent.gamma))}
        rows.append(row)
        _write_json(directory / "noise_measurements.json", rows)
        np.save(directory / "noise_batch_scalar_cross_sums.npy", cross_sums)
        _record_progress(agent, "noise_scale", "measure", iteration + 1, batches, started)
    if not torch.equal(actor_before, _flat_parameters(agent.model)):
        raise RuntimeError("noise-scale measurement changed the frozen actor")
    if any(not torch.equal(value, agent.model.critic.state_dict()[key]) for key, value in critic_before.items()):
        raise RuntimeError("noise-scale measurement changed the frozen critic")
    scaler_after = (agent.running_reward_scaler.ret_rms.var, agent.running_reward_scaler.ret_rms.mean,
                    agent.running_reward_scaler.ret_rms.count, agent.running_reward_scaler.ret)
    if any(not np.array_equal(before, after) for before, after in zip(scaler_before, scaler_after)):
        raise RuntimeError("noise-scale measurement changed the frozen reward scaler")
    _record_progress(agent, "noise_scale", "bootstrap", 0, 2000, started)
    summary = batch_noise_summary(batch_sums, cross_sums, agent.n_envs, agent.n_steps // 250,
                                  bootstrap_reps=2000, seed=agent.seed)
    gram = summary.pop("gram")
    np.save(directory / "noise_batch_gram.npy", gram)
    for name, row in zip(VARIANTS, summary["variants"]):
        row["variant"] = name
    summary.update({"mode": "noise_scale", "status": "completed", "seed": agent.seed,
                    "variant_order": list(VARIANTS), "warmup_batches": warmup,
                    "measurement_batches": batches, "warmup": warmup_rows,
                    "measurements": rows, "critic_explained_var_after_warmup": rows[0]["explained_var"],
                    "critic_explained_var_mean_measurement": float(np.mean([r["explained_var"] for r in rows if r["explained_var"] is not None]))
                        if any(r["explained_var"] is not None for r in rows) else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "sampling": "train sampler; frozen actor, critic and reward scaler during measurement",
                    "gradient_units": "negative mean loss gradient per 5000-pair environment rollout; theory variants equal minus paper estimator / 2500",
                    "artifacts": {"batch_vector_sums": "noise_batch_vector_sums.npy",
                                  "batch_scalar_cross_sums": "noise_batch_scalar_cross_sums.npy", "gram": "noise_batch_gram.npy"}})
    _write_json(directory / "noise_scale.json", summary)
    _write_csv(directory / "noise_scale.csv", summary["variants"])
    lines = ["# Fixed-policy gradient noise", "",
             "Twenty-five fresh batches are the bootstrap units; the 1,000 environment gradients are coupled within a batch by normalization and/or LOO. The specified n-based signal correction is consequently approximate for those variants.", "",
             "| Variant | Signal norm squared [90% CI] | Noise trace | B_env | B_ep | Batch cosine |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary["variants"]:
        lines.append("| {variant} | {u_sq:.6g} {u_sq_ci90} | {tr_sigma:.6g} | {B_env} | {B_ep} | {expected_batch_cosine} |".format(**row))
    lines.extend(["", "Null ratios indicate unresolved positive signal; signed signal estimates are retained. See noise_scale.json for lower bounds, all confidence intervals, the direction-cosine matrix and warmed-critic explained variance.", "", summary["bound_note"], "", summary["cosine_note"]])
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    _record_progress(agent, "noise_scale", "completed", batches, batches, started)
    return summary


def run_calibrate(agent):
    target = float(agent.cfg.train.get("calib_kl_target"))
    if not math.isfinite(target) or target <= 0:
        raise ValueError("calib_kl_target must be finite and positive")
    directory = Path(agent.logdir)
    origin = _flat_parameters(agent.model)
    norm = torch.linalg.vector_norm(origin.double()).item()
    started = time.monotonic()
    rows = []
    for iteration in range(5):
        batch_started = time.monotonic()
        batch = agent._stage3_collect(deterministic=False, update_scaler=False)
        k = episode_indices(batch["firsts"])
        returns = mc_returns(batch["raw_rewards"], batch["firsts"], agent.gamma)
        advantages, _ = loo_advantages(returns, batch["firsts"])
        gradient = actor_gradient(agent, batch, advantages * agent.gamma ** k,
                                  reduce="sum", gamma_denoising=1., normalize=False)
        grad_norm = torch.linalg.vector_norm(gradient.double()).item()
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise FloatingPointError("calibration requires a nonzero finite E2 gradient")
        pairs = diagnostic_pairs(agent, batch, iteration, purpose=3000)
        trials = []

        def evaluate(eta, label):
            if not math.isfinite(eta) or eta <= 0:
                raise FloatingPointError("invalid calibration trial step size")
            try:
                _set_parameters(agent.model, origin - eta * gradient)
                kl = sampled_kl(agent, batch, pairs)
            finally:
                _set_parameters(agent.model, origin)
            trials.append({"eta": float(eta), "kl": kl, "label": label})
            return kl

        eta0 = 1e-4 * norm / grad_norm
        kl0 = evaluate(eta0, "initial")
        bracket_trials = 0
        while not 1e-6 <= kl0 <= 1e-2:
            bracket_trials += 1
            if bracket_trials > 20:
                raise RuntimeError("calibration failed to bracket KL after 20 decade changes")
            eta0 *= 10 if kl0 < 1e-6 else 0.1
            kl0 = evaluate(eta0, "bracket")
        kl10 = evaluate(10 * eta0, "exponent_probe")
        exponent = math.log(kl10 / kl0) / math.log(10) if kl10 > 0 else None
        eta_b = eta0 * math.sqrt(target / kl0)
        kl_b = evaluate(eta_b, "quadratic_prediction")
        warnings = []
        if not (0.5 * target <= kl_b <= 2 * target):
            if kl_b > 0 and eta_b != eta0:
                slope = math.log(kl_b / kl0) / math.log(eta_b / eta0)
            else:
                slope = exponent
            if slope is None or slope <= 0 or not math.isfinite(slope) or kl_b <= 0:
                warnings.append("log-log secant undefined; retaining the quadratic step")
            else:
                eta_b = math.exp(math.log(eta_b) + (math.log(target) - math.log(kl_b)) / slope)
                kl_b = evaluate(eta_b, "one_loglog_secant")
        if exponent is None or not 1.5 <= exponent <= 2.5:
            warnings.append("KL exponent is outside [1.5, 2.5]")
        if not 0.5 * target <= kl_b <= 2 * target:
            warnings.append("final trial KL is outside [0.5, 2] times target after the permitted correction")
        row = {"batch": iteration, "eta0": eta0, "kl_eta0": kl0,
               "kl_10eta0": kl10, "exponent": exponent, "eta_b": eta_b,
               "eta_theory_b": eta_b / 2500, "kl_eta_b": kl_b,
               "gradient_norm": grad_norm, "theta0_norm": norm,
               "seconds": time.monotonic() - batch_started, "trials": trials,
               "warnings": warnings}
        rows.append(row)
        _write_json(directory / "calibration_batches.json", rows)
        _record_progress(agent, "calibrate", "calibrate", iteration + 1, 5, started,
                         eta_b=eta_b, target_kl=target, realized_kl=kl_b, warnings=warnings)
    if not torch.equal(origin, _flat_parameters(agent.model)):
        raise RuntimeError("calibration did not restore the actor")
    eta_star = float(np.median([row["eta_b"] for row in rows]))
    spread = max(row["eta_b"] for row in rows) / min(row["eta_b"] for row in rows)
    summary = {"mode": "calibrate", "status": "completed", "seed": agent.seed,
               "kl_target": target, "eta_star": eta_star, "eta_star_theory": eta_star / 2500,
               "eta_spread_max_over_min": spread, "batches": rows,
               "warnings": ["eta_b spread exceeds 3x"] if spread > 3 else [],
               "elapsed_seconds": time.monotonic() - started,
               "gradient_units": "code gradient equals negative paper gradient / 2500",
               "protocol": "Five fresh train-sampler batches; same 20,000 raw coordinate-sum logprob pairs for every trial within a batch; actor restored after each trial; no optimizer step."}
    _write_json(directory / "calibration.json", summary)
    _write_csv(directory / "calibration.csv", [{k: v for k, v in row.items() if k != "trials"} for row in rows])
    lines = ["# SGD step calibration", "", "Target sampled transition KL: %.9g. Median eta: %.9g; eta_theory: %.9g; max/min spread: %.6g." %
             (target, eta_star, eta_star / 2500, spread), "",
             "| Batch | eta | eta_theory | Realized KL | KL exponent |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {batch} | {eta_b:.9g} | {eta_theory_b:.9g} | {kl_eta_b:.9g} | {exponent} |".format(**row))
    lines.extend(["", "No actor or optimizer update is retained. Per-trial data and calibration warnings are in calibration.json.", "", "Warnings: " + json.dumps(summary["warnings"] + [warning for row in rows for warning in row["warnings"]])])
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    _record_progress(agent, "calibrate", "completed", 5, 5, started)
    return summary


def run_ckpt_eval(agent):
    repeats = int(agent.cfg.train.get("ckpt_eval_repeats", 1))
    if repeats not in (1, 3):
        raise ValueError("ckpt_eval_repeats must be 1 (final checkpoint) or 3 (pretrained)")
    directory = Path(agent.logdir)
    started = time.monotonic()
    rows, pooled = [], {}
    origin = _flat_parameters(agent.model)
    for sampler, deterministic in (("train", False), ("eval", True)):
        all_returns, all_discounted = [], []
        for repetition in range(repeats):
            batch_started = time.monotonic()
            batch = agent._stage3_collect(deterministic=deterministic, n_steps=500, update_scaler=False)
            rewards = as_episodes(batch["raw_rewards"], batch["firsts"])
            undiscounted = rewards.sum(axis=1)
            discounted = discounted_episode_returns(batch["raw_rewards"], batch["firsts"], agent.gamma)
            if len(undiscounted) != 80:
                raise ValueError("checkpoint evaluation requires exactly 80 episodes per rollout")
            pairs = diagnostic_pairs(agent, batch, repetition + int(deterministic) * repeats, purpose=5000)
            saturation = saturation_metrics(agent, batch, pairs)
            actions = np.asarray(batch["actions"])
            row = {"sampler": sampler, "deterministic_flag": deterministic, "repeat": repetition,
                   "undiscounted_return": mean_se(undiscounted), "J_disc": mean_se(discounted),
                   "x0_sat_frac": saturation["x0_sat_frac_mean"], **saturation,
                   "action_oor_frac": float(np.mean(np.abs(actions) > 1)),
                   "episode_returns": undiscounted.tolist(), "episode_J_disc": discounted.tolist(),
                   "seconds": time.monotonic() - batch_started}
            rows.append(row)
            all_returns.extend(undiscounted.tolist())
            all_discounted.extend(discounted.tolist())
            _write_json(directory / "ckpt_eval_rollouts.json", rows)
            _record_progress(agent, "ckpt_eval", sampler, repetition + 1, repeats, started)
        selected = [row for row in rows if row["sampler"] == sampler]
        pooled[sampler] = {"undiscounted_return": mean_se(all_returns), "J_disc": mean_se(all_discounted),
                           "x0_sat_frac": float(np.mean([row["x0_sat_frac"] for row in selected])),
                           "x0_sat_frac_last": float(np.mean([row["x0_sat_frac_last"] for row in selected])),
                           "action_oor_frac": float(np.mean([row["action_oor_frac"] for row in selected]))}
    if not torch.equal(origin, _flat_parameters(agent.model)):
        raise RuntimeError("checkpoint evaluation changed the actor")
    summary = {"mode": "ckpt_eval", "status": "completed", "seed": agent.seed,
               "checkpoint": str(agent.cfg.base_policy_path), "repeats_per_sampler": repeats,
               "samplers": pooled, "rollouts": rows, "elapsed_seconds": time.monotonic() - started,
               "sampling_note": "Seed initialized once; continuous RNG across train repeats then eval repeats. DDPM deterministic=True still adds noise at t>0, and zero noise at t=0.",
               "uncertainty": "Episode standard error uses sample SD (ddof=1)/sqrt(n); pretrained pooled summaries contain 240 episodes per sampler, each final checkpoint 80.",
               "action_units": "Normalized policy action coordinates sent to the environment wrapper, before its affine action unnormalization."}
    _write_json(directory / "ckpt_eval.json", summary)
    _write_csv(directory / "ckpt_eval.csv", [{"sampler": key,
                "undiscounted_mean": values["undiscounted_return"]["mean"],
                "undiscounted_se": values["undiscounted_return"]["se"],
                "J_disc_mean": values["J_disc"]["mean"], "J_disc_se": values["J_disc"]["se"],
                "n_episodes": values["J_disc"]["n"], "x0_sat_frac": values["x0_sat_frac"],
                "x0_sat_frac_last": values["x0_sat_frac_last"], "action_oor_frac": values["action_oor_frac"]}
                for key, values in pooled.items()])
    lines = ["# Final-checkpoint evaluation", "", "| Sampler | Undiscounted return (mean ± s.e.) | Discounted return (mean ± s.e.) | x0 saturation |",
             "|---|---:|---:|---:|"]
    for sampler, values in pooled.items():
        lines.append("| %s | %.6g ± %.6g | %.6g ± %.6g | %.6g |" %
                     (sampler, values["undiscounted_return"]["mean"], values["undiscounted_return"]["se"],
                      values["J_disc"]["mean"], values["J_disc"]["se"], values["x0_sat_frac"]))
    lines.extend(["", summary["sampling_note"], "", summary["uncertainty"], "", summary["action_units"]])
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    _record_progress(agent, "ckpt_eval", "completed", 2 * repeats, 2 * repeats, started)
    return summary


def run_stage3_mode(agent):
    mode = agent.cfg.train.get("mode", "train")
    dispatch = {"noise_scale": run_noise_scale, "calibrate": run_calibrate,
                "ckpt_eval": run_ckpt_eval}
    if mode not in dispatch:
        raise ValueError("unsupported non-training mode: %s" % mode)
    return dispatch[mode](agent)
