"""Approved whole-batch gradient-mean collection, isolated from training.

All vectors are CODE ASCENT gradients: the negative of the inherited Phase-1
loss gradient. V4/V5 equal the episode-average paper estimator divided by2500.
Only the explicitly requested frozen-file and frozen-state checks use SHA256.
"""
import csv
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time
import traceback

import numpy as np
import torch

from agent.finetune.stage3_math import (
    actor_gradient, as_episodes, discounted_episode_returns, episode_indices,
    mean_se, mc_returns, loo_advantages,
)
from agent.finetune.stage3_modes import (
    VARIANTS, _explained_variance, _generator, _strict_json, _variant_coefficients,
    diagnostic_pairs, saturation_metrics,
)


log = logging.getLogger(__name__)
POINTS = ("P0", "D1", "D2", "N1", "N2")
VERSION = 1


def settings(agent):
    cfg = agent.cfg.train
    point = cfg.get("gm_point", None)
    return {"mode": "grad_mean", "gm_step": cfg.get("gm_step", "collect"),
            "gm_point": point,
            "gm_warmup_batches": cfg.get("gm_warmup_batches", 20 if point == "P0" else 5),
            "gm_batches": cfg.get("gm_batches", 50),
            "gm_frozen_path": cfg.get("gm_frozen_path", None),
            "gm_frozen_sha256": cfg.get("gm_frozen_sha256", None),
            "gm_score_sample": cfg.get("gm_score_sample", 0),
            "gm_include_critic": cfg.get("gm_include_critic", False),
            "gm_parent_commit": cfg.get("gm_parent_commit", None),
            "seed": int(agent.seed), "environment_seeds": [int(agent.seed) + i for i in range(agent.n_envs)],
            "normalization": "full 200000-pair batch, unbiased SD, before environment selection",
            "gradient_sign": "ascent = negative inherited Phase1 loss gradient",
            "stored_vector": "float32 group SUM after float64 accumulation; not a mean",
            "model_parameter_dtype": "float32", "statistics_dtype": "float64"}


def _copy(value):
    """CPU tensor/scalar payload compatible with weights_only=True loading."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _copy(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported frozen-state value: " + type(value).__name__)


def state_sha256(value):
    """Canonical tensor/scalar state hash, independent of torch-save layout."""
    digest = hashlib.sha256()
    def visit(child):
        if torch.is_tensor(child):
            array = child.detach().cpu().contiguous().numpy()
            visit(array)
        elif isinstance(child, np.ndarray):
            array = np.ascontiguousarray(child)
            digest.update(b"array\0" + str(array.dtype).encode() + b"\0")
            digest.update(json.dumps(list(array.shape)).encode() + b"\0" + array.tobytes())
        elif isinstance(child, dict):
            digest.update(b"dict\0")
            for key in sorted(child):
                visit(str(key))
                visit(child[key])
            digest.update(b"enddict\0")
        elif isinstance(child, (list, tuple)):
            digest.update(b"list\0")
            for item in child:
                visit(item)
            digest.update(b"endlist\0")
        elif isinstance(child, np.generic):
            visit(child.item())
        elif child is None or isinstance(child, (str, int, float, bool)):
            digest.update(type(child).__name__.encode() + b"\0" +
                          json.dumps(child, allow_nan=False).encode() + b"\0")
        else:
            raise TypeError("Unsupported hashed frozen state: " + type(child).__name__)
    visit(value)
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_state(agent):
    return {"actor": _copy(agent.model.actor.state_dict()),
            "actor_ft": _copy(agent.model.actor_ft.state_dict())}


def scaler_state(agent):
    scaler = agent.running_reward_scaler
    return _copy({"ret_rms": vars(scaler.ret_rms),
                  **{key: value for key, value in vars(scaler).items() if key != "ret_rms"}})


def state_hashes(agent):
    states = {"actor": actor_state(agent), "critic": agent.model.critic.state_dict(),
              "scaler": scaler_state(agent)}
    def finite(value):
        if isinstance(value, dict):
            return all(finite(child) for child in value.values())
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, np.ndarray):
            return bool(np.isfinite(value).all())
        if isinstance(value, (float, np.floating)):
            return math.isfinite(value)
        return True
    if not all(finite(value) for value in states.values()):
        raise FloatingPointError("Nonfinite actor/critic/scaler frozen state")
    return {name + "_sha256": state_sha256(value) for name, value in states.items()}


def _restore_scaler(agent, payload):
    def values(mapping):
        return {key: child.detach().cpu().numpy().copy() if torch.is_tensor(child) else child
                for key, child in mapping.items()}
    agent.running_reward_scaler.ret_rms.__dict__.update(values(payload["ret_rms"]))
    agent.running_reward_scaler.__dict__.update(values({k: v for k, v in payload.items() if k != "ret_rms"}))


def _specs(agent):
    return [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel()}
            for name, p in agent.model.actor_ft.named_parameters()]


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w") as stream:
        json.dump(_strict_json(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _atomic_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in _strict_json(row).items()})
    temporary.replace(path)


def _save_vector(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _progress(agent, phase, completed, target, started, status="running", **extra):
    value = {"mode": "grad_mean", "step": agent.cfg.train.get("gm_step", "collect"),
             "phase": phase, "substage": phase, "completed": completed, "done": completed,
             "target": target, "total": target, "status": status,
             "elapsed_seconds": time.monotonic() - started, **extra}
    _atomic_json(Path(agent.logdir) / "stage3_progress.json", value)
    log.info("stage3_mode_progress %s", json.dumps(_strict_json(value), sort_keys=True))


def seed_environments(agent):
    """Seed private Gym RNGs only for this new mode, before its first reset."""
    if hasattr(agent, "_stage3_mode_obs"):
        raise RuntimeError("grad_mean must start before any measurement-mode rollout")
    seeds = [int(agent.seed) + i for i in range(agent.n_envs)]
    responses = agent.venv.call_sync_arg("grad_mean_seed", "seed", seeds,
                                         indices=list(range(agent.n_envs)))
    if len(responses) != agent.n_envs or any(not ok or value != {
            "seed": seed, "underlying_env_seeded": True, "action_space_seeded": True}
            for (value, ok), seed in zip(responses, seeds)):
        raise RuntimeError("grad_mean underlying environment seeding failed")
    return seeds


def batch_gradients(agent, batch, env_indices=None):
    """Five float64 ascent vectors; model/autograd computations remain float32.

    Full-batch coefficients and V1-V3 normalization precede optional environment
    selection, exactly as in the inherited Phase1 measurement estimator.
    """
    vectors = []
    for advantages, reduction, discount, normalize in _variant_coefficients(agent, batch):
        gradient = actor_gradient(agent, batch, advantages, reduce=reduction,
                                  gamma_denoising=discount, normalize=normalize,
                                  env_indices=env_indices, clamp=False)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite gradient in V%d" % (len(vectors) + 1))
        vectors.append(-gradient.detach().double().cpu().numpy())
    return np.stack(vectors)


def frozen_critic_baseline(agent, batch):
    """Convert the pre-update frozen critic to raw reward units in float64."""
    scaler = agent.running_reward_scaler
    variance = np.asarray(scaler.ret_rms.var, dtype=np.float64)
    if variance.size != 1 or scaler.per_env:
        raise ValueError("V6 requires the approved scalar frozen reward divisor")
    divisor = float(np.sqrt(variance.reshape(-1)[0] + float(scaler.epsilon)))
    if not math.isfinite(divisor) or divisor <= 0:
        raise FloatingPointError("Nonfinite/nonpositive frozen reward divisor")
    return np.asarray(batch["values"], dtype=np.float64) * divisor, divisor


def theory_critic_components(agent, batch):
    """Float64 MC residuals; actor cast/discount order matches existing V4/V5."""
    firsts = batch["firsts"]
    k = episode_indices(firsts)
    returns = mc_returns(np.asarray(batch["raw_rewards"], dtype=np.float64), firsts, agent.gamma)
    baseline, divisor = frozen_critic_baseline(agent, batch)
    residual = returns - baseline
    loo, _ = loo_advantages(returns, firsts)
    discount = agent.gamma ** k
    coefficient = (torch.as_tensor(residual, device=agent.device, dtype=torch.float32)
                   * torch.as_tensor(discount, device=agent.device, dtype=torch.float32))
    return {"returns": returns, "critic_baseline": baseline, "critic_residual": residual,
            "loo_residual": loo, "weighted_float64": residual * discount,
            "coefficient": coefficient, "divisor": divisor, "k": k}


def critic_gradient(agent, batch, components=None):
    components = theory_critic_components(agent, batch) if components is None else components
    gradient = actor_gradient(agent, batch, components["coefficient"], reduce="sum",
                              gamma_denoising=1.0, normalize=False, clamp=False)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Nonfinite gradient in V6")
    return -gradient.detach().double().cpu().numpy()


def critic_baseline_metrics(agent, batch, components):
    """Residual means and discounted energies on the same frozen-policy batch."""
    firsts = batch["firsts"]
    critic = as_episodes(components["critic_residual"], firsts)
    loo = as_episodes(components["loo_residual"], firsts)
    weights = agent.gamma ** (2 * np.arange(critic.shape[1], dtype=np.float64))
    result = {"frozen_reward_divisor": components["divisor"]}
    totals = {"critic": 0.0, "loo": 0.0}
    for start in range(0, 250, 50):
        stop = start + 50
        suffix = "_k%d_%d" % (start, stop)
        energies = {}
        for name, values in (("critic", critic), ("loo", loo)):
            section = values[:, start:stop]
            result["baseline_" + name + "_residual_mean" + suffix] = float(section.mean(dtype=np.float64))
            energy = float(np.mean(np.sum(section ** 2 * weights[start:stop], axis=1, dtype=np.float64), dtype=np.float64))
            result["baseline_" + name + "_energy" + suffix] = energy
            energies[name] = energy
            totals[name] += energy
        result["baseline_energy_ratio" + suffix] = energies["critic"] / energies["loo"] if energies["loo"] > 0 else None
    for name, energy in totals.items():
        result["baseline_" + name + "_energy"] = energy
    result["baseline_energy_ratio"] = totals["critic"] / totals["loo"] if totals["loo"] > 0 else None
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite V6 baseline diagnostic")
    return result


def decision_scores(agent, batch, count=200, iteration=0):
    """Norm of sum of all ten 24-coordinate transition scores per decision."""
    if count != 200:
        raise ValueError("Approved decision-score sample contains exactly200 decisions")
    n = batch["chains_k"].shape[0]
    selected = torch.randperm(n, generator=_generator(agent.seed, iteration, 7711))[:count]
    steps = torch.arange(agent.model.ft_denoising_steps, device=agent.device)
    parameters = list(agent.model.actor_ft.parameters())
    rows = []
    for index in selected.tolist():
        decisions = torch.full_like(steps, index)
        logp = agent.model.get_logprobs_subsample(
            {key: value[decisions] for key, value in batch["obs_k"].items()},
            batch["chains_k"][decisions, steps], batch["chains_k"][decisions, steps + 1], steps)
        score = torch.autograd.grad(logp.sum(), parameters, allow_unused=True)
        squared = sum(value.detach().double().square().sum().item() for value in score if value is not None)
        if not math.isfinite(squared):
            raise FloatingPointError("Nonfinite per-decision score")
        t = index // agent.n_envs
        rows.append({"batch": iteration, "decision_index": index, "environment": index % agent.n_envs,
                     "decision_t": t, "episode_within_environment": t // 250, "k": t % 250,
                     "score_norm": math.sqrt(squared), "score_norm_sq": squared,
                     "denoising_steps": int(agent.model.ft_denoising_steps), "coordinates_per_step": 24})
    return rows


def _batch_metrics(agent, batch, index):
    firsts, rewards = batch["firsts"], np.asarray(batch["raw_rewards"], dtype=np.float64)
    episode_indices(firsts)
    if any(not np.isfinite(np.asarray(batch[key])).all() for key in
           ("values", "gae_adv", "returns", "scaled_rewards")):
        raise FloatingPointError("Nonfinite critic/reward statistics in collected batch")
    returns = as_episodes(rewards, firsts).sum(axis=1, dtype=np.float64)
    discounted = discounted_episode_returns(rewards, firsts, agent.gamma)
    if len(returns) != 80 or not np.isfinite(returns).all() or not np.isfinite(discounted).all():
        raise FloatingPointError("Invalid/nonfinite complete-episode return batch")
    mean, j = mean_se(returns), mean_se(discounted)
    pairs = diagnostic_pairs(agent, batch, index, purpose=7721, count=20000)
    saturation = saturation_metrics(agent, batch, pairs)
    result = {"batch": index, "n_episodes": int(len(returns)),
              "return_mean": mean["mean"], "return_se": mean["se"],
              "return_sum": float(returns.sum(dtype=np.float64)),
              "return_sumsq": float(np.dot(returns, returns)),
              "J_disc_mean": j["mean"], "J_disc_se": j["se"],
              "J_disc_sum": float(discounted.sum(dtype=np.float64)),
              "J_disc_sumsq": float(np.dot(discounted, discounted)),
              "max_abs_reward": float(np.max(np.abs(rewards))),
              "max_abs_R": float(np.max(np.abs(rewards))),
              "explained_variance": _explained_variance(batch),
              **saturation, **agent._stage3_action_metrics(batch["actions"])}
    return result, returns


def _metadata(agent, config):
    from omegaconf import OmegaConf
    include_critic = bool(config.get("gm_include_critic", False)) and config["gm_step"] == "collect"
    variants = list(VARIANTS) + (["theory_critic"] if include_critic else [])
    return {"format_version": VERSION, "mode": "grad_mean", "step": config["gm_step"],
            "point": config["gm_point"], "seed": int(agent.seed), "settings": config,
            "variant_order": variants, "variant_ids": ["V%d" % i for i in range(1, len(variants) + 1)],
            "parameter_specs": _specs(agent), "parameter_count": sum(p.numel() for p in agent.model.actor_ft.parameters()),
            "parent_commit": config["gm_parent_commit"], "source_checkpoint": str(agent.cfg.base_policy_path),
            "config": OmegaConf.to_container(agent.cfg, resolve=True),
            "gradient_units": "CODE ASCENT = -inherited Phase1 loss gradient; V4=-native E2 loss gradient",
            "theory_multiplier_V4_V5": 2500, **({"theory_multiplier_V6": 2500} if include_critic else {}),
            "group_vector_kind": "sum",
            "group_accumulation_dtype": "float64", "stored_vector_dtype": "float32",
            "normalization": config["normalization"], "pairs_per_batch": 200000,
            "episodes_per_batch": 80, "decisions_per_episode": 250,
            "critic_and_scaler_frozen_during_collection": True,
            "policy_rng": "run seed initializes global Torch once; preparation RNG is never restored",
            "environment_seeding": "explicit grad_mean-only hook seeds private Gym and action-space RNGs with run seed+i",
            "diagnostic_rng": "owned CPU generators from SeedSequence; no global RNG consumption",
            "actor_optimizer_steps": 0}


def _validate(agent, config):
    if not isinstance(config.get("gm_include_critic", False), bool):
        raise ValueError("gm_include_critic must be a bool")
    if config["gm_step"] not in ("prepare", "collect") or config["gm_point"] not in POINTS:
        raise ValueError("grad_mean requires gm_step=prepare|collect and a valid gm_point")
    if not isinstance(config["gm_parent_commit"], str) or not config["gm_parent_commit"]:
        raise ValueError("grad_mean requires the explicit parent source revision")
    if (agent.n_envs, agent.n_steps, agent.model.ft_denoising_steps, agent.reward_horizon,
            agent.horizon_steps, agent.action_dim, agent.batch_size) != (40, 500, 10, 4, 4, 6, 50000):
        raise ValueError("grad_mean geometry differs from the approved fixed batch")
    if agent.gamma != .99 or agent.gae_lambda != .95 or not agent.reward_scale_running or agent.reward_scale_const != 1.:
        raise ValueError("grad_mean requires unchanged gamma/GAE/reward-scaler settings")
    if (agent.model.clamp_logprob or agent.model.randn_clip_value != 100 or
            agent.model.min_logprob_denoising_std != .1 or agent.model.get_min_sampling_denoising_std() != .1):
        raise ValueError("grad_mean requires unclamped logprob, noise bound100, and existing0.1 noise floors")
    if any(p.dtype != torch.float32 for p in list(agent.model.actor.parameters()) +
           list(agent.model.actor_ft.parameters()) + list(agent.model.critic.parameters())):
        raise ValueError("grad_mean model/critic parameters must remain float32")
    if sum(p.numel() for p in agent.model.actor_ft.parameters()) != 568392:
        raise ValueError("grad_mean actor geometry differs from the approved568392 coordinates")
    if int(config["gm_warmup_batches"]) != (20 if config["gm_point"] == "P0" else 5):
        raise ValueError("Preparation warmup must be20 forP0 and5 forRL checkpoints")
    if not isinstance(config["gm_batches"], int) or config["gm_batches"] != 50:
        raise ValueError("gm_batches must equal the approved50 whole batches")
    if config["gm_score_sample"] not in (0, 200):
        raise ValueError("gm_score_sample must be0 or200")


def prepare(agent, config):
    directory, started = Path(agent.logdir), time.monotonic()
    point, warmup = config["gm_point"], int(config["gm_warmup_batches"])
    frozen_path = directory / ("frozen_" + point + ".pt")
    checksum_path = directory / ("frozen_" + point + ".sha256")
    if any(p.exists() for p in (frozen_path, checksum_path, directory / "grad_mean.json")):
        raise FileExistsError("Refusing to overwrite preparation evidence")
    before_actor = state_hashes(agent)["actor_sha256"]
    meta = _metadata(agent, config)
    meta.update(status="running", completed_warmup_batches=0, actor_before_sha256=before_actor, warmup=[])
    _atomic_json(directory / "grad_mean.json", meta)
    try:
        for iteration in range(warmup):
            tick = time.monotonic()
            _progress(agent, "critic_warmup", iteration, warmup, started)
            batch = agent._stage3_collect(deterministic=False, update_scaler=True)
            episode_indices(batch["firsts"])
            explained = _explained_variance(batch)
            update = agent._stage3_critic_update(batch)
            if state_hashes(agent)["actor_sha256"] != before_actor:
                raise RuntimeError("Preparation warmup changed the frozen actor")
            meta["warmup"].append({"batch": iteration, "seconds": time.monotonic() - tick,
                                    "explained_var_preupdate": explained, **update})
            meta["completed_warmup_batches"] = iteration + 1
            _atomic_json(directory / "grad_mean.json", meta)
        frozen_hashes = state_hashes(agent)
        _progress(agent, "frozen_critic_ev", 0, 1, started)
        batch = agent._stage3_collect(deterministic=False, update_scaler=False)
        measurements, _ = _batch_metrics(agent, batch, warmup)
        if state_hashes(agent) != frozen_hashes:
            raise RuntimeError("Extra EV batch changed frozen actor/critic/scaler")
        artifact = {"format_version": VERSION, "point": point, "source_checkpoint": str(agent.cfg.base_policy_path),
                    "parent_commit": config["gm_parent_commit"], "prepare_seed": int(agent.seed),
                    "settings": config, "parameter_specs": _specs(agent),
                    **actor_state(agent), "critic": _copy(agent.model.critic.state_dict()),
                    "reward_scaler": scaler_state(agent), **frozen_hashes}
        with frozen_path.open("xb") as stream:
            torch.save(artifact, stream)
        checksum = file_sha256(frozen_path)
        with checksum_path.open("x") as stream:
            stream.write(checksum + "  " + frozen_path.name + "\n")
        meta.update(status="completed", frozen_path=str(frozen_path), frozen_sha256=checksum,
                    actor_after_sha256=frozen_hashes["actor_sha256"], **frozen_hashes,
                    frozen_ev_batch=measurements, elapsed_seconds=time.monotonic() - started,
                    critic_optimizer_steps=sum(r["critic_steps"] for r in meta["warmup"]))
        _atomic_json(directory / "grad_mean.json", meta)
        _progress(agent, "completed", warmup, warmup, started, status="complete")
        return meta
    except BaseException as error:
        meta.update(status="failed", error=type(error).__name__ + ": " + str(error),
                    elapsed_seconds=time.monotonic() - started)
        _atomic_json(directory / "grad_mean.json", meta)
        _progress(agent, "failed", meta["completed_warmup_batches"], warmup, started, status="failed", error=meta["error"])
        raise


def load_frozen(agent, config):
    path = Path(config["gm_frozen_path"]).expanduser().resolve() if config["gm_frozen_path"] else None
    expected = config["gm_frozen_sha256"]
    if path is None or not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Collect requires an existing frozen file and explicit SHA256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError("Frozen-file SHA256 does not match the authorized preparation")
    frozen = torch.load(path, map_location="cpu", weights_only=True)
    if (frozen.get("format_version") != VERSION or frozen.get("point") != config["gm_point"] or
            frozen.get("parameter_specs") != _specs(agent)):
        raise ValueError("Frozen state point/version/actor geometry mismatch")
    agent.model.actor.load_state_dict(frozen["actor"], strict=True)
    agent.model.actor_ft.load_state_dict(frozen["actor_ft"], strict=True)
    agent.model.critic.load_state_dict(frozen["critic"], strict=True)
    _restore_scaler(agent, frozen["reward_scaler"])
    hashes = state_hashes(agent)
    if any(hashes[key] != frozen.get(key) for key in hashes):
        raise ValueError("Loaded actor/critic/scaler state differs from frozen preparation")
    return path, actual, hashes


def collect(agent, config):
    directory, started = Path(agent.logdir), time.monotonic()
    include_critic = bool(config.get("gm_include_critic", False))
    n_variants = 6 if include_critic else 5
    paths = [directory / ("sums_V%d.npy" % i) for i in range(1, n_variants + 1)]
    if any(p.exists() for p in paths + [directory / "batches.csv", directory / "meta.json", directory / "decision_scores.csv"]):
        raise FileExistsError("Refusing to overwrite gradient collection evidence")
    frozen_path, frozen_hash, before = load_frozen(agent, config)
    total = int(config["gm_batches"])
    count = sum(p.numel() for p in agent.model.actor_ft.parameters())
    sums = np.zeros((n_variants, count), dtype=np.float64)
    cross_sum = np.zeros((n_variants, n_variants), dtype=np.float64)
    rows = []
    meta = _metadata(agent, config)
    meta.update(status="running", gm_batches=total, completed_batches=0,
                frozen_path=str(frozen_path), frozen_sha256=frozen_hash,
                **{key.replace("_sha256", "_before_sha256"): value for key, value in before.items()},
                critic_optimizer_steps=0, sums_paths=[str(p) for p in paths])
    _atomic_json(directory / "meta.json", meta)
    try:
        for iteration in range(total):
            tick = time.monotonic()
            _progress(agent, "collect", iteration, total, started)
            batch = agent._stage3_collect(deterministic=False, update_scaler=False)
            rollout_seconds = time.monotonic() - tick
            if iteration == 0:
                meta["first_observations"] = batch["obs_k"]["state"][:agent.n_envs].detach().cpu().tolist()
            gradient_started = time.monotonic()
            vectors = batch_gradients(agent, batch)
            gradient_seconds = time.monotonic() - gradient_started
            cross = vectors @ vectors.T
            components = None
            if include_critic:
                # Keep the old 5x5 BLAS operation and old gradient call order intact.
                # A single 6x6 multiplication can change old reductions bitwise.
                components = theory_critic_components(agent, batch)
                v6_started = time.monotonic()
                v6 = critic_gradient(agent, batch, components)
                gradient_seconds += time.monotonic() - v6_started
                expanded = np.empty((6, 6), dtype=np.float64)
                expanded[:5, :5] = cross
                for index in range(5):
                    expanded[index, 5] = expanded[5, index] = np.dot(vectors[index], v6)
                expanded[5, 5] = np.dot(v6, v6)
                cross = expanded
                vectors = np.concatenate((vectors, v6[None]), axis=0)
            if not np.isfinite(cross).all():
                raise FloatingPointError("Nonfinite whole-batch gradient cross-product")
            diagnostic_started = time.monotonic()
            row, returns = _batch_metrics(agent, batch, iteration)
            if include_critic:
                row.update(critic_baseline_metrics(agent, batch, components))
            if iteration == 0:
                meta["first_episode_returns"] = returns.tolist()
                if config["gm_score_sample"]:
                    score_started = time.monotonic()
                    scores = decision_scores(agent, batch, config["gm_score_sample"], iteration)
                    _atomic_csv(directory / "decision_scores.csv", scores)
                    meta["score_sample"] = {"count": len(scores), "path": str(directory / "decision_scores.csv"),
                                            "seconds": time.monotonic() - score_started,
                                            "definition": "unweighted sum over ten transition logprobs, all24 coordinates; gradient norm in float64"}
            after = state_hashes(agent)
            if before != after:
                raise RuntimeError("Collect changed frozen actor/critic/scaler state")
            sums += vectors
            cross_sum += cross
            if not np.isfinite(sums).all() or not np.isfinite(cross_sum).all():
                raise FloatingPointError("Nonfinite float64 accumulated group statistics")
            row.update(rollout_seconds=rollout_seconds, gradient_seconds=gradient_seconds,
                       diagnostic_seconds=time.monotonic() - diagnostic_started, seconds=time.monotonic() - tick)
            for i in range(n_variants):
                row["V%d_norm_sq" % (i + 1)] = float(cross[i, i])
                for j in range(n_variants):
                    row["V%d_V%d_dot" % (i + 1, j + 1)] = float(cross[i, j])
            rows.append(row)
            meta.update(completed_batches=len(rows), gradient_cross_sum=cross_sum.tolist(),
                        elapsed_seconds=time.monotonic() - started,
                        **{key.replace("_sha256", "_after_sha256"): value for key, value in after.items()})
            _atomic_csv(directory / "batches.csv", rows)
            _atomic_json(directory / "meta.json", meta)
            log.info("GRAD_MEAN_BATCH %s", json.dumps(_strict_json(row), sort_keys=True))
            _progress(agent, "collect", iteration + 1, total, started)
        meta["status"] = "completed"
        return meta
    except BaseException as error:
        meta.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        # Persist finite failed prefixes, never label them completed groups.
        for path, value in zip(paths, sums):
            stored = value.astype(np.float32)
            if not np.isfinite(stored).all():
                meta.update(status="failed", error="Nonfinite float32 stored group sum")
                _atomic_json(directory / "meta.json", meta)
                raise FloatingPointError(meta["error"])
            _save_vector(path, stored)
        terminal_error = None
        try:
            final_hashes = state_hashes(agent)
        except FloatingPointError as error:
            terminal_error = error
            final_hashes = {key: None for key in before}
            meta.update(status="failed", final_state_hash_error=str(error))
        meta.update(elapsed_seconds=time.monotonic() - started, completed_batches=len(rows),
                    gradient_cross_sum=cross_sum.tolist(), gradient_squared_norm_sum=np.diag(cross_sum).tolist(),
                    **{key.replace("_sha256", "_after_sha256"): value for key, value in final_hashes.items()})
        meta["return_totals"] = {key: sum(row[key] for row in rows) for key in
                                 ("n_episodes", "return_sum", "return_sumsq", "J_disc_sum", "J_disc_sumsq")}
        _atomic_json(directory / "meta.json", meta)
        _progress(agent, "completed" if meta["status"] == "completed" else "failed", len(rows), total, started,
                  status="complete" if meta["status"] == "completed" else "failed")
        if terminal_error is not None:
            raise terminal_error


def _direct_children():
    """Linux process identities, restricted to children of this job process."""
    parent = os.getpid()
    pids = set()
    for task in Path("/proc/self/task").iterdir():
        try:
            pids.update(int(value) for value in (task / "children").read_text().split())
        except FileNotFoundError:
            pass  # A completed close thread may disappear during enumeration.
    result = {}
    for pid in pids:
        try:
            fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[1]) == parent:
            result[pid] = {"pid": pid, "parent_pid": parent,
                           "start_ticks": int(fields[19]), "state": fields[0]}
    return result


def cleanup_environment(agent, primary_error=None, close_timeout=10.0, join_timeout=5.0):
    """Close and reap this job's children; cleanup failures make the job fail.

    AsyncVectorEnv.close() can block in recv()/join() even with its timeout
    argument, so its graceful call is bounded externally. Only children whose
    parent is this process are eligible for fallback termination; no process
    group or scheduler job is signalled.
    """
    started, parent = time.monotonic(), os.getpid()
    path = Path(agent.logdir) / "cleanup.json"
    errors, actions, workers = [], [], {}
    record = {"status": "running", "parent_pid": parent,
              "started_unix": time.time(), "close_timeout_seconds": close_timeout,
              "join_timeout_seconds": join_timeout,
              "primary_error": None if primary_error is None else {
                  "type": type(primary_error).__name__, "message": str(primary_error),
                  "traceback": "".join(traceback.format_exception(
                      type(primary_error), primary_error, primary_error.__traceback__))},
              "errors": errors, "actions": actions}
    def fail(phase, error):
        errors.append({"phase": phase, "type": type(error).__name__, "message": str(error)})
    def discover():
        for process in list(getattr(agent.venv, "processes", [])) + list(mp.active_children()):
            if process.pid is None:
                continue
            if getattr(process, "_parent_pid", None) != parent:
                fail("ownership", RuntimeError("Refusing to signal non-owned process %s" % process.pid))
                continue
            workers[process.pid] = process
    def join_workers(timeout):
        deadline = time.monotonic() + timeout
        for pid, process in workers.items():
            try:
                process.join(max(0.0, deadline - time.monotonic()))
            except BaseException as error:
                fail("join_pid_%d" % pid, error)
    def inspect_children():
        try:
            return _direct_children()
        except BaseException as error:
            fail("inspect_direct_children", error)
            return {}
    def join_other_children(timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = [pid for pid in inspect_children() if pid not in workers]
            if not remaining:
                return
            for pid in remaining:
                try:
                    reaped, status = os.waitpid(pid, os.WNOHANG)
                    if reaped:
                        actions.append({"pid": pid, "action": "reaped direct child",
                                        "wait_status": status})
                except ChildProcessError:
                    pass
            if time.monotonic() >= deadline:
                return
            time.sleep(.02)
    def stop_owned(signum):
        # multiprocessing objects are joined only through their own API.
        for pid, process in workers.items():
            try:
                if process.is_alive():
                    process.terminate() if signum == signal.SIGTERM else process.kill()
                    actions.append({"pid": pid, "signal": int(signum), "owner": "multiprocessing child"})
            except BaseException as error:
                fail("signal_pid_%d" % pid, error)
        # Inspect/recheck parent+start ticks before signalling any other child.
        for pid, identity in inspect_children().items():
            if pid in workers:
                continue
            try:
                current = _direct_children().get(pid)
                if current is not None and current["start_ticks"] == identity["start_ticks"]:
                    if current["state"] != "Z":
                        os.kill(pid, signum)
                        actions.append({"pid": pid, "signal": int(signum), "owner": "verified direct child"})
                    os.waitpid(pid, os.WNOHANG)
            except (ProcessLookupError, ChildProcessError):
                pass
            except BaseException as error:
                fail("signal_direct_child_%d" % pid, error)
    close_errors = []
    def close_vector():
        try:
            agent.venv.close()
        except BaseException as error:
            close_errors.append({"phase": "venv.close", "type": type(error).__name__,
                                 "message": str(error), "traceback": traceback.format_exc()})
    close_thread = None
    try:
        discover()
        record["worker_pids_before"] = sorted(workers)
        record["direct_children_before"] = list(inspect_children().values())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, record)
        except BaseException as error:
            fail("write_initial_cleanup_record", error)
        close_thread = threading.Thread(target=close_vector, name="grad-mean-close", daemon=True)
        close_thread.start()
        close_thread.join(close_timeout)
        if close_thread.is_alive():
            fail("venv.close", TimeoutError("Graceful vector-environment close exceeded its deadline"))
        # Avoid simultaneous joins with a still-running close thread; terminate
        # first on that failure path so its blocking reads/joins can unwind.
        discover()
        if close_thread.is_alive():
            stop_owned(signal.SIGTERM)
            close_thread.join(join_timeout)
        if not close_thread.is_alive():
            join_workers(join_timeout)
            join_other_children(join_timeout)
        alive = [pid for pid, process in workers.items() if process.is_alive()]
        remaining = inspect_children()
        if alive or remaining:
            fail("graceful_children_exit", RuntimeError("Children remained after vector-environment close"))
            stop_owned(signal.SIGTERM)
            close_thread.join(join_timeout)
            join_workers(join_timeout)
            join_other_children(join_timeout)
        if any(process.is_alive() for process in workers.values()) or inspect_children():
            stop_owned(signal.SIGKILL)
            close_thread.join(join_timeout)
            join_workers(join_timeout)
            join_other_children(join_timeout)
        # Reap any non-multiprocessing direct-child zombies after signals.
        for pid in list(inspect_children()):
            if pid not in workers:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
    except BaseException as error:
        fail("cleanup_controller", error)
        stop_owned(signal.SIGTERM)
        join_workers(join_timeout)
        join_other_children(join_timeout)
        stop_owned(signal.SIGKILL)
        join_workers(join_timeout)
        join_other_children(join_timeout)
    finally:
        errors.extend(close_errors)
        if close_thread is not None and close_thread.is_alive():
            fail("close_thread", RuntimeError("Vector-environment close thread did not finish"))
        record["workers"] = []
        for pid, process in workers.items():
            try:
                value = {"pid": pid, "exitcode": process.exitcode, "alive": process.is_alive()}
                record["workers"].append(value)
                if value["alive"] or value["exitcode"] != 0:
                    fail("worker_exit", RuntimeError("Worker did not exit cleanly: %r" % value))
            except BaseException as error:
                fail("inspect_worker_%d" % pid, error)
        record["remaining_active_children"] = [process.pid for process in mp.active_children()]
        record["remaining_direct_children"] = list(inspect_children().values())
        record["all_children_exited"] = not (record["remaining_active_children"] or record["remaining_direct_children"])
        if not record["all_children_exited"]:
            fail("children_remaining", RuntimeError("Job still has live or unreaped children"))
        record.update(status="failed" if errors else "passed",
                      elapsed_seconds=time.monotonic() - started, ended_unix=time.time())
        _atomic_json(path, record)
        log.info("GRADMEAN_CLEANUP %s", json.dumps(_strict_json(record), sort_keys=True))
    if errors:
        raise RuntimeError("grad_mean cleanup failed; see %s: %s" % (path, errors))
    return record


def run_grad_mean(agent):
    primary_error = None
    try:
        config = settings(agent)
        _validate(agent, config)
        Path(agent.logdir).mkdir(parents=True, exist_ok=True)
        seed_environments(agent)
        return prepare(agent, config) if config["gm_step"] == "prepare" else collect(agent, config)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            cleanup_environment(agent, primary_error=primary_error)
        except BaseException as cleanup_error:
            if primary_error is not None:
                raise cleanup_error from primary_error
            raise
