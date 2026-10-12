"""Explicit W=0 frozen-critic measurements; legacy grad_mean stays unchanged."""
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from agent.finetune.stage3_math import actor_gradient, episode_indices, mc_returns
from agent.finetune.stage3_modes import _generator, _variant_coefficients


def validate_settings(config):
    if isinstance(config.get("gm_warmup"), bool) or config.get("gm_warmup") != 0:
        raise ValueError("The v5 protocol permits only explicit gm_warmup=0")
    if "gm_scaler_c" not in config:
        raise ValueError("W=0 needs explicit gm_scaler_c: a positive number or authorized null fallback")
    c = config["gm_scaler_c"]
    if c is not None and (isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0):
        raise ValueError("gm_scaler_c must be a finite positive number or explicit null")
    if not isinstance(config.get("gm_v2_raw", False), bool):
        raise ValueError("gm_v2_raw must be boolean")
    if config.get("gm_normalization", "full") not in ("full", "quarter"):
        raise ValueError("gm_normalization must be full or quarter")
    if config["gm_step"] == "norm_check" and not config.get("gm_v2_raw", False):
        raise ValueError("The v5 normalization check requires raw MC-minus-critic V2")


def set_scaler_divisor(agent, divisor):
    """Set the frozen scalar transform without fabricating running moments."""
    scaler = agent.running_reward_scaler
    if scaler.per_env or np.asarray(scaler.ret_rms.var).size != 1:
        raise ValueError("V5 requires the scalar running reward scaler")
    variance = float(divisor) ** 2 - float(scaler.epsilon)
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Requested divisor is not representable by the native scaler")
    # All other fields remain unchanged; they are frozen and never updated.
    scaler.ret_rms.var = np.asarray(variance, dtype=np.float64)
    effective = float(np.sqrt(scaler.ret_rms.var + scaler.epsilon))
    if effective != float(divisor):
        raise ValueError("Native sqrt(var+epsilon) cannot reproduce the supplied divisor exactly")
    return effective


def prepare_without_warmup(agent, config):
    from agent.finetune import grad_mean as gm
    validate_settings(config)
    directory, started = Path(agent.logdir), time.monotonic()
    point = config["gm_point"]
    frozen_path = directory / ("frozen_" + point + ".pt")
    checksum_path = directory / ("frozen_" + point + ".sha256")
    if any(path.exists() for path in (frozen_path, checksum_path, directory / "grad_mean.json")):
        raise FileExistsError("Refusing to overwrite W=0 preparation")
    before = gm.state_hashes(agent)
    meta = gm._metadata(agent, config)
    meta.update(status="running", completed_warmup_batches=0, warmup=[], critic_optimizer_steps=0,
                actor_before_sha256=before["actor_sha256"], critic_before_sha256=before["critic_sha256"],
                scaler_before_sha256=before["scaler_sha256"], scaler_fit_batches=[])
    gm._atomic_json(directory / "grad_mean.json", meta)
    try:
        divisor = config["gm_scaler_c"]
        fallback = divisor is None
        if fallback:
            numerator, denominator = 0.0, 0.0
            for iteration in range(5):
                gm._progress(agent, "frozen_critic_scaler_fit", iteration, 5, started)
                batch = agent._stage3_collect(deterministic=False, update_scaler=False)
                episode_indices(batch["firsts"])
                returns = mc_returns(batch["raw_rewards"], batch["firsts"], agent.gamma)
                values = np.asarray(batch["values"], dtype=np.float64)
                vg, vv = float(np.sum(values * returns, dtype=np.float64)), float(np.sum(values ** 2, dtype=np.float64))
                numerator += vg
                denominator += vv
                meta["scaler_fit_batches"].append(dict(batch=iteration, sum_VG=vg, sum_V2=vv))
                if gm.state_hashes(agent) != before:
                    raise RuntimeError("Scale-fit collection mutated the frozen actor, critic or scaler")
            if denominator <= 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
                raise FloatingPointError("Invalid through-origin frozen-critic scale fit")
            divisor = numerator / denominator
            if not math.isfinite(divisor) or divisor <= 0:
                raise FloatingPointError("Through-origin scale fit is nonpositive/nonfinite; no repair is permitted")
        effective = set_scaler_divisor(agent, divisor)
        frozen_hashes = gm.state_hashes(agent)
        if any(frozen_hashes[key] != before[key] for key in ("actor_sha256", "critic_sha256")):
            raise RuntimeError("W=0 preparation changed checkpoint model weights")
        source = "least_squares_five_batches" if fallback else "provided"
        extra = dict(effective_scaler_c=effective, scaler_c_source=source,
                     scaler_c_fallback_batches=5 if fallback else 0,
                     scaler_c_fit_definition="sum(V*raw_MC_G)/sum(V^2), through origin" if fallback else None,
                     critic_before_sha256=before["critic_sha256"], critic_after_sha256=frozen_hashes["critic_sha256"])
        gm._progress(agent, "frozen_critic_ev", 0, 1, started)
        batch = agent._stage3_collect(deterministic=False, update_scaler=False)
        measurements, _ = gm._batch_metrics(agent, batch, 5 if fallback else 0)
        if gm.state_hashes(agent) != frozen_hashes:
            raise RuntimeError("W=0 EV batch mutated frozen state")
        artifact = {"format_version": gm.VERSION, "point": point,
                    "source_checkpoint": str(agent.cfg.base_policy_path), "parent_commit": config["gm_parent_commit"],
                    "prepare_seed": int(agent.seed), "settings": config, "parameter_specs": gm._specs(agent),
                    **gm.actor_state(agent), "critic": gm._copy(agent.model.critic.state_dict()),
                    "reward_scaler": gm.scaler_state(agent), **frozen_hashes, **extra}
        with frozen_path.open("xb") as stream:
            torch.save(artifact, stream)
        checksum = gm.file_sha256(frozen_path)
        with checksum_path.open("x") as stream:
            stream.write(checksum + "  " + frozen_path.name + "\n")
        meta.update(status="completed", frozen_path=str(frozen_path), frozen_sha256=checksum,
                    actor_after_sha256=frozen_hashes["actor_sha256"], scaler_after_sha256=frozen_hashes["scaler_sha256"],
                    **frozen_hashes, **extra, frozen_ev_batch=measurements, elapsed_seconds=time.monotonic() - started)
        gm._atomic_json(directory / "grad_mean.json", meta)
        gm._progress(agent, "completed", 0, 0, started, status="complete")
        return meta
    except BaseException as error:
        meta.update(status="failed", error=type(error).__name__ + ": " + str(error),
                    elapsed_seconds=time.monotonic() - started)
        gm._atomic_json(directory / "grad_mean.json", meta)
        raise


def pair_permutation(agent, iteration):
    generator = _generator(agent.seed, iteration, 95107)
    total = agent.n_steps * agent.n_envs * agent.model.ft_denoising_steps
    # Training reuses the fifth/final critic-epoch permutation for its actor step.
    for _ in range(5):
        permutation = torch.randperm(total, generator=generator, device="cpu")
    return permutation.to(agent.device)


def normalized_gradient(agent, batch, advantages, permutation, normalization):
    """Mean-coordinate score gradient, exact native per-chunk norm ordering."""
    if normalization not in ("full", "quarter"):
        raise ValueError("Unknown normalization")
    model = agent.model
    parameters = list(model.actor_ft.parameters())
    denoising = model.ft_denoising_steps
    total = len(permutation)
    adv = torch.as_tensor(advantages, device=agent.device, dtype=torch.float32).reshape(-1).detach()
    if normalization == "full":
        repeated = adv.repeat_interleave(denoising)
        mean, std = repeated.mean(), repeated.std()
    gradients = [torch.zeros_like(parameter) for parameter in parameters]
    for start in range(0, total, agent.batch_size):
        selected = permutation[start:start + agent.batch_size]
        decisions = torch.div(selected, denoising, rounding_mode="floor")
        steps = selected % denoising
        coefficients = adv[decisions].clone()
        if normalization == "quarter":
            mean, std = coefficients.mean(), coefficients.std()
        coefficients = (coefficients - mean) / (std + 1e-8)
        discount = torch.tensor([.99 ** (denoising - int(i) - 1) for i in steps.detach().cpu()], device=agent.device)
        coefficients *= discount
        logp = model.get_logprobs_subsample({key: value[decisions] for key, value in batch["obs_k"].items()},
            batch["chains_k"][decisions, steps], batch["chains_k"][decisions, steps + 1], steps)
        logp = logp[:, :agent.reward_horizon, :].mean(dim=(-1, -2))
        loss = -(coefficients * logp).mean() * (len(selected) / total)
        values = torch.autograd.grad(loss, parameters, allow_unused=True)
        for target, value in zip(gradients, values):
            if value is not None:
                target.add_(value.detach())
    return torch.cat([value.reshape(-1) for value in gradients])


def batch_gradients_v5(agent, batch, env_indices=None):
    from agent.finetune import grad_mean as gm
    if env_indices is not None:
        raise ValueError("V5 production collection always measures the full batch")
    variants = list(_variant_coefficients(agent, batch))
    if agent.cfg.train.get("gm_v2_raw", False):
        raw = gm.theory_critic_components(agent, batch)["critic_residual"]
        variants[1] = (raw, "mean", .99, True)
    normalization = agent.cfg.train.get("gm_normalization", "full")
    permutation = pair_permutation(agent, getattr(agent, "_gm_v5_iteration", 0)) if normalization == "quarter" else None
    vectors = []
    for variant_index, (advantages, reduction, discount, normalize) in enumerate(variants):
        if variant_index == 1 and normalize and normalization == "quarter":
            gradient = normalized_gradient(agent, batch, advantages, permutation, "quarter")
        else:
            gradient = actor_gradient(agent, batch, advantages, reduce=reduction,
                                      gamma_denoising=discount, normalize=normalize, clamp=False)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite V5 measurement gradient")
        vectors.append(-gradient.detach().double().cpu().numpy())
    return np.stack(vectors)


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def diagnostic_state(agent):
    """All state that an extra gradient pass is forbidden to mutate."""
    return dict(rng=_rng_state(), model=agent.model.state_dict(),
                actor_optimizer=agent.actor_optimizer.state_dict(),
                critic_optimizer=agent.critic_optimizer.state_dict(),
                gradients=[parameter.grad for parameter in agent.model.parameters()])


def normalization_check(agent, config):
    from agent.finetune import grad_mean as gm
    directory, started = Path(agent.logdir), time.monotonic()
    target = directory / "normalization_check.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite normalization evidence")
    _, frozen_hash, before = gm.load_frozen(agent, config)
    report = gm._metadata(agent, config)
    report.update(status="running", batch_count=0, rows=[], frozen_sha256=frozen_hash,
                  **{key.replace("_sha256", "_before_sha256"): value for key, value in before.items()},
                  actor_optimizer_steps=0, critic_optimizer_steps=0,
                  permutation="dedicated CPU SeedSequence(seed,batch,95107), fifth of five full pair permutations")
    gm._atomic_json(target, report)
    for iteration in range(2):
        gm._progress(agent, "normalization_check", iteration, 2, started)
        batch = agent._stage3_collect(deterministic=False, update_scaler=False)
        advantages = gm.theory_critic_components(agent, batch)["critic_residual"]
        diagnostic_before = gm.state_sha256(diagnostic_state(agent))
        permutation = pair_permutation(agent, iteration)
        full = actor_gradient(agent, batch, advantages, reduce="mean", gamma_denoising=.99,
                              normalize=True, clamp=False).double()
        quarter = normalized_gradient(agent, batch, advantages, permutation, "quarter").double()
        if not bool(torch.isfinite(full).all() and torch.isfinite(quarter).all()):
            raise FloatingPointError("Nonfinite normalization diagnostic")
        norm_full, norm_quarter = full.norm().item(), quarter.norm().item()
        if norm_full <= 0 or norm_quarter <= 0:
            raise FloatingPointError("Zero normalization diagnostic gradient")
        cosine = (torch.dot(full, quarter) / (norm_full * norm_quarter)).item()
        ratio = norm_quarter / norm_full
        unchanged = (gm.state_hashes(agent) == before and
                     gm.state_sha256(diagnostic_state(agent)) == diagnostic_before)
        if not unchanged:
            raise RuntimeError("Normalization diagnostic mutated frozen state or global RNG")
        report["rows"].append(dict(batch=iteration, cosine=cosine, quarter_over_full_norm=ratio,
                                   full_norm=norm_full, quarter_norm=norm_quarter,
                                   full_acceptable=cosine >= .999 and abs(ratio - 1.) <= .01,
                                   state_and_rng_unchanged=True))
        report["batch_count"] = iteration + 1
        gm._atomic_json(target, report)
    after = gm.state_hashes(agent)
    report.update(status="completed", selected_normalization="full" if all(row["full_acceptable"] for row in report["rows"]) else "quarter",
                  elapsed_seconds=time.monotonic() - started,
                  **{key.replace("_sha256", "_after_sha256"): value for key, value in after.items()})
    gm._atomic_json(target, report)
    gm._progress(agent, "completed", 2, 2, started, status="complete")
    return report
