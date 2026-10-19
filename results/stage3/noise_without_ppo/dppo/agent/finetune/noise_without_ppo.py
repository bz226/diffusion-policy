"""Paired frozen-policy loss gradients for the approved without_PPO study.

This module has a separate CLI and is never imported by normal training.  The
measurement uses native four-chunk advantage normalization and the fifth PPO
permutation, and never calls either optimizer's step method.
"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from agent.finetune import grad_mean as gm
from agent.finetune.grad_mean_v5 import set_scaler_divisor
from agent.finetune.stage3_math import as_episodes, discounted_episode_returns, episode_indices, mc_returns
from util.reward_scaling import RunningRewardScaler

LOG = logging.getLogger(__name__)
UPDATE_MAP = {0: 0, 35: 32, 70: 63, 105: 95, 139: 126}


def frozen_divisor(scaler):
    variance = np.asarray(scaler.ret_rms.var, dtype=np.float64)
    if scaler.per_env or variance.size != 1:
        raise ValueError("Measurement requires the original scalar reward scaler")
    value = float(np.sqrt(variance.reshape(-1)[0] + scaler.epsilon))
    if not math.isfinite(value) or value <= 0:
        raise FloatingPointError("Nonpositive/nonfinite scaler divisor")
    return value


def mc_critic_advantages(batch, gamma, prebatch_divisor):
    """Raw MC return minus the pre-batch critic baseline, in float64."""
    rewards = np.asarray(batch["raw_rewards"], dtype=np.float64)
    firsts = np.asarray(batch["firsts"], dtype=np.float64)
    episode_indices(firsts)
    values = batch["values"] if "values" in batch else batch["values_k"]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(rewards.shape)
    return mc_returns(rewards, firsts, gamma) - float(prebatch_divisor) * values


def gae_advantages(agent, scaled_rewards, values, terminated, final_observation=None,
                   final_values=None):
    """The native DPPO float64 GAE recursion, without changing critic targets."""
    rewards = np.asarray(scaled_rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(rewards.shape)
    terminated = np.asarray(terminated, dtype=np.float64)
    if final_values is None:
        if final_observation is None:
            if not bool(np.all(terminated[-1] == 1)):
                raise ValueError("A nonterminal final transition needs bootstrap values")
            final_values = np.zeros((1, rewards.shape[1]), dtype=np.float32)
        else:
            with torch.no_grad():
                final_values = agent.model.critic({"state": torch.from_numpy(
                    final_observation["state"]).float().to(agent.device)}).reshape(1, -1).cpu().numpy()
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextvalues = final_values if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - terminated[t]
        delta = (rewards[t] * agent.reward_scale_const + agent.gamma * nextvalues * nonterminal - values[t])
        advantages[t] = lastgaelam = delta + agent.gamma * agent.gae_lambda * nonterminal * lastgaelam
    return advantages


def pair_permutation(agent, seed):
    """Fifth random permutation, on the native training device."""
    generator = torch.Generator(device=agent.device)
    generator.manual_seed(int(seed))
    total = agent.n_steps * agent.n_envs * agent.model.ft_denoising_steps
    for _ in range(agent.update_epochs):
        indices = torch.randperm(total, generator=generator, device=agent.device)
    return indices


def _tensor(batch, tensor_key, numpy_key, agent):
    source = batch[tensor_key] if tensor_key in batch else batch[numpy_key]
    return torch.as_tensor(source, device=agent.device, dtype=torch.float32).reshape(-1)


def native_gradient(agent, batch, advantages, permutation, actor_loss="ratio"):
    """Read native actor .grad before its optimizer step, restoring owned state.

    All four 50,000-pair losses are normalized independently by model.loss and
    weighted by chunk size / 200,000 exactly like _single_actor_step.  Returned
    vectors are loss gradients (not ascent vectors), in float32 training units.
    """
    if actor_loss not in ("ratio", "score"):
        raise ValueError("actor_loss must be ratio or score")
    total = agent.n_steps * agent.n_envs * agent.model.ft_denoising_steps
    if len(permutation) != total:
        raise ValueError("A full native pair permutation is required")
    model = agent.model
    original_loss = getattr(model, "actor_loss", None)
    if original_loss is None and actor_loss != "ratio":
        raise ValueError("The model must expose the existing score-loss flag")
    original_gradients = [parameter.grad for parameter in model.parameters()]
    returns = _tensor(batch, "returns_k", "returns", agent)
    values = _tensor(batch, "values_k", "values", agent)
    advantages = torch.as_tensor(advantages, device=agent.device, dtype=torch.float32).reshape(-1)
    oldlogprobs = batch["logprobs_k"] if "logprobs_k" in batch else batch["oldlogprobs"]
    oldlogprobs = torch.as_tensor(oldlogprobs, device=agent.device, dtype=torch.float32)
    metrics = {"ratio": 0.0, "clipfrac": 0.0, "approx_kl": 0.0, "pg_loss": 0.0,
               "chunk_sizes": [], "actor_loss": actor_loss}
    try:
        if original_loss is not None:
            model.actor_loss = actor_loss
        agent.actor_optimizer.zero_grad()
        for start in range(0, total, agent.batch_size):
            selected = permutation[start:start + agent.batch_size]
            decisions, steps = torch.unravel_index(selected,
                (agent.n_steps * agent.n_envs, model.ft_denoising_steps))
            losses = model.loss(
                {"state": batch["obs_k"]["state"][decisions]},
                batch["chains_k"][decisions, steps], batch["chains_k"][decisions, steps + 1],
                steps, returns[decisions], values[decisions], advantages[decisions],
                oldlogprobs[decisions, steps], use_bc_loss=agent.use_bc_loss,
                reward_horizon=agent.reward_horizon)
            pg_loss, entropy_loss, _, clipfrac, approx_kl, ratio, bc_loss, _ = losses
            loss = pg_loss + entropy_loss * agent.ent_coef + bc_loss * agent.bc_loss_coeff
            weight = len(selected) / total
            (loss * weight).backward()
            for key, value in (("ratio", ratio), ("clipfrac", clipfrac),
                               ("approx_kl", approx_kl), ("pg_loss", pg_loss)):
                metrics[key] += weight * float(value.detach().item() if torch.is_tensor(value) else value)
            metrics["chunk_sizes"].append(len(selected))
        vector = torch.cat([parameter.grad.detach().reshape(-1).clone() if parameter.grad is not None
                            else torch.zeros_like(parameter).reshape(-1)
                            for parameter in model.actor_ft.parameters()])
        if not bool(torch.isfinite(vector).all()):
            raise FloatingPointError("Nonfinite measured loss gradient")
        return vector, metrics
    finally:
        if original_loss is not None:
            model.actor_loss = original_loss
        for parameter, gradient in zip(model.parameters(), original_gradients):
            parameter.grad = gradient


def seed_record(spec, batch_index):
    """Disjoint point/job/batch blocks; 40 private env streams and two Torch streams."""
    p, j, b = int(spec["point_index"]), int(spec["job_index"]), int(batch_index)
    if not (0 <= p < 5 and 0 <= j < 100 and 0 <= b < 100):
        raise ValueError("Seed namespace requires point<5, job<100, batch<100")
    base = int(spec.get("seed_base", 50000000)) + p * 1000000 + j * 10000 + b * 100
    return {"environment_seeds": list(range(base, base + 40)),
            "diffusion_seed": base + 50, "permutation_seed": base + 51,
            "python_seed": base + 52, "numpy_seed": base + 53}


def seeded_batch(agent, spec, index, update_scaler):
    record = seed_record(spec, index)
    expected = record["environment_seeds"]
    responses = agent.venv.call_sync_arg("grad_mean_seed", "seed", expected,
                                          indices=list(range(agent.n_envs)))
    if len(responses) != agent.n_envs or any(not ok or value != {
            "seed": seed, "underlying_env_seeded": True, "action_space_seeded": True}
            for (value, ok), seed in zip(responses, expected)):
        raise RuntimeError("Private MuJoCo reset generator was not seeded")
    # Native rollouts end exactly at episode boundaries.  We discard the
    # already-autoreset observation and explicitly reset from a fresh private
    # seed, preserving its initial-state distribution while separating batches.
    for name in ("_stage3_mode_obs", "_stage3_mode_done"):
        if hasattr(agent, name):
            delattr(agent, name)
    random.seed(record["python_seed"])
    np.random.seed(record["numpy_seed"])
    torch.manual_seed(record["diffusion_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(record["diffusion_seed"])
    batch = agent._stage3_collect(deterministic=False, update_scaler=update_scaler)
    episode_indices(batch["firsts"])
    if not np.array_equal(np.flatnonzero(batch["firsts"][:, 0]), np.array([0, 250, 500])):
        raise AssertionError("Unexpected episode boundaries")
    recomputed = gae_advantages(agent, batch["scaled_rewards"], batch["values"], batch["terminated"],
                               final_observation=agent._stage3_mode_obs)
    if not np.array_equal(recomputed, batch["gae_adv"]):
        raise AssertionError("Measurement GAE differs from the native rollout's recursion")
    return batch, record


def checkpoint_critic_hash(agent, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {key[len("critic."):]: value for key, value in payload["model"].items()
                if key.startswith("critic.")}
    if not expected:
        raise ValueError("Checkpoint has no trained/untrained critic state")
    expected_hash = gm.state_sha256(expected)
    actual_hash = gm.state_sha256(agent.model.critic.state_dict())
    if expected_hash != actual_hash:
        raise AssertionError("Restored critic differs bitwise from the checkpoint")
    return actual_hash


def _model_copy(agent):
    return {key: value.detach().cpu().clone() for key, value in agent.model.state_dict().items()}


def _assert_model_equal(agent, before):
    if any(not torch.equal(value.detach().cpu(), before[key])
           for key, value in agent.model.state_dict().items()):
        raise AssertionError("Measurement changed an actor or critic parameter/buffer")


def prepare(agent, spec):
    directory = Path(agent.logdir)
    path = directory / "prepare.json"
    before = _model_copy(agent)
    critic_hash = checkpoint_critic_hash(agent, spec["checkpoint"])
    meta = {"status": "running", "operation": "prepare", "spec": spec,
            "critic_checkpoint_sha256": critic_hash, "warmup_batches": 0,
            "actor_optimizer_steps": 0, "critic_optimizer_steps": 0, "fit_batches": []}
    gm._atomic_json(path, meta)
    if int(spec["checkpoint_itr"]) == 0:
        c = frozen_divisor(agent.running_reward_scaler)
        source = "exact_initial_scaler_state_updated_per_measurement_batch"
    else:
        numerator, denominator = 0.0, 0.0
        for index in range(5):
            batch, seeds = seeded_batch(agent, spec, index, update_scaler=False)
            returns = mc_returns(batch["raw_rewards"], batch["firsts"], agent.gamma)
            values = np.asarray(batch["values"], dtype=np.float64)
            vg, vv = float(np.sum(values * returns)), float(np.sum(values ** 2))
            numerator += vg
            denominator += vv
            meta["fit_batches"].append({"batch": index, "sum_VG": vg, "sum_V2": vv, "seeds": seeds})
            gm._atomic_json(path, meta)
            _assert_model_equal(agent, before)
        if denominator <= 0 or not math.isfinite(numerator + denominator):
            raise FloatingPointError("Invalid five-batch scale fit")
        c = numerator / denominator
        if not math.isfinite(c) or c <= 0:
            raise FloatingPointError("Nonpositive/nonfinite scale fit; no repair permitted")
        set_scaler_divisor(agent, c)
        source = "least_squares_five_fresh_batches_frozen_critic_through_origin"
    _assert_model_equal(agent, before)
    meta.update(status="measurement_complete_pending_cleanup", scaler_c=c,
                scaler_c_source=source, scaler_approximation=int(spec["checkpoint_itr"]) != 0,
                critic_after_sha256=checkpoint_critic_hash(agent, spec["checkpoint"]),
                actor_and_critic_unchanged=True)
    gm._atomic_json(path, meta)
    return meta, path


def collect(agent, spec):
    frozen = json.loads(Path(spec["frozen_path"]).read_text())
    if frozen["status"] != "completed" or frozen["spec"]["checkpoint"] != spec["checkpoint"]:
        raise ValueError("Collection requires the completed matching preparation")
    before = _model_copy(agent)
    critic_hash = checkpoint_critic_hash(agent, spec["checkpoint"])
    if critic_hash != frozen["critic_checkpoint_sha256"]:
        raise AssertionError("Prepared and collected critics differ")
    initial = int(spec["checkpoint_itr"]) == 0
    total = int(spec.get("batches", 50))
    if total != 50 and not (spec.get("gate_only", False) and total in (1, 2)):
        raise ValueError("Production jobs require exactly 50 batches")
    if not initial:
        set_scaler_divisor(agent, frozen["scaler_c"])
    rows, records = [], []
    sums = np.zeros((2, sum(parameter.numel() for parameter in agent.model.actor_ft.parameters())), dtype=np.float64)
    directory = Path(agent.logdir)
    path = directory / "meta.json"
    meta = {"status": "running", "operation": "collect", "point": spec["point"],
            "point_index": int(spec["point_index"]), "checkpoint_itr": int(spec["checkpoint_itr"]),
            "updates": int(spec["updates"]), "K": 0, "target_batches": total,
            "source_checkpoint": spec["checkpoint"], "parent_revision": spec["parent_revision"],
            "spec": spec, "scaler_c": frozen["scaler_c"], "scaler_c_source": frozen["scaler_c_source"],
            "scaler_approximation": frozen["scaler_approximation"], "seed_records": records,
            "critic_checkpoint_sha256": critic_hash, "critic_warmup_batches": 0,
            "actor_optimizer_steps": 0, "critic_optimizer_steps": 0,
            "gradient_sign": "native actor loss gradient passed to optimizer",
            "estimators": ["without_PPO", "without_PPO_and_GAE"],
            "estimator_ids": ["V1", "V2"], "normalization": "separate native 50000-pair chunks",
            "permutation": "fifth of five native-device random permutations, same permutation for both estimators",
            "dtype": "float32 model/gradient, float64 group sums and scalar products",
            "original_reset_at_iteration": bool(agent.reset_at_iteration),
            "measurement_reset": "fresh private environment seed and reset at each aligned batch boundary",
            "episodes_per_batch": 80, "pairs_per_batch": 200000}
    gm._atomic_json(path, meta)
    for index in range(total):
        if initial:
            agent.running_reward_scaler = RunningRewardScaler(agent.n_envs)
        pre_c = frozen_divisor(agent.running_reward_scaler)
        batch, seeds = seeded_batch(agent, spec, index, update_scaler=initial)
        post_c = frozen_divisor(agent.running_reward_scaler)
        tick = time.perf_counter()
        permutation = pair_permutation(agent, seeds["permutation_seed"])
        v1, metrics1 = native_gradient(agent, batch, batch["gae_adv"], permutation, actor_loss="ratio")
        v2_adv = mc_critic_advantages(batch, agent.gamma, pre_c)
        v2, metrics2 = native_gradient(agent, batch, v2_adv, permutation, actor_loss="score")
        if metrics1["ratio"] != 1.0 or metrics1["clipfrac"] != 0.0:
            raise AssertionError("Native single-step ratio/clip invariant failed: %r" % metrics1)
        vectors = np.stack([v1.double().cpu().numpy(), v2.double().cpu().numpy()])
        products = vectors @ vectors.T
        if not bool(np.isfinite(vectors).all() and np.isfinite(products).all()):
            raise FloatingPointError("Nonfinite gradient or scalar product")
        if products[0, 0] <= 0 or products[1, 1] <= 0:
            raise FloatingPointError("Zero gradient makes within-batch cosine undefined")
        gradient_seconds = time.perf_counter() - tick
        returns = as_episodes(batch["raw_rewards"], batch["firsts"]).sum(axis=1)
        discounted = discounted_episode_returns(batch["raw_rewards"], batch["firsts"], agent.gamma)
        sums += vectors
        row = {"batch": index, "norm_sq_V1": float(products[0, 0]),
               "norm_sq_V2": float(products[1, 1]), "cross_V1_V2": float(products[0, 1]),
               "within_batch_cosine": float(products[0, 1] / np.sqrt(products[0, 0] * products[1, 1])),
               "episode_returns": returns.tolist(),
               "initial_observations": batch["obs_k"]["state"][:agent.n_envs].detach().cpu().tolist(),
               "return_mean": float(returns.mean()),
               "return_se": float(returns.std(ddof=1) / np.sqrt(len(returns))),
               "J_disc_mean": float(discounted.mean()), "J_disc_se": float(discounted.std(ddof=1) / np.sqrt(len(discounted))),
               "prebatch_scaler_c": pre_c, "postbatch_scaler_c": post_c,
               "rollout_seconds": batch["timing"]["rollout_seconds"], "gradient_seconds": gradient_seconds,
               "seeds": seeds, "V1_ratio": metrics1["ratio"], "V2_ratio": metrics2["ratio"],
               "V1_clipfrac": metrics1["clipfrac"], "V2_clipfrac": metrics2["clipfrac"]}
        rows.append(row)
        records.append(seeds)
        meta["K"] = index + 1
        gm._atomic_csv(directory / "batches.csv", rows)
        gm._atomic_json(path, meta)
        gm._atomic_json(directory / "progress.json", {"status": "running", "completed": index + 1,
                                                       "target": total, "last_batch": row})
        LOG.info("PAIRED_NOISE batch=%d/%d rollout=%.3f gradient=%.3f norm1=%.8g norm2=%.8g cosine=%.8g",
                 index + 1, total, row["rollout_seconds"], gradient_seconds,
                 np.sqrt(products[0, 0]), np.sqrt(products[1, 1]), row["within_batch_cosine"])
        _assert_model_equal(agent, before)
    for variant in range(2):
        gm._save_vector(directory / ("sums_V%d.npy" % (variant + 1)), sums[variant])
    meta.update(status="measurement_complete_pending_cleanup", actor_and_critic_unchanged=True,
                critic_after_sha256=checkpoint_critic_hash(agent, spec["checkpoint"]),
                stored_vector_dtype="float64", group_vector_kind="sum")
    gm._atomic_json(path, meta)
    return meta, path


def make_config(spec):
    from omegaconf import OmegaConf
    source = Path(__file__).resolve().parents[2]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_path = source / "cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp.yaml"
    cfg = OmegaConf.load(str(config_path))
    cfg.wandb = None
    cfg.logdir = str(Path(spec["out"]).resolve())
    cfg.base_policy_path = str(Path(spec["checkpoint"]).resolve())
    cfg.device = spec.get("device", "cuda:0")
    cfg.seed = seed_record(spec, 0)["diffusion_seed"]
    if spec.get("normalization_path"):
        cfg.normalization_path = str(Path(spec["normalization_path"]).resolve())
    cfg.train.n_train_itr = 140
    cfg.train.save_model_freq = 35
    cfg.train.target_kl = None
    cfg.train.actor_single_step = True
    cfg.model.clip_ploss_coef = 1e6
    cfg.model.clip_ploss_coef_base = 1e6
    cfg.model.clamp_logprob = False
    cfg.model.randn_clip_value = 100
    OmegaConf.resolve(cfg)
    return cfg


def validate(agent, spec):
    if UPDATE_MAP.get(int(spec["checkpoint_itr"])) != int(spec["updates"]):
        raise ValueError("Checkpoint/update mapping differs from the native schedule")
    if (agent.n_envs, agent.n_steps, agent.model.ft_denoising_steps, agent.batch_size,
            agent.reward_horizon, agent.horizon_steps, agent.action_dim, agent.update_epochs) != (40, 500, 10, 50000, 4, 4, 6, 5):
        raise ValueError("Unexpected measurement geometry")
    if (agent.gamma, agent.gae_lambda, agent.model.gamma_denoising) != (.99, .95, .99):
        raise ValueError("Unexpected discount settings")
    if agent.max_grad_norm is not None or agent.use_bc_loss or agent.ent_coef != 0:
        raise ValueError("The measured actor loss must have no extra active terms or gradient clipping")
    if (agent.model.clamp_logprob or agent.model.randn_clip_value != 100 or
            agent.model.logprob_reduce != "mean" or not agent.model.norm_adv):
        raise ValueError("Native without_PPO loss/sampling flags differ")
    if (agent.model.min_logprob_denoising_std != .1 or
            agent.model.get_min_sampling_denoising_std() != .1 or
            agent.model.clip_ploss_coef != 1e6 or agent.model.clip_ploss_coef_base != 1e6):
        raise ValueError("Native noise floors or inactive ratio-clip bounds differ")
    if agent.reset_at_iteration:
        raise ValueError("Expected native reset_at_iteration=false")
    if sum(p.numel() for p in agent.model.actor_ft.parameters()) != 568392:
        raise ValueError("Unexpected actor parameter count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    if spec.get("operation") not in ("prepare", "collect"):
        raise ValueError("operation must be prepare or collect")
    directory = Path(spec["out"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("prepare.json", "meta.json", "cleanup.json", "failure.json")):
        raise FileExistsError("Refusing to overwrite previous job evidence")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    from omegaconf import OmegaConf
    from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
    cfg = make_config(spec)
    OmegaConf.save(cfg, str(directory / "resolved_config.yaml"))
    LOG.info("PAIRED_NOISE_SETTINGS %s", json.dumps(spec, sort_keys=True))
    LOG.info("PAIRED_NOISE_ESTIMATORS V1=ratio_GAE V2=score_raw_MC_minus_prebatch_cV normalization=native_50000_pair_chunks warmup=0")
    started, agent, error, result = time.monotonic(), None, None, None
    try:
        # Retain the partially built object so an initialization failure after
        # vector-env creation still closes only this process's own workers.
        agent = TrainPPODiffusionAgent.__new__(TrainPPODiffusionAgent)
        agent.logdir = str(directory)
        agent.__init__(cfg)
        validate(agent, spec)
        result = prepare(agent, spec) if spec["operation"] == "prepare" else collect(agent, spec)
    except BaseException as caught:
        error = caught
        gm._atomic_json(directory / "failure.json", {"status": "failed", "type": type(caught).__name__,
                                                      "message": str(caught), "spec": spec})
        raise
    finally:
        if agent is not None and hasattr(agent, "venv"):
            gm.cleanup_environment(agent, primary_error=error)
    if result is not None:
        meta, path = result
        meta.update(status="completed", elapsed_seconds=time.monotonic() - started, clean_exit=True)
        gm._atomic_json(path, meta)
        gm._atomic_json(directory / "progress.json", {"status": "completed", "completed": meta.get("K", len(meta.get("fit_batches", []))),
                                                       "elapsed_seconds": meta["elapsed_seconds"]})


if __name__ == "__main__":
    main()
