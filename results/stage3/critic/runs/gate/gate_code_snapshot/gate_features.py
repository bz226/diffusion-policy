#!/usr/bin/env python
"""Isolated C1/C2/C3 correctness workers; Python 3.8, no campaign submission."""
import argparse
import copy
import importlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from gate_common import compare, freeze, require, rng_state, transfer
from gate_replay import make_agent


def json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def error(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    delta = actual - reference
    return {"max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(reference.ravel()), np.finfo(np.float64).tiny))}


def brute_returns(rewards, firsts, gamma):
    """Independent forward double sum, stopping at each observed boundary."""
    rewards = np.asarray(rewards, dtype=np.float64)
    result = np.zeros_like(rewards)
    for env in range(rewards.shape[1]):
        boundaries = np.flatnonzero(firsts[:, env])
        require(boundaries[0] == 0 and boundaries[-1] == len(rewards), "incomplete saved episodes")
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            for decision in range(begin, end):
                for future in range(decision, end):
                    result[decision, env] += gamma ** (future - decision) * rewards[future, env]
    return result


def e2_critic_overrides():
    return {"model.actor_loss": "score", "train.adv_estimator": "mc_critic",
            "train.decision_discount": True, "model.gamma_denoising": 1.,
            "model.logprob_reduce": "sum", "model.norm_adv": False,
            "train.actor_optimizer": "sgd", "train.actor_lr": 1e-4,
            "train.actor_lr_scheduler.min_lr": 1e-4}


def c1(args):
    agent, saved, cfg = make_agent(args.repo, args.fixture, "cpu", args.output, e2_critic_overrides())
    from agent.finetune.critic_baseline import mc_critic_advantages, reward_scaler_divisor
    from agent.finetune.stage3_math import episode_indices
    batch = transfer(saved["batch"], "cpu")
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved["reward_scaler_before"]))
    divisor = np.sqrt(np.asarray(agent.running_reward_scaler.ret_rms.var, dtype=np.float64)
                      + agent.running_reward_scaler.epsilon)
    require(np.array_equal(divisor, reward_scaler_divisor(agent)), "independent start divisor differs")
    raw_returns = brute_returns(saved["raw_reward_trajs"], saved["firsts_trajs"], agent.gamma)
    values = batch["values_k"].numpy().astype(np.float64).reshape(raw_returns.shape)
    expected = raw_returns - divisor * values
    direct = mc_critic_advantages(saved["raw_reward_trajs"], saved["firsts_trajs"], agent.gamma, values, divisor)
    helper_error = error(expected, direct)
    require(direct.dtype == np.float64 and helper_error["relative_l2"] <= 1e-12,
            "C1 float64 mc_critic differs from independent G-cV")
    # The saved first training iteration can have c very close to1. Exercise
    # nonunit divisors too, so an accidentally omitted multiplication cannot pass.
    divisor_cases = {}
    for name, candidate in (("nonunit_scalar", np.asarray(3.75, dtype=np.float64)),
                            ("per_environment", np.linspace(2., 6., raw_returns.shape[1]))):
        value = mc_critic_advantages(saved["raw_reward_trajs"], saved["firsts_trajs"], agent.gamma, values, candidate)
        comparison = error(raw_returns - values * candidate, value)
        require(value.dtype == np.float64 and comparison["relative_l2"] <= 1e-12,
                "C1 nonunit divisor formula/broadcast failed")
        divisor_cases[name] = comparison
    agent._stage3_iteration_start()
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved["reward_scaler_after"]))
    agent._stage3_actions = saved["action_trajs"].copy()
    agent._stage3_rollout_finished(saved["raw_reward_trajs"], saved["firsts_trajs"], False)
    actual = agent._stage3_prepare_batch(
        saved["raw_reward_trajs"], saved["firsts_trajs"], saved["terminated_trajs"],
        saved["action_trajs"], batch["obs_k"], batch["chains_k"], batch["returns_k"],
        batch["values_k"], batch["advantages_k"], batch["logprobs_k"], saved["scaled_reward_trajs"])
    cache_error = error(expected, agent._critic_baseline_raw_adv)
    require(agent._critic_baseline_raw_adv.dtype == np.float64 and cache_error["relative_l2"] <= 1e-12,
            "C1 live preparation cache differs from G-cV")
    k = episode_indices(saved["firsts_trajs"])
    require(np.array_equal(k, np.tile((np.arange(500) % 250)[:, None], (1, 40))), "wrong within-episode k")
    # Native code rounds raw float64 residual to float32, then applies gamma^k.
    target = torch.as_tensor(direct, dtype=torch.float32).reshape(-1) * torch.as_tensor(
        agent.gamma ** k, dtype=torch.float32).reshape(-1)
    discount_check = compare(target, actual, True, "post_raw_cast_decision_discount")
    require(discount_check["passed"] and agent.model.norm_adv is False,
            "C1 discount must follow raw residual, with normalization disabled")
    torch.save({"raw_returns_explicit": raw_returns, "raw_residual_explicit": expected,
                "native_raw_residual": agent._critic_baseline_raw_adv,
                "expected_post_discount": target, "actual_post_discount": actual,
                "divisor": divisor, "audit": agent._critic_baseline_audit}, args.output / "evidence.pt")
    return {"passed": True, "helper": helper_error, "preparation": cache_error,
            "nonunit_divisor_checks": divisor_cases,
            "post_discount": discount_check, "norm_adv": agent.model.norm_adv,
            "divisor": json_value(divisor), "pairs": 200000, "episodes": 80,
            "precision": "G and cV float64; native float32 cast before decision discount"}


def config(args, output, overrides):
    from omegaconf import OmegaConf
    saved = torch.load(args.fixture, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(saved["config"])
    cfg.logdir = str(output)
    cfg.device = "cuda:0"
    cfg.model.device = "cuda:0"
    cfg.train.n_train_itr = 140
    cfg.train.save_model_freq = 35
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def construct(args, output, overrides):
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from agent.finetune.grad_mean import cleanup_environment
    cfg = config(args, output, overrides)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'gate_config.yaml', resolve=True)
    cls = get_class(cfg._target_)
    agent = cls.__new__(cls)
    try:
        cls.__init__(agent, cfg)
    except BaseException as exc:
        if hasattr(agent, "venv"):
            if not hasattr(agent, "logdir"):
                agent.logdir = str(cfg.logdir)
            cleanup_environment(agent, primary_error=exc)
        raise
    return agent


def close(agent):
    from agent.finetune.grad_mean import cleanup_environment
    record = cleanup_environment(agent)
    require(record["status"] == "passed" and record["all_children_exited"], "environment cleanup failed")
    require(not mp.active_children(), "gate worker still has children")
    return record


def c3(args):
    from agent.finetune.grad_mean import actor_state, scaler_state, state_sha256
    cold = construct(args, args.output / "cold", {})
    try:
        before_actor = actor_state(cold)
        before_optimizers = freeze({"actor": cold.actor_optimizer.state_dict(),
                                   "critic": cold.critic_optimizer.state_dict()})
    finally:
        cold_cleanup = close(cold)
    warm = construct(args, args.output / "warm", {"train.critic_init_path": str(args.frozen)})
    try:
        payload = torch.load(args.frozen, map_location="cpu", weights_only=True)
        checks = {"critic": compare(payload["critic"], freeze(warm.model.critic.state_dict()), True),
                  "scaler": compare(payload["reward_scaler"], scaler_state(warm), True),
                  "actor_and_actor_ft": compare(before_actor, actor_state(warm), True),
                  "fresh_optimizer_states": compare(before_optimizers, freeze({"actor": warm.actor_optimizer.state_dict(),
                                                 "critic": warm.critic_optimizer.state_dict()}), True)}
        hashes = {"stored_critic": payload["critic_sha256"],
                  "loaded_critic": state_sha256(warm.model.critic.state_dict()),
                  "stored_scaler": payload["scaler_sha256"], "loaded_scaler": state_sha256(scaler_state(warm)),
                  "actor_cold": state_sha256(before_actor), "actor_warm": state_sha256(actor_state(warm))}
        require(all(value["passed"] for value in checks.values()), "C3 initialization changed unexpected state")
        require(hashes["stored_critic"] == hashes["loaded_critic"]
                and hashes["stored_scaler"] == hashes["loaded_scaler"]
                and hashes["actor_cold"] == hashes["actor_warm"], "C3 hash mismatch")
        report = {"passed": True, "comparisons": checks, "hashes": hashes,
                  "constructor_audit": warm._critic_init_audit,
                  "frozen_path": str(args.frozen), "cold_cleanup": cold_cleanup}
    finally:
        warm_cleanup = close(warm)
    report["warm_cleanup"] = warm_cleanup
    return report


class CompletedIteration(Exception):
    pass


def c2(args):
    from agent.finetune.critic_baseline import reward_scaler_divisor
    from agent.finetune.grad_mean import state_sha256
    overrides = e2_critic_overrides()
    overrides.update({"train.stage3_diag": True, "train.baseline_diag": True})
    agent = construct(args, args.output / "native", overrides)
    start_original, prepare_original, record_original = (
        agent._stage3_iteration_start, agent._stage3_prepare_batch, agent._stage3_record)
    scaler_class = type(agent.running_reward_scaler)
    scaler_call = scaler_class.__call__
    evidence = {"actor_steps": 0, "critic_steps": 0}
    step_actor, step_critic = agent.actor_optimizer.step, agent.critic_optimizer.step
    def actor_step(*pos, **kw):
        evidence["actor_steps"] += 1
        return step_actor(*pos, **kw)
    def critic_step(*pos, **kw):
        evidence["critic_steps"] += 1
        return step_critic(*pos, **kw)
    def start():
        # Snapshot independently before the implementation captures its c.
        if agent.itr == 1:
            evidence["critic_start"] = freeze(agent.model.critic.state_dict())
            evidence["divisor_start"] = reward_scaler_divisor(agent).copy()
            evidence["critic_start_sha256"] = state_sha256(evidence["critic_start"])
            evidence["divisor_start_sha256"] = state_sha256(evidence["divisor_start"])
        return start_original()
    def scale(instance, *pos, **kw):
        if instance is agent.running_reward_scaler and agent.itr == 1:
            require("divisor_before_batch_scaler_update" not in evidence, "multiple scaler calls in training iteration")
            evidence["divisor_before_batch_scaler_update"] = reward_scaler_divisor(agent).copy()
        return scaler_call(instance, *pos, **kw)
    def prepare(*pos, **kw):
        require(agent.itr == 1, "unexpected training iteration")
        evidence["critic_use"] = freeze(agent.model.critic.state_dict())
        evidence["critic_use_sha256"] = state_sha256(evidence["critic_use"])
        evidence["divisor_used"] = agent._critic_baseline_divisor.copy()
        evidence["divisor_used_sha256"] = state_sha256(evidence["divisor_used"])
        require(evidence["critic_steps"] == 0 and evidence["actor_steps"] == 0,
                "a parameter update happened before baseline use")
        before_rng = rng_state()
        result = prepare_original(*pos, **kw)
        evidence["prepare_rng_inert"] = compare(before_rng, rng_state(), True)
        evidence["native_audit"] = copy.deepcopy(agent._critic_baseline_audit)
        evidence["raw_advantages"] = agent._critic_baseline_raw_adv.copy()
        evidence["raw_returns"] = agent._critic_baseline_raw_returns.copy()
        evidence["preupdate_values"] = agent._critic_baseline_values.copy()
        return result
    def record(result, eval_mode):
        record_original(result, eval_mode)
        if agent.itr == 1:
            require(not eval_mode, "live gate itr1 was not training")
            evidence["iteration_record"] = copy.deepcopy(result)
            raise CompletedIteration()
    agent._stage3_iteration_start = start
    agent._stage3_prepare_batch = prepare
    agent._stage3_record = record
    agent.actor_optimizer.step, agent.critic_optimizer.step = actor_step, critic_step
    scaler_class.__call__ = scale
    caught = False
    try:
        try:
            agent.run()
        except CompletedIteration:
            caught = True
        require(caught, "live gate did not finish exactly one training iteration")
        checks = {"critic_start_vs_use": compare(evidence["critic_start"], evidence["critic_use"], True),
                  "divisor_start_vs_pre_scaler": compare(evidence["divisor_start"], evidence["divisor_before_batch_scaler_update"], True),
                  "divisor_start_vs_used": compare(evidence["divisor_start"], evidence["divisor_used"], True),
                  "prepare_rng": evidence["prepare_rng_inert"]}
        require(all(value["passed"] for value in checks.values()), "C2 pre-update critic/divisor invariant failed")
        require(evidence["critic_start_sha256"] == evidence["critic_use_sha256"]
                and evidence["divisor_start_sha256"] == evidence["divisor_used_sha256"], "C2 independent hash mismatch")
        require(evidence["actor_steps"] == 1 and evidence["critic_steps"] == 20,
                "live C2 iteration did not finish one actor and twenty critic steps")
        torch.save(evidence, args.output / "evidence.pt")
        report = {"passed": True, "comparisons": checks,
                  "hashes": {key: value for key, value in evidence.items() if key.endswith("sha256")},
                  "native_audit": evidence["native_audit"],
                  "actor_steps": evidence["actor_steps"], "critic_steps": evidence["critic_steps"],
                  "iteration_record": evidence["iteration_record"],
                  "protocol": "Unmodified 140-iteration loop configuration; harness stops after eval0 and the complete training1 record, before iteration2."}
    finally:
        scaler_class.__call__ = scaler_call
        cleanup = close(agent)
    report["cleanup"] = cleanup
    return report


def main():
    def interrupted(signum, frame):
        raise RuntimeError("feature worker interrupted by signal %d" % signum)
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", choices=("C1", "C2", "C3"), required=True)
    args = parser.parse_args()
    for name in ("repo", "fixture", "frozen", "output"):
        setattr(args, name, getattr(args, name).resolve())
    require(not args.output.exists(), "gate output already exists")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    report = {"feature": args.feature, "passed": False, "status": "running", "command": sys.argv}
    try:
        require(not torch.are_deterministic_algorithms_enabled(), "deterministic algorithms must stay disabled")
        report.update({"C1": c1, "C2": c2, "C3": c3}[args.feature](args))
        require(not mp.active_children(), "feature worker left child processes")
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (args.output / "feature.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
