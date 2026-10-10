#!/usr/bin/env python
"""Feature checks B1--B8/B10 using the live saved batch and production helpers."""
import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from gate_common import freeze, require, restore_rng, transfer
from gate_replay import make_agent


def relative_l2(actual, expected):
    actual = torch.as_tensor(actual).detach().double().reshape(-1)
    expected = torch.as_tensor(expected).detach().double().reshape(-1)
    delta = actual - expected
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected)).item()}


def full_gradient(agent, batch, indices, *, actor_loss, reduction, gamma_denoising,
                  norm_adv, advantages):
    model = agent.model
    model.actor_loss = actor_loss
    model.logprob_reduce = reduction
    model.gamma_denoising = gamma_denoising
    model.norm_adv = norm_adv
    model.clamp_logprob = False
    params = list(model.actor_ft.parameters())
    accumulated = [torch.zeros_like(p) for p in params]
    for start in range(0, len(indices), agent.batch_size):
        inds = indices[start:start + agent.batch_size]
        decisions, denoising = torch.unravel_index(inds, (agent.n_steps * agent.n_envs,
                                                         model.ft_denoising_steps))
        loss = model.loss({key: value[decisions] for key, value in batch["obs_k"].items()},
                          batch["chains_k"][decisions, denoising],
                          batch["chains_k"][decisions, denoising + 1], denoising,
                          batch["returns_k"][decisions], batch["values_k"][decisions],
                          advantages[decisions].clone(), batch["logprobs_k"][decisions, denoising],
                          use_bc_loss=False, reward_horizon=agent.reward_horizon)[0]
        gradient = torch.autograd.grad(loss * (len(inds) / len(indices)), params)
        for target, value in zip(accumulated, gradient):
            target.add_(value.detach())
    return torch.cat([value.reshape(-1) for value in accumulated])


def brute_mc(rewards, firsts, gamma):
    result = np.empty_like(rewards, dtype=np.float64)
    for env in range(rewards.shape[1]):
        for t in range(len(rewards)):
            after = np.flatnonzero(firsts[t + 1:, env])
            stop = min(len(rewards), t + 1 + int(after[0])) if len(after) else len(rewards)
            result[t, env] = np.dot(rewards[t:stop, env], gamma ** np.arange(stop - t))
    return result


def original_gae(raw, terminated):
    reference = Path(__file__).resolve().parents[1] / "reference_stage2"
    source = (reference / "agent/finetune/train_ppo_diffusion_agent.py").read_text()
    loop = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)
                and ast.get_source_segment(source, node.iter) == "reversed(range(self.n_steps))")
    prefix = ast.parse("advantages_trajs = np.zeros_like(reward_trajs)\nlastgaelam = 0\n").body
    fn = ast.FunctionDef(name="original_gae", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=name) for name in ("self", "reward_trajs", "terminated_trajs", "values_trajs", "obs_venv_ts")],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=prefix + [copy.deepcopy(loop), ast.Return(value=ast.Name(id="advantages_trajs", ctx=ast.Load()))],
        decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {"np": np}
    exec(compile(tree, "<unchanged-stage2-GAE-recursion>", "exec"), ns)
    settings = SimpleNamespace(n_steps=len(raw), reward_scale_const=1., gamma=.99, gae_lambda=1.,
                               model=SimpleNamespace(critic=lambda obs: torch.zeros(raw.shape[1])))
    return ns["original_gae"](settings, raw, terminated, np.zeros_like(raw), {})


def math_checks(mathmod, saved):
    checks = {}
    rng = np.random.default_rng(4021)
    random_rewards = rng.normal(size=(43, 4))
    random_firsts = (rng.uniform(size=(44, 4)) < 0.13).astype(float)
    random_firsts[[0, -1]] = 1
    errors = []
    for name, reward, firsts in (("random_boundaries", random_rewards, random_firsts),
                                 ("live_batch", saved["raw_reward_trajs"], saved["firsts_trajs"])):
        actual = mathmod.mc_returns(reward, firsts, .99)
        expected = brute_mc(reward, firsts, .99)
        error = relative_l2(actual, expected)
        require(error["relative_l2"] <= 1e-12, "B2a MC returns disagree with double sum: " + name)
        errors.append({"case": name, **error})
    checks["B2a"] = {"passed": True, "comparisons": errors}
    raw, firsts, terminated = saved["raw_reward_trajs"], saved["firsts_trajs"], saved["terminated_trajs"]
    mc = mathmod.mc_returns(raw, firsts, .99)
    recursion = original_gae(raw, terminated)
    error = relative_l2(recursion, mc)
    ends = firsts[1:] == 1
    checks["B2b"] = {"report_only": True, "matches": error["relative_l2"] <= 1e-12,
                       "recursion": "Exact Stage2 source AST; lambda=1, values=0, raw rewards, reward_scale_const=1",
                       "episode_ends": int(ends.sum()),
                       "terminated_at_ends": int(terminated[ends].sum()),
                       "truncated_at_ends": int(saved["truncated_trajs"][ends].sum()), **error}
    advantage, baseline = mathmod.loo_advantages(mc, firsts)
    episodes = mc.T.reshape(-1, 250)
    explicit = np.array([np.delete(episodes, i, axis=0).mean(axis=0) for i in range(len(episodes))])
    expected_baseline = explicit.reshape(raw.shape[1], raw.shape[0]).T
    error = relative_l2(baseline, expected_baseline)
    sums = advantage.T.reshape(-1, 250).sum(axis=0)
    tolerance = 64 * np.finfo(np.float64).eps * len(episodes) * np.abs(episodes).max()
    require(error["relative_l2"] <= 1e-12 and np.abs(sums).max() <= tolerance, "B3 LOO formula or zero-sum identity failed")
    checks["B3"] = {"passed": True, **error, "max_abs_advantage_sum_per_k": float(np.abs(sums).max()),
                     "sum_roundoff_bound": float(tolerance), "episodes": len(episodes)}
    k = mathmod.episode_indices(firsts)
    expected = np.broadcast_to((np.arange(len(raw)) % 250)[:, None], raw.shape)
    require(np.array_equal(k, expected), "B4 within-episode index differs from t mod 250")
    checks["B4"] = {"passed": True, "index_max_abs": int(np.abs(k - expected).max()),
                     "discount_max_abs": float(np.max(np.abs(.99 ** k - .99 ** expected))),
                     "episode_start_discount": float((.99 ** k)[250, 0]),
                     "last_decision_discount": float((.99 ** k)[249, 0])}

    origin = torch.tensor([10.0, -5.0, 2.0], dtype=torch.float64)
    p = torch.nn.Parameter(origin + torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64))
    inside = p.detach().clone()
    inside_metrics = mathmod.project_ball_([p], origin, 4.0)
    require(torch.equal(p, inside) and not inside_metrics["proj_active"], "B8 inside projection changed parameters")
    old = p.detach().clone() - origin
    outside_metrics = mathmod.project_ball_([p], origin, 1.7)
    after = p.detach() - origin
    radius_error = abs(torch.linalg.vector_norm(after).item() / 1.7 - 1)
    parallel_error = 1 - torch.nn.functional.cosine_similarity(after[None], old[None]).item()
    require(radius_error <= 1e-6 and abs(parallel_error) <= 1e-12 and outside_metrics["proj_active"],
            "B8 outside projection violates radius/parallelism")
    checks["B8"] = {"passed": True, "inside_bitwise": True,
                     "outside_relative_radius_error": radius_error, "one_minus_cosine": parallel_error}

    # Gaussian analytic standard errors for the unbiased mean-square and trace estimates.
    n = dimension = 1000
    variance = np.linspace(.5, 1.5, dimension)
    mean = np.full(dimension, 2 / np.sqrt(dimension))
    vectors = rng.normal(size=(n, dimension)) * np.sqrt(variance) + mean
    moment = mathmod.noise_moments(vectors.sum(axis=0)[None], np.array([[np.sum(vectors * vectors)]]), n)
    u_sq, tr_sigma = float(mean @ mean), float(variance.sum())
    se_signal = np.sqrt(4 * np.sum(mean * mean * variance) / n
                        + 2 * np.sum(variance ** 2) / (n * (n - 1)))
    se_variance = np.sqrt(2 * np.sum(variance ** 2) / (n - 1))
    signal_error = abs(float(moment["u_sq"][0]) - u_sq)
    variance_error = abs(float(moment["tr_cov"][0]) - tr_sigma)
    require(signal_error <= 3 * se_signal and variance_error <= 3 * se_variance,
            "B10 Gaussian signal/variance estimator outside three analytic standard errors")
    checks["B10"] = {"passed": True, "n": n, "dimension": dimension,
                      "true_u_sq": u_sq, "estimated_u_sq": float(moment["u_sq"][0]),
                      "true_tr_sigma": tr_sigma, "estimated_tr_sigma": float(moment["tr_cov"][0]),
                      "se_u_sq": se_signal, "se_tr_sigma": se_variance,
                      "signal_error_in_se": signal_error / se_signal,
                      "variance_error_in_se": variance_error / se_variance}
    return checks, advantage, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--cpu-logprobs", type=Path)
    args = parser.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    require(not args.output.exists(), "feature gate output exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "command": sys.argv, "checks": {}}
    try:
        agent, saved, cfg = make_agent(args.repo, args.fixture.resolve(), args.device, args.output)
        mathmod = importlib.import_module("agent.finetune.stage3_math")
        batch = transfer(saved["batch"], args.device)
        if args.cpu_logprobs:
            batch["logprobs_k"] = torch.load(args.cpu_logprobs, map_location=args.device, weights_only=True)
        checks, loo, k = math_checks(mathmod, saved)
        report["checks"].update(checks)
        restore_rng(saved["rng"])
        for _ in range(agent.update_epochs):
            indices = torch.randperm(agent.n_steps * agent.n_envs * agent.model.ft_denoising_steps, device=args.device)
        common = dict(reduction="mean", gamma_denoising=.99, norm_adv=True,
                      advantages=batch["advantages_k"])
        ratio = full_gradient(agent, batch, indices, actor_loss="ratio", **common)
        score = full_gradient(agent, batch, indices, actor_loss="score", **common)
        error = relative_l2(score, ratio)
        tolerance = 1e-5 if args.device == "cpu" else 1e-4
        require(error["relative_l2"] <= tolerance, "B1 score/ratio gradient mismatch")
        report["checks"]["B1"] = {"passed": True, "tolerance": tolerance, **error}
        agent.adv_estimator = "mc_loo"
        agent.decision_discount = True
        agent.model.norm_adv = False
        agent._stage3_iteration_start()
        raw_adv = agent._stage3_prepare_batch(saved["raw_reward_trajs"], saved["firsts_trajs"],
            saved["terminated_trajs"], saved["action_trajs"], batch["obs_k"], batch["chains_k"],
            batch["returns_k"], batch["values_k"], batch["advantages_k"], batch["logprobs_k"],
            saved["scaled_reward_trajs"])
        expected_adv = (torch.as_tensor(loo, device=args.device, dtype=torch.float32)
                        * torch.as_tensor(.99 ** k, device=args.device, dtype=torch.float32)).reshape(-1)
        require(torch.equal(raw_adv, expected_adv), "B4 production actor coefficients use incorrect decision discount")
        report["checks"]["B4"]["production_e2_coefficients_bitwise_equal"] = True
        common = dict(actor_loss="score", gamma_denoising=1., norm_adv=False, advantages=raw_adv)
        mean = full_gradient(agent, batch, indices, reduction="mean", **common)
        summed = full_gradient(agent, batch, indices, reduction="sum", **common)
        error = relative_l2(summed, mean * 24)
        require(error["relative_l2"] <= 1e-6, "B5 sum/mean gradient does not have factor24")
        report["checks"]["B5"] = {"passed": True, "factor": 24, "tolerance": 1e-6, **error}
        weighted = full_gradient(agent, batch, indices, actor_loss="score", reduction="sum",
                                 gamma_denoising=.99, norm_adv=False, advantages=raw_adv)
        cosine = torch.nn.functional.cosine_similarity(summed.double()[None], weighted.double()[None]).item()
        report["checks"]["B6"] = {"report_only": True, "cosine": cosine,
                                    "expected_above_0999": cosine > .999}
        # Exercise production float32 actor coordinates on this requested device,
        # with the actual pretrained center and a deterministic perturbation.
        actor_parameters = list(agent.model.actor_ft.parameters())
        origin = torch.cat([p.detach().reshape(-1) for p in actor_parameters]).clone()
        radius = float(json.loads((Path(__file__).resolve().parents[1] /
                                   "provenance/anchors.json").read_text())["projection_radius"])
        direction = torch.linspace(-1., 1., origin.numel(), device=args.device, dtype=origin.dtype)
        direction *= (2 * radius / torch.linalg.vector_norm(direction.double())).to(direction.dtype)
        perturbed = origin + direction
        with torch.no_grad():
            offset = 0
            for p in actor_parameters:
                p.copy_(perturbed[offset:offset + p.numel()].view_as(p))
                offset += p.numel()
        original_displacement = perturbed.double() - origin.double()
        inside_result = mathmod.project_ball_(actor_parameters, origin, 3 * radius)
        inside = torch.cat([p.detach().reshape(-1) for p in actor_parameters])
        require(torch.equal(inside, perturbed) and not inside_result["proj_active"],
                "B8 actual actor inside-ball projection changed float32 parameters")
        outside_result = mathmod.project_ball_(actor_parameters, origin, radius)
        projected = torch.cat([p.detach().reshape(-1) for p in actor_parameters]).double() - origin.double()
        radius_error = abs(torch.linalg.vector_norm(projected).item() / radius - 1)
        parallel_error = abs(1 - torch.nn.functional.cosine_similarity(
            projected[None], original_displacement[None]).item())
        require(radius_error <= 1e-6 and parallel_error <= 1e-6 and outside_result["proj_active"],
                "B8 actual actor projection violates float32 radius/parallelism tolerance")
        report["checks"]["B8"]["production_actor"] = {
            "device": args.device, "dtype": str(origin.dtype), "parameters": origin.numel(),
            "nonzero_pretrained_center_norm": torch.linalg.vector_norm(origin.double()).item(),
            "radius": radius, "inside_bitwise": True,
            "outside_relative_radius_error": radius_error, "one_minus_cosine": parallel_error,
            "parallelism_tolerance": 1e-6}
        # The production constructor selects SGD; the mathematical update is checked on actual actor tensors.
        sgd_agent, _, _ = make_agent(args.repo, args.fixture.resolve(), args.device, args.output / "sgd",
                                     {"train.actor_optimizer": "sgd", "train.actor_lr": 1e-4,
                                      "train.actor_lr_scheduler.min_lr": 1e-4})
        require(isinstance(sgd_agent.actor_optimizer, torch.optim.SGD), "B7 constructor did not select SGD")
        opt = sgd_agent.actor_optimizer
        for group in opt.param_groups:
            require(group.get("momentum", 0) == 0 and group.get("weight_decay", 0) == 0
                    and not group.get("nesterov", False), "B7 SGD is not plain momentum-free SGD")
        params = list(sgd_agent.model.actor_ft.parameters())
        before = [p.detach().clone() for p in params]
        offset = 0
        for p in params:
            p.grad = summed[offset:offset + p.numel()].view_as(p).clone()
            offset += p.numel()
        eta = 1e-4
        expected = torch.cat([(p.double() - eta * q.grad.double()).reshape(-1) for p, q in zip(before, params)])
        opt.step()
        actual = torch.cat([p.detach().reshape(-1) for p in params])
        error = relative_l2(actual, expected)
        require(error["relative_l2"] <= 1e-7, "B7 SGD update is not theta-eta gradient")
        lrs = []
        for _ in range(20):
            sgd_agent.actor_lr_scheduler.step()
            lrs.append(float(opt.param_groups[0]["lr"]))
        require(all(lr == eta for lr in lrs), "B7 SGD learning rate is not constant")
        report["checks"]["B7"] = {"passed": True, **error, "eta": eta, "lr_over_20_steps": lrs}
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (args.output / "features.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
