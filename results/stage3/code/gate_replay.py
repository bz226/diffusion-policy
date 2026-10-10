#!/usr/bin/env python
"""One isolated source-tree replay worker for the Stage-3 differential gate."""
import argparse
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from gate_common import (assignment_to, compare, freeze, numeric_settings, require,
                         restore_rng, rng_state, transfer)

FIELDS = ("obs_k", "chains_k", "returns_k", "values_k", "advantages_k", "logprobs_k")


def make_agent(repo, fixture, device, out, overrides=None):
    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    saved = torch.load(fixture, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(saved["config"])
    cfg.device = device
    cfg.model.device = device
    cfg.logdir = str(out / "native")
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, value, force_add=True)
    agent = get_class(cfg._target_)(cfg)
    agent.venv.close()
    agent._gate_constructor_actor_optimizer = freeze(agent.actor_optimizer.state_dict())
    agent._gate_constructor_critic_optimizer = freeze(agent.critic_optimizer.state_dict())
    agent._gate_constructor_actor_scheduler = freeze(agent.actor_lr_scheduler.state_dict())
    agent._gate_constructor_critic_scheduler = freeze(agent.critic_lr_scheduler.state_dict())
    agent.model.load_state_dict(saved["model_state"])
    for name, module in agent.model.named_modules():
        module.training = saved["module_training"][name]
    if not overrides or "train.actor_optimizer" not in overrides:
        agent.actor_optimizer.load_state_dict(copy.deepcopy(saved["actor_optimizer_state"]))
    agent.critic_optimizer.load_state_dict(copy.deepcopy(saved["critic_optimizer_state"]))
    agent.actor_lr_scheduler.load_state_dict(copy.deepcopy(saved["actor_scheduler_state"]))
    agent.critic_lr_scheduler.load_state_dict(copy.deepcopy(saved["critic_scheduler_state"]))
    agent.running_reward_scaler.__dict__.update(copy.deepcopy(saved["reward_scaler_after"]))
    for key, value in saved["attrs"].items():
        if not overrides or ("train." + key) not in overrides:
            setattr(agent, key, value)
    agent.diagnostic_generator.set_state(saved["diagnostic_generator_state"])
    return agent, saved, cfg


def update_body(repo):
    module = importlib.import_module("agent.finetune.train_ppo_diffusion_agent")
    source = (repo / "agent/finetune/train_ppo_diffusion_agent.py").read_text()
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
    run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    loop = next(node for node in run.body if isinstance(node, ast.While))
    update = next(node for node in loop.body if isinstance(node, ast.If)
                  and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.operand, ast.Name)
                  and node.test.operand.id == "eval_mode")
    start = next(i for i, node in enumerate(update.body) if assignment_to(node, "total_steps"))
    body = copy.deepcopy(update.body[start:])
    function = ast.FunctionDef(name="exact_update", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=name) for name in ("self",) + FIELDS],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=body, decorator_list=[])
    function.body.append(ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()),
                                                 args=[], keywords=[])))
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(module.__dict__)
    exec(compile(program, "<stage3-exact-update>", "exec"), namespace)
    return namespace["exact_update"]


def prepare_cpu(agent, saved, out):
    batch = transfer(saved["batch"], "cpu")
    old = torch.empty_like(batch["logprobs_k"])
    # Use the actor's exact final-epoch chunk order. This avoids comparing CPU
    # log-probs produced by different GEMM batch shapes in NC4's exact ratio=1
    # invariant, while providing both code versions the same fixed input.
    generator = torch.Generator(device="cpu")
    generator.set_state(saved["rng"]["torch_cpu"])
    total = agent.n_steps * agent.n_envs * agent.model.ft_denoising_steps
    for _ in range(agent.update_epochs):
        indices = torch.randperm(total, generator=generator)
    with torch.no_grad():
        for start in range(0, total, agent.batch_size):
            pairs = indices[start:start + agent.batch_size]
            decisions, steps = torch.unravel_index(pairs, (agent.n_steps * agent.n_envs,
                                                          agent.model.ft_denoising_steps))
            old[decisions, steps] = agent.model.get_logprobs_subsample(
                {key: value[decisions] for key, value in batch["obs_k"].items()},
                batch["chains_k"][decisions, steps], batch["chains_k"][decisions, steps + 1], steps).detach()
    torch.save(old, out / "cpu_old_logprobs.pt")
    (out / "cpu_preparation.json").write_text(json.dumps({
        "source": str(Path(agent.cfg.logdir).parent),
        "input_fixture": str(saved.get("source_revision")),
        "operation": "Recompute old log-probs once on the reference CPU at the saved theta_old, using the final critic epoch's permutation and actor chunk shapes; supply this identical tensor to both CPU replays. Other saved batch fields remain unchanged.",
        "max_abs_difference_from_gpu_capture": (old - saved["batch"]["logprobs_k"]).abs().max().item(),
        "shape": list(old.shape)}, indent=2) + "\n")


def sampling(agent, saved, device):
    agent.model.load_state_dict(saved["model_state"])
    shape = (40, 4, 6)
    generator = torch.Generator(device=device).manual_seed(91235)
    original_randn, original_like = torch.randn, torch.randn_like
    noises = [original_randn(shape, device=device, generator=generator) for _ in range(21)]
    consumed = []
    def take(kind, tensor_shape):
        i = len(consumed)
        require(i < 21 and tuple(tensor_shape) == shape, "fixed-noise sampler shape/count changed")
        require(kind == ("randn" if i == 0 else "randn_like"), "fixed-noise sampler order changed")
        consumed.append(kind)
        return noises[i].clone()
    def fixed_randn(size, **kwargs):
        return take("randn", size)
    def fixed_like(tensor, **kwargs):
        return take("randn_like", tensor.shape)
    forward = agent.model.forward.__wrapped__
    source = Path(importlib.import_module("model.diffusion.diffusion_vpg").__file__).read_text()
    inner_line = min(node.lineno for node in ast.walk(ast.parse(source)) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == "return_chain"
                     and node.lineno > forward.__code__.co_firstlineno)
    states = []
    def trace(frame, event, arg):
        if frame.f_code is not forward.__code__:
            return None
        if event == "line" and frame.f_lineno == inner_line:
            states.append(freeze(frame.f_locals["x"]))
        return trace
    before = rng_state()
    try:
        torch.randn, torch.randn_like = fixed_randn, fixed_like
        sys.settrace(trace)
        result = agent.model(cond={"state": saved["batch"]["obs_k"]["state"][:40].to(device)},
                             deterministic=False, return_chain=True)
    finally:
        sys.settrace(None)
        torch.randn, torch.randn_like = original_randn, original_like
    require(len(consumed) == 21 and len(states) == 20, "incomplete chain sampling evidence")
    return {"chain": freeze(result.chains), "trajectories": freeze(result.trajectories),
            "all_20_transition_states": torch.stack(states), "noise_tape": freeze(noises),
            "global_rng_before": before, "global_rng_after": rng_state()}


def replay(agent, saved, batch, repo, out, diagnostics=False):
    evidence = {"loss_components": [], "actor_gradients": [], "critic_gradients": [],
                "permutations": []}
    original_loss = agent.model.loss
    def loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        evidence["loss_components"].append(freeze(result))
        return result
    agent.model.loss = loss
    original_actor = agent.actor_optimizer.step
    original_critic = agent.critic_optimizer.step
    def actor(*args, **kwargs):
        evidence["actor_gradients"].append(freeze({name: p.grad for name, p in agent.model.actor_ft.named_parameters()}))
        return original_actor(*args, **kwargs)
    def critic(*args, **kwargs):
        evidence["critic_gradients"].append(freeze({name: p.grad for name, p in agent.model.critic.named_parameters()}))
        return original_critic(*args, **kwargs)
    agent.actor_optimizer.step = actor
    agent.critic_optimizer.step = critic
    original_perm = torch.randperm
    def permutation(*args, **kwargs):
        result = original_perm(*args, **kwargs)
        if kwargs.get("generator") is None:
            evidence["permutations"].append(freeze(result))
        return result
    restore_rng(saved["rng"])
    evidence["initial_rng"] = rng_state()
    evidence["initial_model"] = freeze(agent.model.state_dict())
    evidence["initial_actor_optimizer"] = freeze(agent.actor_optimizer.state_dict())
    evidence["initial_critic_optimizer"] = freeze(agent.critic_optimizer.state_dict())
    evidence["constructor_actor_optimizer"] = agent._gate_constructor_actor_optimizer
    evidence["constructor_critic_optimizer"] = agent._gate_constructor_critic_optimizer
    evidence["constructor_actor_scheduler"] = agent._gate_constructor_actor_scheduler
    evidence["constructor_critic_scheduler"] = agent._gate_constructor_critic_scheduler
    evidence["input_batch"] = freeze(batch)
    if getattr(agent, "stage3_active", False):
        agent._stage3_iteration_start()
        agent._stage3_actions = saved["action_trajs"].copy()
        agent._stage3_rollout_finished(saved["raw_reward_trajs"], saved["firsts_trajs"], False)
        before_diagnostics = rng_state()
        before_measurement = {"model": freeze(agent.model.state_dict()),
                              "actor_optimizer": freeze(agent.actor_optimizer.state_dict()),
                              "critic_optimizer": freeze(agent.critic_optimizer.state_dict()),
                              "actor_grads": freeze([p.grad for p in agent.model.actor_ft.parameters()]),
                              "critic_grads": freeze([p.grad for p in agent.model.critic.parameters()])}
        batch["advantages_k"] = agent._stage3_prepare_batch(
            saved["raw_reward_trajs"], saved["firsts_trajs"], saved["terminated_trajs"],
            saved["action_trajs"], batch["obs_k"], batch["chains_k"], batch["returns_k"],
            batch["values_k"], batch["advantages_k"], batch["logprobs_k"], saved["scaled_reward_trajs"])
        evidence["rng_before_prepare"] = before_diagnostics
        evidence["rng_after_prepare"] = rng_state()
        evidence["prepared_batch"] = freeze(batch)
        after_measurement = {"model": freeze(agent.model.state_dict()),
                             "actor_optimizer": freeze(agent.actor_optimizer.state_dict()),
                             "critic_optimizer": freeze(agent.critic_optimizer.state_dict()),
                             "actor_grads": freeze([p.grad for p in agent.model.actor_ft.parameters()]),
                             "critic_grads": freeze([p.grad for p in agent.model.critic.parameters()])}
        evidence["measurement_state_inert"] = compare(before_measurement, after_measurement, True)
        require(evidence["measurement_state_inert"]["passed"], "diagnostic preparation changed model, optimizer, or gradients")
    update = update_body(repo)
    try:
        torch.randperm = permutation
        local = update(agent, **batch)
    finally:
        torch.randperm = original_perm
    evidence.update({"actor_parameters": freeze(dict(agent.model.actor_ft.named_parameters())),
                     "critic_parameters": freeze(dict(agent.model.critic.named_parameters())),
                     "model_state": freeze(agent.model.state_dict()),
                     "actor_optimizer_state": freeze(agent.actor_optimizer.state_dict()),
                     "critic_optimizer_state": freeze(agent.critic_optimizer.state_dict()),
                     "actor_scheduler_state": freeze(agent.actor_lr_scheduler.state_dict()),
                     "critic_scheduler_state": freeze(agent.critic_lr_scheduler.state_dict()),
                     "rng_after_update": rng_state(),
                     "diagnostic_generator_state": agent.diagnostic_generator.get_state().clone()})
    keys = ("loss", "pg_loss", "entropy_loss", "v_loss", "clipfracs", "approx_kl", "ratio", "bc_loss",
            "eta", "actor_optimizer_steps", "critic_optimizer_steps", "actor_lr", "critic_lr",
            "actor_step_metrics", "explained_var", "diagnostics")
    evidence["logged_values"] = {key: freeze(local[key]) for key in keys if key in local}
    evidence["input_batch_after"] = freeze(batch)
    before_batch = evidence.get("prepared_batch", evidence["input_batch"])
    evidence["update_batch_inert"] = compare(before_batch, evidence["input_batch_after"], True)
    require(evidence["update_batch_inert"]["passed"], "update or diagnostics mutated the saved input batch")
    evidence["stage3_metrics"] = freeze(getattr(agent, "_stage3_metrics", {}))
    evidence["sampling"] = sampling(agent, saved, str(agent.device))
    torch.save(evidence, out / "evidence.pt")
    return {"actor_steps": len(evidence["actor_gradients"]), "critic_steps": len(evidence["critic_gradients"]),
            "loss_calls": len(evidence["loss_components"]), "permutations": len(evidence["permutations"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--cpu-logprobs", type=Path)
    parser.add_argument("--prepare-cpu", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--e2", action="store_true")
    args = parser.parse_args()
    args.repo, args.fixture, args.output = args.repo.resolve(), args.fixture.resolve(), args.output.resolve()
    require(not args.output.exists(), "gate worker output already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "command": sys.argv, "numeric_settings_before": numeric_settings()}
    try:
        require(not torch.are_deterministic_algorithms_enabled(), "deterministic algorithms unexpectedly enabled")
        overrides = {"train.stage3_diag": True} if args.diagnostics else {}
        if args.e2:
            overrides.update({"model.actor_loss": "score", "train.adv_estimator": "mc_loo",
                              "train.decision_discount": True, "model.gamma_denoising": 1.,
                              "model.logprob_reduce": "sum", "model.norm_adv": False})
        agent, saved, cfg = make_agent(args.repo, args.fixture, args.device, args.output, overrides)
        if args.prepare_cpu:
            require(args.device == "cpu", "CPU fixture preparation requires CPU")
            prepare_cpu(agent, saved, args.output)
        else:
            batch = transfer(saved["batch"], args.device)
            if args.cpu_logprobs:
                batch["logprobs_k"] = torch.load(args.cpu_logprobs, map_location=args.device, weights_only=True)
            report.update(replay(agent, saved, batch, args.repo, args.output, args.diagnostics))
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["numeric_settings_after"] = numeric_settings()
        (args.output / "worker.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
