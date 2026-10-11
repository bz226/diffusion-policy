#!/usr/bin/env python
"""G1 worker: a saved batch, without rollout or optimizer updates."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from gate_common import compare, freeze, require, restore_rng, rng_state, transfer
from gate_replay import make_agent


def full_loss_gradient(agent, saved, batch, advantages):
    """E2's actual PPODiffusion.loss, independent of the measurement helper."""
    model = agent.model
    model.actor_loss, model.logprob_reduce = "score", "sum"
    model.gamma_denoising, model.norm_adv, model.clamp_logprob = 1., False, False
    params = list(model.actor_ft.parameters())
    gradients = [torch.zeros_like(p) for p in params]
    total = agent.n_steps * agent.n_envs * model.ft_denoising_steps
    advantages = torch.as_tensor(advantages, device=agent.device, dtype=torch.float32).reshape(-1)
    inputs = transfer(saved["batch"], agent.device)
    for start in range(0, total, agent.batch_size):
        pairs = torch.arange(start, min(start + agent.batch_size, total), device=agent.device)
        decisions = torch.div(pairs, model.ft_denoising_steps, rounding_mode="floor")
        denoise = pairs % model.ft_denoising_steps
        loss = model.loss(
            {key: val[decisions] for key, val in batch["obs_k"].items()},
            batch["chains_k"][decisions, denoise], batch["chains_k"][decisions, denoise + 1],
            denoise, inputs["returns_k"][decisions], inputs["values_k"][decisions],
            advantages[decisions].clone(), inputs["logprobs_k"][decisions, denoise],
            use_bc_loss=False, reward_horizon=agent.reward_horizon)[0]
        values = torch.autograd.grad(loss * (len(pairs) / total), params)
        for target, value in zip(gradients, values):
            target.add_(value.detach())
    return -torch.cat([x.reshape(-1) for x in gradients]).double().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("reference", "edited", "e2"), required=True)
    args = parser.parse_args()
    args.repo, args.fixture, args.output = args.repo.resolve(), args.fixture.resolve(), args.output.resolve()
    require(not args.output.exists(), "G1 output already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"kind": args.kind, "status": "running", "command": sys.argv, "device": "cpu"}
    try:
        agent, saved, cfg = make_agent(args.repo, args.fixture, "cpu", args.output)
        from agent.finetune.stage3_modes import _variant_coefficients
        from agent.finetune.stage3_math import actor_gradient
        inputs = transfer(saved["batch"], "cpu")
        batch = {"obs_k": inputs["obs_k"], "chains_k": inputs["chains_k"],
                 "raw_rewards": saved["raw_reward_trajs"], "firsts": saved["firsts_trajs"],
                 "scaled_rewards": saved["scaled_reward_trajs"], "values": saved["values_trajs"],
                 "gae_adv": saved["advantages_trajs"], "returns": saved["returns_trajs"],
                 "oldlogprobs": inputs["logprobs_k"], "terminated": saved["terminated_trajs"]}
        coefficients = _variant_coefficients(agent, batch)
        restore_rng(saved["rng"])
        def state():
            return {"rng": rng_state(), "model": freeze(agent.model.state_dict()),
                    "actor_optimizer": freeze(agent.actor_optimizer.state_dict()),
                    "critic_optimizer": freeze(agent.critic_optimizer.state_dict()),
                    "gradients": freeze([p.grad for p in agent.model.parameters()]),
                    "batch": freeze(batch)}
        before = state()
        if args.kind == "reference":
            vectors = np.zeros((5, sum(p.numel() for p in agent.model.actor_ft.parameters())), dtype=np.float64)
            for env in range(agent.n_envs):
                for v, (advantages, reduction, discount, normalize) in enumerate(coefficients):
                    gradient = actor_gradient(agent, batch, advantages, reduce=reduction,
                        gamma_denoising=discount, normalize=normalize, env_indices=[env])
                    vectors[v] -= gradient.double().cpu().numpy() / agent.n_envs
                print("G1 reference env=%d/%d" % (env + 1, agent.n_envs), flush=True)
        elif args.kind == "edited":
            from agent.finetune.grad_mean import batch_gradients
            vectors = batch_gradients(agent, batch)
        else:
            vectors = full_loss_gradient(agent, saved, batch, coefficients[3][0])[None]
        require(np.isfinite(vectors).all(), "nonfinite saved-batch gradients")
        report["inert"] = compare(before, state(), True)
        require(report["inert"]["passed"], "G1 measurements mutated batch/model/optimizer/gradients/RNG")
        np.save(args.output / "gradients.npy", vectors)
        report.update(status="passed", shape=list(vectors.shape), ascent_sign="negative native loss gradient")
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (args.output / "worker.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
