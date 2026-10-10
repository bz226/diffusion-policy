#!/usr/bin/env python
"""Capture full live fixtures at the immutable Stage-2 tip, before any update.

Only tracing and copies are added; the reference scientific source is unchanged.
Both DPPO-default and NC4 fixtures include their own on-policy chains/log-probs.
"""
import argparse
import ast
import copy
import importlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
REFERENCE = STUDY / "reference_stage2"
PIN = "ab46b150fa34b5a5b457cd4062cd4c5ad830d964"
FIELDS = ("obs_k", "chains_k", "returns_k", "values_k", "advantages_k", "logprobs_k")
ATTRS = ("itr", "n_steps", "n_envs", "batch_size", "update_epochs", "use_bc_loss",
         "reward_horizon", "ent_coef", "vf_coef", "bc_loss_coeff", "learn_eta",
         "n_critic_warmup_itr", "max_grad_norm", "target_kl", "seed", "gamma",
         "gae_lambda", "reward_scale_const", "reward_scale_running", "actor_single_step")


def freeze(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(freeze(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]}


def git_command(*args):
    metadata = REFERENCE / ".git"
    if metadata.is_file():
        text = metadata.read_text().strip()
        if not text.startswith("gitdir: "):
            raise RuntimeError("reference worktree has an invalid Git metadata pointer")
        metadata = (REFERENCE / text[len("gitdir: "):]).resolve()
    if not metadata.is_dir():
        raise RuntimeError("reference Git metadata unavailable; parent discovery forbidden")
    return subprocess.check_output(["git", "--git-dir=" + str(metadata),
                                   "--work-tree=" + str(REFERENCE), *args], text=True).strip()


def verify_reference():
    if git_command("rev-parse", "HEAD") != PIN or git_command("status", "--porcelain"):
        raise RuntimeError("Stage-2 reference is not clean at the required tip")


class Captured(Exception):
    pass


def capture(name, out):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    cls = importlib.import_module("agent.finetune.train_ppo_diffusion_agent").TrainPPODiffusionAgent
    overrides = ["seed=0", "wandb=null", "train.n_train_itr=140",
                 "train.save_model_freq=35", "logdir=" + str(out / "native")]
    if name == "nc4":
        overrides += ["model.clip_ploss_coef=1e6", "model.clip_ploss_coef_base=1e6",
                      "train.target_kl=null", "+model.clamp_logprob=false",
                      "model.randn_clip_value=100", "+train.actor_single_step=true",
                      "train.actor_lr=1e-4", "train.actor_lr_scheduler.min_lr=1e-4"]
    with initialize_config_dir(version_base=None,
                               config_dir=str(REFERENCE / "cfg/gym/finetune/halfcheetah-v2")):
        cfg = compose(config_name="ft_ppo_diffusion_mlp", overrides=overrides)
    OmegaConf.resolve(cfg)
    out.mkdir(parents=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    source = (REFERENCE / "agent/finetune/train_ppo_diffusion_agent.py").read_text()
    lines = source.splitlines()
    update_line = next(i + 1 for i, line in enumerate(lines)
                       if line.strip() == "total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps"
                       and i > next(j for j, text in enumerate(lines) if "def run(self):" in text))
    scale_line = next(i + 1 for i, line in enumerate(lines) if line.strip() == "if self.reward_scale_running:")
    transition_line = next(i + 1 for i, line in enumerate(lines)
                           if line.strip() == "done_venv = terminated_venv | truncated_venv")
    agent = cls(cfg)
    initial = {"model_state": freeze(agent.model.state_dict()),
               "actor_optimizer_state": freeze(agent.actor_optimizer.state_dict()),
               "critic_optimizer_state": freeze(agent.critic_optimizer.state_dict()),
               "actor_scheduler_state": freeze(agent.actor_lr_scheduler.state_dict()),
               "critic_scheduler_state": freeze(agent.critic_lr_scheduler.state_dict()),
               "reward_scaler_state": freeze(vars(agent.running_reward_scaler)),
               "rng": rng_state()}
    truncated = np.zeros((agent.n_steps, agent.n_envs), dtype=np.bool_)
    actions = np.zeros((agent.n_steps, agent.n_envs, agent.act_steps, agent.action_dim))
    seen = set()
    saved = {}
    before_scaling = {}

    def tracer(frame, event, arg):
        if frame.f_code is not cls.run.__code__:
            return None
        if event != "line" or agent.itr != 1:
            return tracer
        local = frame.f_locals
        if frame.f_lineno == transition_line:
            step = int(local["step"])
            if step in seen:
                raise RuntimeError("capture saw duplicate training transition")
            seen.add(step)
            truncated[step] = local["truncated_venv"]
            actions[step] = local["action_venv"]
        elif frame.f_lineno == scale_line:
            before_scaling.update(raw_reward_trajs=freeze(local["reward_trajs"]),
                                  reward_scaler_before=freeze(vars(agent.running_reward_scaler)))
        elif frame.f_lineno == update_line:
            if len(seen) != agent.n_steps or "raw_reward_trajs" not in before_scaling:
                raise RuntimeError("capture omitted rollout transitions or raw rewards")
            saved.update({"batch": freeze({key: local[key] for key in FIELDS}),
                          "raw_reward_trajs": before_scaling["raw_reward_trajs"],
                          "scaled_reward_trajs": freeze(local["reward_trajs"]),
                          "firsts_trajs": freeze(local["firsts_trajs"]),
                          "terminated_trajs": freeze(local["terminated_trajs"]),
                          "truncated_trajs": truncated.copy(), "action_trajs": actions.copy(),
                          "values_trajs": freeze(local["values_trajs"]),
                          "advantages_trajs": freeze(local["advantages_trajs"]),
                          "returns_trajs": freeze(local["returns_trajs"]),
                          "final_observation": freeze(local["obs_venv"]),
                          "model_state": freeze(agent.model.state_dict()),
                          "actor_optimizer_state": freeze(agent.actor_optimizer.state_dict()),
                          "critic_optimizer_state": freeze(agent.critic_optimizer.state_dict()),
                          "actor_scheduler_state": freeze(agent.actor_lr_scheduler.state_dict()),
                          "critic_scheduler_state": freeze(agent.critic_lr_scheduler.state_dict()),
                          "reward_scaler_before": before_scaling["reward_scaler_before"],
                          "reward_scaler_after": freeze(vars(agent.running_reward_scaler)),
                          "rng": rng_state(),
                          "diagnostic_generator_state": agent.diagnostic_generator.get_state().clone(),
                          "attrs": {key: getattr(agent, key) for key in ATTRS},
                          "module_training": {key: module.training for key, module in agent.model.named_modules()},
                          "config": OmegaConf.to_container(cfg, resolve=True),
                          "condition": name, "source_revision": PIN,
                          "capture_source_line": update_line,
                          "captured_before_any_optimizer_step": True,
                          "initial_before_eval": initial,
                          "rollout_metrics": {key: freeze(local[key]) for key in
                                              ("avg_episode_reward", "avg_best_reward", "success_rate",
                                               "num_episode_finished", "episode_reward", "cnt_train_step")}})
            torch.save(saved, out / "training_batch.pt")
            raise Captured()
        return tracer

    try:
        sys.settrace(tracer)
        agent.run()
        raise RuntimeError("capture breakpoint not reached")
    except Captured:
        pass
    finally:
        sys.settrace(None)
        agent.venv.close()
    boundaries = [np.flatnonzero(saved["firsts_trajs"][:, e]).tolist() for e in range(agent.n_envs)]
    report = {"condition": name, "status": "captured", "saved_batch": str(out / "training_batch.pt"),
              "source_revision": PIN, "source_clean": True, "overrides": overrides,
              "training_iteration": 1, "optimizer_steps": 0,
              "evaluation_env_transitions": 80000, "training_env_transitions": 80000,
              "pair_count": 200000, "episode_boundaries_per_env": boundaries,
              "terminated_count": int(np.count_nonzero(saved["terminated_trajs"])),
              "truncated_count": int(np.count_nonzero(truncated)),
              "actor_optimizer_state_entries": len(saved["actor_optimizer_state"]["state"]),
              "critic_optimizer_state_entries": len(saved["critic_optimizer_state"]["state"]),
              "batch_shapes": {key: list(value["state"].shape if key == "obs_k" else value.shape)
                               for key, value in saved["batch"].items()},
              "fields": list(saved)}
    (out / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    del agent
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STUDY / "runs/gate_capture")
    parser.add_argument("--timeout-seconds", type=int, default=840)
    args = parser.parse_args()
    output = args.output.resolve()
    if STUDY not in output.parents or output.exists():
        raise ValueError("capture output must be a new directory inside Stage3")
    if not 0 < args.timeout_seconds <= 840:
        raise ValueError("capture execution cap must be <=840s within a15min allocation")
    output.mkdir(parents=True)
    start = time.monotonic()
    report = {"status": "running", "command": sys.argv,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "reference_revision": PIN, "reference_worktree": str(REFERENCE),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "captures": []}
    def timeout(signum, frame):
        raise TimeoutError("capture reached its execution cap")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        verify_reference()
        if not torch.cuda.is_available():
            raise RuntimeError("live capture requires the allocated GPU")
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("deterministic algorithms must remain disabled")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["torch_threads"] = torch.get_num_threads()
        sys.path.insert(0, str(REFERENCE))
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        for name in ("default", "nc4"):
            print("CAPTURE START " + name, flush=True)
            report["captures"].append(capture(name, output / name))
            print("CAPTURE COMPLETE " + name, flush=True)
        verify_reference()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - start
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (output / "capture_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
