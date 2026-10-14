"""Opt-in observational state capture for the approved DPPO step-noise study.

No optimizer or sampler behavior is changed by the capture hooks.  Native update
replay compiles the existing update-loop AST instead of maintaining a second
implementation.  All saved data are local, trusted experiment artifacts.
"""
import ast
import copy
import csv
import json
import logging
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch


log = logging.getLogger(__name__)
CAPTURE_ITERATIONS = (0, 35, 70, 105, 139)


def _copy(value):
    """Deep CPU snapshot; preserve integer optimizer keys and tensor metadata."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        result = copy.copy(value)
        result.clear()
        result.update((key, _copy(child)) for key, child in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_copy(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy(child) for child in value)
    return copy.deepcopy(value)


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": []}
    # Do not initialize CUDA merely to observe an otherwise CPU-only process.
    if torch.cuda.is_initialized():
        result["torch_cuda"] = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved and current CUDA generator counts differ")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def scaler_divisor(agent):
    scaler = agent.running_reward_scaler
    value = np.asarray(np.sqrt(scaler.ret_rms.var + scaler.epsilon))
    if value.size != 1:
        raise ValueError("DPPO step-noise study requires a scalar reward divisor")
    return float(value)


def _scaler_state(agent):
    scaler = agent.running_reward_scaler
    return {"ret_rms": _copy(vars(scaler.ret_rms)),
            "fields": _copy({key: value for key, value in vars(scaler).items() if key != "ret_rms"})}


def snapshot(agent):
    """Full training state with no RNG draws or modification of live tensors."""
    return {"format_version": 1, "itr": int(agent.itr),
            "model": _copy(agent.model.state_dict()),
            "actor_optimizer": _copy(agent.actor_optimizer.state_dict()),
            "critic_optimizer": _copy(agent.critic_optimizer.state_dict()),
            "actor_scheduler": _copy(agent.actor_lr_scheduler.state_dict()),
            "critic_scheduler": _copy(agent.critic_lr_scheduler.state_dict()),
            "reward_scaler": _scaler_state(agent), "rng": rng_state(),
            "diagnostic_rng": agent.diagnostic_generator.get_state().clone(),
            "model_counters": {"ft_denoising_steps_cnt": int(agent.model.ft_denoising_steps_cnt),
                               "ft_denoising_steps": int(agent.model.ft_denoising_steps)},
            "model_training": bool(agent.model.training)}


def restore(agent, state, restore_rng=False):
    """Restore parameters, moments and scalers; fresh sampling RNG is the default."""
    agent.model.load_state_dict(state["model"], strict=True)
    agent.actor_optimizer.load_state_dict(copy.deepcopy(state["actor_optimizer"]))
    agent.critic_optimizer.load_state_dict(copy.deepcopy(state["critic_optimizer"]))
    agent.actor_lr_scheduler.load_state_dict(copy.deepcopy(state["actor_scheduler"]))
    agent.critic_lr_scheduler.load_state_dict(copy.deepcopy(state["critic_scheduler"]))
    agent.running_reward_scaler.ret_rms.__dict__.clear()
    agent.running_reward_scaler.ret_rms.__dict__.update(_copy(state["reward_scaler"]["ret_rms"]))
    for key in tuple(vars(agent.running_reward_scaler)):
        if key != "ret_rms":
            delattr(agent.running_reward_scaler, key)
    agent.running_reward_scaler.__dict__.update(_copy(state["reward_scaler"]["fields"]))
    agent.itr = int(state["itr"])
    for key, value in state["model_counters"].items():
        setattr(agent.model, key, copy.deepcopy(value))
    agent.model.train(state["model_training"])
    agent.diagnostic_generator.set_state(state["diagnostic_rng"].cpu())
    if restore_rng:
        globals()["restore_rng"](state["rng"])


def _directory(agent):
    return Path(agent.logdir) / "ppo_steps"


def _save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite DPPO capture: " + str(path))
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
    temporary.rename(path)


def validate_capture(agent):
    if not isinstance(agent.ppo_steps_capture, bool):
        raise TypeError("ppo_steps_capture must be bool")
    if (agent.actor_single_step or agent.model.actor_loss != "ratio" or
            not agent.model.clamp_logprob or agent.model.logprob_reduce != "mean" or
            not agent.model.norm_adv or agent.model.randn_clip_value != 3 or
            agent.max_grad_norm is not None or agent.learn_eta or agent.use_bc_loss or
            agent.stage3_active or agent.stage3_mode != "train"):
        raise ValueError("Capture requires the unchanged DPPO default training path")
    if (agent.n_steps, agent.n_envs, agent.model.ft_denoising_steps, agent.batch_size,
            agent.update_epochs, agent.n_train_itr, agent.save_model_freq) != (500, 40, 10, 50000, 5, 140, 35):
        raise ValueError("Capture geometry or run length differs from the approved rerun")
    directory = _directory(agent)
    directory.mkdir(parents=True, exist_ok=False)
    from omegaconf import OmegaConf
    settings = OmegaConf.to_container(agent.cfg, resolve=True)
    with (directory / "config.json").open("x") as stream:
        json.dump(settings, stream, indent=2, allow_nan=False)
        stream.write("\n")
    agent._ppo_steps_capture_scales = []
    log.info("ppo_steps_capture=True full_state_iterations=%s max_grad_norm=%r reset_at_iteration=%r",
             CAPTURE_ITERATIONS, agent.max_grad_norm, agent.reset_at_iteration)


def capture_iteration_start(agent):
    agent._ppo_steps_capture_c_start = scaler_divisor(agent)
    log.info("DPPO_REWARD_SCALE itr=%d phase=start c=%.17g", agent.itr, agent._ppo_steps_capture_c_start)


def capture_raw_buffer(agent, **buffer):
    if agent.itr == 36:
        agent._ppo_steps_capture_raw_buffer = _copy(buffer)


def capture_scaling(agent):
    log.info("DPPO_REWARD_SCALE itr=%d phase=after_scaler_update c=%.17g", agent.itr, scaler_divisor(agent))


def capture_update_input(agent, **batch):
    if agent.itr != 36:
        return
    payload = {"format_version": 1, "itr": int(agent.itr), "batch": _copy(batch),
               "raw_buffer": agent._ppo_steps_capture_raw_buffer,
               "pre_loop_rng": rng_state(), "pre_update_state": snapshot(agent),
               "source_full_state": "full_state_35.pt"}
    _save(_directory(agent) / "replay_itr36_input.pt", payload)
    del agent._ppo_steps_capture_raw_buffer


def capture_update_output(agent, actor_optimizer_steps, critic_optimizer_steps):
    if agent.itr != 36:
        return
    payload = {"format_version": 1, "itr": int(agent.itr), "state": snapshot(agent),
               "actor_optimizer_steps": int(actor_optimizer_steps),
               "critic_optimizer_steps": int(critic_optimizer_steps)}
    _save(_directory(agent) / "replay_itr36_expected.pt", payload)
    log.info("DPPO_REPLAY_INPUT_READY itr=36 actor_steps=%d critic_steps=%d",
             actor_optimizer_steps, critic_optimizer_steps)


def capture_checkpoint(agent, **counters):
    if agent.itr not in CAPTURE_ITERATIONS:
        return
    payload = snapshot(agent)
    payload["counters"] = _copy(counters)
    path = _directory(agent) / ("full_state_%d.pt" % agent.itr)
    _save(path, payload)
    log.info("DPPO_FULL_STATE_READY itr=%d path=%s", agent.itr, path)


def capture_iteration_metrics(agent, record, eval_mode):
    after = scaler_divisor(agent)
    record["reward_scaler_c_start"] = agent._ppo_steps_capture_c_start
    record["reward_scaler_c_after"] = after
    row = {"itr": int(agent.itr), "eval": bool(eval_mode),
           "c_start": agent._ppo_steps_capture_c_start, "c_after": after}
    agent._ppo_steps_capture_scales.append(row)
    path = _directory(agent) / "reward_scale.csv"
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(agent._ppo_steps_capture_scales)
    temporary.replace(path)


def capture_iteration_finished(agent):
    """Publish monitoring progress only after the native iteration checks pass."""
    completed = int(agent.itr) + 1
    record = {"completed": completed, "total": int(agent.n_train_itr),
              "status": "complete" if completed == agent.n_train_itr else "running"}
    path = Path(agent.logdir) / "progress.json"
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native_update_source():
    """Return the untouched native loop AST, from setup through its actor branch."""
    path = Path(__file__).with_name("train_ppo_diffusion_agent.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainPPODiffusionAgent")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    loop = next(node for node in method.body if isinstance(node, ast.While))
    training = next(node for node in loop.body if isinstance(node, ast.If) and
                    isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) and
                    isinstance(node.test.operand, ast.Name) and node.test.operand.id == "eval_mode")
    start = next(index for index, node in enumerate(training.body) if isinstance(node, ast.Assign) and
                 any(isinstance(target, ast.Name) and target.id == "total_steps" for target in node.targets))
    stop = next(index for index, node in enumerate(training.body[start:], start) if
                isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) and
                isinstance(node.test.values[0], ast.Attribute) and node.test.values[0].attr == "actor_single_step")
    return ast.Module(body=copy.deepcopy(training.body[start:stop + 1]), type_ignores=[]), path


def native_update(agent, batch):
    """Execute exactly the current default native actor+critic loop on a saved batch."""
    if agent.actor_single_step or agent.stage3_active:
        raise ValueError("Exact replay is restricted to the DPPO default loop")
    required = ("obs_k", "chains_k", "returns_k", "values_k", "advantages_k", "logprobs_k")
    def device_copy(value):
        if torch.is_tensor(value):
            return value.to(agent.device).clone()
        if isinstance(value, dict):
            return {key: device_copy(child) for key, child in value.items()}
        return copy.deepcopy(value)
    local = {key: device_copy(batch[key]) for key in required}
    local["self"] = agent
    tree, path = native_update_source()
    code = compile(ast.fix_missing_locations(tree), str(path), "exec")
    exec(code, {"torch": torch, "np": np, "math": math, "log": log,
                "perf_counter": perf_counter}, local)
    return {"actor_optimizer_steps": local["actor_optimizer_steps"],
            "critic_optimizer_steps": local["critic_optimizer_steps"],
            "clipfracs": local["clipfracs"], "approx_kl": local["approx_kl"],
            "ratio": local["ratio"], "last_update_epoch": local["update_epoch"],
            "last_batch": local["batch"], "target_kl_fired": bool(local["flag_break"])}
