#!/usr/bin/env python
"""Saved real-batch differential check, with no changes to scientific source.

The pristine run is interrupted immediately before its first update. Both exact
update bodies are extracted from their source ASTs, without modifying any numeric
statement, and replayed from the same saved batch, weights, optimizer and RNG.
CPU requires identical bytes. CUDA uses predeclared atol=1e-7, rtol=1e-5.
"""
import argparse
import ast
import copy
import importlib
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import types

from source_git import source_git_command

STUDY = Path(__file__).resolve().parents[1]
PIN = "cc7234ad7ff39a8f32de3af903606723a16f0648"
EXPECTED = "ab46b150fa34b5a5b457cd4062cd4c5ad830d964"
MODEL = "model/diffusion/diffusion_ppo.py"
AGENT = "agent/finetune/train_ppo_diffusion_agent.py"
FIELDS = ("obs_k", "chains_k", "returns_k", "values_k", "advantages_k", "logprobs_k")
ATTRS = ("itr", "n_steps", "n_envs", "batch_size", "update_epochs", "use_bc_loss",
         "reward_horizon", "ent_coef", "vf_coef", "bc_loss_coeff", "learn_eta",
         "n_critic_warmup_itr", "max_grad_norm", "target_kl", "seed")
LOSS_FIELDS = ("loss", "pg_loss", "entropy_loss", "v_loss", "clipfrac", "approx_kl",
               "ratio", "bc_loss", "eta")
ATOL, RTOL = 1e-7, 1e-5


def require(value, message):
    if not value:
        raise AssertionError(message)


def freeze(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(freeze(item) for item in value)
    return copy.deepcopy(value)


def transfer(value, device):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {key: transfer(item, device) for key, item in value.items()}
    return copy.deepcopy(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def bits_equal(a, b):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return False
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def compare(reference, edited, bitwise, name="value"):
    """Relative error is |new-old|/max(|old|,float64 tiny), no hidden floor."""
    result = {"passed": True, "bitwise_equal": True, "max_abs": 0.0, "max_rel": 0.0,
              "elements": 0, "tensors": 0, "first_failure": None}

    def fail(label):
        result["passed"] = False
        if result["first_failure"] is None:
            result["first_failure"] = label

    def walk(a, b, label):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape:
                fail(label + ": tensor dtype/shape")
                return
            result["tensors"] += 1
            result["elements"] += a.numel()
            exact = bits_equal(a, b)
            result["bitwise_equal"] &= exact
            x, y = a.detach().cpu().double(), b.detach().cpu().double()
            if x.numel():
                finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                if not finite:
                    fail(label + ": nonfinite")
                    return
                delta = (y - x).abs()
                relative = delta / x.abs().clamp(min=torch.finfo(torch.float64).tiny)
                result["max_abs"] = max(result["max_abs"], delta.max().item())
                result["max_rel"] = max(result["max_rel"], relative.max().item())
                accepted = exact if bitwise or not a.is_floating_point() else bool(
                    (delta <= ATOL + RTOL * x.abs()).all())
                if not accepted:
                    fail(label + ": numerical difference")
        elif isinstance(a, np.ndarray):
            walk(torch.from_numpy(a.copy()), torch.from_numpy(b.copy()), label)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                fail(label + ": dictionary keys")
                return
            for key in a:
                walk(a[key], b[key], label + "." + str(key))
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                fail(label + ": sequence structure")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, label + "." + str(index))
        elif isinstance(a, (float, np.floating)):
            walk(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64), label)
        elif a != b:
            result["bitwise_equal"] = False
            fail(label + ": nonnumeric value")

    walk(reference, edited, name)
    if math.isinf(result["max_rel"]):
        result["max_rel"] = "inf"
    return result


def rng_comparison(before, after):
    states = {key: compare(before[key], after[key], True, key)["passed"] for key in before}
    return {"passed": all(states.values()), "states": states,
            "cuda_generator_count": len(before["torch_cuda"])}


def numeric_settings():
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def assignment_to(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def update_ast(source):
    parsed = ast.parse(source)
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef))
    run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    loop = next(node for node in run.body if isinstance(node, ast.While))
    update = next(node for node in loop.body if isinstance(node, ast.If) and
                  isinstance(node.test, ast.UnaryOp) and isinstance(node.test.operand, ast.Name)
                  and node.test.operand.id == "eval_mode")
    start = next(i for i, node in enumerate(update.body) if assignment_to(node, "total_steps"))
    return update.body[start:], update.body[start].lineno


def compile_update(source, module, label):
    body, start_line = update_ast(source)
    names = ("self",) + FIELDS
    function = ast.FunctionDef(name="exact_source_update", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=name) for name in names], vararg=None,
        kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=copy.deepcopy(body), decorator_list=[])
    function.body.append(ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()),
                                                  args=[], keywords=[])))
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(module.__dict__)
    exec(compile(program, "<exact-source-update-" + label + ">", "exec"), namespace)
    actor_lines, critic_lines, permutation_lines = set(), set(), set()
    for node in ast.walk(function):
        if assignment_to(node, "num_batch"):
            permutation_lines.add(node.lineno)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "step" and isinstance(func.value, ast.Attribute):
                if func.value.attr == "actor_optimizer":
                    actor_lines.add(node.lineno)
                if func.value.attr == "critic_optimizer":
                    critic_lines.add(node.lineno)
    require(len(actor_lines) == len(critic_lines) == len(permutation_lines) == 1,
            "could not identify exact update instrumentation boundaries")
    return namespace["exact_source_update"], {"start_line": start_line,
        "end_line": body[-1].end_lineno, "actor_lines": sorted(actor_lines),
        "critic_lines": sorted(critic_lines), "permutation_lines": sorted(permutation_lines)}


def load_reference(repo):
    sources = {path: subprocess.check_output(source_git_command(repo, "show", PIN + ":" + path),
                                            text=True) for path in (MODEL, AGENT)}
    modules = {}
    for path, suffix in ((MODEL, "model"), (AGENT, "agent")):
        module = types.ModuleType("fixed_batch_pristine_" + suffix)
        exec(compile(sources[path], "<pristine-" + path + ">", "exec"), module.__dict__)
        modules[path] = module
    return modules, sources


def instantiate_model(cfg, model_module, cls, device):
    from hydra.utils import instantiate
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.device = device
    original = model_module.PPODiffusion
    model_module.PPODiffusion = cls
    try:
        return instantiate(model_cfg)
    finally:
        model_module.PPODiffusion = original


class BatchCaptured(Exception):
    pass


def capture_batch(cfg, model_module, old_modules, old_sources, output):
    old_cls = old_modules[MODEL].PPODiffusion
    agent_cls = old_modules[AGENT].TrainPPODiffusionAgent
    original = model_module.PPODiffusion
    model_module.PPODiffusion = old_cls
    try:
        agent = agent_cls(cfg)
    finally:
        model_module.PPODiffusion = original
    capture_line = update_ast(old_sources[AGENT])[1]
    captured = {}

    def trace(frame, event, arg):
        if frame.f_code is not agent_cls.run.__code__:
            return None
        if event == "line" and frame.f_lineno == capture_line:
            require(agent.itr == 1, "expected first training iteration")
            captured.update({"batch": freeze({name: frame.f_locals[name] for name in FIELDS}),
                "model_state": freeze(agent.model.state_dict()),
                "actor_optimizer_state": freeze(agent.actor_optimizer.state_dict()),
                "critic_optimizer_state": freeze(agent.critic_optimizer.state_dict()),
                "rng": rng_state(), "attrs": {name: getattr(agent, name) for name in ATTRS},
                "module_training": {name: module.training for name, module in agent.model.named_modules()},
                "source_revision": PIN, "capture_source_line": capture_line,
                "captured_before_any_optimizer_step": True})
            torch.save(captured, output / "training_batch.pt")
            raise BatchCaptured()
        return trace

    try:
        sys.settrace(trace)
        agent.run()
        raise AssertionError("capture breakpoint was not reached")
    except BatchCaptured:
        pass
    finally:
        sys.settrace(None)
        agent.venv.close()
    require(captured["batch"]["obs_k"]["state"].shape == (20000, 1, 17), "not full observation batch")
    require(captured["batch"]["chains_k"].shape == (20000, 11, 4, 6), "not full chain batch")
    require(captured["batch"]["logprobs_k"].shape == (20000, 10, 4, 6), "not full old-logprob batch")
    del agent
    torch.cuda.empty_cache()
    return captured


def replay_one(cfg, model_module, cls, agent_cls, module, source, saved, device, output, label):
    model = instantiate_model(cfg, model_module, cls, device)
    model.load_state_dict(saved["model_state"])
    for name, submodule in model.named_modules():
        submodule.training = saved["module_training"][name]
    agent = agent_cls.__new__(agent_cls)
    for name, value in saved["attrs"].items():
        setattr(agent, name, value)
    agent.model, agent.device = model, device
    agent.actor_single_step = False
    agent.diagnostic_generator = torch.Generator(device="cpu").manual_seed(agent.seed)
    agent.actor_optimizer = torch.optim.AdamW(model.actor_ft.parameters(), lr=1e-4, weight_decay=0)
    agent.critic_optimizer = torch.optim.AdamW(model.critic.parameters(), lr=1e-3, weight_decay=0)
    agent.actor_optimizer.load_state_dict(copy.deepcopy(saved["actor_optimizer_state"]))
    agent.critic_optimizer.load_state_dict(copy.deepcopy(saved["critic_optimizer_state"]))
    batch = transfer(saved["batch"], device)
    function, boundaries = compile_update(source, module, label)
    evidence = {"loss_components": [], "actor_gradients": [], "critic_gradients": [],
                "permutations": [], "actor_steps": 0, "critic_steps": 0,
                "module_training_before": saved["module_training"], "boundaries": boundaries}

    def trace(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        if event == "line":
            if frame.f_lineno in boundaries["permutation_lines"]:
                evidence["permutations"].append(freeze(frame.f_locals["inds_k"]))
            elif frame.f_lineno in boundaries["actor_lines"]:
                evidence["loss_components"].append(freeze({name: frame.f_locals[name] for name in LOSS_FIELDS}))
                evidence["actor_gradients"].append(freeze({name: param.grad for name, param in model.actor_ft.named_parameters()}))
                evidence["critic_gradients"].append(freeze({name: param.grad for name, param in model.critic.named_parameters()}))
                require(all(value is not None for value in evidence["actor_gradients"][-1].values()),
                        "an actor parameter has no gradient at the observation point")
                require(all(value is not None for value in evidence["critic_gradients"][-1].values()),
                        "a critic parameter has no gradient at the observation point")
                evidence["actor_steps"] += 1
                print("FULL UPDATE %s %s minibatch %d/20" % (device, label, evidence["actor_steps"]), flush=True)
            elif frame.f_lineno in boundaries["critic_lines"]:
                evidence["critic_steps"] += 1
        return trace

    initial = {"batch": freeze(batch), "model_state": freeze(model.state_dict()),
               "actor_optimizer_state": freeze(agent.actor_optimizer.state_dict()),
               "critic_optimizer_state": freeze(agent.critic_optimizer.state_dict())}
    restore_rng(saved["rng"])
    initial["rng"] = rng_state()
    before = time.monotonic()
    try:
        sys.settrace(trace)
        function(agent, **batch)
    finally:
        sys.settrace(None)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    evidence["seconds"] = time.monotonic() - before
    evidence["actor_parameters"] = freeze(dict(model.actor_ft.named_parameters()))
    evidence["critic_parameters"] = freeze(dict(model.critic.named_parameters()))
    evidence["model_state"] = freeze(model.state_dict())
    evidence["actor_optimizer_state"] = freeze(agent.actor_optimizer.state_dict())
    evidence["critic_optimizer_state"] = freeze(agent.critic_optimizer.state_dict())
    evidence["rng_after_update"] = rng_state()
    evidence["saved_batch_unchanged"] = compare(saved["batch"], freeze(batch), True, "saved_batch")
    require(evidence["saved_batch_unchanged"]["passed"], "replay changed the saved batch inputs")
    require(evidence["actor_steps"] == evidence["critic_steps"] == 20,
            "full baseline phase did not execute exactly 20 actor/critic steps")
    require(len(evidence["permutations"]) == 5, "full phase did not execute 5 epochs")
    for indices in evidence["permutations"]:
        require(torch.equal(indices.sort().values, torch.arange(200000)), "incomplete permutation")
    torch.save(evidence, output / (label + "_update_evidence.pt"))
    return agent, batch, initial, evidence


def sampling_check(cfg, model_module, classes, saved, device, output, bitwise):
    shape = (40, 4, 6)
    generator = torch.Generator(device=device).manual_seed(91235)
    noises = [torch.randn(shape, device=device, generator=generator) for _ in range(21)]
    observations = {"state": saved["batch"]["obs_k"]["state"][:40].to(device)}
    samples, full_chains, counts = [], [], []
    for cls in classes:
        model = instantiate_model(cfg, model_module, cls, device)
        model.load_state_dict(saved["model_state"])
        for name, submodule in model.named_modules():
            submodule.training = saved["module_training"][name]
        original_randn, original_like = torch.randn, torch.randn_like
        consumed = []

        def take_noise(kind, tensor_shape, requested_device):
            index = len(consumed)
            require(index < 21, "sampling consumed too many fixed-noise tensors")
            require(tuple(tensor_shape) == shape, "sampling noise shape changed")
            require(torch.device(requested_device) == torch.device(device), "sampling noise device changed")
            require(kind == ("randn" if index == 0 else "randn_like"), "sampling noise order changed")
            consumed.append(kind)
            return noises[index].clone()

        def fixed_randn(size, **kwargs):
            return take_noise("randn", size, kwargs.get("device", "cpu"))

        def fixed_like(tensor, **kwargs):
            return take_noise("randn_like", tensor.shape, kwargs.get("device", tensor.device))

        # Observe x after every complete denoising transition, including any final clamp.
        forward = model.forward.__wrapped__
        forward_source = Path(importlib.import_module("model.diffusion.diffusion_vpg").__file__).read_text()
        forward_tree = ast.parse(forward_source)
        inner_line = min(node.lineno for node in ast.walk(forward_tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name) and node.test.id == "return_chain"
            and node.lineno > forward.__code__.co_firstlineno)
        chain = []

        def trace(frame, event, arg):
            if frame.f_code is not forward.__code__:
                return None
            if event == "line" and frame.f_lineno == inner_line:
                chain.append(freeze(frame.f_locals["x"]))
            return trace

        try:
            torch.randn, torch.randn_like = fixed_randn, fixed_like
            sys.settrace(trace)
            result = model(cond=observations, deterministic=False, return_chain=True)
        finally:
            sys.settrace(None)
            torch.randn, torch.randn_like = original_randn, original_like
        require(len(consumed) == 21 and len(chain) == 20, "not all 20 denoising transitions observed")
        samples.append({"chain": freeze(result.chains), "trajectories": freeze(result.trajectories)})
        full_chains.append(torch.stack(chain))
        counts.append(len(consumed))
        del model
    comparisons = {key: compare(samples[0][key], samples[1][key], bitwise, key)
                   for key in ("chain", "trajectories")}
    comparisons["all_20_transition_states"] = compare(full_chains[0], full_chains[1], bitwise)
    torch.save({"noise_tape": freeze(noises), "pristine": samples[0], "edited": samples[1],
                "full_20_transition_states": full_chains}, output / "sampling_evidence.pt")
    return {"passed": all(item["passed"] for item in comparisons.values()),
            "noise_calls": counts, "returned_chain_states": 11, "observed_transition_states": 20,
            "comparisons": comparisons}


def device_gate(cfg, model_module, agent_module, old_modules, old_sources, saved, device, output):
    output.mkdir()
    bitwise = device == "cpu"
    reference = replay_one(cfg, model_module, old_modules[MODEL].PPODiffusion,
        old_modules[AGENT].TrainPPODiffusionAgent, old_modules[AGENT], old_sources[AGENT],
        saved, device, output, "pristine")
    edited = replay_one(cfg, model_module, model_module.PPODiffusion,
        agent_module.TrainPPODiffusionAgent, agent_module, (STUDY / "dppo" / AGENT).read_text(),
        saved, device, output, "edited")
    old_agent, old_batch, old_initial, old = reference
    new_agent, new_batch, new_initial, new = edited
    initial_comparisons = {key: compare(old_initial[key], new_initial[key], True, key)
                           for key in old_initial}
    categories = ("loss_components", "actor_gradients", "critic_gradients", "actor_parameters",
                  "critic_parameters", "model_state", "actor_optimizer_state", "critic_optimizer_state",
                  "permutations", "rng_after_update")
    comparisons = {key: compare(old[key], new[key], bitwise or key in ("permutations", "rng_after_update"), key)
                   for key in categories}
    minibatches = [{"minibatch": index, "epoch": index // 4,
                   **{key: compare(old[key][index], new[key][index], bitwise, key)
                      for key in ("loss_components", "actor_gradients", "critic_gradients")}}
                  for index in range(20)]
    before = rng_state()
    diagnostics = new_agent._post_update_diagnostics(new_batch["obs_k"], new_batch["chains_k"], new_batch["logprobs_k"])
    isolation = rng_comparison(before, rng_state())
    isolation["diagnostics"] = diagnostics
    torch.save({"before": before, "after": rng_state()}, output / "diagnostic_rng_evidence.pt")
    update = {"passed": all(x["passed"] for x in comparisons.values()) and all(x["passed"] for x in initial_comparisons.values()),
              "epochs": 5, "minibatches": 20, "actor_steps": 20, "critic_steps": 20,
              "same_saved_batch": initial_comparisons["batch"]["passed"],
              "same_initial_model_state": initial_comparisons["model_state"]["passed"],
              "same_initial_optimizer_states": all(initial_comparisons[key]["passed"] for key in
                                                     ("actor_optimizer_state", "critic_optimizer_state")),
              "same_initial_rng_state": initial_comparisons["rng"]["passed"],
              "initial_comparisons": initial_comparisons, "comparisons": comparisons,
              "minibatch_comparisons": minibatches,
              "pristine_seconds": old["seconds"], "edited_seconds": new["seconds"]}
    del reference, edited, old_agent, new_agent, old_batch, new_batch, old_initial, new_initial, old, new
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    sampling = sampling_check(cfg, model_module, (old_modules[MODEL].PPODiffusion, model_module.PPODiffusion),
                              saved, device, output, bitwise)
    return {"passed": update["passed"] and sampling["passed"] and isolation["passed"],
            "required_bitwise": bitwise, "tolerance": {"atol": 0 if bitwise else ATOL, "rtol": 0 if bitwise else RTOL},
            "full_update": update, "sampling": sampling, "rng_isolation": isolation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "runs/verification_full_update")
    parser.add_argument("--timeout-seconds", type=int, default=1710)
    args = parser.parse_args()
    output = args.output.resolve()
    require(STUDY in output.parents and not output.exists(), "output must be new and inside study")
    require(0 < args.timeout_seconds <= 1710, "verification remaining-budget cap exceeded")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "passed": False, "command": sys.argv,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "gpu_tolerance_predeclared": {"atol": ATOL, "rtol": RTOL},
              "relative_error_definition": "abs(edited-reference)/max(abs(reference),float64_tiny)",
              "timeout_seconds": args.timeout_seconds}

    def save():
        with open(output / "verification.json", "w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)

    def timeout(signum, frame):
        raise TimeoutError("revised fixed-batch gate reached its remaining approved cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    save()
    try:
        repo = STUDY / "dppo"
        command = lambda *parts: subprocess.check_output(source_git_command(repo, *parts), text=True).strip()
        require(command("rev-parse", "HEAD") == EXPECTED and not command("status", "--porcelain"),
                "source commit changed or worktree dirty")
        require(set(command("diff", "--name-only", PIN).splitlines()) == {MODEL, AGENT}, "unauthorized source files changed")
        report["source_commit"], report["reference_commit"] = EXPECTED, PIN
        (output / "source.diff").write_text(command("diff", PIN) + "\n")
        sys.path.insert(0, str(repo))
        global torch, np
        import torch
        import numpy as np
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        require(torch.cuda.is_available(), "full gate requires allocated GPU")
        report["numeric_settings_before"] = numeric_settings()
        report["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
        require(not report["deterministic_algorithms_enabled"], "deterministic algorithms unexpectedly enabled")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        report["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True)
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(repo / "cfg/gym/finetune/halfcheetah-v2")):
            cfg = compose(config_name="ft_ppo_diffusion_mlp", overrides=["seed=0", "wandb=null",
                "train.n_train_itr=140", "train.save_model_freq=35", "logdir=" + str(output / "capture/native")])
        OmegaConf.resolve(cfg)
        OmegaConf.save(cfg, output / "capture_config.yaml", resolve=True)
        model_module = importlib.import_module("model.diffusion.diffusion_ppo")
        agent_module = importlib.import_module("agent.finetune.train_ppo_diffusion_agent")
        old_modules, old_sources = load_reference(repo)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        report["active_phase"] = "capture_pristine_real_batch"
        save()
        print("FULL GATE START capture pristine real batch", flush=True)
        capture_start = time.monotonic()
        capture_batch(cfg, model_module, old_modules, old_sources, output)
        saved = torch.load(output / "training_batch.pt", map_location="cpu", weights_only=False)
        report["capture"] = {"seconds": time.monotonic() - capture_start,
            "saved_batch_path": str(output / "training_batch.pt"), "training_iteration": 1,
            "training_environment_transitions": 80000, "evaluation_environment_transitions": 80000,
            "env_step_denoising_pairs": 200000,
            "optimizer_steps_performed": 0, "reference_source": PIN,
            "actor_optimizer_state_entries": len(saved["actor_optimizer_state"]["state"]),
            "critic_optimizer_state_entries": len(saved["critic_optimizer_state"]["state"]),
            "batch_shapes": {key: list(value["state"].shape if key == "obs_k" else value.shape)
                             for key, value in saved["batch"].items()}}
        save()
        for key, device in (("cpu", "cpu"), ("gpu", "cuda:0")):
            report["active_phase"] = key + "_full_update"
            save()
            print("FULL GATE START " + key, flush=True)
            report[key] = device_gate(cfg, model_module, agent_module, old_modules, old_sources,
                                      saved, device, output / key)
            save()
            require(report[key]["passed"], key + " differential gate failed; see numerical comparisons")
            print("FULL GATE PASS " + key, flush=True)
        report["numeric_settings_after"] = numeric_settings()
        report["deterministic_settings_unchanged"] = report["numeric_settings_before"] == report["numeric_settings_after"]
        require(report["deterministic_settings_unchanged"], "numeric settings changed")
        require(command("rev-parse", "HEAD") == EXPECTED and not command("status", "--porcelain"), "scientific source changed")
        report["status"], report["passed"] = "passed", True
        report.pop("active_phase", None)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
    print("FULL GATE COMPLETE passed", flush=True)


if __name__ == "__main__":
    main()
