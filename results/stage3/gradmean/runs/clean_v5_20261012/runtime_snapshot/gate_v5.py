#!/usr/bin/env python
"""CPU frozen-prepare replay and W=0 scientific gates; no fresh rollouts.

F3 compares old/new default preparation on identical saved CPU input batches,
initial model, optimizers, schedules, scaler and RNG, with identical provenance.
This is not a comparison against a historical GPU-produced archive.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from gate_common import compare, freeze, numeric_settings, require, restore_rng, rng_state

SUBTREE = "results/stage3_gradmean/dppo"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def saved_state(agent, gm):
    return {"model": freeze(agent.model.state_dict()),
            "actor_optimizer": freeze(agent.actor_optimizer.state_dict()),
            "critic_optimizer": freeze(agent.critic_optimizer.state_dict()),
            "actor_scheduler": copy.deepcopy(agent.actor_lr_scheduler.state_dict()),
            "critic_scheduler": copy.deepcopy(agent.critic_lr_scheduler.state_dict()),
            "scaler": gm.scaler_state(agent), "rng": rng_state(),
            "gradients": freeze([p.grad for p in agent.model.parameters()])}


def restore_state(agent, gm, state):
    agent.model.load_state_dict(state["model"])
    agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
    agent.actor_lr_scheduler.load_state_dict(state["actor_scheduler"])
    agent.critic_lr_scheduler.load_state_dict(state["critic_scheduler"])
    gm._restore_scaler(agent, state["scaler"])
    for parameter, gradient in zip(agent.model.parameters(), state["gradients"]):
        parameter.grad = None if gradient is None else gradient.clone()
    restore_rng(state["rng"])


def build_agent(args):
    sys.path.insert(0, str(args.source))
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from agent.finetune import grad_mean as gm
    config = OmegaConf.create(json.loads(args.prepare.read_text())["config"])
    overrides = {"device": "cpu", "model.device": "cpu", "seed": args.seed,
                 "base_policy_path": str(args.checkpoint), "model.network_path": str(args.checkpoint),
                 "logdir": str(args.output), "wandb": None, "train.mode": "grad_mean",
                 "train.gm_step": "prepare", "train.gm_point": "D1", "train.gm_warmup_batches": 5,
                 "train.gm_batches": 50, "train.gm_score_sample": 0, "train.gm_include_critic": True,
                 "train.gm_parent_commit": args.base_commit, "train.gm_frozen_path": None,
                 "train.gm_frozen_sha256": None, "model.clamp_logprob": False,
                 "model.randn_clip_value": 100, "model.actor_loss": "score"}
    if args.worker == "features":
        overrides.update({"train.gm_warmup": 0, "train.gm_scaler_c": 1.000000005,
                          "train.gm_v2_raw": True, "train.gm_normalization": "full"})
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, force_add=True)
    agent = get_class(config._target_)(config)
    # The saved-batch check never steps/reset the simulator.
    agent.venv.close()
    settings = gm.settings(agent)
    gm._validate(agent, settings)
    return agent, gm, settings


def replay_collector(agent, fixture):
    """Replay observations/rewards/chains; recompute native targets and values."""
    def collect(deterministic=False, n_steps=None, update_scaler=True):
        require(not deterministic and n_steps is None, "unexpected replay sampler request")
        agent.model.train()
        batch = freeze(fixture)
        rewards, firsts = batch["raw_rewards"], batch["firsts"]
        steps = rewards.shape[0]
        with torch.no_grad():
            chunks = []
            for start in range(0, steps * agent.n_envs, agent.logprob_batch_size):
                cond = {key: value[start:start + agent.logprob_batch_size]
                        for key, value in batch["obs_k"].items()}
                chunks.append(agent.model.critic(cond).cpu().numpy().reshape(-1))
            values = np.concatenate(chunks).reshape(steps, agent.n_envs).astype(np.float64)
            scaled = (agent.running_reward_scaler(rewards.T, firsts[:-1].T).T if update_scaler
                      else agent.running_reward_scaler.transform(rewards.T).T)
            advantages, last = np.zeros_like(rewards), 0
            require(np.all(batch["terminated"][-1] == 1), "replay ends must be terminal")
            for t in reversed(range(steps)):
                nxt = np.zeros((1, agent.n_envs)) if t == steps - 1 else values[t + 1]
                nonterminal = 1 - batch["terminated"][t]
                delta = scaled[t] * agent.reward_scale_const + agent.gamma * nxt * nonterminal - values[t]
                advantages[t] = last = delta + agent.gamma * agent.gae_lambda * nonterminal * last
            batch.update(values=values, scaled_rewards=scaled, gae_adv=advantages,
                         returns=advantages + values, timing={"rollout_seconds": 0.0})
        return batch
    return collect


def native_quarter_gradient(agent, batch, advantages, permutation):
    parameters = list(agent.model.actor_ft.parameters())
    gradients = [torch.zeros_like(parameter) for parameter in parameters]
    adv = torch.as_tensor(advantages, dtype=torch.float32).reshape(-1)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32).reshape(-1)
    values = torch.as_tensor(batch["values"], dtype=torch.float32).reshape(-1)
    denoising = agent.model.ft_denoising_steps
    total = len(permutation)
    for start in range(0, total, agent.batch_size):
        pairs = permutation[start:start + agent.batch_size]
        decisions, indices = torch.div(pairs, denoising, rounding_mode="floor"), pairs % denoising
        losses = agent.model.loss({key: value[decisions] for key, value in batch["obs_k"].items()},
            batch["chains_k"][decisions, indices], batch["chains_k"][decisions, indices + 1],
            indices, returns[decisions], values[decisions], adv[decisions],
            batch["oldlogprobs"][decisions, indices], use_bc_loss=False, reward_horizon=agent.reward_horizon)
        chunk = torch.autograd.grad(losses[0] * (len(pairs) / total), parameters, allow_unused=True)
        for target, value in zip(gradients, chunk):
            if value is not None:
                target.add_(value.detach())
    return torch.cat([value.reshape(-1) for value in gradients])


def feature_checks(agent, gm, settings, fixture, args):
    from agent.finetune.grad_mean_v5 import batch_gradients_v5, normalized_gradient, pair_permutation, set_scaler_divisor
    from agent.finetune.stage3_math import actor_gradient
    from agent.finetune.stage3_modes import _variant_coefficients
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = {key[len("critic."):]: value for key, value in checkpoint["model"].items() if key.startswith("critic.")}
    require(expected, "checkpoint has no critic")
    initial = saved_state(agent, gm)
    loaded = compare(expected, agent.model.critic.state_dict(), True)
    require(loaded["passed"], "F1 checkpoint critic load mismatch")
    agent._stage3_collect = replay_collector(agent, fixture)
    meta = gm.prepare(agent, settings)
    after = saved_state(agent, gm)
    for key in ("model", "actor_optimizer", "critic_optimizer", "actor_scheduler", "critic_scheduler", "gradients", "rng"):
        require(compare(initial[key], after[key], True)["passed"], "W0 changed " + key)
    stored = torch.load(meta["frozen_path"], map_location="cpu", weights_only=True)
    exact = compare(expected, stored["critic"], True)
    require(exact["passed"] and meta["completed_warmup_batches"] == 0 and meta["critic_optimizer_steps"] == 0, "F1 failure")
    divisor = float(np.sqrt(np.asarray(agent.running_reward_scaler.ret_rms.var) + agent.running_reward_scaler.epsilon))
    require(divisor == settings["gm_scaler_c"] == meta["effective_scaler_c"], "F2 divisor mismatch")
    result = {"F1": {"passed": True, "checkpoint_critic": loaded, "frozen_critic": exact,
                      "critic_optimizer_steps": 0, "completed_warmup_batches": 0},
              "F2": {"passed": True, "requested": settings["gm_scaler_c"], "effective": divisor}}
    scaler_before = gm.scaler_state(agent)
    audit = json.loads(args.divisor_audit.read_text())
    divisor_checks = []
    for iteration in ("0", "35", "70", "105"):
        requested = audit["checkpoints"][iteration]["divisor"]
        effective = set_scaler_divisor(agent, requested)
        require(effective == requested, "F2 supplied checkpoint divisor differs")
        divisor_checks.append(dict(checkpoint_itr=int(iteration), requested=requested,
                                   effective=effective, passed=True))
    # A known through-origin fit, independent of the production fallback sample.
    probe_values = np.asarray([1., 2., 3.], dtype=np.float64)
    probe_returns = 512.25 * probe_values
    fitted = float(np.sum(probe_values * probe_returns) / np.sum(probe_values ** 2))
    require(set_scaler_divisor(agent, fitted) == fitted == 512.25, "F2 fitted divisor differs")
    gm._restore_scaler(agent, scaler_before)
    result["F2"].update(supplied_divisors=divisor_checks, synthetic_fitted_divisor=fitted,
                         synthetic_fit_exact=True)
    batch = agent._stage3_collect(update_scaler=False)
    components = gm.theory_critic_components(agent, batch)
    # Explicit float64 residual definition, before actor float32 conversion.
    expected_adv = components["returns"] - divisor * np.asarray(batch["values"], dtype=np.float64)
    residual = components["critic_residual"]
    require(np.array_equal(expected_adv, residual), "V2 raw residual definition differs")
    before_diag = saved_state(agent, gm)
    permutation = pair_permutation(agent, 0)
    require(agent.model.actor_loss == "score" and agent.model.norm_adv and
            agent.model.logprob_reduce == "mean" and agent.model.gamma_denoising == .99 and
            not agent.model.clamp_logprob, "quarter gate flags differ from training")
    measured = normalized_gradient(agent, batch, residual, permutation, "quarter").double()
    native = native_quarter_gradient(agent, batch, residual, permutation).double()
    difference = measured - native
    rel = float(difference.norm().item() / native.norm().item())
    require(torch.isfinite(measured).all() and rel <= 1e-6, "native quarter gradient mismatch")
    inert = compare(before_diag, saved_state(agent, gm), True)
    require(inert["passed"], "quarter gradient diagnostic changed state")
    result["quarter_native_gradient"] = {"passed": True, "relative_l2": rel,
        "max_abs": difference.abs().max().item(), "pairs": len(permutation), "chunk_size": agent.batch_size,
        "definition": "four native model.loss score terms weighted by chunk_size/full_size", "inert": inert}
    result["raw_V2_definition"] = {"passed": True, "max_abs": 0.0, "relative_l2": 0.0,
                                     "definition": "float64 raw MC return - frozen divisor * pre-update V"}
    # Native inherited definitions vs explicitly flagged quarter production path.
    reference_vectors = {}
    for index, (advantages, reduction, discount, normalize) in enumerate(_variant_coefficients(agent, batch)):
        if index != 1:
            reference_vectors[index] = -actor_gradient(agent, batch, advantages, reduce=reduction,
                gamma_denoising=discount, normalize=normalize, clamp=False).double().numpy()
    v6_before = gm.critic_gradient(agent, batch)
    agent.cfg.train.gm_normalization = "quarter"
    current_vectors = batch_gradients_v5(agent, batch)
    v6_after = gm.critic_gradient(agent, batch)
    untouched = {"V%d" % (index + 1): compare(value, current_vectors[index], True)
                 for index, value in reference_vectors.items()}
    untouched["V6"] = compare(v6_before, v6_after, True)
    require(all(value["passed"] for value in untouched.values()), "non-V2 estimator changed")
    require(compare(-measured.numpy(), current_vectors[1], True)["passed"], "production V2 differs from quarter helper")
    agent.cfg.train.gm_normalization = "full"
    require(compare(before_diag, saved_state(agent, gm), True)["passed"], "variant gate changed state")
    result["other_estimators_unchanged"] = {"passed": True, "comparisons": untouched}
    return result


def worker(args):
    require(not args.output.exists(), "worker output exists")
    args.output.mkdir(parents=True)
    agent, gm, failure = None, None, None
    started = time.monotonic()
    report = {"status": "running", "worker": args.worker, "device": "cpu", "numeric_settings": numeric_settings()}
    try:
        agent, gm, settings = build_agent(args)
        fixture = torch.load(args.batch, map_location="cpu", weights_only=False)
        require(fixture["raw_rewards"].shape == (500, 40), "saved batch must have full geometry")
        agent._stage3_collect = replay_collector(agent, fixture)
        if args.worker == "features":
            report.update(feature_checks(agent, gm, settings, fixture, args))
        else:
            if args.worker == "old":
                torch.save(saved_state(agent, gm), args.output / "initial.pt")
            else:
                restore_state(agent, gm, torch.load(args.initial, map_location="cpu", weights_only=False))
            torch.save(saved_state(agent, gm), args.output / "input.pt")
            metadata = gm.prepare(agent, settings)
            torch.save(saved_state(agent, gm), args.output / "final.pt")
            report.update(frozen_path=metadata["frozen_path"], frozen_sha256=metadata["frozen_sha256"],
                          completed_warmup_batches=metadata["completed_warmup_batches"],
                          critic_optimizer_steps=metadata["critic_optimizer_steps"], settings=settings)
        report["status"] = "passed"
    except BaseException as error:
        failure = error
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if agent is not None:
            try:
                report["cleanup"] = gm.cleanup_environment(agent, primary_error=failure)
            except BaseException as error:
                report.update(status="failed", cleanup_error=str(error))
                raise
            finally:
                report["seconds"] = time.monotonic() - started
                write(args.output / "worker.json", report)
        else:
            report["seconds"] = time.monotonic() - started
            write(args.output / "worker.json", report)


def snapshot(source, target, parent, revision, allow_new):
    inventory = subprocess.check_output(["git", "-C", str(parent), "ls-tree", "-r", "--name-only",
                                         revision, "--", SUBTREE], text=True).splitlines()
    require(inventory, "empty scientific source inventory")
    names = [Path(name).relative_to(SUBTREE) for name in inventory]
    if allow_new:
        names.append(Path("agent/finetune/grad_mean_v5.py"))
    target.mkdir(parents=True)
    for relative in sorted(set(names)):
        src, dst = source / relative, target / relative
        require(src.is_file(), "missing source " + str(src))
        if not allow_new:
            expected = subprocess.check_output(["git", "-C", str(parent), "show", revision + ":" + SUBTREE + "/" + relative.as_posix()])
            require(src.read_bytes() == expected, "old source is not the approved base: " + str(relative))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    return len(set(names))


def main(args):
    require(not args.output.exists(), "gate output exists; preserve prior evidence")
    require(not torch.cuda.is_available(), "CPU gate must have no visible GPU")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"passed": False, "status": "running", "device": "cpu", "base_commit": args.base_commit,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "expected_gpu_count": 0,
              "source_snapshot": str(args.output / "source_snapshot"),
              "saved_batch_source": str(args.batch), "checkpoint": str(args.checkpoint), "seed": args.seed,
              "F3_protocol": "identical saved CPU batch/model/optimizer/scheduler/scaler/RNG and identical provenance; exact old/new frozen-file bytes",
              "workers": []}
    try:
        report["source_files"] = snapshot(args.source, args.output / "source_snapshot", args.parent, args.base_commit, True)
        reference = args.output / "reference_snapshot"
        report["reference_files"] = snapshot(args.reference, reference, args.parent, args.base_commit, False)
        for name in ("gate_v5.py", "gate_common.py"):
            shutil.copy2(str(Path(__file__).parent / name), str(args.output / name))
        for name in ("old", "new", "features"):
            command = [sys.executable, "-B", str(args.output / "gate_v5.py"), "--worker", name,
                       "--source", str(reference if name == "old" else args.output / "source_snapshot"),
                       "--prepare", str(args.prepare), "--checkpoint", str(args.checkpoint),
                       "--batch", str(args.batch), "--output", str(args.output / name),
                       "--divisor-audit", str(args.divisor_audit),
                       "--base-commit", args.base_commit, "--seed", str(args.seed)]
            if name == "new":
                command += ["--initial", str(args.output / "old/initial.pt")]
            report["workers"].append({"name": name, "command": command})
            write(args.output / "gate_summary.json", report)
            with (args.output / (name + ".out")).open("x") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=5400)
            require(completed.returncode == 0, name + " gate worker failed; see its log")
            evidence = json.loads((args.output / name / "worker.json").read_text())
            require(evidence["status"] == "passed" and evidence["cleanup"]["status"] == "passed", name + " incomplete")
        first = torch.load(args.output / "old/input.pt", map_location="cpu", weights_only=False)
        second = torch.load(args.output / "new/input.pt", map_location="cpu", weights_only=False)
        inputs = compare(first, second, True)
        outputs = compare(torch.load(args.output / "old/final.pt", map_location="cpu", weights_only=False),
                          torch.load(args.output / "new/final.pt", map_location="cpu", weights_only=False), True)
        old_frozen = args.output / "old/frozen_D1.pt"
        new_frozen = args.output / "new/frozen_D1.pt"
        same_bytes = old_frozen.read_bytes() == new_frozen.read_bytes()
        report["F3"] = {"passed": inputs["passed"] and outputs["passed"] and same_bytes,
                        "identical_saved_inputs": inputs, "final_states": outputs,
                        "frozen_files_byte_identical": same_bytes, "frozen_file_bytes": old_frozen.stat().st_size,
                        "old_frozen": str(old_frozen), "new_frozen": str(new_frozen),
                        "warmup_batches": 5, "critic_steps": 100}
        features = json.loads((args.output / "features/worker.json").read_text())
        for key in ("F1", "F2", "quarter_native_gradient", "raw_V2_definition", "other_estimators_unchanged"):
            report[key] = features[key]
        require(report["F3"]["passed"], "F3 default replay differs")
        report.update(passed=True, status="passed")
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        write(args.output / "gate_summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--divisor-audit", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--worker", choices=("old", "new", "features"))
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--seed", type=int, default=4900000)
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
