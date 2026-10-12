#!/usr/bin/env python
"""CPU H1-H4 gate. Two actual 500x40 batches in each isolated source process.

The two-batch length is a test-only direct call to collect; production settings
are validated at50 before copying the local harness dictionary with batches=2.
"""
import argparse
import copy
import csv
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

from gate_common import compare, freeze, require, rng_state, numeric_settings

PARENT = "44b2998228e1a93aaef544e8030c30640d17dc06"
TIMING = {"rollout_seconds", "gradient_seconds", "diagnostic_seconds", "seconds"}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_agent(args, enabled):
    sys.path.insert(0, str(args.source))
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from agent.finetune import grad_mean as gm
    prepare = json.loads(args.prepare.read_text())
    cfg = OmegaConf.create(prepare["config"])
    overrides = {"device": "cpu", "model.device": "cpu", "seed": args.seed,
                 "logdir": str(args.output), "wandb": None, "train.mode": "grad_mean",
                 "train.gm_step": "collect", "train.gm_batches": 50,
                 "train.gm_score_sample": 0, "train.gm_parent_commit": PARENT,
                 "train.gm_frozen_path": prepare["frozen_path"],
                 "train.gm_frozen_sha256": prepare["frozen_sha256"],
                 "model.clamp_logprob": False, "model.randn_clip_value": 100}
    if enabled:
        overrides["train.gm_include_critic"] = True
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    agent = get_class(cfg._target_)(cfg)
    settings = gm.settings(agent)
    gm._validate(agent, settings)
    return agent, gm, settings, prepare


def worker(args):
    require(not args.output.exists(), "worker output must be new")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "device": "cpu", "worker": args.worker,
              "command": sys.argv, "settings": numeric_settings()}
    agent = None
    failure = None
    try:
        agent, gm, settings, prepare = build_agent(args, args.worker != "old")
        if args.worker in ("old", "new"):
            # Load once for the initial state evidence; collect checks/loads again.
            gm.load_frozen(agent, settings)
            report["initial_frozen_hashes"] = gm.state_hashes(agent)
            gm.seed_environments(agent)
            initial_rng = rng_state()
            original_collect = agent._stage3_collect
            batch_rng = []
            count = [0]
            def capture(*a, **kw):
                value = original_collect(*a, **kw)
                batch_rng.append(rng_state())
                if count[0] == 0 and args.worker == "old":
                    torch.save(freeze(value), args.output / "saved_batch.pt")
                count[0] += 1
                return value
            agent._stage3_collect = capture
            short = dict(settings, gm_batches=2)
            meta = gm.collect(agent, short)
            report["production_validated_batches"] = settings["gm_batches"]
            report["gate_only_batches"] = short["gm_batches"]
            report["final_frozen_hashes"] = gm.state_hashes(agent)
            torch.save({"initial": initial_rng, "after_rollouts": batch_rng,
                        "final": rng_state()}, args.output / "rng.pt")
            require(meta["status"] == "completed" and meta["completed_batches"] == 2,
                    "short collection incomplete")
        else:
            gm.load_frozen(agent, settings)
            # No fresh rollout: this is the first actual old-source H3 batch.
            batch = torch.load(args.batch, map_location="cpu", weights_only=False)
            from agent.finetune.stage3_math import actor_gradient, loo_advantages
            from agent.finetune.stage3_modes import _variant_coefficients
            before = {"rng": rng_state(), "model": freeze(agent.model.state_dict()),
                      "actor_optimizer": freeze(agent.actor_optimizer.state_dict()),
                      "critic_optimizer": freeze(agent.critic_optimizer.state_dict()),
                      "grads": freeze([p.grad for p in agent.model.parameters()]), "batch": freeze(batch),
                      "scaler": gm.scaler_state(agent)}
            components = gm.theory_critic_components(agent, batch)
            rewards = np.asarray(batch["raw_rewards"], dtype=np.float64)
            firsts = batch["firsts"]
            explicit = np.zeros_like(rewards)
            for start in range(0, len(rewards), 250):
                for k in range(250):
                    explicit[start + k] = np.sum(
                        rewards[start + k:start + 250] *
                        (agent.gamma ** np.arange(250 - k, dtype=np.float64))[:, None], axis=0, dtype=np.float64)
            scaler = agent.running_reward_scaler
            divisor = float(np.sqrt(np.asarray(scaler.ret_rms.var, dtype=np.float64) + scaler.epsilon))
            k = np.broadcast_to((np.arange(len(rewards)) % 250)[:, None], rewards.shape)
            expected = (agent.gamma ** k) * (explicit - divisor * np.asarray(batch["values"], dtype=np.float64))
            delta = components["weighted_float64"] - expected
            relative = float(np.linalg.norm(delta) / np.linalg.norm(expected))
            report["H1"] = {"passed": relative <= 1e-12, "relative_l2": relative,
                            "max_abs": float(np.max(np.abs(delta))), "divisor": divisor,
                            "float64_definition": "gamma^k*(raw MC G-c*pre-update frozen V)",
                            "actor_cast": "float32 residual then float32 gamma^k, identical to V4/V5"}
            require(report["H1"]["passed"], "H1 weight mismatch")
            inherited = _variant_coefficients(agent, batch)
            old_baseline = gm.frozen_critic_baseline
            gradients = []
            for index, name in ((3, "LOO_to_V4"), (4, "zero_to_V5")):
                base = (components["returns"] - loo_advantages(components["returns"], firsts)[0]
                        if index == 3 else np.zeros_like(rewards))
                gm.frozen_critic_baseline = lambda _agent, _batch, base=base: (base.copy(), divisor)
                try:
                    v6 = gm.critic_gradient(agent, batch)
                finally:
                    gm.frozen_critic_baseline = old_baseline
                adv, reduction, discount, norm = inherited[index]
                target = -actor_gradient(agent, batch, adv, reduce=reduction,
                                         gamma_denoising=discount, normalize=norm, clamp=False).double().cpu().numpy()
                diff = v6 - target
                rel = float(np.linalg.norm(diff) / np.linalg.norm(target))
                gradients.append({"name": name, "passed": rel <= 1e-6,
                                  "relative_l2": rel, "max_abs": float(np.max(np.abs(diff)))})
            report["H2"] = {"passed": all(x["passed"] for x in gradients), "comparisons": gradients}
            require(report["H2"]["passed"], "H2 gradient identity failed")
            after = {"rng": rng_state(), "model": freeze(agent.model.state_dict()),
                     "actor_optimizer": freeze(agent.actor_optimizer.state_dict()),
                     "critic_optimizer": freeze(agent.critic_optimizer.state_dict()),
                     "grads": freeze([p.grad for p in agent.model.parameters()]), "batch": freeze(batch),
                     "scaler": gm.scaler_state(agent)}
            report["saved_batch_inert"] = compare(before, after, True)
            require(report["saved_batch_inert"]["passed"], "H1/H2 mutated saved state")
        report["status"] = "passed"
    except BaseException as error:
        failure = error
        report.update(status="failed", error=type(error).__name__ + ": " + str(error),
                      traceback=traceback.format_exc())
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


def source_snapshot(source, target, parent):
    subtree = "results/stage3_gradmean/dppo"
    names = subprocess.check_output(["git", "-C", str(parent), "ls-tree", "-r", "--name-only",
                                     PARENT, "--", subtree], text=True).splitlines()
    require(names, "empty source inventory")
    target.mkdir(parents=True)
    for name in names:
        relative = Path(name).relative_to(subtree)
        src = source / relative
        dst = target / relative
        require(src.is_file(), "missing source " + str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    return len(names)


def main(args):
    require(not args.output.exists(), "gate output must be new")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "passed": False, "device": "cpu", "parent_commit": PARENT,
              "source_snapshot": str(args.output / "source_snapshot"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv,
              "seed": args.seed, "cuda_generator_count": 0, "workers": []}
    try:
        # Frozen copies protect the gate while other queues continue unchanged.
        report["source_files"] = source_snapshot(args.source, args.output / "source_snapshot", args.parent)
        old = args.output / "reference_snapshot"
        report["reference_files"] = source_snapshot(args.reference, old, args.parent)
        for path in old.rglob("*"):
            if path.is_file():
                name = "results/stage3_gradmean/dppo/" + path.relative_to(old).as_posix()
                original = subprocess.check_output(["git", "-C", str(args.parent), "show", PARENT + ":" + name])
                require(path.read_bytes() == original, "reference source differs from44b: " + name)
        for name in ("v6_gate.py", "gate_common.py"):
            shutil.copy2(str(Path(__file__).parent / name), str(args.output / name))
        for name in ("old", "new", "identities"):
            command = [sys.executable, "-B", str(args.output / "v6_gate.py"), "--worker", name,
                       "--source", str(old if name == "old" else args.output / "source_snapshot"),
                       "--prepare", str(args.prepare), "--output", str(args.output / name),
                       "--seed", str(args.seed)]
            if name == "identities":
                command += ["--batch", str(args.output / "old/saved_batch.pt")]
            report["workers"].append({"name": name, "command": command})
            write(args.output / "gate_summary.json", report)
            with (args.output / (name + ".out")).open("x") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=6600)
            require(completed.returncode == 0, name + " worker failed: " + str(completed.returncode))
            evidence = json.loads((args.output / name / "worker.json").read_text())
            require(evidence["status"] == "passed", name + " worker failed")
            require(evidence["cleanup"]["status"] == "passed" and evidence["cleanup"]["all_children_exited"]
                    and not evidence["cleanup"]["remaining_active_children"]
                    and not evidence["cleanup"]["remaining_direct_children"],
                    name + " cleanup failed")
            report["workers"][-1]["seconds"] = evidence["seconds"]
        identity = json.loads((args.output / "identities/worker.json").read_text())
        report["H1"], report["H2"] = identity["H1"], identity["H2"]
        left, right = args.output / "old", args.output / "new"
        sums = [compare(np.load(left / ("sums_V%d.npy" % v)),
                        np.load(right / ("sums_V%d.npy" % v)), True, "V%d" % v) for v in range(1, 6)]
        with (left / "batches.csv").open() as stream:
            lrows = list(csv.DictReader(stream))
        with (right / "batches.csv").open() as stream:
            rrows = list(csv.DictReader(stream))
        require(len(lrows) == len(rrows) == 2, "H3 requires two full batches")
        scalar_diffs = []
        for index, (a, b) in enumerate(zip(lrows, rrows)):
            for key in a:
                if key not in TIMING and a[key] != b.get(key):
                    scalar_diffs.append({"batch": index, "key": key, "old": a[key], "new": b.get(key)})
        rng = compare(torch.load(left / "rng.pt", weights_only=False),
                      torch.load(right / "rng.pt", weights_only=False), True)
        report["H3"] = {"passed": all(x["passed"] for x in sums) and not scalar_diffs and rng["passed"],
                        "sums": sums, "scalar_differences": scalar_diffs, "rng": rng,
                        "excluded_keys": sorted(TIMING), "actual_rollouts_per_source": 2,
                        "decisions_per_rollout": 500, "envs": 40}
        require(report["H3"]["passed"], "H3 old/new invariance failed")
        prepare = json.loads(args.prepare.read_text())
        states = {}
        for label in ("old", "new"):
            meta = json.loads((args.output / label / "meta.json").read_text())
            states[label] = {}
            for name in ("actor", "critic", "scaler"):
                expected = prepare[name + "_sha256"]
                before = meta[name + "_before_sha256"]
                after = meta[name + "_after_sha256"]
                states[label][name] = {"prepare": expected, "before": before, "after": after,
                                       "passed": expected == before == after}
        report["H4"] = {"passed": all(v["passed"] for s in states.values() for v in s.values()),
                        "states": states, "prepare_record": str(args.prepare)}
        require(report["H4"]["passed"], "H4 frozen-state mismatch")
        report.update(status="passed", passed=True)
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        write(args.output / "gate_summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("old", "new", "identities"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--seed", type=int, default=2900000)
    args = parser.parse_args()
    # No deterministic-algorithm settings; these are bounded CPU gate resources.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    worker(args) if args.worker else main(args)
