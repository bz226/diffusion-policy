#!/usr/bin/env python
"""G1/G5 saved-batch gate; requires the fixed G2 gate to have passed first."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from gate_common import compare, require, numeric_settings

HERE = Path(__file__).resolve().parents[1]
PARENT = HERE.parents[2]
REFERENCE = PARENT / "results/stage3_frs/dppo"
EDITED = PARENT / "results/stage3_gradmean/dppo"
CODE = HERE / "code"


def source_files(root):
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()
            and not any(part in ("__pycache__", ".git", ".pytest_cache") for part in p.relative_to(root).parts)
            and p.suffix not in (".pyc", ".pyo")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7140)
    args = parser.parse_args()
    output = args.output.resolve()
    require(HERE in output.parents and not output.exists(), "Use a fresh gate directory inside Addendum G")
    require(0 < args.timeout_seconds <= 7140, "Gate must fit approved two-hour allocation")
    g2 = json.loads(args.g2.read_text())
    require(g2.get("passed") is True and g2.get("gate") == "G2-G1", "G2 must pass before G1/G5 allocation")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"stage": "offline", "passed": False, "status": "running", "commands": [],
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "G2": str(args.g2.resolve()),
              "reference": str(REFERENCE), "edited": str(EDITED), "G1": {}, "G5": {},
              "numeric_settings": numeric_settings(), "gate_wall_limit_seconds": 7200,
              "parent_commit": subprocess.check_output(["git", "-C", str(PARENT), "rev-parse", "HEAD"], text=True).strip()}
    def save():
        (output / "gate_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def run(script, name, flags):
        report["active_phase"] = name
        command = [sys.executable, "-B", str(CODE / script), "--output", str(output / name), *flags]
        report["commands"].append(command)
        save()
        remaining = args.timeout_seconds - (time.monotonic() - started)
        require(remaining > 0, "Offline gate reached its time cap")
        print("GATE START " + name, flush=True)
        with (output / (name + ".out")).open("x") as stream:
            result = subprocess.run(command, cwd=EDITED, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
        require(result.returncode == 0, name + " failed; inspect saved output")
        return output / name
    def load(path):
        return torch.load(path / "evidence.pt", map_location="cpu", weights_only=False)
    try:
        require(torch.cuda.is_available(), "G1/G5 allocation requires GPU for differential replay")
        require(not torch.are_deterministic_algorithms_enabled(), "Do not change deterministic-algorithm settings")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        files = source_files(EDITED)
        source_bytes = {name: path.read_bytes() for name, path in files.items()}
        snapshot = output / "source_snapshot"
        for name, path in files.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        report["source_snapshot"] = str(snapshot)
        report["source_files"] = len(files)
        code_snapshot = output / "gate_code_snapshot"
        code_snapshot.mkdir()
        for path in CODE.glob("gate*.py"):
            shutil.copy2(path, code_snapshot / path.name)
        fixture = PARENT / "results/stage3/runs/gate_capture/nc4/training_batch.pt"
        g1_paths = {}
        for kind, repo in (("reference", REFERENCE), ("edited", EDITED), ("e2", REFERENCE)):
            g1_paths[kind] = run("gate_gradients.py", "G1_" + kind,
                                ["--repo", str(repo), "--fixture", str(fixture), "--kind", kind])
        vectors = {key: np.load(path / "gradients.npy") for key, path in g1_paths.items()}
        comparisons = []
        for i in range(5):
            old, new = vectors["reference"][i], vectors["edited"][i]
            absolute = np.abs(new - old)
            relative = float(np.linalg.norm(new - old) / np.linalg.norm(old))
            comparisons.append({"variant": "V%d" % (i + 1), "relative_l2": relative,
                                "max_abs": float(absolute.max()), "passed": relative <= 1e-6})
        old, new = vectors["e2"][0], vectors["edited"][3]
        relative = float(np.linalg.norm(new - old) / np.linalg.norm(old))
        report["G1"] = {"passed": all(x["passed"] for x in comparisons) and relative <= 1e-6,
                        "per_variant": comparisons, "V4_vs_negative_E2_loss_gradient": {
                            "relative_l2": relative, "max_abs": float(np.abs(new - old).max()), "passed": relative <= 1e-6},
                        "phase1_reference_sign": "negative native loss gradient, mean over40 envs"}
        save()
        require(report["G1"]["passed"], "G1 saved-batch definitions differ")
        keys = ("initial_rng", "initial_model", "initial_actor_optimizer", "initial_critic_optimizer",
                "constructor_actor_optimizer", "constructor_critic_optimizer", "constructor_actor_scheduler",
                "constructor_critic_scheduler", "input_batch", "loss_components", "actor_gradients",
                "critic_gradients", "permutations", "actor_parameters", "critic_parameters", "model_state",
                "actor_optimizer_state", "critic_optimizer_state", "actor_scheduler_state", "critic_scheduler_state",
                "rng_after_update", "diagnostic_generator_state", "logged_values", "sampling", "input_batch_after")
        strict = {"initial_rng", "permutations", "rng_after_update", "diagnostic_generator_state"}
        for condition in ("default", "nc4"):
            fixture = PARENT / "results/stage3/runs/gate_capture" / condition / "training_batch.pt"
            prepared = run("gate_replay.py", "G5_prepare_cpu_" + condition,
                           ["--repo", str(REFERENCE), "--fixture", str(fixture), "--device", "cpu", "--prepare-cpu"])
            for device in ("cpu", "cuda:0"):
                name = condition + "_" + device.replace(":", "")
                flags = ["--fixture", str(fixture), "--device", device]
                if device == "cpu":
                    flags += ["--cpu-logprobs", str(prepared / "cpu_old_logprobs.pt")]
                before = run("gate_replay.py", "G5_" + name + "_reference", ["--repo", str(REFERENCE), *flags])
                after = run("gate_replay.py", "G5_" + name + "_edited", ["--repo", str(EDITED), *flags])
                old, new = load(before), load(after)
                checks = {key: compare(old[key], new[key], device == "cpu" or key in strict, key) for key in keys}
                expected_actor = 20 if condition == "default" else 1
                require(len(old["actor_gradients"]) == len(new["actor_gradients"]) == expected_actor, "Actor step count changed")
                require(len(old["critic_gradients"]) == len(new["critic_gradients"]) == 20, "Critic step count changed")
                report["G5"][name] = {"passed": all(x["passed"] for x in checks.values()), "comparisons": checks,
                                       "actor_steps": expected_actor, "critic_steps": 20}
                save()
                require(report["G5"][name]["passed"], "G5 differential check failed: " + name)
                del old, new
        current = source_files(EDITED)
        require(current.keys() == source_bytes.keys() and all(p.read_bytes() == source_bytes[n] for n, p in current.items()),
                "Scientific source changed during gate")
        report.update(passed=True, status="passed", scientific_source_unchanged=True)
        report.pop("active_phase", None)
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    main()
