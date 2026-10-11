#!/usr/bin/env python
"""Bounded saved-batch FRS gate. This entry point never collects trajectories."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from gate_common import compare, numeric_settings, require, rng_comparison

STUDY = Path(__file__).resolve().parents[1]
PARENT = STUDY.parents[1]
CODE = STUDY / "code"
REFERENCE = PARENT / "results/stage3/dppo"
EDITED = STUDY / "dppo"
CAPTURE = PARENT / "results/stage3/runs/gate_capture/nc4/training_batch.pt"
STATS = PARENT / "results/stage3/ns_theta0_seed1000/noise_batch_vector_sums.npy"


def source_files(root):
    return {str(p.relative_to(root)): p for p in root.rglob("*")
            if p.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".git")
                                       for part in p.relative_to(root).parts)
            and p.suffix not in (".pyc", ".pyo")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--out", type=Path, default=STUDY / "runs/gate")
    parser.add_argument("--device", choices=("cpu", "cuda:0", "both"), default="both")
    parser.add_argument("--timeout-seconds", type=int, default=3540)
    args = parser.parse_args()
    output = args.output.resolve()
    require(STUDY in output.parents and not output.exists(), "gate output must be new inside this study")
    require(0 < args.timeout_seconds <= 3540, "gate must fit the approved one-hour allocation")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "passed": False, "command": sys.argv,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "allocation_cap_seconds": 3600,
              "reference": str(REFERENCE), "edited": str(EDITED),
              "fixture": str(CAPTURE), "processing_seed": 4000,
              "cached_measurement_seed": 1000, "cached_variant": "V4",
              "gpu_differential_tolerance": {"atol": 1e-7, "rtol": 1e-5},
              "cpu_differential_criterion": "bitwise",
              "formula_criterion": "relative L2 error of updated parameters <= 1e-7; displacement rounding also reported",
              "numeric_settings_before": numeric_settings(),
              "commands": [], "differential": {}, "diagnostics": {}, "features": {}}

    def save():
        (output / "gate_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def run(script, name, extra):
        report["active_phase"] = name
        command = [sys.executable, "-B", str(CODE / script), "--output", str(output / name), *extra]
        report["commands"].append(command)
        save()
        remaining = args.timeout_seconds - (time.monotonic() - started)
        require(remaining > 0, "gate time limit reached")
        print("GATE START " + name, flush=True)
        with (output / (name + ".out")).open("w") as stream:
            process = subprocess.run(command, cwd=str(EDITED), stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=remaining)
        require(process.returncode == 0, name + " failed; inspect its saved output")
        print("GATE COMPLETE " + name, flush=True)
        return output / name

    def load(path):
        return torch.load(path / "evidence.pt", map_location="cpu", weights_only=False)

    try:
        require(not torch.are_deterministic_algorithms_enabled(), "deterministic algorithms unexpectedly enabled")
        require(not any(p.name == ".git" for p in EDITED.rglob("*")), "ordinary source snapshot contains nested Git metadata")
        require(CAPTURE.is_file(), "approved saved training batch missing; do not collect a replacement")
        report["source_import"] = json.loads((STUDY / "provenance/dppo_source_snapshot.json").read_text())
        # FRS1's approved step size is 1e-4. The unchanged SGD branch receives
        # the same explicit value on both sides of the differential check.
        actor_lr = "0.0001"
        report["frs_lr"] = 1e-4
        report["sgd_differential_lr"] = 1e-4
        report["parent_commit"] = subprocess.check_output(
            ["git", "-C", str(PARENT), "rev-parse", "HEAD"], text=True).strip()
        report["parent_branch"] = subprocess.check_output(
            ["git", "-C", str(PARENT), "branch", "--show-current"], text=True).strip()
        files = source_files(EDITED)
        source_bytes = {name: path.read_bytes() for name, path in files.items()}
        snapshot = output / "source_snapshot"
        for name, path in files.items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(destination))
        report["source_snapshot"] = str(snapshot)
        report["source_file_count"] = len(files)
        diffs = []
        for name in sorted(set(files) | set(source_files(REFERENCE))):
            old_path = REFERENCE / name
            old = old_path.read_bytes() if old_path.is_file() else b""
            new = source_bytes.get(name, b"")
            if old == new:
                continue
            try:
                diffs.extend(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                                                fromfile="stage3/" + name, tofile="stage3_frs/" + name))
            except UnicodeDecodeError:
                diffs.append("Binary source differs: " + name + "\n")
        (output / "source_changes.diff").write_text("".join(diffs))
        report["source_changes"] = str(output / "source_changes.diff")
        for path in CODE.glob("gate*.py"):
            destination = output / "gate_code_snapshot" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(str(path), str(destination))

        devices = ["cpu", "cuda:0"] if args.device == "both" else [args.device]
        if "cuda:0" in devices:
            require(torch.cuda.is_available(), "GPU gate requested without CUDA")
            report["gpu_name"] = torch.cuda.get_device_name(0)
        common = ["--fixture", str(CAPTURE)]
        prepared = run("gate_replay.py", "prepare_cpu",
                       ["--repo", str(REFERENCE), *common, "--device", "cpu", "--prepare-cpu"])
        cpu_old = prepared / "cpu_old_logprobs.pt"
        report["cpu_oldlogprob_preparation"] = str(cpu_old)

        # Two conversions share the original Phase-1 measurement seed 1000;
        # processing seed 4000 introduces no new measurements or randomness.
        rescale = []
        report["cached_conversion"] = {}
        for repeat in ("A", "B"):
            path = run("gate_replay.py", "rescale_seed4000_" + repeat,
                       ["--repo", str(EDITED), *common, "--device", "cpu", "--convert-stats",
                        "--rescale-source", str(STATS)])
            rescale.append(path / "native/rescale.pt")
            require(rescale[-1].is_file(), "cached rescale artifact missing")
            report["cached_conversion"][repeat] = json.loads((path / "worker.json").read_text())
        a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in rescale]
        keys = ("parameter_specs", "r", "r_tilde", "floor", "epsilon", "global_median_r")
        report["cached_conversion"]["repeat_comparison"] = {
            key: compare(a[key], b[key], True, key) for key in keys}
        require(all(x["passed"] for x in report["cached_conversion"]["repeat_comparison"].values()),
                "repeated cached conversion is not bitwise identical")
        report["rescale_path"] = str(rescale[0])
        report["rescale_repeat_reference_path"] = str(rescale[1])
        del a, b

        keys = ("initial_rng", "initial_model", "initial_actor_optimizer", "initial_critic_optimizer",
                "constructor_actor_optimizer", "constructor_critic_optimizer",
                "constructor_actor_scheduler", "constructor_critic_scheduler", "input_batch",
                "loss_components", "actor_gradients", "critic_gradients", "permutations",
                "actor_parameters", "critic_parameters", "model_state", "actor_optimizer_state",
                "critic_optimizer_state", "actor_scheduler_state", "critic_scheduler_state",
                "rng_after_update", "diagnostic_generator_state", "logged_values", "sampling", "input_batch_after")
        strict = {"initial_rng", "permutations", "rng_after_update", "diagnostic_generator_state"}
        for device in devices:
            extra = [*common, "--device", device]
            if device == "cpu":
                extra += ["--cpu-logprobs", str(cpu_old)]
            for case, flags in (("nc4_adamw", []), ("e2_adamw", ["--e2", "--actor-optimizer", "adamw"]),
                                ("e2_sgd", ["--e2", "--actor-optimizer", "sgd", "--actor-lr", actor_lr])):
                name = case + "_" + device.replace(":", "")
                before = run("gate_replay.py", name + "_reference", ["--repo", str(REFERENCE), *extra, *flags])
                after = run("gate_replay.py", name + "_edited", ["--repo", str(EDITED), *extra, *flags])
                old, new = load(before), load(after)
                comparisons = {key: compare(old[key], new[key], device == "cpu" or key in strict, key)
                               for key in keys}
                require(len(old["actor_gradients"]) == len(new["actor_gradients"]) == 1,
                        "saved NC4 batch must receive exactly one actor update")
                require(len(old["critic_gradients"]) == len(new["critic_gradients"]) == 20,
                        "saved batch must retain exactly twenty critic updates")
                result = {"passed": all(x["passed"] for x in comparisons.values()), "comparisons": comparisons,
                          "actor_steps": 1, "critic_steps": 20}
                report["differential"][name] = result
                save()
                require(result["passed"], "default-path differential failed: " + name)
                del old, new

            # Full-update comparison tests diagnostics with the actual FRS optimizer.
            flags = ["--repo", str(EDITED), *extra, "--e2", "--actor-optimizer", "sgd_diag",
                     "--rescale-path", str(rescale[0]), "--actor-lr", actor_lr]
            offpath = run("gate_replay.py", "frs_off_" + device.replace(":", ""), flags)
            onpath = run("gate_replay.py", "frs_on_" + device.replace(":", ""), [*flags, "--diagnostics"])
            off, on = load(offpath), load(onpath)
            inert_keys = ("prepared_batch", "actor_parameters", "critic_parameters", "model_state",
                          "actor_optimizer_state", "critic_optimizer_state", "rng_after_update", "input_batch_after",
                          "actor_gradients", "critic_gradients", "permutations")
            strict_inert = {"rng_after_update", "permutations", "input_batch_after", "prepared_batch"}
            comparisons = {key: compare(off[key], on[key], device == "cpu" or key in strict_inert, key)
                           for key in inert_keys}
            isolation = rng_comparison(on["rng_before_prepare"], on["rng_after_prepare"])
            result = {"passed": all(x["passed"] for x in comparisons.values()) and isolation["passed"],
                      "comparisons": comparisons, "diagnostic_rng_isolation": isolation,
                      "measurement_state_inert": on["measurement_state_inert"],
                      "saved_batch_unchanged": on["update_batch_inert"]}
            report["diagnostics"][device] = result
            save()
            require(result["passed"], "FRS diagnostics changed training or global RNG")
            del off, on
            path = run("gate_frs_features.py", "features_" + device.replace(":", ""),
                       ["--repo", str(EDITED), "--device", device, "--rescale-path", str(rescale[0]),
                        "--replay-evidence", str(offpath / "evidence.pt")])
            report["features"][device] = json.loads((path / "features.json").read_text())
            require(report["features"][device]["passed"], "FRS feature gate failed")

        current_files = source_files(EDITED)
        require(set(current_files) == set(source_bytes) and
                all(path.read_bytes() == source_bytes[name] for name, path in current_files.items()),
                "scientific source changed during gate")
        report["scientific_source_unchanged_during_gate"] = True
        report["requested_checks_passed"] = True
        report["passed"] = args.device == "both"
        report["status"] = "passed" if report["passed"] else "partial_device_checks_passed"
        report.pop("active_phase", None)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["numeric_settings_after"] = numeric_settings()
        save()


if __name__ == "__main__":
    main()
