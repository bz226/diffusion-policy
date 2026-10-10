#!/usr/bin/env python
"""Stage-3 pre-campaign gate: exact defaults replay and required feature checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from gate_common import compare, require, rng_comparison

STUDY = Path(__file__).resolve().parents[1]
CODE = STUDY / "code"
REFERENCE = STUDY / "reference_stage2"
EDITED = STUDY / "dppo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STUDY / "runs/gate")
    parser.add_argument("--timeout-seconds", type=int, default=2640)
    args = parser.parse_args()
    output = args.output.resolve()
    require(STUDY in output.parents and not output.exists(), "gate output must be new inside Stage3")
    require(0 < args.timeout_seconds <= 2640, "gate exceeds remaining45min allocation")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "passed": False, "command": sys.argv,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "reference_commit": "ab46b150fa34b5a5b457cd4062cd4c5ad830d964",
              "reference": str(REFERENCE), "edited": str(EDITED),
              "gpu_tolerance": {"atol": 1e-7, "rtol": 1e-5},
              "cpu_criterion": "bitwise", "commands": [], "A": {}, "B9": {}}
    def save():
        (output / "gate_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def run(script, name, extra):
        report["active_phase"] = name
        command = [sys.executable, "-B", str(CODE / script), "--output", str(output / name), *extra]
        report["commands"].append(command)
        save()
        print("GATE START " + name, flush=True)
        remaining = args.timeout_seconds - (time.monotonic() - started)
        require(remaining > 0, "gate cap exhausted")
        with (output / (name + ".out")).open("w") as stream:
            completed = subprocess.run(command, cwd=str(EDITED), stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=remaining)
        require(completed.returncode == 0, name + " failed; inspect " + str(output / (name + ".out")))
        print("GATE COMPLETE " + name, flush=True)
        return output / name
    def load(path):
        return torch.load(path / "evidence.pt", map_location="cpu", weights_only=False)
    try:
        capture = json.loads((STUDY / "runs/gate_capture/capture_summary.json").read_text())
        require(capture["status"] == "complete", "capture not complete")
        report["capture_summary"] = str(STUDY / "runs/gate_capture/capture_summary.json")
        require(torch.cuda.is_available(), "gate allocation requires GPU")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        sys.path.insert(0, str(REFERENCE))
        source_paths = ["agent/finetune/train_ppo_diffusion_agent.py", "agent/finetune/train_ppo_agent.py",
                        "model/diffusion/diffusion_ppo.py", "agent/finetune/stage3_support.py",
                        "agent/finetune/stage3_math.py", "agent/finetune/stage3_modes.py"]
        source_bytes = {name: (EDITED / name).read_bytes() for name in source_paths}
        for name, data in source_bytes.items():
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        report["source_snapshot"] = str(output / "source_snapshot")
        report["source_diff"] = subprocess.check_output(
            ["git", "--git-dir=" + str(EDITED / ".git"), "--work-tree=" + str(EDITED),
             "diff", "ab46b150fa34b5a5b457cd4062cd4c5ad830d964"], text=True)
        (output / "source.diff").write_text(report.pop("source_diff"))
        report["source_diff_path"] = str(output / "source.diff")
        cpulog = {}
        for condition in ("default", "nc4"):
            fixture = STUDY / "runs/gate_capture" / condition / "training_batch.pt"
            preparation = run("gate_replay.py", "prepare_cpu_" + condition,
                              ["--repo", str(REFERENCE), "--fixture", str(fixture), "--device", "cpu", "--prepare-cpu"])
            cpulog[condition] = preparation / "cpu_old_logprobs.pt"
            for device in ("cpu", "cuda:0"):
                name = condition + "_" + device.replace(":", "")
                extra = ["--fixture", str(fixture), "--device", device]
                if device == "cpu":
                    extra += ["--cpu-logprobs", str(cpulog[condition])]
                reference = run("gate_replay.py", name + "_reference", ["--repo", str(REFERENCE), *extra])
                edited = run("gate_replay.py", name + "_edited", ["--repo", str(EDITED), *extra])
                old, new = load(reference), load(edited)
                keys = ("initial_rng", "initial_model", "initial_actor_optimizer", "initial_critic_optimizer",
                        "constructor_actor_optimizer", "constructor_critic_optimizer",
                        "constructor_actor_scheduler", "constructor_critic_scheduler",
                        "input_batch", "loss_components", "actor_gradients", "critic_gradients", "permutations",
                        "actor_parameters", "critic_parameters", "model_state", "actor_optimizer_state",
                        "critic_optimizer_state", "actor_scheduler_state", "critic_scheduler_state",
                        "rng_after_update", "diagnostic_generator_state", "logged_values", "sampling",
                        "input_batch_after")
                strict = {"initial_rng", "permutations", "rng_after_update", "diagnostic_generator_state"}
                comparisons = {key: compare(old[key], new[key], device == "cpu" or key in strict, key) for key in keys}
                expected_steps = 20 if condition == "default" else 1
                require(len(old["actor_gradients"]) == len(new["actor_gradients"]) == expected_steps,
                        name + " actor step count mismatch")
                require(len(old["critic_gradients"]) == len(new["critic_gradients"]) == 20,
                        name + " critic step count mismatch")
                result = {"passed": all(item["passed"] for item in comparisons.values()),
                          "comparisons": comparisons, "actor_steps": expected_steps, "critic_steps": 20,
                          "cpu_oldlogprob_preparation": str(cpulog[condition]) if device == "cpu" else None}
                report["A"][name] = result
                save()
                require(result["passed"], "A defaults differential check failed for " + name)
                del old, new
        fixture = STUDY / "runs/gate_capture/nc4/training_batch.pt"
        report["features"] = {}
        for device in ("cpu", "cuda:0"):
            extra = ["--repo", str(EDITED), "--fixture", str(fixture), "--device", device]
            if device == "cpu":
                extra += ["--cpu-logprobs", str(cpulog["nc4"])]
            path = run("gate_features.py", "features_" + device.replace(":", ""), extra)
            report["features"][device] = json.loads((path / "features.json").read_text())
            save()
            require(report["features"][device]["status"] == "passed", "feature gate failed")
        extra = ["--repo", str(EDITED), "--fixture", str(fixture), "--device", "cpu",
                 "--cpu-logprobs", str(cpulog["nc4"]), "--e2"]
        offpath = run("gate_replay.py", "B9_cpu_off", extra)
        onpath = run("gate_replay.py", "B9_cpu_on", [*extra, "--diagnostics"])
        off, on = load(offpath), load(onpath)
        keys = ("prepared_batch", "actor_parameters", "critic_parameters", "model_state",
                "actor_optimizer_state", "critic_optimizer_state", "rng_after_update", "input_batch_after",
                "actor_gradients", "critic_gradients", "permutations")
        comparisons = {key: compare(off[key], on[key], True, key) for key in keys}
        isolation = rng_comparison(on["rng_before_prepare"], on["rng_after_prepare"])
        result = {"passed": all(item["passed"] for item in comparisons.values()) and isolation["passed"],
                  "comparisons": comparisons, "diagnostic_rng_isolation": isolation,
                  "measurement_state_inert": on["measurement_state_inert"],
                  "saved_batch_unchanged": on["update_batch_inert"],
                  "new_diagnostic_metrics": {key: value for key, value in on["stage3_metrics"].items()
                                             if not key.endswith("_seconds")}}
        report["B9"] = result
        save()
        require(result["passed"], "B9 diagnostics changed training or global RNG")
        require(all((EDITED / name).read_bytes() == data for name, data in source_bytes.items()),
                "scientific source changed while gate was running")
        report["scientific_source_unchanged_during_gate"] = True
        report["passed"], report["status"] = True, "passed"
        report.pop("active_phase", None)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
