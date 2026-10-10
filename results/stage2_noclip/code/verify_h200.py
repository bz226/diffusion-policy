#!/usr/bin/env python
"""Repeat the approved full-update GPU gate on H200 using the existing saved batch.

No environment is constructed and no new batch or CPU test is run. The scientific
source and numerical settings are untouched. The original CPU gate remains the
evidence for CPU bitwise equality.
"""
import argparse
import csv
import importlib
import io
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import verify_full_update as gate

STUDY = Path(__file__).resolve().parents[1]
ORIGIN = STUDY / "runs/verification_full_update"


def maxima(node):
    """Summarize the existing comparison records without recomputing metrics."""
    absolute, relative, bitwise = [], [], []

    def visit(value):
        if isinstance(value, dict):
            if "max_abs" in value and "max_rel" in value:
                absolute.append(value["max_abs"])
                relative.append(value["max_rel"])
                bitwise.append(value["bitwise_equal"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    gate.require(absolute, "no numerical comparisons were recorded")
    return {"max_abs": max(absolute),
            "max_rel": "inf" if "inf" in relative else max(relative),
            "bitwise_equal": all(bitwise), "comparison_records": len(absolute)}


def actual_gpu(torch):
    """Bind hardware provenance to this Python process's CUDA context."""
    torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    inventory_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id,index",
        "--format=csv,noheader,nounits"], text=True)
    inventory = [dict(zip(("name", "uuid", "driver_version", "pci_bus_id", "index"),
                         [part.strip() for part in row]))
                 for row in csv.reader(io.StringIO(inventory_text)) if row]
    processes_text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
    own_uuids = {row[1].strip() for row in csv.reader(io.StringIO(processes_text))
                 if len(row) == 2 and row[0].strip() == str(os.getpid())}
    gate.require(len(own_uuids) == 1, "cannot uniquely identify this process's GPU UUID")
    selected = [item for item in inventory if item["uuid"] in own_uuids]
    gate.require(len(selected) == 1, "active GPU UUID was missing from hardware inventory")
    return selected[0], inventory_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "runs/verification_h200")
    parser.add_argument("--timeout-seconds", type=int, default=570)
    args = parser.parse_args()
    output = args.output.resolve()
    gate.require(STUDY in output.parents and not output.exists(), "output must be new and inside study")
    gate.require(0 < args.timeout_seconds <= 570, "H200 verification exceeds its approved process cap")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "passed": False, "command": sys.argv,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "timeout_seconds": args.timeout_seconds,
              "allocation_cap_seconds": 600,
              "origin_batch": str(ORIGIN / "training_batch.pt"),
              "origin_capture_config": str(ORIGIN / "capture_config.yaml"),
              "original_gate": str(ORIGIN / "verification.json"),
              "new_environment_transitions": 0,
              "cpu_rerun": False,
              "gpu_tolerance_predeclared": {"atol": gate.ATOL, "rtol": gate.RTOL},
              "relative_error_definition": "abs(edited-reference)/max(abs(reference),float64_tiny)"}

    def save():
        temporary = output / "verification.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "verification.json")

    def timeout(signum, frame):
        raise TimeoutError("H200 saved-batch verification reached its approved process cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(args.timeout_seconds)
    save()
    try:
        repo = STUDY / "dppo"
        command = lambda *parts: subprocess.check_output(["git", "-C", str(repo), *parts], text=True).strip()
        gate.require(command("rev-parse", "HEAD") == gate.EXPECTED and not command("status", "--porcelain"),
                     "source commit changed or worktree is dirty")
        gate.require(set(command("diff", "--name-only", gate.PIN).splitlines()) == {gate.MODEL, gate.AGENT},
                     "unauthorized scientific source files changed")
        report.update(source_commit=gate.EXPECTED, reference_commit=gate.PIN,
                      source_verified_clean_before=True)
        (output / "source.diff").write_text(command("diff", gate.PIN) + "\n")
        original_report = json.loads((ORIGIN / "verification.json").read_text())
        gate.require(original_report["passed"] and original_report["status"] == "passed"
                     and original_report["source_commit"] == gate.EXPECTED
                     and original_report["reference_commit"] == gate.PIN,
                     "original gate is not passed for the required source revisions")
        original_cpu = original_report["cpu"]
        original_cpu_maxima = maxima(original_cpu)
        gate.require(original_cpu["passed"] and original_cpu["required_bitwise"]
                     and original_cpu_maxima["bitwise_equal"], "original CPU bitwise gate did not pass")
        report["cpu_gate_reused"] = {"report": str(ORIGIN / "verification.json"),
                                     "passed": True, **original_cpu_maxima}
        sys.path.insert(0, str(repo))
        import torch
        import numpy as np
        from omegaconf import OmegaConf
        gate.torch, gate.np = torch, np
        gate.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                     "H200 verification requires exactly one visible CUDA GPU")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        gate.require("H200" in report["gpu_name"], "allocated CUDA device is not an H200")
        report["numeric_settings_before"] = gate.numeric_settings()
        report["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
        gate.require(not report["deterministic_algorithms_enabled"], "deterministic algorithms unexpectedly enabled")
        gate.require(report["numeric_settings_before"] == original_report["numeric_settings_before"],
                     "numerical or thread settings differ from the original gate")
        selected_gpu, inventory_text = actual_gpu(torch)
        gate.require(selected_gpu["name"] == report["gpu_name"], "CUDA and NVIDIA inventory GPU names differ")
        report.update(gpu_uuid=selected_gpu["uuid"], driver_version=selected_gpu["driver_version"],
                      active_gpu=selected_gpu, hardware=inventory_text,
                      device_capability=list(torch.cuda.get_device_capability(0)),
                      compiled_cuda_arches=torch.cuda.get_arch_list(),
                      torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        cfg = OmegaConf.load(ORIGIN / "capture_config.yaml")
        model_module = importlib.import_module("model.diffusion.diffusion_ppo")
        agent_module = importlib.import_module("agent.finetune.train_ppo_diffusion_agent")
        old_modules, old_sources = gate.load_reference(repo)
        saved = torch.load(ORIGIN / "training_batch.pt", map_location="cpu", weights_only=False)
        gate.require(saved["source_revision"] == gate.PIN and saved["captured_before_any_optimizer_step"],
                     "saved batch is not the original pristine pre-update capture")
        report["capture"] = dict(original_report["capture"])
        report["capture"]["reused_without_new_sampling"] = True
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        report["active_phase"] = "gpu_full_update"
        save()
        print("H200 GATE START existing saved-batch full-update replay", flush=True)
        report["gpu"] = gate.device_gate(cfg, model_module, agent_module, old_modules,
                                          old_sources, saved, "cuda:0", output / "gpu")
        report["gpu_maximum_differences"] = maxima(report["gpu"])
        save()
        gate.require(report["gpu"]["passed"], "H200 differential gate failed; see comparisons")
        report["numeric_settings_after"] = gate.numeric_settings()
        report["deterministic_settings_unchanged"] = report["numeric_settings_before"] == report["numeric_settings_after"]
        gate.require(report["deterministic_settings_unchanged"], "numerical settings changed")
        gate.require(command("rev-parse", "HEAD") == gate.EXPECTED and not command("status", "--porcelain"),
                     "scientific source changed during verification")
        report["source_verified_clean_after"] = True
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
    print("H200 GATE COMPLETE passed", flush=True)


if __name__ == "__main__":
    main()
