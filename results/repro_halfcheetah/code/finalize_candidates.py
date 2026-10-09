#!/usr/bin/env python3
"""Produce only the approved two plots and return table after three verified seeds."""
import json
import os
import datetime as dt
import time
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parallel_path = root / "provenance/parallel_seed2.json"
if parallel_path.exists():
    parallel = json.loads(parallel_path.read_text())
    job = os.environ.get("SLURM_JOB_ID")
    if parallel.get("status") != "assigned":
        raise RuntimeError("Parallel seed 2 assignment is not ready")
    if job == str(parallel["original_job_id"]):
        print("Candidate analysis is assigned to parallel job %s." % parallel["parallel_job_id"])
        sys.exit(0)
    if job != str(parallel["parallel_job_id"]):
        raise RuntimeError("Only the assigned parallel job may finalize candidates")
    # Usually seed 1 finishes first. Respect its existing deadline if it does not.
    while True:
        first = json.loads((root / "runs/seed1/manifest.json").read_text())
        if first.get("status") == "complete":
            break
        if first.get("status") not in ("running", "starting"):
            raise RuntimeError("Seed 1 did not complete successfully")
        deadline = dt.datetime.fromisoformat(first["deadline_utc"]).timestamp()
        if time.time() >= deadline:
            raise RuntimeError("Seed 1 has not completed within its approved deadline")
        time.sleep(1)
command = [sys.executable, str(root / "code/analyze_results.py")]
for seed, run_id in enumerate(("seed0_handoff", "seed1", "seed2")):
    manifest = json.loads((root / "runs" / run_id / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("verified_rows") != 140
            or manifest.get("checkpoint_tensors_finite") is not True
            or manifest.get("final_training_env_steps") != 10080000):
        raise RuntimeError("Seed %d has not passed full verification" % seed)
    command += ["--seed%d" % seed, manifest["result_path"]]
command += ["--output-dir", str(root / "runs/three_seed_analysis/candidates")]
subprocess.run([sys.executable, str(root / "code/run_logged.py"),
                "--label", "approved_three_seed_analysis", "--timeout", "180",
                "--cwd", str(root), "--"] + command, check=True)
