#!/usr/bin/env python3
"""Produce only the approved two plots and return table after three verified seeds."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
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
