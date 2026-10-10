"""Stage-3 per-run supervision and read-only progress inspection.

The worker owns exactly one child process group. It never calls scancel, signals
other jobs, retries training, or changes scientific overrides.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dppo"
PYTHON = ROOT.parent / "repro_halfcheetah/env/bin/python"
TERMINAL = {"complete", "failed", "skipped"}
SLURM_TERMINAL = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
                  "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}
MODE_OUTPUTS = {"calibrate": "calibration.json", "noise_scale": "noise_scale.json",
                "ckpt_eval": "ckpt_eval.json"}
NONFINITE_TEXT = re.compile(r"(?:^|[\s=:,])(?:nan|[-+]?inf(?:inity)?)(?:$|[\s,|}])", re.I)
SETTINGS_PREFIX = "STAGE3_SETTINGS "


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(str(temporary), str(path))


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def safe_json(value):
    """Preserve nonfinite failure evidence as strings; never replace it with zero."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def command(argv, journal=None, check=True, timeout=60):
    result = subprocess.run([str(x) for x in argv], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    record = {"utc": utc(), "command": [str(x) for x in argv], "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if journal:
        append_json(journal, record)
    if check and result.returncode:
        raise RuntimeError("Command failed: {}: {}".format(argv[0], result.stderr.strip()))
    return record


def validate_source(commit, journal=None):
    if not (SOURCE / ".git").exists():
        raise ValueError("Stage-3 Git metadata is missing; do not discover the parent repository")
    top = command(["git", "-C", SOURCE, "rev-parse", "--show-toplevel"], journal)["stdout"].strip()
    current = command(["git", "-C", SOURCE, "rev-parse", "HEAD"], journal)["stdout"].strip()
    dirty = command(["git", "-C", SOURCE, "status", "--porcelain", "--untracked-files=normal"], journal)["stdout"]
    if Path(top).resolve() != SOURCE or current != commit or dirty:
        raise ValueError("Stage-3 source differs from the passed, committed gate: " + dirty)
    return {"commit": current, "clean": True, "path": str(SOURCE)}


def validate_gate(path, commit):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError("Stage-3 gate must be a local study artifact")
    record = read_json(path)
    if not isinstance(record, dict) or record.get("passed") is not True:
        raise ValueError("Stage-3 full gate has not passed")
    bound_commit = record.get("scientific_source_commit", record.get("stage3_commit", record.get("source_commit")))
    if bound_commit != commit:
        raise ValueError("Passed gate is not bound to the exact Stage-3 commit")
    return {"path": str(path), "passed": True, "commit": commit}


def first_nonfinite(value, prefix=""):
    """None is an explicitly undefined diagnostic, not a numerical failure."""
    if value is None or isinstance(value, (str, bool, int)):
        return None
    if isinstance(value, dict):
        for name, child in value.items():
            bad = first_nonfinite(child, prefix + "." + str(name))
            if bad:
                return bad
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            bad = first_nonfinite(child, prefix + "[{}]".format(index))
            if bad:
                return bad
    elif hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
        # Result entries should be scalar, but inspect accidental numeric arrays.
        import numpy as np
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            return prefix or "<root>"
    else:
        try:
            if not math.isfinite(float(value)):
                return prefix or "<root>"
        except (TypeError, ValueError):
            pass
    return None


def expected_settings(run):
    values = {"actor_loss": "score", "adv_estimator": "gae", "decision_discount": False,
              "gamma_denoising": 0.99, "logprob_reduce": "mean", "norm_adv": True,
              "clamp_logprob": False, "randn_clip_value": 100, "actor_single_step": True,
              "actor_optimizer": "adamw", "lr": 1e-4, "actor_beta1": 0.9,
              "scheduler_min_lr": 1e-4, "scheduler_max_lr": 1e-4, "proj_radius": None,
              "mode": run["mode"], "stage3_diag": True, "ns_warmup_batches": 20,
              "ns_batches": 25, "calib_kl_target": None, "ckpt_eval_repeats": 1,
              "denoised_clip_value": 1.0, "min_sampling_denoising_std": 0.1,
              "min_logprob_denoising_std": 0.1, "max_grad_norm": None}
    for item in run["overrides"]:
        key, raw = item.lstrip("+").split("=", 1)
        name = key.rsplit(".", 1)[-1]
        if key == "train.actor_lr":
            name = "lr"
        elif key == "train.actor_lr_scheduler.min_lr":
            name = "scheduler_min_lr"
        if name not in values:
            continue
        if raw in ("true", "false", "null"):
            value = {"true": True, "false": False, "null": None}[raw]
        else:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        values[name] = value
    values["scheduler_max_lr"] = values["lr"]
    values["optimizer_class"] = "SGD" if values["actor_optimizer"] == "sgd" else "AdamW"
    values["betas"] = None if values["actor_optimizer"] == "sgd" else [values["actor_beta1"], 0.999]
    values["momentum"] = 0 if values["actor_optimizer"] == "sgd" else None
    return values


def validate_settings(run, actual):
    expected = expected_settings(run)
    differences = {key: {"expected": value, "actual": actual.get(key, "<missing>")}
                   for key, value in expected.items() if actual.get(key, "<missing>") != value}
    if differences:
        raise ValueError("Effective startup settings differ from approved flags: " + json.dumps(differences))
    return expected


def observe(run):
    """Read progress without touching sampler, optimizer or source state."""
    out = {"utc": utc(), "run_id": run["run_id"], "mode": run["mode"]}
    logdir = Path(run["logdir"])
    result_path = logdir / "result.pkl"
    if result_path.exists():
        try:
            with result_path.open("rb") as stream:
                rows = pickle.load(stream)
            if not isinstance(rows, list):
                raise ValueError("result.pkl must be a list")
            out["rows"] = len(rows)
            if rows:
                out["last"] = safe_json(rows[-1])
                out["last_itr"] = int(rows[-1]["itr"])
                out["last_step"] = int(rows[-1]["step"])
                evals = [r for r in rows if "eval_episode_reward" in r]
                out["latest_eval"] = safe_json(evals[-1]) if evals else None
                if evals and all(first_nonfinite(r.get("eval_episode_reward")) is None for r in evals):
                    initial = float(evals[0]["eval_episode_reward"])
                    collapses = [r for r in evals if float(r["eval_episode_reward"]) < 0.5 * initial]
                    if collapses:
                        out["reward_collapse"] = {"first_itr": collapses[0]["itr"], "initial": initial,
                                                  "first_return": float(collapses[0]["eval_episode_reward"])}
                for row in rows:
                    bad = first_nonfinite(row)
                    if bad:
                        out["nonfinite"] = {"itr": row.get("itr"), "key": bad,
                                            "row": safe_json(row)}
                        out["failure_iteration"] = row.get("itr")
                        out["failure_iteration_basis"] = "nonfinite value in saved result row"
                        break
        except (EOFError, pickle.UnpicklingError, OSError) as exc:
            out["transient_result_read"] = type(exc).__name__
    for filename in ("stage3_progress.json", "failure.json", MODE_OUTPUTS.get(run["mode"], "")):
        if not filename:
            continue
        path = logdir / filename
        if path.exists():
            try:
                value = read_json(path)
                out[filename] = safe_json(value)
                # Noise-scale nonpositive signal estimates and undefined ratios
                # are legitimate; their status is handled in the scientific JSON.
                if isinstance(value, dict) and value.get("status") in ("failed", "nonfinite", "collapsed"):
                    out["mode_failure"] = {"file": str(path), "status": value["status"]}
                if filename == "failure.json":
                    out["mode_failure"] = {"file": str(path), "failure": safe_json(value)}
                    if isinstance(value, dict) and "itr" in value:
                        out["failure_iteration"] = value["itr"]
                        out["failure_iteration_basis"] = "explicit failure.json record"
            except (ValueError, OSError) as exc:
                out["transient_mode_read"] = type(exc).__name__
    console = Path(run["console_log"])
    if console.exists():
        with console.open("rb") as stream:
            stream.seek(max(0, console.stat().st_size - 100000))
            tail = stream.read().decode("utf-8", errors="replace")
        lines = tail.splitlines()
        out["console_tail"] = lines[-8:]
        traceback_start = None
        for line_index, line in enumerate(lines):
            # Restrict to iteration metrics; avoid harmless error prose, config
            # target_kl=null, or documented undefined diagnostic explanations.
            iteration_line = re.search(r"\b(\d+): step (\d+)\b", line)
            if iteration_line:
                out["last_iteration_log_line"] = line
                out["last_logged_iteration"] = int(iteration_line.group(1))
                if NONFINITE_TEXT.search(line):
                    out["nonfinite_log_line"] = line
                    out.setdefault("failure_iteration", int(iteration_line.group(1)))
                    out.setdefault("failure_iteration_basis", "nonfinite value in iteration log line")
            if "Traceback (most recent call last)" in line:
                out["traceback_seen"] = True
                traceback_start = line_index
            if "FloatingPointError:" in line and re.search(r"non[- ]?finite|\bnan\b|\binf\b", line, re.I):
                out["nonfinite_exception"] = line
            if "STAGE3_METRICS " in line:
                try:
                    out["last_stage3_metrics"] = safe_json(json.loads(line.split("STAGE3_METRICS ", 1)[1]))
                except ValueError:
                    out["last_stage3_metrics_line"] = line
        if traceback_start is not None:
            out["traceback_tail"] = lines[traceback_start:][-100:]
            if "failure_iteration" not in out and run["mode"] == "train" and "last_itr" in out:
                out["failure_iteration_candidate"] = out["last_itr"] + 1
                out["failure_iteration_candidate_basis"] = "inferred next iteration after saved prefix; crash prevented an explicit iteration record"
        # Startup settings can be far outside the tail late in a run.
        with console.open("rb") as stream:
            startup_lines = stream.read(100000).decode("utf-8", errors="replace").splitlines()
        for line in startup_lines:
            if SETTINGS_PREFIX in line:
                payload = line.split(SETTINGS_PREFIX, 1)[1]
                try:
                    out["effective_settings"] = json.loads(payload)
                except ValueError:
                    out["startup_settings_parse_error"] = payload
    return out


def verify_complete(run, observation):
    logdir = Path(run["logdir"])
    if run["mode"] == "train":
        with (logdir / "result.pkl").open("rb") as stream:
            rows = pickle.load(stream)
        if len(rows) != 140:
            raise ValueError("Expected 140 iterations, got " + str(len(rows)))
        n_steps = 2000 if run["condition"] == "BATCH4" else 500
        for index, row in enumerate(rows):
            expected = (index - index // 10) * n_steps * 40 * 4
            if row.get("itr") != index or row.get("step") != expected:
                raise ValueError("Iteration/step grid mismatch at " + str(index))
            reward = "eval_episode_reward" if index % 10 == 0 else "train_episode_reward"
            if reward not in row or first_nonfinite(row):
                raise ValueError("Missing return or nonfinite metric at " + str(index))
        checkpoints = [logdir / "checkpoint" / ("state_{}.pt".format(i)) for i in (0, 35, 70, 105, 139)]
        if not all(p.is_file() and p.stat().st_size > 0 for p in checkpoints):
            raise ValueError("Missing required checkpoint")
        import torch
        for path in checkpoints:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            if state.get("itr") != int(path.stem.split("_")[-1]):
                raise ValueError("Checkpoint iteration mismatch: " + str(path))
            if any(not torch.isfinite(t).all().item() for t in state["model"].values() if torch.is_tensor(t)):
                raise ValueError("Nonfinite checkpoint tensor: " + str(path))
        return {"verified_rows": len(rows), "final_step": rows[-1]["step"],
                "checkpoints": list(map(str, checkpoints)), "checkpoint_tensors_finite": True}
    path = logdir / MODE_OUTPUTS[run["mode"]]
    value = read_json(path)
    if not isinstance(value, dict) or value.get("status") not in ("complete", "completed"):
        raise ValueError("Missing or failed mode output: " + str(path))
    if first_nonfinite(value):
        raise ValueError("Nonfinite mode output: " + str(path))
    if run["mode"] == "calibrate":
        eta = value.get("eta_star")
        if not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta <= 0:
            raise ValueError("Calibration did not produce finite positive eta_star")
        if len(value.get("batches", [])) != 5 or value.get("kl_target") != expected_settings(run)["calib_kl_target"]:
            raise ValueError("Calibration must contain five fresh batches at the prescribed KL target")
    if run["mode"] == "noise_scale":
        if not (logdir / "SUMMARY.md").is_file() or not list(logdir.glob("*.csv")):
            raise ValueError("Noise-scale SUMMARY.md or CSV missing")
        expected = expected_settings(run)
        if (value.get("measurement_batches") != 25 or value.get("n_env_rollouts") != 1000 or
                value.get("warmup_batches") != expected["ns_warmup_batches"] or
                len(value.get("variants", [])) != 5 or len(value.get("measurements", [])) != 25):
            raise ValueError("Noise-scale sample counts differ from the approved protocol")
    if run["mode"] == "ckpt_eval":
        repeats = expected_settings(run)["ckpt_eval_repeats"]
        if value.get("repeats_per_sampler") != repeats or len(value.get("rollouts", [])) != 2 * repeats:
            raise ValueError("Checkpoint-evaluation rollout count differs from the protocol")
        if set(value.get("samplers", {})) != {"train", "eval"}:
            raise ValueError("Both checkpoint-evaluation samplers are required")
        for sampler in value["samplers"].values():
            if sampler.get("J_disc", {}).get("n") != 80 * repeats:
                raise ValueError("Checkpoint evaluation must contain 80 episodes per rollout")
    return {"verified_mode_output": str(path)}


def stop_own_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10)
        return
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def worker(spec_path, commit, gate):
    run = read_json(spec_path)
    if not isinstance(run, dict):
        raise ValueError("Missing run specification")
    logdir = Path(run["logdir"]).resolve()
    if logdir != ROOT / run["run_id"]:
        raise ValueError("Worker output must be the approved absolute run directory")
    record_dir = ROOT / "runs" / run["run_id"]
    record_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = record_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("A worker manifest already exists; automatic retries are forbidden")
    journal = record_dir / "commands.jsonl"
    start_clock = time.monotonic()
    manifest = dict(run, status="starting", started_utc=utc(), source_commit=commit,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), monitor_status="active",
                    source_gate_path=str(Path(gate).resolve()), spec_path=str(Path(spec_path).resolve()))
    write_json(manifest_path, manifest)
    child = None
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    try:
        manifest["gate"] = validate_gate(gate, commit)
        manifest["source_before"] = validate_source(commit, journal)
        if not os.environ.get("SLURM_JOB_ID", "").isdigit():
            raise ValueError("GPU worker requires its own Slurm allocation")
        gpu = command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], journal)
        gpu_lines = gpu["stdout"].strip().splitlines()
        if len(gpu_lines) != 1 or "RTX A6000" not in gpu_lines[0]:
            raise ValueError("Exactly one RTX A6000 is required")
        if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 40:
            raise ValueError("Forty allocated CPUs are required")
        if len(os.sched_getaffinity(0)) < 40:
            raise ValueError("CPU affinity is smaller than the approved 40 CPUs")
        if any(os.environ.get(k) is not None for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
            raise ValueError("Stage-1/2 thread variables must remain unset")
        manifest["hardware"] = {"gpu": gpu["stdout"].strip(), "cpu_affinity_count": len(os.sched_getaffinity(0)),
                                 "slurm_node": os.environ.get("SLURM_JOB_NODELIST")}
        manifest["environment"] = {k: os.environ.get(k) for k in ("DPPO_DATA_DIR", "DPPO_LOG_DIR", "PYTHONPATH",
              "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        argv = [str(PYTHON), "-B", str(SOURCE / "script/run.py"),
                "--config-dir=" + str(SOURCE / "cfg/gym/finetune/halfcheetah-v2"),
                "--config-name=ft_ppo_diffusion_mlp", "train.n_train_itr=140", "train.save_model_freq=35",
                "wandb=null", "seed=" + str(run["seed"]), "logdir=" + str(logdir)] + run["overrides"]
        manifest["command"] = argv
        append_json(journal, {"utc": utc(), "command": argv, "cwd": str(SOURCE), "action": "launch_own_child"})
        with Path(run["console_log"]).open("xb") as stream:
            child = subprocess.Popen(argv, cwd=str(SOURCE), stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        manifest.update(status="running", child_pid=child.pid, subprocess_started_utc=utc())
        write_json(manifest_path, manifest)
        next_record = time.monotonic()
        last_observation = {}
        prior_counter = None
        stagnant_checks = 0
        while child.poll() is None:
            if interrupted:
                raise RuntimeError("Worker received signal " + str(interrupted[0]))
            if time.monotonic() - start_clock >= run["cap_seconds"] - 30:
                raise RuntimeError("Approved per-run cap reached; no extension or retry")
            observation = observe(run)
            last_observation = observation
            if (observation.get("nonfinite") or observation.get("nonfinite_log_line") or
                    observation.get("nonfinite_exception") or observation.get("mode_failure")):
                manifest["failure_observation"] = observation
                manifest["collapse"] = True
                raise RuntimeError("Nonfinite scientific output or explicit mode failure")
            if observation.get("effective_settings"):
                validate_settings(run, observation["effective_settings"])
                manifest["effective_settings"] = observation["effective_settings"]
            if time.monotonic() >= next_record:
                mode_progress = observation.get("stage3_progress.json", {})
                counter = (observation.get("last_itr"), mode_progress.get("substage"), mode_progress.get("done"),
                           mode_progress.get("environments_completed"))
                stagnant_checks = stagnant_checks + 1 if counter == prior_counter else 0
                prior_counter = counter
                observation["consecutive_scheduled_checks_without_progress"] = stagnant_checks
                if stagnant_checks >= 2:
                    observation["suspected_stall"] = True
                observation["gpu_observation"] = command(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                    journal, check=False)
                append_json(record_dir / "monitor_history.jsonl", observation)
                write_json(record_dir / "progress.json", observation)
                next_record = time.monotonic() + run["monitor_interval_seconds"]
            time.sleep(15)
        manifest["returncode"] = child.returncode
        final = observe(run)
        manifest["last_observation"] = final
        if final.get("effective_settings"):
            validate_settings(run, final["effective_settings"])
            manifest["effective_settings"] = final["effective_settings"]
        if final.get("reward_collapse"):
            manifest["collapse"] = True
            manifest["reward_collapse"] = final["reward_collapse"]
        append_json(record_dir / "monitor_history.jsonl", final)
        write_json(record_dir / "progress.json", final)
        if child.returncode:
            raise RuntimeError("Training process exited with code " + str(child.returncode))
        if "effective_settings" not in manifest:
            raise ValueError("Required startup effective-settings line was not recorded")
        if (final.get("nonfinite") or final.get("nonfinite_log_line") or
                final.get("nonfinite_exception") or final.get("mode_failure")):
            manifest["collapse"] = True
            raise RuntimeError("Run ended with nonfinite scientific output")
        manifest.update(verify_complete(run, final))
        manifest["source_after"] = validate_source(commit, journal)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", failure_reason=str(exc), failure_type=type(exc).__name__)
        if child:
            stop_own_child(child)
            manifest["returncode"] = child.returncode
        try:
            manifest["last_observation"] = observe(run)
            failure = manifest["last_observation"]
            if (failure.get("nonfinite") or failure.get("nonfinite_log_line") or
                    failure.get("nonfinite_exception") or failure.get("mode_failure") or failure.get("reward_collapse")):
                manifest["collapse"] = True
        except Exception as observe_exc:
            manifest["last_observation_error"] = str(observe_exc)
    finally:
        manifest.update(ended_utc=utc(), wall_seconds=time.monotonic() - start_clock, monitor_status="paused")
        write_json(manifest_path, safe_json(manifest))
    print(json.dumps({"run_id": run["run_id"], "status": manifest["status"],
                      "failure_reason": manifest.get("failure_reason")}), flush=True)
    return 0 if manifest["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("worker")
    run.add_argument("--spec", required=True)
    run.add_argument("--commit", required=True)
    run.add_argument("--gate", required=True)
    check = sub.add_parser("inspect")
    check.add_argument("--spec", required=True)
    args = parser.parse_args()
    if args.action == "worker":
        return worker(args.spec, args.commit, args.gate)
    print(json.dumps(observe(read_json(args.spec)), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
