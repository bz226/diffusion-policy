#!/usr/bin/env python3
"""Bound one approved Stage 2 process and preserve its native outputs."""
import argparse
import copy
import datetime as dt
import json
import math
import numbers
import os
import pickle
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "cc7234ad7ff39a8f32de3af903606723a16f0648"
CONFIG = "cfg/gym/finetune/halfcheetah-v2"
DIAGNOSTICS = ("approx_kl", "clipfrac", "actor_optimizer_steps", "critic_optimizer_steps",
               "actor_lr", "critic_lr", "kl_true_per_action", "logratio_p99", "clamp_hit_frac")
ENV_KEYS = ("STUDY_ROOT", "DPPO_DATA_DIR", "DPPO_LOG_DIR", "PYTHONPATH", "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_GPUS", "SLURM_MEM_PER_NODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH",
            "TRITON_CACHE_DIR", "D4RL_DATASET_DIR", "PYTHONDONTWRITEBYTECODE")
NONFINITE = re.compile(r"(?<![\w./])[-+]?(?:nan|inf(?:inity)?)(?![\w./])", re.I)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def utc(timestamp=None):
    return dt.datetime.fromtimestamp(time.time() if timestamp is None else timestamp,
                                     dt.timezone.utc).isoformat(timespec="seconds")


def save(path, value):
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("Missing or nonnumeric " + label)
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite value: " + label)
    return float(value)


def finite_tree(value, label, np, torch=None):
    if torch is not None and torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            raise ValueError("Nonfinite tensor: " + label)
    elif isinstance(value, numbers.Number):
        if not bool(np.isfinite(value)):
            raise ValueError("Nonfinite value: " + label)
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("Nonfinite array: " + label)
        if value.dtype == object:
            for index, item in enumerate(value.flat):
                finite_tree(item, label + "[%d]" % index, np, torch)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, label + "." + str(key), np, torch)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, label + "[%d]" % index, np, torch)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run-id", "expected-commit"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-seconds", type=int, required=True)
    p.add_argument("--estimate-seconds", type=float, required=True)
    p.add_argument("--fixed-batch-gate")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    match = re.fullmatch(r"(P1|P2|R0_rerun|NC[123]|NC4_lr(?:1e-4|1e-3|3e-3))_seed([012])", a.run_id)
    if not match or (match.group(1) in ("P1", "P2", "R0_rerun") and match.group(2) != "0"):
        p.error("Unapproved run ID")
    a.condition, a.seed = match.group(1), int(match.group(2))
    if a.condition == "R0_rerun":
        a.condition = "R0"
    a.probe = a.condition in ("P1", "P2")
    if not a.probe and not a.fixed_batch_gate:
        p.error("Full runs require the passed revised fixed-batch gate")
    a.rows = 20 if a.probe else 140
    if not re.fullmatch(r"[0-9a-f]{40}", a.expected_commit):
        p.error("Supply the full expected commit")
    if a.probe and a.expected_commit != BASE:
        p.error("Probes require pristine pinned source")
    if not 0 < a.timeout_seconds <= (1800 if a.probe else 10800):
        p.error("Timeout exceeds the approved cap")
    if not math.isfinite(a.estimate_seconds) or a.estimate_seconds <= 0:
        p.error("Runtime estimate must be finite and positive")
    if a.command[:1] == ["--"]:
        a.command = a.command[1:]
    return a


class Supervisor:
    def __init__(self, a):
        self.a, self.repo, self.run_dir = a, ROOT / "dppo", ROOT / "runs" / a.run_id
        self.native, self.console = self.run_dir / "native", ROOT / (a.run_id + ".out")
        self.started, self.clock = time.time(), time.monotonic()
        self.deadline = self.clock + a.timeout_seconds
        self.interval = math.ceil(max(1800, a.estimate_seconds / 6))
        self.next_monitor = self.clock + self.interval
        self.process, self.interrupted, self.records = None, None, []
        self.last_signature, self.bad_pickle_since, self.last_progress = None, None, 0
        self.last_monitor_signature, self.stalls = None, 0
        self.checking_itr, self.collection_itr = None, -1
        self.trajectory_observer = None
        self.saved_valid_rows = 0
        self.observed_nonfinite, self.nonfinite_iteration = False, None
        self.pending_nonfinite = None
        self.progress = {"run_id": a.run_id, "status": "starting", "completed_rows": 0,
                         "training_env_steps": 0, "bytes": 0, "last_line": None,
                         "last_diagnostic_line": None, "last_diagnostics": None}
        if Path.cwd().resolve() != self.repo or Path(os.environ.get("STUDY_ROOT", "")).resolve() != ROOT:
            raise ValueError("Use the Stage 2 environment and source directory")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.native.exists() or self.console.exists():
            raise ValueError("Refusing an existing native directory or root console")
        self.manifest = {"run_id": a.run_id, "condition": a.condition, "seed": a.seed,
                         "status": "starting", "start_utc": utc(self.started),
                         "deadline_utc": utc(self.started + a.timeout_seconds),
                         "timeout_seconds": a.timeout_seconds, "estimate_seconds": a.estimate_seconds,
                         "monitor_interval_seconds": self.interval, "monitor_status": "scheduled",
                         "expected_rows": a.rows, "expected_commit": a.expected_commit,
                         "command": a.command, "command_shell_display": shlex.join(a.command),
                         "cwd": str(self.repo), "logdir": str(self.native),
                         "result_path": str(self.native / "result.pkl"), "console_log": str(self.console),
                         "environment": {k: os.environ.get(k) for k in ENV_KEYS},
                         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "verification_commands": []}
        # Exclusive creation is also the launch reservation if two dispatchers race.
        with (self.run_dir / "manifest.json").open("x") as stream:
            json.dump(self.manifest, stream, indent=2)

    def checked_command(self, command):
        r = subprocess.run(command, cwd=str(self.repo), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=30)
        self.manifest["verification_commands"].append({"command": command, "returncode": r.returncode,
                                                       "output": r.stdout})
        if r.returncode:
            raise ValueError("Verification command failed: " + shlex.join(command) + "\n" + r.stdout)
        return r.stdout.strip()

    def source(self):
        commit = self.checked_command(["git", "rev-parse", "HEAD"])
        if commit != self.a.expected_commit:
            raise ValueError("Unexpected source commit")
        self.checked_command(["git", "diff", "--exit-code", "HEAD", "--"])
        changed = self.checked_command(["git", "diff", "--name-only", BASE, "HEAD", "--"]).splitlines()
        allowed = {"model/diffusion/diffusion_ppo.py", "agent/finetune/train_ppo_diffusion_agent.py"}
        if set(changed) - allowed or (self.a.probe and changed):
            raise ValueError("Unexpected source/config modifications")
        return {"commit": commit, "tracked_files_clean": True, "changed_from_base": changed}

    def validate_gate(self):
        if not self.a.probe:
            from fixed_batch_gate import validate_gate
            self.manifest["fixed_batch_gate"] = validate_gate(self.a.fixed_batch_gate, self.a.expected_commit)

    def preflight(self):
        from preflight import inspect
        cmd = self.a.command
        position = 2 if len(cmd) > 1 and cmd[1] == "-u" else 1
        if len(cmd) <= position + 2 or cmd[position:position + 3] != [
                "script/run.py", "--config-dir=" + CONFIG, "--config-name=ft_ppo_diffusion_mlp"]:
            raise ValueError("Use the prescribed script and config arguments")
        import shutil
        if Path(shutil.which(cmd[0]) or cmd[0]).resolve() != Path(sys.executable).resolve():
            raise ValueError("Training must use the shared Stage 1 interpreter")
        overrides = cmd[position + 3:]
        expected = {"train.n_train_itr": self.a.rows, "train.save_model_freq": 35,
                    "seed": self.a.seed, "wandb": None, "logdir": str(self.native)}
        c = self.a.condition
        if c.startswith("NC"):
            expected.update({"model.clip_ploss_coef": 1e6, "model.clip_ploss_coef_base": 1e6,
                             "train.target_kl": None})
        if c in ("NC2", "NC3") or c.startswith("NC4"):
            expected.update({"model.clamp_logprob": False, "model.randn_clip_value": 100})
        if c == "NC3":
            expected["model.logprob_reduce"] = "sum"
        if c.startswith("NC4"):
            lr = float(c.split("lr", 1)[1])
            expected.update({"train.actor_single_step": True, "train.actor_lr": lr,
                             "train.actor_lr_scheduler.min_lr": lr})
        supplied = [token.split("=", 1)[0].lstrip("+") for token in overrides]
        if len(supplied) != len(set(supplied)) or set(supplied) != set(expected):
            raise ValueError("CLI override keys differ from the approved condition")
        checked = inspect(overrides)
        cfg = checked["resolved_configuration"]
        baseline = json.loads((ROOT.parent / "repro_halfcheetah/runs/seed0_handoff/config.json").read_text())
        wanted = copy.deepcopy(baseline["resolved_configuration"])
        for key, value in expected.items():
            node, parts = wanted, key.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        if cfg != wanted:
            raise ValueError("Resolved configuration differs from Stage 1 beyond approved overrides")
        if checked["source_commit"] != self.a.expected_commit:
            raise ValueError("Import preflight source revision changed")
        save(self.run_dir / "preflight.json", checked)
        save(self.run_dir / "config.json", {"resolved_configuration": cfg, "overrides": overrides})
        self.cfg = cfg
        if os.environ.get("SLURM_CPUS_PER_TASK") != "40" or len(os.sched_getaffinity(0)) < 40:
            raise ValueError("Each run requires an allocation of 40 CPUs")
        gpu = self.checked_command(["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
                                    "--format=csv,noheader"])
        from hardware_exception import validate_hardware
        exception = validate_hardware(gpu, self.a, os.environ, ROOT)
        if exception is not None:
            self.manifest["hardware_exception"] = exception
        self.manifest["hardware"] = {"gpu_query": gpu, "uname": list(os.uname()),
                                     "cpu_affinity_count": len(os.sched_getaffinity(0))}

    def read_results(self, final=False):
        import numpy as np
        path = self.native / "result.pkl"
        if not path.exists():
            if final:
                raise ValueError("Native result.pkl is missing")
            return
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self.last_signature and not final and self.bad_pickle_since is None:
            return
        try:
            with path.open("rb") as stream:
                rows = pickle.load(stream)
        except (EOFError, pickle.UnpicklingError, OSError) as error:
            self.bad_pickle_since = self.bad_pickle_since or time.monotonic()
            if final or time.monotonic() - self.bad_pickle_since > 10:
                raise ValueError("Native result file remains unreadable: " + str(error))
            return  # DPPO writes result.pkl in place; wait for its write to finish.
        self.bad_pickle_since, self.last_signature = None, signature
        if not isinstance(rows, list) or not len(self.records) <= len(rows) <= self.a.rows:
            raise ValueError("Unexpected native result count")
        for itr, row in enumerate(rows):
            self.checking_itr = itr
            finite_tree(row, "itr%d" % itr, np)
            if (not isinstance(row, dict) or isinstance(row.get("itr"), bool)
                    or not isinstance(row.get("itr"), numbers.Integral) or row["itr"] != itr
                    or isinstance(row.get("step"), bool) or not isinstance(row.get("step"), numbers.Integral)
                    or row["step"] != (itr - itr // 10) * 80000):
                raise ValueError("Invalid native iteration/step schedule")
            metric = "eval_episode_reward" if itr % 10 == 0 else "train_episode_reward"
            other = "train_episode_reward" if itr % 10 == 0 else "eval_episode_reward"
            scalar(row.get(metric), "itr%d.%s" % (itr, metric))
            if other in row or scalar(row.get("time"), "iteration time") <= 0:
                raise ValueError("Unexpected native metric schedule or duration")
            if not self.a.probe and itr % 10:
                for key in DIAGNOSTICS:
                    scalar(row.get(key), "itr%d.%s" % (itr, key))
                if self.a.condition.startswith("NC4"):
                    if (scalar(row.get("actor_step_ratio"), "actor_step_ratio") != 1.0
                            or scalar(row.get("actor_step_clipfrac"), "actor_step_clipfrac") != 0.0):
                        raise ValueError("NC4 literal ratio=1 and clip fraction=0 invariant failed")
                    if row["actor_optimizer_steps"] != 1 or row["critic_optimizer_steps"] != 20:
                        raise ValueError("NC4 optimizer step count invariant failed")
        self.records = rows
        self.checking_itr = None
        if len(rows) > self.saved_valid_rows:
            directory = self.run_dir / "observed_results"
            directory.mkdir(exist_ok=True)
            path = directory / ("itr_%03d.pkl" % rows[-1]["itr"])
            with path.open("xb") as stream:
                pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
            self.manifest.update({"last_valid_result_path": str(path), "last_valid_result_rows": len(rows),
                                  "last_valid_result_note": "External snapshot of the validated finite native prefix; never replaces raw failure evidence."})
            self.saved_valid_rows = len(rows)
        if rows:
            self.progress.update({"completed_rows": len(rows), "last_completed_itr": rows[-1]["itr"],
                                  "training_env_steps": rows[-1]["step"]})
            training = [row for row in rows if row["itr"] % 10]
            if training:
                self.progress["last_diagnostics"] = {key: training[-1][key] for key in DIAGNOSTICS
                                                       if key in training[-1]}

    def write_progress(self, force=False):
        now = time.monotonic()
        if force or now - self.last_progress >= 1:
            self.progress.update({"updated_utc": utc(), "elapsed_seconds": now - self.clock})
            save(self.run_dir / "progress.json", self.progress)
            self.last_progress = now

    def monitor(self, event):
        signature = (self.progress["bytes"], self.progress["completed_rows"])
        if event == "scheduled":
            self.stalls = self.stalls + 1 if signature == self.last_monitor_signature else 0
        self.last_monitor_signature = signature
        self.write_progress(True)
        with (self.run_dir / "monitor_history.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(self.progress, event=event, consecutive_stalled_checks=self.stalls)) + "\n")
        save(self.run_dir / "manifest.json", self.manifest)
        if self.stalls >= 2 and event != "terminal":
            raise ValueError("Two scheduled checks without progress")

    def stop_group(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)

    def line(self, raw, terminated=True):
        text = ANSI.sub("", raw.decode("utf-8", errors="replace")).strip()
        self.progress["last_line"] = text[-2000:]
        if "stage2_flags " in text:
            c = self.a.condition
            expected_flags = {"clamp_logprob": not (c in ("NC2", "NC3") or c.startswith("NC4")),
                              "logprob_reduce": "sum" if c == "NC3" else "mean",
                              "actor_single_step": c.startswith("NC4")}
            expected_text = "stage2_flags " + " ".join(key + "=" + str(value) for key, value in expected_flags.items())
            if text.split("stage2_flags ", 1)[1] != expected_text.split("stage2_flags ", 1)[1]:
                self.manifest["startup_flag_validation_error"] = text
                raise ValueError("Startup flag printout differs from approved condition")
            self.manifest["effective_startup_flags"] = expected_flags
        if re.search(r"\bProcessed step 0 of \d+", text):
            self.collection_itr += 1
            self.progress["collection_iteration"] = self.collection_itr
        if any(key in text for key in DIAGNOSTICS):
            self.progress["last_diagnostic_line"] = text[-4000:]
        nonfinite = NONFINITE.search(text)
        summary = re.search(r"(?:^| - )(\d+): step\s+\d+\s*\|", text)
        if nonfinite:
            self.observed_nonfinite = True
            if self.nonfinite_iteration is None:
                self.nonfinite_iteration = int(summary.group(1)) if summary else (
                    self.collection_itr if self.collection_itr >= 0 else None)
            self.progress["last_nonfinite_line"] = text[-4000:]
        if nonfinite:
            message = "Printed NaN/Inf: " + text[-800:]
            if summary and any(key in text for key in ("kl_true_per_action", "logratio_p99", "clamp_hit_frac")):
                # The trainer writes this diagnostic row immediately after its summary.
                # Give that native evidence at most two seconds to reach result.pkl.
                if self.pending_nonfinite is None:
                    self.pending_nonfinite = {"iteration": int(summary.group(1)), "message": message,
                                              "deadline": time.monotonic() + 2, "line": text[-4000:]}
                return
            raise ValueError(message)

    def pending_nonfinite_row_written(self):
        try:
            with (self.native / "result.pkl").open("rb") as stream:
                rows = pickle.load(stream)
        except (OSError, EOFError, pickle.UnpicklingError):
            return False
        itr = self.pending_nonfinite["iteration"]
        written = isinstance(rows, list) and len(rows) > itr and isinstance(rows[itr], dict) and rows[itr].get("itr") == itr
        if written:
            self.manifest["nonfinite_result_row_observed"] = True
        return written

    def capture(self):
        pending, leader_exit = b"", None
        with self.console.open("xb", buffering=0) as output:
            process_start = time.monotonic()
            self.manifest["subprocess_start_utc"] = utc()
            self.process = subprocess.Popen(self.a.command, cwd=str(self.repo), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
            self.manifest.update({"status": "running", "pid": self.process.pid, "monitor_status": "running"})
            self.progress["status"] = "running"
            selector = selectors.DefaultSelector()
            selector.register(self.process.stdout, selectors.EVENT_READ)
            failure, failure_clock = None, None
            try:
                self.monitor("start")
                while selector.get_map():
                    if failure_clock is not None and time.monotonic() - failure_clock >= 5:
                        break  # A detached descendant must not retain this supervisor forever.
                    try:
                        now, code = time.monotonic(), self.process.poll()
                        if code is not None and leader_exit is None:
                            leader_exit = now
                        if self.interrupted:
                            raise ValueError("Received " + self.interrupted)
                        if code not in (None, 0):
                            raise ValueError("DPPO exited with status %s" % code)
                        if now >= self.deadline:
                            raise ValueError("Approved wall-clock cap reached")
                        if leader_exit is not None and now - leader_exit > 10:
                            raise ValueError("Exited process left its output pipe open")
                        if failure is None:
                            if self.pending_nonfinite is not None:
                                if (self.pending_nonfinite_row_written() or code is not None
                                        or now >= self.pending_nonfinite["deadline"]):
                                    raise ValueError(self.pending_nonfinite["message"])
                            else:
                                self.read_results()
                                self.observe_trajectory()
                            if now >= self.next_monitor:
                                self.monitor("scheduled")
                                self.next_monitor += self.interval
                    except Exception as error:
                        if failure is None:
                            failure = error
                            self.stop_group()
                            failure_clock = time.monotonic()
                    for key, _ in selector.select(0.5):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        output.write(chunk)  # Preserve raw output before any scientific gate.
                        self.progress["bytes"] += len(chunk)
                        pending += chunk
                        while b"\n" in pending:
                            raw, pending = pending.split(b"\n", 1)
                            try:
                                self.line(raw)
                            except Exception as error:
                                if failure is None:
                                    failure = error
                                    self.stop_group()
                                    failure_clock = time.monotonic()
                        self.write_progress()
                if pending:
                    try:
                        self.line(pending, terminated=False)
                    except Exception as error:
                        if failure is None:
                            failure = error
                            self.stop_group()
                code = self.process.wait(timeout=10)
                self.manifest.update({"returncode": code, "subprocess_end_utc": utc(),
                                      "subprocess_wall_seconds": time.monotonic() - process_start})
                if failure is not None:
                    raise failure
                if self.pending_nonfinite is not None:
                    self.pending_nonfinite_row_written()
                    raise ValueError(self.pending_nonfinite["message"])
                if code != 0:
                    raise ValueError("DPPO exited with status %s" % code)
            finally:
                selector.close()

    def verify(self):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        self.read_results(final=True)
        if len(self.records) != self.a.rows:
            raise ValueError("Incomplete native results")
        if not self.a.probe and "effective_startup_flags" not in self.manifest:
            raise ValueError("Missing effective startup flag printout")
        self.observe_trajectory(final=True)
        cfg = OmegaConf.load(self.native / ".hydra/config.yaml")
        if OmegaConf.to_container(cfg, resolve=True) != self.cfg:
            raise ValueError("Native saved configuration differs from preflight")
        checkpoints = []
        for itr in ((0, 19) if self.a.probe else (0, 35, 70, 105, 139)):
            path = self.native / "checkpoint" / ("state_%d.pt" % itr)
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("itr") != itr or "model" not in payload:
                raise ValueError("Invalid checkpoint: " + str(path))
            finite_tree(payload, str(path), np, torch)
            checkpoints.append(str(path))
        self.manifest.update({"verified_rows": len(self.records), "checkpoint_tensors_finite": True,
                              "checkpoints": checkpoints, "final_training_env_steps": self.records[-1]["step"],
                              "native_iteration_time_mean_seconds": float(np.mean([r["time"] for r in self.records]))})

    def observe_trajectory(self, final=False):
        if self.trajectory_observer is None:
            return
        flags = self.trajectory_observer.check(self.records, final=final)
        self.manifest["trajectory_red_flags"] = self.trajectory_observer.report["red_flags"]
        for flag in flags:
            print("R0 trajectory red flag (record only, continuing): " + json.dumps(flag), flush=True)

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: setattr(self, "interrupted", signal.Signals(signum).name))
        try:
            self.manifest["execution_phase"] = "startup_validation"
            self.manifest["source_before"] = self.source()
            self.validate_gate()
            self.preflight()
            if self.a.condition == "R0":
                from trajectory_observer import TrajectoryObserver
                self.trajectory_observer = TrajectoryObserver(ROOT, self.run_dir)
                self.manifest["trajectory_observation_path"] = str(self.trajectory_observer.status_path)
            if self.interrupted or time.monotonic() >= self.deadline:
                raise ValueError("Interrupted or timed out before launch")
            self.manifest["execution_phase"] = "training"
            self.capture()
            self.manifest["execution_phase"] = "terminal_verification"
            self.progress["status"] = "verifying"
            self.write_progress(True)
            self.verify()
            self.manifest["source_after"] = self.source()
            if self.interrupted or time.monotonic() >= self.deadline:
                raise ValueError("Interrupted or timed out during verification")
            self.manifest["status"] = self.progress["status"] = "complete"
        except Exception as error:
            self.stop_group()
            self.manifest.update({"status": "failed", "failure_reason": str(error),
                                  "returncode": self.process.poll() if self.process else None,
                                  "failing_iteration": self.nonfinite_iteration if self.nonfinite_iteration is not None else
                                                       self.checking_itr if self.checking_itr is not None else
                                                       self.collection_itr if self.collection_itr >= 0 else None,
                                  "nonfinite_failure": self.observed_nonfinite or str(error).startswith(("Nonfinite", "Printed NaN/Inf"))})
            self.progress["status"] = "failed"
            try:
                self.manifest["source_after"] = self.source()
            except Exception as source_error:
                self.manifest["source_after_error"] = str(source_error)
        finally:
            if self.trajectory_observer is not None:
                self.observe_trajectory(final=True)
            self.manifest.update({"end_utc": utc(), "wall_seconds_including_verification": time.monotonic() - self.clock,
                                  "monitor_status": "paused", "last_diagnostics": self.progress["last_diagnostics"],
                                  "last_diagnostic_line": self.progress["last_diagnostic_line"],
                                  "last_nonfinite_line": self.progress.get("last_nonfinite_line"),
                                  "nonfinite_summary_line": self.pending_nonfinite["line"] if self.pending_nonfinite else None,
                                  "checkpoints_found": [str(p) for p in sorted((self.native / "checkpoint").glob("state_*.pt"))]})
            self.progress["monitor_status"] = "paused"
            self.monitor("terminal")
        print("%s: %s; %s" % (self.a.run_id, self.manifest["status"], self.run_dir / "manifest.json"), flush=True)
        return 0 if self.manifest["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(Supervisor(parse_args()).run())
    except Exception as error:
        print("Supervisor refused or could not record run: %s" % error, file=sys.stderr, flush=True)
        sys.exit(1)
