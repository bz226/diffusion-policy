#!/usr/bin/env python3
"""Execute one prescribed DPPO run with external gates and a scheduled monitor."""

import argparse
import datetime as dt
import json
import math
import numbers
import os
import pickle
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"
STUDY = Path(__file__).resolve().parents[1]
ENV_KEYS = (
    "DPPO_DATA_DIR", "DPPO_LOG_DIR", "DPPO_HARDWARE_RECORD", "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MUJOCO_GL",
    "MUJOCO_PY_MUJOCO_PATH", "MUJOCO_PY_FORCE_CPU", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
    "MPLCONFIGDIR", "D4RL_SUPPRESS_IMPORT_ERROR", "SLURM_JOB_ID", "SLURM_JOBID",
    "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS",
)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NONFINITE = re.compile(r"(?<![\w./])[-+]?(?:nan|inf(?:inity)?)(?![\w./])", re.I)
EVAL = re.compile(r"\beval:.*?\bavg episode reward\s+([^\s|]+)")
COLLECTION = re.compile(r"Processed step (\d+) of (\d+)")
TRAIN = re.compile(r"\b(\d+): step\s+(\d+)\s*\|")
CHECKPOINT = re.compile(r"Saved model to (.+)/checkpoint/state_0\.pt\s*$")


def utc(timestamp=None):
    return dt.datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp, dt.timezone.utc
    ).isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("%s is not a real scalar" % label)
    if not math.isfinite(float(value)):
        raise ValueError("%s is not finite" % label)
    return float(value)


def finite_tree(value, label, np, torch=None):
    if torch is not None and torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("Nonfinite checkpoint tensor: " + label)
    elif isinstance(value, numbers.Number):
        if not bool(np.isfinite(value)):
            raise ValueError("Nonfinite result: " + label)
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ValueError("Nonfinite result array: " + label)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, "%s.%s" % (label, key), np, torch)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, "%s[%d]" % (label, index), np, torch)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, choices=("smoke", "seed0", "seed1", "seed2"))
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--estimate-seconds", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.requested_timeout_seconds = args.timeout_seconds
    if args.run_id != "smoke" and args.timeout_seconds == 7200:
        amendment = json.loads((STUDY / "provenance" / "budget_amendment.json").read_text())
        if (amendment["approved_cap_seconds"] != 8100
                or args.run_id not in amendment["scope"]):
            parser.error("Budget amendment does not cover this run")
        args.timeout_seconds = 8100
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    cap = 1800 if args.run_id == "smoke" else 8100
    if not 0 < args.timeout_seconds <= cap:
        parser.error("Timeout must be positive and no greater than the approved %ds" % cap)
    if not math.isfinite(args.estimate_seconds) or args.estimate_seconds <= 0:
        parser.error("--estimate-seconds must be finite and positive")
    handoff_dir = STUDY / "runs" / "seed0_handoff"
    if args.run_id in ("seed1", "seed2") and handoff_dir.exists():
        manifest = json.loads((handoff_dir / "manifest.json").read_text())
        if (manifest.get("status") != "complete"
                or manifest.get("requires_original_console_reconciliation") is not False):
            parser.error("Seed 0 handoff must be fully reconciled before another seed")
        reconciliation = json.loads((handoff_dir / "console_reconciliation.json").read_text())
        if reconciliation.get("status") != "passed":
            parser.error("Seed 0 console reconciliation has not passed")
    if not args.command:
        parser.error("Supply the exact DPPO command after --")
    expected = ["--config-dir=cfg/gym/finetune/halfcheetah-v2",
                "--config-name=ft_ppo_diffusion_mlp"]
    if args.run_id == "smoke":
        expected += ["train.n_train_itr=2", "wandb=null"]
    else:
        expected += ["train.n_train_itr=140", "train.save_model_freq=35",
                     "seed=" + args.run_id[-1], "wandb=null"]
    scripts = [i for i, part in enumerate(args.command) if part == "script/run.py"]
    if len(scripts) != 1 or args.command[scripts[0] + 1:] != expected:
        parser.error("Command must contain script/run.py followed by the prescribed arguments")
    prefix = args.command[:scripts[0]]
    if not prefix or prefix[1:] not in ([], ["-u"]):
        parser.error("Use a Python executable, optionally -u, before script/run.py")
    return args


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.cwd = Path.cwd().resolve()
        if self.cwd != (STUDY / "dppo").resolve():
            raise ValueError("Launch from the study's dppo/ directory")
        self.run_dir = STUDY / "runs" / args.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.start_wall = time.time()
        self.start_clock = time.monotonic()
        self.interval = math.ceil(max(1800, args.estimate_seconds / 6))
        self.next_check = self.start_clock + self.interval
        self.deadline = self.start_clock + args.timeout_seconds
        self.seed = 42 if args.run_id == "smoke" else int(args.run_id[-1])
        self.expected_rows = 2 if args.run_id == "smoke" else 140
        self.logdir = None
        self.process = None
        self.stop_reason = None
        self.interrupted = None
        self.last_progress_write = 0.0
        self.previous_monitor_signature = None
        self.stalled_checks = 0
        self.mirror_stdout = True
        self.progress = {"run_id": args.run_id, "status": "starting", "bytes": 0,
                         "lines": 0, "collection_iteration": -1, "rollout_step": None,
                         "last_completed_itr": None, "training_env_steps": 0,
                         "initial_eval_return": None, "last_line": ""}
        self.manifest = {
            "run_id": args.run_id, "seed": self.seed, "status": "starting",
            "start_utc": utc(self.start_wall), "end_utc": None,
            "command": args.command, "command_shell_display": shlex.join(args.command),
            "cwd": str(self.cwd), "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", os.environ.get("SLURM_JOBID")),
            "hardware_record": os.environ.get("DPPO_HARDWARE_RECORD", str(STUDY / "setup")),
            "timeout_seconds": args.timeout_seconds,
            "requested_timeout_seconds": args.requested_timeout_seconds,
            "budget_amendment": str(STUDY / "provenance" / "budget_amendment.json") if self.seed != 42 else None,
            "deadline_utc": utc(self.start_wall + args.timeout_seconds),
            "estimate_seconds": args.estimate_seconds,
            "estimate_basis": (
                "Provisional 60s smoke estimate using the paper's 16.8s/training iteration"
                if args.run_id == "smoke" else
                "14 evaluation and 126 training iterations from measured smoke, replaced by completed full native timing when available"
            ),
            "runtime_estimate_records": (
                None if args.run_id == "smoke" else
                str(STUDY / "provenance" / "runtime_estimates.jsonl")
            ),
            "monitor_interval_seconds": self.interval, "monitor_status": "scheduled",
            "console_log": str(self.run_dir / "console.log"), "verification_commands": [],
        }
        atomic_json(self.run_dir / "manifest.json", self.manifest)

    def source_check(self, phase):
        results = {}
        for label, command in (
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("tracked_diff", ["git", "diff", "--exit-code", "HEAD", "--"]),
        ):
            result = subprocess.run(command, cwd=str(self.cwd), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=30)
            self.manifest["verification_commands"].append(command)
            results[label] = {"returncode": result.returncode, "output": result.stdout}
        atomic_json(self.run_dir / ("source_" + phase + ".json"), results)
        if results["commit"]["returncode"] or results["commit"]["output"].strip() != COMMIT:
            raise ValueError("Pinned commit verification failed")
        if results["tracked_diff"]["returncode"]:
            raise ValueError("Tracked DPPO source/config differs from pinned commit")
        return {"commit": COMMIT, "tracked_files_clean": True}

    def discover_logdir(self):
        if self.logdir is not None:
            return
        log_root = Path(os.environ["DPPO_LOG_DIR"]).expanduser().resolve()
        candidates = [path.parent.parent for path in
                      (log_root / "gym-finetune").glob("*/*_%d/.hydra/config.yaml" % self.seed)
                      if path.stat().st_mtime >= self.start_wall - 2]
        if len(candidates) > 1:
            raise ValueError("Ambiguous newly created native log directories")
        if len(candidates) == 1:
            self.accept_logdir(candidates[0])

    def accept_logdir(self, path):
        path = Path(path).expanduser().resolve()
        root = Path(os.environ["DPPO_LOG_DIR"]).expanduser().resolve()
        if root not in path.parents or STUDY not in path.parents:
            raise ValueError("Native log directory is outside the approved study/log root")
        if self.logdir is not None and path != self.logdir:
            raise ValueError("Native log directory changed within the run")
        self.logdir = path
        self.manifest["logdir"] = str(path)
        self.manifest["result_path"] = str(path / "result.pkl")
        self.manifest["native_config"] = str(path / ".hydra" / "config.yaml")

    def consume_line(self, raw):
        line = ANSI.sub("", raw.decode("utf-8", errors="replace")).rstrip("\r\n")
        self.progress["lines"] += 1
        self.progress["last_line"] = line[-1600:]
        if NONFINITE.search(line):
            raise ValueError("Printed NaN/Inf token: " + line[-400:])
        match = CHECKPOINT.search(line)
        if match:
            self.accept_logdir(match.group(1))
        match = COLLECTION.search(line)
        if match:
            step = int(match.group(1))
            if step == 0:
                self.progress["collection_iteration"] += 1
            self.progress["rollout_step"] = step
        match = TRAIN.search(line)
        if match:
            self.progress["last_completed_itr"] = int(match.group(1))
            self.progress["training_env_steps"] = int(match.group(2))
        match = EVAL.search(line)
        if match:
            reward = finite(float(match.group(1)), "printed evaluation reward")
            self.progress["last_completed_itr"] = self.progress["collection_iteration"]
            self.progress["last_eval_return"] = reward
            if self.progress["initial_eval_return"] is None:
                self.progress["initial_eval_return"] = reward
                if not 3850 <= reward <= 4650:
                    raise ValueError("Initial evaluation %.4f is outside [3850, 4650]" % reward)

    def write_progress(self, force=False):
        now = time.monotonic()
        if force or now - self.last_progress_write >= 1:
            self.progress.update({"updated_utc": utc(),
                                  "elapsed_seconds": now - self.start_clock,
                                  "deadline_utc": self.manifest["deadline_utc"],
                                  "logdir": str(self.logdir) if self.logdir else None})
            atomic_json(self.run_dir / "progress.json", self.progress)
            self.last_progress_write = now

    def update_study(self, terminal=False):
        path = STUDY / "experiment.md"
        text = path.read_text()
        section = re.compile(r"(## Current status and next step\n).*?(?=\n## |\Z)", re.S)
        if not section.search(text):
            raise ValueError("experiment.md lacks Current status and next step section")
        next_step = ("Verify the next authorized stage; no automatic retry or extension."
                     if self.progress["status"] == "complete" else
                     "Execution stopped; report the recorded failure before any further run."
                     if terminal else "Continue the same authorized run; stop on any failure gate.")
        note = "%s: `%s` %s; last completed iteration %s, training steps %s, elapsed %.1fs / %ds. " % (
            utc(), self.args.run_id, self.progress["status"], self.progress["last_completed_itr"],
            self.progress["training_env_steps"], time.monotonic() - self.start_clock,
            self.args.timeout_seconds)
        note += ("Monitor paused. " if terminal else "Scheduled monitor interval %ds, based on %.1fs estimated runtime. " %
                 (self.interval, self.args.estimate_seconds))
        if self.stop_reason:
            note += "Failure: %s. " % self.stop_reason.replace("\n", " ")
        note += "Next: %s [Run record](runs/%s/manifest.json).\n" % (next_step, self.args.run_id)
        temporary = path.with_name("experiment.md.supervisor.pending")
        temporary.write_text(section.sub(lambda match: match.group(1) + note, text, count=1))
        temporary.replace(path)

    def monitor(self, event, terminal=False):
        self.discover_logdir()
        snapshot = dict(self.progress)
        snapshot.update({"checked_utc": utc(), "event": event,
                         "elapsed_seconds": time.monotonic() - self.start_clock,
                         "deadline_utc": self.manifest["deadline_utc"],
                         "disk_free_bytes": shutil.disk_usage(STUDY).free,
                         "load_average": list(os.getloadavg()),
                         "monitor_status": "paused" if terminal else "running"})
        if self.process is not None:
            snapshot["process_pid"] = self.process.pid
            snapshot["process_returncode"] = self.process.poll()
            try:
                status = Path("/proc/%d/status" % self.process.pid).read_text()
                rss = re.search(r"^VmRSS:\s+(\d+) kB", status, re.M)
                snapshot["main_process_rss_bytes"] = int(rss.group(1)) * 1024 if rss else None
            except OSError:
                snapshot["main_process_rss_bytes"] = None
        if self.logdir is not None:
            native_result = self.logdir / "result.pkl"
            snapshot["native_result_bytes"] = native_result.stat().st_size if native_result.exists() else 0
        signature = (self.progress["bytes"], self.progress["collection_iteration"],
                     self.progress["rollout_step"], self.progress["last_completed_itr"])
        if event == "scheduled":
            self.stalled_checks = self.stalled_checks + 1 if signature == self.previous_monitor_signature else 0
        self.previous_monitor_signature = signature
        snapshot["consecutive_scheduled_checks_without_progress"] = self.stalled_checks
        with (self.run_dir / "monitor_history.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
        self.write_progress(force=True)
        self.update_study(terminal=terminal)
        atomic_json(self.run_dir / "manifest.json", self.manifest)
        if self.stalled_checks >= 2 and not terminal:
            raise ValueError("Confirmed stall: two consecutive scheduled checks without progress")

    def stop_group(self):
        if self.process is None:
            return
        for sig, wait_seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                continue
            # Kill surviving descendants even if the group leader already exited.
            if sig == signal.SIGTERM:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return

    def capture(self):
        process_start_clock = time.monotonic()
        self.manifest["subprocess_start_utc"] = utc()
        self.process = subprocess.Popen(self.args.command, cwd=str(self.cwd),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True, bufsize=0)
        self.manifest.update({"status": "running", "pid": self.process.pid,
                              "monitor_status": "running"})
        self.progress["status"] = "running"
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        pending = b""
        leader_exit_clock = None
        try:
            with (self.run_dir / "console.log").open("xb", buffering=0) as console:
                self.monitor("start")
                while selector.get_map():
                    now = time.monotonic()
                    if not self.stop_reason:
                        returncode = self.process.poll()
                        if returncode is not None and leader_exit_clock is None:
                            leader_exit_clock = now
                        if self.interrupted:
                            self.stop_reason = "Supervisor received " + self.interrupted
                        elif returncode is not None and returncode != 0:
                            self.stop_reason = "DPPO command exited with status %d" % returncode
                        elif now >= self.deadline:
                            self.stop_reason = "Approved wall-clock timeout reached"
                        elif leader_exit_clock is not None and now - leader_exit_clock >= 10:
                            self.stop_reason = "Exited DPPO process left descendants holding its output pipe open"
                        elif now >= self.next_check:
                            try:
                                self.monitor("scheduled")
                            except Exception as error:
                                self.stop_reason = str(error)
                            self.next_check += self.interval
                        if self.stop_reason:
                            self.stop_group()
                    events = selector.select(timeout=0.5)
                    for key, _ in events:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        console.write(chunk)
                        if self.mirror_stdout:
                            try:
                                sys.stdout.buffer.write(chunk)
                                sys.stdout.buffer.flush()
                            except BrokenPipeError:
                                self.mirror_stdout = False
                        self.progress["bytes"] += len(chunk)
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if not self.stop_reason:
                                try:
                                    self.consume_line(line)
                                except Exception as error:
                                    self.stop_reason = str(error)
                                    self.stop_group()
                        self.write_progress()
                if pending and not self.stop_reason:
                    self.consume_line(pending)
            returncode = self.process.wait(timeout=10)
            self.manifest["returncode"] = returncode
            self.manifest["subprocess_end_utc"] = utc()
            self.manifest["subprocess_wall_seconds"] = time.monotonic() - process_start_clock
            if self.stop_reason:
                raise ValueError(self.stop_reason)
            if returncode != 0:
                raise ValueError("DPPO command exited with status %d" % returncode)
        finally:
            selector.close()

    def save_configuration(self):
        from omegaconf import OmegaConf
        self.discover_logdir()
        if self.logdir is None:
            raise ValueError("Native log directory was not found")
        native_config = self.logdir / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(native_config)
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
        OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
        # Hydra's stored ${now:...} must not become the verification time.
        cfg.logdir = str(self.logdir)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if not (self.run_dir / "config.yaml").exists():
            with (self.run_dir / "config.yaml").open("x") as stream:
                stream.write(OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True))
        atomic_json(self.run_dir / "config.json", {
            "native_config": str(native_config), "resolved_configuration": resolved,
            "base_policy_path": resolved["base_policy_path"],
            "normalization_path": resolved["normalization_path"],
            "logdir": str(self.logdir), "result_path": str(self.logdir / "result.pkl")})
        self.manifest["resolved_config"] = str(self.run_dir / "config.yaml")
        return resolved

    def verify(self):
        import numpy as np
        import torch

        resolved = self.save_configuration()
        checks = {"seed": self.seed, "act_steps": 4, "horizon_steps": 4,
                  "denoising_steps": 20, "ft_denoising_steps": 10, "wandb": None,
                  "env_name": "halfcheetah-medium-v2"}
        if any(resolved[key] != value for key, value in checks.items()):
            raise ValueError("Resolved configuration disagrees with the approved run")
        if (resolved["env"]["n_envs"] != 40 or resolved["train"]["n_steps"] != 500
                or resolved["train"]["val_freq"] != 10
                or resolved["train"]["n_train_itr"] != self.expected_rows
                or resolved["train"]["save_model_freq"] != (100 if self.expected_rows == 2 else 35)):
            raise ValueError("Resolved training budget or unchanged environment settings disagree")
        with (self.logdir / "result.pkl").open("rb") as stream:
            records = pickle.load(stream)
        if not isinstance(records, list) or len(records) != self.expected_rows:
            raise ValueError("Incorrect number of native result records")
        times = []
        for itr, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError("Invalid result record at iteration %d" % itr)
            finite_tree(record, "iteration%d" % itr, np)
            if any(isinstance(record.get(key), bool) or not isinstance(record.get(key), numbers.Integral)
                   for key in ("itr", "step")):
                raise ValueError("Invalid iteration/step type")
            if record["itr"] != itr or record["step"] != (itr - itr // 10) * 80000:
                raise ValueError("Native iteration/step schedule mismatch")
            metric = "eval_episode_reward" if itr % 10 == 0 else "train_episode_reward"
            other = "train_episode_reward" if itr % 10 == 0 else "eval_episode_reward"
            if other in record:
                raise ValueError("Evaluation/training metric schedule mismatch")
            reward = finite(record.get(metric), metric)
            elapsed = finite(record.get("time"), "iteration time")
            if elapsed <= 0:
                raise ValueError("Nonpositive iteration duration")
            times.append(elapsed)
            if itr == 0 and not 3850 <= reward <= 4650:
                raise ValueError("Saved initial evaluation is outside [3850, 4650]")
        if self.progress["initial_eval_return"] is None:
            raise ValueError("First evaluation gate was not observed in live output")
        checkpoints = []
        for itr in ((0, 1) if self.expected_rows == 2 else (0, 35, 70, 105, 139)):
            path = self.logdir / "checkpoint" / ("state_%d.pt" % itr)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("Missing or empty checkpoint: " + str(path))
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or checkpoint.get("itr") != itr or "model" not in checkpoint:
                raise ValueError("Invalid checkpoint content: " + str(path))
            finite_tree(checkpoint, str(path), np, torch)
            del checkpoint
            checkpoints.append(str(path))
        self.manifest["source_after"] = self.source_check("after")
        self.manifest.update({"checkpoints": checkpoints, "verified_rows": len(records),
                              "initial_eval_return": float(records[0]["eval_episode_reward"]),
                              "final_training_env_steps": records[-1]["step"],
                              "native_iteration_time_sum_seconds": float(np.sum(times)),
                              "native_iteration_time_mean_seconds": float(np.mean(times)),
                              "checkpoint_tensors_finite": True,
                              "resolved_config": str(self.run_dir / "config.yaml")})
        self.progress.update({"last_completed_itr": self.expected_rows - 1,
                              "training_env_steps": records[-1]["step"]})

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: setattr(self, "interrupted", signal.Signals(signum).name))
        try:
            for key, suffix in (("DPPO_DATA_DIR", "data"), ("DPPO_LOG_DIR", "log")):
                if Path(os.environ[key]).resolve() != self.cwd / suffix:
                    raise ValueError("%s must point to the clone's %s/" % (key, suffix))
            self.manifest["source_before"] = self.source_check("before")
            self.capture()
            self.progress["status"] = "verifying"
            self.write_progress(force=True)
            self.verify()
            if self.interrupted:
                raise ValueError("Supervisor received " + self.interrupted)
            self.progress["status"] = "complete"
            self.manifest["status"] = "complete"
        except Exception as error:
            self.stop_reason = str(error)
            self.stop_group()
            self.progress["status"] = "failed"
            self.manifest.update({"status": "failed", "failure_reason": self.stop_reason,
                                  "returncode": self.process.poll() if self.process else None})
            try:
                self.discover_logdir()
                if self.logdir is not None and (self.logdir / ".hydra" / "config.yaml").is_file():
                    self.save_configuration()
            except Exception as config_error:
                self.manifest["failure_config_capture_error"] = str(config_error)
            if "source_after" not in self.manifest:
                try:
                    self.manifest["source_after"] = self.source_check("after")
                except Exception as source_error:
                    self.manifest["source_after_error"] = str(source_error)
        finally:
            self.manifest.update({"end_utc": utc(),
                                  "wall_seconds_including_verification": time.monotonic() - self.start_clock,
                                  "monitor_status": "paused"})
            self.progress["monitor_status"] = "paused"
            try:
                self.monitor("terminal", terminal=True)
            except Exception as error:
                self.manifest["terminal_monitor_error"] = str(error)
                self.manifest["status"] = "failed"
                self.progress["status"] = "failed"
                self.stop_reason = self.stop_reason or "Terminal record verification failed: " + str(error)
                self.manifest["failure_reason"] = self.stop_reason
            self.write_progress(force=True)
            atomic_json(self.run_dir / "manifest.json", self.manifest)
        print("Supervisor %s: %s. Monitor paused. Record: %s" %
              (self.args.run_id, self.manifest["status"], self.run_dir / "manifest.json"), flush=True)
        if self.stop_reason:
            print("Stopped: " + self.stop_reason, file=sys.stderr, flush=True)
        return 0 if self.manifest["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(Supervisor(parse_args()).run())
    except Exception as error:
        print("Supervisor refused launch: %s" % error, file=sys.stderr)
        sys.exit(1)
