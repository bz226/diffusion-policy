#!/usr/bin/env python3
"""Transfer only seed0's external watchdog; never restart or modify DPPO."""

import argparse
import copy
import datetime as dt
import fcntl
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from supervise_run import ANSI, COLLECTION, EVAL, NONFINITE, TRAIN, STUDY, Supervisor, atomic_json, utc


def process_stat(pid, expected_start=None):
    raw = Path("/proc/%d/stat" % pid).read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    if len(fields) < 50:
        raise RuntimeError("Linux process stat lacks field 52")
    result = {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
              "pgrp": int(fields[2]), "start_ticks": int(fields[19]),
              "raw_exit_status": int(fields[49])}
    if expected_start is not None and result["start_ticks"] != expected_start:
        raise RuntimeError("Process identity changed for PID %d" % pid)
    if result["state"] == "Z":
        status = result["raw_exit_status"]
        if os.WIFEXITED(status):
            result["returncode"] = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            result["returncode"] = -os.WTERMSIG(status)
        else:
            raise RuntimeError("Unrecognized zombie exit status")
    else:
        result["returncode"] = None
    return result


class ObservedProcess:
    """Read an unreaped child's kernel status without claiming parent ownership."""
    def __init__(self, pid, start_ticks):
        self.pid, self.start_ticks = pid, start_ticks
        self.exit_evidence = None

    def poll(self):
        evidence = process_stat(self.pid, self.start_ticks)
        if evidence["state"] == "Z":
            self.exit_evidence = evidence
        return evidence["returncode"]

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = self.poll()
            if code is not None:
                return code
            time.sleep(0.05)
        raise subprocess.TimeoutExpired("observed DPPO PID %d" % self.pid, timeout)


def checked_pipe(watchdog_pid, pipe_fd, dppo_pid):
    source = Path("/proc/%d/fd/%d" % (watchdog_pid, pipe_fd))
    pipe_name = os.readlink(str(source))
    if not pipe_name.startswith("pipe:[") or pipe_name != os.readlink("/proc/%d/fd/1" % dppo_pid):
        raise RuntimeError("Watchdog pipe does not match DPPO stdout")
    fdinfo = Path("/proc/%d/fdinfo/%d" % (watchdog_pid, pipe_fd)).read_text()
    flags = int(next(line.split()[1] for line in fdinfo.splitlines() if line.startswith("flags:")), 8)
    if flags & os.O_ACCMODE != os.O_RDONLY:
        raise RuntimeError("Expected the watchdog's read end of stdout")
    fd = os.open(str(source), os.O_RDONLY | os.O_NONBLOCK)
    if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        os.close(fd)
        raise RuntimeError("Duplicated pipe is not read-only")
    return fd, pipe_name


def pause_watchdog(pid, start_ticks):
    process_stat(pid, start_ticks)
    os.kill(pid, signal.SIGSTOP)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process_stat(pid, start_ticks)["state"] == "T":
            return {"state": "T", "confirmed_utc": utc()}
        time.sleep(0.02)
    raise RuntimeError("Watchdog did not enter stopped state")


class AdoptedRun(Supervisor):
    def __init__(self, options):
        # Deliberately do not call Supervisor.__init__ or launch a subprocess.
        original_dir = STUDY / "runs" / "seed0"
        original = json.loads((original_dir / "manifest.json").read_text())
        amendment_path = STUDY / "provenance" / "budget_amendment.json"
        amendment = json.loads(amendment_path.read_text())
        if (amendment.get("approved_cap_seconds") != 8100 or "seed0" not in amendment.get("scope", [])
                or original.get("run_id") != "seed0" or original.get("status") != "running"
                or original.get("pid") != options.dppo_pid or original.get("timeout_seconds") != 7200):
            raise RuntimeError("Original run or explicit budget amendment does not match")
        self.cwd = Path.cwd().resolve()
        if self.cwd != Path(original["cwd"]).resolve() or self.cwd != (STUDY / "dppo").resolve():
            raise RuntimeError("Launch adopter from the original dppo directory")
        if os.environ.get("SLURM_JOB_ID") != original.get("slurm_job_id"):
            raise RuntimeError("Adopter must run inside the original Slurm allocation")
        for key in ("DPPO_LOG_DIR", "DPPO_DATA_DIR"):
            if os.environ.get(key) != original["environment"][key]:
                raise RuntimeError("Original %s must be retained" % key)
        identity = process_stat(options.dppo_pid, options.expected_start_ticks)
        if identity["ppid"] != options.watchdog_pid or identity["pgrp"] != options.dppo_pid:
            raise RuntimeError("DPPO parent/group identity does not match")
        command = Path("/proc/%d/cmdline" % options.dppo_pid).read_bytes().rstrip(b"\0").split(b"\0")
        if [part.decode() for part in command] != original["command"]:
            raise RuntimeError("DPPO command line differs from the preserved original command")
        self.watchdog_identity = process_stat(options.watchdog_pid)
        if self.watchdog_identity["state"] in ("T", "Z", "X"):
            raise RuntimeError("Original watchdog is not running normally")
        self.start_wall = dt.datetime.fromisoformat(original["start_utc"]).timestamp()
        self.start_clock = time.monotonic() - (time.time() - self.start_wall)
        self.deadline = self.start_clock + 8100
        if time.time() >= self.start_wall + 7200 - 60:
            raise RuntimeError("Refusing a new handoff within 60s of the old deadline")
        self.args = SimpleNamespace(run_id="seed0_handoff", timeout_seconds=8100,
                                    requested_timeout_seconds=8100,
                                    estimate_seconds=options.remaining_estimate_seconds,
                                    command=original["command"])
        self.interval = math.ceil(max(1800, options.remaining_estimate_seconds / 6))
        self.next_check = time.monotonic() + self.interval
        self.seed, self.expected_rows = 0, 140
        self.logdir = Path(original["logdir"]) if original.get("logdir") else None
        self.process = ObservedProcess(options.dppo_pid, options.expected_start_ticks)
        self.run_dir = STUDY / "runs" / "seed0_handoff"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.original_dir, self.options = original_dir, options
        self.native_log = None
        self.native_log_offset = 0
        self.native_pending = b""
        self.stop_reason = self.interrupted = None
        self.last_progress_write = 0.0
        self.previous_monitor_signature = None
        self.stalled_checks = 0
        self.mirror_stdout = True
        self.paused = False
        self.pipe_fd = None
        self.scientific_failure = False
        self.progress = {"run_id": "seed0_handoff", "status": "preparing_handoff",
                         "bytes": 0, "lines": 0, "collection_iteration": -1,
                         "rollout_step": None, "last_completed_itr": None,
                         "training_env_steps": 0, "initial_eval_return": None, "last_line": ""}
        self.manifest = copy.deepcopy(original)
        self.manifest.update({"run_id": "seed0_handoff", "status": "preparing_handoff",
                              "original_run_id": "seed0", "handoff_start_utc": utc(),
                              "original_manifest": str(original_dir / "manifest.json"),
                              "timeout_seconds": 8100, "requested_timeout_seconds": 8100,
                              "deadline_utc": utc(self.start_wall + 8100),
                              "budget_amendment": str(amendment_path),
                              "console_log": str(self.run_dir / "console.log"),
                              "estimate_seconds": options.remaining_estimate_seconds,
                              "estimate_basis": "Estimated remaining authorized work at watchdog handoff",
                              "monitor_interval_seconds": self.interval,
                              "monitor_status": "scheduled", "verification_commands": [],
                              "dppo_identity_at_handoff": identity,
                              "original_watchdog_identity": self.watchdog_identity,
                              "dppo_restarted": False,
                              "original_control_timeout_is_not_dppo_crash": True,
                              "requires_original_console_reconciliation": True})
        atomic_json(self.run_dir / "original_manifest.json", original)
        atomic_json(self.run_dir / "original_progress.json",
                    json.loads((original_dir / "progress.json").read_text()))
        atomic_json(self.run_dir / "manifest.json", self.manifest)

    def stop_group(self):
        # Used only after a verified numerical failure, child crash, or approved cap.
        process_stat(self.process.pid, self.process.start_ticks)
        super().stop_group()

    def consume_live_line(self, raw):
        try:
            self.consume_line(raw)
        except ValueError as error:
            if str(error).startswith(("Printed NaN/Inf token:", "Initial evaluation", "printed evaluation reward")):
                self.scientific_failure = True
            raise

    def poll_native_log(self):
        with self.native_log.open("rb") as stream:
            stream.seek(self.native_log_offset)
            data = stream.read()
        self.native_log_offset += len(data)
        self.native_pending += data
        while b"\n" in self.native_pending:
            line, self.native_pending = self.native_pending.split(b"\n", 1)
            text = ANSI.sub("", line.decode("utf-8", errors="replace"))
            if NONFINITE.search(text):
                self.scientific_failure = True
                raise ValueError("Native DPPO log contains a NaN/Inf token: " + text[-400:])
        self.progress["native_log_bytes_checked"] = self.native_log_offset

    def update_study(self, terminal=False):
        if self.progress["status"] != "complete_pending_console_reconciliation":
            return super().update_study(terminal=terminal)
        path = STUDY / "experiment.md"
        text = path.read_text()
        section = re.compile(r"(## Current status and next step\n).*?(?=\n## |\Z)", re.S)
        message = (
            "%s: seed0 DPPO exited normally with code 0; all 140 result rows and required "
            "checkpoints passed verification. The handoff monitor is paused. Next: reconcile "
            "the original supervisor's final console and return code before any remaining seed. "
            "[Handoff record](runs/seed0_handoff/manifest.json).\n"
        ) % utc()
        if not section.search(text):
            raise RuntimeError("Missing current study status section")
        temporary = path.with_name("experiment.md.handoff.pending")
        temporary.write_text(section.sub(lambda match: match.group(1) + message, text, count=1))
        temporary.replace(path)

    def capture_adopted(self, console, pending):
        selector = selectors.DefaultSelector()
        selector.register(self.pipe_fd, selectors.EVENT_READ)
        leader_exit_clock = None
        first_fragment = True
        try:
            self.monitor("handoff_started")
            while selector.get_map() or self.process.poll() is None:
                self.poll_native_log()
                if self.interrupted:
                    raise RuntimeError("Adopter interrupted by " + self.interrupted)
                if time.monotonic() >= self.deadline:
                    self.scientific_failure = True
                    raise RuntimeError("Approved 8100-second wall-clock cap reached")
                code = self.process.poll()
                if code is not None and leader_exit_clock is None:
                    leader_exit_clock = time.monotonic()
                if code is not None and code != 0:
                    self.scientific_failure = True
                    raise RuntimeError("DPPO itself exited with status %d" % code)
                if leader_exit_clock is not None and selector.get_map() and time.monotonic() - leader_exit_clock >= 10:
                    self.scientific_failure = True
                    raise RuntimeError("Exited DPPO process left descendants holding stdout open for 10s")
                if time.monotonic() >= self.next_check:
                    try:
                        self.monitor("scheduled")
                    except ValueError as error:
                        if "Confirmed stall" in str(error):
                            self.scientific_failure = True
                        raise
                    self.next_check += self.interval
                for key, _ in selector.select(timeout=0.5):
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fd)
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
                        if first_fragment:
                            # One in-flight old-reader chunk may precede this fragment.
                            # Native run.log independently gates all numeric metrics.
                            first_fragment = False
                        else:
                            self.consume_live_line(line)
                    self.write_progress()
                if not selector.get_map():
                    time.sleep(0.05)
            if pending:
                if not first_fragment:
                    self.consume_live_line(pending)
            self.poll_native_log()
            evidence = process_stat(self.process.pid, self.process.start_ticks)
            if evidence["state"] != "Z" or evidence["raw_exit_status"] != 0 or evidence["returncode"] != 0:
                self.scientific_failure = True
                raise RuntimeError("Missing identity-matched kernel evidence of normal DPPO exit 0")
            self.manifest.update({"returncode": 0, "kernel_exit_evidence": evidence,
                                  "subprocess_end_utc": utc(),
                                  "subprocess_wall_seconds": time.monotonic() - self.start_clock})
            atomic_json(self.run_dir / "kernel_exit_evidence.json", evidence)
        finally:
            selector.close()

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, frame: setattr(self, "interrupted", signal.Signals(number).name))
        try:
            # Import validators and verify writable records/source before pausing anything.
            import numpy
            import torch
            import omegaconf
            self.manifest["source_at_handoff"] = self.source_check("handoff_before")
            self.discover_logdir()
            if self.logdir is None:
                raise RuntimeError("Native log directory must exist before handoff")
            self.native_log = self.logdir / "run.log"
            with self.native_log.open("rb") as stream:
                stream.read(1)  # Verify the independent numeric gate is readable before STOP.
            self.manifest["native_numeric_gate_log"] = str(self.native_log)
            self.pipe_fd, pipe_name = checked_pipe(self.options.watchdog_pid, self.options.pipe_fd,
                                                  self.options.dppo_pid)
            self.manifest["adopted_pipe"] = pipe_name
            with (self.run_dir / "console.log").open("xb", buffering=0) as console:
                self.paused = True  # Ensure finally sends CONT even if confirmation fails.
                self.manifest["watchdog_stop_confirmation"] = pause_watchdog(
                    self.options.watchdog_pid, self.watchdog_identity["start_ticks"])
                self.manifest["watchdog_paused_utc"] = utc()
                prefix = (self.original_dir / "console.log").read_bytes()
                with (self.run_dir / "original_console_prefix.log").open("xb") as stream:
                    stream.write(prefix)
                self.manifest["original_console_prefix_bytes"] = len(prefix)
                self.manifest["original_console_reconciliation_note"] = (
                    "The old reader may retain one unlogged chunk. Full original console after "
                    "CONT plus the handoff console must be reconciled before another seed. "
                    "Native run.log is independently checked every 0.5s for all numeric metrics."
                )
                parts = prefix.split(b"\n")
                pending = parts.pop()
                for line in parts:
                    self.consume_live_line(line)
                self.progress["bytes"] = len(prefix)
                if self.progress["initial_eval_return"] is None:
                    raise RuntimeError("Original console lacks the already-completed initial evaluation gate")
                self.progress["status"] = "running"
                self.manifest["status"] = "running"
                self.capture_adopted(console, pending)
            self.progress["status"] = "verifying"
            self.write_progress(force=True)
            self.verify()
            self.manifest["status"] = self.progress["status"] = "complete_pending_console_reconciliation"
            self.manifest["verification_complete"] = True
        except Exception as error:
            self.stop_reason = str(error)
            self.manifest.update({"status": "failed", "failure_reason": self.stop_reason})
            self.progress["status"] = "failed"
            if self.paused and self.scientific_failure:
                try:
                    self.stop_group()
                except Exception as cleanup_error:
                    self.manifest["termination_error"] = str(cleanup_error)
        finally:
            self.manifest.update({"end_utc": utc(), "monitor_status": "paused",
                                  "wall_seconds_including_verification": time.monotonic() - self.start_clock})
            self.progress["monitor_status"] = "paused"
            try:
                self.monitor("handoff_terminal", terminal=True)
            except Exception as record_error:
                self.manifest["terminal_monitor_error"] = str(record_error)
                self.manifest["status"] = self.progress["status"] = "failed"
            try:
                # Persist verified evidence BEFORE the original batch can exit.
                self.write_progress(force=True)
                atomic_json(self.run_dir / "manifest.json", self.manifest)
            finally:
                try:
                    if self.pipe_fd is not None:
                        os.close(self.pipe_fd)
                finally:
                    if self.paused:
                        try:
                            process_stat(self.options.watchdog_pid, self.watchdog_identity["start_ticks"])
                            os.kill(self.options.watchdog_pid, signal.SIGCONT)
                        except Exception as resume_error:
                            print("CRITICAL: original watchdog could not be resumed: %s" % resume_error,
                                  file=sys.stderr, flush=True)
                            raise
        print("Handoff outcome: %s; DPPO returncode: %s; record: %s" %
              (self.manifest["status"], self.manifest.get("returncode"), self.run_dir), flush=True)
        if self.stop_reason:
            print(self.stop_reason, file=sys.stderr, flush=True)
        return 0 if self.manifest["status"] == "complete_pending_console_reconciliation" else 1


def reconcile_only():
    """Read the resumed original parent's evidence before releasing later seeds."""
    directory = STUDY / "runs" / "seed0_handoff"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete_pending_console_reconciliation":
        raise RuntimeError("No verified handoff awaits reconciliation")
    original = json.loads((STUDY / "runs" / "seed0" / "manifest.json").read_text())
    if original.get("monitor_status") != "paused" or original.get("returncode") != 0:
        raise RuntimeError("Original parent has not recorded a terminal DPPO exit 0")
    if original.get("status") != "complete" and original.get("failure_reason") != "Approved wall-clock timeout reached":
        raise RuntimeError("Original supervisor reports a failure other than its superseded control deadline")
    evidence = manifest.get("kernel_exit_evidence", {})
    if (not manifest.get("verification_complete") or manifest.get("verified_rows") != 140
            or manifest.get("final_training_env_steps") != 10080000
            or manifest.get("checkpoint_tensors_finite") is not True
            or evidence.get("state") != "Z" or evidence.get("raw_exit_status") != 0
            or evidence.get("pid") != manifest.get("pid")
            or evidence.get("start_ticks") != manifest["dppo_identity_at_handoff"]["start_ticks"]):
        raise RuntimeError("Completed native artifacts/kernel exit evidence are incomplete")
    prefix = (directory / "original_console_prefix.log").read_bytes()
    original_console = (STUDY / "runs" / "seed0" / "console.log").read_bytes()
    if not original_console.startswith(prefix):
        raise RuntimeError("Original console prefix was modified")
    combined = original_console + (directory / "console.log").read_bytes()
    evaluations, training, collections = [], [], []
    for raw in combined.splitlines():
        line = ANSI.sub("", raw.decode("utf-8", errors="replace"))
        if NONFINITE.search(line):
            raise RuntimeError("Reconciled full console contains NaN/Inf: " + line[-400:])
        match = EVAL.search(line)
        if match:
            evaluations.append(float(match.group(1)))
        match = TRAIN.search(line)
        if match:
            training.append((int(match.group(1)), int(match.group(2))))
        match = COLLECTION.search(line)
        if match:
            collections.append((int(match.group(1)), int(match.group(2))))
    if (len(evaluations) != 14 or not 3850 <= evaluations[0] <= 4650
            or training != [(itr, (itr - itr // 10) * 80000) for itr in range(140) if itr % 10]
            or collections != [(step, 500) for _ in range(140) for step in range(0, 500, 10)]):
        raise RuntimeError("Reconciled complete console does not match the full native iteration schedule")
    native = Path(manifest["native_numeric_gate_log"]).read_text()
    if NONFINITE.search(ANSI.sub("", native)):
        raise RuntimeError("Final native log contains NaN/Inf")
    result = {"status": "passed", "checked_utc": utc(), "original_parent_returncode": 0,
              "original_supervisor_status": original["status"],
              "superseded_control_timeout_only": original.get("failure_reason") == "Approved wall-clock timeout reached",
              "original_prefix_bytes": len(prefix), "final_original_console_bytes": len(original_console),
              "recovered_inflight_chunk_bytes": len(original_console) - len(prefix),
              "evaluation_lines": len(evaluations), "training_lines": len(training),
              "rollout_counter_lines": len(collections), "complete_stream_finite": True}
    with (directory / "original_controller_terminal_manifest.json").open("x") as stream:
        json.dump(original, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (directory / "original_controller_console_tail.log").open("xb") as stream:
        stream.write(original_console[len(prefix):])
    with (directory / "console_reconciliation.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (directory / "manifest_before_reconciliation.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest.update({"status": "complete", "requires_original_console_reconciliation": False,
                     "console_reconciliation": str(directory / "console_reconciliation.json")})
    atomic_json(directory / "manifest.json", manifest)
    progress = json.loads((directory / "progress.json").read_text())
    progress.update({"status": "complete", "updated_utc": utc()})
    atomic_json(directory / "progress.json", progress)
    study_path = STUDY / "experiment.md"
    study_text = study_path.read_text()
    section = re.compile(r"(## Current status and next step\n).*?(?=\n## |\Z)", re.S)
    if not section.search(study_text):
        raise RuntimeError("Missing current study status section")
    already_started = [seed for seed in ("seed1", "seed2") if (STUDY / "runs" / seed).exists()]
    next_step = ("Inspect and continue the existing remaining-run records (%s); do not launch duplicates." %
                 ", ".join(already_started) if already_started else
                 "Run seeds 1 and 2 once each under the approved 8100-second caps.")
    message = (
        "%s: seed0 verified complete at 10,080,000 training steps. DPPO exited normally "
        "with code 0; full console reconciliation passed, including all 140 iterations. "
        "The original watchdog's superseded control deadline is recorded separately "
        "from DPPO's exit status. Monitoring is paused. Next: %s "
        "[Handoff record](runs/seed0_handoff/manifest.json).\n"
    ) % (utc(), next_step)
    temporary = study_path.with_name("experiment.md.reconciliation.pending")
    temporary.write_text(section.sub(lambda match: match.group(1) + message, study_text, count=1))
    temporary.replace(study_path)
    print("Seed0 handoff reconciliation PASSED; verified DPPO exit 0 and complete finite stream.")
    return 0


def self_test():
    """Benign independent pipe/parent/child handoff; contains no DPPO execution."""
    directory = STUDY / "provenance" / ("handoff_self_test_" + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    parent_code = r'''
import json, os, pathlib, selectors, subprocess, sys, time
p = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-u", "-c", "import time; [(print(i, flush=True), time.sleep(.02)) for i in range(150)]"], stdout=subprocess.PIPE, start_new_session=True)
(p / "identity.json").write_text(json.dumps({"pid": child.pid, "fd": child.stdout.fileno()}))
with (p / "original.log").open("wb", buffering=0) as out:
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    while selector.get_map():
        for key, _ in selector.select(timeout=.5):
            part = os.read(key.fd, 4096)
            if not part:
                selector.unregister(key.fileobj)
            else:
                out.write(part)
code = child.wait()
(p / "parent_returncode.json").write_text(json.dumps({"returncode": code}))
'''
    parent = subprocess.Popen([sys.executable, "-u", "-c", parent_code, str(directory)])
    fd = None
    paused = False
    try:
        deadline = time.monotonic() + 12
        while not (directory / "identity.json").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Self-test parent did not initialize")
            time.sleep(0.02)
        time.sleep(0.15)
        identity = json.loads((directory / "identity.json").read_text())
        original_parent = process_stat(parent.pid)
        child = process_stat(identity["pid"])
        fd, pipe_name = checked_pipe(parent.pid, identity["fd"], identity["pid"])
        paused = True
        stop_confirmation = pause_watchdog(parent.pid, original_parent["start_ticks"])
        adopted = b""
        history = []
        next_monitor = time.monotonic()
        eof = False
        while not eof or process_stat(identity["pid"], child["start_ticks"])["state"] != "Z":
            if time.monotonic() >= deadline:
                raise RuntimeError("Self-test bounded handoff timed out")
            try:
                data = os.read(fd, 4096)
                if not data:
                    eof = True
                adopted += data
            except BlockingIOError:
                pass
            if time.monotonic() >= next_monitor:
                history.append({"utc": utc(), "adopted_bytes": len(adopted)})
                next_monitor += 0.1
            time.sleep(0.01)
        evidence = process_stat(identity["pid"], child["start_ticks"])
        if evidence["returncode"] != 0:
            raise RuntimeError("Self-test child exit status is not zero")
        (directory / "adopted.log").write_bytes(adopted)
        os.kill(parent.pid, signal.SIGCONT)
        paused = False
        if parent.wait(timeout=5) != 0:
            raise RuntimeError("Self-test original parent failed after resumption")
        observed = json.loads((directory / "parent_returncode.json").read_text())["returncode"]
        combined = (directory / "original.log").read_bytes() + adopted
        if observed != 0 or [int(line) for line in combined.splitlines()] != list(range(150)):
            raise RuntimeError("Self-test stream or parent exit-code reconciliation failed")
        result = {"status": "passed", "synthetic_only": True, "values": 150,
                  "all_values_finite": True, "pipe": pipe_name,
                  "kernel_exit_evidence": evidence, "parent_observed_returncode": observed,
                  "watchdog_stop_confirmation": stop_confirmation,
                  "scheduled_monitor_samples": history}
        atomic_json(directory / "result.json", result)
        print("Synthetic handoff passed: %s" % directory)
        return 0
    finally:
        if paused and parent.poll() is None:
            os.kill(parent.pid, signal.SIGCONT)
        if fd is not None:
            os.close(fd)
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true")
    parser.add_argument("--perform-handoff", action="store_true")
    parser.add_argument("--watchdog-pid", type=int)
    parser.add_argument("--dppo-pid", type=int)
    parser.add_argument("--expected-start-ticks", type=int)
    parser.add_argument("--pipe-fd", type=int)
    parser.add_argument("--remaining-estimate-seconds", type=float)
    options = parser.parse_args()
    if options.reconcile_only:
        if options.self_test or options.perform_handoff:
            parser.error("Reconciliation is separate from self-test and handoff")
        return reconcile_only()
    if options.self_test:
        if options.perform_handoff:
            parser.error("Self-test and real handoff are mutually exclusive")
        return self_test()
    if not options.perform_handoff:
        parser.error("Real handoff requires explicit --perform-handoff")
    if any(getattr(options, key) is None for key in
           ("watchdog_pid", "dppo_pid", "expected_start_ticks", "pipe_fd", "remaining_estimate_seconds")):
        parser.error("Supply both PIDs, expected DPPO start ticks, pipe FD, and remaining estimate")
    if not math.isfinite(options.remaining_estimate_seconds) or options.remaining_estimate_seconds <= 0:
        parser.error("Remaining estimate must be finite and positive")
    return AdoptedRun(options).run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("Watchdog adoption stopped: %s" % error, file=sys.stderr, flush=True)
        sys.exit(1)
