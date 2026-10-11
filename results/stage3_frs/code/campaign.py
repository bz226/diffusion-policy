"""Bounded FRS1 addendum queue. No retries, optional conditions, or source edits."""
import argparse
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

from monitor import (ROOT, SOURCE, PARENT, SLURM_TERMINAL, TERMINAL, append_json, command,
                     observe, read_json, safe_json, utc, validate_gate, validate_source,
                     verify_complete, write_json, source_files)

BUDGET = 14 * 3600
GATE_CAP = 3600
TRAIN_CAP = 4 * 3600
EVAL_CAP = 900
MAX_PARALLEL = 3
JOURNAL = ROOT / "runs/campaign_commands.jsonl"
STATE = ROOT / "runs/campaign.json"
GATE_DEFAULT = ROOT / "provenance/committed_gate.json"
COMMON = ["model.clip_ploss_coef=1e6", "model.clip_ploss_coef_base=1e6", "train.target_kl=null",
          "+model.clamp_logprob=false", "model.randn_clip_value=100", "+train.actor_single_step=true",
          "+model.actor_loss=score", "+train.stage3_diag=true"]
E2 = ["+train.adv_estimator=mc_loo", "+train.decision_discount=true", "model.gamma_denoising=1.0",
      "+model.logprob_reduce=sum", "+model.norm_adv=false"]


def run_specs(gate):
    artifact = read_json(gate, {}).get("rescale_path")
    if not isinstance(artifact, str) or not Path(artifact).is_absolute():
        raise ValueError("Gate must name the exact absolute frozen-scale artifact")
    specs = []
    for seed in (0, 1, 2):
        name = "FRS1_seed" + str(seed)
        specs.append({"run_id": name, "condition": "FRS1", "seed": seed, "mode": "train",
                      "overrides": COMMON + E2 + ["+train.actor_optimizer=sgd_diag",
                          "+train.rescale_floor=0.001", "+train.rescale_path=" + artifact,
                          "train.actor_lr=0.0001", "train.actor_lr_scheduler.min_lr=0.0001"],
                      "logdir": str(ROOT / name), "console_log": str(ROOT / (name + ".out")),
                      "cap_seconds": TRAIN_CAP, "estimate_seconds": 8100,
                      "estimate_basis": "E2_beta0/SGD10 all-diagnostic A6000 runs took 7204-7446 s; add allowance to 2h15m",
                      "monitor_interval_seconds": 1800, "gpu_type": "a6000", "gpu_count": 1,
                      "cpus": 40, "memory_gib": 64})
    for seed in (0, 1, 2):
        source = "FRS1_seed" + str(seed)
        name = "ckpt_" + source + "_seed5000"
        specs.append({"run_id": name, "condition": "ckpt_FRS1", "training_run": source,
                      "seed": 5000, "mode": "ckpt_eval", "overrides": COMMON + ["+train.mode=ckpt_eval",
                          "base_policy_path=" + str(ROOT / source / "checkpoint/state_139.pt")],
                      "logdir": str(ROOT / name), "console_log": str(ROOT / (name + ".out")),
                      "cap_seconds": EVAL_CAP, "estimate_seconds": 300,
                      "estimate_basis": "Stage3 final 80-episode-per-sampler evaluations took 45-46 s; allow 5 min",
                      "monitor_interval_seconds": 1800, "gpu_type": "a6000", "gpu_count": 1,
                      "cpus": 40, "memory_gib": 64})
    return specs


def validate_spec(run, gate):
    expected = {spec["run_id"]: spec for spec in run_specs(gate)}
    if run.get("run_id") not in expected or run != expected[run["run_id"]]:
        raise ValueError("Run specification differs from the approved FRS1-only six-job plan")
    return True


def validate_approval():
    approval = read_json(ROOT / "provenance/approval.json", {})
    required = {"approved": True, "conditions": ["FRS1"], "seeds": [0, 1, 2],
                "gpu_hour_cap": 14, "max_parallel_gpu_allocations": 3,
                "training_cap_seconds": TRAIN_CAP, "gate_cap_seconds": GATE_CAP,
                "ckpt_eval_cap_seconds": EVAL_CAP, "retries": 0, "gpu_type": "a6000",
                "cpus_per_gpu_job": 40, "memory_gib_per_gpu_job": 64}
    bad = {key: [value, approval.get(key)] for key, value in required.items() if approval.get(key) != value}
    if bad:
        raise ValueError("Missing or inconsistent addendum approval: " + json.dumps(bad))
    return approval


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Refusing to overwrite different existing plan evidence: " + str(path))
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


def plan(gate, commit):
    validate_approval()
    bound = validate_gate(gate, commit)
    source = validate_source(commit, JOURNAL, gate)
    specs = run_specs(gate)
    reserved = GATE_CAP + sum(spec["cap_seconds"] for spec in specs)
    if reserved > BUDGET:
        raise ValueError("Addendum plan exceeds 14 GPU-hours")
    for spec in specs:
        write_once(ROOT / "runs/specs" / (spec["run_id"] + ".json"), spec)
    record = {"scope": "FRS1 only, seeds 0-2, final checkpoint evaluation per completed seed",
              "parent_commit": commit, "gate": bound, "source": source,
              "gpu_budget_seconds": BUDGET, "reserved_gpu_seconds": reserved,
              "gate_reserved_seconds": GATE_CAP, "max_parallel_gpu_allocations": MAX_PARALLEL,
              "training_cap_seconds": TRAIN_CAP, "evaluation_cap_seconds": EVAL_CAP,
              "training_monitor_interval_seconds": 1800, "retries": 0, "runs": specs}
    write_once(ROOT / "runs/queue_plan.json", record)
    return record


def bind_source(gate, commit):
    """Match committed bytes/Git execute flags and exact current/gate execute bits."""
    gate = Path(gate).resolve()
    if ROOT not in gate.parents:
        raise ValueError("Original gate must be local to the FRS study")
    original = read_json(gate, {})
    if original.get("passed") is not True or original.get("original_gate_path"):
        raise ValueError("Source binding requires the original passed numerical gate")
    current = command(["git", "-C", str(PARENT), "rev-parse", "HEAD"], JOURNAL)["stdout"].strip()
    branch = command(["git", "-C", str(PARENT), "branch", "--show-current"], JOURNAL)["stdout"].strip()
    if current != commit or branch != "stage3-theory":
        raise ValueError("Commit binding requires the current parent stage3-theory revision")
    snapshot = Path(original.get("source_snapshot", "")).resolve()
    if ROOT not in snapshot.parents or snapshot == SOURCE:
        raise ValueError("Original gate source snapshot must be a separate local artifact")
    actual, expected = source_files(SOURCE), source_files(snapshot)
    if not expected or actual.keys() != expected.keys():
        raise ValueError("Current scientific source file set differs from numerical gate")
    prefix = str(SOURCE.relative_to(PARENT)) + "/"
    argv = ["git", "-C", str(PARENT), "archive", "--format=tar", commit, prefix[:-1]]
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    append_json(JOURNAL, {"utc": utc(), "command": argv, "returncode": result.returncode,
                         "stderr": result.stderr.decode("utf-8", errors="replace"),
                         "stdout": "Git archive inspected in memory; no archive artifact written"})
    if result.returncode:
        raise ValueError("Cannot inspect committed ordinary FRS source")
    committed = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError("Committed study source must contain ordinary files only")
            name = member.name[len(prefix):]
            committed[name] = (archive.extractfile(member).read(), member.mode & 0o111)
    if committed.keys() != expected.keys():
        raise ValueError("Committed scientific source file set differs from numerical gate snapshot")
    for name, path in expected.items():
        data, executable = path.read_bytes(), path.stat().st_mode & 0o111
        # Git records one executable flag (the owner's execute bit), not all
        # permission bits. Local source must still retain the gate's exact bits.
        committed_data, committed_executable = committed[name]
        if (committed_data != data or bool(committed_executable & 0o100) != bool(executable & 0o100) or
                actual[name].read_bytes() != data or
                actual[name].stat().st_mode & 0o111 != executable):
            raise ValueError("Committed/current source differs from numerical gate: " + name)
    artifact = Path(original.get("rescale_path", "")).resolve()
    if ROOT not in artifact.parents or not artifact.is_file():
        raise ValueError("Original passed gate's frozen rescale artifact is missing")
    bound = dict(original, original_gate_path=str(gate), reference_parent_commit=original.get("parent_commit"),
                 parent_commit=commit, parent_branch=branch, committed_source_byte_identical=True,
                 committed_source_files_compared=len(expected), bound_utc=utc(),
                 binding_note="Original pre-commit numerical gate preserved unchanged; committed and current ordinary source match its full snapshot byte-for-byte. The committed Git executable flag matches; current source retains the gate snapshot's full execute permission bits.")
    destination = ROOT / "provenance/committed_gate.json"
    write_once(destination, bound)
    return bound


def accounting(job, journal=JOURNAL):
    output = command(["sacct", "--allocations", "--noheader", "--parsable2", "--jobs=" + str(job),
                      "--format=JobIDRaw,State,ElapsedRaw,AllocTRES%180"], journal)["stdout"]
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] == str(job):
            tres = dict(part.split("=", 1) for part in fields[3].split(",") if "=" in part)
            count = int(tres.get("gres/gpu", "0"))
            return {"job_id": str(job), "scheduler_state": fields[1].split()[0].rstrip("+"),
                    "elapsed_seconds": int(fields[2]), "allocated_tres": fields[3],
                    "gpu_count": count, "allocated_gpu_seconds": int(fields[2]) * count}
    return {"job_id": str(job), "scheduler_state": None}


def scheduler_jobs():
    output = command(["squeue", "--noheader", "--user=" + os.environ.get("USER", "zbao7"),
                      "--format=%i|%T|%b|%j|%P"], JOURNAL)["stdout"]
    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        job, state, gres, name, partition = line.split("|", 4)
        gpus = sum(int(re.search(r"[:=](\d+)$", token).group(1))
                   for token in gres.split(",") if "gpu" in token and re.search(r"[:=](\d+)$", token))
        jobs.append({"job_id": job.strip(), "state": state.strip(), "gpus": gpus,
                     "name": name.strip(), "partition": partition.strip()})
    return jobs


def available_slots(submitted, jobs):
    owned = {x["job_id"] for x in submitted if x.get("scheduler_state") not in SLURM_TERMINAL}
    others = [j for j in jobs if j["job_id"] not in owned]
    return max(0, min(MAX_PARALLEL - len(owned), 12 - len(owned) - sum(j["gpus"] for j in others),
                      12 - len(jobs), 20 - len(jobs)))


class Campaign:
    def __init__(self, args):
        self.args = args
        self.stop = False
        self.started = time.monotonic()
        self.lock = (ROOT / "runs/campaign.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.plan = plan(args.gate, args.commit)
        self.specs = {run["run_id"]: run for run in self.plan["runs"]}
        if STATE.exists():
            raise ValueError("Campaign evidence already exists; automatic restart or retry is forbidden")
        submission = read_json(ROOT / "provenance/gate_submission.json", {})
        if not str(submission.get("job_id", "")).isdigit() or not 0 < submission.get("allocation_cap_seconds", 0) <= GATE_CAP:
            raise ValueError("Gate allocation record must document one approved at-most-one-hour allocation")
        allocation = accounting(submission["job_id"])
        if allocation.get("scheduler_state") != "COMPLETED" or allocation.get("gpu_count") != 1:
            raise ValueError("Passed gate must finish its one-GPU allocation before training launches")
        if allocation["allocated_gpu_seconds"] > GATE_CAP:
            raise ValueError("Gate exceeded its aggregate one-GPU-hour cap")
        self.state = {"status": "running", "started_utc": utc(), "parent_commit": args.commit,
                      "gate": str(args.gate), "gate_allocation": allocation, "submitted": [],
                      "skipped": [], "monitor_status": "active", "gpu_budget_seconds": BUDGET,
                      "all_planned_caps_seconds": self.plan["reserved_gpu_seconds"]}
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda signum, frame: setattr(self, "stop", True))
        self.next_monitor = 0
        self.record()

    def record(self):
        self.state["reserved_gpu_seconds"] = GATE_CAP + sum(e["cap_seconds"] for e in self.state["submitted"])
        self.state["measured_gpu_seconds_including_gate"] = (self.state["gate_allocation"]["allocated_gpu_seconds"] +
            sum(e.get("allocated_gpu_seconds", 0) for e in self.state["submitted"]))
        if max(self.state["reserved_gpu_seconds"], self.state["measured_gpu_seconds_including_gate"]) > BUDGET:
            raise RuntimeError("Hard addendum GPU-hour budget exceeded")
        self.state["updated_utc"] = utc()
        write_json(STATE, safe_json(self.state))

    def inspect_jobs(self):
        for entry in self.state["submitted"]:
            if entry.get("scheduler_state") in SLURM_TERMINAL and entry.get("terminal_recorded"):
                continue
            status = accounting(entry["job_id"])
            if status.get("scheduler_state") is None:
                entry.setdefault("unknown_since", time.time())
                if time.time() - entry["unknown_since"] > 300:
                    raise RuntimeError("Owned scheduler job unobservable for five minutes")
                continue
            entry.pop("unknown_since", None)
            entry.update(status)
            if status["scheduler_state"] in SLURM_TERMINAL:
                path = ROOT / "runs" / entry["run_id"] / "manifest.json"
                manifest = read_json(path, {})
                if status["scheduler_state"] != "COMPLETED" and manifest.get("status") == "complete":
                    manifest.update(status="failed", prior_worker_status="complete", scheduler_observation=status,
                                    failure_reason="Scheduler did not complete successfully despite worker output",
                                    monitor_status="paused")
                    write_json(path, safe_json(manifest))
                if manifest.get("status") not in TERMINAL:
                    observation = observe(self.specs[entry["run_id"]])
                    manifest.update(self.specs[entry["run_id"]], status="failed", ended_utc=utc(),
                                    monitor_status="paused", scheduler_observation=status,
                                    last_observation=observation,
                                    failure_reason="Allocation terminated before worker completion verification")
                    write_json(path, safe_json(manifest))
                entry.update(terminal_recorded=True, run_status=manifest["status"])
        self.record()

    def submit(self, run):
        validate_source(self.args.commit, JOURNAL, self.args.gate)
        validate_gate(self.args.gate, self.args.commit)
        validate_spec(run, self.args.gate)
        if any(e["run_id"] == run["run_id"] for e in self.state["submitted"]):
            raise ValueError("Duplicate submission is forbidden")
        if Path(run["console_log"]).exists() or (ROOT / "runs" / run["run_id"] / "manifest.json").exists():
            raise ValueError("Existing run evidence prohibits launch")
        if self.state["reserved_gpu_seconds"] + run["cap_seconds"] > BUDGET:
            raise ValueError("Submission would exceed 14 GPU-hours")
        argv = ["sbatch", "--parsable", "--account=soal", "--partition=soal", "--nodes=1", "--ntasks=1",
                "--cpus-per-task=40", "--mem=64G", "--gres=gpu:a6000:1", "--no-requeue",
                "--export=PATH,HOME,USER,LOGNAME,LANG", "--time=" + str(math.ceil(run["cap_seconds"] / 60)),
                "--signal=B:TERM@30", "--job-name=frs-" + run["run_id"], "--chdir=" + str(SOURCE),
                "--output=" + str(ROOT / "setup" / (run["run_id"] + "-%j.out")),
                str(ROOT / "code/run_worker.sh"), "--spec", str(ROOT / "runs/specs" / (run["run_id"] + ".json")),
                "--commit", self.args.commit, "--gate", str(self.args.gate)]
        self.state["pending_submission_intent"] = {"utc": utc(), "run_id": run["run_id"], "command": argv}
        self.record()
        result = command(argv, JOURNAL, check=False)
        job = result["stdout"].strip().split(";")[0]
        if result["returncode"] or not job.isdigit():
            self.state["submission_failure"] = result
            raise RuntimeError("Submission failed or ambiguous; no retry")
        self.state["submitted"].append({"run_id": run["run_id"], "job_id": job, "submitted_utc": utc(),
                                        "cap_seconds": run["cap_seconds"], "scheduler_state": "PENDING", "command": argv})
        self.state.pop("pending_submission_intent", None)
        self.record()

    def dispatch(self):
        if any(e.get("unknown_since") for e in self.state["submitted"]):
            return
        slots = available_slots(self.state["submitted"], scheduler_jobs())
        if not self.state["submitted"]:
            if slots >= 3:
                for seed in (0, 1, 2):
                    self.submit(self.specs["FRS1_seed" + str(seed)])
            return
        seen = {e["run_id"] for e in self.state["submitted"]} | set(self.state["skipped"])
        for run in self.specs.values():
            if run["run_id"] in seen or run["mode"] != "ckpt_eval":
                continue
            training = read_json(ROOT / "runs" / run["training_run"] / "manifest.json", {})
            entry = next(e for e in self.state["submitted"] if e["run_id"] == run["training_run"])
            if not entry.get("terminal_recorded"):
                continue
            if training.get("status") != "complete":
                skipped = dict(run, status="skipped", reason="Training seed did not complete; no substitute checkpoint evaluation",
                               monitor_status="paused", recorded_utc=utc())
                write_once(ROOT / "runs" / run["run_id"] / "manifest.json", skipped)
                self.state["skipped"].append(run["run_id"])
                self.record()
            elif slots > 0:
                self.submit(run)
                slots -= 1

    def monitor(self, terminal=False):
        observation = {"utc": utc(), "status": self.state["status"], "terminal": terminal,
                       "allocated_gpu_hours": self.state["measured_gpu_seconds_including_gate"] / 3600,
                       "reserved_gpu_hours": self.state["reserved_gpu_seconds"] / 3600,
                       "budget_gpu_hours": 14, "runs": []}
        for run in self.specs.values():
            manifest = read_json(ROOT / "runs" / run["run_id"] / "manifest.json", {})
            observation["runs"].append({"run_id": run["run_id"], "status": manifest.get("status", "not_started"),
                                        "progress": observe(run)})
        append_json(ROOT / "runs/campaign_monitor_history.jsonl", safe_json(observation))
        write_json(ROOT / "runs/campaign_progress.json", safe_json(observation))
        report = ROOT / "experiment.md"
        if report.exists():
            start, end = "<!-- frs-status-start -->", "<!-- frs-status-end -->"
            text = report.read_text()
            if start in text and end in text:
                statuses = "; ".join(item["run_id"] + ": " + item["status"] for item in observation["runs"])
                block = start + "\nLast monitor: " + observation["utc"] + "; **" + self.state["status"] + "**. " + statuses
                block += ". Monitor " + ("paused" if terminal else "active, every 30 minutes") + ".\n" + end
                report.write_text(text[:text.index(start)] + block + text[text.index(end) + len(end):])
        self.next_monitor = time.monotonic() + 1800

    def run(self):
        try:
            while True:
                if self.stop or time.monotonic() - self.started >= self.args.timeout_seconds - 60:
                    raise RuntimeError("Controller interrupted or time cap reached; admissions stopped, workers retain own caps")
                self.inspect_jobs()
                self.dispatch()
                submitted = self.state["submitted"]
                if len(submitted) + len(self.state["skipped"]) == len(self.specs) and all(e.get("terminal_recorded") for e in submitted):
                    for entry in submitted:
                        if entry.get("run_status") == "complete":
                            verify_complete(self.specs[entry["run_id"]], observe(self.specs[entry["run_id"]]))
                    failed = [e["run_id"] for e in submitted if e.get("run_status") != "complete"]
                    self.state.update(status="complete_with_failures" if failed else "complete",
                                      failures=failed, ended_utc=utc(), monitor_status="paused")
                    self.record()
                    self.monitor(terminal=True)
                    return 0
                if time.monotonic() >= self.next_monitor:
                    self.monitor()
                time.sleep(30)
        except Exception as exc:
            self.state.update(status="blocked", failure_reason=str(exc), ended_utc=utc(), monitor_status="paused")
            self.record()
            self.monitor(terminal=True)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bind-source", "plan", "run"))
    parser.add_argument("--gate", type=Path, default=GATE_DEFAULT)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    args = parser.parse_args()
    args.gate = args.gate.resolve()
    if not 60 <= args.timeout_seconds <= 86400:
        raise ValueError("Controller must have an at-most-24-hour cap")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "setup").mkdir(exist_ok=True)
    if args.action == "bind-source":
        print(json.dumps(bind_source(args.gate, args.commit), indent=2))
        return 0
    if args.action == "plan":
        print(json.dumps(plan(args.gate, args.commit), indent=2))
        return 0
    return Campaign(args).run()


if __name__ == "__main__":
    sys.exit(main())
