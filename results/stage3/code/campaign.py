"""Bounded, resumable Slurm dispatcher for the user-approved Stage-3 waves.

This process receives no GPU. Every scientific process has a distinct allocation,
directory and timeout. Failed runs are recorded once and never retried. Stopping
this controller stops admission only; independent owned workers retain their caps.
"""
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import sys
import time

from monitor import (ROOT, SOURCE, TERMINAL, SLURM_TERMINAL, append_json, command,
                     observe, read_json, safe_json, utc, validate_gate, validate_source,
                     write_json)

KLSTAR = 0.000584517336086918
RADIUS = 8.686025494700294
GPU_BUDGET_SECONDS = 200 * 3600
GATE_RESERVED_SECONDS = 3600  # capture 900s + remaining CPU/GPU gate allocation 2700s
MAX_PARALLEL = 9
STATE = ROOT / "runs/campaign.json"
JOURNAL = ROOT / "runs/campaign_commands.jsonl"
COMMON = ["model.clip_ploss_coef=1e6", "model.clip_ploss_coef_base=1e6", "train.target_kl=null",
          "+model.clamp_logprob=false", "model.randn_clip_value=100", "+train.actor_single_step=true",
          "+model.actor_loss=score", "+train.stage3_diag=true"]
E2 = ["+train.adv_estimator=mc_loo", "+train.decision_discount=true", "model.gamma_denoising=1.0",
      "+model.logprob_reduce=sum", "+model.norm_adv=false"]
TRAIN_CONDITIONS = ["E0", "E1", "E2", "SGD1", "SGD3", "SGD10", "E2_beta0", "E2_nobase", "SGD03"]
WAVE2_CONDITIONS = ["PROJ3", "PROJ10", "BATCH4"]
NC4_CHECKPOINT = ROOT.parent / "stage2_noclip/runs/NC4_lr1e-4_seed0/native/checkpoint/state_139.pt"


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Refusing to overwrite an existing job specification: " + str(path))
        return
    with path.open("x") as stream:
        stream.write(content)


def run_spec(condition, seed, extra, wave, mode="train"):
    name = condition + "_seed" + str(seed)
    cap, estimate = 14400, 10800
    if condition == "BATCH4":
        cap, estimate = 57600, 43200
    elif mode == "noise_scale":
        cap, estimate = 10800, (5400 if condition == "ns_theta0" else 4500)
    elif mode == "calibrate":
        cap, estimate = 3600, 1200
    elif mode == "ckpt_eval":
        cap, estimate = (1800, 480) if condition == "ckpt_theta0" else (900, 300)
    return {"run_id": name, "condition": condition, "seed": seed, "mode": mode, "wave": wave,
            "overrides": COMMON + list(extra), "logdir": str(ROOT / name),
            "console_log": str(ROOT / (name + ".out")), "cap_seconds": cap,
            "estimate_seconds": estimate, "monitor_interval_seconds": max(1800, estimate // 6),
            "gpu_type": "a6000", "gpu_count": 1, "cpus": 40, "memory_gib": 64,
            "estimate_basis": "NC4 1e-4: 8489–8674 seconds; central Stage3 estimate includes split-gradient and saturation passes"}


def wave0():
    return [run_spec("cal", 3000, ["+train.mode=calibrate", "+train.calib_kl_target=" + repr(KLSTAR)] + E2, 0, "calibrate"),
            run_spec("ns_theta0", 1000, ["+train.mode=noise_scale", "+train.ns_warmup_batches=20", "+train.ns_batches=25"], 0, "noise_scale"),
            run_spec("ns_ckpt", 2000, ["+train.mode=noise_scale", "+train.ns_warmup_batches=5", "+train.ns_batches=25",
                                      "base_policy_path=" + str(NC4_CHECKPOINT)], 0, "noise_scale")]


def sgd(eta):
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return E2 + ["+train.actor_optimizer=sgd", "train.actor_lr=" + repr(eta),
                 "train.actor_lr_scheduler.min_lr=" + repr(eta)]


def wave1(eta):
    extras = {"E0": [], "E1": ["+train.adv_estimator=mc_loo"], "E2": E2,
              "SGD1": sgd(eta), "SGD3": sgd(3 * eta), "SGD10": sgd(10 * eta),
              "E2_beta0": E2 + ["+train.actor_beta1=0.0"],
              "E2_nobase": [x if not x.startswith("+train.adv_estimator=") else "+train.adv_estimator=mc_none" for x in E2],
              "SGD03": sgd(eta / 3)}
    return [run_spec(name, seed, extras[name], 1) for name in TRAIN_CONDITIONS for seed in (0, 1, 2)]


def wave2(eta):
    extras = {"PROJ3": sgd(3 * eta) + ["+train.proj_radius=" + repr(RADIUS)],
              "PROJ10": sgd(10 * eta) + ["+train.proj_radius=" + repr(RADIUS)],
              "BATCH4": sgd(3 * eta) + ["train.n_steps=2000"]}
    return [run_spec(name, seed, extras[name], 2) for name in WAVE2_CONDITIONS for seed in (0, 1, 2)]


def evaluation(source_run=None):
    if source_run is None:
        return run_spec("ckpt_theta0", 5000, ["+train.mode=ckpt_eval", "+train.ckpt_eval_repeats=3"], 3, "ckpt_eval")
    return run_spec("ckpt_" + source_run["run_id"], 5000,
                    ["+train.mode=ckpt_eval", "base_policy_path=" + str(Path(source_run["logdir"]) / "checkpoint/state_139.pt")],
                    3, "ckpt_eval")


def save_specs(runs, filename):
    """Human job files spell out COMMON too; workers execute their JSON counterparts."""
    grouped = {}
    for run in runs:
        key = (run["condition"], tuple(run["overrides"]))
        grouped.setdefault(key, []).append(run["seed"])
        text = json.dumps(run, indent=2, allow_nan=False) + "\n"
        write_once(ROOT / "runs/specs" / (run["run_id"] + ".json"), text)
    lines = ["# NAME SEEDS OVERRIDES; every effective condition override is written in full."]
    for (name, overrides), seeds in grouped.items():
        lines.append(name + " " + ",".join(map(str, seeds)) + " " + " ".join(overrides))
    write_once(ROOT / filename, "\n".join(lines) + "\n")


def prepare():
    if not NC4_CHECKPOINT.is_file():
        raise ValueError("Approved NC4 checkpoint is unavailable")
    anchors = read_json(ROOT / "provenance/anchors.json", {})
    if anchors.get("kl_star") != KLSTAR or anchors.get("projection_radius") != RADIUS:
        raise ValueError("Queue constants disagree with the recorded data-derived anchors")
    save_specs(wave0(), "jobs_wave0.txt")
    plan = {"kl_star": KLSTAR, "projection_radius": RADIUS,
            "kl_source": "Pooled median over the 378 Stage2 NC4 1e-4 training iterations",
            "radius_source": "1.25 times maximum final actor_ft distance over Stage1 seeds 0,1,2",
            "gate_reserved_seconds": GATE_RESERVED_SECONDS, "gpu_budget_seconds": GPU_BUDGET_SECONDS,
            "max_parallel_gpu_allocations": MAX_PARALLEL, "max_training_runs": 36,
            "max_gpu_hours_from_caps": 197.5, "retries": 0,
            "normal_estimate_seconds": 10800, "batch4_estimate_seconds": 43200,
            "normal_monitor_interval_seconds": 1800, "batch4_monitor_interval_seconds": 7200,
            "optional_included": ["E2_nobase", "SGD03", "BATCH4"],
            "optional_excluded": ["per-denoising-step Phase1 variance", "Stage2 degraded-checkpoint evaluations"],
            "normal_training_cap_seconds": 14400, "batch4_cap_seconds": 57600,
            "wave0_specs": [r["run_id"] for r in wave0()],
            "later_job_files": "Written with numeric eta_star after verified calibration; no placeholders are executable"}
    write_once(ROOT / "runs/queue_plan.json", json.dumps(plan, indent=2) + "\n")
    return plan


def scheduler_jobs(journal=JOURNAL):
    text = command(["squeue", "--noheader", "--user=" + os.environ.get("USER", "zbao7"),
                    "--format=%i|%T|%b|%j|%P"], journal)["stdout"]
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        job, state, resources, name, partition = line.split("|", 4)
        gpu = 0
        for token in resources.split(","):
            if "gpu" in token:
                match = re.search(r"[:=](\d+)$", token)
                gpu += int(match.group(1)) if match else 1
        jobs.append({"job_id": job.strip(), "state": state.strip(), "gpus": gpu,
                     "name": name.strip(), "partition": partition.strip()})
    return jobs


def accounting(job_id, journal=JOURNAL):
    text = command(["sacct", "--allocations", "--noheader", "--parsable2", "--jobs=" + str(job_id),
                    "--format=JobIDRaw,State,ElapsedRaw,AllocTRES%160"], journal)["stdout"]
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] == str(job_id):
            state = fields[1].split()[0].rstrip("+")
            return {"job_id": str(job_id), "state": state, "elapsed_seconds": int(fields[2]), "allocated_tres": fields[3]}
    return {"job_id": str(job_id), "state": None}


def remaining_slots(state, all_jobs, maximum):
    owned = {entry["job_id"] for entry in state["submitted"] if entry.get("scheduler_state") not in SLURM_TERMINAL}
    own_count = len(owned)
    others = [job for job in all_jobs if job["job_id"] not in owned]
    gpu_other = sum(job["gpus"] for job in others)
    # Be conservative: submitted allocations count against running capacity too,
    # so admission cannot intentionally oversubscribe the observed account caps.
    return max(0, min(maximum - own_count, 12 - own_count - gpu_other,
                      12 - len(all_jobs), 20 - len(all_jobs)))


def projection_decision(condition, runs):
    maxima = []
    reason = "all matching SGD seeds completed and all distances are below 0.8 R"
    for seed in (0, 1, 2):
        name = condition + "_seed" + str(seed)
        run = next(r for r in runs if r["run_id"] == name)
        manifest = read_json(ROOT / "runs" / name / "manifest.json", {})
        if manifest.get("status") != "complete":
            return {"skip": False, "reason": "A matching SGD seed failed; slack cannot be established from an incomplete run",
                    "source_condition": condition, "radius": RADIUS, "threshold": 0.8 * RADIUS}
        with (Path(run["logdir"]) / "result.pkl").open("rb") as stream:
            rows = pickle.load(stream)
        values = [float(r["theta_dist_preproj"]) for r in rows if "train_episode_reward" in r]
        if len(values) != 126 or not all(math.isfinite(x) for x in values):
            raise ValueError("Missing finite theta_dist_preproj in " + name)
        maxima.append(max(values))
    return {"skip": max(maxima) < 0.8 * RADIUS, "seed_maxima": maxima, "maximum": max(maxima),
            "radius": RADIUS, "threshold": 0.8 * RADIUS, "source_condition": condition,
            "reason": reason if max(maxima) < 0.8 * RADIUS else "Projection skip threshold was reached"}


class Campaign:
    def __init__(self, args):
        self.args = args
        self.interrupted = False
        self.clock_start = time.monotonic()
        ROOT.joinpath("runs").mkdir(exist_ok=True)
        self.lock = (ROOT / "runs/campaign.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        approval = read_json(args.approval)
        if (not isinstance(approval, dict) or not approval.get("approval") or
                approval.get("gpu_hour_cap") != 200 or
                set(approval.get("optional_conditions_included", [])) != {"E2_nobase", "SGD03", "BATCH4"} or
                approval.get("max_parallel_gpu_allocations") != 9 or
                approval.get("automatic_retries") is not False):
            raise ValueError("Approval must explicitly bind 200 GPU-hours and the three listed optional conditions")
        if not os.environ.get("SLURM_JOB_ID", "").isdigit():
            raise ValueError("Run the controller in a CPU-only Slurm allocation")
        if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE") not in (None, "", "0"):
            raise ValueError("Controller must be CPU-only")
        validate_gate(args.gate, args.commit)
        validate_source(args.commit, JOURNAL)
        if not Path(args.finalizer).is_file() or Path(args.finalizer).resolve() != ROOT / "code/analyze_stage3.py":
            raise ValueError("The approved CPU-only Stage-3 finalizer must exist before launch")
        prepare()
        self.state = read_json(STATE)
        if self.state:
            if not args.resume:
                raise ValueError("Campaign already exists; explicit --resume adopts the same jobs without retries")
            if self.state["commit"] != args.commit or self.state["approval"] != str(Path(args.approval).resolve()):
                raise ValueError("Cannot resume a different scope or source")
            old_controller = self.state.get("controller_job_id")
            if old_controller != os.environ["SLURM_JOB_ID"]:
                old = accounting(old_controller)
                if old["state"] not in SLURM_TERMINAL:
                    raise ValueError("Previous controller must be terminal before adoption")
            self.state.setdefault("controller_adoptions", []).append({"utc": utc(), "previous": old_controller,
                                                                     "new": os.environ["SLURM_JOB_ID"]})
        else:
            self.state = {"status": "running", "started_utc": utc(), "commit": args.commit,
                          "gate": str(Path(args.gate).resolve()), "approval": str(Path(args.approval).resolve()),
                          "gpu_budget_seconds": GPU_BUDGET_SECONDS, "gate_reserved_seconds": GATE_RESERVED_SECONDS,
                          "submitted": [], "runs": wave0(), "wave1_created": False,
                          "wave2_created": False, "evaluations_created": False, "retries": 0,
                          "projection_decisions": {}, "monitor_status": "active"}
        self.state.update(status="running", controller_job_id=os.environ["SLURM_JOB_ID"], max_parallel=args.max_parallel)
        gate_jobs = []
        for filename in ("capture_submission.json", "gate_submission.json"):
            submission = read_json(ROOT / "provenance" / filename)
            if submission:
                job = str(submission.get("job_id", ""))
                if not job.isdigit():
                    raise ValueError("Gate allocation record lacks a numeric job_id")
                observed = accounting(job)
                if observed["state"] not in SLURM_TERMINAL:
                    raise ValueError("Capture/gate allocations must be terminal before wave0")
                gate_jobs.append(dict(observed, submission_record=str(ROOT / "provenance" / filename)))
        if len(gate_jobs) != 2 or len({job["job_id"] for job in gate_jobs}) != 2:
            raise ValueError("Both unique capture and gate allocation records are required")
        if sum(job["elapsed_seconds"] for job in gate_jobs) > GATE_RESERVED_SECONDS:
            raise ValueError("Capture and gate allocations exceed the approved aggregate one-hour budget")
        self.state["gate_allocations"] = gate_jobs
        self.next_report = time.monotonic()
        self.last_progress = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "interrupted", True))
        self.record()

    def record(self):
        self.state["updated_utc"] = utc()
        self.state["reserved_gpu_seconds"] = GATE_RESERVED_SECONDS + sum(x["cap_seconds"] for x in self.state["submitted"])
        self.state["measured_gpu_seconds_submitted"] = sum(x.get("elapsed_seconds", 0) for x in self.state["submitted"])
        self.state["measured_gate_gpu_seconds"] = sum(x.get("elapsed_seconds", 0) for x in self.state.get("gate_allocations", []))
        self.state["measured_gpu_seconds_including_gate"] = (self.state["measured_gpu_seconds_submitted"] +
                                                              self.state["measured_gate_gpu_seconds"])
        if self.state["reserved_gpu_seconds"] > GPU_BUDGET_SECONDS:
            raise RuntimeError("Hard reserved GPU-hour budget exceeded")
        write_json(STATE, self.state)

    def manifest(self, run):
        return read_json(ROOT / "runs" / run["run_id"] / "manifest.json", {})

    def status(self, run):
        if run.get("skip"):
            return "skipped"
        return self.manifest(run).get("status", "planned")

    def refresh(self):
        jobs = scheduler_jobs()
        active = {j["job_id"]: j for j in jobs}
        unknown = False
        for entry in self.state["submitted"]:
            if entry.get("scheduler_state") in SLURM_TERMINAL:
                continue
            if entry["job_id"] in active:
                entry["scheduler_state"] = active[entry["job_id"]]["state"]
                entry.pop("unobservable_since", None)
                continue
            observed = accounting(entry["job_id"])
            entry.update(scheduler_state=observed["state"], elapsed_seconds=observed.get("elapsed_seconds", 0))
            if observed["state"] is None:
                unknown = True
                entry.setdefault("unobservable_since", time.time())
                if time.time() - entry["unobservable_since"] > 300:
                    raise RuntimeError("Owned allocation unobservable for five minutes: " + entry["job_id"])
                continue
            if observed["state"] in SLURM_TERMINAL:
                run = next(r for r in self.state["runs"] if r["run_id"] == entry["run_id"])
                manifest = self.manifest(run)
                if manifest.get("status") not in TERMINAL:
                    # Scheduler termination can prevent a worker's final write.
                    # Retain partial evidence and do not turn it into a retry.
                    manifest.update(run, status="failed", failure_reason="Scheduler terminated before terminal verification",
                                    scheduler_observation=observed, ended_utc=utc(), monitor_status="paused")
                    manifest["last_observation"] = observe(run)
                    write_json(ROOT / "runs" / run["run_id"] / "manifest.json", safe_json(manifest))
        self.state["dispatch_paused_for_unknown_scheduler_state"] = unknown
        self.state["observed_user_jobs"] = jobs
        self.record()
        return jobs

    def create_later_waves(self):
        cal = next(r for r in self.state["runs"] if r["condition"] == "cal")
        if not self.state["wave1_created"] and self.status(cal) == "complete":
            value = read_json(Path(cal["logdir"]) / "calibration.json")
            eta = float(value["eta_star"])
            if not math.isfinite(eta) or eta <= 0:
                raise ValueError("Invalid eta_star in completed calibration")
            runs = wave1(eta)
            save_specs(runs, "jobs_wave1.txt")
            self.state.update(eta_star=eta, eta_theory=eta / 2500, wave1_created=True)
            self.state["runs"].extend(runs)
            self.record()
        if self.state["wave1_created"] and not self.state["wave2_created"]:
            sgds = [r for r in self.state["runs"] if r["condition"] in ("SGD1", "SGD3", "SGD10", "SGD03")]
            if sgds and all(self.status(r) in TERMINAL for r in sgds):
                runs = wave2(self.state["eta_star"])
                for proj, source in (("PROJ3", "SGD3"), ("PROJ10", "SGD10")):
                    decision = projection_decision(source, self.state["runs"])
                    self.state["projection_decisions"][proj] = decision
                    if decision["skip"]:
                        for run in runs:
                            if run["condition"] == proj:
                                run["skip"] = decision
                                write_json(ROOT / "runs" / run["run_id"] / "manifest.json",
                                           dict(run, status="skipped", monitor_status="paused", recorded_utc=utc()))
                save_specs(runs, "jobs_wave2.txt")
                self.state["runs"].extend(runs)
                self.state["wave2_created"] = True
                self.record()
        if self.state["wave2_created"] and not self.state["evaluations_created"]:
            training = [r for r in self.state["runs"] if r["mode"] == "train"]
            if all(self.status(r) in TERMINAL for r in training):
                runs = [evaluation(r) for r in training if self.status(r) == "complete"] + [evaluation()]
                save_specs(runs, "jobs_ckpt_eval.txt")
                self.state["runs"].extend(runs)
                self.state["evaluations_created"] = True
                self.record()

    def submit(self, run):
        if any(x["run_id"] == run["run_id"] for x in self.state["submitted"]):
            raise ValueError("Automatic duplicate submission is forbidden")
        reserve = GATE_RESERVED_SECONDS + sum(x["cap_seconds"] for x in self.state["submitted"]) + run["cap_seconds"]
        if reserve > GPU_BUDGET_SECONDS:
            raise RuntimeError("Next allocation would exceed the hard GPU-hour reservation budget")
        validate_source(self.args.commit, JOURNAL)
        validate_gate(self.args.gate, self.args.commit)
        if (ROOT / "runs" / run["run_id"] / "manifest.json").exists() or Path(run["console_log"]).exists():
            raise ValueError("Existing run evidence forbids a fresh launch: " + run["run_id"])
        spec = ROOT / "runs/specs" / (run["run_id"] + ".json")
        argv = ["sbatch", "--parsable", "--account=soal", "--partition=soal", "--nodes=1", "--ntasks=1",
                "--cpus-per-task=40", "--mem=64G", "--gres=gpu:a6000:1", "--no-requeue",
                "--export=PATH,HOME,USER,LOGNAME,LANG",
                "--time=" + str(math.ceil(run["cap_seconds"] / 60)), "--signal=B:TERM@30",
                "--job-name=stage3-" + run["run_id"], "--chdir=" + str(SOURCE),
                "--output=" + str(ROOT / "setup" / (run["run_id"] + "-%j.out")),
                str(ROOT / "code/run_worker.sh"), "--spec", str(spec), "--commit", self.args.commit,
                "--gate", str(Path(self.args.gate).resolve())]
        # Persist intent before invoking sbatch. If the response is ambiguous,
        # stop; never submit again and risk an unrecorded duplicate allocation.
        intent = {"utc": utc(), "run_id": run["run_id"], "command": argv, "reserved_gpu_seconds": reserve}
        self.state["pending_submission_intent"] = intent
        self.record()
        result = command(argv, JOURNAL, check=False)
        job = result["stdout"].strip().split(";")[0]
        if result["returncode"] or not job.isdigit():
            self.state["submission_failure"] = result
            self.record()
            raise RuntimeError("Slurm submission failed or was ambiguous; no automatic retry")
        self.state["submitted"].append({"run_id": run["run_id"], "job_id": job,
                                        "submitted_utc": utc(), "cap_seconds": run["cap_seconds"],
                                        "scheduler_state": "PENDING", "command": argv})
        self.state.pop("pending_submission_intent", None)
        self.record()

    def dispatch(self, jobs):
        if self.state["dispatch_paused_for_unknown_scheduler_state"]:
            return
        available = remaining_slots(self.state, jobs, self.args.max_parallel)
        seen = {x["run_id"] for x in self.state["submitted"]}
        planned = [r for r in self.state["runs"] if r["run_id"] not in seen and not r.get("skip")]
        groups = []
        for run in planned:
            if run["mode"] == "train":
                if any(group[0]["condition"] == run["condition"] for group in groups):
                    continue
                group = [r for r in planned if r["condition"] == run["condition"]]
                # Resume may adopt a partially submitted condition but never retry a seed.
                groups.append(group)
            else:
                groups.append([run])
        for group in groups:
            if len(group) <= available:
                for run in group:
                    self.submit(run)
                available -= len(group)

    def report(self, final=False):
        statuses = {r["run_id"]: self.status(r) for r in self.state["runs"]}
        observations = {}
        for run in self.state["runs"]:
            if statuses[run["run_id"]] not in ("planned", "skipped"):
                observations[run["run_id"]] = observe(run)
        snapshot = {"utc": utc(), "status": self.state["status"], "statuses": statuses,
                    "observations": observations, "reserved_gpu_hours": self.state["reserved_gpu_seconds"] / 3600,
                    "budget_gpu_hours": 200, "max_parallel": self.args.max_parallel,
                    "controller_job_id": self.state["controller_job_id"]}
        append_json(ROOT / "runs/campaign_monitor_history.jsonl", safe_json(snapshot))
        write_json(ROOT / "runs/campaign_progress.json", safe_json(snapshot))
        # Edit only our marker block, preserving the user's report and prose.
        report = ROOT / "experiment.md"
        if report.exists():
            text = report.read_text()
            start, end = "<!-- stage3-status-start -->", "<!-- stage3-status-end -->"
            counts = {s: list(statuses.values()).count(s) for s in set(statuses.values())}
            block = (start + "\nLast monitor: " + snapshot["utc"] + "; campaign **" + self.state["status"] +
                     "**. Run states: " + ", ".join(k + " " + str(v) for k, v in sorted(counts.items())) +
                     ". [Live record](runs/campaign_progress.json); [command journal](runs/campaign_commands.jsonl).\n" + end)
            if start in text and end in text:
                text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
            else:
                text = text.rstrip() + "\n\n" + block + "\n"
            report.write_text(text)
        self.next_report = time.monotonic() + 1800

    def run(self):
        try:
            if self.state.get("pending_submission_intent"):
                raise RuntimeError("Unresolved prior sbatch intent; reconcile scheduler evidence before resuming")
            while True:
                if self.interrupted:
                    raise RuntimeError("Controller received a signal; admission stopped, existing workers retain their own caps")
                if time.monotonic() - self.clock_start >= self.args.timeout_seconds - 60:
                    raise RuntimeError("Controller time cap reached; admission stopped, existing workers retain their own caps")
                jobs = self.refresh()
                self.create_later_waves()
                cal = next(r for r in self.state["runs"] if r["condition"] == "cal")
                if self.status(cal) == "failed" and all(self.status(r) in TERMINAL for r in self.state["runs"]):
                    raise RuntimeError("Calibration failed; no eta is available for later waves")
                self.dispatch(jobs)
                statuses = [self.status(r) for r in self.state["runs"]]
                all_scheduler_terminal = all(x.get("scheduler_state") in SLURM_TERMINAL for x in self.state["submitted"])
                if self.state["evaluations_created"] and all(s in TERMINAL for s in statuses) and all_scheduler_terminal:
                    self.state.update(status="finalizing", runs_ended_utc=utc())
                    self.record()
                    finalizer = command([sys.executable, "-B", self.args.finalizer], JOURNAL, check=False, timeout=1800)
                    self.state["finalizer"] = finalizer
                    if finalizer["returncode"]:
                        raise RuntimeError("All runs terminal, but analysis finalizer failed; no automatic retry")
                    self.state.update(status="complete", ended_utc=utc(), monitor_status="paused")
                    self.record()
                    self.report(final=True)
                    print("Stage3 campaign terminal: all selected runs verified, failed, or prospectively skipped.", flush=True)
                    return 0
                if time.monotonic() >= self.next_report:
                    self.record()
                    self.report()
                time.sleep(30)
        except Exception as exc:
            self.state.update(status="blocked", error=str(exc), ended_utc=utc(), monitor_status="paused",
                              owned_workers_cancelled=False)
            self.record()
            self.report(final=True)
            print("Stage3 controller stopped admission: " + str(exc), file=sys.stderr, flush=True)
            return 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("--commit", required=True)
    run.add_argument("--gate", required=True)
    run.add_argument("--approval", required=True)
    run.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PAR", "9")))
    run.add_argument("--timeout-seconds", type=int, default=432000)
    run.add_argument("--finalizer", default=str(ROOT / "code/analyze_stage3.py"))
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), indent=2))
        return 0
    if not 1 <= args.max_parallel <= MAX_PARALLEL:
        raise ValueError("MAX_PAR must be between one and nine")
    # Whole three-seed conditions must be admitted together.
    if args.max_parallel < 3:
        raise ValueError("MAX_PAR needs at least three slots for the required parallel seed groups")
    if not 60 <= args.timeout_seconds <= 604800:
        raise ValueError("Controller cap must be between one minute and seven days")
    ROOT.joinpath("setup").mkdir(exist_ok=True)
    return Campaign(args).run()


if __name__ == "__main__":
    sys.exit(main())
