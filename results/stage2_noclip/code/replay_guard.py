"""R0-only live exact-replay gate, using frozen saved-row/console prefixes."""
import datetime as dt
import json
import pickle
import re
import time
from pathlib import Path

from replay_compare import ReplayInputError, _payload, compare_replay

BASE = "cc7234ad7ff39a8f32de3af903606723a16f0648"
SUMMARY = re.compile(r"^\d+: step ")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path, value, exclusive=False):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + ".pending")
        temporary.write_text(text)
        temporary.replace(path)


class ReplayGuard:
    def __init__(self, study_root, run_dir, console):
        self.root, self.run_dir, self.console = map(Path, (study_root, run_dir, console))
        self.directory = self.run_dir / "replay"
        self.directory.mkdir(exist_ok=False)
        self.status_path = self.run_dir / "replay_status.json"
        self.last_count, self.consumed_bytes, self.summary_ends = 0, 0, []
        self.wait_started, self.finished = None, False
        self.references, self.baseline = {}, []
        for name in ("P1", "P2"):
            manifest_path = self.root / "runs" / (name + "_seed0") / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("status") != "complete" or manifest.get("returncode") != 0
                    or manifest.get("verified_rows") != 20 or manifest.get("checkpoint_tensors_finite") is not True
                    or manifest.get("source_before", {}).get("commit") != BASE
                    or manifest.get("source_after", {}).get("commit") != BASE):
                raise ValueError("Replay requires fully verified pristine " + name)
            self.references[name.lower() + "_result"] = manifest["result_path"]
            self.references[name.lower() + "_log"] = str(self.root / (name + "_seed0.out"))
        stage1 = self.root.parent / "repro_halfcheetah"
        for name in ("seed0_handoff", "seed1", "seed2"):
            manifest = json.loads((stage1 / "runs" / name / "manifest.json").read_text())
            if manifest.get("status") != "complete" or manifest.get("verified_rows") != 140:
                raise ValueError("Canonical Stage 1 reference is not complete: " + name)
            self.baseline.append(manifest["result_path"])
        probe_report = compare_replay(**self.references)
        self.probe_report_path = self.directory / "probe_comparison.json"
        write_json(self.probe_report_path, probe_report, exclusive=True)
        self.last_report = {"comparison_status": "waiting_r0", "r0_passed": None,
                            "deterministic_probe": probe_report["deterministic_probe"],
                            "completed_compared_rows": 0, "updated_utc": timestamp(),
                            "probe_comparison_path": str(self.probe_report_path),
                            "baseline_results": self.baseline}
        write_json(self.status_path, self.last_report, exclusive=True)

    def observe_line(self, raw, terminated=True):
        # The supervisor calls this only after writing these raw bytes to .out.
        self.consumed_bytes += len(raw) + int(terminated)
        text = ANSI.sub("", raw.decode("utf-8", errors="strict"))
        _, payload = _payload(text)
        if payload.startswith("eval: ") or SUMMARY.match(payload):
            self.summary_ends.append(self.consumed_bytes)

    def failure(self, message, detail=None, snapshot=None):
        report = dict(self.last_report, comparison_status="invalid_input", r0_passed=False,
                      updated_utc=timestamp(), error=str(message), first_difference=detail)
        if snapshot:
            report["snapshot"] = snapshot
        write_json(self.status_path, report)
        write_json(self.directory / "failure.json", report, exclusive=True)
        self.last_report = report
        raise ValueError("R0 replay failed: " + str(message))

    def check(self, records, final=False):
        if self.finished:
            return
        eligible = min(len(records), len(self.summary_ends))
        if final and (len(records) != 140 or eligible != len(records)):
            self.failure("Final R0 result/complete-console count differs from 140",
                         {"saved_rows": len(records), "captured_summaries": len(self.summary_ends)})
        if eligible <= self.last_count and not final:
            return
        if eligible == 0:
            return
        boundary = self.summary_ends[eligible - 1]
        with self.console.open("rb") as stream:
            console_prefix = stream.read(boundary)
        if len(console_prefix) != boundary:
            # This is an explicit incomplete-I/O case, never a numeric tolerance.
            self.wait_started = self.wait_started or time.monotonic()
            if final or time.monotonic() - self.wait_started > 5:
                self.failure("Captured console bytes remain unavailable")
            return
        self.wait_started = None
        label = "final" if final else "rows_%03d" % eligible
        result_path = self.directory / (label + ".result.pkl")
        log_path = self.directory / (label + ".out")
        report_path = self.directory / (label + ".json")
        # Freeze the same eligible prefix in both sources before comparing.
        with result_path.open("xb") as stream:
            pickle.dump(records[:eligible], stream, protocol=pickle.HIGHEST_PROTOCOL)
        with log_path.open("xb") as stream:
            stream.write(console_prefix)
        snapshot = {"saved_rows": eligible, "console_bytes": boundary,
                    "result_path": str(result_path), "console_path": str(log_path)}
        try:
            full = compare_replay(**self.references, r0_result=result_path, r0_log=log_path,
                                  baseline_results=self.baseline)
        except (ReplayInputError, EOFError, pickle.UnpicklingError, OSError) as error:
            self.failure(error, getattr(error, "first_difference", None), snapshot)
        # The full immutable equality mask lives once in probe_comparison.json.
        report = {key: value for key, value in full.items() if key not in ("probe_comparison", "exceptions")}
        report["exceptions"] = {"R0": full["exceptions"].get("R0", [])}
        report.update({"updated_utc": timestamp(), "completed_compared_rows": eligible,
                       "probe_comparison_path": str(self.probe_report_path),
                       "baseline_results": self.baseline, "snapshot": snapshot,
                       "history_path": str(report_path)})
        write_json(report_path, report, exclusive=True)
        write_json(self.status_path, report)
        self.last_report, self.last_count = report, eligible
        if report["r0_passed"] is False:
            detail = report.get("r0_comparison", {}).get("first_difference")
            raise ValueError("R0 exact replay mismatch: " + json.dumps(detail, allow_nan=False))
        if eligible >= 20 and report["r0_passed"] is not True:
            self.failure("Twenty completed iterations did not produce an exact replay decision")
        if final:
            self.finished = True
