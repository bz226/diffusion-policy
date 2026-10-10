"""Synthetic stdout/pickle failure tests; no DPPO, environment, model or GPU run."""
import json
import math
import pickle
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from supervise_run import Supervisor


SUMMARY = ("[2026-10-09 00:00:00,000][agent][INFO] - 1: step 80000 | loss 0.1 | t: 1.0"
           " | kl_true_per_action=nan | logratio_p99=1.0 | clamp_hit_frac=0.0")


class NonfiniteEvidenceTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix="supervisor_unit_", dir=str(root / "tmp"))
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def supervisor(self, script):
        obj = Supervisor.__new__(Supervisor)
        obj.a = types.SimpleNamespace(run_id="synthetic", condition="NC1", command=[sys.executable, "-u", "-c", script])
        obj.repo = obj.run_dir = self.root
        obj.native = self.root / "native"
        obj.native.mkdir()
        obj.console = self.root / "synthetic.out"
        obj.clock = time.monotonic()
        obj.deadline = obj.clock + 12
        obj.next_monitor = obj.clock + 1800
        obj.interval = 1800
        obj.process = obj.interrupted = obj.trajectory_observer = None
        obj.checking_itr = None
        obj.collection_itr = -1
        obj.observed_nonfinite = False
        obj.nonfinite_iteration = obj.pending_nonfinite = None
        obj.records = []
        obj.progress = dict(status="starting", bytes=0, last_line=None, last_diagnostics=None,
                            last_diagnostic_line=None, completed_rows=0, training_env_steps=0)
        obj.manifest = {}
        obj.source = lambda: {"test": "synthetic"}
        obj.preflight = lambda: None
        obj.validate_gate = lambda: None
        obj.read_results = lambda *args, **kwargs: None
        obj.write_progress = lambda *args, **kwargs: None
        obj.monitor = lambda event: None
        obj.verify = lambda: None
        return obj

    def test_summary_waits_for_native_nonfinite_row(self):
        result = self.root / "native/result.pkl"
        script = ("import pickle,time\n"
                  "print(%r,flush=True)\n" % SUMMARY +
                  "time.sleep(0.25)\n"
                  "with open(%r,'wb') as stream: pickle.dump([{'itr':0},{'itr':1,'kl_true_per_action':float('nan')}],stream)\n" % str(result) +
                  "time.sleep(10)\n")
        obj = self.supervisor(script)
        self.assertEqual(obj.run(), 1)
        self.assertTrue(result.exists())
        with result.open("rb") as stream:
            self.assertTrue(math.isnan(pickle.load(stream)[1]["kl_true_per_action"]))
        self.assertTrue(obj.manifest["nonfinite_failure"])
        self.assertTrue(obj.manifest["nonfinite_result_row_observed"])
        self.assertEqual(obj.manifest["failing_iteration"], 1)
        self.assertIn("nan", obj.manifest["nonfinite_summary_line"])
        json.dumps(obj.manifest, allow_nan=False)

    def test_summary_grace_is_bounded(self):
        obj = self.supervisor("import time\nprint(%r,flush=True)\ntime.sleep(10)\n" % SUMMARY)
        started = time.monotonic()
        self.assertEqual(obj.run(), 1)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 1.9)
        self.assertLess(elapsed, 6)
        self.assertFalse((obj.native / "result.pkl").exists())
        self.assertTrue(obj.manifest["nonfinite_failure"])

    def test_nonzero_exit_before_drain_keeps_diagnostic_and_original_failure(self):
        obj = self.supervisor("import sys\nprint(%r,flush=True)\nsys.exit(7)\n" % SUMMARY)
        real_popen = subprocess.Popen
        def exited_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            process.wait(timeout=5)  # Force poll() to fail before the first pipe read.
            return process
        with patch("supervise_run.subprocess.Popen", side_effect=exited_process):
            self.assertEqual(obj.run(), 1)
        self.assertEqual(obj.manifest["returncode"], 7)
        self.assertEqual(obj.manifest["failure_reason"], "DPPO exited with status 7")
        self.assertTrue(obj.manifest["nonfinite_failure"])
        self.assertEqual(obj.manifest["failing_iteration"], 1)
        self.assertIn("kl_true_per_action=nan", obj.manifest["last_diagnostic_line"])
        self.assertIn(SUMMARY, obj.console.read_text())

    def test_other_nonfinite_line_stops_without_grace(self):
        obj = self.supervisor("import time\nprint('optimizer tensor nan',flush=True)\ntime.sleep(10)\n")
        self.assertEqual(obj.run(), 1)
        self.assertIsNone(obj.pending_nonfinite)
        self.assertTrue(obj.manifest["nonfinite_failure"])
        self.assertEqual(obj.manifest["failure_reason"], "Printed NaN/Inf: optimizer tensor nan")


if __name__ == "__main__":
    unittest.main()
