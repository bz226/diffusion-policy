"""Mechanical queue/monitor checks; no Slurm calls, model execution, or training."""
import json
import io
import math
from pathlib import Path
import pickle
import tempfile
import tarfile
import unittest
from unittest.mock import patch

import campaign as c
import monitor as m


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="frs-runtime-")
        self.root = Path(self.temp.name)
        self.scale = self.root / "rescale.pt"
        self.scale.write_bytes(b"mock-artifact")
        self.gate = self.root / "gate.json"
        original = {"passed": True, "parent_commit": "precommit",
                    "rescale_path": str(self.scale), "source_snapshot": str(self.root / "snapshot")}
        self.original_gate = self.root / "original_gate.json"
        self.original_gate.write_text(json.dumps(original))
        self.gate.write_text(json.dumps(dict(original, parent_commit="fixture-commit",
                    original_gate_path=str(self.original_gate), reference_parent_commit="precommit",
                    committed_source_byte_identical=True)))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_six_job_scope_and_caps(self):
        specs = c.run_specs(self.gate)
        self.assertEqual(len(specs), 6)
        self.assertEqual(c.GATE_CAP + sum(s["cap_seconds"] for s in specs), 13.75 * 3600)
        self.assertLessEqual(c.GATE_CAP + sum(s["cap_seconds"] for s in specs), c.BUDGET)
        for index, run in enumerate(specs):
            self.assertTrue(c.validate_spec(run, self.gate))
            self.assertEqual(run["cpus"], 40)
            self.assertEqual(run["memory_gib"], 64)
            self.assertEqual(run["gpu_type"], "a6000")
            self.assertEqual(run["monitor_interval_seconds"], 1800)
            if index < 3:
                self.assertEqual(run["seed"], index)
                self.assertEqual(run["cap_seconds"], 14400)
                expected = m.expected_settings(run)
                self.assertEqual(expected["actor_optimizer"], "sgd_diag")
                self.assertEqual(expected["optimizer_class"], "FrozenRescaledSGD")
                self.assertEqual(expected["adv_estimator"], "mc_loo")
                self.assertEqual(expected["rescale_path"], str(self.scale))
                self.assertEqual(expected["rescale_floor"], .001)
                self.assertEqual(expected["lr"], 1e-4)
                self.assertEqual(expected["scheduler_min_lr"], 1e-4)
                self.assertIsNone(expected["betas"])
                self.assertEqual(expected["momentum"], 0)
            else:
                self.assertEqual(run["mode"], "ckpt_eval")
                self.assertEqual(run["seed"], 5000)
                self.assertEqual(run["cap_seconds"], 900)
                self.assertEqual(m.expected_settings(run)["ckpt_eval_repeats"], 1)

    def test_spec_mutation_is_rejected(self):
        original = c.run_specs(self.gate)[0]
        for field, wrong in [("cpus", 39), ("seed", 3), ("cap_seconds", 18000), ("gpu_type", "h200nvl"),
                             ("condition", "FRS3"), ("estimate_seconds", 99999)]:
            run = dict(original, **{field: wrong})
            with self.assertRaises(ValueError):
                c.validate_spec(run, self.gate)
        run = dict(original, overrides=original["overrides"] + ["env.n_envs=20"])
        with self.assertRaises(ValueError):
            c.validate_spec(run, self.gate)

    def test_startup_settings_misspelling_rejected(self):
        run = c.run_specs(self.gate)[0]
        expected = m.expected_settings(run)
        m.validate_settings(run, expected)
        for key in ["rescale_path", "rescale_floor", "actor_optimizer", "optimizer_class"]:
            wrong = dict(expected)
            wrong.pop(key)
            with self.assertRaises(ValueError):
                m.validate_settings(run, wrong)

    def test_source_no_nested_git(self):
        source = self.root / "source"
        source.mkdir()
        (source / "a.py").write_text("x=1\n")
        (source / "__pycache__").mkdir()
        (source / "__pycache__/a.pyc").write_bytes(b"ignore")
        self.assertEqual(list(m.source_files(source)), ["a.py"])
        (source / ".git").write_text("gitdir: somewhere")
        with self.assertRaises(ValueError):
            m.source_files(source)

    def test_source_parent_branch_revision_bytes_and_mode(self):
        source, snapshot = self.root / "source", self.root / "snapshot"
        source.mkdir(); snapshot.mkdir()
        (source / "a.py").write_text("a=1\n")
        (snapshot / "a.py").write_text("a=1\n")
        def mocked(argv, *args, **kwargs):
            self.assertEqual(argv[:3], ["git", "-C", str(self.root)])
            return {"stdout": "stage3-theory\n" if "branch" in argv else "fixture-commit\n"}
        with patch.object(m, "ROOT", self.root), patch.object(m, "SOURCE", source), patch.object(m, "PARENT", self.root), patch.object(m, "command", mocked):
            self.assertTrue(m.validate_source("fixture-commit", gate=self.gate)["byte_identical"])
            with self.assertRaises(ValueError):
                m.validate_source("wrong-revision", gate=self.gate)
            (source / "a.py").write_text("a=2\n")
            with self.assertRaises(ValueError):
                m.validate_source("fixture-commit", gate=self.gate)
            (source / "a.py").write_text("a=1\n")
            (source / "a.py").chmod(0o755)
            with self.assertRaises(ValueError):
                m.validate_source("fixture-commit", gate=self.gate)
        with patch.object(m, "command", lambda *a, **kw: {"stdout": "main\n"}):
            with self.assertRaises(ValueError):
                m.validate_source("fixture-commit", gate=self.gate)

    def test_gate_bound_and_local(self):
        with patch.object(m, "ROOT", self.root), patch.object(m, "validate_rescale", lambda *a: {"bitwise_payload_equal": True}):
            self.assertTrue(m.validate_gate(self.gate, "fixture-commit")["passed"])
            with self.assertRaises(ValueError):
                m.validate_gate(self.gate, "wrong")
            with self.assertRaises(ValueError):
                m.validate_gate(self.original_gate, "precommit")
            self.scale.unlink()
            with self.assertRaises(ValueError):
                m.validate_gate(self.gate, "fixture-commit")

    def test_nonfinite_diagnostics_null_vs_actual_nonfinite(self):
        self.assertIsNone(m.first_nonfinite({"grad_cos_prev": None, "max_ratio": 0.0}))
        self.assertEqual(m.first_nonfinite({"max_ratio": float("inf")}), ".max_ratio")
        self.assertEqual(m.first_nonfinite({"median_ratio": float("nan")}), ".median_ratio")

    def test_observe_failure_keeps_iteration_and_last_diagnostics(self):
        run = c.run_specs(self.gate)[0]
        run.update(logdir=str(self.root), console_log=str(self.root / "out"))
        rows = [{"itr": 0, "step": 0, "eval_episode_reward": 4200.},
                {"itr": 1, "step": 80000, "train_episode_reward": 4100., "max_ratio": float("inf")}]
        with (self.root / "result.pkl").open("wb") as stream:
            pickle.dump(rows, stream)
        observation = m.observe(run)
        self.assertEqual(observation["failure_iteration"], 1)
        self.assertEqual(observation["nonfinite"]["row"]["max_ratio"], "inf")
        self.assertEqual(observation["last"]["train_episode_reward"], 4100.)

    def test_training_completion_requires_rescale_diagnostics(self):
        run = c.run_specs(self.gate)[0]
        run["logdir"] = str(self.root)
        rows = [{"itr": i, "step": (i - i // 10) * 80000,
                 "eval_episode_reward" if i % 10 == 0 else "train_episode_reward": 4000.} for i in range(140)]
        with (self.root / "result.pkl").open("wb") as stream:
            pickle.dump(rows, stream)
        with self.assertRaisesRegex(ValueError, "ratio diagnostics"):
            m.verify_complete(run, {})

    def test_accounting_counts_generic_gpu_once(self):
        response = {"stdout": "123|COMPLETED|100|cpu=40,gres/gpu:a6000=1,gres/gpu=1,mem=64G|\n"}
        with patch.object(c, "command", lambda *a, **kw: response):
            result = c.accounting("123")
        self.assertEqual(result["gpu_count"], 1)
        self.assertEqual(result["allocated_gpu_seconds"], 100)

    def test_admission_respects_account_and_owned_limits(self):
        self.assertEqual(c.available_slots([], []), 3)
        jobs = [{"job_id": str(i), "gpus": 1} for i in range(11)]
        self.assertEqual(c.available_slots([], jobs), 1)
        self.assertEqual(c.available_slots([], jobs + [{"job_id": "12", "gpus": 0}]), 0)
        own = [{"job_id": "owned" + str(i), "scheduler_state": "PENDING"} for i in range(3)]
        self.assertEqual(c.available_slots(own, []), 0)

    def test_write_once_never_replaces_different_evidence(self):
        path = self.root / "record.json"
        c.write_once(path, {"a": 1})
        c.write_once(path, {"a": 1})
        with self.assertRaises(ValueError):
            c.write_once(path, {"a": 2})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_bind_committed_bytes_preserves_original_gate(self):
        source = self.root / "dppo"
        snapshot = self.root / "snapshot"
        source.mkdir(); snapshot.mkdir()
        for base in (source, snapshot):
            (base / "a.py").write_text("a=1\n")
        before = self.original_gate.read_bytes()
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:") as archive:
            entry = tarfile.TarInfo("dppo/a.py")
            entry.size = 4
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(b"a=1\n"))
        class Result:
            returncode = 0
            stderr = b""
            stdout = archive_bytes.getvalue()
        def mocked(argv, *a, **kw):
            return {"stdout": "stage3-theory\n" if "branch" in argv else "fixture-commit\n"}
        with patch.object(c, "ROOT", self.root), patch.object(c, "SOURCE", source), patch.object(c, "PARENT", self.root), \
                patch.object(c, "JOURNAL", self.root / "commands.jsonl"), patch.object(c, "command", mocked), \
                patch.object(c.subprocess, "run", lambda *a, **kw: Result()):
            result = c.bind_source(self.original_gate, "fixture-commit")
            self.assertTrue(result["committed_source_byte_identical"])
            self.assertEqual(result["reference_parent_commit"], "precommit")
            self.assertEqual(result["parent_commit"], "fixture-commit")
            self.assertEqual(self.original_gate.read_bytes(), before)
            (source / "a.py").write_text("a=2\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                c.bind_source(self.original_gate, "fixture-commit")

    def test_rescale_payload_check_detects_coordinate_changes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Run with the existing Stage1 Python environment for torch artifact checks")
        repeat = self.root / "repeat.pt"
        payload = {"r": {"weight": torch.tensor([1., 2.], dtype=torch.float64)},
                   "r_tilde": {"weight": torch.tensor([1., 2.], dtype=torch.float64)},
                   "floor": .001, "processing_seed": 4000}
        torch.save(payload, str(self.scale))
        torch.save(payload, str(repeat))
        record = {"rescale_path": str(self.scale), "rescale_repeat_reference_path": str(repeat)}
        with patch.object(m, "ROOT", self.root):
            self.assertTrue(m.validate_rescale(record)["bitwise_payload_equal"])
            payload["r"]["weight"][0] = 3.
            torch.save(payload, str(repeat))
            with self.assertRaises(ValueError):
                m.validate_rescale(record)


if __name__ == "__main__":
    unittest.main()
