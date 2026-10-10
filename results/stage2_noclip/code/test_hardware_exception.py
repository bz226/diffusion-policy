"""CPU-only mocks for the single pending-job H200 authorization."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

from hardware_exception import SOURCE, validate_hardware
from test_revised_regression import gate_fixture


class HardwareExceptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h200_fixture_", dir=str(Path(__file__).resolve().parent))
        self.root = Path(self.temp.name)
        self.original_dir = self.root / "runs/verification_full_update"
        self.original_dir.mkdir(parents=True)
        self.h200_dir = self.root / "runs/verification_h200"
        self.h200_dir.mkdir(parents=True)
        self.provenance = self.root / "provenance"
        self.provenance.mkdir()
        original = gate_fixture(commit=SOURCE, root=self.original_dir)
        (self.original_dir / "verification.json").write_text(json.dumps(original))
        h200 = gate_fixture(commit=SOURCE, root=self.h200_dir)
        h200.update(gpu_name="NVIDIA H200 NVL", source_verified_clean_before=True,
                    source_verified_clean_after=True, timeout_seconds=570, elapsed_seconds=100,
                    original_gate=str(self.original_dir / "verification.json"),
                    origin_batch=original["capture"]["saved_batch_path"],
                    origin_capture_config=str(self.original_dir / "capture.json"))
        (self.original_dir / "capture.json").write_text("{}\n")
        self.report = h200
        self.authorization = {
            "approved": True, "run_id": "NC1_seed2", "job_id": "17777249", "source_commit": SOURCE,
            "from_gpu": "NVIDIA RTX A6000", "to_gpu": "NVIDIA H200", "slurm_gres": "gpu:h200nvl:1",
            "gpus": 1, "cpus": 40, "memory_gib": 64, "runtime_cap_seconds": 10800,
            "verification_allocation_cap_seconds": 600,
            "h200_gate_path": str(self.h200_dir / "verification.json"),
            "existing_fixed_batch_gate": str(self.original_dir / "verification.json"),
            "deterministic_settings_changed": False, "other_runs_unchanged": True,
            "scientific_source_changes": False}
        self.args = types.SimpleNamespace(run_id="NC1_seed2", expected_commit=SOURCE,
            condition="NC1", seed=2, rows=140, timeout_seconds=10800,
            fixed_batch_gate=str(self.original_dir / "verification.json"))
        self.environment = {"SLURM_JOB_ID": "17777249", "SLURM_CPUS_PER_TASK": "40", "SLURM_MEM_PER_NODE": "65536"}
        self.query = "NVIDIA H200 NVL, GPU-synthetic, 0000:00:00.0, synthetic, 143771 MiB"

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, query=None):
        (self.provenance / "h200_seed2_authorization.json").write_text(json.dumps(self.authorization))
        (self.h200_dir / "verification.json").write_text(json.dumps(self.report))
        return validate_hardware(self.query if query is None else query, self.args, self.environment, self.root)

    def test_only_approved_pending_job_is_allowed(self):
        self.assertTrue(self.validate()["h200_fixed_batch_gate"]["passed"])
        self.assertEqual(self.validate()["slurm_job_id"], "17777249")

    def test_a6000_needs_no_h200_exception(self):
        for run in ("R0_rerun_seed0", "NC1_seed0", "NC1_seed2", "NC3_seed2", "NC4_lr3e-3_seed1"):
            self.args.run_id = run
            self.assertIsNone(validate_hardware("NVIDIA RTX A6000, GPU-synthetic", self.args, {}, self.root))

    def test_wrong_run_job_or_gpu_is_rejected(self):
        self.args.run_id = "NC1_seed1"
        with self.assertRaises(ValueError):
            self.validate()
        self.args.run_id = "NC1_seed2"
        self.environment["SLURM_JOB_ID"] = "17777250"
        with self.assertRaises(ValueError):
            self.validate()
        self.environment["SLURM_JOB_ID"] = "17777249"
        for query in ("NVIDIA H100, GPU-synthetic", "NVIDIA H200 SXM, GPU-synthetic", self.query + "\n" + self.query, ""):
            with self.assertRaises(ValueError):
                self.validate(query)

    def test_authorization_cannot_change_resources_or_source(self):
        original = copy.deepcopy(self.authorization)
        for key, value in (("approved", False), ("source_commit", "b" * 40), ("cpus", 80),
                           ("gpus", 2), ("memory_gib", 128), ("runtime_cap_seconds", 14400),
                           ("job_id", "17777250"), ("run_id", "NC2_seed2"), ("scientific_source_changes", True)):
            self.authorization = copy.deepcopy(original)
            self.authorization[key] = value
            with self.assertRaises(ValueError):
                self.validate()

    def test_actual_allocation_and_limits_must_match(self):
        for key, value in (("SLURM_MEM_PER_NODE", "131072"), ("SLURM_CPUS_PER_TASK", "80")):
            old = self.environment[key]
            self.environment[key] = value
            with self.assertRaises(ValueError):
                self.validate()
            self.environment[key] = old
        for key, value in (("rows", 1000), ("seed", 1), ("timeout_seconds", 14400), ("expected_commit", "b" * 40)):
            old = getattr(self.args, key)
            setattr(self.args, key, value)
            with self.assertRaises(ValueError):
                self.validate()
            setattr(self.args, key, old)

    def test_gate_cannot_be_summarized_as_passed_when_a_minibatch_failed(self):
        self.report["gpu"]["full_update"]["minibatch_comparisons"][19]["actor_gradients"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "minibatch19 actor_gradients"):
            self.validate()

    def test_all_rngs_and_saved_inputs_required(self):
        for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
            self.report["gpu"]["rng_isolation"]["states"][key] = False
            with self.assertRaisesRegex(ValueError, "RNG"):
                self.validate()
            self.report["gpu"]["rng_isolation"]["states"][key] = True
        self.report["gpu"]["full_update"]["initial_comparisons"]["rng"]["bitwise_equal"] = False
        with self.assertRaisesRegex(ValueError, "initial rng"):
            self.validate()

    def test_sampling_evidence_required(self):
        (self.h200_dir / "gpu/sampling_evidence.pt").unlink()
        with self.assertRaisesRegex(ValueError, "saved gate evidence"):
            self.validate()

    def test_no_settings_changes_or_new_batch(self):
        original = copy.deepcopy(self.report)
        for key, value in (("numeric_settings_before", {"deterministic_algorithms": True}),
                           ("source_verified_clean_after", False), ("origin_batch", "unapproved.pt"),
                           ("timeout_seconds", 601), ("elapsed_seconds", 601)):
            self.report = copy.deepcopy(original)
            self.report[key] = value
            with self.assertRaises(ValueError):
                self.validate()


if __name__ == "__main__":
    unittest.main()
