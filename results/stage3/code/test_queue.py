"""Scheduler-free checks for admission safety, numerical failure handling and waves."""
import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import campaign
import monitor


class QueueSafetyTests(unittest.TestCase):
    def test_approved_maximum_is_bounded_including_every_evaluation(self):
        train = campaign.wave1(0.0123) + campaign.wave2(0.0123)
        modes = campaign.wave0() + [campaign.evaluation(r) for r in train] + [campaign.evaluation()]
        self.assertEqual(len(train), 36)
        self.assertEqual(len({r["run_id"] for r in train + modes}), 76)
        self.assertEqual(sum(r["cap_seconds"] for r in train + modes) + campaign.GATE_RESERVED_SECONDS,
                         int(197.5 * 3600))
        self.assertLess(sum(r["cap_seconds"] for r in train + modes) + campaign.GATE_RESERVED_SECONDS,
                        campaign.GPU_BUDGET_SECONDS)

    def test_other_jobs_reduce_capacity(self):
        state = {"submitted": [{"job_id": str(i), "scheduler_state": "RUNNING"} for i in range(3)]}
        jobs = [{"job_id": str(i), "gpus": 1} for i in range(3)]
        jobs += [{"job_id": "controller", "gpus": 0}, {"job_id": "interactive", "gpus": 1}]
        self.assertEqual(campaign.remaining_slots(state, jobs, 9), 6)
        jobs += [{"job_id": "other" + str(i), "gpus": 1} for i in range(4)]
        self.assertEqual(campaign.remaining_slots(state, jobs, 9), 3)
        jobs += [{"job_id": "cpu" + str(i), "gpus": 0} for i in range(3)]
        self.assertEqual(campaign.remaining_slots(state, jobs, 9), 0)

    def test_null_diagnostic_is_allowed_but_nan_and_inf_are_failures(self):
        self.assertIsNone(monitor.first_nonfinite({"grad_cos_prev": None, "split_dot": -3.0}))
        self.assertEqual(monitor.first_nonfinite({"kl_true_per_action": math.inf}), ".kl_true_per_action")
        self.assertEqual(monitor.first_nonfinite({"grad_norm": math.nan}), ".grad_norm")
        self.assertEqual(monitor.first_nonfinite(math.inf), "<root>")

    def test_effective_flags_must_match_instead_of_silently_ignoring(self):
        run = campaign.wave1(0.0123)[9]
        settings = monitor.expected_settings(run)
        monitor.validate_settings(run, settings)
        settings["actor_optimizer"] = "adamw"
        with self.assertRaises(ValueError):
            monitor.validate_settings(run, settings)
        for run in campaign.wave1(0.0123) + campaign.wave2(0.0123) + campaign.wave0() + [campaign.evaluation()]:
            actual = monitor.expected_settings(run)
            self.assertEqual(actual["mode"], run["mode"])
            if run["condition"].startswith("SGD") or run["condition"].startswith("PROJ") or run["condition"] == "BATCH4":
                self.assertEqual(actual["optimizer_class"], "SGD")
                self.assertEqual(actual["momentum"], 0)

    def test_projection_skip_requires_all_complete_and_strict_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for seed in (0, 1, 2):
                run = {"run_id": "SGD3_seed" + str(seed), "logdir": str(root / ("native" + str(seed)))}
                runs.append(run)
                path = root / "runs" / run["run_id"]
                path.mkdir(parents=True)
                (path / "manifest.json").write_text(json.dumps({"status": "complete"}))
                logdir = Path(run["logdir"])
                logdir.mkdir()
                with (logdir / "result.pkl").open("wb") as stream:
                    pickle.dump([{"train_episode_reward": 1.0, "theta_dist_preproj": 0.1}] * 126, stream)
            with mock.patch.object(campaign, "ROOT", root):
                self.assertTrue(campaign.projection_decision("SGD3", runs)["skip"])
                with (Path(runs[1]["logdir"]) / "result.pkl").open("wb") as stream:
                    pickle.dump([{"train_episode_reward": 1.0, "theta_dist_preproj": 0.8 * campaign.RADIUS}] * 126, stream)
                self.assertFalse(campaign.projection_decision("SGD3", runs)["skip"])
                (root / "runs/SGD3_seed1/manifest.json").write_text(json.dumps({"status": "failed"}))
                self.assertFalse(campaign.projection_decision("SGD3", runs)["skip"])

    def test_wave_specs_have_exact_seeds_and_no_duplicate_keys(self):
        runs = campaign.wave1(0.0123) + campaign.wave2(0.0123)
        for condition in campaign.TRAIN_CONDITIONS + campaign.WAVE2_CONDITIONS:
            self.assertEqual([r["seed"] for r in runs if r["condition"] == condition], [0, 1, 2])
        for run in runs + campaign.wave0():
            keys = [x.split("=", 1)[0].lstrip("+") for x in run["overrides"]]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(run["monitor_interval_seconds"], max(1800, run["estimate_seconds"] // 6))
            self.assertNotIn("<", " ".join(run["overrides"]))
        e2none = next(r for r in runs if r["condition"] == "E2_nobase")
        self.assertIn("+train.adv_estimator=mc_none", e2none["overrides"])

    def test_log_observation_ignores_null_and_finds_reward_collapse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.out"
            log.write_text('STAGE3_SETTINGS {"actor_loss":"score"}\n'
                           '1: step 80000 | grad_cos_prev=null | kl_true_per_action=0.001\n')
            with (root / "result.pkl").open("wb") as stream:
                pickle.dump([{"itr": 0, "step": 0, "eval_episode_reward": 4000.},
                             {"itr": 10, "step": 720000, "eval_episode_reward": 1000., "grad_cos_prev": None}], stream)
            run = {"run_id": "test", "mode": "train", "logdir": str(root), "console_log": str(log)}
            out = monitor.observe(run)
            self.assertNotIn("nonfinite", out)
            self.assertNotIn("nonfinite_log_line", out)
            self.assertEqual(out["reward_collapse"]["first_itr"], 10)
            self.assertEqual(out["effective_settings"], {"actor_loss": "score"})
            with log.open("a") as stream:
                stream.write('11: step 800000 | kl_true_per_action=inf\n')
            self.assertIn("nonfinite_log_line", monitor.observe(run))

    def test_preexisting_jobfile_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.txt"
            campaign.write_once(path, "original\n")
            campaign.write_once(path, "original\n")
            with self.assertRaises(ValueError):
                campaign.write_once(path, "changed\n")
            self.assertEqual(path.read_text(), "original\n")

    def test_failure_record_preserves_iteration_beyond_saved_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.out"
            log.write_text('3: step 240000 | kl_true_per_action=0.001\n'
                           'STAGE3_METRICS {"grad_norm":4.0,"grad_cos_prev":null}\n'
                           'Traceback (most recent call last):\n'
                           '  File "train.py", line 1\n'
                           'FloatingPointError: nonfinite raw x0 prediction\n')
            with (root / "result.pkl").open("wb") as stream:
                pickle.dump([{"itr": 3, "step": 240000, "train_episode_reward": 4000.}], stream)
            run = {"run_id": "test", "mode": "train", "logdir": str(root), "console_log": str(log)}
            out = monitor.observe(run)
            self.assertEqual(out["failure_iteration_candidate"], 4)
            self.assertNotIn("failure_iteration", out)
            self.assertEqual(out["last_stage3_metrics"]["grad_norm"], 4.0)
            self.assertIn("nonfinite_exception", out)
            (root / "failure.json").write_text(json.dumps({"itr": 4, "reason": "nonfinite", "values": {"grad_norm": "inf"}}))
            out = monitor.observe(run)
            self.assertEqual(out["failure_iteration"], 4)
            self.assertNotIn("failure_iteration_candidate", out)
            self.assertEqual(out["last_itr"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
