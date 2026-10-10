"""Focused synthetic regression tests; no training or source imports."""

import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest

from replay_compare import ReplayInputError, compare_replay, _gross_deviations


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="replay_fixture_", dir=str(Path(__file__).resolve().parent))
        self.directory = Path(self.temp.name)
        self.runs = {}
        for role in ("P1", "P2", "R0"):
            self.write_run(role)

    def tearDown(self):
        self.temp.cleanup()

    def write_run(self, role, count=20):
        rows = []
        lines = ["irrelevant package startup warning"]
        if role == "R0":
            lines.append("[2026-10-09 01:02:03,001][agent][INFO] - stage2_flags clamp_logprob=True logprob_reduce=mean actor_single_step=False")
        for iteration in range(count):
            row = {"itr": iteration, "step": (iteration - iteration // 10) * 80000, "time": 50.0 if role == "P1" else 90.0}
            lines.extend("Processed step %d of 500" % step for step in range(0, 500, 10))
            prefix = "[2026-10-09 01:02:%02d,001][agent.finetune.train_ppo_diffusion_agent][INFO] - " % iteration
            if iteration % 10:
                row["train_episode_reward"] = 3800.0 + iteration
                for epoch in range(5):
                    for batch in range(4):
                        lines.append(prefix + "approx_kl: %s, update_epoch: %d, num_batch: 4" % (repr((epoch * 4 + batch) * 1e-6), epoch))
                summary = "%d: step %8d | loss   0.0054 | pg loss  -0.0008 | value loss   0.0123 | bc loss   0.0000 | reward %8.4f | eta   1.0000 | t:%8.4f" % (iteration, row["step"], row["train_episode_reward"], row["time"])
                if role == "R0":
                    row.update(kl_true_per_action=0.01, logratio_p99=0.02, clamp_hit_frac=0.03, approx_kl=19e-6, clipfrac=0.1, actor_optimizer_steps=20, critic_optimizer_steps=20, actor_lr=1e-4, critic_lr=1e-3)
                    summary += " | kl_true_per_action=0.01 | logratio_p99=0.02 | clamp_hit_frac=0.03"
            else:
                row.update(eval_success_rate=1.0, eval_episode_reward=4200.0 + iteration, eval_best_reward=6.0)
                summary = "eval: success rate   1.0000 | avg episode reward %8.4f | avg best reward   6.0000" % row["eval_episode_reward"]
            if iteration == 0 or (iteration == 19 and role != "R0"):
                lines.append(prefix + "Saved model to /different/%s/native/checkpoint/state_%d.pt" % (role, iteration))
            lines.append(prefix + summary)
            rows.append(row)
        result, log = self.directory / (role + ".pkl"), self.directory / (role + ".out")
        self.runs[role] = [result, log]
        self.set_rows(role, rows)
        log.write_text("\n".join(lines) + "\n")

    def rows(self, role):
        with self.runs[role][0].open("rb") as stream:
            return pickle.load(stream)

    def set_rows(self, role, rows):
        with self.runs[role][0].open("wb") as stream:
            pickle.dump(rows, stream)

    def change_log(self, role, old, new, count=1):
        path = self.runs[role][1]
        original = path.read_text()
        self.assertIn(old, original)
        path.write_text(original.replace(old, new, count))

    def compare(self, r0=True):
        arguments = self.runs["P1"] + self.runs["P2"]
        if r0:
            arguments += self.runs["R0"]
        return compare_replay(*arguments)

    def test_exact_replay_ignores_only_declared_metadata(self):
        report = self.compare()
        self.assertTrue(report["deterministic_probe"])
        self.assertTrue(report["r0_passed"])
        self.assertEqual(report["r0_comparison"]["mismatch_count"], 0)
        self.assertGreater(report["probe_comparison"]["value_count"], 4000)
        self.assertEqual(report["probe_comparison"]["value_count"], report["r0_comparison"]["enforced_value_count"])
        self.assertEqual(sum(e["kind"] == "probe_terminal_checkpoint" for e in report["exceptions"]["P1"]), 1)
        json.dumps(report, allow_nan=False)

    def test_probes_only(self):
        report = self.compare(r0=False)
        self.assertTrue(report["deterministic_probe"])
        self.assertIsNone(report["r0_passed"])

    def test_full_precision_pickle_difference_is_not_rounded_away(self):
        rows = self.rows("R0")
        rows[7]["train_episode_reward"] += 1e-10
        self.set_rows("R0", rows)
        report = self.compare()
        self.assertFalse(report["r0_passed"])
        mismatch = report["r0_comparison"]["first_difference"]
        self.assertEqual((mismatch["iteration"], mismatch["field"]), (7, "train_episode_reward"))

    def test_nondeterminism_does_not_disable_later_stable_fields(self):
        rows = self.rows("P2")
        rows[2]["train_episode_reward"] += 5
        self.set_rows("P2", rows)
        rows = self.rows("R0")
        rows[2]["train_episode_reward"] += 1000
        rows[16]["train_episode_reward"] += 1
        self.set_rows("R0", rows)
        report = self.compare()
        self.assertFalse(report["deterministic_probe"])
        self.assertEqual(report["probe_comparison"]["first_difference"]["iteration"], 2)
        self.assertEqual(report["r0_comparison"]["first_difference"]["iteration"], 16)
        self.assertEqual(report["r0_comparison"]["unconstrained_value_count"], 1)

    def test_log_mask_is_per_token_not_per_line(self):
        self.change_log("P2", "1: step    80000 | loss   0.0054", "1: step    80000 | loss   0.0055")
        self.change_log("R0", "1: step    80000 | loss   0.0054 | pg loss  -0.0008", "1: step    80000 | loss   0.0100 | pg loss  -0.0009")
        report = self.compare()
        self.assertFalse(report["deterministic_probe"])
        self.assertFalse(report["r0_passed"])
        self.assertEqual(report["r0_comparison"]["unconstrained_value_count"], 1)
        self.assertEqual(report["r0_comparison"]["mismatch_count"], 1)

    def test_missing_legacy_key_fails(self):
        rows = self.rows("R0")
        del rows[4]["train_episode_reward"]
        self.set_rows("R0", rows)
        with self.assertRaises(ReplayInputError) as caught:
            self.compare()
        self.assertEqual(caught.exception.first_difference["iteration"], 4)

    def test_unknown_additive_key_fails(self):
        rows = self.rows("R0")
        rows[3]["unapproved"] = 1
        self.set_rows("R0", rows)
        with self.assertRaises(ReplayInputError):
            self.compare()

    def test_nonfinite_new_metric_fails(self):
        rows = self.rows("R0")
        rows[8]["kl_true_per_action"] = math.inf
        self.set_rows("R0", rows)
        with self.assertRaises(ReplayInputError) as caught:
            self.compare()
        self.assertEqual(caught.exception.first_difference["iteration"], 8)

    def test_unknown_log_line_fails_without_realignment(self):
        self.change_log("R0", "Processed step 20 of 500", "unexpected extra event\nProcessed step 20 of 500")
        with self.assertRaises(ReplayInputError) as caught:
            self.compare()
        self.assertEqual(caught.exception.first_difference["iteration"], 0)

    def test_terminal_checkpoint_exception_cannot_hide_wrong_iteration(self):
        self.change_log("P2", "state_19.pt", "state_18.pt")
        with self.assertRaises(ReplayInputError):
            self.compare()

    def test_partial_r0_waits_but_detects_early_failure(self):
        self.write_run("R0", count=2)
        report = self.compare()
        self.assertIsNone(report["r0_passed"])
        self.assertEqual(report["comparison_status"], "waiting_r0")
        rows = self.rows("R0")
        rows[1]["train_episode_reward"] += 0.1
        self.set_rows("R0", rows)
        self.assertFalse(self.compare()["r0_passed"])

    def test_probe_requires_all_twenty_iterations(self):
        self.write_run("P2", count=19)
        with self.assertRaises(ReplayInputError):
            self.compare()

    def test_required_diagnostics_must_be_logged(self):
        self.change_log("R0", " | logratio_p99=0.02", "")
        with self.assertRaises(ReplayInputError):
            self.compare()

    def test_startup_flags_must_be_defaults(self):
        self.change_log("R0", "logprob_reduce=mean", "logprob_reduce=sum")
        with self.assertRaises(ReplayInputError):
            self.compare()

    def test_sample_std_interval_and_strict_boundary(self):
        paths = []
        for seed, value in enumerate([0.0, 1.0, 2.0]):
            path = self.directory / ("baseline%d.pkl" % seed)
            path.write_bytes(pickle.dumps([{"itr": 0, "step": 0, "eval_episode_reward": value}]))
            paths.append(path)
        checks = _gross_deviations([{"itr": 0, "step": 0, "eval_episode_reward": 6.0}], paths)
        self.assertEqual(checks[0]["sample_std_ddof1"], 1.0)
        self.assertFalse(checks[0]["red_flag"])
        checks = _gross_deviations([{"itr": 0, "step": 0, "eval_episode_reward": 6.0 + 1e-12}], paths)
        self.assertTrue(checks[0]["red_flag"])

    def test_unrelated_startup_warning_is_not_a_flag_printout(self):
        for role in self.runs:
            self.change_log(role, "irrelevant package startup warning", "warning example text: clamp_logprob=True actor_single_step=False")
        self.assertTrue(self.compare()["r0_passed"])

    def test_matching_warning_inside_iteration_is_compared(self):
        for role in self.runs:
            self.change_log(role, "Processed step 20 of 500", "[2026-10-09 01:01:01,001][worker][WARNING] - diagnostic warning 123\nProcessed step 20 of 500")
        self.assertTrue(self.compare()["r0_passed"])
        self.change_log("R0", "diagnostic warning 123", "diagnostic warning 124")
        self.assertFalse(self.compare()["r0_passed"])

    def test_checks_gross_deviation_after_replay_interval(self):
        self.write_run("R0", count=21)
        rows = self.rows("P2")
        rows[2]["train_episode_reward"] += 1
        self.set_rows("P2", rows)
        paths = []
        for seed in range(3):
            rows = self.rows("R0")
            for row in rows:
                if "eval_episode_reward" in row:
                    row["eval_episode_reward"] += seed - 1
            path = self.directory / ("long_baseline%d.pkl" % seed)
            path.write_bytes(pickle.dumps(rows))
            paths.append(path)
        rows = self.rows("R0")
        rows[20]["eval_episode_reward"] = 9000.0
        self.set_rows("R0", rows)
        report = compare_replay(*(self.runs["P1"] + self.runs["P2"] + self.runs["R0"]), baseline_results=paths)
        self.assertTrue(report["r0_passed"])
        self.assertEqual([item["iteration"] for item in report["gross_eval_red_flags"]], [20])

    def test_actual_stage1_twenty_iteration_console_and_schema(self):
        baseline = Path(__file__).resolve().parents[2] / "repro_halfcheetah"
        manifest_path = baseline / "runs/seed0_handoff/manifest.json"
        console_path = baseline / "runs/seed0/console.log"
        if not manifest_path.exists() or not console_path.exists():
            self.skipTest("Retained Stage 1 evidence is unavailable")
        manifest = json.loads(manifest_path.read_text())
        with open(manifest["result_path"], "rb") as stream:
            rows = pickle.load(stream)[:20]
        lines = []
        for line in console_path.read_text().splitlines():
            if "][INFO] - 19: step " in line:
                prefix = line.split("19: step ")[0]
                lines.append(prefix + "Saved model to /synthetic/probe/checkpoint/state_19.pt")
                lines.append(line)
                break
            lines.append(line)
        self.assertIn("][INFO] - 19: step ", lines[-1])
        for role in ("P1", "P2"):
            self.set_rows(role, rows)
            self.runs[role][1].write_text("\n".join(lines) + "\n")
        report = self.compare(r0=False)
        self.assertTrue(report["deterministic_probe"])
        self.assertEqual(report["probe_comparison"]["iterations"], 20)


if __name__ == "__main__":
    unittest.main()
