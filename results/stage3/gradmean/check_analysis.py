"""Deterministic software integration check, entirely inside a temporary tree.

No random vectors, environments, model inference, or real checkpoint loading.
Requires the existing analysis environment (NumPy and matplotlib).
"""
import csv
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
import types

import numpy as np

import analyze
from analysis_core import PointData, cross_point_cosines, discover, path_rows
from analysis_references import drift_comparisons


ROOT = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix="gradmean_analysis_check_", dir="/tmp") as temporary:
        base = Path(temporary)
        study = base / "stage3/gradmean"
        (study / "code").mkdir(parents=True)
        for name in ("analyze.py", "analysis_core.py", "analysis_references.py", "gradmean_math.py"):
            shutil.copy2(ROOT / name, study / name)
        for name in ("analysis_plots.py", "checkpoint_distances.py", "analysis_report.py"):
            shutil.copy2(ROOT / "code" / name, study / "code" / name)
        specs = [{"name": "weight", "shape": [4], "dtype": "torch.float32", "numel": 4}]
        originals = {}
        admitted = []
        for pi, point in enumerate(("P0", "D1", "D2", "N1", "N2")):
            for group in range(6):
                directory = study / "collect" / ("%s_j%02d" % (point, group))
                directory.mkdir(parents=True)
                fixed = (np.arange(200, dtype=np.float64).reshape(50, 4) % 11 - 5) / 17
                fixed += np.array([1.2, .8, .5, .4]) + group / 30 + pi / 10
                vectors = np.stack([fixed * (i + 1) + np.array([.1, -.1, .2, 0]) * i for i in range(5)], axis=1)
                sums = vectors.sum(axis=0)
                cross = np.einsum("bvp,bwp->vw", vectors, vectors)
                for v in range(5):
                    np.save(directory / ("sums_V%d.npy" % (v + 1)), sums[v].astype(np.float32))
                meta = {"point": point, "seed": 100000 + pi * 10000 + group * 100,
                        "status": "completed", "completed_batches": 50, "group_vector_kind": "sum",
                        "theory_multiplier_V4_V5": 2500, "episodes_per_batch": 80, "pairs_per_batch": 200000,
                        "frozen_sha256": point, "parameter_specs": specs, "parameter_count": 4,
                        "gradient_cross_sum": cross.tolist()}
                for kind in ("actor", "critic", "scaler"):
                    meta[kind + "_before_sha256"] = meta[kind + "_after_sha256"] = "fixed-" + point
                (directory / "meta.json").write_text(json.dumps(meta))
                manifest = study / "runs" / directory.name
                manifest.mkdir(parents=True)
                (manifest / "manifest.json").write_text(json.dumps({"status": "complete", "subprocess_exitcode": 0,
                    "remaining_process_group_members": []}))
                (directory / "cleanup.json").write_text(json.dumps({"status": "passed", "all_children_exited": True,
                    "remaining_active_children": [], "remaining_direct_children": [], "errors": [], "primary_error": None,
                    "workers": [{"exitcode": 0, "alive": False} for _ in range(40)]}))
                admitted.append({"run_id": directory.name, "terminal_recorded": True, "run_status": "complete",
                                 "scheduler_state": "COMPLETED", "exit_code": "0:0"})
                with (directory / "batches.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["n_episodes", "return_sum", "J_disc_sum",
                        "explained_variance", "x0_sat_frac_mean", "x0_sat_frac_last", "action_oor_frac",
                        "policy_action_oor_frac", "max_abs_reward"])
                    writer.writeheader()
                    for batch in range(50):
                        writer.writerow(dict(n_episodes=80, return_sum=80 * (4000 + pi * 100 + group + batch / 10),
                            J_disc_sum=80 * (1000 + pi * 20 + group + batch / 20), explained_variance=.5 + group / 100,
                            x0_sat_frac_mean=.2, x0_sat_frac_last=.3, action_oor_frac=.1,
                            policy_action_oor_frac=.05, max_abs_reward=10 + group / 10))
                if group == 0:
                    with (directory / "decision_scores.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["k", "score_norm_sq"])
                        writer.writeheader()
                        for i in range(200):
                            writer.writerow({"k": (i * 7) % 250, "score_norm_sq": 1 + i / 100})
                originals[str(directory / "meta.json")] = (directory / "meta.json").read_bytes()
        # Native science can be complete while required process cleanup fails.
        # Preserve that original, and include only its successful replacement.
        original = study / "collect/P0_j00"
        replacement = study / "collect/P0_j00_retry1"
        shutil.copytree(original, replacement)
        replacement_meta = json.loads((replacement / "meta.json").read_text())
        replacement_meta["seed"] = 1000000
        (replacement / "meta.json").write_text(json.dumps(replacement_meta))
        shutil.copytree(study / "runs/P0_j00", study / "runs/P0_j00_retry1")
        (study / "runs/P0_j00/manifest.json").write_text(json.dumps({"status": "failed", "subprocess_exitcode": 1}))
        admitted[0].update(run_status="failed", scheduler_state="FAILED", exit_code="1:0")
        admitted.append({"run_id": "P0_j00_retry1", "terminal_recorded": True, "run_status": "complete",
                         "scheduler_state": "COMPLETED", "exit_code": "0:0"})
        (study / "runs/campaign.json").write_text(json.dumps({"submitted": admitted}))
        points, inventory = discover(study)
        assert not next(row for row in inventory if row["path"] == str(original))["included"]
        assert next(row for row in inventory if row["path"] == str(replacement))["included"]
        assert points["P0"].n == 6
        for point in points.values():
            for remove in (None, 0, 3):
                ids = point.ids if remove is None else np.delete(point.ids, remove)
                total, q = point.sums[:, ids, :].sum(axis=1), point.q[ids].sum(axis=0)
                k = len(ids) * 50
                expected = (total @ total.T - q) / (k * (k - 1))
                assert np.allclose(point.moments(ids)["cross_signal"], expected, rtol=1e-12, atol=1e-12)
        print("Full/delete-group cross moments match direct scalar reconstruction", flush=True)
        sparse = {point: PointData(point, [points[point].paths[0]]) for point in ("P0", "D1")}
        sparse_cosines = cross_point_cosines(sparse)
        assert len(sparse_cosines) == 50
        assert all(row["cosine_low"] is None for row in sparse_cosines)
        sparse_paths = path_rows(sparse, {"P0": np.zeros(4), "D1": np.ones(4)})
        assert len(sparse_paths) == 6 and sparse_paths[0]["status"] == "estimated"
        assert sparse_paths[0]["measured_delta_J_low"] is None
        assert all(row["status"] == "point_not_collected" for row in sparse_paths[1:])
        assert all(row["B_ep_type"] == "unavailable" for row in sparse["P0"].public_rows(0.))
        print("Single-group and missing-policy partial outputs retain unavailable uncertainty", flush=True)
        for condition in ("SGD1", "SGD3", "SGD10", "BATCH4"):
            for seed in range(3):
                run = base / "stage3" / ("%s_seed%d" % (condition, seed))
                manifest = base / "stage3/runs" / run.name
                run.mkdir()
                manifest.mkdir(parents=True)
                rows, updates = [], 0
                for itr in range(140):
                    if itr % 10 == 0:
                        rows.append({"itr": itr, "eval_episode_reward": 4000.})
                    else:
                        rows.append({"itr": itr, "train_episode_reward": 4000., "J_disc_train": 100 + 3 * updates + seed,
                                     "actor_lr": .0001, "actor_optimizer_steps": 1})
                        updates += 1
                with (run / "result.pkl").open("wb") as stream:
                    pickle.dump(rows, stream)
                (manifest / "manifest.json").write_text(json.dumps({"status": "complete", "logdir": str(run)}))
        seeds, _ = drift_comparisons(base / "stage3", points["P0"])
        assert all(row["J_disc_slope_per_update"] == 3. for row in seeds)
        assert all(abs(row["squared_gradient_drift_proxy"] - 3 / (.0001 / 2500)) < 1e-6 for row in seeds)
        print("All 126-observation drift fits and theory conversion match the exact linear fixture", flush=True)
        fake = types.ModuleType("checkpoint_distances")
        fixed_vectors = {p: np.array([i, i / 2, i / 3, i / 4], dtype=float) for i, p in enumerate(points)}
        fake.load_checkpoint_vectors = lambda **kwargs: (fixed_vectors, {
            "parameter_specs": specs, "parameter_count": 4, "scope": "deterministic software fixture"})
        fake.distance_summary = lambda vectors: {"distance_from_P0": {
            point: float(np.linalg.norm(value - vectors["P0"])) for point, value in vectors.items()}}
        sys.modules["checkpoint_distances"] = fake
        analyze.ROOT, analyze.STAGE3 = study, base / "stage3"
        sys.path.insert(0, str(ROOT / "code"))
        output = study / "runs/fixture_analysis"
        sys.argv = ["analyze.py", "--output", str(output), "--partial"]
        analyze.main()
        status = json.loads((output / "analysis_status.json").read_text())
        assert status["status"] == "complete" and all(Path(path).is_file() for path in status["outputs"])
        result = json.loads((output / "candidates/results.json").read_text())
        assert len(result["main"]) == 25
        assert len(result["tables"]["same_point_cosines"]) == 125
        assert len(result["tables"]["cross_point_cosines"]) == 50
        assert len(result["tables"]["path_alignment"]) == 6
        assert all(row["u_sq"] is None and row["trace_ep"] is None and row["effective_rank"] is None
                   for row in result["main"] if row["variant"] in ("V1", "V2", "V3"))
        summary = output / "candidates/SUMMARY.md"
        for target in re.findall(r"\]\(([^)]+)\)", summary.read_text()):
            assert (summary.parent / target).resolve().exists(), target
        report = output / "candidates/REPORT.md"
        report_text = report.read_text()
        assert all("## Q%d." % i in report_text for i in range(1, 11))
        assert "fixed-policy gradient" in report_text
        for target in re.findall(r"\]\(([^)]+)\)", report_text):
            assert (report.parent / target.strip("<>")).resolve().exists(), target
        assert not (study / "REPORT.md").exists(), "Interim analysis overwrote the current report"
        from analysis_report import build_report, metadata_inventory, write_reports
        metadata = metadata_inventory(study, json.loads((study / "runs/campaign.json").read_text()))
        # Format missing policies, unresolved bounds and all-zero metadata
        # without producing measurements or rerunning any estimator.
        incomplete = dict(result, main=[dict(result["main"][3], B_ep_type="lower_bound", B_ep=None,
            B_ep_lower_bound=10000., expected_batch_cos=None, expected_batch_cos_upper_bound=.089,
            K_needed_type="lower_bound", K_needed=532.5)])
        rendered = build_report(report, study, output / "candidates", incomplete, [], {}, {}, status, metadata)
        assert "≥ 1e+04 (plug-in)" in rendered and "≤ 0.089 (plug-in)" in rendered
        # Formatting-only final mirror exercise reuses the saved fixture tables.
        final = study / "runs/final"
        (final / "candidates").mkdir(parents=True)
        (final / "internal").mkdir()
        old_body = "# Previous current report\n"
        (study / "REPORT.md").write_text(old_body)
        final_paths = write_reports(study, final / "candidates", result, status["figures"], {}, {}, status, final=True)
        assert Path(final_paths["current_report"]).is_file()
        assert (final / "internal/report_before_final_analysis.md").read_text() == old_body
        for report in (final / "candidates/REPORT.md", study / "REPORT.md"):
            for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
                assert (report.parent / target.strip("<>")).resolve().exists(), target
        assert all(Path(path).read_bytes() == content for path, content in originals.items())
        print(json.dumps({"deterministic_integration": "passed", "figures": len(status["figures"]),
                          "files": len(status["outputs"]), "summary_words": len(summary.read_text().split()),
                          "raw_records_unchanged": True, "new_scientific_samples": 0}), flush=True)


if __name__ == "__main__":
    main()
