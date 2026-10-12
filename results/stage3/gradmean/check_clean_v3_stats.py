"""Mechanical checks of clean-v3 statistics; synthetic arrays only, no rollout."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

import clean_v3_stats as stats


def fixture(directory, index, vectors, point="P0"):
    directory.mkdir(parents=True)
    meta = {"point": point, "status": "completed", "completed_batches": 50,
            "group_vector_kind": "sum", "theory_multiplier_V4_V5": 2500,
            "theory_multiplier_V6": 2500, "episodes_per_batch": 80,
            "pairs_per_batch": 200000, "seed": index, "frozen_sha256": "fixture",
            "parameter_specs": [{"name": "fixture", "shape": [vectors.shape[-1]]}],
            "parameter_count": vectors.shape[-1], "variant_ids": list(stats.VARIANTS),
            "gradient_cross_sum": np.einsum("bvp,bwp->vw", vectors, vectors).tolist()}
    for name in ("actor", "critic", "scaler"):
        meta[name + "_before_sha256"] = meta[name + "_after_sha256"] = "unchanged"
    (directory / "meta.json").write_text(json.dumps(meta))
    for v in range(6):
        np.save(directory / ("sums_V%d.npy" % (v + 1)), vectors[:, v].sum(axis=0).astype(np.float32))
    with (directory / "batches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["J_disc_sum", "return_sum", "n_episodes"])
        writer.writeheader()
        for b in range(50):
            writer.writerow(dict(J_disc_sum=80 * (index + b), return_sum=160 * (index + b), n_episodes=80))
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core, source = stats._load_core(args.root)
    checks = []
    def check(name, truth):
        if not truth:
            raise AssertionError(name)
        checks.append(name)
    def close(a, b):
        return np.allclose(a, b, rtol=2e-12, atol=2e-12)
    interval = stats.jackknife(4., list(np.linspace(3., 5., 10)))
    check("t90 df9 critical", abs(interval["critical"] - 1.8331129326536335) < 1e-12)
    deleted = np.linspace(3., 5., 10)
    expected_se = np.sqrt(.9 * ((deleted - deleted.mean()) ** 2).sum())
    check("literal t jackknife width", close(interval["lo"], 4 - interval["critical"] * expected_se))
    check("undefined replicate is unavailable interval", stats.jackknife(1., [None, 1.])["status"] == "estimate_without_interval")
    rng = np.random.default_rng(884412)
    vectors = rng.normal(size=(10, 50, 6, 9)) + np.arange(1, 7)[None, None, :, None] * .4
    with tempfile.TemporaryDirectory(prefix="clean-v3-stat-check-") as temp:
        root = Path(temp)
        paths = [fixture(root / ("P0_j%02d" % i), i, value) for i, value in enumerate(vectors)]
        data = core.PointData("P0", paths, variants=stats.VARIANTS)
        summary = stats.statistics_from_data(data)
        # Recompute the full and each deletion directly from saved float32 sums,
        # using the saved float64 within-batch moments as the true inputs.
        def explicit(ids):
            k = 50 * len(ids)
            total = data.sums[:, ids].sum(axis=1)
            q = data.q[ids].sum(axis=0)
            cross = total @ total.T
            return ((cross - q) / (k * (k - 1)),
                    (np.diag(q) - np.diag(cross) / k) / (k - 1))
        full_signal, full_trace = explicit(np.arange(10))
        deleted_values = [explicit(np.delete(np.arange(10), i)) for i in range(10)]
        check("pairwise U-statistic excludes same-batch products", close(data.full_moments["cross_signal"], full_signal))
        check("batch variance direct recomputation", close(data.full_moments["trace"], full_trace))
        check("theory squared scaling and episode factor", close(summary["estimators"]["V6"]["trace_ep"]["value"],
              80 * 2500 ** 2 * full_trace[5]))
        full_B = 80 * full_trace[5] / full_signal[3, 3]
        deleted_B = [80 * trace[5] / signal[3, 3] for signal, trace in deleted_values]
        independent_B = stats.jackknife(full_B, deleted_B)
        check("joint deletion of V6 numerator and V4 denominator", close(
            summary["estimators"]["V6"]["B_ep"]["se"], independent_B["se"]))
        check("B per-update scales value and interval by80", close(summary["estimators"]["V5"]["B_update"]["hi"],
            summary["estimators"]["V5"]["B_ep"]["hi"] / 80))
        check("trained V2 uses its own mean", close(summary["trained"]["V2"]["B_ep"]["value"],
            80 * full_trace[1] / full_signal[1, 1]))
        check("trained cosine excludes same-batch cross products", close(summary["trained"]["V1"]["cos_V4"]["value"],
            full_signal[0, 3] / np.sqrt(full_signal[0, 0] * full_signal[3, 3])))
        check("variance comparison uses paired trace cohort", close(summary["variance_ratios"]["V5_over_V6"]["value"],
            full_trace[4] / full_trace[5]))
        check("reward group jackknife uses t90", summary["metrics"]["J_disc"]["critical"] == interval["critical"])
        # Signal resolution must use the wider t interval, not old normal1.645.
        data.full_moments["signal"] = data.full_moments["signal"].copy()
        data.full_moments["signal"][3] = -.01
        for i, moment in enumerate(data.deleted_moments):
            moment["signal"] = moment["signal"].copy()
            moment["signal"][3] = -.01 + .01 * (i - 4.5)
        unresolved = stats._noise_scale(data, "V6", "V4")
        check("nonpositive full signal reports bound without sqrt", unresolved["status"] == "lower_bound" and unresolved["value"] is None)
        check("bound uses t upper signal limit", close(unresolved["lower_bound"],
            80 * data.full_moments["trace"][5] / stats._signal(data, "V4")["hi"]))
        check("negative full signal norm is unavailable", stats._norm(data, "V4")["value"] is None)
        data.full_moments["signal"][3] = .01
        check("negative deleted signal leaves norm interval unavailable", stats._norm(data, "V4")["status"] == "estimate_without_interval")
        json.dumps(summary, allow_nan=False)
        check("metric schema strict JSON", True)
        admin = root / "runs/clean_v3_collection_20261012"
        c_path = fixture(root / "collect/C1_j00", 500, vectors[0], point="D1")
        cleanup = dict(status="passed", all_children_exited=True, remaining_active_children=[],
                       remaining_direct_children=[], errors=[], primary_error=None,
                       workers=[dict(exitcode=0, alive=False) for _ in range(40)])
        (c_path / "cleanup.json").write_text(json.dumps(cleanup))
        worker = admin / "workers/C1_j00"
        worker.mkdir(parents=True)
        manifest = dict(status="complete", subprocess_exitcode=0, remaining_process_group_members=[],
                        external_point_id="C1", collector_point="D1", started_utc="2026-10-12T01:00:00+00:00")
        (worker / "manifest.json").write_text(json.dumps(manifest))
        entry = dict(run_id="C1_j00", status="complete", scheduler_state="COMPLETED", exit_code="0:0", job_id="1")
        (admin / "campaign.json").write_text(json.dumps(dict(entries=[entry])))
        groups, inventory = stats.discover_groups(root, core)
        check("C controller schema and internal alias accepted", len(groups["C1"]) == 1 and groups["C1"][0]["collector_point"] == "D1")
        entry["status"] = "running"
        (admin / "campaign.json").write_text(json.dumps(dict(entries=[entry])))
        groups, inventory = stats.discover_groups(root, core)
        check("native completion alone cannot admit unfinished scheduler job", not groups["C1"] and "scheduler_not_complete" in inventory[0]["exclusion_reasons"])
    records = [{"run_id": "z", "job_id": "2", "started_utc": "2026-10-12T01:00:00+00:00"},
               {"run_id": "a", "job_id": "1", "started_utc": "2026-10-12T02:00:00+00:00"}]
    check("selection follows actual launch instead of name", sorted(records, key=stats._launch_key)[0]["run_id"] == "z")
    result = {"passed": True, "checks": checks, "count": len(checks), "source": str(source),
              "python": __import__("sys").version, "synthetic_only": True, "rollouts": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
