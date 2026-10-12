"""Independently recompute a clean-v3 CSV from its exact saved group inputs.

No PointData, clean_v3_stats, or plotting helper is used. This is read-only
scientific verification; its only write is the requested concise check record.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def verify(render, expected_points=None):
    render = Path(render).resolve()
    record = json.loads((render / "render_record.json").read_text())
    with (render / "points_table.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if record["version"] != 3:
        raise ValueError("Expected a version-3 render")
    if expected_points is not None and len(record["statistics"]["points"]) != expected_points:
        raise ValueError("Wrong number of completed point cohorts")
    evidence = {"compared_numeric_cells": 0, "max_relative_difference": 0., "worst": None}
    source_groups = {}
    for point, data in record["statistics"]["points"].items():
        jobs = data["selected_jobs"]
        if len(jobs) != 10:
            raise ValueError("Main point must have ten groups")
        matrices, grouped, rewards = [], [], []
        source_groups[point] = [job["path"] for job in jobs]
        for job in jobs:
            directory = Path(job["path"])
            meta = json.loads((directory / "meta.json").read_text())
            if meta["completed_batches"] != 50 or meta["variant_ids"] != ["V%d" % v for v in range(1, 7)]:
                raise ValueError("Wrong group geometry or estimator order")
            matrices.append(meta["gradient_cross_sum"])
            grouped.append(np.stack([np.load(directory / ("sums_V%d.npy" % v)).astype(np.float64)
                                     for v in range(1, 7)]))
            with (directory / "batches.csv").open(newline="") as stream:
                batch_rows = list(csv.DictReader(stream))
            if len(batch_rows) != 50 or any(int(row["n_episodes"]) != 80 for row in batch_rows):
                raise ValueError("Wrong measurement episode counts")
            rewards.append([sum(float(row[key + "_sum"]) for row in batch_rows) / 4000.
                            for key in ("J_disc", "return")])
        q = np.asarray(matrices, dtype=np.float64)
        g = np.asarray(grouped, dtype=np.float64)
        rr = np.asarray(rewards, dtype=np.float64)
        del grouped
        n = len(jobs)
        def moment(ids):
            total = g[ids].sum(axis=0, dtype=np.float64)
            batch_cross = q[ids].sum(axis=0, dtype=np.float64)
            k = 50 * len(ids)
            total_cross = total @ total.T
            return ((total_cross - batch_cross) / (k * (k - 1)),
                    (np.diag(batch_cross) - np.diag(total_cross) / k) / (k - 1))
        moments = [moment(np.arange(n))] + [moment(np.delete(np.arange(n), i)) for i in range(n)]
        def jk(operation):
            values = [operation(*value) for value in moments]
            full, deleted = values[0], values[1:]
            if full is None or any(value is None for value in deleted):
                return [full, None, None, None]
            deleted = np.asarray(deleted)
            se = np.sqrt((n - 1.) / n * np.sum((deleted - deleted.mean()) ** 2))
            width = t.ppf(.95, n - 1) * se
            return [full, full - width, full + width, se]
        def compare(row, key, expected):
            for suffix, value in zip(("", "_lo", "_hi", "_se"), expected):
                actual = float(row[key + suffix]) if row[key + suffix] else None
                if value is None:
                    if actual is not None:
                        raise ValueError("Expected undefined: " + point + ":" + key + suffix)
                else:
                    if actual is None:
                        raise ValueError("Missing numeric CSV cell: " + point + ":" + key + suffix)
                    error = abs(actual - value) / max(abs(value), 1.)
                    if error > evidence["max_relative_difference"]:
                        evidence["max_relative_difference"] = float(error)
                        evidence["worst"] = [point, key, suffix]
                    if error >= 2e-8:
                        raise ValueError("Independent recomputation differs: " + point + ":" + key + suffix)
                evidence["compared_numeric_cells"] += 1
        def noise_scale(variance_index, signal_index):
            signal = jk(lambda u, z: u[signal_index, signal_index])
            answer = jk(lambda u, z: 80 * z[variance_index] / u[signal_index, signal_index]
                        if u[signal_index, signal_index] > 0 else None)
            if signal[1] is None or signal[0] <= 0 or signal[1] <= 0:
                bound = (80 * moments[0][1][variance_index] / signal[2]
                         if signal[2] and signal[2] > 0 else None)
                return [None] * 4, bound
            return answer, None
        for row in [item for item in rows if item["point"] == point]:
            if (row["K"], row["m"], row["groups"]) != ("500", "40000", "10"):
                raise ValueError("CSV main cohort size mismatch")
            v = int(row["estimator"][1:]) - 1
            compare(row, "u_norm", jk(lambda u, z: 2500 * np.sqrt(u[3, 3]) if u[3, 3] > 0 else None))
            compare(row, "trace_ep", jk(lambda u, z: 80 * 2500 ** 2 * z[v]))
            scale, bound = noise_scale(v, 3)
            compare(row, "B_ep", scale)
            compare(row, "B_update", [value / 80 if value is not None else None for value in scale])
            if bound is not None:
                actual = float(row["B_ep_lower_bound"])
                if abs(actual - bound) / max(abs(bound), 1.) >= 2e-8:
                    raise ValueError("Main noise-scale bound mismatch")
            for key, a, b in (("variance_V6_over_V4", 5, 3), ("variance_V5_over_V6", 4, 5)):
                compare(row, key, jk(lambda u, z: z[a] / z[b]))
            for v1 in (0, 1):
                scale, bound = noise_scale(v1, v1)
                compare(row, "V%d_B_ep" % (v1 + 1), scale)
                if bound is not None:
                    actual = float(row["V%d_B_ep_lower_bound" % (v1 + 1)])
                    if abs(actual - bound) / max(abs(bound), 1.) >= 2e-8:
                        raise ValueError("Trained-estimator bound mismatch")
                compare(row, "V%d_cos_V4" % (v1 + 1), jk(lambda u, z:
                    u[v1, 3] / np.sqrt(u[v1, v1] * u[3, 3]) if u[v1, v1] > 0 and u[3, 3] > 0 else None))
            for index, key in enumerate(("J_disc", "return")):
                values = rr[:, index]
                full = values.mean()
                deleted = np.asarray([np.delete(values, i).mean() for i in range(n)])
                se = np.sqrt((n - 1.) / n * np.sum((deleted - deleted.mean()) ** 2))
                width = t.ppf(.95, n - 1) * se
                compare(row, key, [full, full - width, full + width, se])
        del g
    evidence.update(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                    render=str(render), version=3, completed_points=list(record["statistics"]["points"]),
                    rows=len(rows), completed_point_rows=sum(row["K"] == "500" for row in rows),
                    method="Independent float64 recomputation of pairwise U signal, trace, joint t jackknife ratios, same-batch-excluded cosines, and grouped rewards; no analysis/statistics helpers imported.",
                    comparisons="Main CSV values and intervals, paired variance ratios, own-mean trained estimators, unresolved-signal bounds.",
                    source_files=[str(render / "render_record.json"), str(render / "points_table.csv")],
                    source_groups=source_groups,
                    files_per_group=["meta.json", "batches.csv"] + ["sums_V%d.npy" % v for v in range(1, 7)],
                    tolerance="abs difference / max(abs reference,1) < 2e-8",
                    no_simulation=True, source_script=str(Path(__file__).resolve()))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--expected-points", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an earlier verification record")
    result = verify(args.render, args.expected_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "completed_points", "compared_numeric_cells",
                      "max_relative_difference", "worst")}, indent=2))


if __name__ == "__main__":
    main()
