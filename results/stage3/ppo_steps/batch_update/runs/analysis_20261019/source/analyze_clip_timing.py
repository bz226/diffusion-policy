#!/usr/bin/env python3
"""DPPO clip timing from stored Part C results; Python 3.8, CPU only.

Two explicit phases prevent interpretation after a failed prerequisite gate.
The gates phase reads each existing group sum exactly once and preserves small
sufficient statistics. The analyze phase requires a passing gates.json.
No models, gradients, samplers, GPUs, or scheduler APIs are imported.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

POINTS = ((35, 32), (70, 63), (105, 95))
G = 10
N = 50
P = 568392
FIELDS = ("switched_off_frac", "clipfrac", "approx_kl", "mean_abs_ratio_minus_one")
FIELD_NAMES = ("f", "clipfrac", "approx_kl", "abs_ratio")
EXPECTED_C = [0, 0, 0, 0, .226, .304, .373, .436, .719, .847,
              .962, 1.066, 1.385, 1.547, 1.692, 1.823, 2.167, 2.350, 2.515, 2.664]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite %s" % path)
    path.write_text(json.dumps(safe_json(value), indent=2, sort_keys=True) + "\n")


def pushes(beta=.9):
    """Expected own earlier pushes, summing complete earlier epochs only."""
    result = np.zeros(20, dtype=np.float64)
    for s in range(1, 21):
        for epoch in range((s - 1) // 4):
            result[s - 1] += sum(1.0 - beta ** (s - t)
                                 for t in range(4 * epoch + 1, 4 * epoch + 5)) / 4.0
    return result


def jk(estimate, replicates, status="ok"):
    """Ordinary delete-one-group interval centered on the full estimate."""
    reps = np.asarray(replicates, dtype=np.float64)
    valid = np.isfinite(reps)
    value = float(estimate) if np.isfinite(estimate) else None
    out = {"estimate": value, "lo": None, "hi": None, "se": None,
           "status": status, "valid_replicates": int(valid.sum()),
           "total_replicates": int(len(reps))}
    if value is None:
        out["status"] = "undefined_estimate"
    elif not np.all(valid):
        out["status"] = "interval_undefined_nonfinite_replicates"
    elif len(reps) > 1:
        se = math.sqrt((len(reps) - 1.0) / len(reps) *
                       float(np.sum((reps - reps.mean()) ** 2)))
        radius = float(student_t.ppf(.95, len(reps) - 1)) * se
        out.update(lo=value - radius, hi=value + radius, se=se)
    return out


def undefined(status):
    return {"estimate": None, "lo": None, "hi": None, "se": None,
            "status": status, "valid_replicates": 0, "total_replicates": G}


def sqrt_interval(square):
    """Transform signed squared-norm interval, keeping signed input elsewhere."""
    out = dict(square)
    value = square["estimate"]
    out["estimate"] = math.sqrt(value) if value is not None and value > 0 else None
    out["lo"] = math.sqrt(max(0.0, square["lo"])) if square["lo"] is not None else None
    out["hi"] = math.sqrt(max(0.0, square["hi"])) if square["hi"] is not None else None
    out["se"] = None
    out["status"] = "sqrt_transform_of_signed_interval"
    if out["estimate"] is None:
        out["status"] = "nonpositive_squared_norm"
    return out


def between_group_moments(group_means):
    """M and joint leave-group M without constructing a (G*S)^2 Gram.

    group_means is [groups, steps, coordinates], float64. Returned native
    totals are squared norms of group SUMS (not group means), for variance.
    This pure function is also suitable for small explicit arithmetic checks.
    """
    a = np.asarray(group_means, dtype=np.float64)
    n = a.shape[0]
    total = a.sum(axis=0, dtype=np.float64)
    tt = np.matmul(total, total.T)
    self_gram = np.stack([np.matmul(x, x.T) for x in a])
    diag_sum = self_gram.sum(axis=0, dtype=np.float64)
    full = (tt - diag_sum) / (n * (n - 1.0))
    leave = []
    leave_total_norm = []
    for i, x in enumerate(a):
        tx = np.matmul(total, x.T)
        reduced_tt = tt - tx - tx.T + self_gram[i]
        leave.append((reduced_tt - diag_sum + self_gram[i]) /
                     ((n - 1.0) * (n - 2.0)))
        leave_total_norm.append(np.diag(reduced_tt))
    return full, np.asarray(leave), np.diag(tt), np.asarray(leave_total_norm)


def native_moments(total_norm_sq, q, count):
    q = np.asarray(q, dtype=np.float64)
    signal = (total_norm_sq - q) / (count * (count - 1.0))
    variance = (q - total_norm_sq / count) / (count - 1.0)
    return signal, variance


def validate_groups(root, itr, update):
    found = []
    for path in sorted((root / "runs").glob("C%d_j*/meta.json" % itr)):
        meta = read_json(path)
        if meta.get("status") == "completed":
            found.append((path.parent, meta))
    if len(found) != G:
        raise ValueError("Expected 10 completed groups at %d updates, found %d" %
                         (update, len(found)))
    seeds = set()
    for path, meta in found:
        if (meta.get("part") != "C" or meta.get("checkpoint_itr") != itr or
                meta.get("updates") != update or meta.get("K") != N or
                meta.get("completed_batches") != N or meta.get("parameter_count") != P or
                meta.get("stored_dtype") != "float64" or meta.get("group_vector_kind") != "sum"):
            raise ValueError("Unexpected metadata: %s" % path)
        if meta["seed"] in seeds:
            raise ValueError("Duplicate group seed")
        seeds.add(meta["seed"])
    return found


def load_point(root, itr, update):
    groups = validate_groups(root, itr, update)
    inventory = []
    scalars = np.empty((G, 20, len(FIELDS)), dtype=np.float64)
    q = np.empty((G, 20), dtype=np.float64)
    available = all((path / ("sums_step%02d.npy" % s)).exists()
                    for path, _ in groups for s in range(1, 21))
    vectors = np.empty((G, 20, P), dtype=np.float64) if available else None
    f1_values = []
    abs1_values = []
    for g, (path, meta) in enumerate(groups):
        rows = read_csv(path / "batches.csv")
        indexed = {}
        for row in rows:
            key = (int(row["batch"]), int(row["step"]))
            if key in indexed:
                raise ValueError("Duplicate batch/step in %s" % path)
            indexed[key] = row
        if set(indexed) != {(b, s) for b in range(N) for s in range(1, 21)}:
            raise ValueError("Not all 50 batches reached all 20 steps: %s" % path)
        for s in range(1, 21):
            by_step = [indexed[(b, s)] for b in range(N)]
            data = np.asarray([[float(r[key]) for key in FIELDS] for r in by_step], dtype=np.float64)
            norms = np.asarray([float(r["grad_sq"]) for r in by_step], dtype=np.float64)
            if not np.all(np.isfinite(data)) or not np.all(np.isfinite(norms)):
                raise ValueError("Nonfinite stored batch scalar")
            scalars[g, s - 1] = data.mean(axis=0)
            q[g, s - 1] = norms.sum(dtype=np.float64)
            if s == 1:
                f1_values.extend(data[:, 0].tolist())
                abs1_values.extend(data[:, 3].tolist())
            if available:
                value = np.load(str(path / ("sums_step%02d.npy" % s)), allow_pickle=False)
                if value.shape != (P,) or value.dtype != np.float64 or not np.all(np.isfinite(value)):
                    raise ValueError("Invalid float64 step sum: %s, step %d" % (path, s))
                vectors[g, s - 1] = value / N
        inventory.append({"directory": str(path), "seed": meta["seed"], "K": N,
                          "source_commit": meta["commit"], "meta": str(path / "meta.json"),
                          "scalars": str(path / "batches.csv"),
                          "group_sums": [str(path / ("sums_step%02d.npy" % s))
                                         for s in range(1, 21)]})
    result = {"scalar_groups": scalars, "q_groups": q,
              "f1_values": np.asarray(f1_values), "abs1_values": np.asarray(abs1_values)}
    if available:
        m, leave, total_norm, leave_norm = between_group_moments(vectors)
        result.update(M=m, M_leave=leave,
                      total_norm_sq=total_norm * N ** 2,
                      leave_total_norm_sq=leave_norm * N ** 2)
        del vectors
    return result, inventory, available


def phase_gates(root, output):
    gates = {"G1": {"passed": True, "details": []},
             "G2": {"passed": True, "details": []},
             "G3": {"passed": True, "details": []},
             "G4": {"passed": True, "details": []}}
    table_path = root / "figs/noise_actual_table.csv"
    table = [r for r in read_csv(table_path) if r["method"] == "DPPO"]
    reference = {(int(r["update"]), int(r["step"])): r for r in table}
    if len(reference) != 60:
        raise ValueError("Expected exactly 60 DPPO noise_actual_table rows")
    values = pushes()
    gates["G4"]["passed"] = bool(np.array_equal(np.round(values, 3), np.asarray(EXPECTED_C)))
    gates["G4"]["details"] = {"computed": values.tolist(), "expected_rounded": EXPECTED_C}
    archive = {"C": values}
    inventory = {"reference_table": str(table_path), "points": []}
    for itr, update in POINTS:
        print("Reading stored sums at %d updates" % update, flush=True)
        try:
            data, items, available = load_point(root, itr, update)
        except Exception as exc:
            gates["G1"]["passed"] = False
            gates["G1"]["details"].append({"update": update, "error": repr(exc)})
            continue
        archive.update({"u%d_%s" % (update, key): val for key, val in data.items()})
        inventory["points"].append({"update": update, "checkpoint_itr": itr,
                                     "groups": items, "group_sums_available": available})
        gates["G1"]["details"].append({"update": update, "K": 500, "groups": G,
                                       "per_batch_scalars": True, "group_sums": available,
                                       "fallback": None if available else "timing_only"})
        f1 = data["f1_values"]
        abs1 = data["abs1_values"]
        ok = bool(np.all(f1 == 0.0) and abs1.mean() < 1e-4)
        gates["G2"]["passed"] &= ok
        gates["G2"]["details"].append({"update": update, "passed": ok,
                                       "f1_mean": float(f1.mean()), "f1_max": float(f1.max()),
                                       "mean_abs_ratio1": float(abs1.mean()),
                                       "max_batch_abs_ratio1": float(abs1.max())})
        if not available:
            gates["G3"]["details"].append({"update": update, "status": "skipped_missing_group_sums"})
            continue
        reconstructed_signal, reconstructed_variance = native_moments(
            data["total_norm_sq"], data["q_groups"].sum(axis=0), G * N)
        for key, reconstructed in (("signal_sq", reconstructed_signal),
                                   ("variance", reconstructed_variance)):
            stored = np.asarray([float(reference[(update, s)][key])
                                 for s in range(1, 21)], dtype=np.float64)
            difference = np.abs(reconstructed - stored)
            relative_error = float(np.max(difference / np.maximum(np.abs(stored), 1e-300)))
            evidence = {"max_relative_error": relative_error,
                        "max_absolute_error": float(np.max(difference)),
                        "rtol": 1e-8, "atol": 0.0,
                        "passed": bool(np.allclose(reconstructed, stored, rtol=1e-8, atol=0))}
            gates["G1"]["details"][-1]["native_" + key + "_table_agreement"] = evidence
            gates["G1"]["passed"] &= evidence["passed"]
        for s in range(20):
            squared = jk(data["M"][s, s], data["M_leave"][:, s, s])
            transformed = sqrt_interval(squared)
            ref = reference[(update, s + 1)]
            ref_lo, ref_hi = float(ref["mean_norm_lo"]), float(ref["mean_norm_hi"])
            positive = squared["estimate"] > 0
            overlap = positive and max(transformed["lo"], ref_lo) <= min(transformed["hi"], ref_hi)
            detail = {"update": update, "step": s + 1, "passed": bool(overlap),
                      "between_group_signal_sq": squared, "between_group_mean_norm": transformed,
                      "stored_mean_norm": float(ref["mean_norm"]),
                      "stored_norm_interval": [ref_lo, ref_hi],
                      "reason": "intervals_overlap" if overlap else
                      ("nonpositive_Mss_sqrt_undefined" if not positive else "intervals_do_not_overlap")}
            gates["G3"]["details"].append(detail)
            gates["G3"]["passed"] &= bool(overlap)
    gates["passed"] = all(gates[key]["passed"] for key in ("G1", "G2", "G3", "G4"))
    gates["G3_definition"] = "Between-group sqrt interval (transformed signed Mss t9 interval) overlaps stored norm interval; full Mss must be positive."
    gates["G5"] = {"status": "performed_by_driver_after_all_outputs"}
    np.savez(str(output / "grouped_moments.npz"), **archive)
    save_json(output / "input_inventory.json", inventory)
    save_json(output / "reference_rows.json", table)
    save_json(output / "gates.json", gates)
    print("Prerequisite gates: %s" % ("PASS" if gates["passed"] else "FAIL"), flush=True)
    return 0 if gates["passed"] else 2


def shape_values(matrix, c):
    """Pure shape/direction function; undefined square roots stay NaN."""
    diag = np.diag(matrix)
    dsq = diag - 2 * matrix[:, 0] + matrix[0, 0]
    dsq[0] = 0.0  # algebraic identity, not an estimated negative-value repair
    d = np.full(20, np.nan, dtype=np.float64)
    d[dsq > 0] = np.sqrt(dsq[dsq > 0])
    d[0] = 0.0
    k = float(np.dot(c[4:8], d[4:8]) / np.dot(c[4:8], c[4:8]))
    k56 = float(np.dot(c[4:6], d[4:6]) / np.dot(c[4:6], c[4:6]))
    q = np.full(20, np.nan)
    q56 = np.full(20, np.nan)
    if k > 0:
        q[4:] = d[4:] / (k * c[4:])
    if k56 > 0:
        q56[4:] = d[4:] / (k56 * c[4:])
    cross = matrix[:, 19] - matrix[:, 0] - matrix[0, 19] + matrix[0, 0]
    cosd = np.full(20, np.nan)
    valid = (d > 0) & np.isfinite(d)
    if d[19] > 0:
        cosd[valid] = cross[valid] / (d[valid] * d[19])
    cos0 = np.full(20, np.nan)
    if diag[0] > 0:
        valid0 = diag > 0
        cos0[valid0] = matrix[valid0, 0] / np.sqrt(diag[valid0] * diag[0])
    component = (matrix[0, 19] - matrix[0, 0]) / d[19] if d[19] > 0 else float("nan")
    return {"delta_sq": dsq, "delta_norm": d, "k": k, "k56": k56,
            "q": q, "q56": q56, "cos_delta20": cosd,
            "cos_onpolicy": cos0, "component": component}


def first_crossing(values, threshold, first_step=1, less=False):
    for i in range(first_step - 1, len(values)):
        if math.isfinite(values[i]) and ((values[i] < threshold) if less else (values[i] >= threshold)):
            return float(i + 1)
    return float("nan")


def reference_metric(row, key):
    value, low, high = (float(row[key + suffix]) for suffix in ("", "_lo", "_hi"))
    return {"estimate": value, "lo": low, "hi": high,
            # The stored norm interval may be a nonlinear square-root
            # transformation, so do not infer a standard error from its width.
            "se": None,
            "status": "stored_noise_actual_table_interval"}


def phase_analyze(root, output):
    gates = read_json(output / "gates.json")
    if not gates["passed"]:
        raise RuntimeError("Prerequisite gate failed; analysis is forbidden")
    a = np.load(str(output / "grouped_moments.npz"), allow_pickle=False)
    c = a["C"]
    refs = {(int(r["update"]), int(r["step"])): r for r in read_json(output / "reference_rows.json")}
    rows, summaries = [], []
    for itr, update in POINTS:
        prefix = "u%d_" % update
        scalars = a[prefix + "scalar_groups"]
        full_scalars = scalars.mean(axis=0)
        leave_scalars = (scalars.sum(axis=0)[None, :, :] - scalars) / (G - 1)
        summary = {"method": "DPPO", "update": update, "checkpoint_itr": itr}
        f = full_scalars[:, 0]
        half = first_crossing(f, .5 * max(f))
        half_reps = [first_crossing(x[:, 0], .5 * max(x[:, 0])) for x in leave_scalars]
        summary["s_half"] = jk(half, half_reps)
        has_shape = prefix + "M" in a
        if has_shape:
            m, ml = a[prefix + "M"], a[prefix + "M_leave"]
            shapes = shape_values(m, c)
            shape_reps = [shape_values(x, c) for x in ml]
            delta20_sq = jk(shapes["delta_sq"][19], [x["delta_sq"][19] for x in shape_reps])
            delta20_resolved = delta20_sq["lo"] is not None and delta20_sq["lo"] > 0
            mean1sq = jk(m[0, 0], ml[:, 0, 0])
            mean1_resolved = mean1sq["lo"] is not None and mean1sq["lo"] > 0
            summary["between_group_mean1_sq"] = mean1sq
            summary["between_group_mean1_resolved"] = mean1_resolved
            stored1 = refs[(update, 1)]
            native1_resolved = float(stored1["signal_sq_lo"]) > 0
            summary["stored_mean1_resolved"] = native1_resolved
            summary["mean1_or_upper"] = reference_metric(stored1, "mean_norm")
            summary["mean1_or_upper"]["comparison_value"] = float(stored1["mean_norm"] if native1_resolved else stored1["mean_norm_hi"])
            summary["mean1_or_upper"]["comparison_kind"] = "point" if native1_resolved else "upper_bound"
            for key in ("k", "k56", "component"):
                summary[key] = jk(shapes[key], [x[key] for x in shape_reps])
            for qkey, skey in (("q", "s_dep"), ("q56", "s_dep56")):
                crossing = first_crossing(shapes[qkey], .8, first_step=9, less=True)
                reps = [first_crossing(x[qkey], .8, first_step=9, less=True) for x in shape_reps]
                summary[skey] = jk(crossing, reps)
                if not math.isfinite(crossing):
                    summary[skey]["status"] = "not_reached_through_20"
                for step in (9, 12, 16, 20):
                    summary["%s_%d" % (qkey, step)] = jk(shapes[qkey][step - 1],
                                                        [x[qkey][step - 1] for x in shape_reps])
            qgroups = a[prefix + "q_groups"]
            _, variance = native_moments(a[prefix + "total_norm_sq"], qgroups.sum(axis=0), G * N)
            _, variance_reps = native_moments(a[prefix + "leave_total_norm_sq"],
                                              qgroups.sum(axis=0)[None, :] - qgroups, (G - 1) * N)
        for i in range(20):
            row = {"method": "DPPO", "update": update, "checkpoint_itr": itr,
                   "step": i + 1, "C": float(c[i]), "K": 500, "groups": 10}
            for j, key in enumerate(FIELD_NAMES):
                row[key] = jk(full_scalars[i, j], leave_scalars[:, i, j])
            ref = refs[(update, i + 1)]
            row["mean_norm"] = reference_metric(ref, "mean_norm")
            row["variance"] = reference_metric(ref, "variance")
            if has_shape:
                # The point matches stored table arithmetic; only the ratio's
                # uncertainty is newly computed, with a shared deleted group.
                variance_point = float(ref["variance"]) / float(refs[(update, 1)]["variance"])
                row["variance_ratio"] = jk(variance_point, variance_reps[:, i] / variance_reps[:, 0])
                row["variance_recomputed"] = float(variance[i])
                row["between_group_mean_sq"] = jk(m[i, i], ml[:, i, i])
                row["delta_sq"] = jk(shapes["delta_sq"][i], [x["delta_sq"][i] for x in shape_reps])
                row["delta_resolved"] = row["delta_sq"]["lo"] is not None and row["delta_sq"]["lo"] > 0
                row["delta_norm"] = sqrt_interval(row["delta_sq"])
                for key in ("q", "q56", "cos_delta20", "cos_onpolicy"):
                    row[key] = jk(shapes[key][i], [x[key][i] for x in shape_reps])
                if not row["delta_resolved"] or not delta20_resolved:
                    row["cos_delta20"] = undefined("unresolved_delta_or_delta20")
                if not mean1_resolved:
                    row["cos_onpolicy"] = undefined("unresolved_onpolicy_mean")
            else:
                row["delta_resolved"] = False
                for key in ("variance_ratio", "delta_sq", "delta_norm", "q", "q56", "cos_delta20", "cos_onpolicy"):
                    row[key] = undefined("missing_group_sums")
            rows.append(row)
        summaries.append(summary)
    result = {"method": "DPPO", "rows": rows, "summaries": summaries,
              "interval": "90% delete-one-group jackknife, t quantile on 9 df",
              "mean_norm_and_variance": "Unchanged estimates and intervals from noise_actual_table",
              "shape": "Between-group M; all ratios use jointly deleted groups; cosines are never clipped",
              "resolution": "Delta and on-policy direction resolution uses signed between-group squared-norm interval; stored norm resolution also reported",
              "undefined_replicates": "An interval is unavailable if any required leave-group value is undefined",
              "threshold_intervals": "s_half and s_dep use the same joint delete-group rule; censored departure replicas yield no interval",
              "signed_norms": "Mss and delta_sq are retained without clipping; square-root interval endpoints intersect the nonnegative domain",
              "gates": gates}
    save_json(output / "analysis.json", result)
    write_tables(output, result)
    return 0


def format_metric(item):
    if not isinstance(item, dict):
        return str(item)
    if item.get("estimate") is None:
        return "N/A (%s)" % item.get("status", "undefined")
    value = "%.6g" % item["estimate"]
    if item.get("lo") is not None and item.get("hi") is not None:
        return "%s [%.6g, %.6g]" % (value, item["lo"], item["hi"])
    return value + " [interval unavailable]"


def write_tables(output, result):
    flat = []
    for kind, items in (("step", result["rows"]), ("summary", result["summaries"])):
        for source in items:
            row = {"row_type": kind}
            for key, value in source.items():
                if isinstance(value, dict):
                    for suffix, item in value.items():
                        row[key + "_" + suffix] = item
                else:
                    row[key] = value
            flat.append(row)
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with (output / "clip_timing_table.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# DPPO clip timing", "", "500 batches per checkpoint, 10 independent groups of 50. All intervals are 90% delete-one-group jackknife intervals using t on 9 degrees of freedom. Norm and variance columns preserve the stored per-gradient table. N/A indicates an undefined estimate or interval, without clipping or substituting a value.", ""]
    sections = [("Clip timing", ("C", "f", "clipfrac", "approx_kl", "abs_ratio")),
                ("Mean and variance", ("mean_norm", "variance", "variance_ratio")),
                ("Extra mean and direction", ("delta_sq", "delta_norm", "delta_resolved", "q", "q56", "cos_delta20", "cos_onpolicy"))]
    for title, keys in sections:
        lines.extend(["## " + title, "", "| Updates | Step | " + " | ".join(keys) + " |",
                      "|---|---|" + "---|" * len(keys)])
        for row in result["rows"]:
            lines.append("| %d | %d | %s |" % (row["update"], row["step"], " | ".join(format_metric(row[k]) for k in keys)))
        lines.append("")
    keys = ("s_half", "k", "k56", "s_dep", "s_dep56", "q_9", "q_12", "q_16", "q_20", "component", "mean1_or_upper")
    lines.extend(["## Checkpoint summaries", "", "| Updates | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)])
    for row in result["summaries"]:
        lines.append("| %d | %s |" % (row["update"], " | ".join(format_metric(row.get(k, undefined("missing_group_sums"))) for k in keys)))
    lines.extend(["", "Signed squared-norm intervals determine resolution. A square-root interval is the monotone transformation after intersecting its signed interval with the nonnegative domain. Cosines are never clipped. Direction cosines are omitted when the required mean is unresolved. The on-policy component is compared with the existing mean norm if resolved, otherwise its stored upper interval endpoint; comparison_kind and comparison_value appear in CSV and JSON.", ""])
    (output / "clip_timing_table.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("gates", "analyze"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "gates":
        if any((args.output / name).exists() for name in ("gates.json", "grouped_moments.npz", "input_inventory.json")):
            raise FileExistsError("Gate outputs already exist; use a new run directory")
        return phase_gates(args.root.resolve(), args.output.resolve())
    return phase_analyze(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
