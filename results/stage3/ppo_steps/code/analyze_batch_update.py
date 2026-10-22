#!/usr/bin/env python3
"""Linear combinations of stored DPPO step-gradient moments, Python 3.8.

This consumes only the compact, gated float64 moments prepared separately.
It never imports a model, computes a gradient, samples a rollout, or uses a GPU.
The same ten deleted groups are used jointly throughout every derived ratio.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_clip_timing import jk, read_csv, read_json, save_json, sqrt_interval

UPDATES = (32, 63, 95)
GROUPS = 10
STEPS = 20
METRIC_NAMES = ("signal_sq", "mean_norm", "variance", "R", "expected_cosine",
                "inner_epoch1", "cos_epoch1", "a", "R_on")


def metric(estimate, replicates, status="ok", positive=False):
    out = jk(estimate, replicates, status=status)
    lo, hi = out["lo"], out["hi"]
    out["resolved"] = bool(lo is not None and hi is not None and
                           (lo > 0 if positive else (lo > 0 or hi < 0)))
    return out


def unavailable(reason):
    return {"estimate": None, "lo": None, "hi": None, "se": None,
            "status": reason, "resolved": False,
            "valid_replicates": 0, "total_replicates": GROUPS}


def norm_metric(square):
    out = sqrt_interval(square)
    out["resolved"] = bool(square["resolved"])
    return out


def ratio(numerator, denominator, positive=True):
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return np.nan
    if denominator <= 0 if positive else denominator == 0:
        return np.nan
    return numerator / denominator


def sqrt_positive(value):
    return math.sqrt(value) if np.isfinite(value) and value > 0 else np.nan


def cosine(inner, square_x, square_y):
    if square_x <= 0 or square_y <= 0:
        return np.nan
    return ratio(inner, math.sqrt(square_x * square_y))


def quad(matrix, left, right=None):
    if right is None:
        right = left
    return float(np.dot(left, np.dot(matrix, right)))


def coefficients():
    result = []
    for epoch in range(1, 6):
        vec = np.zeros(STEPS, dtype=np.float64)
        vec[(epoch - 1) * 4:epoch * 4] = .25
        result.append(("epoch%d" % epoch, epoch, vec))
    result.append(("batch", None, np.ones(STEPS, dtype=np.float64) / STEPS))
    return result


def moment_metrics(m, v, lm, lv, coeff, ref, ref_resolved):
    signal = quad(m, coeff)
    ls = np.asarray([quad(x, coeff) for x in lm])
    variance = quad(v, coeff)
    lvar = np.asarray([quad(x, coeff) for x in lv])
    inner = quad(m, coeff, ref)
    li = np.asarray([quad(x, coeff, ref) for x in lm])
    ref_signal = quad(m, ref)
    lr = np.asarray([quad(x, ref) for x in lm])
    out = {"signal_sq": metric(signal, ls, positive=True),
           "variance": metric(variance, lvar, positive=True),
           "inner_epoch1": metric(inner, li)}
    out["mean_norm"] = norm_metric(out["signal_sq"])
    ratios = np.asarray([ratio(a, b) for a, b in zip(lvar, ls)])
    relvar = ratio(variance, signal)
    out["R"] = metric(relvar, ratios)
    out["R"]["mean_resolved"] = out["signal_sq"]["resolved"]
    out["expected_cosine"] = metric(
        ratio(1.0, sqrt_positive(1.0 + relvar)),
        [ratio(1.0, sqrt_positive(1.0 + val)) for val in ratios])
    if ref_resolved and out["signal_sq"]["resolved"]:
        out["cos_epoch1"] = metric(cosine(inner, signal, ref_signal),
            [cosine(a, b, c) for a, b, c in zip(li, ls, lr)])
    else:
        out["cos_epoch1"] = unavailable("involved_mean_unresolved")
    if ref_resolved:
        component = ratio(inner, sqrt_positive(ref_signal))
        lc = np.asarray([ratio(a, sqrt_positive(b)) for a, b in zip(li, lr)])
        out["a"] = metric(component, lc)
        out["R_on"] = metric(ratio(variance, component ** 2),
            [ratio(a, b ** 2) for a, b in zip(lvar, lc)])
    else:
        out["a"] = unavailable("epoch1_mean_unresolved")
        out["R_on"] = unavailable("epoch1_mean_unresolved")
    return out


def eigensystem(gamma):
    values, vectors = np.linalg.eigh(gamma)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def turning_metrics(m, lm):
    extra = np.zeros((16, STEPS), dtype=np.float64)
    for j in range(16):
        extra[j, j + 4] = 1.0
        extra[j, 0] = -1.0
    gamma = np.matmul(np.matmul(extra, m), extra.T)
    leave_gamma = np.asarray([np.matmul(np.matmul(extra, x), extra.T) for x in lm])
    entries = [[metric(gamma[i, j], leave_gamma[:, i, j], positive=(i == j))
                for j in range(16)] for i in range(16)]
    cosines = []
    for i in range(16):
        row = []
        for j in range(16):
            if entries[i][i]["resolved"] and entries[j][j]["resolved"]:
                row.append(metric(cosine(gamma[i, j], gamma[i, i], gamma[j, j]),
                    [cosine(x[i, j], x[i, i], x[j, j]) for x in leave_gamma]))
            else:
                row.append(unavailable("involved_extra_mean_unresolved"))
        cosines.append(row)
    vals, vecs = eigensystem(gamma)
    # Fix full-sample signs deterministically. Only signs, never rotations, are
    # aligned in deleted groups, preserving the requested principal coordinates.
    for col in range(16):
        anchor = int(np.argmax(np.abs(vecs[:, col])))
        if vecs[anchor, col] < 0:
            vecs[:, col] *= -1.0
    lvals, lvecs, overlaps = [], [], []
    for lg in leave_gamma:
        ev, evec = eigensystem(lg)
        overlaps.append([])
        for col in range(16):
            overlap = float(np.dot(evec[:, col], vecs[:, col]))
            if overlap < 0:
                evec[:, col] *= -1.0
            overlaps[-1].append(abs(overlap))
        lvals.append(ev)
        lvecs.append(evec)
    lvals, lvecs, overlaps = map(np.asarray, (lvals, lvecs, overlaps))
    trace = float(vals.sum())
    ltrace = lvals.sum(axis=1)
    shares = [metric(ratio(vals[i], trace),
              [ratio(row[i], total) for row, total in zip(lvals, ltrace)])
              for i in range(16)]
    cumulative = [metric(ratio(float(vals[:n].sum()), trace),
                  [ratio(float(row[:n].sum()), total) for row, total in zip(lvals, ltrace)])
                  for n in (1, 2, 3)]
    coords = []
    for i in range(16):
        coords.append([metric(vecs[i, j] * sqrt_positive(vals[j]),
                       [ev[i, j] * sqrt_positive(el[j]) for el, ev in zip(lvals, lvecs)])
                       for j in range(2)])
    def cancellation(array):
        diag = np.diag(array)
        if np.any(diag <= 0) or array.sum() <= 0:
            return np.nan
        return math.sqrt(float(array.sum())) / float(np.sqrt(diag).sum())
    kappa = metric(cancellation(gamma), [cancellation(x) for x in leave_gamma])
    return {"steps": list(range(5, 21)), "Gamma": entries, "cosine": cosines,
            "eigenvalues": [metric(vals[i], lvals[:, i]) for i in range(16)],
            "eigenvalue_shares": shares, "cumulative_top_shares": cumulative,
            "coordinates": coords,
            "coordinate_stability": {
                "sign_rule": "largest absolute full-sample loading positive; deleted vectors sign-aligned",
                "min_leave_absolute_eigenvector_overlap": overlaps.min(axis=0).tolist(),
                "top2_positive_in_all_replicates": bool(np.all(lvals[:, :2] > 0)),
                "top2_order_stable_by_best_full_vector_match": [bool(all(
                    int(np.argmax(np.abs(np.dot(vecs.T, ev[:, j])))) == j
                    for ev in lvecs)) for j in range(2)],
                "negative_eigenvalues_retained": int(np.sum(vals < 0))},
            "kappa": kappa}


def analyze_point(archive, update):
    keys = ("M", "V", "M_P", "V_P", "M_cross")
    data = {key: np.asarray(archive["u%d_%s" % (update, key)], dtype=np.float64)
            for key in keys}
    leave = {key: np.asarray(archive["u%d_%s_leave" % (update, key)], dtype=np.float64)
             for key in keys}
    if any(x.shape != (20, 20) for x in data.values()) or any(
            x.shape != (10, 20, 20) for x in leave.values()):
        raise ValueError("Unexpected gated moment shape")
    if any(not np.isfinite(x).all() for x in list(data.values()) + list(leave.values())):
        raise ValueError("Nonfinite gated moment")
    m, v, lm, lv = data["M"], data["V"], leave["M"], leave["V"]
    combos = coefficients()
    ref, batch = combos[0][2], combos[-1][2]
    refsq = metric(quad(m, ref), [quad(x, ref) for x in lm], positive=True)
    quantities = [{"name": name, "epoch": epoch,
                   "metrics": moment_metrics(m, v, lm, lv, coeff, ref, refsq["resolved"])}
                  for name, epoch, coeff in combos]
    def summary_at(matrix, cov):
        s1, sb = quad(matrix, ref), quad(matrix, batch)
        v1, vb = quad(cov, ref), quad(cov, batch)
        inner = quad(matrix, batch, ref)
        return {"R_batch_over_epoch1": ratio(ratio(vb, sb), ratio(v1, s1)),
                "signal_kept": ratio(inner, s1),
                "R_on_batch_over_epoch1": ratio(ratio(vb * s1, inner ** 2), ratio(v1, s1))}
    summary = summary_at(m, v)
    lsumm = [summary_at(a, b) for a, b in zip(lm, lv)]
    summaries = {key: metric(value, [row[key] for row in lsumm])
                 for key, value in summary.items()}
    if not refsq["resolved"]:
        for key in ("signal_kept", "R_on_batch_over_epoch1"):
            summaries[key] = unavailable("epoch1_mean_unresolved")
    turning = turning_metrics(m, lm)
    summaries["kappa"] = turning["kappa"]
    mp, vp, lmp, lvp = data["M_P"], data["V_P"], leave["M_P"], leave["V_P"]
    pref = metric(quad(mp, ref), [quad(x, ref) for x in lmp], positive=True)
    adam = {"epoch1": moment_metrics(mp, vp, lmp, lvp, ref, ref, pref["resolved"]),
            "batch": moment_metrics(mp, vp, lmp, lvp, batch, ref, pref["resolved"])}
    mc, lmc = data["M_cross"], leave["M_cross"]
    den = metric(quad(mc, ref), [quad(x, ref) for x in lmc], positive=True)
    num = metric(quad(mc, ref, batch), [quad(x, ref, batch) for x in lmc])
    adam["gain_numerator"] = num
    adam["gain_denominator"] = den
    if den["resolved"]:
        adam["relative_gain"] = metric(ratio(quad(mc, ref, batch), quad(mc, ref)),
            [ratio(quad(x, ref, batch), quad(x, ref)) for x in lmc])
    else:
        adam["relative_gain"] = unavailable("onpolicy_gain_denominator_unresolved")
    adam["R_batch_over_epoch1"] = metric(
        ratio(ratio(quad(vp, batch), quad(mp, batch)), ratio(quad(vp, ref), quad(mp, ref))),
        [ratio(ratio(quad(b, batch), quad(a, batch)), ratio(quad(b, ref), quad(a, ref)))
         for a, b in zip(lmp, lvp)])
    return {"update": update, "quantities": quantities, "summaries": summaries,
            "turning": turning, "adam": adam}


def imported_metric(row, name, positive=False):
    def number(key):
        val = row.get(key)
        if val in (None, "", "nan", "None"):
            return None
        return float(val)
    estimate, lo, hi = (number(name + suffix) for suffix in ("", "_lo", "_hi"))
    return {"estimate": estimate, "lo": lo, "hi": hi, "se": None,
            "status": "copied_existing_table", "resolved": bool(
                lo is not None and hi is not None and
                (lo > 0 if positive else (lo > 0 or hi < 0))),
            "upper_unbounded": bool(hi is not None and math.isinf(hi) and hi > 0),
            "lower_unbounded": bool(lo is not None and math.isinf(lo) and lo < 0)}


def reference_rows(root, explicit=None):
    primary = root.parent / "noise_without_ppo"
    candidates = list(primary.glob("noise_without_ppo_table.csv"))
    candidates += list(primary.glob("figs/noise_without_ppo_table.csv"))
    candidates += list(primary.glob("runs/**/candidates/noise_without_ppo_table.csv"))
    if explicit:
        candidates = [Path(explicit)]
    chosen = candidates[0] if candidates else root / "figs/noise_actual_table.csv"
    method = "without_PPO" if candidates else "without_PPO_and_GAE"
    rows = read_csv(chosen)
    selected = {}
    for row in rows:
        row_method = row.get("method", row.get("run", row.get("estimator")))
        if row_method != method:
            continue
        raw_update = row.get("update", row.get("updates", row.get("update_count")))
        if raw_update in (None, ""):
            continue
        update = int(raw_update)
        if update not in UPDATES:
            continue
        relname = "relative_variance" if "relative_variance" in row else "R"
        relvar = imported_metric(row, relname)
        relvar["mean_resolved"] = str(row.get("mean_resolved", "")).lower() == "true"
        if row.get("relative_variance_lower_bound") not in (None, ""):
            relvar["lower_bound_when_mean_unresolved"] = float(row["relative_variance_lower_bound"])
        selected[update] = {"method": method, "source": str(chosen.resolve()), "row": row,
                            "metrics": {"R": relvar}}
    if set(selected) != set(UPDATES):
        raise ValueError("Reference table lacks required named-method rows: %s" % chosen)
    return selected


def walk_metrics(value, path=()):
    if isinstance(value, dict):
        if {"estimate", "lo", "hi"}.issubset(value):
            yield path, value
        else:
            for key, child in value.items():
                yield from walk_metrics(child, path + (str(key),))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from walk_metrics(child, path + (str(idx),))


def resolve_gate_metrics(value):
    """Keep the gate's estimates verbatim, adding the requested resolution flag."""
    if isinstance(value, dict):
        out = {key: resolve_gate_metrics(child) for key, child in value.items()}
        if {"estimate", "lo", "hi"}.issubset(out) and "resolved" not in out:
            lo, hi = out["lo"], out["hi"]
            out["resolved"] = bool(lo is not None and hi is not None and (lo > 0 or hi < 0))
        return out
    if isinstance(value, list):
        return [resolve_gate_metrics(child) for child in value]
    return value


def format_metric(value):
    if value["estimate"] is None:
        return "N/A (%s)" % value["status"]
    def fmt(number):
        return "%.6g" % number
    if value.get("lo") is None or value.get("hi") is None:
        return "%s (interval unavailable)" % fmt(value["estimate"])
    return "%s [%s, %s]" % (fmt(value["estimate"]), fmt(value["lo"]), fmt(value["hi"]))


def write_tables(output, analysis):
    dest = output / "batch_update_table.csv"
    if dest.exists():
        raise FileExistsError(str(dest))
    fields = ("method", "update", "section", "quantity", "metric", "estimate", "lo", "hi",
              "se", "resolved", "status", "valid_replicates", "total_replicates")
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in analysis["points"]:
            for path, value in walk_metrics(point):
                method = point["reference"]["method"] if path[0] == "reference" else "DPPO"
                # All matrix coordinates and gate ratios remain machine-readable.
                record = {"method": method, "update": point["update"],
                          "section": path[0], "quantity": "/".join(path[1:-1]),
                          "metric": path[-1]}
                record.update({key: value.get(key) for key in fields if key in value})
                writer.writerow(record)
    lines = ["# DPPO batch update and extra-mean turning", "",
             "Estimates with 90% delete-one-group jackknife intervals (10 groups, t9). "
             "Each checkpoint uses 500 batches of 80 episodes. Variances refer to one "
             "batch gradient; the batch average is the raw average of 20 gradients, "
             "not AdamW's actual displacement. Signed squared norms and all matrix "
             "entries are in the CSV. Unresolved direction-dependent quantities are N/A.", ""]
    for point in analysis["points"]:
        lines.extend(["## %d updates" % point["update"], "",
                      "| Quantity | Mean norm | Variance | Relative variance | Expected cosine |",
                      "|---|---:|---:|---:|---:|"])
        for row in point["quantities"]:
            met = row["metrics"]
            label = "Epoch %d" % row["epoch"] if row["epoch"] else "Batch average"
            lines.append("| %s | %s | %s | %s | %s |" % (label, *[
                format_metric(met[key]) for key in ("mean_norm", "variance", "R", "expected_cosine")]))
        lines.extend(["", "| Quantity | Cosine with epoch 1 | Component along epoch 1 | Relative variance against epoch 1 | Cross inner product with epoch 1 |",
                      "|---|---:|---:|---:|---:|"])
        for row in point["quantities"]:
            label = "Epoch %d" % row["epoch"] if row["epoch"] else "Batch average"
            lines.append("| %s | %s | %s | %s | %s |" % (label, *[
                format_metric(row["metrics"][key]) for key in ("cos_epoch1", "a", "R_on", "inner_epoch1")]))
        lines.extend(["", "| Summary | Estimate [90% interval] |", "|---|---:|"])
        for key, value in point["summaries"].items():
            lines.append("| %s | %s |" % (key, format_metric(value)))
        lines.extend(["", "| Frozen Adam metric | Estimate [90% interval] |", "|---|---:|"])
        for key in ("epoch1", "batch"):
            lines.append("| R(P %s) | %s |" % (key, format_metric(point["adam"][key]["R"])))
        for key in ("R_batch_over_epoch1", "relative_gain", "gain_numerator", "gain_denominator"):
            lines.append("| %s | %s |" % (key, format_metric(point["adam"][key])))
        lines.extend(["", "| Principal coordinate | Signed eigenvalue | Share of signed trace |",
                      "|---|---:|---:|"])
        turning = point["turning"]
        for idx, (eigenvalue, share) in enumerate(zip(turning["eigenvalues"], turning["eigenvalue_shares"]), 1):
            lines.append("| %d | %s | %s |" % (idx, format_metric(eigenvalue), format_metric(share)))
        for idx, value in enumerate(turning["cumulative_top_shares"], 1):
            lines.append("| Top %d cumulative | — | %s |" % (idx, format_metric(value)))
        lines.extend(["", "| Step | PC1 | PC2 | Extra mean resolved |", "|---|---:|---:|---|"])
        for idx, coords in enumerate(turning["coordinates"]):
            lines.append("| %d | %s | %s | %s |" % (idx + 5,
                format_metric(coords[0]), format_metric(coords[1]), turning["Gamma"][idx][idx]["resolved"]))
        ref = point["reference"]
        lines.extend(["", "Reference: **%s**, R = %s; measured at its own parameters. "
                      "Source: `%s`." % (ref["method"], format_metric(ref["metrics"]["R"]), ref["source"]), ""])
        lines.extend(["| G3 comparison | Ratio [90% interval] |", "|---|---:|"])
        for path, value in walk_metrics(point.get("gate_ratios", {})):
            lines.append("| %s | %s |" % (" / ".join(path), format_metric(value)))
        lines.append("")
    path = output / "batch_update_table.md"
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path,
                        help="Existing results/stage3/ppo_steps root")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    gates = read_json(args.input / "gates.json")
    for name in ("G1", "G2", "G3", "G4"):
        if not gates.get(name, {}).get("passed", False):
            raise ValueError("Required gate %s did not pass" % name)
    args.output.mkdir(parents=True, exist_ok=True)
    references = reference_rows(args.root, args.reference)
    with np.load(str(args.input / "moments.npz"), allow_pickle=False) as archive:
        points = [analyze_point(archive, update) for update in UPDATES]
    for point in points:
        point["reference"] = references[point["update"]]
        # The exact gate payload is retained, even when its format carries
        # additional validity detail beyond interval-bearing ratios.
        details = gates["G3"].get("details", [])
        if isinstance(details, list):
            point["gate_ratios"] = resolve_gate_metrics(
                [x for x in details if x.get("update") == point["update"]])
        else:
            point["gate_ratios"] = resolve_gate_metrics(details.get(str(point["update"]), {}))
    analysis = {"method": "DPPO", "K": 500, "groups": 10, "batches_per_group": 50,
                "interval": "90% delete-one-group jackknife, t9, joint ratios",
                "input": str(args.input.resolve()), "gates": gates, "points": points,
                "notes": ["Batch average is G/20, not the AdamW parameter displacement.",
                          "Epoch 1 is a first-order on-policy reference with minibatch normalization.",
                          "Cosines are not clipped; signed eigenvalues, including negative values, are retained.",
                          "Principal-coordinate signs align to the full sample; eigenvector identity stability is reported."]}
    save_json(args.output / "reference_rows.json", references)
    save_json(args.output / "analysis.json", analysis)
    write_tables(args.output, analysis)
    print("Wrote derived tables and analysis.json to %s" % args.output, flush=True)


if __name__ == "__main__":
    main()
