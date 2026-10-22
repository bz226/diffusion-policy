#!/usr/bin/env python3
"""Read-only Part C sufficient statistics and gates for batch-update analysis.

Python 3.8; no rollout, model construction, gradient computation, or GPU use.
The only torch operation is CPU deserialization of the saved optimizer state.
Raw-input gates finish before frozen-Adam metric statistics are evaluated.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_clip_timing import (
    G, N, P, POINTS, between_group_moments, jk, native_moments,
    read_csv, read_json, safe_json, save_json, validate_groups,
)


def matrix_moments(group_means):
    """Between-group mean Gram M and batch covariance Gram V, jointly deleted.

    V represents 50 times the sample covariance of the 10 group means. Thus
    c.T @ V @ c is the batch-level covariance trace of any combination c.
    """
    a = np.asarray(group_means, dtype=np.float64)
    n = len(a)
    m, m_leave, total_norm, leave_norm = between_group_moments(a)
    total = a.sum(axis=0, dtype=np.float64)
    tt = total @ total.T
    own = np.stack([x @ x.T for x in a])
    own_sum = own.sum(axis=0, dtype=np.float64)
    variance = N * (own_sum - tt / n) / (n - 1.0)
    variance_leave = []
    for index, x in enumerate(a):
        tx = total @ x.T
        remaining_tt = tt - tx - tx.T + own[index]
        variance_leave.append(N * (own_sum - own[index] - remaining_tt / (n - 1.0)) /
                              (n - 2.0))
    return {
        "M": m, "M_leave": m_leave,
        "V": variance, "V_leave": np.asarray(variance_leave),
        "total_norm_sq": total_norm * N ** 2,
        "leave_total_norm_sq": leave_norm * N ** 2,
        "total_gram": tt * N ** 2,
    }


def arithmetic_check():
    """Small deterministic arrays check formulas, with no random draws."""
    array = np.arange(5 * 4 * 7, dtype=np.float64).reshape(5, 4, 7)
    array = np.sin(array / 11.0) + np.cos(array / 7.0)
    result = matrix_moments(array)
    mean_expected = sum(array[i] @ array[j].T for i in range(5)
                        for j in range(5) if i != j) / 20.0
    centered = array - array.mean(axis=0)
    variance_expected = N * sum(x @ x.T for x in centered) / 4.0
    differences = {
        "mean_gram_max_abs": float(np.max(np.abs(result["M"] - mean_expected))),
        "covariance_gram_max_abs": float(np.max(np.abs(result["V"] - variance_expected))),
    }
    leave_mean_error = 0.0
    leave_variance_error = 0.0
    for omitted in range(5):
        reduced = np.delete(array, omitted, axis=0)
        expected = sum(reduced[i] @ reduced[j].T for i in range(4)
                       for j in range(4) if i != j) / 12.0
        center = reduced - reduced.mean(axis=0)
        expected_v = N * sum(x @ x.T for x in center) / 3.0
        leave_mean_error = max(leave_mean_error,
                               float(np.max(np.abs(expected - result["M_leave"][omitted]))))
        leave_variance_error = max(leave_variance_error,
                                   float(np.max(np.abs(expected_v - result["V_leave"][omitted]))))
    differences.update(leave_mean_max_abs=leave_mean_error,
                       leave_covariance_max_abs=leave_variance_error)
    differences["passed"] = all(value < 1e-11 for value in differences.values())
    return differences


def load_vectors(groups):
    vectors = np.empty((G, 20, P), dtype=np.float64)
    for index, (directory, unused_meta) in enumerate(groups):
        for step in range(1, 21):
            path = directory / ("sums_step%02d.npy" % step)
            value = np.load(str(path), allow_pickle=False)
            if value.shape != (P,) or value.dtype != np.float64 or not np.all(np.isfinite(value)):
                raise ValueError("Invalid float64 step sum: %s" % path)
            vectors[index, step - 1] = value / N
    return vectors


def load_point(root, itr, update):
    groups = validate_groups(root, itr, update)
    specs = groups[0][1]["parameter_specs"]
    source = groups[0][1]["source_state"]
    q = np.empty((G, 21), dtype=np.float64)
    inventory = []
    vectors = load_vectors(groups)
    consistency = []
    for index, (directory, meta) in enumerate(groups):
        if meta["parameter_specs"] != specs or meta["source_state"] != source:
            raise ValueError("Mixed parameter ordering or source state: %s" % directory)
        rows = read_csv(directory / "batches.csv")
        indexed = {(int(row["batch"]), int(row["step"])): row for row in rows}
        if len(rows) != N * 20 or set(indexed) != {(b, s) for b in range(N) for s in range(1, 21)}:
            raise ValueError("Expected unique 50 by 20 batch-step scalars: %s" % directory)
        for step in range(1, 21):
            values = np.asarray([float(indexed[(b, step)]["grad_sq"]) for b in range(N)])
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Invalid stored gradient norm: %s" % directory)
            q[index, step - 1] = values.sum(dtype=np.float64)
        totals = read_csv(directory / "batch_totals.csv")
        if len(totals) != N or {int(row["batch"]) for row in totals} != set(range(N)):
            raise ValueError("Expected 50 unique batch totals: %s" % directory)
        total_norms = np.asarray([float(row["G_sq"]) for row in totals], dtype=np.float64)
        if not np.all(np.isfinite(total_norms)) or np.any(total_norms < 0):
            raise ValueError("Invalid stored G squared norm: %s" % directory)
        q[index, 20] = total_norms.sum(dtype=np.float64)
        summed_steps = vectors[index].sum(axis=0, dtype=np.float64) * N
        stored_path = directory / "sums_G.npy"
        fallback = not stored_path.exists()
        if fallback:
            total = summed_steps
            error = 0.0
        else:
            total = np.load(str(stored_path), allow_pickle=False)
            if total.shape != (P,) or total.dtype != np.float64 or not np.all(np.isfinite(total)):
                raise ValueError("Invalid stored sum G: %s" % stored_path)
            denominator = float(np.linalg.norm(total))
            difference = float(np.linalg.norm(total - summed_steps))
            error = difference / denominator if denominator > 0 else (0.0 if difference == 0 else float("inf"))
        consistency.append({"update": update, "group": directory.name,
                            "relative_l2": error, "tolerance": 1e-8,
                            "passed": bool(error <= 1e-8),
                            "fallback": "G built from step sums" if fallback else None})
        inventory.append({"directory": str(directory), "seed": meta["seed"], "K": N,
                          "source_commit": meta["commit"], "meta": str(directory / "meta.json"),
                          "scalars": str(directory / "batches.csv"),
                          "totals": str(directory / "batch_totals.csv"),
                          "sum_G": str(stored_path) if not fallback else None,
                          "group_sums": [str(directory / ("sums_step%02d.npy" % s)) for s in range(1, 21)]})
    return vectors, q, groups, consistency, inventory


def interval_overlap(first, second):
    return (first["lo"] is not None and second["lo"] is not None and
            max(first["lo"], second["lo"]) <= min(first["hi"], second["hi"]))


def estimator_checks(vectors, q, moments, update):
    """G3 compares signed squared signals and covariance traces for 20 steps/G."""
    total_means = vectors.sum(axis=1, dtype=np.float64)
    total = total_means.sum(axis=0, dtype=np.float64)
    g_norm = float(np.dot(total, total)) * N ** 2
    g_leave_norm = np.asarray([float(np.dot(total - row, total - row)) * N ** 2
                              for row in total_means])
    total_norms = np.r_[moments["total_norm_sq"], g_norm]
    leave_norms = np.column_stack((moments["leave_total_norm_sq"], g_leave_norm))
    native_signal, native_var = native_moments(total_norms, q.sum(axis=0), G * N)
    leave_signal, leave_var = native_moments(leave_norms, q.sum(axis=0)[None, :] - q, (G - 1) * N)
    group_signal = np.r_[np.diag(moments["M"]), moments["M"].sum()]
    group_var = np.r_[np.diag(moments["V"]), moments["V"].sum()]
    group_signal_leave = np.column_stack((np.diagonal(moments["M_leave"], axis1=1, axis2=2),
                                         moments["M_leave"].sum(axis=(1, 2))))
    group_var_leave = np.column_stack((np.diagonal(moments["V_leave"], axis1=1, axis2=2),
                                      moments["V_leave"].sum(axis=(1, 2))))
    rows = []
    for index in range(21):
        row = {"update": update, "quantity": "g_%d" % (index + 1) if index < 20 else "G"}
        for name, value_a, reps_a, value_b, reps_b in (
                ("signal_sq", group_signal[index], group_signal_leave[:, index],
                 native_signal[index], leave_signal[:, index]),
                ("variance", group_var[index], group_var_leave[:, index],
                 native_var[index], leave_var[:, index])):
            group_metric, batch_metric = jk(value_a, reps_a), jk(value_b, reps_b)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = jk(np.divide(value_a, value_b), np.divide(reps_a, reps_b))
            row[name] = {"group": group_metric, "batch": batch_metric,
                         "group_over_batch": ratio,
                         "passed": bool(interval_overlap(group_metric, batch_metric))}
        row["passed"] = row["signal_sq"]["passed"] and row["variance"]["passed"]
        rows.append(row)
    return rows


def adam_diagonal(checkpoint, specs, itr):
    """Frozen bias-corrected Adam P in the stored actor_ft parameter order."""
    import torch
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if int(state["itr"]) != itr:
        raise ValueError("Checkpoint iteration does not match metadata")
    model = state["model"]
    optimizer = state["actor_optimizer"]
    parameters = []
    for group in optimizer["param_groups"]:
        for identifier in group["params"]:
            parameters.append((identifier, group))
    if len(parameters) != len(specs) or len(optimizer["state"]) != len(specs):
        raise ValueError("Optimizer parameter count differs from canonical specs")
    model_names = [name[len("actor_ft."):] for name in model if name.startswith("actor_ft.")]
    if model_names != [spec["name"] for spec in specs]:
        raise ValueError("Model actor_ft order differs from stored gradient parameter order")
    diagonal = []
    details = []
    for spec, (identifier, group) in zip(specs, parameters):
        entry = optimizer["state"][identifier]
        variance = entry["exp_avg_sq"].detach().cpu().numpy().astype(np.float64)
        shape = tuple(spec["shape"])
        if tuple(model["actor_ft." + spec["name"]].shape) != shape or variance.shape != shape:
            raise ValueError("Actor/optimizer shape mismatch: %s" % spec["name"])
        step = float(entry["step"].item()) if hasattr(entry["step"], "item") else float(entry["step"])
        beta2, eps = float(group["betas"][1]), float(group["eps"])
        if step <= 0 or not 0 < beta2 < 1 or eps <= 0 or group.get("amsgrad", False):
            raise ValueError("Unsupported or invalid AdamW metric state")
        corrected = variance / (1.0 - beta2 ** step)
        scale = 1.0 / (np.sqrt(corrected) + eps)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Invalid frozen Adam diagonal")
        diagonal.append(scale.reshape(-1))
        details.append({"name": spec["name"], "shape": list(shape), "optimizer_id": identifier,
                        "step": step, "beta2": beta2, "epsilon": eps})
    combined = np.concatenate(diagonal)
    if combined.shape != (P,):
        raise ValueError("Flattened Adam diagonal has unexpected size")
    return combined, {"checkpoint": str(checkpoint), "iteration": itr,
                      "definition": "1 / (sqrt(exp_avg_sq / (1 - beta2**step)) + epsilon)",
                      "parameter_order": details, "minimum": float(combined.min()),
                      "maximum": float(combined.max())}


def run(root, output):
    if not root.is_absolute() or not output.is_absolute():
        raise ValueError("Use absolute input and output paths")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("gates.json", "moments.npz", "inventory.json"):
        if (output / name).exists():
            raise FileExistsError("Refusing to overwrite %s" % (output / name))
    cache_path = root / "clip_timing/runs/analysis_20261019/candidates/grouped_moments.npz"
    gates = {name: {"passed": True, "details": []} for name in ("G1", "G2", "G3", "G4")}
    gates["arithmetic_check"] = arithmetic_check()
    inventory = {"clip_timing_cache": str(cache_path), "points": [], "adam_metrics": []}
    archive = {}
    if not gates["arithmetic_check"]["passed"]:
        gates["passed"] = False
        save_json(output / "gates.json", gates)
        return 2
    with np.load(str(cache_path), allow_pickle=False) as cache:
        for itr, update in POINTS:
            print("Raw gates: DPPO after %d updates" % update, flush=True)
            try:
                vectors, q, groups, consistency, items = load_point(root, itr, update)
                moments = matrix_moments(vectors)
            except Exception as error:
                gates["G1"]["passed"] = False
                gates["G1"]["details"].append({"update": update, "error": repr(error)})
                continue
            prefix = "u%d_" % update
            archive.update({prefix + key: moments[key] for key in ("M", "M_leave", "V", "V_leave")})
            inventory["points"].append({"update": update, "checkpoint_itr": itr, "groups": items})
            gates["G1"]["details"].append({"update": update, "groups": G, "K": G * N,
                                            "step_count": 20, "parameter_count": P,
                                            "all_inputs_present": True})
            gates["G2"]["details"].extend(consistency)
            gates["G2"]["passed"] &= all(row["passed"] for row in consistency)
            checks = estimator_checks(vectors, q, moments, update)
            gates["G3"]["details"].extend(checks)
            gates["G3"]["passed"] &= all(row["passed"] for row in checks)
            for name in ("M", "M_leave"):
                expected = cache[prefix + name]
                actual = moments[name]
                difference = np.abs(expected - actual)
                passed = bool(np.allclose(expected, actual, rtol=1e-12, atol=1e-20))
                gates["G4"]["details"].append({"update": update, "quantity": name,
                    "passed": passed, "bitwise_equal": bool(np.array_equal(expected, actual)),
                    "max_absolute_difference": float(difference.max()),
                    "max_relative_difference": float(np.max(difference / np.maximum(np.abs(expected), 1e-300))),
                    "rtol": 1e-12, "atol": 1e-20})
                gates["G4"]["passed"] &= passed
            del vectors
    gates["G3_definition"] = "Signed signal-squared and covariance-trace 90% t9 delete-one-group intervals must overlap, for each step and G; ratios use joint deletion."
    gates["G5"] = {"status": "performed_by_driver_after_all_outputs"}
    gates["passed"] = all(gates[name]["passed"] for name in ("G1", "G2", "G3", "G4"))
    if gates["passed"]:
        try:
            for itr, update in POINTS:
                print("Frozen Adam metric: DPPO after %d updates" % update, flush=True)
                groups = validate_groups(root, itr, update)
                meta = groups[0][1]
                diagonal, details = adam_diagonal(Path(meta["source_state"]), meta["parameter_specs"], itr)
                inventory["adam_metrics"].append(details)
                vectors = load_vectors(groups)
                # sqrt(P) gives <E X, P E Y>; P gives norms/covariances of P X.
                vectors *= np.sqrt(diagonal)[None, None, :]
                cross = matrix_moments(vectors)
                prefix = "u%d_" % update
                archive[prefix + "M_cross"] = cross["M"]
                archive[prefix + "M_cross_leave"] = cross["M_leave"]
                vectors *= np.sqrt(diagonal)[None, None, :]
                weighted = matrix_moments(vectors)
                for name in ("M", "M_leave", "V", "V_leave"):
                    key = name.replace("_leave", "_P_leave") if name.endswith("_leave") else name + "_P"
                    archive[prefix + key] = weighted[name]
                del vectors
        except Exception as error:
            gates["G1"]["passed"] = False
            gates["G1"]["details"].append({"adam_state_error": repr(error)})
            gates["passed"] = False
    np.savez(str(output / "moments.npz"), **archive)
    save_json(output / "inventory.json", inventory)
    save_json(output / "gates.json", gates)
    print("Batch-update prerequisites: %s" % ("PASS" if gates["passed"] else "FAIL"), flush=True)
    return 0 if gates["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return run(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
