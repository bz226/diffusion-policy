#!/usr/bin/env python3
"""Generate the approved Stage 2 candidate artifacts after all runs terminate.

Input matrix: a JSON list of {condition, seed, run_id, manifest, ...}, containing
the 19 full runs (R0 and the six three-seed ablation groups), excluding probes.
Paths are absolute or relative to the matrix directory. Read trusted local
pickle files only. Importing this module neither loads results nor plots them.
"""

import argparse
import csv
from datetime import datetime
import io
import json
import math
from pathlib import Path
import pickle
import statistics


CONDITIONS = ["NC1", "NC2", "NC3", "NC4_lr1e-4", "NC4_lr1e-3", "NC4_lr3e-3"]
LABELS = {"baseline": "Stage 1 baseline", "R0": "R0 (default flags)", "NC1": "NC1", "NC2": "NC2", "NC3": "NC3", "NC4_lr1e-4": "NC4, LR 1e−4", "NC4_lr1e-3": "NC4, LR 1e−3", "NC4_lr3e-3": "NC4, LR 3e−3"}
COLORS = {"baseline": "#202020", "R0": "#202020", "NC1": "#0072B2", "NC2": "#D55E00", "NC3": "#009E73", "NC4_lr1e-4": "#56B4E9", "NC4_lr1e-3": "#CC79A7", "NC4_lr3e-3": "#E69F00"}
METRICS = [("kl_true_per_action", "Post-update KL estimate per action (log scale)"), ("logratio_p99", "99th percentile of |summed log-ratio|"), ("clamp_hit_frac", "Post-update coordinate clamp-hit fraction"), ("approx_kl", "DPPO approximate KL (last minibatch)"), ("clipfrac", "DPPO clip fraction (minibatch mean)")]
ENDPOINTS = [(0, 0), (70, 5040000), (130, 9360000)]
TERMINAL = {"complete", "failed", "timeout", "cancelled", "stopped"}


def number(value):
    if hasattr(value, "shape") and value.shape != ():
        raise ValueError("Expected scalar metric, found array")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected numeric metric: %r" % value)
    return value


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return value


def resolve(path, relative_to):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _read_result_rows(result_path, condition):
    with result_path.open("rb") as stream:
        raw_rows = pickle.load(stream)
    if not isinstance(raw_rows, list):
        raise ValueError("result.pkl is not a list: %s" % result_path)
    if len(raw_rows) > 140:
        raise ValueError("result.pkl exceeds the prescribed 140 iterations")
    rows = []
    for index, row in enumerate(raw_rows):
        if row.get("itr") != index or row.get("step") != (index - index // 10) * 80000:
            raise ValueError("Missing/misaligned iteration in %s at %d" % (result_path, index))
        expected_reward = "eval_episode_reward" if index % 10 == 0 else "train_episode_reward"
        if expected_reward not in row:
            raise ValueError("Missing %s at iteration %d" % (expected_reward, index))
        if condition != "baseline" and index % 10:
            missing = {key for key, _ in METRICS} - set(row)
            if missing:
                raise ValueError("Missing requested diagnostics at %s iteration %d: %r" % (result_path, index, missing))
        rows.append({key: number(value) for key, value in row.items()})
    return rows


def load_run(manifest_path, condition, seed, run_id=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    status = manifest["status"]
    if status not in TERMINAL:
        raise ValueError("%s is %s; final analysis waits for terminal runs" % (manifest_path, status))
    if manifest.get("seed") != seed:
        raise ValueError("Manifest seed disagrees with matrix: %s" % manifest_path)
    if run_id is not None and manifest.get("run_id") != run_id:
        raise ValueError("Manifest run_id disagrees with matrix: %s" % manifest_path)
    result_path = resolve(manifest["result_path"], manifest_path.parent)
    native_read_error = fallback_read_error = None
    effective_result_path = result_path
    result_source = "native"
    read_errors = (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, KeyError, AttributeError, ImportError)
    try:
        rows = _read_result_rows(result_path, condition)
    except read_errors as error:
        if status == "complete":
            raise ValueError("Completed native results cannot be read: %s (%s)" % (result_path, error)) from error
        native_read_error = type(error).__name__ + ": " + str(error)
        rows, effective_result_path, result_source = [], None, "unavailable"
        fallback = manifest.get("last_valid_result_path")
        if fallback:
            fallback_path = resolve(fallback, manifest_path.parent)
            try:
                if fallback_path == result_path:
                    raise ValueError("Declared fallback path equals unreadable native path")
                rows = _read_result_rows(fallback_path, condition)
                if manifest.get("last_valid_result_rows") is not None and len(rows) != manifest["last_valid_result_rows"]:
                    raise ValueError("Declared fallback row count disagrees with snapshot")
                effective_result_path, result_source = fallback_path, "declared_validated_prefix"
            except read_errors as fallback_error:
                fallback_read_error = type(fallback_error).__name__ + ": " + str(fallback_error)
                rows, effective_result_path = [], None
    if status == "complete" and (len(rows) != 140 or rows[-1]["step"] != 10080000):
        raise ValueError("Completed run lacks the prescribed 140 iterations: %s" % result_path)
    return {"condition": condition, "seed": seed, "run_id": manifest.get("run_id"), "manifest_path": str(manifest_path), "manifest": manifest, "status": status, "result_path": str(result_path), "effective_result_path": str(effective_result_path) if effective_result_path is not None else None, "result_source": result_source, "fallback_used": result_source == "declared_validated_prefix", "native_read_error": native_read_error, "fallback_read_error": fallback_read_error, "rows": rows}


def load_campaign(matrix_path, baseline_manifests, expected_r0_run_id="R0_rerun_seed0"):
    matrix_path = Path(matrix_path).resolve()
    matrix = json.loads(matrix_path.read_text())
    if not isinstance(matrix, list):
        raise ValueError("Run matrix must be a list")
    expected = {("R0", 0)} | {(condition, seed) for condition in CONDITIONS for seed in range(3)}
    observed = [(entry["condition"], entry["seed"]) for entry in matrix]
    if len(observed) != 19 or set(observed) != expected:
        raise ValueError("Matrix must contain exactly the 19 prescribed full runs")
    r0_entry = next(entry for entry in matrix if entry["condition"] == "R0")
    if r0_entry["run_id"] != expected_r0_run_id:
        raise ValueError("Matrix R0 run is not the explicitly selected diagnostics baseline")
    runs = [load_run(resolve(entry["manifest"], matrix_path.parent), entry["condition"], entry["seed"], entry["run_id"]) for entry in matrix]
    if len(baseline_manifests) != 3:
        raise ValueError("Exactly three Stage 1 manifests are required")
    baseline_runs = []
    for path in baseline_manifests:
        manifest = json.loads(Path(path).read_text())
        baseline_runs.append(load_run(path, "baseline", manifest["seed"]))
    if {run["seed"] for run in baseline_runs} != {0, 1, 2}:
        raise ValueError("Stage 1 must include seeds 0, 1 and 2 exactly once")
    return sorted(baseline_runs, key=lambda run: run["seed"]) + runs


def metric_series(run, metric, horizontal="step"):
    return {row[horizontal]: row[metric] for row in run["rows"] if metric in row}


def complete_case_stats(runs, metric, horizontal="step"):
    """Population mean/std only where all three requested seeds are finite."""
    if len(runs) != 3 or {run["seed"] for run in runs} != {0, 1, 2}:
        raise ValueError("Three-seed summaries require exactly seeds 0, 1, 2")
    series = [metric_series(run, metric, horizontal) for run in runs]
    grid = sorted(set().union(*(set(item) for item in series)))
    means, stds = [], []
    for x in grid:
        if all(x in item and math.isfinite(item[x]) for item in series):
            values = [item[x] for item in series]
            means.append(statistics.mean(values))
            stds.append(statistics.pstdev(values))
        else:
            means.append(math.nan)
            stds.append(math.nan)
    return grid, means, stds


def summarize_run(run):
    rows = run["rows"]
    by_iteration = {row["itr"]: row for row in rows}
    evaluations = [row["eval_episode_reward"] for row in rows if "eval_episode_reward" in row]
    scientific_nonfinite = [{"iteration": row["itr"], "metric": key, "value": value} for row in rows for key, value in row.items() if key != "time" and not math.isfinite(value)]
    nonfinite_failure = bool(run["manifest"].get("nonfinite_failure", False))
    initial = by_iteration.get(0, {}).get("eval_episode_reward")
    collapse = (bool(scientific_nonfinite) or nonfinite_failure)
    initial_valid = initial is not None and math.isfinite(initial)
    if initial_valid:
        collapse = collapse or any(math.isfinite(value) and value < initial * 0.5 for value in evaluations)
    elif not collapse:
        collapse = None
    kl = [row["kl_true_per_action"] for row in rows if "kl_true_per_action" in row]
    if not kl:
        median, median_status = None, ("unavailable_baseline" if run["condition"] == "baseline" else "no_training_measurements")
    elif any(math.isnan(value) for value in kl):
        median, median_status = None, "undefined_nan_present"
    else:
        median = statistics.median(kl)
        median_status = "defined" if not math.isnan(median) else "undefined_infinite_midpoint"
        if math.isnan(median):
            median = None
    minimum = min(evaluations) if evaluations and not any(math.isnan(value) for value in evaluations) else None
    summary = {"condition": run["condition"], "seed": run["seed"], "run_id": run["run_id"], "manifest": run["manifest_path"], "status": run["status"], "completed_iterations": len(rows), "minimum_eval_return": minimum, "minimum_eval_status": "defined" if minimum is not None else ("no_evaluations" if not evaluations else "undefined_nan_present"), "median_kl_true_per_action": median, "median_kl_status": median_status, "kl_measurement_count": len(kl), "nonfinite_kl_count": sum(not math.isfinite(value) for value in kl), "collapse": collapse, "collapse_observation_scope": "full_run" if run["status"] == "complete" else "saved_partial_run_and_failure_record", "nonfinite_failure": nonfinite_failure, "saved_nonfinite_count": len(scientific_nonfinite), "failing_iteration": run["manifest"].get("failing_iteration"), "failure_reason": run["manifest"].get("failure_reason"), "scientific_nonfinite_values": scientific_nonfinite}
    summary.update({key: run.get(key) for key in ("result_source", "fallback_used", "native_read_error", "fallback_read_error", "effective_result_path")})
    for iteration, step in ENDPOINTS:
        row = by_iteration.get(iteration)
        summary["eval_return_itr%d" % iteration] = row["eval_episode_reward"] if row is not None else None
        summary["eval_step_itr%d" % iteration] = row["step"] if row is not None else None
        if row is not None and row["step"] != step:
            raise ValueError("Endpoint step does not match approved endpoint")
    return summary


def _timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None


def resources(runs):
    records = []
    for run in runs:
        manifest = run["manifest"]
        query = manifest.get("hardware", {}).get("gpu_query", "")
        gpu_rows = list(csv.reader(io.StringIO(query))) if query else []
        uuid = gpu_rows[0][1].strip() if len(gpu_rows) == 1 and len(gpu_rows[0]) > 1 else None
        start = _timestamp(manifest.get("subprocess_start_utc"))
        end = _timestamp(manifest.get("subprocess_end_utc"))
        times = {"train": [row["time"] for row in run["rows"] if row["itr"] % 10 and "time" in row], "eval": [row["time"] for row in run["rows"] if row["itr"] % 10 == 0 and "time" in row]}
        records.append({"condition": run["condition"], "seed": run["seed"], "run_id": run["run_id"], "gpu_uuid": uuid, "gpu_query": query or None, "start_epoch_seconds": start, "end_epoch_seconds": end, "wall_seconds": manifest.get("subprocess_wall_seconds"), "train_iteration_mean_seconds": statistics.mean(times["train"]) if times["train"] else None, "eval_iteration_mean_seconds": statistics.mean(times["eval"]) if times["eval"] else None, "train_iteration_count": len(times["train"]), "eval_iteration_count": len(times["eval"])})
    for record in records:
        uuid, start, end = record["gpu_uuid"], record["start_epoch_seconds"], record["end_epoch_seconds"]
        if uuid is None or start is None or end is None:
            record["maximum_simultaneous_study_runs_on_gpu"] = None
            continue
        events = []
        for other in records:
            a, b = other["start_epoch_seconds"], other["end_epoch_seconds"]
            if other["gpu_uuid"] == uuid and a is not None and b is not None and max(start, a) < min(end, b):
                events.extend([(max(start, a), 1), (min(end, b), -1)])
        count = peak = 0
        for _, change in sorted(events):
            count += change
            peak = max(peak, count)
        record["maximum_simultaneous_study_runs_on_gpu"] = peak
    condition_times = {}
    for condition in ["baseline", "R0"] + CONDITIONS:
        members = [record for record in records if record["condition"] == condition]
        starts = [record["start_epoch_seconds"] for record in members]
        ends = [record["end_epoch_seconds"] for record in members]
        condition_times[condition] = max(ends) - min(starts) if all(value is not None for value in starts + ends) else None
    return {"runs": records, "condition_elapsed_seconds_first_start_to_last_end": condition_times, "sharing_scope": "Overlap among recorded study runs only; external GPU activity is not inferred"}


def _plot_group(axis, members, condition, metric, horizontal, np):
    color = COLORS[condition]
    logarithmic = metric == "kl_true_per_action"
    for run in members:
        series = metric_series(run, metric, horizontal)
        x = np.array(sorted(series), dtype=float)
        y = np.array([series[item] for item in x], dtype=float)
        valid = np.isfinite(y) & ((y > 0) if logarithmic else True)
        axis.plot(x, np.where(valid, y, np.nan), color=color, alpha=0.24 if len(members) == 3 else 1, linewidth=0.9 if len(members) == 3 else 1.8, label=LABELS[condition] if len(members) == 1 else None)
        if run["status"] != "complete" and valid.any():
            last = np.flatnonzero(valid)[-1]
            axis.scatter([x[last]], [y[last]], color=color, marker="x", s=32, zorder=6)
    if len(members) == 3:
        x, mean, std = complete_case_stats(members, metric, horizontal)
        x, mean, std = np.asarray(x), np.asarray(mean), np.asarray(std)
        valid = np.isfinite(mean) & ((mean > 0) if logarithmic else True)
        lower, upper = mean - std, mean + std
        band_valid = valid & np.isfinite(lower) & np.isfinite(upper)
        if logarithmic:
            band_valid &= lower > 0
        axis.plot(x, np.where(valid, mean, np.nan), color=color, linewidth=2, label=LABELS[condition])
        axis.fill_between(x, lower, upper, where=band_valid, color=color, alpha=0.14)


def generate(matrix_path, baseline_manifests, output_dir, expected_r0_run_id="R0_rerun_seed0", selection_provenance=None):
    """Write exactly three figures and one CSV, plus captions and provenance."""
    runs = load_campaign(matrix_path, baseline_manifests, expected_r0_run_id)
    summaries = [summarize_run(run) for run in runs]
    study = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir).resolve()
    if study not in output_dir.parents or output_dir.name != "candidates":
        raise ValueError("Outputs must use a new candidates/ directory within Stage 2")
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing candidate evidence")
    record_path = output_dir.parent / "analysis_record.json"
    if record_path.exists():
        raise FileExistsError("Refusing to overwrite existing analysis record")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {condition: [run for run in runs if run["condition"] == condition] for condition in ["baseline", "R0"] + CONDITIONS}
    output_dir.mkdir(parents=True)
    outputs = []
    for metric, filename, title in [("eval_episode_reward", "eval_return_vs_env_steps.png", "Evaluation return"), ("train_episode_reward", "train_return_vs_env_steps.png", "Training return (exploration noise)")]:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
        for axis, conditions, panel in zip(axes, [["baseline", "NC1", "NC2", "NC3"], ["baseline"] + CONDITIONS[3:]], ["Cumulative clipping removals", "One actor step per batch"]):
            for condition in conditions:
                _plot_group(axis, groups[condition], condition, metric, "step", np)
            axis.set(title=panel, xlabel="Training environment steps", ylabel="Undiscounted episode return")
            axis.grid(alpha=0.2)
            axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
            axis.legend(fontsize=9)
        figure.suptitle(title)
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)
        outputs.append(filename)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, METRICS):
        for condition in ["R0"] + CONDITIONS:
            _plot_group(axis, groups[condition], condition, metric, "itr", np)
        axis.set(title=title, xlabel="Iteration", ylabel=metric)
        if metric == "kl_true_per_action":
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="upper left", frameon=False)
    failures = ["%s seed %d: %s" % (run["condition"], run["seed"], run["status"]) for run in runs if run["condition"] != "baseline" and run["status"] != "complete"]
    nonpositive = sum(row.get("kl_true_per_action", 1) <= 0 for run in runs for row in run["rows"])
    notes = "Thin lines: individual seeds\nBold lines/bands: mean ± population std\nBands require all three seeds\nR0: one run, no uncertainty band\n×: last finite point of a failed/stopped run\nNonpositive KL observations omitted from log axis: %d" % nonpositive
    if failures:
        notes += "\n\nRun failures (see table):\n" + "\n".join(failures)
    legend_axis.text(0.02, 0.5, notes, va="top", fontsize=8, transform=legend_axis.transAxes)
    filename = "diagnostics_vs_iteration.png"
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    outputs.append(filename)

    table_fields = ["condition", "seed", "run_id", "manifest", "status", "completed_iterations", "result_source", "fallback_used", "eval_return_itr0", "eval_step_itr0", "eval_return_itr70", "eval_step_itr70", "eval_return_itr130", "eval_step_itr130", "minimum_eval_return", "minimum_eval_status", "median_kl_true_per_action", "median_kl_status", "kl_measurement_count", "nonfinite_kl_count", "collapse", "collapse_observation_scope", "saved_nonfinite_count", "nonfinite_failure", "failing_iteration"]
    with (output_dir / "condition_seed_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: "NA" if summary[key] is None else serializable(summary[key]) for key in table_fields})
    outputs.append("condition_seed_results.csv")
    completed = sum(run["status"] == "complete" for run in runs if run["condition"] != "baseline")
    collapse_count = sum(summary["collapse"] is True for summary in summaries if summary["condition"] != "baseline")
    observation = "%d of 19 Stage 2 full runs completed; %d runs met the prescribed observed-collapse criterion." % (completed, collapse_count)
    fallback_count = sum(run["fallback_used"] for run in runs)
    unavailable_count = sum(run["result_source"] == "unavailable" for run in runs)
    if fallback_count or unavailable_count:
        observation += " Unreadable failed-run native results required %d explicitly recorded prefix fallbacks; %d runs have no readable measurements." % (fallback_count, unavailable_count)
    captions = {
        outputs[0]: "Evaluation return versus cumulative training environment steps compares Stage 1 with cumulative removals NC1–NC3 (left), and with the three NC4 actor learning rates (right). Thin lines are individual seeds; bold lines and shading are mean ± population standard deviation over all three seeds at the same saved step, with no survivor averaging. " + observation + " Evaluations are undiscounted 1000-step returns; the final planned evaluation is at 9.36 million steps, not 10 million, and × marks the last finite point of a failed/stopped run.",
        outputs[1]: "Training return versus cumulative training environment steps uses the same conditions and uncertainty convention as the evaluation figure. These are on-rollout undiscounted episode returns with exploration noise, measured before each iteration's update, so their level is not directly interchangeable with evaluation return. " + observation + " Missing or nonfinite values are not imputed, and mean/bands appear only where all three requested seeds contribute finite observations.",
        outputs[2]: "Post-update diagnostics are shown against iteration for the single default-flags R0 and all ablation conditions; evaluation-only iterations have no diagnostics. The KL estimate uses 20,000 pairs, unclamped log-probabilities summed over 24 coordinates, and 10 × mean[expm1(d) − d]; the p99 uses |d| and clamp-hit fraction counts coordinates outside [−5, 2], while native approximate KL is the last minibatch and clip fraction the minibatch mean. Thin lines and mean ± population-standard-deviation bands use three independent seeds where all are present; R0 has no band, and nonpositive KL observations or band bounds are omitted from the log axis without replacement. " + observation + " This transition-based Monte Carlo diagnostic is not an exact KL of the final action distribution.",
        outputs[3]: "One row per condition and seed records evaluation at iterations 0, 70 and 130 (actual steps 0, 5.04 million and 9.36 million), minimum observed evaluation, median logged diagnostic KL, and collapse. Missing required iterations are NA rather than earlier substituted endpoints; medians include infinities and explicitly mark undefined NaN cases, and Stage 1 has no diagnostic KL. Collapse means an observed evaluation below half that run's initial evaluation or any recorded scientific NaN/inf, including a supervisor nonfinite failure; finite crashes remain a separate status and partial-run absence of collapse does not establish later stability. " + observation,
    }
    if selection_provenance:
        selection_note = " R0 and NC1 seed 1 use the authorized fresh replacements R0_recovery_seed0 and NC1_recovery_seed1, irrespective of their outcomes. The interrupted R0_rerun_seed0 and NC1_seed1 remain preserved in analysis_record.json as excluded provenance and are not extra seeds or spliced into the replacements."
        captions = {name: caption + selection_note for name, caption in captions.items()}
    with (output_dir / "captions.txt").open("x") as stream:
        stream.write("\n\n".join(name + "\n" + caption for name, caption in captions.items()) + "\n")
    record = {"status": "candidate_outputs_generated", "matrix": str(Path(matrix_path).resolve()), "candidate_directory": str(output_dir), "artifacts": outputs, "captions": "captions.txt", "summaries": summaries, "resources": resources(runs), "sources": [{"condition": run["condition"], "seed": run["seed"], "manifest": run["manifest_path"], "result_path": run["result_path"], "effective_result_path": run["effective_result_path"], "result_source": run["result_source"], "fallback_used": run["fallback_used"], "native_read_error": run["native_read_error"], "fallback_read_error": run["fallback_read_error"], "logdir": run["manifest"].get("logdir"), "checkpoints": run["manifest"].get("checkpoints", run["manifest"].get("checkpoints_found", [])), "status": run["status"], "last_diagnostics": run["manifest"].get("last_diagnostics"), "failure_reason": run["manifest"].get("failure_reason")} for run in runs], "curation_status": "unreviewed; explicit user approval required"}
    for source, run in zip(record["sources"], runs):
        source["run_id"] = run["run_id"]
    record["selected_r0_run_id"] = expected_r0_run_id
    if selection_provenance:
        record["recovery_selection"] = selection_provenance
    with record_path.open("x") as stream:
        json.dump(serializable(record), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--baseline-manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--r0-run-id", default="R0_rerun_seed0")
    args = parser.parse_args()
    record_path = generate(args.matrix, args.baseline_manifest, args.output_dir, args.r0_run_id)
    print(record_path)


if __name__ == "__main__":
    main()
