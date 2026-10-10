#!/usr/bin/env python
"""CPU-only Stage-3 finalizer. Reads native evidence; never trains or curates."""

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["E0", "E1", "E2", "E2_beta0", "E2_nobase", "SGD03", "SGD1", "SGD3", "SGD10", "PROJ3", "PROJ10", "BATCH4"]
PHASES = [("Phase 2: AdamW estimator ladder", CONDITIONS[:5]),
          ("Phase 3: plain SGD", CONDITIONS[5:9] + ["E2", "E2_beta0"]),
          ("Phase 4: projection / batch size", ["SGD3", "SGD10", "PROJ3", "PROJ10", "BATCH4"])]
COLORS = dict(zip(CONDITIONS, ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628", "#8c564b",
                              "#17becf", "#e41a1c", "#9467bd", "#f781bf", "#bcbd22", "#000000"]))
VARIANTS = ["nc4", "mc_critic", "mc_loo", "theory_loo", "theory_nobase"]
METRICS = ["eval_itr0", "eval_itr130", "train_last10", "J_disc_last10",
           "ckpt_train_return", "ckpt_eval_return", "ckpt_train_J_disc", "ckpt_eval_J_disc",
           "delta_train_J_disc", "delta_eval_J_disc", "median_kl", "theta_dist_final",
           "median_B_env", "zero_update_fraction", "x0_saturation_mean", "x0_saturation_last",
           "action_oor_mean", "projection_fraction", "seconds_per_training_iteration", "actor_lr"]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(child) for child in value]
    value = scalar(value)
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(data), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Preserve an explicit, readable empty-result artifact after early failures.
        path.write_text("run,condition,seed,itr,step\n")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(safe(value)) if isinstance(value, (list, tuple, dict)) else safe(value)
                             for key, value in row.items()})


def numeric_bad(value, prefix=""):
    if isinstance(value, dict):
        return [entry for key, child in value.items() for entry in numeric_bad(child, prefix + "." + str(key))]
    if isinstance(value, (list, tuple)):
        return [entry for index, child in enumerate(value) for entry in numeric_bad(child, prefix + "[" + str(index) + "]")]
    value = scalar(value)
    return [{"key": prefix, "value": repr(value)}] if isinstance(value, float) and not math.isfinite(value) else []


def mean_sd(values, required=3):
    valid = [float(value) for value in values if finite(value)]
    complete = len(valid) == required and len(values) == required
    return {"mean": float(np.mean(valid)) if complete else None,
            "sd": float(np.std(valid, ddof=1)) if complete and required > 1 else None,
            "n_available": len(valid), "n_required": required,
            "per_seed": [float(value) if finite(value) else None for value in values]}


def window_noise(rows):
    """Nonoverlapping iteration windows [0,9], [10,19], ...; no partial windows."""
    indexed = {int(row["itr"]): row for row in rows}
    windows = []
    for start in range(0, 140, 10):
        expected = list(range(start + 1, start + 10))
        selected = [indexed[i] for i in expected if i in indexed]
        complete = len(selected) == 9 and all(finite(r.get("split_dot")) and finite(r.get("split_diff_sq")) for r in selected)
        dot = float(sum(r["split_dot"] for r in selected)) if complete else None
        difference = float(sum(r["split_diff_sq"] for r in selected)) if complete else None
        windows.append({"itr_start": start, "itr": start + 9,
                        "step": indexed.get(start + 9, {}).get("step"),
                        "n_training_iterations": len(selected), "complete": complete,
                        "sum_split_dot": dot, "sum_split_diff_sq": difference,
                        "B_env": 10 * difference / dot if complete and dot > 0 else None,
                        "reason": None if complete and dot > 0 else "nonpositive_split_dot" if complete else "incomplete_window"})
    return windows


def read_rows(path):
    with Path(path).open("rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list):
        raise ValueError("result.pkl is not a list: " + str(path))
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("itr") != index:
            raise ValueError("non-contiguous iteration sequence: " + str(path))
    return [{key: scalar(value) for key, value in row.items()} for row in rows]


def load_runs(root, campaign, allow_partial):
    runs = []
    for spec in campaign.get("runs", []):
        if spec["mode"] != "train":
            continue
        manifest_path = root / "runs" / spec["run_id"] / "manifest.json"
        manifest = read_json(manifest_path, {})
        status = manifest.get("status", "planned")
        if status not in ("complete", "failed", "skipped") and not allow_partial:
            raise ValueError("run is not terminal: " + spec["run_id"])
        path = Path(spec["logdir"]) / "result.pkl"
        error, rows = None, []
        if path.exists():
            try:
                rows = read_rows(path)
            except (OSError, EOFError, pickle.UnpicklingError, ValueError) as exc:
                if status == "complete":
                    raise
                error = str(exc)
        n_steps = 2000 if spec["condition"] == "BATCH4" else 500
        for row in rows:
            expected = (row["itr"] - row["itr"] // 10) * n_steps * 40 * 4
            if row.get("step") != expected:
                raise ValueError("step grid differs from the prescribed rollout: " + spec["run_id"])
        checkpoints = [str(Path(spec["logdir"]) / "checkpoint" / ("state_%d.pt" % itr)) for itr in (0, 35, 70, 105, 139)]
        if status == "complete" and (len(rows) != 140 or not all(Path(p).is_file() for p in checkpoints)):
            raise ValueError("claimed complete run lacks rows/checkpoints: " + spec["run_id"])
        evaluation_id = "ckpt_" + spec["run_id"] + "_seed5000"
        evaluation_manifest = read_json(root / "runs" / evaluation_id / "manifest.json", {})
        evaluation = read_json(root / evaluation_id / "ckpt_eval.json", {})
        if evaluation_manifest.get("status") != "complete":
            evaluation = {}
        runs.append({"spec": spec, "manifest": manifest, "manifest_path": str(manifest_path),
                     "status": status, "rows": rows, "windows": window_noise(rows),
                     "result_path": str(path), "read_error": error, "checkpoints": checkpoints,
                     "checkpoint_eval": evaluation, "checkpoint_eval_status": evaluation_manifest.get("status", "not_available"),
                     "checkpoint_eval_path": str(root / evaluation_id / "ckpt_eval.json")})
    return runs


def reference_runs(root):
    references = {"Stage 1 DPPO": [], "Stage 2 NC4 1e-4": []}
    stage1 = root.parent / "repro_halfcheetah"
    for seed, run_id in enumerate(("seed0_handoff", "seed1", "seed2")):
        manifest = read_json(stage1 / "runs" / run_id / "manifest.json")
        if not manifest or manifest.get("status") != "complete":
            raise ValueError("missing completed Stage-1 reference")
        references["Stage 1 DPPO"].append({"seed": seed, "rows": read_rows(manifest["result_path"]),
                                          "path": manifest["result_path"]})
        path = root.parent / "stage2_noclip/runs" / ("NC4_lr1e-4_seed%d" % seed) / "native/result.pkl"
        references["Stage 2 NC4 1e-4"].append({"seed": seed, "rows": read_rows(path), "path": str(path)})
    return references


def summarize(run, theta0):
    rows, spec = run["rows"], run["spec"]
    train = [row for row in rows if "train_episode_reward" in row]
    indexed = {row["itr"]: row for row in rows}
    complete = run["status"] == "complete" and len(rows) == 140
    ev0 = indexed.get(0, {}).get("eval_episode_reward")
    evaluations = [row for row in rows if finite(row.get("eval_episode_reward"))]
    collapse_rows = [row for row in evaluations if finite(ev0) and row["eval_episode_reward"] < 0.5 * ev0]
    bad = numeric_bad(rows)
    observation = run["manifest"].get("last_observation", {})
    failure_text = str(run["manifest"].get("failure_reason", ""))
    nonfinite = bool(bad or any(observation.get(key) for key in ("nonfinite", "nonfinite_log_line", "nonfinite_exception"))
                     or any(word in failure_text.lower() for word in ("nonfinite", "non-finite", "floatingpointerror")))
    last10 = train[-10:] if complete and len(train) >= 10 else []
    def avg(key, values=train):
        xs = [row.get(key) for row in values]
        return float(np.mean(xs)) if xs and all(finite(x) for x in xs) else None
    def med(key):
        xs = [row[key] for row in train if finite(row.get(key))]
        return float(np.median(xs)) if xs else None
    windows = [row["B_env"] for row in run["windows"] if finite(row["B_env"])]
    zero_streak, longest_zero_streak = 0, 0
    for row in train:
        zero_streak = zero_streak + 1 if row.get("update_norm") == 0 else 0
        longest_zero_streak = max(longest_zero_streak, zero_streak)
    zero_rows = [row for row in train if row.get("update_norm") == 0]
    summary = {"run": spec["run_id"], "condition": spec["condition"], "seed": spec["seed"],
               "status": run["status"], "completed_iterations": len(rows),
               "summary_scope": "full run" if complete else "observed prefix only; final metrics unavailable",
               "eval_itr0": ev0 if finite(ev0) else None,
               "eval_itr130": indexed.get(130, {}).get("eval_episode_reward"),
               "eval_itr130_env_steps": indexed.get(130, {}).get("step"),
               "minimum_eval_return": min((row["eval_episode_reward"] for row in evaluations), default=None),
               "train_last10": avg("train_episode_reward", last10), "J_disc_last10": avg("J_disc_train", last10),
               "median_kl": med("kl_true_per_action") if complete else None,
               "median_kl_observed_prefix": med("kl_true_per_action"),
               "theta_dist_final": indexed.get(139, {}).get("theta_dist") if complete else None,
               "median_B_env": float(np.median(windows)) if complete and windows else None,
               "defined_noise_windows": len(windows),
               "nonpositive_noise_windows": sum(row["reason"] == "nonpositive_split_dot" for row in run["windows"]),
               "zero_update_fraction": float(np.mean([row["update_norm"] == 0 for row in train])) if complete else None,
               "longest_zero_update_streak": longest_zero_streak,
               "x0_saturation_during_zero_updates": avg("x0_sat_frac_mean", zero_rows),
               "x0_saturation_mean": avg("x0_sat_frac_mean") if complete else None,
               "x0_saturation_last": avg("x0_sat_frac_last") if complete else None,
               "action_oor_mean": avg("action_oor_frac") if complete else None,
               "policy_action_oor_mean": avg("policy_action_oor_frac") if complete else None,
               "projection_fraction": float(np.mean([bool(row["proj_active"]) for row in train])) if complete else None,
               "seconds_per_training_iteration": avg("time") if complete else None,
               "actor_lr": train[0].get("actor_lr") if train else None,
               "wall_seconds": run["manifest"].get("wall_seconds"),
               "started_utc": run["manifest"].get("started_utc"), "ended_utc": run["manifest"].get("ended_utc"),
               "collapse": True if collapse_rows or nonfinite else False if rows else None, "nonfinite": nonfinite,
               "first_reward_collapse_itr": collapse_rows[0]["itr"] if collapse_rows else None,
               "nonfinite_evidence": bad, "failure_reason": run["manifest"].get("failure_reason"),
               "result_read_error": run["read_error"],
               "checkpoint_eval_status": run["checkpoint_eval_status"],
               "logdir": spec["logdir"], "result_path": run["result_path"], "checkpoints": run["checkpoints"],
               "checkpoint_eval_path": run["checkpoint_eval_path"], "manifest_path": run["manifest_path"]}
    for sampler in ("train", "eval"):
        result = run["checkpoint_eval"].get("samplers", {}).get(sampler, {})
        zero = theta0.get("samplers", {}).get(sampler, {})
        j = result.get("J_disc", {}).get("mean")
        j0 = zero.get("J_disc", {}).get("mean")
        summary.update({"ckpt_" + sampler + "_return": result.get("undiscounted_return", {}).get("mean"),
                        "ckpt_" + sampler + "_return_se": result.get("undiscounted_return", {}).get("se"),
                        "ckpt_" + sampler + "_J_disc": j,
                        "ckpt_" + sampler + "_J_disc_se": result.get("J_disc", {}).get("se"),
                        "delta_" + sampler + "_J_disc": j - j0 if finite(j) and finite(j0) else None,
                        "ckpt_" + sampler + "_x0_saturation": result.get("x0_sat_frac"),
                        "ckpt_" + sampler + "_action_oor": result.get("action_oor_frac")})
    return summary


def aggregate(summaries):
    grouped = {}
    for condition in CONDITIONS:
        selected = {row["seed"]: row for row in summaries if row["condition"] == condition}
        grouped[condition] = {"condition": condition,
            "statuses": [selected.get(seed, {}).get("status", "not_launched") for seed in range(3)],
            "collapse_per_seed": [selected.get(seed, {}).get("collapse") for seed in range(3)],
            "collapse_count": sum(bool(selected.get(seed, {}).get("collapse")) for seed in range(3)),
            "metrics": {key: mean_sd([selected.get(seed, {}).get(key) for seed in range(3)]) for key in METRICS}}
        times = [(row.get("started_utc"), row.get("ended_utc")) for row in selected.values()]
        grouped[condition]["wall_span_seconds"] = ((max(dt.datetime.fromisoformat(end) for _, end in times)
              - min(dt.datetime.fromisoformat(start) for start, _ in times)).total_seconds()
              if len(times) == 3 and all(start and end for start, end in times) else None)
    return grouped


def comparison(grouped, first, second, metric):
    a, b = grouped[first]["metrics"][metric], grouped[second]["metrics"][metric]
    if a["mean"] is None or b["mean"] is None:
        return {"difference": None, "threshold": None, "interpretation": "unresolved: incomplete comparable three-seed data"}
    difference = b["mean"] - a["mean"]
    threshold = 2 * math.sqrt((a["sd"] ** 2 + b["sd"] ** 2) / 2)
    return {"difference": difference, "threshold": threshold,
            "interpretation": "unresolved under the prescribed two-pooled-SD rule" if abs(difference) < threshold else
            "exceeds the prescribed two-pooled-SD threshold; no additional significance test"}


def series(run, metric, horizontal):
    source = run["windows"] if metric == "B_env" else run["rows"]
    return {row[horizontal]: float(row[metric]) for row in source
            if finite(row.get(horizontal)) and finite(row.get(metric))}


def plot_group(ax, selected, metric, horizontal, label, color, logscale=False, linestyle="-"):
    maps = [series(run, metric, horizontal) for run in selected]
    for values in maps:
        x = sorted(values)
        y = np.array([values[key] for key in x])
        if logscale:
            y = np.where(y > 0, y, np.nan)
        ax.plot(np.array(x) / (1e6 if horizontal == "step" else 1), y, color=color, alpha=.22, linewidth=.65, linestyle=linestyle)
    if len(selected) != 3:
        return
    common = sorted(set.intersection(*(set(values) for values in maps)))
    if not common:
        return
    ys = np.array([[values[x] for x in common] for values in maps])
    mean, sd = ys.mean(axis=0), ys.std(axis=0, ddof=1)
    x = np.array(common) / (1e6 if horizontal == "step" else 1)
    lower, upper = mean - sd, mean + sd
    if logscale:
        mean = np.where(mean > 0, mean, np.nan)
        valid = lower > 0
        lower, upper = np.where(valid, lower, np.nan), np.where(valid, upper, np.nan)
    ax.plot(x, mean, color=color, label=label, linewidth=1.5, linestyle=linestyle)
    ax.fill_between(x, lower, upper, color=color, alpha=.10)


def figures(root, runs, references, anchors):
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("eval_episode_reward", "(a) Eval return", False),
              ("train_episode_reward", "(b) Train return", False),
              ("J_disc_train", "(c) Discounted training objective", False),
              ("kl_true_per_action", "(d) Sampled transition KL", True),
              ("theta_dist", "(e) Distance from pretrained actor", False),
              ("B_env", "(f) Window noise scale B_env", True),
              ("grad_cos_prev", "(g) Consecutive gradient cosine", False),
              ("x0_sat_frac_mean", "(h) Saturation / action violation", False)]
    omissions = {"nonpositive_kl_points": 0, "nonpositive_B_env_points": 0,
                 "undefined_noise_windows": 0, "nonpositive_lower_sd_bands": 0}
    for run in runs:
        omissions["nonpositive_kl_points"] += sum(finite(r.get("kl_true_per_action")) and r["kl_true_per_action"] <= 0 for r in run["rows"])
        omissions["nonpositive_B_env_points"] += sum(finite(r.get("B_env")) and r["B_env"] <= 0 for r in run["windows"])
        omissions["undefined_noise_windows"] += sum(r["reason"] == "nonpositive_split_dot" for r in run["windows"])
    for condition in CONDITIONS:
        selected = [run for run in runs if run["spec"]["condition"] == condition]
        if len(selected) == 3:
            for metric in ("kl_true_per_action", "B_env"):
                maps = [series(run, metric, "itr") for run in selected]
                for x in set.intersection(*(set(m) for m in maps)):
                    ys = np.array([m[x] for m in maps])
                    omissions["nonpositive_lower_sd_bands"] += int(ys.mean() - ys.std(ddof=1) <= 0)
    figdir = root / "figs"
    figdir.mkdir(exist_ok=True)
    def draw(rows, filename):
        fig, axes = plt.subplots(len(rows), 8, figsize=(32, 4.1 * len(rows)), squeeze=False)
        for row_index, (title, conditions, horizontal) in enumerate(rows):
            for column, (metric, panel_title, logscale) in enumerate(panels):
                ax = axes[row_index, column]
                for condition in conditions:
                    selected = sorted([run for run in runs if run["spec"]["condition"] == condition], key=lambda r: r["spec"]["seed"])
                    if not selected:
                        continue
                    plot_group(ax, selected, metric, horizontal, condition, COLORS[condition], logscale)
                    if metric == "x0_sat_frac_mean":
                        plot_group(ax, selected, "action_oor_frac", horizontal, condition + " action |a|>1", COLORS[condition], linestyle="--")
                    if metric == "B_env":
                        for run in selected:
                            undefined = [w for w in run["windows"] if w["reason"] == "nonpositive_split_dot" and finite(w.get(horizontal))]
                            ax.scatter([w[horizontal] / (1e6 if horizontal == "step" else 1) for w in undefined],
                                       [.025] * len(undefined), color=COLORS[condition], marker="x", s=15,
                                       alpha=.45, transform=ax.get_xaxis_transform())
                for reference, color, style in (("Stage 1 DPPO", "#555555", "--"), ("Stage 2 NC4 1e-4", "#999999", ":")):
                    if metric not in ("eval_episode_reward", "train_episode_reward", "kl_true_per_action"):
                        continue
                    source = references[reference]
                    maps = [{r[horizontal]: r[metric] for r in item["rows"] if finite(r.get(metric))} for item in source]
                    common = sorted(set.intersection(*(set(m) for m in maps)))
                    if common:
                        values = np.array([[m[x] for x in common] for m in maps]).mean(axis=0)
                        if logscale:
                            values = np.where(values > 0, values, np.nan)
                        ax.plot(np.array(common) / (1e6 if horizontal == "step" else 1), values,
                                color=color, linestyle=style, linewidth=1.1, label=reference)
                if metric == "theta_dist":
                    ax.axhline(anchors["projection_radius"], color="gray", linestyle=":", linewidth=.9)
                if metric == "B_env":
                    ax.axhline(40, color="gray", linestyle=":", linewidth=.8)
                if logscale:
                    ax.set_yscale("log")
                if metric == "grad_cos_prev":
                    ax.axhline(0, color="gray", linewidth=.5)
                ax.set_title(panel_title, fontsize=10)
                ax.set_xlabel("Training environment steps (millions)" if horizontal == "step" else "Iteration", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=.18)
                if column == 0:
                    ax.set_ylabel(title, fontsize=10)
                    ax.legend(fontsize=7, frameon=False, loc="best")
                if column == 7:
                    ax.text(.02, .98, "solid: x0 saturation\ndashed: executed |a| > 1", transform=ax.transAxes,
                            va="top", fontsize=7)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figdir / (filename + "." + extension), dpi=180)
        plt.close(fig)
    draw([(title, conditions, "step") for title, conditions in PHASES], "stage3_phase_comparison")
    if any(run["spec"]["condition"] == "BATCH4" and run["rows"] for run in runs):
        draw([("BATCH4 vs SGD3: iteration", ["SGD3", "BATCH4"], "itr"),
              ("BATCH4 vs SGD3: samples", ["SGD3", "BATCH4"], "step")], "stage3_batch4_comparison")
    caption = ("The three rows compare the AdamW estimator ladder, the SGD sweep, and projection/batch-size conditions; columns show the eight quantities named on the axes. "
               "Thin lines are individual training seeds; bold curves and bands are the mean ± sample SD (ddof=1) only where all three seeds have finite comparable observations, with no smoothing or extrapolation. "
               "Gray dashed/dotted curves are the Stage-1 DPPO and Stage-2 NC4 1e-4 means where those metrics exist; the distance guide is the fixed projection radius, and B_env=40 marks unit training-batch SNR under the approximate independent-env interpretation. "
               "B_env uses complete nonoverlapping iteration windows [0,9], [10,19], etc., each containing nine training iterations, and equals 10 times summed split-gradient squared differences divided by summed split dots; bottom-axis crosses mark nonpositive denominators, not numerical values. "
               "Log axes omit exact/nonpositive observations and nonpositive lower SD bands without epsilon replacement; raw values and failure prefixes remain in the CSVs. "
               "In the final panel, solid curves show pre-update x0 saturation and dashed curves the fraction of physical executed action coordinates outside [-1,1]. "
               "BATCH4 has four times as many samples per iteration; its second figure repeats the comparison against iteration and environment samples. "
               "Shared full-batch normalization/LOO coefficients couple split estimates, so noise scales are approximate; three seeds do not establish a general ordering.")
    (figdir / "captions.md").write_text("# Stage-3 figures\n\n" + caption + "\n\nLog/undefined observations: " + json.dumps(omissions) + "\n")
    return omissions, caption


def fmt(value, digits=4):
    if not finite(value):
        return "—"
    value = float(value)
    return ("%." + str(digits) + "g") % value


def estimate_text(value):
    if value["mean"] is None:
        return "unresolved (%d/3 available)" % value["n_available"]
    return fmt(value["mean"]) + " ± " + fmt(value["sd"])


def cell(grouped, condition, metric, seeds=False):
    values = grouped[condition]["metrics"][metric]
    result = estimate_text(values)
    if seeds:
        result = " / ".join(fmt(x) for x in values["per_seed"]) + "; " + result
    return result


def md_table(headers, rows):
    if not rows:
        return "No matching observations in the saved records."
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(escape, row)) + " |" for row in rows]
    return "\n".join(lines)


def link(root, path, label=None):
    path = Path(path)
    target = os.path.relpath(path, root)
    return "[%s](%s)" % (label or path.name, target)


def mode_evidence(root, campaign, condition, filename):
    spec = next((run for run in campaign.get("runs", []) if run["condition"] == condition), None)
    if not spec:
        return {}, "not launched"
    manifest = read_json(root / "runs" / spec["run_id"] / "manifest.json", {})
    status = manifest.get("status", "planned")
    output = read_json(Path(spec["logdir"]) / filename, {}) if status == "complete" else {}
    return output, status


def comparison_sentence(grouped, first, second, metric):
    value = comparison(grouped, first, second, metric)
    return "%s→%s: change %s, two-pooled-SD threshold %s; %s." % (
        first, second, fmt(value["difference"]), fmt(value["threshold"]), value["interpretation"])


def noise_uncertainty_review(root):
    review = read_json(Path(root) / "runs/noise_ci_review.json", {})
    return review if review.get("status") == "uncertainty_unreliable" else {}


def noise_uncertainty_warning():
    return ("**Phase-1 uncertainty is unreliable.** The saved raw percentile bootstrap intervals for signal norm² are strongly upward shifted. "
            "CI-based positivity/resolution claims, derived confidence intervals and noise-scale lower bounds must not be used. "
            "Signed point moments and saved point-derived ratios/cosines below are provisional; no replacement intervals were computed. "
            "Training curves and across-seed sample SDs are unaffected. [Read-only statistical audit](runs/noise_ci_review.json).")


def phase1_table(value, unreliable=False):
    keys = ("u_sq", "tr_sigma", "B_env", "B_ep", "snr_training_batch", "expected_batch_cosine")
    labels = ("Signal norm²", "Noise trace", "B_env", "B_ep", "Batch SNR", "Expected batch cosine")
    if unreliable:
        return md_table(["Variant"] + [label + " (point only)" for label in labels],
                        [[row["variant"]] + [fmt(row.get(key)) for key in keys] for row in value["variants"]])
    cis = ("u_sq_ci90", "tr_sigma_ci90", "B_env_ci90", "B_ep_ci90", "snr_training_batch_ci90", "expected_batch_cosine_ci90")
    rows = []
    for row in value["variants"]:
        entries = [fmt(row.get(key)) + " [" + ", ".join(fmt(x) for x in row.get(ci) or (None, None)) + "]"
                   for key, ci in zip(keys, cis)]
        rows.append([row["variant"]] + entries + [fmt(row.get("B_env_lower_bound"))])
    return md_table(["Variant"] + [label + " [CI]" for label in labels] + ["B_env lower bound"], rows)


def report(root, campaign, runs, summaries, grouped, references, anchors, noise, calibration, theta0,
           omissions, caption, allow_partial):
    completed = sum(run["status"] == "complete" for run in runs)
    failed = [row for row in summaries if row["status"] == "failed"]
    collapsed = [row for row in summaries if row["collapse"]]
    gate_path = Path(campaign.get("gate", root / "runs/gate/gate_summary.json"))
    gate = read_json(gate_path, {})
    uncertainty_unreliable = bool(noise_uncertainty_review(root))
    lines = ["# Stage 3: Girsanov-score estimator and projected SGD", "",
             "**Question.** How do the estimator, optimizer and projection affect gradient noise, discounted return and stability when moving from NC4 to the prescribed discretized score algorithm?", "",
             "**Status.** %d completed training runs, %d failed runs, %d observed collapse flags; prospective skips remain explicit below. %s" %
             (completed, len(failed), len(collapsed), "This is a partial report; unresolved work is not presented as a result." if allow_partial else "All selected runs are terminal; completed/failed/skipped states are preserved."), "",
             "The interpretation is limited by three training seeds, batch-coupled noise estimates and initial Gym states that are not paired by the existing seed wrapper. No additional significance tests, rescue changes or automatic retries were used.", ""]
    if uncertainty_unreliable:
        lines += [noise_uncertainty_warning(), ""]
    lines += ["## Verification gate", "", "Gate status: **%s**. Full machine-readable evidence: %s." % (gate.get("status", "unavailable"), link(root, gate_path)), ""]
    gate_rows = []
    for name, item in gate.get("A", {}).items():
        comparisons = item.get("comparisons", {})
        absolute = max((value.get("max_abs", 0) for value in comparisons.values() if finite(value.get("max_abs"))), default=None)
        relative_values = [value.get("max_rel") for value in comparisons.values()]
        relative = "inf" if "inf" in relative_values else max((x for x in relative_values if finite(x)), default=None)
        gate_rows.append([name, item.get("passed"), fmt(absolute), relative if relative == "inf" else fmt(relative), item.get("actor_steps"), item.get("critic_steps")])
    if gate_rows:
        lines += [md_table(["Default replay", "Pass", "Max absolute difference", "Max relative difference", "Actor steps", "Critic steps"], gate_rows), ""]
    for device, feature in gate.get("features", {}).items():
        lines += ["Feature checks on **%s**:" % device, "", md_table(["Check", "Evidence"],
                  [[name, "`" + json.dumps(safe(value), sort_keys=True) + "`"] for name, value in feature.get("checks", {}).items()]), ""]
    lines += ["B9 diagnostics-inert gate: **%s**; parameters, optimizer state, gradients and global RNG comparison details remain in the linked gate." % gate.get("B9", {}).get("passed", "unavailable"), "",
              "CPU defaults require bitwise equality; GPU tolerances are taken from the saved gate. B2b and B6 are report-only checks. A missing/failed gate does not authorize scientific launches.", "",
              "## Phase 1: fixed-policy gradient noise", "",
              "Each policy point uses 25 fresh batches, 40 environments per batch and two episodes per environment. Gradients are negative mean-loss gradients per 5,000-pair env rollout; sign does not affect norms or pairwise cosines. The prescribed n-based signal correction is approximate for full-batch-normalized and LOO-coupled variants.", ""]
    if uncertainty_unreliable:
        lines += [noise_uncertainty_warning(), "",
                  "The saved 90% intervals used 2,000 whole-batch resamples. They remain in the raw records for provenance but are omitted from these tables, together with the invalid confidence-based flags and bounds. This bootstrap problem also affects V5, which has no normalization or LOO coupling. B10 checked point moments, not interval coverage.", ""]
    else:
        lines += ["Intervals are 90% batch-bootstrap intervals from 2,000 resamples; clustering the bootstrap does not remove centering bias.", ""]
    for point in ("ns_theta0", "ns_ckpt"):
        value, status = noise[point]
        lines += ["### " + ("Pretrained θ₀" if point == "ns_theta0" else "NC4 final θ checkpoint"), "", "Status: %s." % status, ""]
        if not value:
            lines += ["Measurement unavailable; no quantities inferred.", ""]
            continue
        source = next((spec for spec in campaign.get("runs", []) if spec["condition"] == point), None)
        if source:
            lines += ["Original measurement and uncertainty fields (preserved): " + link(root, Path(source["logdir"]) / "noise_scale.json") + ".", ""]
        lines += [phase1_table(value, uncertainty_unreliable), "",
                  ("Saved provisional mean-direction cosine points (not clipped; dashes retain unavailable entries and do not imply validated signal resolution):" if uncertainty_unreliable else
                   "Mean-direction cosine matrix (signed estimates are not clipped; unresolved entries are unavailable):"), "",
                  md_table(["Variant"] + VARIANTS, [[name] + [fmt(x) for x in row] for name, row in zip(VARIANTS, value["cosine_matrix"])]), "",
                  "Warm-up batches: %s; warmed-critic explained variance on the first frozen measurement batch: %s. %s" %
                  (value.get("warmup_batches"), fmt(value.get("critic_explained_var_after_warmup")), "No valid Phase-1 confidence bound is reported." if uncertainty_unreliable else value.get("bound_note", "")), ""]
    lines += ["## Calibration and projection anchors", "",
              "KL* = **%s**, the pooled median of 378 training diagnostics from the three Stage-2 NC4 1e-4 runs. R = **%s**, 1.25 times the largest Stage-1 final actor distance. The anchor measurements were fixed before wave 0." %
              (fmt(anchors["kl_star"], 10), fmt(anchors["projection_radius"], 10)), "",
              md_table(["Reference", "Seed 0 distance", "Seed 1 distance", "Seed 2 distance"],
                       [["Stage 1"] + [fmt(x) for x in anchors["stage1_final_distances"]],
                        ["Stage 2 NC4 1e-4"] + [fmt(x) for x in anchors["nc4_final_distances"]]]), ""]
    if calibration:
        lines += ["η* = **%s**; η*_theory = **%s**; max/min η_b = **%s**." %
                  (fmt(calibration["eta_star"], 10), fmt(calibration["eta_star_theory"], 10), fmt(calibration["eta_spread_max_over_min"])), "",
                  md_table(["Batch", "η_b", "η_theory", "KL(η_b)", "Local exponent", "Warnings"],
                  [[row["batch"], fmt(row["eta_b"], 10), fmt(row["eta_theory_b"], 10), fmt(row["kl_eta_b"]), fmt(row["exponent"]), "; ".join(row["warnings"]) or "none"] for row in calibration["batches"]]), "",
                  "Calibration-wide warnings: " + ("; ".join(calibration.get("warnings", [])) or "none") + ".", ""]
    else:
        lines += ["Calibration unavailable; no SGD step size is inferred.", ""]
    lines += ["## Training curves", "", "![Stage-3 phase comparison](figs/stage3_phase_comparison.png)", "", caption, "",
              "Omitted log-axis / undefined observations: `" + json.dumps(omissions, sort_keys=True) + "`. Values remain in the raw tables.", ""]
    if (root / "figs/stage3_batch4_comparison.png").exists():
        lines += ["![BATCH4 versus SGD3 by iteration and samples](figs/stage3_batch4_comparison.png)", "",
                  "The same eight quantities are shown against iteration in the first row and environment samples in the second. Aggregation and uncertainty follow the preceding caption; BATCH4 collects 320 episodes instead of 80 at the same SGD3 step size. The budgets therefore differ by four in samples at equal iteration.", ""]
    lines += ["## Results tables", "",
              "Each cell lists seed 0 / seed 1 / seed 2, followed by mean ± sample SD (ddof=1). Means require all three comparable seed values; missing failures are never replaced by survivor-only means. Last-10 summaries use the final ten training iterations of a completed 140-iteration run. Final evaluation is itr 130 (9.36M env steps; BATCH4 37.44M); state_139 contains the final post-update actor. The complete machine-readable tables are [per seed](seed_results.csv), [per condition](condition_results.csv), [all iteration values](all_runs_long.csv) and [noise windows](noise_windows.csv).", ""]
    table_groups = [(["Eval itr 0", "Eval itr 130", "Train last 10", "Discounted train last 10"],
                     ["eval_itr0", "eval_itr130", "train_last10", "J_disc_last10"]),
                    (["Final train-sampler return", "Final eval-sampler return", "Final train J_disc", "Final eval J_disc", "Δ train J_disc", "Δ eval J_disc"],
                     ["ckpt_train_return", "ckpt_eval_return", "ckpt_train_J_disc", "ckpt_eval_J_disc", "delta_train_J_disc", "delta_eval_J_disc"]),
                    (["Median KL", "Final θ distance", "Median B_env", "Zero-update fraction", "Projection fraction"],
                     ["median_kl", "theta_dist_final", "median_B_env", "zero_update_fraction", "projection_fraction"]),
                    (["Mean x0 saturation", "Last-step x0 saturation", "Action outside range", "Seconds / training itr"],
                     ["x0_saturation_mean", "x0_saturation_last", "action_oor_mean", "seconds_per_training_iteration"])]
    for labels, keys in table_groups:
        lines += [md_table(["Condition"] + labels, [[condition] + [cell(grouped, condition, key, True) for key in keys] for condition in CONDITIONS]), ""]
    lines += [md_table(["Condition", "Run states (0/1/2)", "Observed collapse flags (0/1/2)"],
                       [[condition, " / ".join(grouped[condition]["statuses"]), " / ".join(str(x) for x in grouped[condition]["collapse_per_seed"])] for condition in CONDITIONS]), "",
              "Collapse means an observed eval return below half that run's itr-0 value or a numerical nonfinite failure. Null diagnostics caused by unavailable previous gradients or nonpositive split dots are undefined measurements, not numerical failures. Failed-run finite prefixes remain in plots/long tables and their status remains explicit; final summaries are unavailable when the required run endpoint was not reached.", "",
              "### Final pretrained evaluation", ""]
    if theta0:
        lines += [md_table(["Sampler", "Undiscounted mean ± episode s.e.", "J_disc mean ± episode s.e.", "Episodes", "x0 saturation"],
                  [[sampler, fmt(value["undiscounted_return"]["mean"]) + " ± " + fmt(value["undiscounted_return"]["se"]),
                    fmt(value["J_disc"]["mean"]) + " ± " + fmt(value["J_disc"]["se"]), value["J_disc"]["n"], fmt(value["x0_sat_frac"])] for sampler, value in theta0["samplers"].items()]), "",
                  "Final checkpoints each contribute 80 episodes per sampler; θ₀ contributes three rollouts (240 episodes per sampler). ΔJ_disc subtracts the corresponding pooled θ₀ mean; the reported across-seed SD does not incorporate θ₀'s episode-level estimation uncertainty.", "",
                  theta0.get("sampling_note", ""), ""]
    else:
        lines += ["Pretrained checkpoint evaluation unavailable; ΔJ_disc is unresolved.", ""]
    lines += ["## Answers to Q1–Q8", ""]
    if uncertainty_unreliable:
        lines += ["**Q1–Q3 uncertainty limit.** Phase-1 numbers are provisional point estimates only: the saved percentile CIs cannot establish signal positivity, noise dominance, an estimator ranking, or a change between policies. No confidence lower bounds are used. " +
                  "See the [statistical audit](runs/noise_ci_review.json); the separate training-seed sample-SD comparisons remain available.", ""]
    ns0, nsck = noise["ns_theta0"][0], noise["ns_ckpt"][0]
    v0 = {row["variant"]: row for row in ns0.get("variants", [])}
    vk = {row["variant"]: row for row in nsck.get("variants", [])}
    lines += ["### Q1. Noise at m=80 episodes", ""]
    if v0:
        headers = ["Variant", "B_ep (provisional point)", "Expected batch-gradient cosine (provisional point)"] if uncertainty_unreliable else ["Variant", "B_ep", "Expected batch-gradient cosine", "Signal CI includes zero"]
        rows = [[name, fmt(v0[name]["B_ep"]), fmt(v0[name]["expected_batch_cosine"])] +
                ([] if uncertainty_unreliable else [v0[name]["signal_ci_includes_zero"]]) for name in VARIANTS]
        lines += [md_table(headers, rows), ""]
        theory = v0["theory_nobase"]
        classification = ("noise exceeds signal at this batch size" if theory["B_ep"] > 80 else "the point estimate does not exceed the training batch size") if finite(theory.get("B_ep")) else "the signal is unresolved, so a point noise-dominance classification is unavailable"
        if uncertainty_unreliable:
            lines += ["V5's provisional point estimate gives B_ep=%s and expected batch cosine=%s. This suggests noise-dominated gradients when B_ep exceeds 80, but the unavailable valid uncertainty prevents a confidence-based conclusion. No lower bound is reported." %
                      (fmt(theory.get("B_ep")), fmt(theory.get("expected_batch_cosine"))), ""]
        else:
            lines += ["For V5, %s; B_env lower bound %s. This uses the prescribed moment/expected-cosine approximation." % (classification, fmt(theory.get("B_env_lower_bound"))), ""]
    else:
        lines += ["Unresolved: pretrained noise measurement unavailable.", ""]
    lines += ["### Q2. Which estimator change moves noise?", ""]
    if v0:
        pairs = [(0, 1, "GAE→MC with critic"), (1, 2, "critic→LOO and raw rewards under normalization"),
                 (2, 3, "decision discount plus E2 rescalings"), (3, 4, "remove LOO baseline")]
        table, changes = [], []
        for a, b, label in pairs:
            left, right = v0[VARIANTS[a]]["B_env"], v0[VARIANTS[b]]["B_env"]
            ratio = right / left if finite(left) and finite(right) and left > 0 else None
            table.append([label, fmt(left), fmt(right), fmt(ratio), fmt(ns0["cosine_matrix"][a][b])])
            if finite(ratio) and ratio > 0:
                changes.append((abs(math.log(ratio)), label))
        ranking = ("Largest absolute log-ratio among the resolved point estimates: " + max(changes)[1] + ". This is a descriptive ranking; uncertainty is in the Phase-1 tables." if len(changes) == 4 else "A complete noise-change ranking is unresolved because one or more signal estimates are unavailable.")
        if uncertainty_unreliable:
            ranking = ("Largest absolute log-ratio among the available provisional points: " + max(changes)[1] + ". This is descriptive only; no ranking is established with valid uncertainty." if len(changes) == 4 else
                       "A complete point ranking is unavailable because some saved signal-derived values are missing. The remaining provisional ratios and cosines have no validated Phase-1 confidence intervals.")
        lines += [md_table(["Joint intervention", "Before B_env", "After B_env", "After/before", "Mean-direction cosine"], table), "", ranking, ""]
    else:
        lines += ["Unresolved: pretrained noise measurement unavailable.", ""]
    lines += ["### Q3. Noise across policies and training", ""]
    if v0 and vk:
        lines += [md_table(["Variant", "θ₀ B_env", "θ_ckpt B_env", "Checkpoint / pretrained"],
                  [[name, fmt(v0[name]["B_env"]), fmt(vk[name]["B_env"]), fmt(vk[name]["B_env"] / v0[name]["B_env"])
                    if finite(v0[name]["B_env"]) and v0[name]["B_env"] > 0 and finite(vk[name]["B_env"]) else "unresolved"] for name in VARIANTS]), ""]
        if uncertainty_unreliable:
            lines += ["These cross-policy ratios are descriptive comparisons of saved provisional point estimates. The invalid Phase-1 percentile intervals cannot resolve whether the underlying noise scale changed.", ""]
    trajectory_noise = []
    for condition in ("E0", "E2", "SGD03", "SGD1", "SGD3", "SGD10"):
        selected = sorted([run for run in runs if run["spec"]["condition"] == condition], key=lambda r: r["spec"]["seed"])
        first = mean_sd([run["windows"][0]["B_env"] for run in selected])
        last = mean_sd([run["windows"][-1]["B_env"] for run in selected])
        trajectory_noise.append([condition, estimate_text(first), estimate_text(last), cell(grouped, condition, "median_B_env")])
    lines += [md_table(["Condition", "Window 0–9 B_env", "Window 130–139 B_env", "Median window B_env"], trajectory_noise), "",
              "Only complete windows with positive summed split dot define B_env. Excluding undefined windows from a median is explicit in noise_windows.csv; it is not evidence that those windows had low noise.", "",
              "### Q4. E0 versus E1 versus E2", "",
              md_table(["Condition", "Eval itr130", "Train last10", "J_disc last10", "Final train-sampler ΔJ_disc"],
                       [[condition] + [cell(grouped, condition, key) for key in ("eval_itr130", "train_last10", "J_disc_last10", "delta_train_J_disc")] for condition in ("E0", "E1", "E2")]), ""]
    for first, second in (("E0", "E1"), ("E1", "E2")):
        for key in ("eval_itr130", "train_last10", "J_disc_last10"):
            lines += ["%s: %s" % (key, comparison_sentence(grouped, first, second, key)), ""]
    e1j, e2j = grouped["E1"]["metrics"]["J_disc_last10"]["mean"], grouped["E2"]["metrics"]["J_disc_last10"]["mean"]
    e1r, e2r = grouped["E1"]["metrics"]["train_last10"]["mean"], grouped["E2"]["metrics"]["train_last10"]["mean"]
    if all(finite(x) for x in (e1j, e2j, e1r, e2r)):
        lines += ["The E1→E2 point means %s a higher discounted objective alongside a lower undiscounted training return. This intervention changes decision weighting and the listed normalization/reduction choices jointly, so the change is not isolated to γ^k." % ("show" if e2j > e1j and e2r < e1r else "do not show"), ""]
    lines += ["### Q5. Calibrated SGD", "",
              md_table(["Condition", "η (code)", "η_theory", "Mean seed-median KL", "KL / KL*", "Final train ΔJ_disc", "Collapse seeds"],
                       [[condition, cell(grouped, condition, "actor_lr"),
                         fmt(grouped[condition]["metrics"]["actor_lr"]["mean"] / 2500) if condition.startswith("SGD") and grouped[condition]["metrics"]["actor_lr"]["mean"] is not None else "—",
                         cell(grouped, condition, "median_kl"), fmt(grouped[condition]["metrics"]["median_kl"]["mean"] / anchors["kl_star"])
                         if grouped[condition]["metrics"]["median_kl"]["mean"] is not None else "—",
                         cell(grouped, condition, "delta_train_J_disc"), str(grouped[condition]["collapse_per_seed"])]
                        for condition in ("SGD03", "SGD1", "SGD3", "SGD10", "E2", "E2_beta0")]), ""]
    for other in ("E2", "E2_beta0"):
        lines += [comparison_sentence(grouped, other, "SGD1", "ckpt_train_J_disc"), ""]
    lines += ["η_theory=η/2500 applies to plain SGD; AdamW learning rates are listed for reference and do not define a scalar step on the raw gradient. Positive ΔJ_disc is a point increase over θ₀, not a further significance claim.", "",
              "### Q6. Projection", ""]
    for condition, source in (("PROJ3", "SGD3"), ("PROJ10", "SGD10")):
        decision = campaign.get("projection_decisions", {}).get(condition, {})
        if decision.get("skip"):
            lines += ["%s was prospectively skipped: maximum matching %s preprojection distance %s < 0.8R=%s. %s" %
                      (condition, source, fmt(decision.get("maximum")), fmt(decision.get("threshold")), decision.get("reason", "")), ""]
        else:
            lines += ["%s active-iteration fraction: %s. %s" %
                      (condition, cell(grouped, condition, "projection_fraction", True), comparison_sentence(grouped, source, condition, "ckpt_train_J_disc")), ""]
    lines += ["### Q7. Saturation and frozen updates", "",
              md_table(["Condition", "x0 saturation", "Final-step x0 saturation", "Physical |a|>1", "Zero-update fraction", "Observed collapses"],
                       [[condition] + [cell(grouped, condition, key) for key in ("x0_saturation_mean", "x0_saturation_last", "action_oor_mean", "zero_update_fraction")]
                        + [str(grouped[condition]["collapse_per_seed"])] for condition in CONDITIONS]), "",
              "These are measured associations. Retaining the x0 clamp and observing saturation does not isolate its causal effect; update_norm=0 is a parameter-level check, not an inference from small KL or repeated returns.", "",
              md_table(["Run with observed zero updates", "Longest consecutive training-update streak", "x0 saturation during zero updates", "Observed collapse"],
                       [[row["run"], row["longest_zero_update_streak"], fmt(row["x0_saturation_during_zero_updates"]), row["collapse"]]
                        for row in summaries if row["longest_zero_update_streak"] > 0]), "",
              "### Q8. Early split-gradient cross-check", ""]
    crosscheck = []
    for condition, variant in (("E1", "mc_loo"), ("E2", "theory_loo")):
        selected = sorted([run for run in runs if run["spec"]["condition"] == condition], key=lambda r: r["spec"]["seed"])
        values = mean_sd([run["windows"][0]["B_env"] for run in selected])
        crosscheck.append([condition + " / " + variant, estimate_text(values), fmt(v0.get(variant, {}).get("B_env")),
                           "unreliable; omitted" if uncertainty_unreliable else str(v0.get(variant, {}).get("B_env_ci90"))])
    lines += [md_table(["Training / fixed-policy variant", "First-window B_env", "Phase-1 θ₀ B_env", "Phase-1 uncertainty" if uncertainty_unreliable else "Phase-1 90% CI"], crosscheck), "",
              "The first window already spans evolving training policies; agreement is a consistency check, not an equality requirement. E0 is excluded because its critic starts untrained whereas V1 uses a warmed critic.", ""]
    return lines


def deviations(grouped):
    def effect(first, second):
        result = comparison(grouped, first, second, "eval_itr130")
        return ("Joint %s→%s eval change %s; %s. Individual contribution not isolated." %
                (first, second, fmt(result["difference"]), result["interpretation"]))
    estimator = effect("E0", "E1")
    weighting = effect("E1", "E2")
    return [
        ["Data reuse", "PPO, 5 epochs × 4 minibatches", "one step per fresh batch", "one step", "NC4 (Stage 2)", "Not changed in Stage 3; E0 versus prior NC4 is the control comparison."],
        ["Ratio clip ε=0.01, KL stop, log-prob clamp [−5,2]", "on", "none", "off (inert without reuse)", "Stage 2", "Not separately measured in Stage 3."],
        ["Noise truncation ±3σ", "on", "Gaussian", "off (randn_clip_value=100)", "Stage 2", "Not separately measured; finite ±100 setting retained exactly."],
        ["Return estimate", "GAE λ=0.95 (cut at episode end)", "MC G_k", "MC G_k", "E1", estimator],
        ["Baseline", "critic V(s) inside GAE", "none", "LOO mean of G_k at the same k", "E1", estimator + " No-baseline final train J_disc: " + comparison_sentence(grouped, "E2", "E2_nobase", "ckpt_train_J_disc")],
        ["γ^k on decision k", "absent (the update is not the gradient of any function)", "present", "present", "E2", weighting],
        ["Denoising-step weights", "0.99^(9−i)", "1", "1", "E2", "B6 reports the fixed-batch gradient cosine; training effect jointly measured in E1→E2."],
        ["Log-prob reduction", "mean of 24 coords (a factor 1/24)", "sum", "sum", "E2", "B5 verifies the factor of 24; training effect jointly measured in E1→E2."],
        ["Advantage normalization", "per minibatch", "none", "none", "E2", weighting],
        ["Reward scaling for the actor", "running std (+ clip ±10)", "none (a constant only rescales η)", "raw rewards", "E2", "Joint estimator comparisons; reward_clip_frac is retained per iteration. Separate scaling/clipping intervention not measured."],
        ["Optimizer", "AdamW (0.9, 0.999), wd 0", "SGD", "SGD, momentum 0", "SGD*", "Final train J_disc: " + comparison_sentence(grouped, "E2", "SGD1", "ckpt_train_J_disc")],
        ["Step size", "1e-4", "constant η ≤ 1/L", "η* from KL calibration", "SGD*", "Empirical KL calibration and prescribed η sweep; L and the theoretical smoothness inequality are not measured."],
        ["Projection", "none", "Π_Θ, Θ convex", "ball B(θ₀, R)", "PROJ*", "Active fractions: PROJ3 " + cell(grouped, "PROJ3", "projection_fraction") + "; PROJ10 " + cell(grouped, "PROJ10", "projection_fraction") + ". See Q6 for prospective skips and matching-SGD comparisons."],
        ["Sampling/log-prob noise floor 0.1", "on", "ellipticity (σ ≥ 0.1)", "on", "—", "not measured"],
        ["x̂₀ clamp ±1", "on", "violates smoothness (flat where saturated)", "on, logged", "—", "Mean saturation: E2 " + cell(grouped, "E2", "x0_saturation_mean") + "; SGD10 " + cell(grouped, "SGD10", "x0_saturation_mean") + ". Causal removal not measured."],
        ["First 10 of 20 denoising steps frozen", "on", "fine (θ enters only the last 10 steps)", "on", "—", "not measured"],
        ["Chunk T_a=4, γ per decision", "on", "decision-level MDP", "on", "—", "not measured"],
        ["Horizon", "250 decisions, terminal at the limit", "J_H with H=250 (here J_H is the objective)", "same", "—", "Alignment asserted; B2b reports native termination and GAE-cut behavior."],
        ["Episodes per batch", "80", "m", "80 (BATCH4: 320)", "—", "Final train J_disc: " + comparison_sentence(grouped, "SGD3", "BATCH4", "ckpt_train_J_disc") + " Sample budgets differ by four at equal iteration."],
        ["Critic", "trained, used by GAE", "none", "trained but unused by the actor from E1 on", "E1", "Warmed-critic explained variance measured in Phase 1; estimator change jointly assessed at E0→E1."],
    ]


def finish_report(root, lines, campaign, runs, summaries, grouped, references, anchors, gate, calibration, allow_partial):
    uncertainty_unreliable = bool(noise_uncertainty_review(root))
    lines += ["## Deviations from the analyzed algorithm", "",
              "The first five columns reproduce the specified design comparison. The added column distinguishes direct measurements from joint interventions and unmeasured theoretical assumptions.", "",
              md_table(["Choice", "DPPO (Stage 1)", "Paper", "Stage 3", "Changed in", "Measured effect"], deviations(grouped)), "",
              "**Reward-unit clarification.** The supplied comparison table labels the actor reward-scaling change E2. The literal MC implementation in §2.2 already constructs E1 actor advantages from raw rewards, before E1's per-minibatch advantage normalization; E2 then removes that normalization and exposes the raw reward units. The table's original five columns are preserved, while the implemented sequence follows §2.2. Critic reward scaling and targets remain unchanged.", "",
              "## Control sanity check", ""]
    nc4 = [next(row["eval_episode_reward"] for row in reference["rows"] if row["itr"] == 130)
           for reference in references["Stage 2 NC4 1e-4"]]
    nc4_stats = mean_sd(nc4)
    e0 = grouped["E0"]["metrics"]["eval_itr130"]
    if e0["mean"] is not None:
        threshold = 2 * math.sqrt((e0["sd"] ** 2 + nc4_stats["sd"] ** 2) / 2)
        flag = abs(e0["mean"] - nc4_stats["mean"]) > threshold
        lines += ["E0 eval mean %s versus prior NC4 mean %s; two-pooled-SD tolerance %s. **Sanity red flag: %s.** This is a reporting check and did not stop training." %
                  (fmt(e0["mean"]), fmt(nc4_stats["mean"]), fmt(threshold), flag), ""]
    else:
        lines += ["E0 sanity comparison unresolved: three complete endpoint values are unavailable.", ""]
    lines += ["## Provenance, resources and run artifacts", "",
              "Stage-2 tip: `ab46b150fa34b5a5b457cd4062cd4c5ad830d964`; base: `cc7234ad7ff39a8f32de3af903606723a16f0648`. Stage-3 tested commit: `%s`. " % campaign.get("commit", gate.get("scientific_source_commit", "unavailable")) +
              "No Stage-2 snapshot commit was needed according to [approval/provenance](provenance/approval.json). "
              "The exact source change is in %s; software/environment and actual commands are recorded per allocation." % link(root, gate.get("source_diff_path", root / "runs/gate/source.diff")), "",
              "Approved cap: 200 allocated GPU-hours, at most nine parallel RTX A6000 jobs, 40 CPU cores per job. Recorded allocated use including the gate: %s GPU-hours; reserved caps: %s GPU-hours." %
              (fmt(campaign.get("measured_gpu_seconds_including_gate", 0) / 3600), fmt(campaign.get("reserved_gpu_seconds", 0) / 3600)), "",
              "The controller launches all three seeds of a condition together when slots allow; it never changes the run configuration to reduce resource demand. No automatic retry follows a numerical, alignment or allocation failure. [Campaign record](runs/campaign.json), [submission/command journal](runs/campaign_commands.jsonl), [queue plan](runs/queue_plan.json), [anchor measurements](provenance/anchors.json).", "",
              md_table(["Condition", "Wall span across its three allocated runs (seconds)"],
                       [[condition, fmt(grouped[condition]["wall_span_seconds"])] for condition in CONDITIONS]), "",
              md_table(["Run", "Status", "Allocated/wall seconds", "Log / results", "Final checkpoint", "Final evaluation"],
                  [[row["run"], row["status"], fmt(row["wall_seconds"]),
                    link(root, row["manifest_path"], "manifest") + " · " + link(root, row["result_path"], "result.pkl"),
                    link(root, row["checkpoints"][-1], "state_139") if Path(row["checkpoints"][-1]).is_file() else "unavailable",
                    link(root, row["checkpoint_eval_path"], "ckpt_eval") if Path(row["checkpoint_eval_path"]).is_file() else row["checkpoint_eval_status"]]
                   for row in summaries]), "",
              "Every checkpoint path, console log and effective command is retained in [analysis_record.json](analysis_record.json) and the per-run manifest. Failed checkpoint evaluations remain separate from their completed source training runs.", ""]
    mode_rows = []
    for spec in campaign.get("runs", []):
        if spec["mode"] == "train":
            continue
        manifest_path = root / "runs" / spec["run_id"] / "manifest.json"
        manifest = read_json(manifest_path, {})
        mode_rows.append([spec["run_id"], spec["mode"], manifest.get("status", "not available"),
                          fmt(manifest.get("wall_seconds")), link(root, manifest_path, "manifest"), manifest.get("failure_reason", "—")])
    lines += [md_table(["Measurement/evaluation job", "Mode", "Status", "Wall seconds", "Evidence", "Failure"], mode_rows), "",
              "### Exact job files and commands", "",
              "The [launcher](run_queue.sh), [worker](code/run_worker.sh) and [environment exports](code/environment.sh) are retained. Each manifest contains the exact argument vector, absolute paths, hardware and environment; [all_commands.json](all_commands.json) collects those vectors without reconstructing shell text.", ""]
    for filename in ("jobs_wave0.txt", "jobs_wave1.txt", "jobs_wave2.txt", "jobs_ckpt_eval.txt"):
        path = root / filename
        if path.exists():
            lines += [link(root, path, filename), "", "```text", path.read_text().rstrip(), "```", ""]
    failed = [row for row in summaries if row["status"] == "failed"]
    lines += ["## Failures, limits and departures", ""]
    if failed:
        lines += [md_table(["Run", "Last saved itr", "Cause", "Observed collapse", "Nonfinite evidence"],
                           [[row["run"], row["completed_iterations"] - 1, row["failure_reason"], row["collapse"], json.dumps(row["nonfinite_evidence"])] for row in failed]), ""]
    else:
        lines += ["No failed training allocation is recorded in this report's selected manifest set.", ""]
    lines += ["The unchanged Gym wrapper seeds global NumPy but does not seed the underlying environment's private generator; shared checkpoint-evaluation seeds align policy RNG streams, not all initial states. No seeding intervention was added. Pretrained evaluation interleaves train/eval per repeat so repeat 0 shares the policy-RNG prefix with one-repeat checkpoint jobs.", "",
              "The theoretical Gaussian score is implemented with the requested randn_clip_value=100 and the retained x0 clamp; this is the prescribed finite-precision discrete implementation, not a proof of the continuous-time theorem or smoothness assumptions. The sampled Stage-2 KL diagnostic measures fine-tuned denoising transitions, not exact final-action marginal KL.", "",
              ("LOO/full-batch normalization couples gradients within a batch, making the prescribed moment correction and split-window B_env approximate. Separately, the Phase-1 raw percentile uncertainty has the diagnosed upward-shift problem, including for the uncoupled V5 variant; its CI-based conclusions and lower bounds are withheld. Signed point moments and original measurements are preserved; no replacement intervals or new scientific samples were generated. [Audit](runs/noise_ci_review.json)." if uncertainty_unreliable else
               "LOO/full-batch normalization couples gradients within a batch. The prescribed moment correction and split-window B_env are approximate; 90% uncertainty resamples whole batches, retaining signed signal estimates and marking unresolved ratios unavailable. Derived intervals that would require dropping nonpositive bootstrap moments are left unavailable, not silently conditioned on favorable resamples."), "",
              "Phase-1 per-denoising variance decomposition and the optional Stage-2 collapsed-checkpoint saturation sweeps were not included in the approved budget. Included optional training conditions are E2_nobase, SGD03 and BATCH4. Saturation associations do not identify the effect of removing the clamp.", "",
              "All generated figures and tables are review candidates. Nothing is copied to curated without explicit artifact/name approval.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return lines


def generate(root=ROOT, allow_partial=False):
    root = Path(root).resolve()
    campaign = read_json(root / "runs/campaign.json", {})
    if not campaign:
        raise ValueError("campaign record unavailable; do not fabricate an executed study")
    if not allow_partial and (not campaign.get("evaluations_created") or campaign.get("status") not in ("finalizing", "complete")):
        raise ValueError("finalizer requires all waves and checkpoint evaluations to be terminal")
    training_slots = {(run["condition"], run["seed"]) for run in campaign.get("runs", []) if run["mode"] == "train"}
    if not allow_partial and training_slots != {(condition, seed) for condition in CONDITIONS for seed in range(3)}:
        raise ValueError("finalizer requires every approved condition/seed slot, including prospective skips")
    for spec in campaign.get("runs", []):
        manifest = read_json(root / "runs" / spec["run_id"] / "manifest.json", {})
        if not allow_partial and manifest.get("status") not in ("complete", "failed", "skipped"):
            raise ValueError("nonterminal manifest: " + spec["run_id"])
    if (root / "analysis_record.json").exists():
        raise ValueError("analysis evidence already exists; preserve it and request a distinct reviewed reanalysis destination")
    anchors = read_json(root / "provenance/anchors.json")
    runs = load_runs(root, campaign, allow_partial)
    references = reference_runs(root)
    theta0, theta0_status = mode_evidence(root, campaign, "ckpt_theta0", "ckpt_eval.json")
    calibration, calibration_status = mode_evidence(root, campaign, "cal", "calibration.json")
    noise = {name: mode_evidence(root, campaign, name, "noise_scale.json") for name in ("ns_theta0", "ns_ckpt")}
    summaries = [summarize(run, theta0) for run in runs]
    grouped = aggregate(summaries)
    write_csv(root / "all_runs_long.csv", [{"run": run["spec"]["run_id"], "condition": run["spec"]["condition"],
              "seed": run["spec"]["seed"], **row} for run in runs for row in run["rows"]])
    write_csv(root / "seed_results.csv", summaries)
    condition_rows = []
    for condition in CONDITIONS:
        entry = grouped[condition]
        flat = {key: value for key, value in entry.items() if key != "metrics"}
        for metric, values in entry["metrics"].items():
            flat.update({metric + "_mean": values["mean"], metric + "_sd": values["sd"],
                         metric + "_n": values["n_available"], metric + "_seed0": values["per_seed"][0],
                         metric + "_seed1": values["per_seed"][1], metric + "_seed2": values["per_seed"][2]})
        condition_rows.append(flat)
    write_csv(root / "condition_results.csv", condition_rows)
    write_csv(root / "noise_windows.csv", [{"run": run["spec"]["run_id"], "condition": run["spec"]["condition"],
              "seed": run["spec"]["seed"], **row} for run in runs for row in run["windows"]])
    mode_commands = []
    for spec in campaign.get("runs", []):
        path = root / "runs" / spec["run_id"] / "manifest.json"
        manifest = read_json(path, {})
        mode_commands.append({"run": spec["run_id"], "manifest": str(path), "command": manifest.get("command"),
                              "overrides": spec.get("overrides"), "status": manifest.get("status", "not_launched"),
                              "environment": manifest.get("environment"), "hardware": manifest.get("hardware"),
                              "console_log": spec.get("console_log")})
    write_json(root / "all_commands.json", mode_commands)
    omissions, caption = figures(root, runs, references, anchors)
    gate = read_json(campaign.get("gate", root / "runs/gate/gate_summary.json"), {})
    lines = report(root, campaign, runs, summaries, grouped, references, anchors, noise, calibration, theta0,
                   omissions, caption, allow_partial)
    finish_report(root, lines, campaign, runs, summaries, grouped, references, anchors, gate, calibration, allow_partial)
    artifacts = ["all_runs_long.csv", "seed_results.csv", "condition_results.csv", "noise_windows.csv", "all_commands.json", "REPORT.md"]
    artifacts += [str(path.relative_to(root)) for path in sorted((root / "figs").glob("stage3_*.*"))]
    for relative in artifacts:
        if not (root / relative).is_file() or (root / relative).stat().st_size == 0:
            raise ValueError("generated artifact is missing or empty: " + relative)
    record = {"status": "partial_candidates" if allow_partial else "completed_candidates",
              "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": sys.argv,
              "source_commit": campaign.get("commit"), "summaries": summaries, "conditions": grouped,
              "reference_sources": {name: [run["path"] for run in values] for name, values in references.items()},
              "calibration_status": calibration_status, "theta0_evaluation_status": theta0_status,
              "noise_statuses": {name: value[1] for name, value in noise.items()},
              "phase1_uncertainty_review": noise_uncertainty_review(root) or None,
              "artifacts": artifacts, "log_omissions": omissions, "sample_sd_ddof": 1,
              "mean_policy": "All three planned seeds must provide finite comparable values; no survivor-only means.",
              "curation_status": "unreviewed; explicit artifact/name approval required"}
    write_json(root / "analysis_record.json", record)
    if not allow_partial:
        complete = sum(run["status"] == "complete" for run in runs)
        failed = sum(run["status"] == "failed" for run in runs)
        collapsed = sum(bool(row["collapse"]) for row in summaries)
        study = ["# Stage 3: from NC4 to projected SGD", "",
                 "**Question.** How do the estimator, optimizer and projection affect gradient noise, discounted return and fine-tuning stability?", "",
                 "**Answer.** The selected campaign is terminal: %d training runs completed, %d failed, and %d observed collapse flags. The numerical answers and uncertainty are in the [full report](REPORT.md); missing or failed seeds remain explicit." % (complete, failed, collapsed), "",
                 "The [phase comparison](figs/stage3_phase_comparison.png) shows the complete selected evidence. All three seed values are required for means and sample-SD bands; failed prefixes remain visible without endpoint extrapolation. Noise estimates are approximate because normalization/LOO couples observations within batches, and shared evaluation seeds do not pair the underlying Gym initial states.", "",
                 "**Setup.** HalfCheetah, pretrained DPPO actor, 140 iterations, three training seeds per condition; the final scheduled eval is itr130 and final checkpoints are additionally evaluated with both samplers. [Methods](methods.md), [gate](runs/gate/gate_summary.json), [per-seed table](seed_results.csv), [condition table](condition_results.csv), [run/command record](analysis_record.json).", "",
                 "**Decision.** Review the candidate figures/tables and decide which to curate. No additional experiments, retries or curation are authorized by completion.", "",
                 "<!-- stage3-status-start -->", "Campaign terminal; all selected run statuses and failures are recorded in [campaign.json](runs/campaign.json).", "<!-- stage3-status-end -->", ""]
        if noise_uncertainty_review(root):
            study[6:6] = [noise_uncertainty_warning(), ""]
        if (root / "curated/INDEX.md").is_file():
            study[6:6] = ["Previously approved interim artifacts remain in the [curated index](curated/INDEX.md). New final-analysis candidates require their own artifact decision.", ""]
        (root / "experiment.md").write_text("\n".join(study))
    print(json.dumps({"status": record["status"], "artifacts": artifacts, "report": str(root / "REPORT.md")}, indent=2))
    return record


def self_test():
    values = mean_sd([1, 2, 3])
    assert values["mean"] == 2 and values["sd"] == 1
    assert mean_sd([1, None, 3])["mean"] is None
    rows = [{"itr": itr, "step": (itr - itr // 10) * 80000, "split_dot": 2., "split_diff_sq": 8.}
            for itr in range(20) if itr % 10]
    windows = window_noise(rows)
    assert windows[0]["B_env"] == windows[1]["B_env"] == 40
    assert windows[2]["B_env"] is None and windows[2]["reason"] == "incomplete_window"
    for row in rows:
        row["split_dot"] = -1.
    assert window_noise(rows)[0]["reason"] == "nonpositive_split_dot"
    assert numeric_bad({"grad_cos_prev": None, "loss": float("inf")}) == [{"key": ".loss", "value": "inf"}]
    grouped = aggregate([{ "condition": "E0", "seed": seed, "status": "complete", "collapse": False,
                           **{metric: seed for metric in METRICS}} for seed in (0, 1, 2)])
    assert grouped["E0"]["metrics"]["eval_itr130"]["sd"] == 1
    assert grouped["E1"]["metrics"]["eval_itr130"]["mean"] is None
    noise_row = {"variant": "theory_loo", "u_sq": -0.5, "tr_sigma": 10., "B_env": None,
                 "u_sq_ci90": [101., 102.], "tr_sigma_ci90": [103., 104.], "B_env_lower_bound": 105.}
    guarded = phase1_table({"variants": [noise_row]}, unreliable=True)
    assert "-0.5" in guarded and "10" in guarded and "point only" in guarded
    assert not any(token in guarded for token in ("[CI]", "lower bound", "101", "102", "103", "104", "105"))
    assert noise_row["u_sq_ci90"] == [101., 102.] and noise_row["B_env_lower_bound"] == 105.
    assert "signal norm² are strongly upward shifted" in noise_uncertainty_warning()
    print("Analysis checks passed: sample-SD, no survivor means, complete noise windows, undefined-denominator handling, nonfinite evidence, report-only Phase-1 CI guard without measurement mutation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        generate(args.root, args.allow_partial)
