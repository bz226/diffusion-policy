#!/usr/bin/env python
"""Read-only AdaGrad-Norm analysis of existing results; Python 3.8 compatible.

Adapted from the approved E1CG analysis's loading, aggregation and plotting
conventions. Never imports training code or launches jobs. Incomplete seeds
remain explicit; final artifacts are published only when evidence is complete.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np


PRIMARY = ("AN1", "AN3", "AN10")
CONDITIONS = PRIMARY + ("E1CG", "E1C", "SGD3", "SGD10", "E2")
FACTORS = {"AN1": 1, "AN3": 3, "AN10": 10}
LABELS = {"AN1": "AdaGrad-Norm, η = d*", "AN3": "AdaGrad-Norm, η = 3d*",
          "AN10": "AdaGrad-Norm, η = 10d*", "E1CG": "AdamW (lr 1e-4)",
          "E1C": "AdamW, no γ^k (reference)", "SGD3": "SGD, 3η* (LOO reference)",
          "SGD10": "SGD, 10η* (LOO reference)", "E2": "AdamW, paper-form LOO (reference)"}
COLORS = {"AN1": "#7CB7D8", "AN3": "#277DAE", "AN10": "#004D73", "E1CG": "#CC79A7"}
ANCHOR_J = 1369.0240467260296
KL_STAR = 0.000584517336086918
METRICS = ("eval_936M", "train_last10", "J_disc_gain_last10", "ckpt_train_return",
           "ckpt_train_J_disc", "ckpt_eval_return", "ckpt_eval_J_disc", "first_KL",
           "mean_KL", "final_theta_dist", "final_saturation")
PRETTY = {"eval_936M": "Eval at 9.36M", "train_last10": "Train last 10 updates",
          "J_disc_gain_last10": "J_disc gain, last 10 updates", "ckpt_train_return": "Final train-sampler return",
          "ckpt_train_J_disc": "Final train-sampler J_disc", "ckpt_eval_return": "Final eval-sampler return",
          "ckpt_eval_J_disc": "Final eval-sampler J_disc", "first_KL": "First-update KL",
          "mean_KL": "Mean KL/update", "final_theta_dist": "Final distance from θ₀",
          "final_saturation": "Final pre-update x̂₀ saturation"}


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def fmt(value, precision=5):
    return ("%." + str(precision) + "g") % value if finite(value) else "N/A"


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path.write_text(json.dumps(safe(data), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(safe(rows))


def updates_before(itr):
    return sum(index % 10 != 0 for index in range(int(itr)))


def average(rows, key):
    values = [row.get(key) for row in rows]
    return float(np.mean(values, dtype=np.float64)) if values and all(finite(v) for v in values) else None


def load_checkpoint(path):
    data = read_json(path)
    return data if data and data.get("status") == "completed" else None


def paths(root, condition, seed):
    stage3 = root.parent
    if condition in PRIMARY:
        return (root / (condition + "_seed%d" % seed),
                root / "evaluations" / ("final_%s_seed%d" % (condition, seed)) / "ckpt_eval.json")
    if condition in ("E1CG", "E1C"):
        directory = stage3 / condition.lower() / (condition + "_seed%d" % seed)
        checkpoint = stage3 / "e1cg/evaluations" / ("segments_%s_s%d_i139" % (condition, seed)) / "ckpt_eval.json"
        return directory, checkpoint
    return (stage3 / (condition + "_seed%d" % seed),
            stage3 / ("ckpt_%s_seed%d_seed5000" % (condition, seed)) / "ckpt_eval.json")


def load_run(condition, seed, directory, collapse_threshold):
    path = directory / "result.pkl"
    output = dict(condition=condition, method=LABELS[condition], seed=seed, run_dir=str(directory),
                  result_path=str(path), status="missing", rows=[], training=[], evaluation=[],
                  errors=[], collapsed=None, first_collapse_update=None)
    if not path.is_file():
        return output
    try:
        with path.open("rb") as stream:
            rows = pickle.load(stream)
    except (EOFError, OSError, pickle.UnpicklingError) as error:
        output.update(status="temporarily_unreadable", errors=[repr(error)])
        return output
    if not isinstance(rows, list):
        raise ValueError("Expected list of iteration records: " + str(path))
    itrs = [int(row["itr"]) for row in rows]
    if itrs != list(range(len(rows))) or len(rows) > 140:
        raise ValueError("Non-contiguous/overlength result: " + str(path))
    training, evaluation, errors = [], [], []
    for row in rows:
        itr = int(row["itr"])
        is_train = "train_episode_reward" in row
        if is_train != (itr % 10 != 0):
            raise ValueError("Unexpected update cadence: " + str(path))
        if int(row["step"]) != 80000 * (updates_before(itr) + int(is_train)):
            raise ValueError("Step count disagrees with update count: " + str(path))
        (training if is_train else evaluation).append(row)
        keys = ("train_episode_reward", "J_disc_train", "kl_true_per_action", "theta_dist", "x0_sat_frac_mean") if is_train else ("eval_episode_reward",)
        for key in keys:
            if not finite(row.get(key)):
                errors.append("itr%d missing/nonfinite %s" % (itr, key))
        if condition in PRIMARY:
            for key in ("actor_lr", "adagrad_b", "adagrad_effective_lr", "adagrad_actor_updates"):
                if not finite(row.get(key)):
                    errors.append("itr%d missing/nonfinite %s" % (itr, key))
            expected_updates = updates_before(itr) + int(is_train)
            if finite(row.get("adagrad_actor_updates")) and row["adagrad_actor_updates"] != expected_updates:
                errors.append("itr%d optimizer update counter mismatch" % itr)
    windows = []
    for end in range(9, len(training)):
        value = average(training[end - 9:end + 1], "train_episode_reward")
        if finite(value):
            windows.append((end + 1, int(training[end]["itr"]), value))
    below = [row for row in windows if row[2] < collapse_threshold]
    complete = len(rows) == 140 and (directory / "checkpoint/state_139.pt").is_file()
    status = "complete" if complete else "partial"
    if (directory / "failure.json").is_file():
        status = "failed"
    output.update(rows=rows, training=training, evaluation=evaluation, errors=errors,
                  status=status, last_itr=itrs[-1] if itrs else None, n_updates=len(training),
                  collapsed=bool(below) if windows else None,
                  first_collapse_update=below[0][0] if below else None,
                  first_collapse_itr=below[0][1] if below else None,
                  min_window10_train=min((r[2] for r in windows), default=None),
                  min_window10_at_update=min(windows, key=lambda r: r[2])[0] if windows else None,
                  numerical_failure=bool(errors), collapse_threshold=collapse_threshold)
    return output


def seed_metrics(run, checkpoint):
    keys = ("condition", "method", "seed", "run_dir", "status", "last_itr", "n_updates", "collapsed",
            "first_collapse_update", "first_collapse_itr", "min_window10_train", "min_window10_at_update", "numerical_failure")
    result = {key: run.get(key) for key in keys}
    result.update({key: None for key in METRICS})
    training = run["training"]
    if training:
        result["first_KL"] = training[0].get("kl_true_per_action")
        result["actor_lr"] = training[0].get("actor_lr")
    if run["status"] == "complete":
        evaluation = {int(row["itr"]): row for row in run["evaluation"]}
        result["eval_936M"] = evaluation[130]["eval_episode_reward"]
        result["train_last10"] = average(training[-10:], "train_episode_reward")
        discounted = average(training[-10:], "J_disc_train")
        result["J_disc_gain_last10"] = discounted - ANCHOR_J if finite(discounted) else None
        result["mean_KL"] = average(training, "kl_true_per_action")
        result["final_theta_dist"] = training[-1].get("theta_dist")
        result["final_saturation"] = training[-1].get("x0_sat_frac_mean")
    result["checkpoint_eval_path"] = str(checkpoint)
    data = load_checkpoint(checkpoint)
    result["checkpoint_eval_status"] = "complete" if data else "missing"
    if data:
        for sampler in ("train", "eval"):
            values = data.get("samplers", {}).get(sampler)
            if values:
                result["ckpt_" + sampler + "_return"] = values["undiscounted_return"]["mean"]
                result["ckpt_" + sampler + "_J_disc"] = values["J_disc"]["mean"]
                result["ckpt_" + sampler + "_episodes"] = values["undiscounted_return"]["n"]
    result["issues"] = "; ".join(run["errors"])
    return result


def summaries(seed_rows):
    output = []
    for condition in CONDITIONS:
        selected = sorted((r for r in seed_rows if r["condition"] == condition), key=lambda r: r["seed"])
        if [row["seed"] for row in selected] != [0, 1, 2]:
            raise ValueError("Each condition must retain all three seed records")
        row = dict(condition=condition, method=LABELS[condition])
        for key in METRICS:
            values = [r.get(key) for r in selected]
            valid = all(finite(value) for value in values)
            row[key + "_mean"] = float(np.mean(values)) if valid else None
            row[key + "_sd"] = float(np.std(values, ddof=1)) if valid else None
            row[key + "_per_seed"] = json.dumps(safe(values))
        for key in ("status", "collapsed", "numerical_failure", "first_collapse_update"):
            row[key + "_per_seed"] = json.dumps([r.get(key) for r in selected])
        row["complete_seeds"] = sum(r["status"] == "complete" for r in selected)
        row["final_evaluations"] = sum(r["checkpoint_eval_status"] == "complete" for r in selected)
        output.append(row)
    return output


def differences(summary):
    by = {row["condition"]: row for row in summary}
    output = []
    for reference in ("E1CG", "SGD3"):
        for condition in PRIMARY:
            for metric in METRICS:
                a, b = by[reference], by[condition]
                ma, mb, sa, sb = (a[metric + "_mean"], b[metric + "_mean"],
                                  a[metric + "_sd"], b[metric + "_sd"])
                valid = all(finite(value) for value in (ma, mb, sa, sb))
                pooled = math.sqrt((sa * sa + sb * sb) / 2) if valid else None
                delta = mb - ma if valid else None
                output.append(dict(reference=reference, condition=condition, metric=metric,
                                   difference=delta, standard_error=pooled * math.sqrt(2 / 3) if valid else None,
                                   pooled_sd=pooled, twice_pooled_sd=2 * pooled if valid else None,
                                   reading=("unresolved" if abs(delta) < 2 * pooled else "exceeds reading-rule threshold")
                                   if valid else "unavailable"))
    return output


def mechanism(runs, calibration):
    rows = []
    for run in runs:
        if run["seed"] != 0 or run["condition"] not in PRIMARY + ("E1CG", "SGD3", "SGD10"):
            continue
        training = run["training"]
        row = dict(condition=run["condition"], method=run["method"], updates=len(training), status=run["status"])
        if not training:
            rows.append(row)
            continue
        norms = np.asarray([r.get("update_norm", np.nan) for r in training], dtype=np.float64)
        kls = np.asarray([r.get("kl_true_per_action", np.nan) for r in training], dtype=np.float64)
        j = np.arange(1, len(training) + 1, dtype=np.float64)
        eta = training[0].get("actor_lr")
        row.update(eta=eta, first_KL=kls[0], mean_KL=float(np.mean(kls)),
                   first20_max_KL=float(np.max(kls[:20])), late10_mean_KL=float(np.mean(kls[-10:])),
                   final_theta_dist=training[-1].get("theta_dist"),
                   cumulative_step_length=float(norms.sum()), root_sum_squared_steps=float(np.linalg.norm(norms)),
                   rms_step_length=float(np.sqrt(np.mean(norms * norms))),
                   mean_step_length=float(norms.mean()), first_step_length=float(norms[0]),
                   positive_KL_updates=int(np.count_nonzero(kls > 0)))
        valid = np.isfinite(kls) & (kls > 0) & (j >= 2)
        row["KL_loglog_slope_after_first"] = float(np.polyfit(np.log(j[valid]), np.log(kls[valid]), 1)[0]) if valid.sum() >= 2 else None
        if run["condition"] in PRIMARY and finite(eta) and eta > 0:
            normalized = norms * np.sqrt(j) / eta
            predicted = KL_STAR * FACTORS[run["condition"]] ** 2
            harmonic = float(np.sum(1.0 / j))
            row.update(normalized_step_median=float(np.median(normalized)), normalized_step_min=float(normalized.min()),
                       normalized_step_max=float(normalized.max()), normalized_step_first=float(normalized[0]),
                       predicted_first_KL=predicted, first_KL_prediction_ratio=float(kls[0] / predicted),
                       median_scaled_KL_ratio=float(np.median(kls * j / predicted)),
                       harmonic_number=harmonic, noise_distance_prediction=eta * math.sqrt(harmonic),
                       final_distance_prediction_ratio=float(training[-1]["theta_dist"] / (eta * math.sqrt(harmonic))),
                       equivalent_constant_rms_step=eta * math.sqrt(harmonic / len(training)),
                       calibration_d_star=calibration.get("d_star"), b0=calibration.get("b0"),
                       final_b=training[-1].get("adagrad_b"))
        rows.append(row)
    return rows


def style(ax):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def save_figure(fig, directory, name, caption, plt):
    fig.savefig(directory / (name + ".png"), dpi=300, bbox_inches="tight", metadata={"Description": caption})
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight", metadata={"Subject": caption})
    (directory / (name + ".caption.md")).write_text(caption + "\n")
    plt.close(fig)


def figures(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    directory = output / "figs"
    directory.mkdir(exist_ok=True)
    by = {(r["condition"], r["seed"]): r for r in runs}
    quantities = (("eval_episode_reward", "Evaluation reward", "Episode return (raw reward)"),
                  ("train_episode_reward", "Training reward", "Episode return (raw reward)"),
                  ("J_disc_train", "Discounted training return", "J_disc (raw reward units)"))
    for three in (False, True):
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 8.1))
        drawn = []
        for condition in ("E1CG",) + PRIMARY:
            selected = [by[(condition, seed)] for seed in ((0, 1, 2) if three else (0,))]
            has_any = False
            for ax, (key, title, label) in zip(axes, quantities):
                series = [{int(r["itr"]): r[key] for r in run["rows"] if key in r and finite(r[key])} for run in selected]
                common = sorted(set.intersection(*(set(s) for s in series)))
                if common:
                    xx = [updates_before(itr) * .08 for itr in common]
                    yy = [float(np.mean([s[itr] for s in series])) for itr in common]
                    ax.plot(xx, yy, color=COLORS[condition], lw=1.0, marker="o", ms=2.1, label=LABELS[condition])
                    has_any = True
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("Training env steps (millions)")
                ax.set_ylabel(label)
                style(ax)
            if has_any:
                drawn.append(condition)
        fig.suptitle("Estimator: without_PPO_and_GAE + γ^k", fontsize=10, y=.997)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .972), frameon=False, fontsize=8, ncol=2)
        fig.tight_layout(h_pad=1.5, rect=(0, 0, 1, .91))
        caption = ("Mean over three seeds per optimizer; 80 episodes per point per seed. " if three else
                   "Seed 0 per optimizer; 80 episodes per point. ")
        caption += ("Estimator: without_PPO_and_GAE + γ^k; only the actor optimizer changes. "
                    "J_disc is the training batch's mean of Σ_k 0.99^k R_k in raw rewards. "
                    "Evaluation keeps the DDPM's noise at every denoising step except the last. "
                    "Points show estimates without uncertainty bands; seed variation is in an_table. ")
        if len(drawn) < 4 or any(by[(c, s)]["status"] != "complete" for c in PRIMARY for s in (range(3) if three else (0,))):
            caption += "This is an interim view: only available iterations common to the included seeds are shown."
        save_figure(fig, directory, "an_curves_3seed" if three else "an_curves_seed0", caption, plt)

    fig, axes = plt.subplots(3, 1, figsize=(4.8, 8.1))
    omitted_kl = 0
    for condition in ("E1CG",) + PRIMARY:
        training = by[(condition, 0)]["training"]
        if not training:
            continue
        xx = np.arange(1, len(training) + 1)
        eta = training[0].get("actor_lr")
        if condition in PRIMARY and finite(eta) and eta > 0:
            yy = [r.get("update_norm", np.nan) * math.sqrt(j) / eta for j, r in zip(xx, training)]
            axes[0].plot(xx, yy, color=COLORS[condition], marker="o", ms=2.1, lw=1.0, label=LABELS[condition])
        for ax, key in ((axes[1], "kl_true_per_action"), (axes[2], "theta_dist")):
            points = [(j, r.get(key)) for j, r in zip(xx, training) if finite(r.get(key)) and (key != "kl_true_per_action" or r[key] > 0)]
            if key == "kl_true_per_action":
                omitted_kl += len(training) - len(points)
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], color=COLORS[condition], marker="o", ms=2.1, lw=1.0, label=LABELS[condition])
    for ax, title, ylabel in zip(axes, ("Step-length schedule", "KL per update", "Distance from initialization"),
                                ("‖Δθⱼ‖ √(j+1) / η (unitless)", "KL per action (nats)", "‖θ − θ₀‖ (parameter L2 units)")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Actor update number")
        ax.set_ylabel(ylabel)
        style(ax)
    axes[1].set_yscale("log")
    fig.suptitle("Estimator: without_PPO_and_GAE + γ^k", fontsize=10, y=.997)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .972), frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 1, .91))
    caption = ("Seed 0: normalized AdaGrad-Norm step length, per-action KL and distance from initialization, against actor update number. "
               "The formula indexes j from zero; the first applied update is shown at 1. "
               "The step-length panel includes only AdaGrad-Norm; the other panels include AdamW. "
               "Filled circles are point estimates, without uncertainty intervals; no smoothing is applied.")
    if omitted_kl:
        caption += " %d nonpositive or unavailable KL points cannot be shown on the log axis." % omitted_kl
    save_figure(fig, directory, "an_steps", caption, plt)


def tables(summary, contrasts, mechanism_rows, output):
    write_csv(output / "an_table.csv", summary)
    write_csv(output / "an_differences.csv", contrasts)
    write_csv(output / "mechanism_seed0.csv", mechanism_rows)
    lines = ["# Optimizer outcomes", "", "Each metric is mean ± sample SD over seeds 0, 1 and 2, followed by all three seed values. "
             "A mean requires three finite values; missing or failed runs are retained. Main rows share the estimator without_PPO_and_GAE + γ^k. "
             "The reference rows change the estimator as stated; they do not isolate optimizer geometry.", "",
             "| Optimizer / reference | " + " | ".join(PRETTY[k] for k in METRICS) + " | Collapse by seed | Completion |",
             "|---|" + "---|" * (len(METRICS) + 2)]
    for row in summary:
        cells = [row["method"]]
        for key in METRICS:
            values = json.loads(row[key + "_per_seed"])
            cells.append("%s ± %s (%s)" % (fmt(row[key + "_mean"]), fmt(row[key + "_sd"]), ", ".join(fmt(v) for v in values)))
        cells.extend([row["collapsed_per_seed"], "%d/3 training; %d/3 final evaluations" % (row["complete_seeds"], row["final_evaluations"])])
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "J_disc gain subtracts 1369.0240467260296, the existing pretrained train-sampler mean; its uncertainty is not included in seed SD. "
              "A run collapses when any ten consecutive training-update returns average below the pretrained train-sampler mean minus 300. "
              "The threshold and first crossing are in seed_metrics.csv; non-finite values and crashes are recorded separately. "
              "Final saturation is measured on the last pre-update rollout, while distance is measured after the final update. "
              "New and AdamW final checkpoint evaluations use 160 episodes per sampler; LOO references retain their saved 80-episode evaluations.", "",
              "## Differences", "", "Each difference is condition minus reference, with SE = pooled seed SD × √(2/3). "
              "Differences smaller than twice pooled SD are called unresolved.", "",
              "| Difference | Metric | Difference ± SE | Twice pooled SD | Reading |", "|---|---|---:|---:|---|"]
    for row in contrasts:
        lines.append("| %s − %s | %s | %s ± %s | %s | %s |" %
                     (LABELS[row["condition"]], LABELS[row["reference"]], PRETTY[row["metric"]], fmt(row["difference"]),
                      fmt(row["standard_error"]), fmt(row["twice_pooled_sd"]), row["reading"]))
    (output / "an_table.md").write_text("\n".join(lines) + "\n")


def interim(runs, output, target):
    if target % 10:
        raise ValueError("Interim target must be an evaluation iteration")
    train_itr = target - 1 if target else 1
    rows = []
    for run in runs:
        if run["seed"] != 0 or run["condition"] not in PRIMARY + ("E1CG",):
            continue
        logs = {int(row["itr"]): row for row in run["rows"]}
        row = dict(condition=run["condition"], method=run["method"], seed=0, evaluation_itr=target,
                   training_itr=train_itr, evaluation=logs.get(target, {}).get("eval_episode_reward"),
                   training_return=logs.get(train_itr, {}).get("train_episode_reward"),
                   J_disc_train=logs.get(train_itr, {}).get("J_disc_train"),
                   KL=logs.get(train_itr, {}).get("kl_true_per_action"), run_status=run["status"])
        rows.append(row)
    write_csv(output / "interim_comparison.csv", rows)
    lines = ["# Matched-iteration interim comparison", "", "Seed 0 per optimizer, estimator without_PPO_and_GAE + γ^k. "
             "Evaluation is at iteration %d; training metrics are from iteration %d, the immediately preceding update. "
             "Values are from 80 episodes; missing values remain N/A." % (target, train_itr), "",
             "| Optimizer | Evaluation | Training return | J_disc | KL/update | Status |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (row["method"], fmt(row["evaluation"]),
                     fmt(row["training_return"]), fmt(row["J_disc_train"]), fmt(row["KL"]), row["run_status"]))
    (output / "INTERIM.md").write_text("\n".join(lines) + "\n")


def contrast_text(contrasts, condition, reference, metric):
    row = next(r for r in contrasts if r["condition"] == condition and r["reference"] == reference and r["metric"] == metric)
    return "%s %s ± %s SE (%s)" % (PRETTY[metric], fmt(row["difference"]), fmt(row["standard_error"]), row["reading"])


def report(root, output, summary, contrasts, mechanism_rows, seed_rows, record):
    by = {r["condition"]: r for r in summary}
    complete = not record["missing"]
    calibration = record["calibration"]
    relative = lambda path: Path(os.path.relpath(str(path), str(output))).as_posix()
    lines = ["# AdaGrad-Norm on the discounted critic-baseline estimator", "",
             "Can a single-scalar adaptive step train without_PPO_and_GAE + γ^k stably, and how does it compare with AdamW and calibrated SGD?", "",
             "**Status.** " + ("All requested training runs and final checkpoint evaluations are complete." if complete else
                              "Partial evidence; final conclusions await the missing records listed below."), "",
             "**Q1 — Stability.**"]
    for condition in PRIMARY:
        row = by[condition]
        crossings = [r.get("first_collapse_update") for r in seed_rows if r["condition"] == condition]
        lines.append("- %s: J_disc gain %s ± %s (seed SD); collapse flags %s; first crossing updates %s." %
                     (LABELS[condition], fmt(row["J_disc_gain_last10_mean"]), fmt(row["J_disc_gain_last10_sd"]),
                      row["collapsed_per_seed"], json.dumps(crossings)))
    eligible = [c for c in PRIMARY if finite(by[c]["J_disc_gain_last10_mean"])]
    best = max(eligible, key=lambda c: by[c]["J_disc_gain_last10_mean"]) if eligible else None
    distance_text = []
    for reference in ("SGD3", "SGD10"):
        reference_distance = by[reference]["final_theta_dist_mean"]
        candidates = [c for c in PRIMARY if finite(by[c]["final_theta_dist_mean"]) and finite(reference_distance)]
        if candidates:
            closest = min(candidates, key=lambda c: abs(by[c]["final_theta_dist_mean"] - reference_distance))
            distance_text.append("%s has mean final distance %s; the closest adaptive condition is %s at %s" %
                                 (LABELS[reference], fmt(reference_distance), LABELS[closest], fmt(by[closest]["final_theta_dist_mean"])))
    lines += ["", "**Q2 — Calibrated SGD.** " +
              (("The largest observed mean gain in this fixed grid is %s. Against SGD at 3η*: " % LABELS[best]) +
               "; ".join(contrast_text(contrasts, best, "SGD3", key) for key in ("J_disc_gain_last10", "eval_936M", "train_last10", "final_theta_dist")) + ". "
               if best else "The grid is incomplete. ") +
              "The available SGD references use LOO and other estimator settings; this comparison cannot by itself isolate the step schedule. "
              "The adaptive grid also uses KL calibration, so its performance cannot establish calibration-free selection of η. "
              + ("; ".join(distance_text) + "." if distance_text else "Distance-matched comparisons are pending."), "",
              "**Q3 — AdamW.**"]
    for condition in PRIMARY:
        lines.append("- %s minus AdamW: %s; %s; %s." %
                     (LABELS[condition], contrast_text(contrasts, condition, "E1CG", "J_disc_gain_last10"),
                      contrast_text(contrasts, condition, "E1CG", "eval_936M"),
                      contrast_text(contrasts, condition, "E1CG", "train_last10")))
    reference_gap = []
    for reference in ("SGD3", "SGD10"):
        a, b = by["E1CG"], by[reference]
        means = (a["J_disc_gain_last10_mean"], b["J_disc_gain_last10_mean"])
        deviations = (a["J_disc_gain_last10_sd"], b["J_disc_gain_last10_sd"])
        if all(finite(value) for value in means + deviations):
            pooled = math.sqrt(sum(value * value for value in deviations) / 2)
            gap = means[1] - means[0]
            reference_gap.append("%s minus AdamW gives J_disc gain difference %s ± %s SE (%s)" %
                                 (LABELS[reference], fmt(gap), fmt(pooled * math.sqrt(2 / 3)),
                                  "unresolved" if abs(gap) < 2 * pooled else "exceeds reading-rule threshold"))
    lines += ["", ("; ".join(reference_gap) + ". " if reference_gap else "SGD/AdamW gap pending. ") +
              "These contextual SGD gaps are confounded by baseline, normalization and denoising-weight choices; "
              "agreement with an adaptive-optimizer gap would not alone establish geometry as its cause."]
    lines += ["", "**Q4 — Mechanism, seed 0.** Step normalization uses j+1 = 1,…,126. "
              "A KL log-log slope of −1 corresponds to the proposed inverse-update decay. The distance ratio uses η√H_N, a noise-dominated heuristic, not a bound.", "",
              "| Optimizer | Normalized step: median [min,max] | First KL / prediction | KL log-log slope | Final distance / η√H_N | RMS step length |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in mechanism_rows:
        if row["condition"] in PRIMARY:
            lines.append("| %s | %s [%s, %s] | %s | %s | %s | %s |" %
                         (row["method"], fmt(row.get("normalized_step_median")), fmt(row.get("normalized_step_min")),
                          fmt(row.get("normalized_step_max")), fmt(row.get("first_KL_prediction_ratio")),
                          fmt(row.get("KL_loglog_slope_after_first")), fmt(row.get("final_distance_prediction_ratio")),
                          fmt(row.get("rms_step_length"))))
    lines += ["", "For a noise-dominated 126-update walk, the equivalent constant RMS step is η√(H₁₂₆/126) ≈ 0.21η; "
              "this compares accumulated squared step lengths, not the total arc length.", "",
              "| Optimizer, seed 0 | First step length | Observed RMS step length | Equivalent constant RMS step prediction | Final distance |",
              "|---|---:|---:|---:|---:|"]
    for row in mechanism_rows:
        lines.append("| %s | %s | %s | %s | %s |" %
                     (row["method"], fmt(row.get("first_step_length")), fmt(row.get("rms_step_length")),
                      fmt(row.get("equivalent_constant_rms_step")), fmt(row.get("final_theta_dist"))))
    lines += ["", "[Full outcomes and differences](an_table.md) · [Seed-0 curves](figs/an_curves_seed0.png) · "
              "[Three-seed means](figs/an_curves_3seed.png) · [Steps, KL and distance](figs/an_steps.png) · "
              "[Seed metrics](seed_metrics.csv) · [Mechanism values](mechanism_seed0.csv).", "",
              "**Calibration.** KL* = %s; d* = %s; distance spread max/min = %s; median gradient norm = %s; b₀ = %s. "
              "The five batch step lengths are %s. Reported calibration warnings: %s." %
              (fmt(calibration.get("kl_target")), fmt(calibration.get("d_star")), fmt(calibration.get("spread")),
               fmt(calibration.get("median_gradient_norm")), fmt(calibration.get("b0")),
               ", ".join(fmt(v) for v in calibration.get("d_b", [])) or "pending",
               json.dumps(record.get("calibration_warnings", []))), "",
              "**Verification and provenance.** Gates: %s. Source revision `%s`, branch `%s`; exact reference revision "
              "`03f82611746954839169a64305777c3d7f91b139`. Saved commands, job names and seeds are in the linked plan; "
              "no existing results were modified by this analysis." %
              ("; ".join("%s %s" % (key, "passed" if record["gates"].get("checks", {}).get(key, {}).get("passed") else "pending/failed")
                         for key in (("G1", "G2", "G3a", "G3b", "G4", "G5") if record.get("amendment") else
                                     ("G1", "G2", "G3", "G4", "G5"))), record.get("execution", {}).get("revision", "pending"),
               record.get("execution", {}).get("branch", "stage3-e1cg-an")), "",
              "[Methods and scope](%s) · [Gate evidence](%s) · [Job file](%s) · [Exact job plan](%s) · "
              "[Calibration record](%s) · [Analysis inputs and omissions](analysis_record.json)." %
              (relative(root / "methods.md"), relative(Path(record["gate_path"])), relative(root / "jobs_e1cg_an.txt"),
               relative(root / "runs/plan_training.json"), relative(Path(record["calibration_record_path"]))), "",
              "**Limitations/deviations.** Reference SGD uses LOO because critic-baseline SGD remains paused. "
              "All seed records remain visible. Across-seed differences smaller than twice pooled SD are unresolved. "
              "New and AdamW checkpoint evaluations have 160 episodes per sampler; LOO reference evaluations retain 80. "
              "AdaGrad-Norm's theorem assumes an unbiased estimator; this normalized, denoising-weighted estimator has not been shown to meet that assumption.", "",
              "**Missing evidence:** " + ("; ".join(record["missing"]) or "none") + "."]
    if record.get("amendment"):
        previous = record.get("prior_failed_gate", {}).get("checks", {}).get("G3", {})
        revised = record.get("amended_G3", {}).get("checks", {})
        lines += ["", "**Approved numerical-check amendment.** The original G3 measured the rounded float32 parameter "
                  "displacement; its relative norm error was %s against a 1e-6 limit, so training stopped. "
                  "The user replaced that check with a float64 formula check and a float32 pre-add/update-rounding "
                  "check; both must pass, followed by the unchanged 12-iteration CPU smoke." % fmt(previous.get("relative_error")), "",
                  "The measured calibration remains d* = %s and b₀ = %s. The exact approved run values are "
                  "η = 0.00263955, 0.00791865 and 0.0263955, with b₀ = 0.00000322129; no calibration or passed "
                  "gate was rerun. [Original G3](%s) · [Amended G3a/G3b evidence](%s) · [Approval](%s)." %
                  (fmt(calibration.get("raw_calibrated_d_star")), fmt(calibration.get("raw_calibrated_b0")),
                   relative(root / "runs/gates/calibrated/gate_summary.json"),
                   relative(root / "runs/gates/amended_G3/gate_summary.json"),
                   relative(Path(calibration["amendment_path"]))), ""]
        for key in ("G3a", "G3b"):
            values = revised.get(key, {})
            selected = {name: value for name, value in values.items()
                        if any(term in name for term in ("relative_error", "relative_norm", "rounding", "pre_add"))
                        and isinstance(value, (int, float, bool))}
            if selected:
                lines.append("%s numerical evidence: %s." % (key, "; ".join("%s = %s" % (name.replace('_', ' '), fmt(value))
                                                                         for name, value in selected.items())))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def complete_bookkeeping(root, summary, record):
    """Update reader-facing records only after verified final publication."""
    if not record.get("final_requested") or record.get("status") != "complete" or record.get("missing"):
        raise ValueError("Completion bookkeeping requires a verified final analysis")
    required = [root / "REPORT.md", root / "methods.md", root / "an_table.csv", root / "an_table.md"]
    required += [root / "figs" / (name + suffix) for name in
                 ("an_curves_seed0", "an_curves_3seed", "an_steps") for suffix in (".png", ".pdf", ".caption.md")]
    if any(not path.is_file() for path in required):
        raise ValueError("Requested final artifacts must exist before marking the study complete")
    by = {row["condition"]: row for row in summary}
    if any(by[condition]["complete_seeds"] != 3 or by[condition]["final_evaluations"] != 3 for condition in PRIMARY):
        raise ValueError("All nine runs and final evaluations must complete before checklist insertion")

    project = root.parents[2]
    checklist = project / "complete_experiment.md"
    original = checklist.read_text()
    study_link = "./" + root.relative_to(project).as_posix() + "/"
    heading = "### Stage 3 AdaGrad-Norm on the discounted critic-baseline estimator"
    entry = (heading + "\n\n"
             "- Experiment folder: [AdaGrad-Norm experiment](" + study_link + ")\n"
             "- Experiment design: [Methods and execution protocol](" + study_link + "methods.md)\n"
             "- Curated: —\n\n")
    marker = "## Comparison analyses\n"
    if heading not in original and study_link not in original:
        if original.count(marker) != 1:
            raise ValueError("Cannot locate unique completed-experiments insertion point")
        revised = original.replace(marker, entry + marker, 1)
    else:
        revised = original  # Preserve an existing entry, including later curation links.

    calibration = record["calibration"]
    outcomes = []
    for condition in PRIMARY:
        row = by[condition]
        collapsed = json.loads(row["collapsed_per_seed"])
        outcomes.append("%s: %s ± %s J_disc gain; %d/3 collapsed" %
                        (LABELS[condition], fmt(row["J_disc_gain_last10_mean"]), fmt(row["J_disc_gain_last10_sd"]),
                         sum(value is True for value in collapsed)))
    text = ("# AdaGrad-Norm on the discounted Monte Carlo critic estimator\n\n"
            "**Question.** Does a single-scalar adaptive step train `without_PPO_and_GAE + γ^k` stably, "
            "and how does it compare with AdamW and the completed SGD references?\n\n"
            "**Status: completed.** All nine training runs reached 140 iterations (126 actor updates), and all nine "
            "final checkpoints were evaluated with both samplers. G1–G5 passed. "
            "The calibrated d* is %s and b₀ is %s.\n\n" % (fmt(calibration["d_star"]), fmt(calibration["b0"])) +
            "**Results.** " + "; ".join(outcomes) + ". Values are means ± sample SD across three seeds; "
            "the [report](REPORT.md) gives the comparisons and limitations.\n\n"
            "[Methods](methods.md) · [Outcome table](an_table.md) · [Seed-0 curves](figs/an_curves_seed0.png) · "
            "[Three-seed curves](figs/an_curves_3seed.png) · [Steps, KL and distance](figs/an_steps.png) · "
            "[Verified analysis inputs](analysis_record.json).\n\n"
            "**Next decision.** Review the artifacts and choose what to curate. No artifacts were automatically "
            "curated and no further sampling is scheduled.\n")
    (root / "experiment.md").write_text(text)
    if revised != original:
        if checklist.read_text() != original:
            raise RuntimeError("Completed-experiment checklist changed during finalization; preserve it and retry bookkeeping")
        temporary = checklist.with_name(checklist.name + ".adagrad-final.tmp")
        temporary.write_text(revised)
        temporary.replace(checklist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--final", action="store_true")
    modes.add_argument("--interim", action="store_true")
    parser.add_argument("--target-itr", type=int, default=70)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    initial_path = root.parent / "ckpt_theta0_seed5000/ckpt_eval.json"
    initial = load_checkpoint(initial_path)
    if not initial:
        raise ValueError("Required saved pretrained checkpoint evaluation is missing")
    anchor = initial["samplers"]["train"]
    if anchor["J_disc"]["mean"] != ANCHOR_J:
        raise ValueError("Saved discounted-return anchor differs from the approved reference")
    threshold = float(anchor["undiscounted_return"]["mean"]) - 300.0
    if updates_before(130) * .08 != 9.36:
        raise AssertionError("Final evaluation x mapping changed")
    plan = read_json(root / "runs/plan_training.json", {})
    calibration_record_path = (root / "runs/calibration_derived_amended.json"
                               if (root / "runs/calibration_derived_amended.json").exists()
                               else root / "runs/calibration_derived.json")
    calibration = plan.get("calibration") or read_json(calibration_record_path, {})
    campaign = read_json(root / "runs/campaign_training/campaign.json", {})
    gate_paths = [Path(check["path"]) for check in plan.get("global_checks", []) if "path" in check]
    gate_paths += [root / "runs/gates/amended_complete/gate_summary.json",
                   root / "runs/gates/calibrated/gate_summary.json", root / "runs/gates/gate_summary.json"]
    gate_path = next((path for path in gate_paths if path.is_file()), gate_paths[0])
    gates = read_json(gate_path, {})
    execution = read_json(root / "provenance/execution_revision.json", plan.get("source_provenance", {}))
    runs, seed_rows, missing = [], [], []
    for condition in CONDITIONS:
        for seed in range(3):
            directory, checkpoint = paths(root, condition, seed)
            run = load_run(condition, seed, directory, threshold)
            runs.append(run)
            seed_rows.append(seed_metrics(run, checkpoint))
            if run["status"] != "complete":
                missing.append("%s seed%d training %s" % (condition, seed, run["status"]))
            if run["errors"]:
                missing.append("%s seed%d numerical/logging issues: %s" % (condition, seed, "; ".join(run["errors"][:5])))
            if not load_checkpoint(checkpoint):
                missing.append("%s seed%d final checkpoint evaluation" % (condition, seed))
    if not gates.get("passed"):
        missing.append("G1–G5 not recorded passed")
    if not calibration or not all(finite(calibration.get(k)) for k in ("d_star", "b0", "kl_target", "median_gradient_norm")):
        missing.append("complete calibration-derived record")
    calibration_raw = read_json(calibration.get("calibration_path") or root / "cal_seed3000/calibration.json", {})
    warning_list = calibration_raw.get("warnings", []) + [warning for b in calibration_raw.get("batches", []) for warning in b.get("warnings", [])]
    created = datetime.now(timezone.utc)
    output = args.output.resolve() if args.output else root / "candidates" / (("final_" if args.final else "interim_") + created.strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    summary = summaries(seed_rows)
    contrasts = differences(summary)
    mechanism_rows = mechanism(runs, calibration)
    record = dict(created_utc=created.isoformat(), command=sys.argv, final_requested=args.final,
                  status="complete" if not missing else "partial", missing=missing,
                  code=str(Path(__file__).resolve()), gates=gates, gate_path=str(gate_path), calibration=calibration,
                  calibration_record_path=str(calibration_record_path),
                  amendment=read_json(calibration.get("amendment_path"), {}) if calibration.get("amendment_path") else {},
                  prior_failed_gate=read_json(root / "runs/gates/calibrated/gate_summary.json", {}),
                  amended_G3=read_json(root / "runs/gates/amended_G3/gate_summary.json", {}),
                  calibration_warnings=warning_list, execution=execution, training_plan=plan,
                  campaign=campaign, reference_path=str(initial_path), reference_J_disc=ANCHOR_J,
                  reference_train_return=anchor["undiscounted_return"]["mean"], collapse_threshold=threshold,
                  runs=[{k: v for k, v in run.items() if k not in ("rows", "training", "evaluation")} for run in runs])
    write_json(output / "analysis_record.json", record)
    write_csv(output / "seed_metrics.csv", seed_rows)
    interim(runs, output, args.target_itr)
    tables(summary, contrasts, mechanism_rows, output)
    figures(runs, output)
    report(root, output, summary, contrasts, mechanism_rows, seed_rows, record)
    if args.final and not missing:
        for name in ("an_table.csv", "an_table.md", "an_differences.csv", "seed_metrics.csv", "mechanism_seed0.csv", "analysis_record.json"):
            shutil.copy2(output / name, root / name)
        (root / "figs").mkdir(exist_ok=True)
        for path in (output / "figs").iterdir():
            shutil.copy2(path, root / "figs" / path.name)
        report(root, root, summary, contrasts, mechanism_rows, seed_rows, record)
        complete_bookkeeping(root, summary, record)
    print(json.dumps({"status": record["status"], "output": str(output), "missing": missing}, sort_keys=True))
    if args.final and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
