#!/usr/bin/env python
"""Read-only approved E1CG training/segment analysis, Python 3.8.

Does not import training code, launch jobs, modify results, or silently drop
failed/missing seeds. Candidate artifacts are regenerated from saved evidence.
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


LABELS = {"E1C": "without_PPO_and_GAE", "E1CG": "without_PPO_and_GAE + γ^k",
          "E1": "E1: MC + LOO", "E2": "E2: discounted MC + LOO",
          "E2C": "paper-form MC + critic"}
COLORS = {"E1C": "#009E73", "E1CG": "#CC79A7"}
METRICS = ("eval_936M", "train_last10", "J_disc_gain_last10", "ckpt_train_return",
           "ckpt_train_J_disc", "ckpt_eval_return", "ckpt_eval_J_disc", "mean_KL",
           "mean_grad_cos", "final_theta_dist", "final_saturation")
PRETTY = {"eval_936M": "Eval at 9.36M", "train_last10": "Train last 10",
          "J_disc_gain_last10": "J_disc gain last 10", "ckpt_train_return": "Final train sampler return",
          "ckpt_train_J_disc": "Final train sampler J_disc", "ckpt_eval_return": "Final eval sampler return",
          "ckpt_eval_J_disc": "Final eval sampler J_disc", "mean_KL": "Mean KL/update",
          "mean_grad_cos": "Mean consecutive-gradient cosine", "final_theta_dist": "Final distance",
          "final_saturation": "Final pre-update saturation"}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def f(value, digits=5):
    return ("%." + str(digits) + "g") % value if finite(value) else "N/A"


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def updates_before(itr):
    return sum(i % 10 != 0 for i in range(int(itr)))


def load_run(condition, seed, directory):
    path = directory / "result.pkl"
    info = dict(condition=condition, method=LABELS[condition], seed=seed,
                run_dir=str(directory), result_path=str(path), status="missing", rows=[], errors=[])
    if not path.is_file():
        return info
    try:
        with path.open("rb") as stream:
            rows = pickle.load(stream)
    except (EOFError, OSError, pickle.UnpicklingError) as error:
        info.update(status="temporarily_unreadable", errors=[str(error)])
        return info
    itrs = [int(row["itr"]) for row in rows]
    if itrs != list(range(len(rows))) or len(rows) > 140:
        raise ValueError("Non-contiguous or overlength training log: " + str(path))
    training = [row for row in rows if "train_episode_reward" in row]
    evaluation = [row for row in rows if "eval_episode_reward" in row]
    for row in rows:
        itr = int(row["itr"])
        is_train = "train_episode_reward" in row
        if is_train != (itr % 10 != 0):
            raise ValueError("Unexpected evaluation/update cadence: " + str(path))
        expected = 80000 * (updates_before(itr) + int(is_train))
        if int(row["step"]) != expected:
            raise ValueError("Step counter disagrees with rollout update count: " + str(path))
    errors = []
    for row in training:
        for key in ("train_episode_reward", "J_disc_train", "kl_true_per_action", "theta_dist", "x0_sat_frac_mean"):
            if not finite(row.get(key)):
                errors.append("itr%d:missing/nonfinite:%s" % (row["itr"], key))
        if row["itr"] != 1 and not finite(row.get("grad_cos_prev")):
            errors.append("itr%d:undefined/nonfinite:grad_cos_prev" % row["itr"])
    for row in evaluation:
        if not finite(row["eval_episode_reward"]):
            errors.append("itr%d:nonfinite:eval" % row["itr"])
    collapsed = bool(errors)
    if evaluation and finite(evaluation[0]["eval_episode_reward"]):
        collapsed = collapsed or any(finite(row["eval_episode_reward"]) and
                                      row["eval_episode_reward"] < .5 * evaluation[0]["eval_episode_reward"]
                                      for row in evaluation)
    complete = len(rows) == 140 and (directory / "checkpoint/state_139.pt").is_file()
    info.update(status="complete" if complete else "partial", rows=rows, training=training,
                evaluation=evaluation, errors=errors, collapsed=collapsed,
                last_itr=itrs[-1] if itrs else None, n_updates=len(training))
    return info


def average(rows, key):
    values = [row.get(key) for row in rows]
    return float(np.mean(values, dtype=np.float64)) if values and all(finite(v) for v in values) else None


def load_ckpt(path):
    data = read(path)
    if not data or data.get("status") != "completed":
        return None
    return data


def run_metrics(run, checkpoint, reference_J):
    row = {key: run.get(key) for key in ("condition", "method", "seed", "run_dir", "status", "collapsed", "last_itr")}
    row.update({key: None for key in METRICS})
    training = run.get("training", [])
    if run["status"] == "complete":
        evaluation = {int(r["itr"]): r for r in run["evaluation"]}
        row["eval_936M"] = evaluation[130]["eval_episode_reward"]
        if updates_before(130) * .08 != 9.36:
            raise ValueError("Final evaluation axis mapping failed")
        row["train_last10"] = average(training[-10:], "train_episode_reward")
        value = average(training[-10:], "J_disc_train")
        row["J_disc_gain_last10"] = value - reference_J if finite(value) else None
        row["mean_KL"] = average(training, "kl_true_per_action")
        row["mean_grad_cos"] = average(training[1:], "grad_cos_prev")
        row["final_theta_dist"] = training[-1].get("theta_dist")
        row["final_saturation"] = training[-1].get("x0_sat_frac_mean")
    row["checkpoint_eval_path"] = str(checkpoint) if checkpoint else None
    data = load_ckpt(checkpoint) if checkpoint else None
    if data:
        for sampler in ("train", "eval"):
            sampler_data = data["samplers"].get(sampler)
            if sampler_data:
                row["ckpt_" + sampler + "_return"] = sampler_data["undiscounted_return"]["mean"]
                row["ckpt_" + sampler + "_J_disc"] = sampler_data["J_disc"]["mean"]
                row["ckpt_" + sampler + "_episodes"] = sampler_data["undiscounted_return"]["n"]
    row["issues"] = ";".join(run["errors"])
    return row


def summarize(seed_rows, conditions):
    output = []
    for condition in conditions:
        selected = sorted([row for row in seed_rows if row["condition"] == condition], key=lambda row: row["seed"])
        if [row["seed"] for row in selected] != [0, 1, 2]:
            raise ValueError("All three seed records must be retained")
        row = {"condition": condition, "method": LABELS[condition]}
        for metric in METRICS:
            values = [r[metric] for r in selected]
            valid = all(finite(v) for v in values)
            row[metric + "_mean"] = float(np.mean(values)) if valid else None
            row[metric + "_sd"] = float(np.std(values, ddof=1)) if valid else None
            row[metric + "_per_seed"] = json.dumps(safe(values))
        row["run_status_per_seed"] = ";".join(r["status"] for r in selected)
        row["collapsed_per_seed"] = json.dumps([r.get("collapsed") for r in selected])
        output.append(row)
    return output


def differences(summary):
    by = {row["condition"]: row for row in summary}
    output = []
    for before, after in (("E1C", "E1CG"), ("E1", "E2")):
        for metric in METRICS:
            a, b = by[before], by[after]
            ma, mb, sa, sb = (a[metric + "_mean"], b[metric + "_mean"],
                              a[metric + "_sd"], b[metric + "_sd"])
            okay = all(finite(v) for v in (ma, mb, sa, sb))
            pooled = math.sqrt((sa ** 2 + sb ** 2) / 2) if okay else None
            delta = mb - ma if okay else None
            output.append(dict(before=before, after=after, metric=metric,
                               difference=delta, pooled_sd=pooled,
                               standard_error=pooled * math.sqrt(2 / 3) if okay else None,
                               twice_pooled_sd=2 * pooled if okay else None,
                               reading="unresolved" if okay and abs(delta) < 2 * pooled else
                               "exceeds reading-rule threshold" if okay else "unavailable"))
    return output


def segments(root, eval_jobs):
    initial = root / "evaluations/segments_theta0/ckpt_eval.json"
    baseline = load_ckpt(initial)
    base = baseline["samplers"]["train"].get("reward_segments") if baseline else None
    base_values = np.asarray([x["mean"] for x in base], dtype=np.float64) if base else None
    rows = []
    for job in eval_jobs:
        if job.get("condition") not in ("E1C", "E1CG") or "checkpoint_itr" not in job:
            continue
        condition, seed, itr = job["condition"], int(job["source_seed"]), int(job["checkpoint_itr"])
        path = Path(job["logdir"]) / "ckpt_eval.json"
        data = load_ckpt(path)
        segment = data["samplers"]["train"].get("reward_segments") if data else None
        row = dict(condition=condition, method=LABELS[condition], seed=seed, checkpoint_itr=itr,
                   updates=updates_before(itr) + int(itr % 10 != 0), source=str(path),
                   baseline_source=str(initial), status="complete" if base and segment else "missing",
                   tilt_T=None)
        for b in range(5):
            row["delta_bin%d" % b] = None
            row["bin%d_mean" % b] = None
        if base and segment:
            if len(segment) != 5 or any(int(x["n"]) != 160 for x in segment):
                raise ValueError("Checkpoint segments require 160 episodes per bin")
            if any(int(x["n"]) != 480 for x in base):
                raise ValueError("Pretrained segments require 480 episodes per bin")
            observed = np.asarray([x["mean"] for x in segment], dtype=np.float64)
            delta = observed - base_values
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite segment estimate")
            for b in range(5):
                row["delta_bin%d" % b] = float(delta[b])
                row["bin%d_mean" % b] = float(observed[b])
                row["bin%d_episode_se" % b] = segment[b]["se"]
                row["bin%d_theta0_mean" % b] = float(base_values[b])
            row["tilt_T"] = float(delta[:2].mean() - delta[3:].mean())
            row["episodes"] = 160
            # Use the stored per-episode bins to verify aggregation when present.
            episode_rows = [row for rollout in data.get("rollouts", []) if rollout["sampler"] == "train"
                            for row in rollout.get("episode_segment_rewards", [])]
            if episode_rows:
                values = np.asarray(episode_rows, dtype=np.float64)
                if values.shape != (160, 5) or not np.allclose(values.mean(axis=0), observed, rtol=1e-12, atol=1e-12):
                    raise ValueError("Segment means disagree with episode records")
        rows.append(row)
    return rows


def style(ax):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def savefig(fig, directory, name, caption, plt):
    fig.savefig(directory / (name + ".png"), dpi=300, bbox_inches="tight", metadata={"Description": caption})
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight", metadata={"Subject": caption})
    (directory / (name + ".caption.md")).write_text(caption + "\n")
    plt.close(fig)


def figures(runs, segment_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    plotdir = output / "figs"
    plotdir.mkdir(exist_ok=True)
    by = {(r["condition"], r["seed"]): r for r in runs}
    fields = (("eval_episode_reward", "Evaluation reward", "Episode return (raw reward)"),
              ("train_episode_reward", "Training reward", "Episode return (raw reward)"),
              ("J_disc_train", "Discounted training return", "J_disc (raw reward units)"))
    for three in (False, True):
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 8.0))
        drawn = False
        for condition in ("E1C", "E1CG"):
            seeds = (0, 1, 2) if three else (0,)
            selected = [by[(condition, seed)] for seed in seeds]
            for ax, (field, title, label) in zip(axes, fields):
                points = [{int(row["itr"]): row[field] for row in r["rows"] if field in row and finite(row[field])}
                          for r in selected]
                common = sorted(set.intersection(*(set(p) for p in points)))
                if common:
                    xx = [.08 * updates_before(itr) for itr in common]
                    yy = [float(np.mean([p[itr] for p in points])) for itr in common]
                    ax.plot(xx, yy, color=COLORS[condition], lw=1.0, marker="o", ms=2.2,
                            label=LABELS[condition])
                    drawn = True
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("Training env steps (millions)")
                ax.set_ylabel(label)
                style(ax)
        if drawn:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995), ncol=1, frameon=False, fontsize=8)
            fig.tight_layout(h_pad=1.6, rect=(0, 0, 1, .91))
            caption = ("Mean over three seeds per method; 80 episodes per point per seed. " if three else
                       "One seed per method, 80 episodes per point. ")
            caption += ("The two runs differ only in weighting decision k's advantage by γ^k. "
                        "J_disc is the training batch's mean of Σ_k 0.99^k R_k, using raw rewards. "
                        "Evaluation keeps the DDPM's noise at every denoising step except the last. "
                        "Only available common iterations are shown; uncertainty is not plotted.")
            savefig(fig, plotdir, "e1cg_curves_3seed" if three else "e1cg_curves_seed0", caption, plt)
        else:
            plt.close(fig)
    complete = {(r["condition"], r["seed"]): r for r in segment_rows if r["checkpoint_itr"] == 139 and r["status"] == "complete"}
    if all((condition, seed) in complete for condition in ("E1C", "E1CG") for seed in range(3)):
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 8.0))
        for seed, ax in enumerate(axes):
            for condition in ("E1C", "E1CG"):
                row = complete[(condition, seed)]
                ax.plot(range(5), [row["delta_bin%d" % b] for b in range(5)], color=COLORS[condition], marker="o", ms=3,
                        lw=1.1, label=LABELS[condition])
            ax.set_xticks(range(5))
            ax.set_xticklabels(("0–49", "50–99", "100–149", "150–199", "200–249"), rotation=30, ha="right", fontsize=7)
            ax.set_title("Seed %d" % seed)
            ax.set_xlabel("Episode segment (decisions)")
            ax.set_ylabel("Δ reward per decision (raw)")
            style(ax)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .995), frameon=False, fontsize=8)
        fig.tight_layout(h_pad=1.6, rect=(0, 0, 1, .91))
        caption = ("Change in mean reward per decision from the pretrained policy, by episode segment, at the final checkpoint; "
                   "training sampler, 160 episodes per point. γ^k puts half its weight on the first 61 decisions. "
                   "The shared pretrained reference uses 480 episodes. Points are estimates; uncertainty is not plotted.")
        savefig(fig, plotdir, "e1cg_gain_by_segment", caption, plt)


def tables(summary, contrasts, segment_rows, output):
    write_csv(output / "e1cg_table.csv", summary)
    write_csv(output / "e1cg_differences.csv", contrasts)
    write_csv(output / "segments_table.csv", segment_rows)
    lines = ["# Training and checkpoint outcomes", "",
             "Each cell is mean ± sample SD across seeds 0–2, followed by per-seed values. All three seeds are required for a cell; "
             "missing/failed seeds remain explicit. KL and consecutive-gradient cosine are means over training updates (only the first undefined cosine is omitted).", "",
             "| Method | " + " | ".join(PRETTY[m] for m in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for row in summary:
        cells = [row["method"]]
        for metric in METRICS:
            values = json.loads(row[metric + "_per_seed"])
            cells.append("%s ± %s (%s)" % (f(row[metric + "_mean"]), f(row[metric + "_sd"]), ", ".join(f(v) for v in values)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "J_disc gain subtracts the saved pretrained train-sampler mean 1369.0240467260296; its uncertainty is not included in training-seed SD. "
              "Distance is after the final update; saturation is from the final training rollout before that update. "
              "New final-checkpoint measurements use 160 episodes per sampler; existing LOO references retain their original 80-episode measurements.", "",
              "| Difference | Metric | Mean difference ± standard error | Twice pooled SD | Reading |", "|---|---|---:|---:|---|"]
    for row in contrasts:
        lines.append("| %s − %s | %s | %s ± %s | %s | %s |" % (LABELS[row["after"]], LABELS[row["before"]],
                     PRETTY[row["metric"]], f(row["difference"]), f(row["standard_error"]), f(row["twice_pooled_sd"]), row["reading"]))
    lines += ["", "SE = pooled sample SD × √(2/3); differences smaller than twice pooled seed SD are unresolved. "
              "No additional significance tests are applied."]
    (output / "e1cg_table.md").write_text("\n".join(lines) + "\n")
    lines = ["# Gains by episode segment", "",
             "Δ is the checkpoint's mean raw reward per decision minus the shared pretrained mean. T is mean Δ in the first two bins minus mean Δ in the last two. "
             "Each checkpoint has 160 training-sampler episodes; the common reference has 480. No seed or episode uncertainty is implied by the point estimates below.", "",
             "| Method | Seed | Checkpoint iteration | Updates | Δ 0–49 | Δ 50–99 | Δ 100–149 | Δ 150–199 | Δ 200–249 | T | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in segment_rows:
        lines.append("| " + " | ".join([row["method"], str(row["seed"]), str(row["checkpoint_itr"]), str(row["updates"])] +
                     [f(row["delta_bin%d" % b]) for b in range(5)] + [f(row["tilt_T"]), row["status"]]) + " |")
    (output / "segments_table.md").write_text("\n".join(lines) + "\n")


def comparison_text(contrasts, metric, before="E1C", after="E1CG"):
    row = next(r for r in contrasts if r["metric"] == metric and r["before"] == before and r["after"] == after)
    return "%s: %s ± %s (SE), %s" % (PRETTY[metric], f(row["difference"]), f(row["standard_error"]), row["reading"])


def interim_comparison(runs, output):
    """Compare the same rollout iteration, with no running-window substitution."""
    by = {(run["condition"], run["seed"]): run for run in runs}
    available = [r for r in runs if r["condition"] == "E1CG" and r.get("evaluation")]
    if not available:
        return
    target = min(70, min(max(int(row["itr"]) for row in r["evaluation"]) for r in available))
    target -= target % 10
    train_itr = target - 1 if target else 1
    rows = []
    for condition in ("E1C", "E1CG"):
        for seed in range(3):
            run = by[(condition, seed)]
            logs = {int(row["itr"]): row for row in run["rows"]}
            rows.append(dict(condition=condition, method=LABELS[condition], seed=seed,
                             evaluation_itr=target, training_itr=train_itr,
                             evaluation=logs.get(target, {}).get("eval_episode_reward"),
                             training_return=logs.get(train_itr, {}).get("train_episode_reward"),
                             J_disc_train=logs.get(train_itr, {}).get("J_disc_train"),
                             run_status=run["status"]))
    write_csv(output / "interim_comparison.csv", rows)
    lines = ["# Matched-iteration interim comparison", "",
             "Evaluation is at iteration %d; training return and discounted return are from iteration %d. "
             "Each row is one seed and 80 rollout episodes. Missing values remain N/A." % (target, train_itr), "",
             "| Method | Seed | Evaluation | Training return | J_disc_train | Run status |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| %s | %d | %s | %s | %s | %s |" %
                     (row["method"], row["seed"], f(row["evaluation"]), f(row["training_return"]),
                      f(row["J_disc_train"]), row["run_status"]))
    (output / "INTERIM.md").write_text("\n".join(lines) + "\n")



def prediction_text(root, contrasts):
    path = root / "alignment_table.csv"
    if not path.is_file():
        return "Part A alignment data are missing; no geometric prediction is available."
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get("pair") == "V2_V6" and row.get("own_critic") == "True"
                and row.get("path_method") == "without_PPO_and_GAE"]
    rows.sort(key=lambda row: int(row["updates"]))
    estimates, valid = [], []
    for row in rows:
        good = (row.get("status") == "resolved" and row.get("interpretation") != "sanity_failed"
                and row.get("sanity_status") != "failed")
        value = float(row["value"]) if good and row.get("value") else None
        valid.append(value)
        estimates.append("%s: %s" % (row["updates"], f(value, 4)))
    criterion = (len(rows) == 5 and all(finite(value) and value >= .9 for value in valid))
    text = "Own-critic c(V2,V6) point estimates (update: cosine) are " + "; ".join(estimates) + ". "
    text += ("All five point estimates meet the ≥0.9 retracing criterion. " if criterion else
             "The ≥0.9-at-every-checkpoint criterion is not met, so Part A does not give the stated retracing prediction. ")
    selected = [row for row in contrasts if row['before'] == 'E1C' and row['after'] == 'E1CG'
                and row['metric'] in ('J_disc_gain_last10', 'eval_936M', 'train_last10')]
    if all(row['reading'] != 'unavailable' for row in selected):
        resolved = [PRETTY[row['metric']] for row in selected if row['reading'] == 'exceeds reading-rule threshold']
        text += ("The learning comparison exceeds the reading-rule threshold for " + ", ".join(resolved) + ". "
                 if resolved else "None of the three primary learning contrasts exceeds the reading-rule threshold. ")
    text += "These noisy Euclidean cosines are indicators; normalization and AdamW preconditioning prevent an exact trajectory prediction."
    return text


def consistency_text(contrasts):
    out = []
    for metric in ('J_disc_gain_last10', 'eval_936M', 'train_last10'):
        selected = {row['before']: row for row in contrasts if row['metric'] == metric}
        primary, loo = selected['E1C'], selected['E1']
        if not finite(primary['difference']) or not finite(loo['difference']):
            out.append(PRETTY[metric] + ": primary comparison pending")
            continue
        same = primary['difference'] * loo['difference'] > 0
        relation = 'same nominal direction' if same else 'opposite nominal directions' if primary['difference'] * loo['difference'] < 0 else 'one contrast is zero'
        unresolved = primary['reading'] == 'unresolved' or loo['reading'] == 'unresolved'
        out.append(PRETTY[metric] + ": " + relation + ("; at least one contrast is unresolved" if unresolved else "; both exceed the reading-rule threshold"))
    return "; ".join(out) + "."


def report(root, output, summary, contrasts, segment_rows, record, final):
    gate_checks = record.get("gates", {}).get("checks", {})
    gate_text = "; ".join("%s %s" % (name, "passed" if gate_checks.get(name, {}).get("passed") else "pending/failed")
                          for name in ("G1", "G2", "G3", "G4", "G5", "G6"))
    lines = ["# Discounting decisions after advantage normalization", "",
             "Does γ^k change what without_PPO_and_GAE learns, and move improvement toward earlier decisions?", "",
             "**Status:** " + ("All required training and checkpoint evaluations are verified complete." if final and not record["missing"] else
                              "Partial analysis; incomplete evidence is listed explicitly and no final conclusion is made."), "",
             "**Q1 — Discounted objective.** " + comparison_text(contrasts, "J_disc_gain_last10") + ".", "",
             "**Q2 — Benchmark returns.** " + comparison_text(contrasts, "eval_936M") + "; " + comparison_text(contrasts, "train_last10") + ".", "",
             "**Q3 — Prediction.** " + prediction_text(root, contrasts), "",
             "**Q4 — Episode tilt.**"]
    for itr in (35, 70, 105, 139):
        values = {}
        for condition in ("E1C", "E1CG"):
            selected = sorted([r for r in segment_rows if r["condition"] == condition and r["checkpoint_itr"] == itr], key=lambda r: r["seed"])
            values[condition] = [r["tilt_T"] for r in selected]
        okay = all(len(v) == 3 and all(finite(x) for x in v) for v in values.values())
        if okay:
            a, b = np.asarray(values["E1C"]), np.asarray(values["E1CG"])
            pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            delta = float(b.mean() - a.mean())
            lines.append("- Checkpoint %d: T without γ^k %s ± %s; with γ^k %s ± %s (seed SD); difference %s ± %s (SE), %s." %
                         (itr, f(a.mean()), f(a.std(ddof=1)), f(b.mean()), f(b.std(ddof=1)), f(delta), f(pooled * math.sqrt(2/3)),
                          "unresolved" if abs(delta) < 2 * pooled else "exceeds reading-rule threshold"))
        else:
            lines.append("- Checkpoint %d: pending/incomplete." % itr)
    lines += ["", "**Q5 — LOO comparison.** " + "; ".join(comparison_text(contrasts, metric, "E1", "E2") for metric in
             ("J_disc_gain_last10", "eval_936M", "train_last10")) + ". "
             + consistency_text(contrasts) + " The LOO comparison also changes denoising weights, log-prob reduction and normalization, so it is not the same single intervention.", "",
             "**Other behavior.** " + "; ".join(comparison_text(contrasts, metric) for metric in
             ("mean_KL", "mean_grad_cos", "final_theta_dist", "final_saturation")) + ".", "",
             "[Outcome and difference table](e1cg_table.md) · [Episode segments](segments_table.md) · "
             "[Seed-0 curves](figs/e1cg_curves_seed0.png) · [Three-seed means](figs/e1cg_curves_3seed.png) · "
             "[Final segment gains](figs/e1cg_gain_by_segment.png).", "",
             "**Verification and provenance.** " + gate_text + ". [Source records](analysis_record.json) preserve every input, source revision, job and gate record. "
             "All three seeds are retained. Final checkpoint sampling is separate from the last logged evaluation at 9.36M. "
             "Part A records retain full-batch-proxy, fitted-scale and older scaled-return caveats. "
             "No new sampling, tuning or modifications to saved results were performed by this analysis.", "",
             "**Missing evidence:** " + ("; ".join(record["missing"]) or "none") + "."]
    relative = lambda path: Path(os.path.relpath(str(path), str(output))).as_posix()
    lines += ["", "**Part A.** [Alignment table](%s) · [Mean-gradient alignment figure](%s) · [Interim interpretation](%s)." %
              (relative(root / "alignment_table.md"), relative(root / "figs/e1cg_alignment.png"),
               relative(root / "runs/part_a/candidates/INTERIM.md"))]
    provenance = record.get("source_provenance") or {}
    campaign = record.get("campaign_summary") or {}
    nodes = sorted({str(row.get("hardware", {}).get("node")) for row in record.get("campaign_records", {}).values()
                    if row.get("hardware", {}).get("node")})
    lines += ["", "**Execution.** Source branch `%s`, revision `%s`; reference revision `b459412cfc272a7b8830c77925271843c27222b2`. "
              "Each training allocation requests one A6000 and 40 CPUs; observed nodes: %s. Completed allocation time is %s GPU-hours of the approved 32; "
              "the optional condition decision is `%s`. Existing jobs were left running." %
              (provenance.get("branch", "pending"), provenance.get("revision", "pending"), ", ".join(nodes) or "not yet allocated",
               f(campaign.get("actual_gpu_hours")), record.get("optional_decision", "pending")), "",
              "[Exact commands, seeds and job records](%s) · [Primary job file](%s) · [Optional job file](%s) · "
              "[Methods and deviations](%s) · [Gate evidence](%s)." %
              (relative(root / "runs/plan.json"), relative(root / "jobs_e1cg.txt"), relative(root / "jobs_e2c.txt"),
               relative(root / "methods.md"), relative(root / "runs/gates/gate_summary.json")), "",
              "The first gate attempt failed before scientific checks because D4RL's default cache was read-only. "
              "The retry used a study-local cache; this was an installation-path fix, with no algorithm change."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    project = root.parents[2]
    stage3 = project / "results/stage3"
    plan = read(root / "runs/plan.json", {})
    campaign = read(root / "runs/campaign/campaign.json", {})
    jobs = plan.get("jobs", [])
    states = campaign.get("jobs", {})
    gates = read(root / "runs/gates/gate_summary.json", {})
    optional = campaign.get("optional_decision") == "admit_three_seeds"
    conditions = ["E1C", "E1CG", "E1", "E2"] + (["E2C"] if optional else [])
    runs, seed_rows, missing = [], [], []
    reference_path = stage3 / "ckpt_theta0_seed5000/ckpt_eval.json"
    reference = read(reference_path)
    reference_J = reference["samplers"]["train"]["J_disc"]["mean"]
    for condition in conditions:
        for seed in range(3):
            if condition == "E1C":
                directory = stage3 / "e1c" / ("E1C_seed%d" % seed)
                checkpoint = root / "evaluations" / ("segments_E1C_s%d_i139" % seed) / "ckpt_eval.json"
            elif condition in ("E1", "E2"):
                directory = stage3 / (condition + "_seed%d" % seed)
                checkpoint = stage3 / ("ckpt_%s_seed%d_seed5000" % (condition, seed)) / "ckpt_eval.json"
            else:
                directory = root / (condition + "_seed%d" % seed)
                checkpoint = root / "evaluations" / (("segments_E1CG_s%d_i139" % seed) if condition == "E1CG" else
                                                       ("final_E2C_seed%d" % seed)) / "ckpt_eval.json"
            run = load_run(condition, seed, directory)
            runs.append(run)
            seed_rows.append(run_metrics(run, checkpoint, reference_J))
            if run["status"] != "complete":
                missing.append("%s seed%d training %s" % (condition, seed, run["status"]))
            if not load_ckpt(checkpoint):
                missing.append("%s seed%d final checkpoint evaluation" % (condition, seed))
    eval_jobs = [job for job in jobs if job.get("mode") == "ckpt_eval"]
    segment_rows = segments(root, eval_jobs)
    if len(segment_rows) != 24 or any(row["status"] != "complete" for row in segment_rows):
        missing.append("Part C segment evaluations incomplete (%d/24)" % sum(r["status"] == "complete" for r in segment_rows))
    if args.final:
        if not gates.get("passed"):
            missing.append("G1–G6 gate record not passed")
        for job in jobs:
            state = states.get(job["id"], {})
            if state.get("status") not in ("complete", "skipped"):
                missing.append("job not terminal-complete: " + job["id"])
    output = root / "runs" / ("final_analysis" if args.final else "interim_analysis") / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(seed_rows, conditions)
    contrasts = differences(summary)
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, missing=missing,
                  status="complete" if not missing else "partial", final_requested=args.final,
                  code=str(Path(__file__).resolve()), source_provenance=plan.get("source_provenance"),
                  optional_decision=campaign.get("optional_decision"), job_records=jobs,
                  campaign_records=states, gates=gates,
                  campaign_summary={key: campaign.get(key) for key in ('actual_gpu_hours', 'reserved_gpu_hours', 'started_utc', 'ended_utc', 'controller_job_id')},
                  reference_J_disc=dict(path=str(reference_path), value=reference_J),
                  training_runs=[{k: v for k, v in r.items() if k not in ("rows", "training", "evaluation")} for r in runs])
    (output / "analysis_record.json").write_text(json.dumps(safe(record), indent=2, allow_nan=False) + "\n")
    write_csv(output / "seed_metrics.csv", seed_rows)
    interim_comparison(runs, output)
    tables(summary, contrasts, segment_rows, output)
    figures(runs, segment_rows, output)
    report(root, output, summary, contrasts, segment_rows, record, args.final)
    # Requested top-level artifacts are published only after full verification.
    if args.final and not missing:
        for name in ("e1cg_table.csv", "e1cg_table.md", "e1cg_differences.csv", "segments_table.csv", "segments_table.md", "analysis_record.json"):
            shutil.copy2(output / name, root / name)
        (root / "figs").mkdir(exist_ok=True)
        for path in (output / "figs").iterdir():
            shutil.copy2(path, root / "figs" / path.name)
        report(root, root, summary, contrasts, segment_rows, record, args.final)
    print(json.dumps({"status": record["status"], "output": str(output), "missing": missing}, sort_keys=True))
    if args.final and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
