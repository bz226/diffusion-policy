#!/usr/bin/env python
"""Clean figures v5. Read saved evidence only; Python 3.8 compatible."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from clean_v5_rewards import load_reward_data
from clean_v5_stats import build_statistics

POINTS = ("W0_000", "W0_035", "W0_070", "W0_105", "W0_139")
ITRS = (0, 35, 70, 105, 139)
UPDATES = (0, 32, 63, 95, 126)
METHODS = (("DPPO", "#0072B2"), ("without_PPO", "#D55E00"),
           ("without_PPO_and_GAE", "#009E73"))
GREEN = "#009E73"
GRAY = "#383838"
UNAVAILABLE = {"value": None, "lo": None, "hi": None, "se": None, "status": "unavailable"}


def finite(x):
    return x is not None and math.isfinite(float(x))


def fmt(x, digits=5):
    return ("%." + str(digits) + "g") % x if finite(x) else "N/A"


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError("Missing estimates must be explicit, not NaN/inf")
    return x


def metric_cell(m):
    if finite(m.get("lower_bound")) and not finite(m.get("value")):
        return "≥%s" % fmt(m["lower_bound"])
    if not finite(m.get("value")):
        return "N/A (upper %s)" % fmt(m["upper_bound"]) if finite(m.get("upper_bound")) else "N/A"
    if not finite(m.get("lo")) or not finite(m.get("hi")):
        return fmt(m["value"]) + " (interval unresolved)"
    return "%s [%s, %s]" % (fmt(m["value"]), fmt(m["lo"]), fmt(m["hi"]))


def add_metric(row, name, m):
    for key in ("value", "lo", "hi", "se", "status", "lower_bound", "upper_bound", "reason"):
        row[name if key == "value" else name + "_" + key] = m.get(key)


def style_axes(ax):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def save_figure(fig, output, name, caption, plt):
    fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight", metadata={"Description": caption})
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight", metadata={"Subject": caption, "Title": name})
    (output / (name + ".caption.md")).write_text(caption + "\n")
    plt.close(fig)


def plot_rewards(rewards, output, plt):
    for sampler, filename, ylabel in (("eval", "eval_reward", "Evaluation reward (episode return)"),
                                      ("train", "train_reward", "Training reward (episode return)")):
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        start = rewards["pooled_start"][sampler]
        for method, color in METHODS:
            rows = rewards["curves"][method][sampler]
            if not rows or rows[0]["update"] != 0:
                raise ValueError("First rollout must use the pretrained actor")
            xx = [r["env_steps"] / 1e6 for r in rows]
            yy = [start["mean"]] + [r["reward"] for r in rows[1:]]
            ax.plot(xx, yy, color=color, lw=1.1, label=method)
            if sampler == "eval":
                if xx[-1] != 9.36:
                    raise ValueError("Final evaluation must be at 9.36M")
                ax.plot(xx[1:], yy[1:], color=color, marker="o", ms=3, lw=0)
        ax.errorbar(0, start["mean"], yerr=start.get("se"), color=GRAY,
                    marker="o", ms=4.5, lw=.9, capsize=2, zorder=5)
        style_axes(ax)
        ax.set_xlabel("Training env steps (millions)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-.22, 10.2)
        ax.set_xticks([0, 2.5, 5, 7.5, 10])
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        fig.tight_layout()
        caption = "One seed per method (seed 0), 80 episodes per point; the shared starting point pools 240 original episodes. "
        if sampler == "eval":
            caption += "Evaluation keeps the DDPM's noise at every denoising step except the last. "
        else:
            caption += "Training adds N(0, 0.1²) noise to every action at the last denoising step, so training returns sit below evaluation returns. "
        caption += "DPPO retains its ±3σ truncation of this nominal Gaussian noise. The pooled starting-point standard error is unavailable because original episode spreads were not saved."
        save_figure(fig, output, filename, caption, plt)


def error_point(ax, x, metric):
    value = metric.get("value")
    if not finite(value) or value <= 0:
        bound = metric.get("lower_bound")
        if finite(bound) and bound > 0:
            ax.plot(x, bound, marker="^", color=GREEN, mfc="none", ms=5, ls="none")
        return
    ax.plot(x, value, marker="o", color=GREEN, ms=4, ls="none", zorder=4)
    lo, hi = metric.get("lo"), metric.get("hi")
    if finite(lo) and finite(hi) and hi > 0:
        # A nonpositive endpoint cannot appear on log axes. Its full interval
        # remains in the table; there is no invented finite lower endpoint.
        bottom = lo if lo > 0 else value
        ax.vlines(x, bottom, hi, color=GREEN, lw=.9)
        ax.hlines(hi, x - 1.4, x + 1.4, color=GREEN, lw=.9)
        if lo > 0:
            ax.hlines(lo, x - 1.4, x + 1.4, color=GREEN, lw=.9)


def plot_noise(stats, variant, output, plt):
    if not stats["points"]:
        return False
    is_run = variant == "V2"
    filename = "gradient_noise_as_run" if is_run else "gradient_noise_paper_form"
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.15))
    panels = (("u_norm", "Mean gradient norm", "Gradient norm\n(training-loss units)" if is_run else "Gradient norm (paper units)"),
              ("trace_ep", "Gradient variance\n(per episode)", "Variance trace\n(training-loss units²)" if is_run else "Variance trace (paper units²)"),
              ("B_ep", "Relative variance\n(per episode)", "Relative variance = B (episodes)"))
    for ax, (field, title, ylabel) in zip(axes, panels):
        ys = []
        for name, x in zip(POINTS, UPDATES):
            p = stats["points"].get(name)
            metric = p["estimators"][variant][field] if p else UNAVAILABLE
            ys.append(metric["value"] if finite(metric.get("value")) and metric["value"] > 0 else np.nan)
            error_point(ax, x, metric)
        ax.plot(UPDATES, ys, color=GREEN, lw=1.2)
        ax.set_yscale("log")
        style_axes(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Updates (one update = 80 episodes)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-5, 132)
        ax.set_xticks(UPDATES)
        ax.margins(y=.22)
    fig.suptitle(("As run" if is_run else "Paper’s form") + "\neach point: m = 40,000 episodes (500 batches of 80)", fontsize=11, y=.99)
    fig.subplots_adjust(left=.085, right=.99, bottom=.21, top=.72, wspace=.62)
    norm = stats.get("normalization", stats.get("runmetadata", {}).get("normalization", "pending"))
    if is_run:
        estimator = "Raw Monte Carlo return minus the critic, normalized %s; mean coordinate log-probability and 0.99 denoising weights, without decision discount." % ("per quarter" if norm == "quarter" else "per batch")
    else:
        estimator = "Discounted raw Monte Carlo return minus the critic, summed coordinate log-probabilities, unit denoising weights and no normalization, in paper units."
    caption = (estimator + " The critic and reward scale at each point are the run's own; at update 0 the critic is untrained. "
               "The scale at update 126 uses the specified least-squares fallback. "
               "One update averages 80 episodes, so its relative variance is B/80. "
               "Error bars are 90% delete-one-group jackknife intervals with t(9) quantiles; intervals are conditional on the measured checkpoints (one training seed).")
    unresolved = any(p["estimators"][variant]["B_ep"].get("value") is None or
                     (finite(p["estimators"][variant]["u_norm"].get("lo")) and p["estimators"][variant]["u_norm"]["lo"] <= 0)
                     for p in stats["points"].values())
    if unresolved:
        caption += " Open triangles mark lower bounds where the signal is unresolved; nonpositive interval endpoints are omitted on log axes and retained in the table."
    pending = [str(x) for name, x in zip(POINTS, UPDATES) if name not in stats["points"]]
    if pending:
        caption += " Pending checkpoints (updates): " + ", ".join(pending) + "."
    save_figure(fig, output, filename, caption, plt)
    return True


def build_table(stats, rewards):
    rows = []
    for name, itr, update in zip(POINTS, ITRS, UPDATES):
        p = stats["points"].get(name)
        checkpoint = rewards["checkpoints"][str(itr)]
        row = {"method": "without_PPO_and_GAE", "update_number": update,
               "checkpoint_file": checkpoint["checkpoint_file"], "K": 500 if p else 0,
               "m": 40000 if p else 0, "groups": 10 if p else 0,
               "status": "measured" if p else "pending ten complete 50-batch jobs",
               "scale_fallback": itr == 139, "point": name}
        for variant, prefix in (("V2", "as_run"), ("V6", "paper_form")):
            for field in ("u_norm", "trace_ep", "B_ep", "B_update"):
                add_metric(row, prefix + "_" + field, p["estimators"][variant][field] if p else UNAVAILABLE)
        add_metric(row, "mean_direction_cosine", p["cos_V2_V6"] if p else UNAVAILABLE)
        add_metric(row, "critic_explained_variance_measured", p["explained_var"] if p else UNAVAILABLE)
        row["critic_explained_variance_logged"] = checkpoint["logged_explained_var"]["value"]
        row["logged_explained_variance_itr"] = checkpoint["logged_explained_var"]["itr"]
        divisor = p.get("divisor", {}) if p else {}
        if isinstance(divisor, (int, float)):
            divisor = {"value": divisor, "status": "fixed"}
        add_metric(row, "c", divisor if divisor else dict(UNAVAILABLE, value=checkpoint["divisor"]))
        source = p.get("divisor_source", {}) if p else {}
        row["c_source"] = source.get("source", checkpoint["divisor_status"]) if isinstance(source, dict) else str(source)
        row["normalization"] = p.get("normalization") if p else "pending check"
        row["selected_jobs"] = ";".join(j["run_id"] if isinstance(j, dict) else str(j) for j in p["selected_jobs"]) if p else ""
        rows.append(row)
    return rows


def write_tables(rows, output):
    with (output / "points_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Gradient noise of without_PPO_and_GAE", "",
             "Each row uses the same 500 batches (40,000 episodes, ten independent jobs) for both estimators. "
             "Brackets are 90% delete-one-group jackknife intervals using t(0.95,9)=1.833113, with joint deletions for ratios and cosines. "
             "Each estimator uses its own pairwise squared-mean estimate. Only B is comparable across their different units. "
             "A dagger marks every estimate affected by the fitted scale at update 126. Gradient intervals condition on that scale; c has its own five-fit-batch jackknife interval with t(4) quantiles. "
             "Bounds indicate an unresolved signal; they are upper-signal plug-in bounds, not two-sided confidence intervals.", "",
             "| Updates | Checkpoint | K | m | As run: ‖u‖ | Trace/episode | B | B/80 | Paper form: ‖u‖ | Trace/episode | B | B/80 | Mean-direction cosine | Critic EV measured | Critic EV logged | c and source |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    def cell(row, field):
        metric = {key: row.get(field if key == "value" else field + "_" + key)
                  for key in ("value", "lo", "hi", "se", "status", "lower_bound", "upper_bound")}
        return metric_cell(metric)
    for row in rows:
        mark = "†" if row["scale_fallback"] and row["K"] else ""
        values = [str(row["update_number"]), row["checkpoint_file"], str(row["K"]), format(row["m"], ",")]
        fields = [prefix + "_" + field for prefix in ("as_run", "paper_form")
                  for field in ("u_norm", "trace_ep", "B_ep", "B_update")]
        fields += ["mean_direction_cosine", "critic_explained_variance_measured"]
        values += [cell(row, field) + mark for field in fields]
        values += [fmt(row["critic_explained_variance_logged"])]
        c = cell(row, "c") if row["scale_fallback"] and row["K"] else fmt(row["c"])
        values += [c + mark + "; " + str(row["c_source"])]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "As-run norm/trace use training-loss units/squared units; paper-form norm/trace use paper units/squared units; B is in episodes. "
              "Exact logged c and logged EV are fixed reference values without sampling intervals. Measured EV uses DPPO's scaled-reward GAE targets. "
              "The final policy has no next training rollout, so its logged EV is unavailable. Selected jobs and normalization are in [render_record.json](render_record.json)."]
    (output / "points_table.md").write_text("\n".join(lines) + "\n")


def write_note(rewards, stats, rows, output):
    missing = [str(x) for name, x in zip(POINTS, UPDATES) if name not in stats["points"]]
    lines = ["# Clean figures, version 5", "",
             ("Reward figures are complete; gradient measurements are pending at updates " + ", ".join(missing) + ".") if missing else
             "All five checkpoints have 500 measured batches. Both gradient figures use exactly the same checkpoints, critics, scales and batches.", "",
             "Checkpoint state_0/35/70/105/139 contains 0/32/63/95/126 actor updates, verified from native counters and save order. "
             "The final evaluation is iteration130, 117 updates, 9.36M training env steps. DPPO uses the original Stage-1 seed0 run.", "",
             "Final seed0 evaluation matches each native log: " + "; ".join(m + " " + fmt(rewards["curves"][m]["eval"][-1]["reward"], 9) for m, _ in METHODS) + ". "
             "Pooled original evaluation/training means are %.6f/%.6f (240 episodes each). Episode spreads were not saved, so the requested starting-point standard errors are unavailable. " %
             (rewards["pooled_start"]["eval"]["mean"], rewards["pooled_start"]["train"]["mean"]) +
             "DPPO retains its original ±3σ truncation of nominal Gaussian noise.", "",
             "The critic has no warm-up. Scales come from the next training iteration's start log; the final scale uses the authorized through-origin least-squares fit over five fresh batches. "
             "The native initial c is 1.000000005 (nominal1 plus the scaler epsilon). All final-point estimates are marked †. "
             "Legacy V2 used clipped scaled returns; this collection explicitly uses raw Monte Carlo returns minus c times the frozen critic.", "",
             "CPU gates and normalization checks: [gate record](../runs/clean_v5_20261012/gate_cpu/) and "
             "[collection record](../runs/clean_v5_20261012/). F3 uses the approved identical-input old/new CPU replay with identical provenance. "
             "Each plotted point requires ten complete50-batch jobs. No unfinished groups or older warmed-critic data enter these figures.", "",
             "| Updates | Predicted consecutive-gradient cosine | Logged mean ± s.e. | Logged n | Measured critic EV | Logged critic EV |",
             "|---:|---:|---:|---:|---:|---:|"]
    for name, itr, update in zip(POINTS, ITRS, UPDATES):
        p = stats["points"].get(name)
        logged = rewards["checkpoints"][str(itr)]
        cosine = logged["grad_cos_prev_window"]
        mark = "†" if p and itr == 139 else ""
        lines.append("| %d | %s | %s ± %s | %d | %s | %s |" %
                     (update, metric_cell(p["predicted_consecutive_cos"] if p else UNAVAILABLE) + mark, fmt(cosine["mean"]),
                      fmt(cosine["sample_se"]), cosine["n"], metric_cell(p["explained_var"] if p else UNAVAILABLE) + mark,
                      fmt(logged["logged_explained_var"]["value"])))
    lines += ["", "The prediction is 1/(1+B_as_run/80). Log windows include ±10 pre-rollout update indices; endpoints are truncated to available data. "
              "The logged standard error is descriptive: successive gradients can be dependent and training changes the policy and critic. "
              "Fresh-batch EV uses the same GAE-target definition as training; update126 has no matching logged rollout.", "",
              "| Updates | K | m | Jobs used |", "|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| %d | %d | %d | %s |" % (row["update_number"], row["K"], row["m"], row["selected_jobs"] or "pending"))
    metadata = stats.get("runmetadata", {})
    gate = metadata.get("gate", {})
    comparison = metadata.get("normalization_check", {})
    if gate.get("passed"):
        lines += ["", "F1–F3 passed on CPU: checkpoint critic and supplied scales are exact; old/new final states and frozen-file bytes agree. "
                  "Native quarter-gradient relative L2 difference: " + fmt(gate.get("quarter_native_gradient", {}).get("relative_l2"), 8) + "."]
    if comparison.get("rows"):
        lines += ["", "Normalization selected **" + str(comparison.get("selected_normalization")) + "**: " +
                  "; ".join("batch%d cosine %s, quarter/full norm ratio %s" %
                            (r["batch"], fmt(r["cosine"], 9), fmt(r["quarter_over_full_norm"], 9)) for r in comparison["rows"]) + "."]
    lines += ["", "The superseded warmed-critic collection finished its already-running work: nine middle-checkpoint groups (450 batches) and two preparations. "
              "Those results remain separate and are not used here. No further jobs from that plan are launched. "
              "Earlier figures remain preserved in the figures history; curated artifacts are unchanged.", "",
              "[Evaluation](eval_reward.png) · [Training](train_reward.png) · [As run](gradient_noise_as_run.png) · "
              "[Paper form](gradient_noise_paper_form.png) · [Table](points_table.md). Captions accompany each figure."]
    (output / "NOTE.md").write_text("\n".join(lines) + "\n")


def publish(candidates, output):
    output.mkdir(parents=True, exist_ok=True)
    existing = [p for p in output.iterdir() if p.is_file()]
    if existing:
        archive = output / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive.mkdir(parents=True)
        for path in existing:
            shutil.copy2(str(path), str(archive / path.name))
    for path in candidates.iterdir():
        if path.is_file():
            shutil.copy2(str(path), str(output / path.name))


def main(args):
    root = args.root.resolve(); project = root.parents[2]
    rewards = load_reward_data(project)
    stats = build_statistics(root, analysis_source=args.analysis_source)
    if any(p["K"] != 500 or p["groups"] != 10 for p in stats["points"].values()):
        raise ValueError("Every rendered point must contain exactly10x50 complete batches")
    run = root / "runs" / ("clean_v5_render_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    candidates = run / "candidates"; candidates.mkdir(parents=True)
    sources = [Path(__file__)] + [Path(__file__).with_name(name) for name in
              ("clean_v5_stats.py", "clean_v5_rewards.py", "clean_v3_stats.py", "clean_v3_rewards.py")]
    for path in sources:
        shutil.copy2(str(path), str(run / path.name))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
    plot_rewards(rewards, candidates, plt)
    for variant in ("V2", "V6"):
        plot_noise(stats, variant, candidates, plt)
    rows = build_table(stats, rewards)
    write_tables(rows, candidates)
    write_note(rewards, stats, rows, candidates)
    missing = [name for name in POINTS if name not in stats["points"]]
    record = {"version": 5, "status": "interim" if missing else "complete", "utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "run": str(run), "rewards": rewards, "statistics": stats,
              "pending_points": missing, "curated": False, "simulation_launched_by_renderer": False,
              "parent_revision": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], universal_newlines=True).strip(),
              "render_source_snapshot": [str(run / path.name) for path in sources]}
    (candidates / "render_record.json").write_text(json.dumps(json_safe(record), indent=2, allow_nan=False) + "\n")
    output = args.output.resolve() if args.output else root / "figs_clean"
    if not args.candidates_only:
        publish(candidates, output)
    print(json.dumps({"status": record["status"], "output": str(output), "run": str(run), "pending_points": missing}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis-source", type=Path)
    parser.add_argument("--candidates-only", action="store_true")
    main(parser.parse_args())
