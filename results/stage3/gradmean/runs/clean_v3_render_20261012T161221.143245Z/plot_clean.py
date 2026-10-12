#!/usr/bin/env python
"""Approved clean figures v3; read-only analysis, compatible with Python 3.8.

No simulations are launched. Earlier renders are preserved before publishing
the stable requested filenames. Curated artifacts are never touched.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from clean_v3_rewards import load_reward_data
from clean_v3_stats import build_statistics

POINTS = ("P0", "D1", "D2", "N1", "N2", "C1", "C2")
PATHS = (("DPPO", ("P0", "D1", "D2"), "#0072B2", -2., "V1"),
         ("without_PPO", ("P0", "N1", "N2"), "#D55E00", 0., "V1"),
         ("without_PPO_and_GAE", ("P0", "C1", "C2"), "#009E73", 2., "V2"))
GRAY = "#383838"
FIGURES = ("eval_reward", "train_reward", "gradient_noise_V6", "gradient_noise_V4", "gradient_noise_V5")
UNAVAILABLE = {"value": None, "lo": None, "hi": None, "se": None, "status": "unavailable"}


def finite(x):
    return x is not None and math.isfinite(float(x))


def fmt(x, digits=3):
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
        return "≥%s (signal unresolved)" % fmt(m["lower_bound"])
    if not finite(m.get("value")):
        if finite(m.get("upper_bound")):
            return "N/A (upper %s)" % fmt(m["upper_bound"])
        return "N/A"
    s = "%s [%s, %s]" % (fmt(m["value"]), fmt(m.get("lo")), fmt(m.get("hi")))
    if m.get("status") not in ("ok", "resolved", "point", "available"):
        s += "*"
    return s


def add_metric(row, name, m):
    for key in ("value", "lo", "hi", "se", "status", "lower_bound", "upper_bound", "reason"):
        row[name if key == "value" else name + "_" + key] = m.get(key)


def style_axes(ax, compact=False):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Actor updates")
    ax.set_xlim(-5, 132)
    ax.set_xticks([0, 63, 126])
    top = ax.secondary_xaxis("top", functions=(lambda x: .08 * x, lambda x: x / .08))
    top.set_xlabel("Training env steps\n(millions)" if compact else "Training env steps (millions)")
    top.set_xticks([0, 5.04, 10.08])
    top.tick_params(labelsize=10 if compact else 9)


def save_figure(fig, output, name, caption, plt):
    # Caption travels with either exported image, not only the Markdown note.
    fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight", metadata={"Description": caption})
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight", metadata={"Subject": caption, "Title": name})
    (output / (name + ".caption.md")).write_text(caption + "\n")
    plt.close(fig)


def plot_rewards(rewards, output, plt):
    for sampler, filename, ylabel in (("eval", "eval_reward", "Evaluation reward (episode return)"),
                                      ("train", "train_reward", "Training reward (episode return)")):
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        start = rewards["pooled_start"][sampler]
        for method, _, color, _, _ in PATHS:
            rows = rewards["curves"][method][sampler]
            if not rows or rows[0]["update"] != 0:
                raise ValueError("Each original first rollout must use the pretrained actor")
            xx = [r["update"] for r in rows]
            yy = [start["mean"]] + [r["reward"] for r in rows[1:]]
            ax.plot(xx, yy, color=color, lw=1.2, label=method)
            if sampler == "eval":
                ax.plot(xx[1:], yy[1:], color=color, marker="o", ms=3, lw=0)
        ax.errorbar(0, start["mean"], yerr=start.get("se"), color=GRAY,
                    marker="o", ms=4.5, lw=.9, capsize=2, zorder=5)
        for x in (63, 126):
            ax.axvline(x, color="#808080", lw=.6, ls="--", alpha=.8)
        style_axes(ax)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        fig.tight_layout()
        caption = rewards["captions"][sampler]
        caption += " The final evaluation is at 117 updates (9.36 million training environment steps)."
        save_figure(fig, output, filename, caption, plt)


def error_point(ax, x, m, color, hollow=False):
    value = m.get("value")
    if not finite(value) or value <= 0:
        bound = m.get("lower_bound")
        if finite(bound) and bound > 0:
            ax.plot(x, bound, marker="^", color=color, mfc="none", ms=6, ls="none", zorder=4)
        return
    ax.plot(x, value, marker="o", color=color, mfc="none" if hollow else color,
            ms=5, ls="none", zorder=4)
    lo, hi = m.get("lo"), m.get("hi")
    if finite(lo) and finite(hi):
        if lo > 0:
            ax.vlines(x, lo, hi, color=color, lw=.85)
            ax.hlines([lo, hi], x - .8, x + .8, color=color, lw=.85)
        elif hi > 0:
            # A negative/zero endpoint cannot be put on a logarithmic axis.
            # Draw only the valid upper arm, with a visible zero-crossing cue.
            ax.vlines(x, value, hi, color=color, lw=.85)
            ax.hlines(hi, x - .8, x + .8, color=color, lw=.85)
            ax.annotate("CI ≤ 0", (x, value), xytext=(0, -13), textcoords="offset points",
                        color=color, fontsize=8, ha="center")


def panel_metric(point, variant, field):
    if point is None:
        return UNAVAILABLE
    return point["metrics"][field] if field == "u_norm" else point["estimators"][variant][field]


def plot_noise(stats, variant, mapping, output, plt):
    from matplotlib.lines import Line2D
    points = stats["points"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 5.25))
    panels = (("u_norm", "Mean gradient\nnorm ‖∇J‖", "Gradient norm (paper units)"),
              ("trace_ep", "Gradient variance\n(per episode)", "Variance trace (paper units²)"),
              ("B_ep", "Relative variance\n(per episode)", "relative variance = B (episodes)"))
    for ax, (field, title, ylabel) in zip(axes, panels):
        for _, names, color, offset, trained in PATHS:
            xs, ys = [], []
            for name in names:
                m = panel_metric(points.get(name), variant, field)
                x = mapping[name]["update_number"] + (0 if name == "P0" else offset)
                xs.append(x)
                ys.append(m["value"] if finite(m.get("value")) and m["value"] > 0 else np.nan)
                if name != "P0":
                    error_point(ax, x, m, color)
            ax.plot(xs, ys, color=color, lw=1.05)
            if variant == "V6" and field == "B_ep":
                # P0 has two trained-estimator means (V1, V2), so cannot be
                # pooled into one honest origin. Keep their values in the table.
                xs, ys = [], []
                for name in names[1:]:
                    p = points.get(name)
                    m = p["trained"][trained]["B_ep"] if p else UNAVAILABLE
                    x = mapping[name]["update_number"] + offset
                    xs.append(x)
                    ys.append(m["value"] if finite(m.get("value")) and m["value"] > 0 else np.nan)
                    error_point(ax, x, m, color, hollow=True)
                ax.plot(xs, ys, color=color, ls=":", lw=1.)
        error_point(ax, 0, panel_metric(points.get("P0"), variant, field), GRAY)
        ax.set_yscale("log")
        style_axes(ax, compact=True)
        ax.set_title(title, fontsize=12, pad=47)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.margins(y=.25)
        unresolved = [p for p in POINTS if p in points and
                      not finite(panel_metric(points[p], variant, field).get("value"))]
        if unresolved:
            ax.text(.02, .02, "Unresolved: " + ", ".join(unresolved), transform=ax.transAxes,
                    fontsize=8, wrap=True, va="bottom")
    count = stats.get("K", 500)
    fig.suptitle("each point: m = %s episodes (%d batches of 80)" % (format(80 * count, ","), count), fontsize=12, y=.985)
    handles = [Line2D([], [], color=c, marker="o", lw=1.2, label=m) for m, _, c, _, _ in PATHS]
    handles.append(Line2D([], [], color=GRAY, marker="o", lw=0, label="Shared pretrained policy"))
    handles.append(Line2D([], [], color=GRAY, marker="^", mfc="none", lw=0, label="Lower bound: signal unresolved"))
    if variant == "V6":
        handles.append(Line2D([], [], color=GRAY, marker="o", mfc="none", ls=":",
                              label="as used in training (relative to its own mean)"))
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=10, ncol=2,
               columnspacing=1.1, handlelength=1.7)
    fig.subplots_adjust(left=.09, right=.99, bottom=.24 if variant == "V6" else .22,
                        top=.68, wspace=.55)
    baseline = {"V4": "leave-one-out baseline", "V5": "no baseline (the paper's own estimator)",
                "V6": "frozen critic baseline"}[variant]
    missing = [p for p in POINTS if p not in points]
    caption = ("The paper's score estimator with " + baseline + " along the DPPO, without_PPO and without_PPO_and_GAE seed-0 paths; "
               "each point: m = %s episodes (%d batches of 80). " % (format(80 * count, ","), count) +
               "Panel A uses V4's pairwise signal estimate on the same batches; panels B/C use this estimator's per-episode variance trace and its ratio to that V4 squared signal. "
               "Paper units multiply code gradients by 2,500; all panels have log y-axes, and error bars are 90%% delete-one-50-batch-group jackknife intervals with t quantiles (%d degrees of freedom), jointly recomputing ratios. " % (count // 50 - 1) +
               "Marker offsets are −2, 0 and +2 updates; the pretrained point is drawn once in gray. "
               "Open triangles are approved upper-signal plug-in lower bounds when signal is unresolved; these are not two-sided confidence intervals, and missing values are not replaced using larger cohorts. "
               "An average over n episodes has relative variance B/n, so one training update (80 episodes) has B/80. "
               "Intervals are conditional on the measured checkpoints (one training seed per path).")
    if variant == "V6":
        caption += (" The critic at P0 was warmed for 20 batches and then frozen; it is not the untrained critic that training starts with. "
                    "At every other point it is that run's own critic, restored from the checkpoint, given 5 warm-up batches and then frozen. "
                    "Hollow dotted overlays use V1 for DPPO and without_PPO and V2 for without_PPO_and_GAE, relative to each estimator's own mean. "
                    "Their two distinct pretrained values are listed in the table and omitted from the overlay to keep a single shared origin. "
                    "These frozen-critic, full-batch-normalized measurements are proxies for training's evolving critic and per-chunk normalization.")
    if missing:
        caption += " Pending complete equal-size measurements: " + ", ".join(missing) + "."
    save_figure(fig, output, "gradient_noise_" + variant, caption, plt)


def build_table(stats, mapping):
    rows = []
    for name in POINTS:
        p = stats["points"].get(name)
        for variant in ("V4", "V5", "V6"):
            row = {"point": name, "method": mapping[name]["method"], "checkpoint_file": mapping[name]["checkpoint_file"],
                   "update_number": mapping[name]["update_number"], "estimator": variant,
                   "K": p["K"] if p else 0, "m": p["m"] if p else 0, "groups": p["groups"] if p else 0,
                   "status": "measured" if p else "pending 500 completed batches", "trained_estimator": "V2" if name.startswith("C") else "V1",
                   "signal_source": "same-cohort V4", "interval": "90% delete-one-group jackknife, t(df=groups-1)"}
            for metric in ("J_disc", "return", "u_norm"):
                add_metric(row, metric, p["metrics"][metric] if p else UNAVAILABLE)
            for metric in ("trace_ep", "B_ep", "B_update"):
                add_metric(row, metric, p["estimators"][variant][metric] if p else UNAVAILABLE)
            for key in ("V6_over_V4", "V5_over_V6"):
                add_metric(row, "variance_" + key, p["variance_ratios"][key] if p else UNAVAILABLE)
            for v in ("V1", "V2"):
                for key in ("B_ep", "cos_V4"):
                    add_metric(row, v + "_" + key, p["trained"][v][key] if p else UNAVAILABLE)
            ref = p["reference_V4"] if p else {}
            row["reference_all_V4_K"] = ref.get("K", 0)
            for key in ("u_norm", "B_ep"):
                add_metric(row, "reference_all_V4_" + key, ref.get(key, UNAVAILABLE))
            row["selected_jobs"] = ";".join(str(j.get("run_id", j.get("job_id", j))) if isinstance(j, dict) else str(j)
                                               for j in p["selected_jobs"]) if p else ""
            rows.append(row)
    return rows


def write_tables(rows, stats, output):
    with (output / "points_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Checkpoint measurements: three seed-0 paths", "",
             "All main estimates use the same 500 V6-enabled batches at each completed point (40,000 episodes, ten independent jobs). "
             "Brackets are 90% delete-one-group jackknife intervals with Student-t quantiles; B jointly recomputes its numerator and denominator. "
             "The shared gradient norm and every main B denominator use V4 on those same batches. Unresolved B is a plug-in lower bound from the upper signal interval, not a two-sided interval; an asterisk flags other unresolved intervals. "
             "Units: gradients in paper units, trace in squared paper units, B in episodes, and returns in raw reward units.", "",
             "| Point | Method | Checkpoint | Updates | Estimator | K | m | Groups | J_disc | Return | ‖u‖ | Trace/episode | B | B/80 |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        p = stats["points"].get(row["point"])
        values = [str(row[k]) for k in ("point", "method", "checkpoint_file", "update_number", "estimator", "K", "m", "groups")]
        metrics = [p["metrics"][k] if p else UNAVAILABLE for k in ("J_disc", "return", "u_norm")]
        metrics += [p["estimators"][row["estimator"]][k] if p else UNAVAILABLE for k in ("trace_ep", "B_ep", "B_update")]
        lines.append("| " + " | ".join(values + [metric_cell(m) for m in metrics]) + " |")
    lines += ["", "Same-cohort baseline comparisons and the trained estimator's B (relative to its own mean). Both pretrained V1 and V2 values are retained because they cannot be pooled.", "",
              "| Point | Critic/LOO variance | No-baseline/critic variance | Trained estimator | B | cos(trained,V4) |",
              "|---|---:|---:|---|---:|---:|"]
    for name in POINTS:
        p = stats["points"].get(name)
        if p is None:
            continue
        variants = ("V1", "V2") if name == "P0" else ("V2",) if name.startswith("C") else ("V1",)
        for v in variants:
            ms = [p["variance_ratios"][k] for k in ("V6_over_V4", "V5_over_V6")]
            lines.append("| %s | %s | %s | %s | %s | %s |" %
                         (name, metric_cell(ms[0]), metric_cell(ms[1]), v,
                          metric_cell(p["trained"][v]["B_ep"]), metric_cell(p["trained"][v]["cos_V4"])))
    lines += ["", "**Reference only: all available V4 batches.** These values never enter the main figures or matched-cohort B denominators.", "",
              "| Point | K | m | Groups | ‖u‖ | B |", "|---|---:|---:|---:|---:|---:|"]
    for name in POINTS:
        p = stats["points"].get(name)
        if p:
            r = p["reference_V4"]
            lines.append("| %s | %d | %d | %d | %s | %s |" %
                         (name, r["K"], r["m"], r["groups"], metric_cell(r["u_norm"]), metric_cell(r["B_ep"])))
    lines += ["", "The 80× whole-batch trace is the stipulated episode-equivalent variance for the coupled leave-one-out baseline. "
              "Measurement critics are warmed and frozen; V1/V2 normalization uses full-batch statistics, whereas the training references normalize per chunk. "
              "Intervals are conditional on these checkpoints and do not measure training-seed variability. Exact selected jobs appear in [render_record.json](render_record.json)."]
    (output / "points_table.md").write_text("\n".join(lines) + "\n")


def publish(candidates, output):
    output.mkdir(parents=True, exist_ok=True)
    existing = [p for p in output.iterdir() if p.is_file()]
    if existing:
        archive = output / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive.mkdir(parents=True)
        for p in existing:
            shutil.copy2(str(p), str(archive / p.name))
    for p in candidates.iterdir():
        if p.is_file():
            shutil.copy2(str(p), str(output / p.name))


def main(args):
    root = args.root.resolve(); project = root.parents[2]
    rewards = load_reward_data(project)
    stats = build_statistics(root, analysis_source=args.analysis_source, batches=500)
    if any(p["K"] != 500 or p["groups"] != 10 for p in stats["points"].values()):
        raise ValueError("Every rendered point must contain exactly 10x50 completed V6 batches")
    mapping = rewards["checkpoint_mapping"]
    run = root / "runs" / ("clean_v3_render_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    candidates = run / "candidates"; candidates.mkdir(parents=True)
    for path in (Path(__file__), Path(__file__).with_name("clean_v3_stats.py"), Path(__file__).with_name("clean_v3_rewards.py")):
        shutil.copy2(str(path), str(run / path.name))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
    plot_rewards(rewards, candidates, plt)
    for variant in ("V6", "V4", "V5"):
        plot_noise(stats, variant, mapping, candidates, plt)
    table = build_table(stats, mapping)
    write_tables(table, stats, candidates)
    missing = [p for p in POINTS if p not in stats["points"]]
    lines = ["# Clean figures, version 3", "",
             "**Scope.** Seed 0 only: DPPO, without_PPO and without_PPO_and_GAE. All three training runs completed; DPPO uses the original measured Stage-1 run, not a rerun. "
             + ("Gradient measurements pending complete 500-batch cohorts: " + ", ".join(missing) + "." if missing else "All seven gradient points have complete 500-batch cohorts."), "",
             "**Mapping.** " + "; ".join("%s: %s → %d updates" % (p, mapping[p]["checkpoint_file"], mapping[p]["update_number"]) for p in POINTS) +
             ". Each run independently verifies evaluation iteration 130 → 117 updates → 9.36M training steps; checkpoints save after the update phase.", "",
             "**Reward checks.** " + "; ".join("%s: final seed-0 evaluation %s" % (m, fmt(rewards["curves"][m]["eval"][-1]["reward"], 8)) for m, _, _, _, _ in PATHS) +
             "; every value matches its native log. Pooled pretrained evaluation/training means: %.6f / %.6f over 240 original episodes each. " % (rewards["pooled_start"]["eval"]["mean"], rewards["pooled_start"]["train"]["mean"]) +
             "Original episode SDs were not saved, so their requested standard-error bars are unavailable and omitted; neither three-run SDs nor fresh rollouts were substituted.", "",
             "**Measurement.** Main figures and table use the same earliest-launched ten V6-enabled jobs per completed point: K=500, m=40,000. "
             "All-available V4 estimates appear only as labelled table references. With ten groups, the 90% jackknife critical value is t(0.95,9)=1.833113. "
             "Unresolved squared signals remain bounds/N/A; greater precision from unmatched cohorts is not substituted.", "",
             "**Display limits.** Main P0 appears once. V1 and V2 have different pretrained B values, so their overlay origins are omitted and both values remain in the table. "
             "V1/V2 use warmed, frozen critics and full-batch normalization in collection; they are proxies for the evolving, per-chunk-normalized training estimators. "
             "DPPO's nominal Gaussian sampler retains its ±3σ truncation. All artifacts remain uncurated candidates.", "",
             "**Collection provenance.** The added points use their own checkpoint critics, five warm-up batches and frozen reward scalers, with unchanged V6 scientific source. "
             "The collector internally accepts only the original point labels, so C1/C2 use explicit D1/D2 aliases in separate output folders; original D data are never mixed. "
             "[Collection record](../runs/clean_v3_collection_20261012/plan.json) records source revisions, seeds, commands and the unchanged aggregate compute cap.", "",
             "| Point | K | m | Groups | Selected jobs (launch order) |", "|---|---:|---:|---:|---|"]
    for p in POINTS:
        info = stats["points"].get(p)
        if info:
            selected = next(r["selected_jobs"] for r in table if r["point"] == p)
            lines.append("| %s | %d | %d | %d | %s |" % (p, info["K"], info["m"], info["groups"], selected))
        else:
            lines.append("| %s | pending | pending | pending | — |" % p)
    lines += ["", "[Evaluation](eval_reward.png) · [Training](train_reward.png) · [Critic gradient noise](gradient_noise_V6.png) · "
              "[LOO](gradient_noise_V4.png) · [No baseline](gradient_noise_V5.png) · [Table](points_table.md). Captions accompany every figure; "
              "[render record](render_record.json) preserves exact inputs, intervals and checks."]
    (candidates / "NOTE.md").write_text("\n".join(lines) + "\n")
    record = {"version": 3, "status": "interim" if missing else "complete", "utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "run": str(run), "mapping": mapping, "rewards": rewards, "statistics": stats,
              "pending_points": missing, "curated": False, "simulation_launched_by_renderer": False}
    (candidates / "render_record.json").write_text(json.dumps(json_safe(record), indent=2, allow_nan=False) + "\n")
    output = args.output.resolve() if args.output else root / "figs_clean"
    publish(candidates, output)
    print(json.dumps({"status": record["status"], "output": str(output), "run": str(run), "pending_points": missing}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean"))
    parser.add_argument("--analysis-source", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
