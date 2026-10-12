#!/usr/bin/env python
"""Read completed measurements and render the approved clean-figures v2.

Python 3.8. Never launches a simulation or alters raw measurement files.
Prior renders at the requested stable filenames are retained in history/.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys

import numpy as np

POINTS = ("P0", "D1", "D2", "N1", "N2")
PATHS = (("DPPO", ("P0", "D1", "D2"), "#0072B2", -1.5),
         ("without_PPO", ("P0", "N1", "N2"), "#D55E00", 1.5))
GRAY = "#383838"
EPISODES = 80
FACTOR = 2500.


def load_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return value is not None and math.isfinite(float(value))


def fmt(value, digits=3):
    return "N/A" if not finite(value) else ("%." + str(digits) + "g") % value


def interval_cell(row, field):
    value = row.get(field)
    if not finite(value):
        bound = row.get(field + "_lower_bound")
        return "≥" + fmt(bound) + " (bound)" if finite(bound) else "N/A"
    return "%s [%s, %s]" % (fmt(value), fmt(row.get(field + "_low")), fmt(row.get(field + "_high")))


def add_interval(row, field, result, scale=1.):
    for old, suffix in (("value", ""), ("se", "_se"), ("lower", "_low"), ("upper", "_high")):
        row[field + suffix] = float(result[old]) * scale if finite(result.get(old)) else None
    row[field + "_reason"] = result.get("reason")


def metric(summary, key):
    return {"value": summary["estimate"][key], **summary["jackknife90"][key]}


def serializable(value):
    """Keep provenance strict JSON even if future readers return NumPy scalars."""
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite render provenance must be explicitly unavailable")
    return value


def load_analysis(project, requested=None):
    """Main-repository renderer copies always import the isolated approved code."""
    code = Path(requested).resolve() if requested else (Path(project) /
        ".codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean").resolve()
    if not (code / "analysis_v6.py").is_file():
        raise ValueError("Approved isolated V6 analysis source is unavailable: " + str(code))
    sys.path.insert(0, str(code))
    modules = {name: importlib.import_module(name) for name in ("analysis_core", "analysis_v6", "gradmean_math")}
    for name, module in modules.items():
        if Path(module.__file__).resolve().parent != code:
            raise RuntimeError(name + " was imported from another tree; use a fresh renderer process")
    return code, modules["analysis_core"], modules["analysis_v6"]


def rollout_updates(row):
    """Logged train step includes current rollout; its policy predates that update."""
    completed = int(row["step"]) // 80000
    if int(row["step"]) % 80000:
        raise ValueError("Unexpected fresh-batch geometry")
    return completed - int("train_episode_reward" in row)


def read_rewards(project, checkpoints):
    record = load_json(project / "results/stage3/analysis_record.json")["reference_sources"]
    source = {"DPPO": record["Stage 1 DPPO"],
              "without_PPO": record["Stage 2 NC4 1e-4"]}
    runs, mapping, provenance = {}, {}, {}
    for method, paths in source.items():
        if len(paths) != 3:
            raise ValueError("Exactly three reference seeds required")
        runs[method] = []
        for seed, name in enumerate(paths):
            path = Path(name)
            with path.open("rb") as stream:
                rows = pickle.load(stream)
            if [r["itr"] for r in rows] != list(range(140)):
                raise ValueError("Reference training must have all 140 iterations")
            for row in rows:
                itr = int(row["itr"])
                expected = sum(t % 10 != 0 for t in range(itr))
                if rollout_updates(row) != expected:
                    raise ValueError("Rollout update counter differs from evaluation schedule")
            final_eval = [row for row in rows if row["itr"] == 130][0]
            if rollout_updates(final_eval) != 117 or final_eval["step"] != 9360000:
                raise ValueError("Final evaluation is not update 117 /9.36M steps")
            runs[method].append(rows)
        provenance[method] = list(paths)
    for point in POINTS:
        path = Path(checkpoints[point]["checkpoint"])
        method = "pretrained" if point == "P0" else "DPPO" if point.startswith("D") else "without_PPO"
        if point == "P0":
            updates, itr, measured = 0, None, None
        else:
            match = re.fullmatch(r"state_(\d+)\.pt", path.name)
            if not match:
                raise ValueError("Unrecognized measured checkpoint")
            itr = int(match[1])
            measured = path.parent.parent / "result.pkl"
            with measured.open("rb") as stream:
                measured_rows = pickle.load(stream)
            saved_row = next(row for row in measured_rows if row["itr"] == itr)
            # Native save_model() occurs after the update phase.
            updates = sum("train_episode_reward" in row for row in measured_rows if row["itr"] <= itr)
            if int(saved_row["step"]) != updates * 80000:
                raise ValueError("Checkpoint update count conflicts with logged training steps")
            if point.endswith("2") and updates != 126:
                raise ValueError("Final checkpoint does not contain126 updates")
        mapping[point] = {"method": method, "checkpoint_file": path.name, "checkpoint_itr": itr,
                          "updates": updates, "training_env_steps": 80000 * updates,
                          "checkpoint_path": str(path), "measured_result": str(measured) if measured else None}
    if mapping["D1"]["measured_result"] != mapping["D2"]["measured_result"] or mapping["N1"]["measured_result"] != mapping["N2"]["measured_result"]:
        raise ValueError("Middle/final points must belong to the same path")
    measured = {}
    for method, point in (("DPPO", "D1"), ("without_PPO", "N1")):
        path = Path(mapping[point]["measured_result"])
        with path.open("rb") as stream:
            measured[method] = pickle.load(stream)
        provenance[method + " measured seed"] = str(path)
    provenance["DPPO_measured_seed_is_rerun"] = mapping["D1"]["measured_result"] != source["DPPO"][0]
    return runs, measured, mapping, provenance


def build_table(points, v6, mapping, core):
    v6_rows = {row["point"]: row for row in v6["tables"]["v6_main_results"]}
    ratio_rows = {(row["point"], row["contrast"]): row for row in v6["tables"]["v6_variance_ratios"]}
    output = []
    for name in POINTS:
        point = points.get(name)
        for variant in ("V4", "V5", "V6"):
            six = v6_rows.get(name, {})
            count = int(six.get("K_V6", 0)) if variant == "V6" else point.k if point else 0
            row = {"point": name, "estimator": variant, "method": mapping[name]["method"],
                   "checkpoint_file": mapping[name]["checkpoint_file"], "update_number": mapping[name]["updates"],
                   "K": count, "m": EPISODES * count, "groups": count // 50,
                   "K_signal_V4": point.k if point else 0, "m_signal_V4": EPISODES * point.k if point else 0,
                   "signal_source": "V4 all batches", "interval": "90% delete-one-50-batch-group jackknife"}
            if point is not None:
                for field in ("J_disc", "return"):
                    add_interval(row, field, core.group_mean(point.group_rewards[field]))
                row["K_returns"] = point.k
                add_interval(row, "u_norm", metric(point.summaries["V4"], "signal_norm"), FACTOR)
                if variant == "V6" and six:
                    for dest, src in (("trace_ep", "V6_trace_ep"), ("B_ep", "V6_B_ep")):
                        for suffix in ("", "_se", "_low", "_high"):
                            row[dest + suffix] = six.get(src + suffix)
                    row["B_ep_lower_bound"] = six.get("V6_B_ep_lower_bound")
                    row["B_ep_reason"] = six.get("V6_B_ep_uncertainty_reason")
                elif variant != "V6":
                    index = ("V1", "V2", "V3", "V4", "V5").index(variant)
                    add_interval(row, "trace_ep", metric(point.summaries[variant], "covariance_trace"), EPISODES * FACTOR ** 2)
                    value = point.scalar(lambda moment: float(EPISODES * moment["trace"][index] / moment["signal"][3])
                                         if moment["signal"][3] > 0 else None)
                    if point.summaries["V4"]["scale_reporting"]["kind"] != "point":
                        upper = point.summaries["V4"]["jackknife90"]["signal_sq"].get("upper")
                        row["B_ep_lower_bound"] = float(EPISODES * point.full_moments["trace"][index] / upper) if finite(upper) and upper > 0 else None
                        value = {"value": None, "reason": "V4 signal unresolved; upper-signal plug-in bound"}
                    add_interval(row, "B_ep", value)
                for suffix in ("", "_se", "_low", "_high", "_lower_bound"):
                    row["B_update" + suffix] = row.get("B_ep" + suffix) / EPISODES if finite(row.get("B_ep" + suffix)) else None
                v1 = point.summaries["V1"]
                v1_stat = metric(v1, "B_ep") if v1["scale_reporting"]["kind"] == "point" else {"value": None, "reason": "V1 signal unresolved"}
                add_interval(row, "V1_B_ep", v1_stat)
                row["V1_B_ep_lower_bound"] = v1["estimate"].get("B_ep_lower_bound")
                row["V1_m"] = EPISODES * point.k
                cos = point.scalar(lambda m: core.cosine(m["cross_signal"][0, 3], m["signal"][0], m["signal"][3]))
                add_interval(row, "cos_V1_V4", cos)
            for short, label in (("variance_V6_over_V4", "V6_trace_over_V4_trace"),
                                 ("variance_V5_over_V6", "V5_trace_over_V6_trace")):
                contrast = ratio_rows.get((name, label), {})
                for suffix in ("", "_se", "_low", "_high"):
                    row[short + suffix] = contrast.get("ratio" + suffix)
            row["paired_variance_K"] = int(six.get("K_V6", 0))
            row["availability"] = "measured" if count else "no completed batches for this estimator"
            output.append(row)
    return output


def style_axes(ax, compact=False):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Actor updates")
    ax.set_xlim(-5, 133)
    ax.set_xticks([0, 63, 126])
    top = ax.secondary_xaxis("top", functions=(lambda x: .08 * x, lambda x: x / .08))
    top.set_xlabel("Training env steps\n(millions)" if compact else "Training env steps (millions)")
    top.set_xticks([0, 5.04, 10.08])


def error_point(ax, x, row, field, color, marker="o", hollow=False):
    value, low, high = row.get(field), row.get(field + "_low"), row.get(field + "_high")
    if not finite(value) or value <= 0:
        bound = row.get(field + "_lower_bound")
        if finite(bound) and bound > 0:
            ax.plot(x, bound, marker="^", color=color, mfc="none", ms=6, ls="none")
            ax.annotate("lower bound", (x, bound), xytext=(0, 5), textcoords="offset points", fontsize=8, ha="center")
        return False
    ax.plot(x, value, marker=marker, color=color, mfc="none" if hollow else color, ms=5, ls="none", zorder=4)
    if finite(low) and finite(high):
        if low > 0:
            ax.vlines(x, low, high, color=color, linewidth=.85)
            ax.plot([x, x], [low, high], ls="none", marker="_", ms=5, color=color)
        else:
            ax.vlines(x, value, high, color=color, linewidth=.85)
            ax.annotate("CI crosses 0", (x, value), xytext=(0, -12), textcoords="offset points", fontsize=7, ha="center")
    return True


def save_figure(fig, output, name, caption, plt):
    fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight", metadata={"Description": caption})
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight", metadata={"Subject": caption, "Title": name})
    (output / (name + ".caption.md")).write_text(caption + "\n")
    plt.close(fig)


def plot_rewards(runs, measured, mapping, provenance, output, plt):
    sanity = {}
    for field, name, ylabel in (("eval_episode_reward", "eval_reward", "Evaluation reward (episode return)"),
                                 ("train_episode_reward", "train_reward", "Training reward (episode return)")):
        fig, ax = plt.subplots(figsize=(5.2, 3.45))
        for method, _, color, _ in PATHS:
            values = [[row for row in run if field in row] for run in runs[method]]
            xs = [[rollout_updates(row) for row in group] for group in values]
            if xs[0] != xs[1] or xs[0] != xs[2]:
                raise ValueError("Seed reward points are not aligned")
            data = np.asarray([[row[field] for row in group] for group in values], dtype=float)
            if not np.isfinite(data).all():
                raise ValueError("Nonfinite reference reward")
            avg, sd = data.mean(0), data.std(0, ddof=1)
            ax.fill_between(xs[0], avg - sd, avg + sd, color=color, alpha=.15, linewidth=0)
            ax.plot(xs[0], avg, color=color, label=method, linewidth=1.5,
                    marker="o" if name == "eval_reward" else None, markersize=3)
            seed = [row for row in measured[method] if field in row]
            ax.plot([rollout_updates(row) for row in seed], [row[field] for row in seed], color=color, linewidth=.55, alpha=.8)
            if name == "eval_reward":
                sanity[method] = {"final_itr": 130, "updates": 117, "training_env_steps": 9360000,
                                  "mean": float(avg[-1]), "sample_sd": float(sd[-1]), "seed_values": data[:, -1].tolist()}
        for x in sorted({row["updates"] for row in mapping.values()}):
            ax.axvline(x, color="#808080", linewidth=.65, linestyle="--", alpha=.7)
        style_axes(ax)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        fig.tight_layout()
        sampler = ("Evaluation uses the deterministic=True DDPM sampler: noise floor 0.001 at nonfinal denoising steps and zero final-step noise. Each point averages 80 complete 1000-step episodes per seed."
                   if name == "eval_reward" else
                   "Each unsmoothed training point averages 80 complete 1000-step episodes collected with the training sampler and denoising noise floor 0.1; these exploratory episodes score lower than evaluation episodes.")
        caption = (sampler + " Lines and shaded bands show the mean±1 sample standard deviation (ddof=1) across three independent training seeds for each method. "
                   "Thin lines follow the measured seed 0 paths" + (", using the separate seed 0 DPPO rerun" if provenance["DPPO_measured_seed_is_rerun"] else "") +
                   "; dashed vertical lines mark the measured checkpoint update counts. One actor update consumes80,000 fresh environment steps; a rollout is placed at the number of updates before it, while a checkpoint is placed after its saved iteration's update. "
                   "The final evaluation is update 117 (9.36 million steps), before the final checkpoint at update 126.")
        save_figure(fig, output, name, caption, plt)
    return sanity


def plot_noise(table, variant, output, plt):
    by = {row["point"]: row for row in table if row["estimator"] == variant}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.7))
    panels = (("u_norm", "Mean gradient\nnorm ‖∇J‖", "Gradient norm (paper units)"),
              ("trace_ep", "Gradient variance\n(per episode)", "Variance trace (paper units²)"),
              ("B_ep", "Relative variance\n(per episode)", "relative variance = B (episodes)"))
    for index, (ax, (field, title, ylabel)) in enumerate(zip(axes, panels)):
        for method, names, color, offset in PATHS:
            xx, yy = [], []
            for point in names:
                row = by[point]
                x = row["update_number"] + (0 if point == "P0" else offset)
                value = row.get(field)
                xx.append(x); yy.append(value if finite(value) and value > 0 else np.nan)
                if point == "P0":
                    continue
                if error_point(ax, x, row, field, color):
                    m = row["m_signal_V4"] if index == 0 else row["m"]
                    other_name = ("N" if point.startswith("D") else "D") + point[1:]
                    other_value = by[other_name].get(field)
                    above = value >= other_value if finite(other_value) else offset < 0
                    label_y = row.get(field + ("_high" if above else "_low"))
                    if not finite(label_y) or label_y <= 0:
                        label_y = value
                    ax.annotate("m=" + ("%gk" % (m / 1000.) if m >= 1000 else str(m)), (x, label_y),
                                xytext=(0, 6 if above else -12), textcoords="offset points", ha="center", color=color, fontsize=8)
            ax.plot(xx, yy, color=color, linewidth=1.05, label=method)
        p0 = by["P0"]
        if error_point(ax, 0, p0, field, GRAY):
            m = p0["m_signal_V4"] if index == 0 else p0["m"]
            ax.annotate("m=" + ("%gk" % (m / 1000.) if m >= 1000 else str(m)),
                        (0, p0[field]), xytext=(4, -15 if index == 2 else 9), textcoords="offset points", fontsize=8, color=GRAY)
        if variant == "V6" and index == 2:
            for _, names, color, offset in PATHS:
                xx, yy = [], []
                for point in names:
                    row = by[point]; x = row["update_number"] + (0 if point == "P0" else offset)
                    value = row.get("V1_B_ep")
                    xx.append(x); yy.append(value if finite(value) and value > 0 else np.nan)
                    if point != "P0":
                        error_point(ax, x, row, "V1_B_ep", color, hollow=True)
                ax.plot(xx, yy, color=color, linewidth=.95, linestyle=":")
            error_point(ax, 0, by["P0"], "V1_B_ep", GRAY, hollow=True)
        ax.set_yscale("log"); style_axes(ax, compact=True)
        ax.set_title(title, fontsize=12); ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.margins(y=.25)
        missing = [name for name in POINTS if not finite(by[name].get(field)) and
                   not finite(by[name].get(field + "_lower_bound"))]
        if len(missing) == len(POINTS) and not (variant == "V6" and index == 2):
            ax.set_yticks([]); ax.set_yticks([], minor=True)
        if missing:
            ax.text(.03, .03, variant + " unavailable\n" + ", ".join(missing[:3]) +
                    ("\n" + ", ".join(missing[3:]) if len(missing) > 3 else ""),
                    transform=ax.transAxes, fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color, marker="o", lw=1.2, label=method) for method, _, color, _ in PATHS]
    handles.append(Line2D([], [], color=GRAY, marker="o", lw=0, label="Shared pretrained policy"))
    if variant == "V6":
        handles.append(Line2D([], [], color=GRAY, marker="o", mfc="none", ls=":",
                              label="as used in training: GAE + critic\n(relative to its own mean)"))
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=10, ncol=2, columnspacing=1.2)
    fig.tight_layout(rect=(0, .23 if variant == "V6" else .14, 1, 1), w_pad=1.4)
    baseline = {"V4": "leave-one-out baseline", "V5": "no baseline (the paper's estimator)", "V6": "frozen critic baseline"}[variant]
    counts = "; ".join("%s: V4 K=%d, displayed-variance K=%d" % (name, by[name]["K_signal_V4"], by[name]["K"]) for name in POINTS)
    caption = ("The paper's score estimator with " + baseline + " is measured at shared P0 and along DPPO and without_PPO seed 0 paths. "
               "Panel A uses the square root of V4's all-batch pairwise squared-signal estimate; panels B/C use this figure's estimator trace and that same V4 signal, in paper units (code gradients×2,500). "
               "All y axes are logarithmic; error bars are 90% delete-one-group (50 batches) jackknife intervals, jointly recomputing numerator and denominator for relative variance. For V6 B, independent legacy/new-cohort jackknife variances are summed, and new groups are deleted from both the V6 numerator and V4 denominator. "
               "Offsets of −1.5/+1.5 updates separate the DPPO/without_PPO markers; shared P0 is drawn in gray. Labels m=80K are measurement episode counts, not the 80-episode training batch: " + counts + ". "
               "An average over n episodes has relative variance B/n, so one training update (80 episodes) has B/80. For the leave-one-out estimator, the displayed episode trace is the stipulated 80× whole-batch trace, a batch-equivalent scale because episodes share the baseline. "
               "Intervals are conditional on the measured checkpoints (one training seed per path). Nonpositive bounds are labeled instead of truncated to an artificial log-axis floor.")
    if variant == "V6":
        caption += (" The critic at P0 was warmed for 20 batches then frozen; it is not the untrained critic at the start of training. At D1, D2, N1, N2 it is each run's own checkpoint critic, warmed for 5 batches then frozen. "
                    "Hollow dotted V1 points use B relative to V1's own mean; V1 measurement episode counts are " +
                    "; ".join(name + ": " + str(by[name].get("V1_m", 0)) for name in POINTS) +
                    ". V1 uses full-batch normalization and a frozen warmed critic; it is a measurement proxy for the training estimator, whose normalization is per optimizer chunk and critic evolves.")
    save_figure(fig, output, "gradient_noise_" + variant, caption, plt)


def write_tables(table, output):
    columns = list(dict.fromkeys(key for row in table for key in row))
    with (output / "points_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(table)
    lines = ["# DPPO and without_PPO: checkpoint measurements", "",
             "Each estimate is followed by its 90% group-jackknife interval. K is the variance estimator's batch count; m=80K is its measurement episode count. The shared mean-gradient norm uses all V4 batches, listed separately. J_disc and return use all completed batches at the point. B uses the displayed trace divided by the shared all-V4 squared signal; B/80 is relative variance of one 80-episode training update.", "",
             "| Point | Method | Checkpoint | Updates | Estimator | K | m | Groups | V4 signal K | J_disc | Return | Mean gradient norm | Trace/episode | B/episode | B/80 |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| " + " | ".join(str(row[k]) for k in ("point", "method", "checkpoint_file", "update_number", "estimator", "K", "m", "groups", "K_signal_V4")) + " | " +
                     " | ".join(interval_cell(row, k) for k in ("J_disc", "return", "u_norm", "trace_ep", "B_ep", "B_update")) + " |")
    lines += ["", "Matched baseline variance ratios use only batches containing V6. V1's dimensionless B is relative to its own mean, and its cosine with V4 excludes same-batch products. These fixed-policy measurement intervals do not measure training-seed uncertainty.", "",
              "| Point | Matched K | Critic/LOO variance | No-baseline/critic variance | V1 B | cos(V1,V4) |", "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        if row["estimator"] == "V6":
            lines.append("| %s | %d | %s | %s | %s | %s |" % (row["point"], row["paired_variance_K"],
                         interval_cell(row, "variance_V6_over_V4"), interval_cell(row, "variance_V5_over_V6"),
                         interval_cell(row, "V1_B_ep"), interval_cell(row, "cos_V1_V4")))
    (output / "points_table.md").write_text("\n".join(lines) + "\n")


def preserve_render(output):
    names = ["eval_reward", "train_reward", "gradient_noise_V6", "gradient_noise_V4", "gradient_noise_V5"]
    files = [output / (name + suffix) for name in names for suffix in (".png", ".pdf", ".caption.md")]
    files += [output / name for name in ("points_table.csv", "points_table.md", "NOTE.md", "render_record.json")]
    existing = [path for path in files if path.exists()]
    if existing:
        archive = output / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive.mkdir(parents=True)
        for path in existing:
            shutil.copy2(str(path), str(archive / path.name))


def main(args):
    root = args.root.resolve(); project = root.parents[2]
    code, core, v6_module = load_analysis(project, args.analysis_source)
    checkpoints = load_json(root / "provenance/checkpoints.json")
    runs, measured, mapping, provenance = read_rewards(project, checkpoints)
    points, inventory = core.discover(root)
    v6 = v6_module.analyze_v6(points, load_json(root / "runs/campaign.json"))
    table = build_table(points, v6, mapping, core)
    output = args.output.resolve() if args.output else root / "figs_clean"
    output.mkdir(parents=True, exist_ok=True); preserve_render(output)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
    sanity = plot_rewards(runs, measured, mapping, provenance, output, plt)
    for variant in ("V6", "V4", "V5"):
        plot_noise(table, variant, output, plt)
    write_tables(table, output)
    missing = v6["missing_points"]
    lines = ["# Clean figures, version2", "",
             "**Mapping.** " + "; ".join("%s: %s → %d updates" % (name, mapping[name]["checkpoint_file"], mapping[name]["updates"]) for name in POINTS) + ". Checkpoints save after the update phase; reward rollouts precede that phase. Final evaluation: iteration 130 → 117 updates → 9.36 million training environment steps.", "",
             "**Sanity check.** " + "; ".join("%s final evaluation %.3f ± %.3f (sample sd, three seeds)" % (method, r["mean"], r["sample_sd"]) for method, r in sanity.items()) + ".", "",
             "**Sample sizes.** " + "; ".join("%s: V4/V5 K=%d,m=%d; V6 K=%d,m=%d" % (name, v6["counts"][name]["K_all"],80*v6["counts"][name]["K_all"],v6["counts"][name]["K_V6"],80*v6["counts"][name]["K_V6"]) for name in POINTS) + ".", "",
             "**Availability and limits.** " + ("No V6 batches yet at " + ", ".join(missing) + "; missing main-figure measurements remain blank. " if missing else "All five points have V6 measurements. ") +
             "V1 is a warmed/frozen, full-batch-normalized measurement proxy for the evolving, per-chunk-normalized training estimator. The original adaptive stopping rule was replaced by exactly 10 new V6 groups per point for compute reasons, before seeing new V6 data; already-running jobs also count. The table distinguishes the V6 cohort from the larger shared-signal cohort. Figures and tables are uncurated candidates.", "",
             "[Evaluation](eval_reward.png) · [Training](train_reward.png) · [Critic gradient noise](gradient_noise_V6.png) · [LOO](gradient_noise_V4.png) · [No baseline](gradient_noise_V5.png) · [Table](points_table.md). Captions accompany every PNG/PDF; [render record](render_record.json) lists exact inputs."]
    (output / "NOTE.md").write_text("\n".join(lines) + "\n")
    record = {"status": "complete", "utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "analysis_source": str(code), "mapping": mapping, "reward_sources": provenance,
              "sanity": sanity, "counts": v6["counts"], "missing_V6_points": missing,
              "group_inventory": inventory, "consistency_failure": v6["consistency_failure"],
              "source_file": str(Path(__file__).resolve()), "expected_files": [name + ext for name in ("eval_reward", "train_reward", "gradient_noise_V6", "gradient_noise_V4", "gradient_noise_V5") for ext in (".png", ".pdf", ".caption.md")] + ["points_table.csv", "points_table.md", "NOTE.md"],
              "no_training_or_collection_launched": True, "curated": False}
    (output / "render_record.json").write_text(json.dumps(serializable(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "missing_V6_points": missing, "sanity": sanity}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean"))
    parser.add_argument("--analysis-source", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
