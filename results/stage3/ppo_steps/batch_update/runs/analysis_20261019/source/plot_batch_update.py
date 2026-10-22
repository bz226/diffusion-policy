#!/usr/bin/env python3
"""Plot stored DPPO batch-update estimates; never recompute gradients/statistics.

Python 3.8 compatible. All figures, captions and plotting coordinates are new
candidate files. Undefined or unresolved norms/ratios are omitted explicitly.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from plot_clip_timing import bounds_check


BLUE = "#004d7a"
EPOCH_COLOR = "#CC79A7"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#909090"
UPDATES = (32, 63, 95)
NAMES = (
    "dppo_epoch_averages", "dppo_batch_vs_onpolicy",
    "dppo_extra_mean_cosines", "dppo_extra_mean_path",
)


def finite(metric):
    value = metric.get("estimate") if isinstance(metric, dict) else metric
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def passed(result):
    if isinstance(result, dict):
        return result.get("passed", result.get("pass")) is True or result.get("status") in ("pass", "passed")
    return result is True or result in ("pass", "passed")


def gates_pass(data, preservation):
    gates = data["gates"]
    gates = gates.get("checks", gates)
    if isinstance(gates, list):
        gates = {row.get("name", row.get("gate")): row for row in gates}
    for name in ("G1", "G2", "G3", "G4"):
        if not passed(gates.get(name)):
            raise ValueError("Refusing to plot before {} passes".format(name))
    if not passed(gates.get("G5")):
        if preservation is None:
            raise ValueError("G5 requires --preservation-check or embedded gate evidence")
        evidence = json.loads(preservation.read_text(encoding="utf-8"))
        evidence = evidence.get("G5", evidence)
        if not passed(evidence):
            raise ValueError("Refusing to plot before the preservation check passes")


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    points = {int(row["update"]): row for row in data["points"]}
    if sorted(points) != list(UPDATES):
        raise ValueError("Expected exactly the checkpoints at 32, 63 and 95 updates")
    for point in points.values():
        quantities = {row["name"]: row for row in point["quantities"]}
        if sorted(quantities) != sorted(["epoch{}".format(i) for i in range(1, 6)] + ["batch"]):
            raise ValueError("Missing epoch or batch estimate")
        point["by_name"] = quantities
        if point["turning"]["steps"] != list(range(5, 21)):
            raise ValueError("Expected turning estimates at steps 5 through 20")
    return data, points


def estimate(point, quantity, key):
    metrics = point["by_name"][quantity]["metrics"]
    if key in ("mean_norm", "R") and not metrics["signal_sq"]["resolved"]:
        return None
    # Signed components/cosines are valid even when their own intervals include
    # zero. The analysis sets these to null when the reference is unresolved.
    value = finite(metrics[key])
    if key in ("mean_norm", "variance", "R", "R_on") and value is not None and value <= 0:
        return None
    return value


def decade_limits(values):
    values = [value for value in values if value is not None and value > 0]
    if not values:
        return None
    low = math.floor(math.log10(min(values)))
    high = math.ceil(math.log10(max(values)))
    if low == high:
        high += 1
    return (10.0 ** low, 10.0 ** high)


def style(axis, log=False):
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", which="major", linewidth=0.6, color="#e2e2e2")
    axis.grid(False, axis="x", which="both")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.minorticks_off()
    axis.tick_params(labelsize=9)
    if log:
        axis.set_yscale("log")


def line(axis, x, y, color=BLUE, label=None, linestyle="-"):
    # NaNs leave gaps; no lines bridge unresolved points.
    axis.plot(x, [np.nan if value is None else value for value in y],
              color=color, linewidth=1.3, marker="o", markersize=3.8,
              markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.3,
              label=label, linestyle=linestyle)


def missing_text(omissions):
    return "Omitted as unresolved or unavailable: {}.".format("; ".join(omissions) if omissions else "none")


def row_labels(figure, axes):
    for update, row in zip(UPDATES, axes):
        box = row[0].get_position()
        figure.text(0.013, box.y0 + box.height / 2.0, "{} updates".format(update),
                    rotation=90, ha="center", va="center", fontsize=10.5)


def epoch_figure(points):
    keys = ("mean_norm", "variance", "R", "cos_epoch1")
    titles = ("Mean gradient norm", "Gradient variance", "Relative variance", "Alignment with epoch 1")
    labels = (
        r"$\|\mathrm{E}\,\bar G_e\|$" + "\n(training-loss units)",
        r"$\mathrm{tr}\,\mathrm{Cov}(\bar G_e)$" + "\n(training-loss units$^2$)",
        r"$R(\bar G_e)$ (no units)",
        r"$\cos(\mathrm{E}\,\bar G_e,\mathrm{E}\,\bar G_1)$",
    )
    limits = {}
    for key in keys[:3]:
        limits[key] = decade_limits([estimate(point, name, key) for point in points.values()
                                     for name in point["by_name"]])
    limits["cos_epoch1"] = (-1.0, 1.0)
    figure, axes = plt.subplots(3, 4, figsize=(13.8, 8.8), squeeze=False)
    figure.subplots_adjust(left=0.092, right=0.987, bottom=0.075, top=0.89, wspace=0.49, hspace=0.30)
    figure.suptitle("DPPO", fontsize=15, y=0.98)
    figure.supxlabel("Epoch (4 PPO steps per epoch)", fontsize=11, y=0.022)
    figure.legend([Line2D([0], [0], color=BLUE, marker="o"),
                   Line2D([0], [0], color=BLUE, linestyle="--", alpha=0.4)],
                  ["DPPO, epoch average", "DPPO, batch average"],
                  loc="upper center", bbox_to_anchor=(0.5, 0.957), ncol=2, frameon=False, fontsize=9)
    records, omissions = [], []
    for row_index, update in enumerate(UPDATES):
        point = points[update]
        for col, key in enumerate(keys):
            axis = axes[row_index, col]
            values = [estimate(point, "epoch{}".format(epoch), key) for epoch in range(1, 6)]
            batch = estimate(point, "batch", key)
            if key == "cos_epoch1" and any(value is not None and not -1 <= value <= 1 for value in values + [batch]):
                raise ValueError("A finite line cosine lies outside [-1,1] at {} updates; requires an explicit display decision".format(update))
            line(axis, list(range(1, 6)), values)
            if batch is not None:
                axis.axhline(batch, linestyle="--", color=BLUE, linewidth=1, alpha=0.35)
            style(axis, log=col < 3)
            if limits[key] is not None:
                axis.set_ylim(*limits[key])
            axis.set_xlim(0.75, 5.25)
            axis.set_xticks(range(1, 6))
            axis.set_ylabel(labels[col], fontsize=9)
            if row_index == 0:
                axis.set_title(titles[col], fontsize=11, pad=11)
            absent = [str(epoch) for epoch, value in enumerate(values, 1) if value is None]
            if absent:
                omissions.append("{} updates, {}: epochs {}".format(update, titles[col].lower(), ", ".join(absent)))
            if batch is None:
                omissions.append("{} updates, {}: batch reference".format(update, titles[col].lower()))
            records.append({"update": update, "metric": key, "x": list(range(1, 6)), "y": values, "batch_reference": batch, "ylim": limits[key]})
    row_labels(figure, axes)
    caption = (
        "DPPO's average gradient in each epoch of one batch (epoch 1 = steps 1–4), from the rerun's own state after 32, 63 and 95 updates (rows). "
        "To first order every epoch average equals the batch's full on-policy gradient, so departures from epoch 1 come from reuse and the clip. "
        "Dashed: the average over all 20 steps. 500 batches per checkpoint, in training-loss units; points are estimates, 90% intervals are in batch_update_table. "
        + missing_text(omissions)
    )
    return figure, records, caption


def comparison_figure(points):
    methods = {point["reference"]["method"] for point in points.values()}
    if len(methods) != 1:
        raise ValueError("Reference must use one method consistently across checkpoints")
    method = methods.pop()
    if method not in ("without_PPO", "without_PPO_and_GAE"):
        raise ValueError("Unrecognized human-facing reference method")
    reference_values = []
    for update in UPDATES:
        metric = points[update]["reference"]["metrics"]["R"]
        reference_values.append(finite(metric) if metric.get("resolved", False) else None)
    specs = (
        ("epoch1", "R", "on-policy gradient (epoch 1)", EPOCH_COLOR, "-"),
        ("batch", "R", "DPPO, batch average", BLUE, "-"),
        ("batch", "R_on", "DPPO, batch average, against the on-policy direction", BLUE, ":"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.35))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.70, wspace=0.34)
    figure.suptitle("DPPO", fontsize=14, y=0.98)
    records, omissions, rv = [], [], []
    for quantity, key, label, color, dash in specs:
        values = [estimate(points[update], quantity, key) for update in UPDATES]
        line(axes[0], UPDATES, values, color=color, label=label, linestyle=dash)
        rv.extend(values)
        records.append({"panel": "relative_variance", "quantity": quantity, "metric": key, "x": list(UPDATES), "y": values})
        for update, value in zip(UPDATES, values):
            if value is None:
                omissions.append("{} at {} updates".format(label, update))
    axes[0].plot(UPDATES, [np.nan if value is None else value for value in reference_values],
                 color=ORANGE if method == "without_PPO" else GREEN, alpha=0.45, linestyle="--", linewidth=1.2,
                 label="{}, one update (its own run)".format(method))
    rv.extend(reference_values)
    records.append({"panel": "relative_variance", "quantity": method, "metric": "R", "x": list(UPDATES), "y": reference_values})
    for update, value in zip(UPDATES, reference_values):
        if value is None:
            omissions.append("{} reference at {} updates".format(method, update))
    signed = []
    for quantity, key, label, color in (
        ("epoch1", "mean_norm", r"$\|\mathrm{E}\,\bar G_1\|$", EPOCH_COLOR),
        ("batch", "a", r"$a(\bar G)$", BLUE),
    ):
        values = [estimate(points[update], quantity, key) for update in UPDATES]
        line(axes[1], UPDATES, values, color=color, label=label)
        signed.extend(value for value in values if value is not None)
        records.append({"panel": "component", "quantity": quantity, "metric": key, "x": list(UPDATES), "y": values})
        for update, value in zip(UPDATES, values):
            if value is None:
                omissions.append("{} at {} updates".format("epoch-1 norm" if quantity == "epoch1" else "batch component", update))
    axes[0].set_title("Relative variance", fontsize=11)
    axes[0].set_ylabel("Relative variance (no units)", fontsize=10)
    axes[1].set_title("Mean along the on-policy direction", fontsize=11)
    axes[1].set_ylabel("Mean component (training-loss units)", fontsize=10)
    for col, axis in enumerate(axes):
        style(axis, log=col == 0)
        axis.set_xlabel("Updates (one batch = 80 episodes)", fontsize=10)
        axis.set_xticks(UPDATES)
        axis.set_xlim(28, 99)
    limits = decade_limits(rv)
    if limits is not None:
        axes[0].set_ylim(*limits)
    if signed:
        low, high = min(0.0, min(signed)), max(0.0, max(signed))
        padding = (high - low) * 0.10 or 1.0
        axes[1].set_ylim(low - padding, high + padding)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), frameon=False, fontsize=8.8, ncol=1)
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    caption = (
        "Relative variance and mean of DPPO's batch-average gradient against the on-policy gradient at the same parameters (epoch 1's average). "
        "'Against the on-policy direction' divides the variance by the squared component of the mean along the on-policy mean. "
        "Dashed: {}'s one-update gradient, measured in its own run (different parameters). ".format(method)
        + "The mean-component panel is linear so signed components remain visible. " + missing_text(omissions)
    )
    return figure, records, caption


def heatmap_figure(points):
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.9))
    figure.subplots_adjust(left=0.065, right=0.89, bottom=0.20, top=0.82, wspace=0.20)
    figure.suptitle("DPPO", fontsize=14, y=0.97)
    records, omissions, outside = [], [], []
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    for axis, update in zip(axes, UPDATES):
        turning = points[update]["turning"]
        raw = [[finite(metric) for metric in row] for row in turning["cosine"]]
        matrix = np.asarray([[np.nan if value is None else value for value in row] for row in raw], dtype=np.float64)
        image = axis.imshow(np.ma.masked_invalid(matrix), origin="lower", vmin=-1, vmax=1, cmap=cmap,
                            extent=(4.5, 20.5, 4.5, 20.5), interpolation="none")
        axis.set_xticks((5, 9, 13, 17))
        axis.set_yticks((5, 9, 13, 17))
        axis.set_xlabel("PPO step", fontsize=10)
        axis.set_ylabel("PPO step", fontsize=10)
        axis.set_title("{} updates".format(update), fontsize=11)
        axis.tick_params(labelsize=9)
        absent = [str(step) for index, step in enumerate(turning["steps"]) if not turning["Gamma"][index][index]["resolved"]]
        if absent:
            omissions.append("{} updates: all cells involving steps {}".format(update, ", ".join(absent)))
        other_missing = [(i + 5, j + 5) for i in range(16) for j in range(i, 16) if raw[i][j] is None
                         and str(i + 5) not in absent and str(j + 5) not in absent]
        if other_missing:
            omissions.append("{} updates: cells {}".format(update, ", ".join("({},{})".format(*pair) for pair in other_missing)))
        out_pairs = [(i + 5, j + 5, raw[i][j]) for i in range(16) for j in range(i + 1, 16)
                     if raw[i][j] is not None and abs(raw[i][j]) > 1.0]
        outside.extend({"update": update, "step_pair": pair[:2], "estimate": pair[2]} for pair in out_pairs)
        records.append({"update": update, "steps": turning["steps"], "cosine": raw, "color_limits": [-1, 1]})
    color_axis = figure.add_axes([0.915, 0.23, 0.017, 0.53])
    figure.colorbar(image, cax=color_axis, label=r"$\cos(\Delta_s,\Delta_r)$", ticks=(-1, 0, 1))
    caption = (
        "Cosine between DPPO's extra means Δ_s = E g_s − E g_1 at steps s and r, from between-group products of the stored group sums. "
        + missing_text(omissions)
    )
    if outside:
        caption += " Estimates outside [−1, 1] retain their values in the table and reach the fixed color scale's endpoint: {}.".format(
            "; ".join("{} updates, steps {}/{}: {:.6g}".format(row["update"], row["step_pair"][0], row["step_pair"][1], row["estimate"]) for row in outside))
    return figure, records, caption


def path_figure(points):
    selected, omissions, records = {}, [], []
    for update in UPDATES:
        turning = points[update]["turning"]
        selected[update] = []
        omitted = []
        for index, step in enumerate(turning["steps"]):
            coords = [finite(metric) for metric in turning["coordinates"][index]]
            if turning["Gamma"][index][index]["resolved"] and all(value is not None for value in coords):
                selected[update].append((step, coords))
            else:
                omitted.append(str(step))
        if omitted:
            omissions.append("{} updates: steps {}".format(update, ", ".join(omitted)))
    values = [coordinate for row in selected.values() for _, coords in row for coordinate in coords] + [0.0]
    low, high = min(values), max(values)
    padding = (high - low) * 0.10 or 1.0
    limits = (low - padding, high + padding)
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.9))
    figure.subplots_adjust(left=0.085, right=0.89, bottom=0.23, top=0.82, wspace=0.35)
    figure.suptitle("DPPO", fontsize=14, y=0.97)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("DPPO_steps", ["#b9d8e9", BLUE])
    norm = Normalize(vmin=5, vmax=20)
    for axis, update in zip(axes, UPDATES):
        by_step = dict(selected[update])
        xx = [by_step[step][0] if step in by_step else np.nan for step in range(5, 21)]
        yy = [by_step[step][1] if step in by_step else np.nan for step in range(5, 21)]
        axis.plot(xx, yy, color=BLUE, linewidth=0.8, alpha=0.55)
        if selected[update]:
            axis.scatter([coords[0] for _, coords in selected[update]], [coords[1] for _, coords in selected[update]],
                         c=[step for step, _ in selected[update]], cmap=cmap, norm=norm, marker="o", s=27,
                         edgecolors="none", zorder=3)
        axis.axhline(0, color=GRAY, alpha=0.45, linewidth=0.7)
        axis.axvline(0, color=GRAY, alpha=0.45, linewidth=0.7)
        style(axis)
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("{} updates".format(update), fontsize=11)
        axis.set_xlabel("PC1 (training-loss units)", fontsize=9)
        axis.set_ylabel("PC2 (training-loss units)", fontsize=9)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
        records.append({"update": update, "steps": [step for step, _ in selected[update]],
                        "coordinates": [coords for _, coords in selected[update]], "xlim": limits, "ylim": limits})
    color_axis = figure.add_axes([0.915, 0.25, 0.017, 0.51])
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(mappable, cax=color_axis, label="PPO step", ticks=(5, 9, 13, 17, 20))
    caption = (
        "DPPO's extra means Δ_5 … Δ_20 in their top two principal coordinates (uncentered, so the origin is no extra mean), light to dark by step. "
        "Shares of each coordinate are in batch_update_table. " + missing_text(omissions)
    )
    return figure, records, caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preservation-check", type=Path)
    args = parser.parse_args()
    data, points = load(args.analysis)
    gates_pass(data, args.preservation_check)
    target = args.output / "figs"
    paths = [target / (name + suffix) for name in NAMES for suffix in (".png", ".pdf", ".caption.md")]
    paths.append(target / "render_verification.json")
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite an existing plot or verification file")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                               "ps.fonttype": 42, "savefig.facecolor": "white", "axes.unicode_minus": True})
    builds = (epoch_figure, comparison_figure, heatmap_figure, path_figure)
    pending, checks = [], {"analysis": str(args.analysis.resolve()), "figures": {},
                           "style": {"png_dpi": 300, "markers": "filled circles", "intervals_drawn": False}}
    # Construct and validate every figure before writing any, so a display
    # conflict cannot leave an apparently complete partial bundle.
    for name, build in zip(NAMES, builds):
        figure, records, caption = build(points)
        checks["figures"][name] = {"data": records, "render": bounds_check(figure)}
        pending.append((name, figure, caption))
    target.mkdir(parents=True, exist_ok=True)
    for name, figure, caption in pending:
        figure.savefig(str(target / (name + ".png")), dpi=300)
        figure.savefig(str(target / (name + ".pdf")))
        (target / (name + ".caption.md")).write_text(caption + "\n", encoding="utf-8")
        plt.close(figure)
    (target / "render_verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "output": str(target.resolve()), "figures": list(NAMES)}))


if __name__ == "__main__":
    main()
