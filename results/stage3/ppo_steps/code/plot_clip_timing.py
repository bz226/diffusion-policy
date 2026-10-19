#!/usr/bin/env python3
"""Draw the approved DPPO clip-timing analysis from saved scalar estimates.

This script neither computes gradients nor reads model/checkpoint files.  It
requires the statistical gates to pass before rendering, writes only new files,
and preserves the exact plotted coordinates alongside its render checks.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text


BLUE = "#004d7a"
GRAY = "#8b8b8b"
UPDATES = (32, 63, 95)
TICKS = (1, 5, 9, 13, 17)
NORM_LIMITS = (1e-5, 1e-2)
VARIANCE_LIMITS = (1e-5, 1e-3)
STEP_LABEL = "PPO step within the batch (4 per epoch)"


def number(metric):
    value = metric.get("estimate") if isinstance(metric, dict) else metric
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def load_analysis(path):
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    rows = data["rows"]
    summaries = {int(row["update"]): row for row in data["summaries"]}
    grouped = {}
    for update in UPDATES:
        point = sorted(
            (row for row in rows if int(row["update"]) == update),
            key=lambda row: int(row["step"]),
        )
        if [int(row["step"]) for row in point] != list(range(1, 21)):
            raise ValueError("Expected exactly 20 ordered steps at {} updates".format(update))
        grouped[update] = point
        if update not in summaries:
            raise ValueError("Missing summary at {} updates".format(update))
    return data, grouped, summaries


def require_gate_pass(data):
    """Allow a separate gate file to be checked by the caller via --gate-file."""
    gates = data.get("gates")
    if gates is None:
        raise ValueError("Gate evidence is required (--gate-file if not in analysis.json)")
    if isinstance(gates, dict) and "checks" in gates:
        gates = gates["checks"]
    if isinstance(gates, list):
        gates = {item.get("name", item.get("gate")): item for item in gates}
    for name in ("G1", "G2", "G3", "G4"):
        result = gates.get(name) if isinstance(gates, dict) else None
        if isinstance(result, bool):
            passed = result
        elif isinstance(result, dict):
            passed = result.get("passed", result.get("pass", False)) is True
            passed = passed or result.get("status") in ("passed", "pass")
        else:
            passed = result in ("passed", "pass")
        if not passed:
            raise ValueError("Refusing to render before {} passes".format(name))


def style_axis(axis, log=False):
    if log:
        axis.set_yscale("log")
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", which="major", color="#e2e2e2", linewidth=0.65)
    axis.grid(False, axis="x", which="both")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=9)
    axis.minorticks_off()


def points(axis, x, y):
    axis.plot(
        x, y, color=BLUE, linewidth=1.25, marker="o", markersize=3.6,
        markerfacecolor=BLUE, markeredgecolor=BLUE, markeredgewidth=0.4,
    )


def linear_limit(values, minimum_upper=0.0):
    values = [float(value) for value in values if value is not None]
    if not values:
        raise ValueError("No finite values for a required linear panel")
    if min(values) < 0:
        raise ValueError("Negative plotted estimate conflicts with requested zero-based axis")
    upper = max(max(values) * 1.08, minimum_upper)
    return (0.0, upper if upper > 0 else 1.0)


def check_log_values(grouped, key, limits):
    for update, rows in grouped.items():
        for row in rows:
            value = number(row[key])
            if value is None or value <= 0 or not limits[0] <= value <= limits[1]:
                raise ValueError(
                    "{} at {} updates, step {} cannot be displayed in preserved limits {}".format(
                        key, update, row["step"], limits
                    )
                )


def row_labels(figure, axes):
    for update, row in zip(UPDATES, axes):
        bounds = row[0].get_position()
        figure.text(
            0.013, bounds.y0 + bounds.height / 2.0,
            "{} updates".format(update), rotation=90,
            ha="center", va="center", fontsize=10.5, fontweight="medium",
        )


def timing_figure(grouped):
    check_log_values(grouped, "mean_norm", NORM_LIMITS)
    check_log_values(grouped, "variance", VARIANCE_LIMITS)
    limits = {
        "f": linear_limit([number(row["f"]) for rows in grouped.values() for row in rows]),
        "abs_ratio": linear_limit(
            [number(row["abs_ratio"]) for rows in grouped.values() for row in rows], 0.011
        ),
        "mean_norm": NORM_LIMITS,
        "variance": VARIANCE_LIMITS,
    }
    specs = (
        ("f", "Pairs switched off by the clip", "fraction of pairs"),
        ("abs_ratio", r"Mean $|\rho-1|$", r"mean $|\rho-1|$"),
        ("mean_norm", "Mean gradient norm", r"$\|\mathrm{E}\,g_s\|$ (training-loss units)"),
        ("variance", "Gradient variance (one step)", r"$\mathrm{tr}\,\mathrm{Cov}(g_s)$ (training-loss units$^2$)"),
    )
    figure, axes = plt.subplots(3, 4, figsize=(13.2, 8.5), squeeze=False)
    figure.subplots_adjust(left=0.088, right=0.987, bottom=0.085, top=0.902, wspace=0.48, hspace=0.31)
    figure.suptitle("DPPO", fontsize=15, y=0.975)
    figure.supxlabel(STEP_LABEL, fontsize=11, y=0.023)
    plotted = []
    for row_index, update in enumerate(UPDATES):
        rows = grouped[update]
        for col_index, (key, title, ylabel) in enumerate(specs):
            axis = axes[row_index, col_index]
            x = [int(row["step"]) for row in rows]
            y = [number(row[key]) for row in rows]
            if any(value is None for value in y):
                raise ValueError("Missing timing estimate for {} at {} updates".format(key, update))
            points(axis, x, y)
            style_axis(axis, log=key in ("mean_norm", "variance"))
            axis.set_xlim(0.5, 20.5)
            axis.set_xticks(TICKS)
            axis.set_ylim(*limits[key])
            axis.set_ylabel(ylabel, fontsize=9.4)
            if row_index == 0:
                axis.set_title(title, fontsize=11, pad=13)
            if key == "abs_ratio":
                axis.axhline(0.01, color=GRAY, alpha=0.65, linestyle="--", linewidth=1.0)
            plotted.append({"update": update, "metric": key, "x": x, "y": y, "ylim": list(limits[key])})
    row_labels(figure, axes)
    return figure, plotted


def shape_figure(grouped, summaries):
    left_rows = {
        update: [row for row in rows if int(row["step"]) >= 2 and row["delta_resolved"] and number(row["delta_norm"]) is not None]
        for update, rows in grouped.items()
    }
    right_rows = {
        update: [row for row in rows if int(row["step"]) >= 5 and row["delta_resolved"] and number(row["cos_delta20"]) is not None]
        for update, rows in grouped.items()
    }
    max_c = max(float(row["C"]) for rows in grouped.values() for row in rows)
    slopes = {update: number(summaries[update]["k"]) for update in UPDATES}
    if any(value is None or value <= 0 for value in slopes.values()):
        raise ValueError("A steps-5–8 fitted slope is unavailable or nonpositive")
    delta_limits = linear_limit(
        [number(row["delta_norm"]) for rows in left_rows.values() for row in rows]
        + [slope * max_c for slope in slopes.values()]
    )
    # The user approved [-1, 1] to show resolved negative direction cosines.
    # Keep every point estimate unchanged and refuse to hide an out-of-range one.
    cosine_limits = (-1.0, 1.0)
    cosine_values = [number(row["cos_delta20"]) for rows in right_rows.values() for row in rows]
    if any(value < -1.0 or value > 1.0 for value in cosine_values):
        raise ValueError("A resolved cosine lies outside the approved [-1, 1] axis")
    figure, axes = plt.subplots(3, 2, figsize=(8.8, 8.5), squeeze=False)
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.094, top=0.902, wspace=0.42, hspace=0.30)
    figure.suptitle("DPPO", fontsize=15, y=0.975)
    plotted = []
    for row_index, update in enumerate(UPDATES):
        left, right = axes[row_index]
        rows = left_rows[update]
        x = [float(row["C"]) for row in rows]
        y = [number(row["delta_norm"]) for row in rows]
        points(left, x, y)
        left.plot([0.0, max_c], [0.0, slopes[update] * max_c], color=GRAY, alpha=0.65, linestyle="--", linewidth=1.0)
        style_axis(left)
        left.set_xlim(-0.03, max_c * 1.04)
        left.set_ylim(*delta_limits)
        left.set_ylabel(r"$\|\mathrm{E}\,g_s-\mathrm{E}\,g_1\|$ (training-loss units)", fontsize=9.5)
        rows_right = right_rows[update]
        xr = [int(row["step"]) for row in rows_right]
        yr = [number(row["cos_delta20"]) for row in rows_right]
        points(right, xr, yr)
        style_axis(right)
        right.set_xlim(0.5, 20.5)
        right.set_xticks(TICKS)
        right.set_ylim(*cosine_limits)
        right.set_ylabel(r"$\cos(\Delta_s,\Delta_{20})$", fontsize=10)
        if row_index == 0:
            left.set_title("Extra mean vs own pushes applied", fontsize=11, pad=13)
            right.set_title("Direction of the extra mean", fontsize=11, pad=13)
        if row_index == 2:
            left.set_xlabel(r"$C_s$ (own pushes applied, in full pushes)", fontsize=9.5, labelpad=8)
            right.set_xlabel("PPO step within the batch\n(4 per epoch)", fontsize=9.5, labelpad=8)
        plotted.extend([
            {"update": update, "metric": "delta_norm", "steps": [int(row["step"]) for row in rows], "x": x, "y": y, "ylim": list(delta_limits), "reference_slope": slopes[update]},
            {"update": update, "metric": "cos_delta20", "x": xr, "y": yr, "ylim": list(cosine_limits)},
        ])
    row_labels(figure, axes)
    return figure, plotted, left_rows, right_rows


def bounds_check(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    # Matplotlib creates extra log ticks outside the view interval but Axis.draw
    # renders only _update_ticks()'s filtered list.  Do not mistake those unused
    # tick-label Text objects for text that is actually drawn beyond the canvas.
    undrawn_tick_labels = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            drawn_ticks = {id(tick) for tick in axis._update_ticks()}
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(tick) not in drawn_ticks:
                    undrawn_tick_labels.update((id(tick.label1), id(tick.label2)))
    outside = []
    texts = []
    for artist in figure.findobj(match=Text):
        if id(artist) in undrawn_tick_labels or not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        bounds = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
        record = {"text": artist.get_text(), "bounds_pixels": bounds}
        texts.append(record)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > width + 1 or bbox.y1 > height + 1:
            outside.append(record)
    if outside:
        raise ValueError("Canvas text is outside the exported figure: {}".format(outside))
    return {"canvas_pixels": [width, height], "all_visible_text_inside_canvas": True,
            "undrawn_tick_label_objects_excluded": len(undrawn_tick_labels), "text_bounds": texts}


def omission_sentence(grouped, selected, start_step):
    parts = []
    for update in UPDATES:
        present = {int(row["step"]) for row in selected[update]}
        missing = [int(row["step"]) for row in grouped[update] if int(row["step"]) >= start_step and int(row["step"]) not in present]
        if missing:
            parts.append("{} updates: steps {}".format(update, ", ".join(str(step) for step in missing)))
    return "; ".join(parts) if parts else "none"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New candidate directory; figures are written in its figs/ subdirectory")
    parser.add_argument("--gate-file", type=Path, help="Optional gate JSON when analysis.json does not embed gates")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--timing-only", action="store_true", help="Render Figure 1 only")
    selection.add_argument("--shape-only", action="store_true", help="Render Figure 2 only, using the user-approved cosine axis [-1, 1]")
    args = parser.parse_args()
    data, grouped, summaries = load_analysis(args.analysis)
    if args.gate_file is not None:
        with args.gate_file.open("r", encoding="utf-8") as stream:
            gate_data = json.load(stream)
        data["gates"] = gate_data.get("gates", gate_data)
    require_gate_pass(data)
    target = args.output / "figs"
    if args.timing_only:
        names = ("dppo_clip_timing_grid",)
    elif args.shape_only:
        names = ("dppo_reuse_mean_shape_grid",)
    else:
        names = ("dppo_clip_timing_grid", "dppo_reuse_mean_shape_grid")
    expected = [target / (name + suffix) for name in names for suffix in (".png", ".pdf", ".caption.md")]
    expected.append(target / "render_verification.json")
    if any(path.exists() for path in expected):
        raise FileExistsError("Refusing to overwrite an existing plot or verification file")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True, "savefig.facecolor": "white",
    })
    checks = {
        "analysis_file": str(args.analysis.resolve()),
        "figures": {},
        "style": {"point_color": BLUE, "point_marker": "filled circle", "intervals_drawn": False, "png_dpi": 300},
    }
    if not args.shape_only:
        timing, timing_points = timing_figure(grouped)
        checks["figures"]["dppo_clip_timing_grid"] = {"data": timing_points, "render": bounds_check(timing)}
    if args.timing_only:
        checks["shape_figure"] = {
            "status": "withheld",
            "reason": "Not requested in --timing-only mode.",
        }
    else:
        shape, shape_points, left_rows, right_rows = shape_figure(grouped, summaries)
        checks["figures"]["dppo_reuse_mean_shape_grid"] = {"data": shape_points, "render": bounds_check(shape)}
        checks["cosine_axis_approval"] = "User approved [-1, 1] to retain resolved negative cosines; estimates are unchanged."
    caption_timing = (
        "DPPO's 20 PPO steps on one batch, from the rerun's own state after 32, 63 and 95 updates (rows): "
        "the fraction of pairs the clip switches off, the mean |ρ − 1| (dashed: the clip edge ε = 0.01), "
        "the mean gradient norm and the gradient variance. 500 batches of 80 episodes per checkpoint; "
        "gradients in training-loss units. Points are estimates; 90% intervals are in clip_timing_table. "
        "No points are omitted: this figure retains the stored point estimates even where a mean is unresolved.\n"
    )
    figures = []
    captions = []
    if not args.shape_only:
        figures.append(timing)
        captions.append(caption_timing)
    if not args.timing_only:
        caption_shape = (
            "The extra mean that reuse adds to DPPO's step gradient, Δ_s = E g_s − E g_1. "
            "Left: its norm against C_s, the amount of each pair's own earlier pushes applied by step s; "
            "dashed: a line through the origin fitted to steps 5–8. Under the second-order model the points stay on the line. "
            "Right: the cosine between Δ_s and Δ_20. Inner products from between-group products of the stored group sums, "
            "500 batches per checkpoint. Points are estimates; 90% intervals are in clip_timing_table. "
            "Omitted as unresolved or unavailable, left (steps 2–20): {}. "
            "Omitted as unresolved or unavailable, right (steps 5–20): {}.\n"
        ).format(omission_sentence(grouped, left_rows, 2), omission_sentence(grouped, right_rows, 5))
        figures.append(shape)
        captions.append(caption_shape)
    target.mkdir(parents=True, exist_ok=True)
    for name, figure, caption in zip(names, figures, captions):
        figure.savefig(str(target / (name + ".png")), dpi=300)
        figure.savefig(str(target / (name + ".pdf")))
        (target / (name + ".caption.md")).write_text(caption, encoding="utf-8")
        plt.close(figure)
    (target / "render_verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "output": str(target.resolve()), "figures": list(names)}))


if __name__ == "__main__":
    main()
