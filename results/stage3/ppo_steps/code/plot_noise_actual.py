#!/usr/bin/env python3
"""Render stored DPPO gradient statistics in optimizer-gradient units.

The calling analysis supplies the already transformed point estimates, shared
limits and captions. This module performs no statistical estimation, reads no
run data and never overwrites an existing artifact. Compatible with Python 3.8.
"""

import math
from pathlib import Path


UPDATES = (32, 63, 95)
KEYS = ("mean_norm", "variance", "relative_variance")
COLOR = "#004d7a"
REFERENCE_COLOR = "#009E73"
TICKS = (1, 5, 9, 13, 17)
XLABEL = "PPO step within the batch (4 per epoch)"
TITLES = (
    "Mean gradient\nnorm",
    "Gradient\nvariance\n(one step)",
    "Relative\nvariance\n(one step)",
)
YLABELS = (
    r"$\|\mathrm{E}\,g_s\|$" + "\n(training-loss units)",
    r"$\mathrm{tr}\,\mathrm{Cov}(g_s)$" + "\n(training-loss units²)",
    r"$\mathrm{tr}\,\mathrm{Cov}(g_s)/\|\mathrm{E}\,g_s\|^2$" + "\n(no units)",
)
REFERENCE_LABEL = (
    "without_PPO_and_GAE, one update (its own run at N updates)"
)


def _inputs(rows, limits):
    points = {}
    references = {}
    for update in UPDATES:
        selected = sorted(
            [row for row in rows
             if row["method"] == "DPPO" and int(row["update"]) == update],
            key=lambda row: int(row["step"]),
        )
        if [int(row["step"]) for row in selected] != list(range(1, 21)):
            raise ValueError("Expected exactly steps 1–20 at %d updates" % update)
        points[update] = {
            key: [float(row[key]) for row in selected] for key in KEYS
        }
        reference = [row for row in rows
                     if row["method"] == "without_PPO_and_GAE"
                     and int(row["update"]) == update]
        if len(reference) != 1 or reference[0]["step"] is not None:
            raise ValueError("Expected one update reference at %d" % update)
        references[update] = {key: float(reference[0][key]) for key in KEYS}
    checked_limits = {}
    for key in KEYS:
        low, high = (float(value) for value in limits[key])
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            raise ValueError("Invalid shared limits for %s" % key)
        for value in (low, high):
            if abs(math.log10(value) - round(math.log10(value))) > 1e-10:
                raise ValueError("Shared limits must be whole decades")
        all_values = [value for update in UPDATES for value in points[update][key]]
        all_values.extend(references[update][key] for update in UPDATES)
        if any(not math.isfinite(value) or not low < value < high
               for value in all_values):
            raise ValueError("Nonpositive, nonfinite or out-of-bounds %s" % key)
        checked_limits[key] = (low, high)
    return points, references, checked_limits


def _panel(ax, values, key, limits, title, ylabel, font_size, reference=None):
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    x = list(range(1, 21))
    line, = ax.plot(
        x, values, "o-", color=COLOR, markerfacecolor=COLOR,
        markeredgecolor=COLOR, markersize=2.4, linewidth=0.9,
    )
    reference_line = None
    if reference is not None:
        reference_line = ax.axhline(
            reference, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9,
            alpha=0.40, label=REFERENCE_LABEL,
        )
    ax.set_yscale("log")
    ax.set_ylim(limits[key])
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(TICKS)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=font_size, length=2.5, pad=1.5)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_title(title, fontsize=font_size + 0.5, pad=5)
    ax.set_ylabel(ylabel, fontsize=font_size, labelpad=1.5)
    ax.grid(axis="y", which="major", color="0.90", linewidth=0.5)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # These checks inspect rendered objects rather than merely the inputs.
    assert list(line.get_xdata()) == x
    assert list(line.get_ydata()) == values
    assert line.get_marker() == "o"
    assert line.get_markerfacecolor() == COLOR
    assert tuple(ax.get_ylim()) == limits[key]
    assert ax.get_yscale() == "log"
    assert list(ax.get_xticks()) == list(TICKS)
    assert len(ax.lines) == (2 if reference is not None else 1)
    assert len(ax.collections) == 0
    if reference_line is not None:
        assert list(reference_line.get_ydata()) == [reference, reference]
        assert reference_line.get_linestyle() == "--"
        assert reference_line.get_color() == REFERENCE_COLOR
    return {
        "quantity": key,
        "x": x,
        "y": values,
        "ylim": list(ax.get_ylim()),
        "xticks": list(TICKS),
        "log_y": True,
        "reference": reference,
        "coordinates_and_limits_checked": True,
        "data_lines": len(ax.lines),
        "errorbar_or_band_collections": len(ax.collections),
    }


def _caption(captions, name):
    if name in captions:
        return captions[name]
    if name + ".png" in captions:
        return captions[name + ".png"]
    raise KeyError("No supplied caption for %s" % name)


def _check_layout(fig):
    """Check exported-size text bounds and separation of legends/titles."""
    fig.set_dpi(300)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    artists = list(fig.texts)
    titles = []
    for ax in fig.axes:
        artists.extend((ax.title, ax.xaxis.label, ax.yaxis.label))
        if ax.title.get_text():
            titles.append(ax.title.get_window_extent(renderer))
    for legend in fig.legends:
        artists.extend(legend.get_texts())
    checked = []
    for artist in artists:
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        # Half a pixel covers floating-point layout tolerances, not clipping.
        assert box.x0 >= canvas.x0 - 0.5, "Text clipped at left: " + artist.get_text()
        assert box.x1 <= canvas.x1 + 0.5, "Text clipped at right: " + artist.get_text()
        assert box.y0 >= canvas.y0 - 0.5, "Text clipped at bottom: " + artist.get_text()
        assert box.y1 <= canvas.y1 + 0.5, "Text clipped at top: " + artist.get_text()
        checked.append(artist.get_text())
    for legend in fig.legends:
        legend_box = legend.get_window_extent(renderer)
        assert not any(legend_box.overlaps(title_box) for title_box in titles), (
            "Legend overlaps a column title"
        )
    return {
        "text_bounds_inside_canvas": True,
        "legend_column_titles_disjoint": True,
        "checked_text": checked,
        "checked_dpi": 300,
    }


def _save(fig, output, name, caption, checks, plt):
    files = []
    try:
        layout_checks = _check_layout(fig)
        for extension in ("png", "pdf"):
            path = output / (name + "." + extension)
            metadata = {"Description": caption} if extension == "png" else {"Subject": caption}
            fig.savefig(str(path), dpi=300, metadata=metadata)
            files.append(str(path))
        path = output / (name + ".caption.md")
        path.write_text(caption + "\n", encoding="utf-8")
        files.append(str(path))
    finally:
        plt.close(fig)
    return {"name": name, "files": files, "panels": checks, "layout": layout_checks}


def render(rows, limits, out, captions):
    """Write five new DPPO views and return artifact/verification records.

    ``rows`` contains stored estimates for both methods, with keys ``method``,
    ``update``, ``step`` and the three names in ``KEYS``. ``limits`` supplies
    the common whole-decade limits for every new figure in the comparison.
    Captions are keyed by output stem (or PNG filename).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points, references, shared_limits = _inputs(rows, limits)
    output = Path(out)
    names = ["dppo_per_step_noise_actual_u%d" % update for update in UPDATES]
    names.extend(("dppo_per_step_noise_actual_grid", "dppo_per_step_noise_actual_grid_ref"))
    # Refuse partial replacement before writing any output.
    for name in names:
        _caption(captions, name)
        for extension in ("png", "pdf", "caption.md"):
            if (output / (name + "." + extension)).exists():
                raise FileExistsError("Existing artifact: %s" % (output / (name + "." + extension)))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    settings = {
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(settings):
        for update in UPDATES:
            fig, axes = plt.subplots(1, 3, figsize=(3.6, 2.5), constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.035, wspace=0.025)
            checks = []
            for ax, key, title, ylabel in zip(axes, KEYS, TITLES, YLABELS):
                check = _panel(ax, points[update][key], key, shared_limits,
                               title, ylabel, 6.5)
                check["update"] = update
                checks.append(check)
            fig.suptitle("DPPO, %d updates" % update, fontsize=9)
            fig.supxlabel(XLABEL, fontsize=7)
            name = "dppo_per_step_noise_actual_u%d" % update
            records.append(_save(fig, output, name, _caption(captions, name), checks, plt))

        for with_reference in (False, True):
            fig, axes = plt.subplots(3, 3, figsize=(6.4, 6.9))
            fig.subplots_adjust(left=0.17, right=0.98, bottom=0.10,
                                top=0.84 if with_reference else 0.89,
                                wspace=0.60, hspace=0.35)
            checks = []
            for row_index, update in enumerate(UPDATES):
                for col_index, (key, title, ylabel) in enumerate(zip(KEYS, TITLES, YLABELS)):
                    check = _panel(
                        axes[row_index, col_index], points[update][key], key,
                        shared_limits, title if row_index == 0 else "", ylabel, 8,
                        references[update][key] if with_reference else None,
                    )
                    check["update"] = update
                    checks.append(check)
                position = axes[row_index, 0].get_position()
                fig.text(0.018, (position.y0 + position.y1) / 2.0,
                         "%d updates" % update, rotation=90, ha="center",
                         va="center", fontsize=9)
            fig.suptitle("DPPO", fontsize=11, y=0.98)
            fig.supxlabel(XLABEL, fontsize=9, y=0.025)
            if with_reference:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                assert labels == [REFERENCE_LABEL] and len(handles) == 1
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
                           frameon=False, fontsize=7.3, handlelength=2.5)
            name = "dppo_per_step_noise_actual_grid" + ("_ref" if with_reference else "")
            records.append(_save(fig, output, name, _caption(captions, name), checks, plt))
    return records
