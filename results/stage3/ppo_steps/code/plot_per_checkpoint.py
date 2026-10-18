#!/usr/bin/env python3
"""Re-layout saved DPPO Part C point estimates; Python 3.8, no aggregation."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
POINTS = ((35, 32), (70, 63), (105, 95))
KEYS = ("u_norm", "trace_ep", "B")
COLOR = "#004d7a"  # Existing Figure 2's 95-update shade.
SPAN = 4.0
TICKS = (1, 5, 9, 13, 17)
XLABEL = "PPO step within the batch (4 per epoch)"
TITLES = ("Mean gradient\nnorm", "Gradient variance\nper episode", "Relative\nvariance")
YLABELS = (r"$\|\mathrm{E}\,g_s\|$ (loss units)",
           r"$\mathrm{tr}\,\Sigma_{\mathrm{ep},s}$ (loss units$^2$)",
           r"$B_s$ (episodes)")


def load_points(table_bytes):
    rows = [r for r in csv.DictReader(io.StringIO(table_bytes.decode("utf-8")))
            if r["part"] == "C"]
    if len(rows) != 60:
        raise ValueError("Expected the existing 60 complete Part C rows")
    points = {}
    for itr, updates in POINTS:
        selected = sorted([r for r in rows if int(r["updates"]) == updates],
                          key=lambda r: int(r["step"]))
        if [int(r["step"]) for r in selected] != list(range(1, 21)):
            raise ValueError("Missing or duplicate PPO steps at %d updates" % updates)
        for r in selected:
            if (r["method"] != "DPPO" or int(r["checkpoint_itr"]) != itr or
                    int(r["K"]) != 500 or int(r["groups"]) != 10 or r["status"] != "complete"):
                raise ValueError("Unexpected source row: %r" % r)
        values = {key: [float(r[key]) for r in selected] for key in KEYS}
        if any(not math.isfinite(v) or v <= 0 for column in values.values() for v in column):
            raise ValueError("All stored point estimates must be positive and finite")
        points[updates] = values
    return points


def shared_limits(points):
    limits = {}
    for key in KEYS:
        values = [v for point in points.values() for v in point[key]]
        low, high = min(values), max(values)
        if math.log10(high / low) >= SPAN:
            raise ValueError("The requested four-decade range cannot contain %s" % key)
        center = (math.log10(low) + math.log10(high)) / 2.0
        limits[key] = (10 ** (center - SPAN / 2), 10 ** (center + SPAN / 2))
    return limits


def caption(updates=None):
    state = ("after %d updates" % updates if updates is not None
             else "after 32, 63 and 95 updates (rows)")
    return ("DPPO's gradient at each of the 20 PPO steps on one batch, from the run's own state "
            + state + ". 500 batches of 80 episodes per checkpoint (10 groups), in training-loss "
            "units. Points are estimates; 90% intervals are in ppo_steps_table.")


def panel(ax, values, key, limits, title, label, font_size):
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    x = list(range(1, 21))
    line, = ax.plot(x, values, "o-", color=COLOR, markerfacecolor=COLOR,
                    markeredgecolor=COLOR, markersize=2.4, linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylim(limits[key])
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(TICKS)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=font_size, length=2.5, pad=1.5)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_title(title, fontsize=font_size + 0.5, pad=5)
    ax.set_ylabel(label, fontsize=font_size, labelpad=1.5)
    ax.grid(axis="y", which="major", color="0.90", linewidth=0.5)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Verify the drawn coordinates against the saved values, without any refit.
    assert list(line.get_xdata()) == x and list(line.get_ydata()) == values
    assert tuple(ax.get_ylim()) == limits[key]
    assert list(ax.get_xticks()) == list(TICKS)
    assert len(ax.lines) == 1 and len(ax.collections) == 0


def save(fig, out, name, text, plt):
    for ext in ("png", "pdf"):
        metadata = {"Description": text} if ext == "png" else {"Subject": text}
        fig.savefig(str(out / (name + "." + ext)), dpi=300, metadata=metadata)
    (out / (name + ".caption.md")).write_text(text + "\n")
    plt.close(fig)


def render(points, limits, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    names = []
    for itr, updates in POINTS:
        fig, axes = plt.subplots(1, 3, figsize=(3.6, 2.35), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.035, wspace=0.025)
        for ax, key, title, label in zip(axes, KEYS, TITLES, YLABELS):
            panel(ax, points[updates][key], key, limits, title, label, 6.5)
        fig.suptitle("DPPO, %d updates" % updates, fontsize=9)
        fig.supxlabel(XLABEL, fontsize=7)
        name = "dppo_per_step_noise_u%d" % updates
        save(fig, out, name, caption(updates), plt)
        names.append(name)

    fig, axes = plt.subplots(3, 3, figsize=(6.4, 6.4))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.91,
                        wspace=0.57, hspace=0.30)
    for row, (itr, updates) in enumerate(POINTS):
        for col, (key, title, label) in enumerate(zip(KEYS, TITLES, YLABELS)):
            panel(axes[row, col], points[updates][key], key, limits,
                  title if row == 0 else "", label, 8)
        pos = axes[row, 0].get_position()
        fig.text(0.018, (pos.y0 + pos.y1) / 2, "%d updates" % updates,
                 rotation=90, ha="center", va="center", fontsize=9)
    fig.suptitle("DPPO", fontsize=11, y=0.99)
    fig.supxlabel(XLABEL, fontsize=9, y=0.02)
    save(fig, out, "dppo_per_step_noise_grid", caption(), plt)
    names.append("dppo_per_step_noise_grid")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New candidate directory; existing outputs are never overwritten")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new output directory; existing results are preserved")
    table = ROOT / "candidates/final/ppo_steps_table.csv"
    table_bytes = table.read_bytes()
    if table_bytes != (ROOT / "curated/ppo_steps_table.csv").read_bytes():
        raise ValueError("Published and final-candidate estimate tables differ")
    # Only the requested old Figure 2 receives before/after hashes. Other saved
    # copies and the table are protected with byte comparisons, not new hashes.
    originals = [ROOT / "curated" / ("dppo_per_step_noise." + ext) for ext in ("png", "pdf")]
    before = {p: p.read_bytes() for p in originals}
    copies = {}
    for directory in (ROOT / "candidates/final/figs", ROOT / "candidates/layout_review/figs",
                      ROOT / "candidates/part_A/figs", ROOT / "figs"):
        for ext in ("png", "pdf"):
            p = directory / ("dppo_per_step_noise." + ext)
            if p.exists():
                copies[p] = p.read_bytes()
    hashes_before = {str(p.relative_to(ROOT)): hashlib.sha256(data).hexdigest()
                     for p, data in before.items()}
    points = load_points(table_bytes)
    limits = shared_limits(points)
    out.mkdir(parents=True)
    names = render(points, limits, out)
    assert table.read_bytes() == table_bytes
    assert all(p.read_bytes() == data for p, data in copies.items())
    assert all(p.read_bytes() == data for p, data in before.items())
    hashes_after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in originals}
    assert hashes_before == hashes_after
    record = {"status": "complete", "python": sys.version, "table": str(table),
              "columns": list(KEYS), "checkpoints": list(POINTS), "K": 500, "groups": 10,
              "color": COLOR, "span_decades": SPAN, "ylimits": limits, "xticks": list(TICKS),
              "figures": names, "original_figure_hashes_before": hashes_before,
              "original_figure_hashes_after": hashes_after,
              "old_figure_copies_byte_identical": len(copies) + len(before),
              "table_byte_identical": True, "drawn_values_equal_saved_points": True,
              "jobs_launched": False, "statistics_recomputed": False,
              "command": [sys.executable] + sys.argv}
    (out / "render_record.json").write_text(json.dumps(record, indent=2) + "\n")
    note = ["# DPPO per-step gradient noise: checkpoint views", "",
            "The figures use the saved Part C `u_norm`, `trace_ep` and `B` point estimates "
            "unchanged: 20 PPO steps at 32/63/95 updates, 500 batches per checkpoint. "
            "Each column uses the same centered four-decade limits in all four figures.", "",
            "| Quantity | Lower y-limit | Upper y-limit |", "|---|---:|---:|"]
    for title, key in zip(("Mean gradient norm (loss units)", "Variance per episode (loss units²)",
                           "Relative variance B (episodes)"), KEYS):
        note.append("| %s | %.12g | %.12g |" % (title, limits[key][0], limits[key][1]))
    note.extend(["", "Original Figure 2 PNG/PDF SHA-256 values agree before and after; "
                 "all existing copies and the source table are byte-identical. The requested hashes "
                 "and command are in [render_record.json](render_record.json). No jobs were launched "
                 "and no statistical estimates or intervals were recomputed.", "",
                 "File-location detail: existing Figure 2 and its table were stored under "
                 "`curated/` and `candidates/final/`, rather than the previously requested study-root "
                 "`figs/`. The new plots are published to the requested `figs/` path. One shared "
                 "x-axis label per figure avoids repeated long labels at single-column width. "
                 "There are no scientific deviations.", ""])
    (out / "NOTE.md").write_text("\n".join(note))
    print(json.dumps({"output": str(out), "figures": names, "ylimits": limits,
                      "originals_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
