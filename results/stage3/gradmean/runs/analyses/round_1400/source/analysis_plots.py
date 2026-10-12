"""Approved Addendum-G displays from analysis rows; no estimation or sampling.

render() writes only new candidate PNG/PDF files with adjacent captions. All
intervals and point/bound classifications must be supplied by the analyzer.
"""
from pathlib import Path
import math

import numpy as np


POINTS = ("P0", "D1", "D2", "N1", "N2")
VARIANTS = ("V1", "V2", "V3", "V4", "V5")
PATHS = (("DPPO", ("P0", "D1", "D2"), "#3068a0"),
         ("NC4", ("P0", "N1", "N2"), "#be572c"))


def _number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _format(value):
    number = _number(value)
    return "N/A" if number is None else "%.3g" % number


def _counts(rows):
    counts = {}
    for row in rows:
        point = row["point"]
        value = row.get("K")
        if point in counts and counts[point] != value:
            raise ValueError("Inconsistent batch count for " + point)
        counts[point] = value
    return "; ".join("%s: %s batches" % (point, counts.get(point, "N/A")) for point in POINTS)


def _interval(ax, x, y, low, high, color, log_y=False):
    """Do not replace nonpositive bounds with an arbitrary positive floor."""
    low, high = _number(low), _number(high)
    if low is None or high is None:
        return
    if low > high:
        raise ValueError("Reversed confidence interval")
    if log_y and low <= 0:
        if high > 0:
            ax.vlines(x, y, high, color=color, linewidth=1.0)
            ax.plot([x], [high], marker="_", color=color, markersize=7)
        ax.annotate("CI crosses 0", (x, y), xytext=(4, -13),
                    textcoords="offset points", color=color, fontsize=6.8)
        return
    ax.vlines(x, low, high, color=color, linewidth=1.0)
    ax.plot([x, x], [low, high], linestyle="none", marker="_", color=color, markersize=7)


def _path_panel(ax, rows, field, label, log_y=False, bounds=False):
    by_point = {row["point"]: row for row in rows}
    if len(by_point) != len(rows):
        raise ValueError("Duplicate point in path plot")
    unavailable = {}
    for path_name, points, color in PATHS:
        xs, ys = [], []
        for point in points:
            row = by_point.get(point, {})
            x, y = _number(row.get("theta_distance")), _number(row.get(field))
            bound = bounds and row.get("B_ep_type") == "lower_bound"
            if bound:
                y = _number(row.get("B_ep_lower_bound"))
            if x is None or y is None or (log_y and y <= 0):
                xs.append(np.nan if x is None else x)
                ys.append(np.nan)
                unavailable[point] = _format(row.get(field))
                continue
            xs.append(x)
            ys.append(np.nan if bound else y)
            ax.scatter([x], [y], marker="^" if bound else "o", s=30,
                       facecolors="none" if bound else color, edgecolors=color, zorder=3)
            ax.annotate(("≥ " if bound else "") + point, (x, y),
                        xytext=(4, 6 if path_name == "DPPO" else -11),
                        textcoords="offset points", fontsize=7, color=color)
            if not bound:
                _interval(ax, x, y, row.get(field + "_low"), row.get(field + "_high"), color, log_y)
        ax.plot(xs, ys, color=color, linewidth=1.3, label=path_name)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"Actor distance from $\theta_0$ ($\ell_2$)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.22, which="both")
    if unavailable:
        text = ", ".join(point + "=" + unavailable[point] for point in POINTS if point in unavailable)
        ax.text(0.02, 0.02, "Not on axis: " + text, transform=ax.transAxes,
                fontsize=6.8, va="bottom", wrap=True)


def _heatmap(plt, rows, labels, from_key, to_key, title):
    index = {label: i for i, label in enumerate(labels)}
    values = np.full((len(labels), len(labels)), np.nan)
    records = {}
    for row in rows:
        a, b = row[from_key], row[to_key]
        if a not in index or b not in index:
            raise ValueError("Unexpected heatmap label")
        key = (index[a], index[b])
        if key in records:
            raise ValueError("Duplicate heatmap cell")
        records[key] = row
        value = _number(row.get("cosine"))
        if value is not None:
            values[key] = value
    finite = values[np.isfinite(values)]
    limit = max(1.0, float(np.max(np.abs(finite)))) if len(finite) else 1.0
    fig, ax = plt.subplots(figsize=(6.6, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit)
    for i in range(len(labels)):
        for j in range(len(labels)):
            row = records.get((i, j), {})
            value = _number(row.get("cosine"))
            low, high = _number(row.get("cosine_low")), _number(row.get("cosine_high"))
            interval = "CI N/A" if low is None or high is None else "[%s, %s]" % (_format(low), _format(high))
            text = "N/A" if value is None else _format(value) + "\n" + interval
            color = "white" if value is not None and abs(value) > 0.65 * limit else "#161616"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Estimated cosine (unclipped)", shrink=0.85)
    return fig


def render(output_dir, main_rows, convergence_rows, same_cos_rows, cross_cos_rows):
    """Write the approved plots; rows contain precomputed 90% jackknife CIs.

    V4 u_norm and trace_ep must already be in paper units: code gradient ×2500,
    squared units ×2500², and per-episode noise trace =80×batch noise trace.
    Returns [{id, png, pdf, caption}], using absolute artifact paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir = Path(output_dir).resolve()
    names = ["v4_signal_noise_paths", "v1_dimensionless_paths", "v4_rho_convergence"]
    names += ["variant_cosines_" + point for point in POINTS]
    names += ["policy_cosines_" + variant for variant in ("V1", "V4")]
    for name in names:
        for suffix in (".png", ".pdf", ".caption.md"):
            if (output_dir / (name + suffix)).exists():
                raise FileExistsError("Candidate already exists: " + str(output_dir / (name + suffix)))
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = list(main_rows)
    convergence_rows = list(convergence_rows)
    same_cos_rows = list(same_cos_rows)
    cross_cos_rows = list(cross_cos_rows)
    counts = _counts(main_rows)
    artifacts = []

    def save(fig, name, caption):
        png, pdf, text = [output_dir / (name + suffix) for suffix in (".png", ".pdf", ".caption.md")]
        try:
            fig.savefig(png, dpi=180, metadata={"Description": caption})
            fig.savefig(pdf, metadata={"Subject": caption, "Title": name.replace("_", " ")})
            with text.open("x") as stream:
                stream.write(caption + "\n")
        finally:
            plt.close(fig)
        artifacts.append({"id": name, "png": str(png), "pdf": str(pdf), "caption": str(text)})

    v4 = [row for row in main_rows if row["variant"] == "V4"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)
    for ax, field, label, bounds in zip(axes, ("u_norm", "trace_ep", "B_ep"),
            (r"$\|\nabla J\|_2$", r"$\mathrm{tr}(\Sigma_{ep})$", r"$B_{ep}$ (episodes)"),
            (False, False, True)):
        _path_panel(ax, v4, field, label, log_y=True, bounds=bounds)
    axes[0].legend(frameon=False)
    fig.suptitle("V4 signal and noise along the saved DPPO and NC4 paths")
    save(fig, names[0],
         "V4 mean-gradient norm, equivalent per-episode noise trace, and noise-to-signal scale are shown "
         "against Euclidean actor-parameter distance from the pretrained policy; all vertical axes are logarithmic. "
         "Blue joins P0–D1–D2 and orange joins P0–N1–N2, with theory-unit gradients (stored vectors ×2500), "
         "trace scaled by 80 episodes per batch, and 90% delete-one-50-batch-group jackknife intervals. "
         "Available measurements: " + counts + ". Hollow triangles marked ≥ are lower bounds for unresolved signal, "
         "using the upper-signal plug-in bound with no joint confidence guarantee; intervals crossing zero and nonpositive or unavailable values are labeled "
         "instead of being clipped to a positive floor. The paths describe saved checkpoints, and the episode scale "
         "is batch-equivalent for the coupled leave-one-out estimator.")

    v1 = [row for row in main_rows if row["variant"] == "V1"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    _path_panel(axes[0], v1, "B_ep", r"$B_{ep}$ (episodes)", log_y=True, bounds=True)
    _path_panel(axes[1], v1, "expected_batch_cos", "Predicted 80-episode batch cosine")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="#888888", linestyle=":", linewidth=0.8)
    axes[1].axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    fig.suptitle("V1 dimensionless signal-resolution comparisons")
    save(fig, names[1],
         "The V1 NC4 measurement estimator is compared along the DPPO and NC4 checkpoint paths using "
         "only dimensionless noise-to-signal and alignment quantities; its advantage-normalized gradient has no "
         "reported return-unit norm. Horizontal axes are actor distance from P0, B_ep uses a logarithmic axis, "
         "and the predicted 80-episode training-batch cosine uses a linear axis with no clipping. "
         "Intervals are 90% delete-one-50-batch-group jackknife intervals; hollow triangles marked ≥ denote "
         "the upper-signal plug-in bound with no joint confidence guarantee, and unavailable values or zero-crossing log intervals are explicitly labeled. "
         "Available measurements: " + counts + ". V1 uses full-batch advantage normalization and therefore differs "
         "from NC4 training's normalization within optimizer chunks.")

    fig, ax = plt.subplots(figsize=(8.0, 4.7), constrained_layout=True)
    colors = ("#5c5c5c", "#70a2c8", "#20588e", "#e4a071", "#b84d20")
    missing = []
    for point, color in zip(POINTS, colors):
        rows = sorted((row for row in convergence_rows if row["point"] == point), key=lambda row: row["K"])
        xs, ys = [], []
        for row in rows:
            x, y = _number(row["K"]), _number(row.get("rho"))
            xs.append(x)
            ys.append(np.nan if y is None else y)
            if y is not None:
                _interval(ax, x, y, row.get("rho_low"), row.get("rho_high"), color)
        if not rows or not np.isfinite(ys).any():
            missing.append(point)
        ax.plot(xs, ys, color=color, marker="o", markersize=3.5, linewidth=1.2, label=point)
    ax.axhline(0.68, linestyle="--", color="#202020", linewidth=1.0, label="Approved stopping level 0.68")
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Collected full batches K (80 episodes per batch)")
    ax.set_ylabel(r"V4 balanced-split mean cosine $\rho$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    if missing:
        ax.text(0.02, 0.02, "No resolved curve: " + ", ".join(missing), transform=ax.transAxes, fontsize=8)
    save(fig, names[2],
         "V4 direction resolution is measured by the mean cosine of 20 dedicated balanced splits of "
         "independent 50-batch groups at each policy point; the horizontal axis counts full 80-episode batches. "
         "Bars show 90% delete-one-group jackknife intervals with the prescribed split randomization reused "
         "across deletions, and the dashed line marks the approved rho≥0.68 round-boundary stopping criterion. "
         "Signed estimates and intervals remain visible, while unavailable estimates appear as gaps; these "
         "nested prefixes share observations and their intervals are not independent. Final available counts: " + counts + ".")

    for point in POINTS:
        rows = [row for row in same_cos_rows if row["point"] == point]
        fig = _heatmap(plt, rows, VARIANTS, "from_variant", "to_variant", point + ": cross-estimator mean directions")
        save(fig, "variant_cosines_" + point,
             "At " + point + ", cells compare the estimated mean-gradient directions of V1–V5 on paired "
             "rollout batches, using same-batch cross-product correction. Each cell gives the unclipped cosine "
             "and its 90% delete-one-50-batch-group jackknife interval; N/A indicates an undefined signal or interval, "
             "and the color scale expands to retain estimates outside [−1, 1]. These are comparisons of measurement "
             "estimators, with V1–V3 using full-batch advantage normalization; uncertainty can leave the direction "
             "unresolved. Available measurements: " + counts + ".")

    for variant in ("V1", "V4"):
        rows = [row for row in cross_cos_rows if row["variant"] == variant]
        fig = _heatmap(plt, rows, POINTS, "from_point", "to_point", variant + ": cross-policy mean directions")
        save(fig, "policy_cosines_" + variant,
             variant + " mean-gradient directions are compared between the pretrained, DPPO, and NC4 checkpoints "
             "using independent rollout groups at different policy points. Each cell gives an unclipped cosine "
             "and 90% jackknife interval with deletions performed independently across the two policy samples; "
             "N/A denotes an undefined signal or interval, and values outside [−1, 1] remain visible. The comparison "
             "describes local measurement directions at saved policies and does not establish a causal training "
             "mechanism. Available measurements: " + counts + ".")
    return artifacts
