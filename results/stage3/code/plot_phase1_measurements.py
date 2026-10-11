#!/usr/bin/env python
"""Plot saved Phase-1 point estimates only; no bootstrap or new measurement."""

import datetime as dt
import json
import math
import os
from pathlib import Path
import shlex
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/phase1_analysis"
CANDIDATES = OUT / "candidates"
VARIANTS = ("nc4", "mc_critic", "mc_loo", "theory_loo", "theory_nobase")
VARIANT_LABELS = ("V1\nNC4 / GAE", "V2\nMC / critic", "V3\nMC / LOO", "V4\nTheory LOO", "V5\nNo baseline")
POINTS = (("theta0", "ns_theta0_seed1000", "Pretrained policy θ₀", "#377eb8", "o"),
          ("nc4_checkpoint", "ns_ckpt_seed2000", "NC4 1e-4 final checkpoint", "#ff7f00", "s"))


def load_inputs():
    review_path = ROOT / "runs/noise_ci_review.json"
    review = json.loads(review_path.read_text())
    if review.get("status") != "uncertainty_unreliable":
        raise ValueError("Expected the recorded uncertainty limitation before plotting.")
    inputs, validation = {}, {}
    for point, run, label, color, marker in POINTS:
        path = ROOT / run / "noise_scale.json"
        data = json.loads(path.read_text())
        manifest_path = ROOT / "runs" / run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or data.get("status") != "completed":
            raise ValueError("measurement is not complete: " + run)
        if tuple(data["variant_order"]) != VARIANTS or len(data["variants"]) != 5:
            raise ValueError("unexpected variant order: " + run)
        if data["n_batches"] != 25 or data["measurement_batches"] != 25 or len(data["measurements"]) != 25 or data["n_env_rollouts"] != 1000:
            raise ValueError("unexpected measurement size: " + run)
        for name, variant in zip(VARIANTS, data["variants"]):
            if variant["variant"] != name:
                raise ValueError("variant order mismatch: " + run)
            signal, variance = variant["u_sq"], variant["tr_sigma"]
            if not math.isfinite(signal) or not math.isfinite(variance) or variance <= 0:
                raise ValueError("nonfinite point moment: " + run)
            if signal <= 0:
                if any(variant[key] is not None for key in ("B_env", "B_ep", "expected_batch_cosine")):
                    raise ValueError("nonpositive signal has a derived ratio: " + run)
            else:
                expected_benv = variance / signal
                for key, expected in (("B_env", expected_benv), ("B_ep", 2 * expected_benv),
                                      ("expected_batch_cosine", (1 + expected_benv / 40) ** -0.5)):
                    if not math.isclose(variant[key], expected, rel_tol=1e-12, abs_tol=1e-15):
                        raise ValueError("saved derived point inconsistent: %s/%s/%s" % (run, name, key))
        matrix = np.asarray([[np.nan if value is None else value for value in row] for row in data["cosine_matrix"]], dtype=np.float64)
        if matrix.shape != (5, 5) or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-15, equal_nan=True):
            raise ValueError("invalid saved matrix shape/symmetry: " + run)
        if np.isinf(matrix).any():
            raise ValueError("infinite saved matrix: " + run)
        invalid_pairs = [{"row": VARIANTS[i], "column": VARIANTS[j], "saved_value": float(matrix[i, j])}
                         for i in range(5) for j in range(i + 1, 5) if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 1]
        inputs[point] = {"data": data, "matrix": matrix, "path": path, "manifest_path": manifest_path,
                         "manifest": manifest, "label": label, "color": color, "marker": marker}
        validation[point] = {"complete": True, "batches": 25, "per_env_rollouts": 1000,
                             "source_commit": manifest.get("source_commit"),
                             "derived_point_formulas_checked": True, "saved_matrix_symmetric": True,
                             "undefined_variants": [v["variant"] for v in data["variants"] if v["u_sq"] <= 0],
                             "matrix_null_cells": int(np.isnan(matrix).sum()),
                             "out_of_range_unique_pairs": invalid_pairs,
                             "matrix_out_of_range_cells": int((np.abs(matrix) > 1).sum())}
    return inputs, validation, review_path


def save_figure(fig, name):
    paths = []
    for extension in ("png", "pdf"):
        path = CANDIDATES / (name + "." + extension)
        fig.savefig(str(path), dpi=180, facecolor="white")
        paths.append(str(path))
    return paths


def add_variant_key(fig, y1, y2):
    fig.text(0.07, y1, "V1: scaled GAE (critic); V2: scaled MC − critic; V3: raw MC − LOO. V1–V3 normalize coefficients.", fontsize=9.5, color="#444444")
    fig.text(0.07, y2, "V4: raw MC − LOO with decision discount; V5: raw MC with decision discount. V4–V5 do not normalize.", fontsize=9.5, color="#444444")


def noise_plot(plt, inputs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.26, top=0.75, wspace=0.24)
    fig.suptitle("Phase 1: provisional gradient-noise point estimates", x=0.07, y=0.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(0.07, 0.922, "NO VALID CONFIDENCE INTERVALS — existing percentile intervals and positivity claims are unreliable",
             fontsize=10.7, color="#a12b1f", fontweight="bold")
    fields = (("B_ep", "Critical batch size in episodes", r"$B_{\rm ep}$ (episodes; log scale)"),
              ("expected_batch_cosine", "Expected training-batch gradient cosine", "Expected cosine at 80 episodes"))
    x = np.arange(5)
    for axis_index, (key, title, ylabel) in enumerate(fields):
        ax = axes[axis_index]
        for point_index, (point, _, label, color, marker) in enumerate(POINTS):
            data = inputs[point]["data"]
            offset = -0.12 if point_index == 0 else 0.12
            for j, variant in enumerate(data["variants"]):
                value = variant[key]
                if variant["u_sq"] <= 0:
                    ax.text(x[j] + offset, 0.055, "N/A", transform=ax.get_xaxis_transform(),
                            color=color, ha="center", va="center", fontsize=10, fontweight="bold")
                    continue
                ax.plot(x[j] + offset, value, marker=marker, markersize=7.5, linewidth=0, color=color,
                        label=label if j == 0 else None, zorder=3)
                number = ("%.1fk" % (value / 1000)) if key == "B_ep" else ("%.3f" % value)
                ax.annotate(number, (x[j] + offset, value), xytext=(0, 10 if point_index == 0 else -17),
                            textcoords="offset points", ha="center", fontsize=9.2, color=color)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_LABELS, fontsize=9.5)
        ax.set_xlim(-0.55, 4.55)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "B_ep":
            ax.set_yscale("log")
            ax.set_ylim(35, 6e5)
            ax.axhline(80, color="#555555", linestyle="--", linewidth=1.2)
            ax.text(4.52, 88, "Training batch: 80 episodes", ha="right", va="bottom", fontsize=9.2, color="#555555")
        else:
            ax.set_ylim(0, 0.087)
        ax.tick_params(labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.065, 0.87),
               ncol=2, frameon=False, fontsize=11, columnspacing=2)
    fig.text(0.07, 0.157, "N/A: nonpositive corrected squared-signal estimate; no zero or lower-bound substitution. No error bars are justified.",
             fontsize=10, color="#a12b1f")
    add_variant_key(fig, 0.108, 0.073)
    fig.text(0.07, 0.026, "Per policy: 25 fresh batches × 40 environments = 1,000 per-environment gradients; two episodes per environment rollout.",
             fontsize=9.5, color="#444444")
    paths = save_figure(fig, "noise_scale")
    plt.close(fig)
    return paths


def cosine_plot(plt, inputs):
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4))
    fig.subplots_adjust(left=0.07, right=0.865, bottom=0.235, top=0.755, wspace=0.19)
    fig.suptitle("Phase 1: provisional direction-cosine estimates", x=0.07, y=0.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(0.07, 0.922, "OUT-OF-RANGE VALUES ARE INVALID COSINES — saved points are shown without clipping; no valid CIs",
             fontsize=10.5, color="#a12b1f", fontweight="bold")
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e5e5e5")
    for ax, (point, _, label, color, marker) in zip(axes, POINTS):
        matrix = inputs[point]["matrix"]
        plot = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1.6, vmax=1.6, interpolation="nearest")
        ax.set_title(label, fontsize=12, pad=11)
        ax.set_xticks(np.arange(5)); ax.set_yticks(np.arange(5))
        ax.set_xticklabels(["V%d" % i for i in range(1, 6)], fontsize=11)
        ax.set_yticklabels(["V%d" % i for i in range(1, 6)], fontsize=11)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(5):
            for j in range(5):
                value = matrix[i, j]
                if not np.isfinite(value):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=10.5, color="#555555")
                elif abs(value) > 1:
                    ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                           edgecolor="#8d1f16", linewidth=2))
                    ax.text(j, i, "%.2f\nINVALID" % value, ha="center", va="center", fontsize=9.7,
                            color="white" if value > 1.25 else "#3d0c06", fontweight="bold")
                else:
                    ax.text(j, i, "%.2f" % value, ha="center", va="center", fontsize=11, color="#222222")
    cax = fig.add_axes([0.895, 0.27, 0.018, 0.44])
    colorbar = fig.colorbar(plot, cax=cax, ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    colorbar.set_label("Saved corrected estimate (not clipped)", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=10)
    fig.text(0.07, 0.17, "Red-bordered INVALID cells exceed [−1, 1]; they cannot be interpreted as directional alignment.",
             fontsize=10.2, color="#a12b1f")
    fig.text(0.07, 0.135, "Gray N/A cells involve a nonpositive corrected squared-signal estimate. These are not measured zero cosines.",
             fontsize=10, color="#444444")
    add_variant_key(fig, 0.09, 0.056)
    fig.text(0.07, 0.021, "Between-variant gradient norm scales differ; corrected ratios, including in-range entries, remain provisional.",
             fontsize=9.5, color="#444444")
    paths = save_figure(fig, "direction_cosines")
    plt.close(fig)
    return paths


def main():
    owned = [CANDIDATES / (name + "." + ext) for name in ("noise_scale", "direction_cosines") for ext in ("png", "pdf")]
    owned += [CANDIDATES / "figure_captions.md", OUT / "plot_record.json"]
    if any(path.exists() for path in owned):
        raise FileExistsError("Refusing to overwrite existing Phase-1 plot evidence.")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    inputs, validation, review_path = load_inputs()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    outputs = noise_plot(plt, inputs) + cosine_plot(plt, inputs)
    captions = """# Phase-1 figure captions

## Provisional noise scale — noise_scale.png / .pdf

For HalfCheetah-v2, at the fixed pretrained policy and NC4 1e-4 seed-0 final checkpoint, points show the saved critical batch size B_ep = 2 trΣ / û² (episodes, logarithmic left axis) and expected cosine (1 + B_ep/80)⁻¹ᐟ² between an 80-episode gradient and its mean direction (right axis); the horizontal line is the training batch of 80 episodes. Each policy used 25 fresh batches of 40 environments, with two episodes per environment rollout, after critic warm-up (20 and five batches respectively) and then frozen actor, critic and reward scaler; these are point estimates from 1,000 per-environment gradients, not three independent training-seed means. V1 uses scaled GAE with a critic, V2 uses scaled MC minus critic, V3 uses raw MC minus LOO, V4 adds decision discounting to raw MC minus LOO, and V5 drops that baseline; V1–V3 use normalized coefficients, coordinate means and 0.99 denoising weights, while V4–V5 use unnormalized coefficients, coordinate sums and unit denoising weights, so raw gradient norm scales differ between variants. Defined points suggest far larger critical batches than 80 (for example pretrained V5: B_ep≈114,292 and cosine≈0.0264), but negative corrected squared-signal points make V2 at θ₀ and V4 at the checkpoint N/A, without zero or lower-bound substitution. **All points are provisional: the recorded percentile intervals, CI-based positivity flags and lower bounds are unreliable, so no valid confidence intervals or error bars are shown, and apparent differences between variants or policies are not established.**

## Provisional mean-direction ratios — direction_cosines.png / .pdf

The two matrices show saved corrected cosine point estimates between V1–V5 mean-gradient directions, normalized using corrected squared-signal moments, at the same policies and from the same 25-batch measurements as the noise-scale figure. The shared color scale extends from −1.6 to 1.6 to preserve every saved finite value, with rounded numerical annotations, while gray N/A cells involve a nonpositive corrected squared-signal estimate. Red-bordered cells marked INVALID exceed the mathematical cosine range [−1,1]—three unique pairs at θ₀ and four at the checkpoint—so they cannot be read as directional alignment, and they are not clipped to ±1. In-range entries, including their signs, also remain provisional because the saved percentile confidence intervals and positivity claims are unreliable; uncertainty is deliberately not displayed. The widespread invalid and undefined cells limit direction comparisons, in addition to the approximation from shared normalization and LOO coupling, and the variant definitions and differing gradient units are the same as above.

Reproduction and source paths: [plot_record.json](../plot_record.json); [uncertainty audit](../../noise_ci_review.json); [plot source](../../../code/plot_phase1_measurements.py).
"""
    (CANDIDATES / "figure_captions.md").write_text(captions)
    excluded = ["all *_ci90 fields", "signal_ci_includes_zero", "signal_resolved_positive", "B_env_lower_bound"]
    record = {
        "status": "rendered_pending_visual_review", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()), "working_directory": str(Path.cwd()),
        "command": " ".join(shlex.quote(v) for v in [sys.executable, "-B", str(Path(__file__).resolve())]),
        "source_paths": {point: {"measurement": str(value["path"]), "manifest": str(value["manifest_path"])} for point, value in inputs.items()},
        "validation": validation, "uncertainty_review": str(review_path), "uncertainty": "none valid; point estimates provisional",
        "saved_fields_plotted": ["variants.u_sq (availability only)", "variants.B_ep", "variants.expected_batch_cosine", "cosine_matrix"],
        "excluded_unreliable_fields": excluded, "new_bootstrap_or_uncertainty_computations": 0, "new_scientific_samples": 0,
        "negative_signal_policy": "N/A, never zero/epsilon; saved negative signal is preserved in source",
        "out_of_range_cosine_policy": "preserve and label INVALID, no clipping; full signed color scale [-1.6,1.6]",
        "training_batch_episodes": 80, "variant_order": list(VARIANTS),
        "versions": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "outputs": outputs + [str(CANDIDATES / "figure_captions.md")], "figure_sizes_inches": {"noise_scale": [13, 6.8], "direction_cosines": [13, 7.4]}, "png_dpi": 180}
    (OUT / "plot_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": record["outputs"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
