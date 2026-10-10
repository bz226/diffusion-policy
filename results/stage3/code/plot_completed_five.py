#!/usr/bin/env python
"""Render the user-requested completed E0/E1/E2/SGD1/SGD3 comparison only."""

import datetime as dt
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np

from analyze_stage3 import COLORS, ROOT, finite, numeric_bad, read_json, read_rows, reference_runs


CONDITIONS = ("E0", "E1", "E2", "SGD1", "SGD3")
PANELS = (("AdamW: estimator changes", ("E0", "E1", "E2")),
          ("E2 estimator: AdamW and plain SGD", ("E2", "SGD1", "SGD3")))
SOURCE_COMMIT = "1515be0411a78df94e940de22df45834edadd858"
OUT = ROOT / "runs/completed_five_analysis"
CANDIDATES = OUT / "candidates"
REF_STYLES = {"Stage 1 DPPO": ("#444444", "--"),
              "Stage 2 NC4 1e-4": ("#8c8c8c", ":")}
METRICS = ("eval_episode_reward", "train_episode_reward", "J_disc_train",
           "kl_true_per_action", "theta_dist", "grad_cos_prev")


def load_selected():
    runs, source_records = {}, []
    for condition in CONDITIONS:
        runs[condition] = []
        for seed in range(3):
            run_id = "%s_seed%d" % (condition, seed)
            manifest_path = ROOT / "runs" / run_id / "manifest.json"
            manifest = read_json(manifest_path)
            if not manifest or manifest.get("status") != "complete":
                raise ValueError("not complete: " + run_id)
            if manifest.get("source_commit") != SOURCE_COMMIT:
                raise ValueError("unexpected scientific source: " + run_id)
            path = Path(manifest["logdir"]) / "result.pkl"
            rows = read_rows(path)
            if len(rows) != 140 or numeric_bad(rows):
                raise ValueError("incomplete or nonfinite result: " + run_id)
            for row in rows:
                itr = row["itr"]
                if row.get("step") != (itr - itr // 10) * 80000:
                    raise ValueError("unexpected step grid: " + run_id)
                expected = ("eval_episode_reward",) if itr % 10 == 0 else METRICS[1:]
                for metric in expected:
                    value = row.get(metric)
                    # The first gradient has no predecessor. Do not replace it by zero.
                    if metric == "grad_cos_prev" and itr == 1 and value is None:
                        continue
                    if not finite(value):
                        raise ValueError("missing metric %s at %s/%d" % (metric, run_id, itr))
            runs[condition].append({"seed": seed, "rows": rows, "path": str(path)})
            source_records.append({"run": run_id, "condition": condition, "seed": seed,
                                   "result_path": str(path), "manifest_path": str(manifest_path),
                                   "source_commit": manifest["source_commit"], "status": "complete",
                                   "rows": len(rows), "eval_rows": 14, "train_rows": 126})
    return runs, source_records


def series(runs, metric):
    """An arithmetic seed mean at common recorded steps; no smoothing/imputation."""
    maps = [{r["step"]: float(r[metric]) for r in run["rows"] if finite(r.get(metric))}
            for run in runs]
    steps = sorted(set.intersection(*(set(mapping) for mapping in maps)))
    if not steps:
        return None
    values = np.asarray([[mapping[step] for step in steps] for mapping in maps], dtype=np.float64)
    return {"steps": np.asarray(steps, dtype=np.int64), "values": values,
            "mean": values.mean(axis=0), "sd": values.std(axis=0, ddof=1)}


def plot_one(ax, aggregate, color, label, log=False, reference=False, linestyle="-"):
    if aggregate is None:
        return
    x = aggregate["steps"] / 1e6
    mean, sd = aggregate["mean"], aggregate["sd"]
    positive = mean > 0 if log else np.ones(len(mean), dtype=bool)
    shown_mean = np.where(positive, mean, np.nan)
    if not reference:
        for values in aggregate["values"]:
            shown = np.where(values > 0, values, np.nan) if log else values
            ax.plot(x, shown, color=color, alpha=0.22, linewidth=0.7, zorder=1)
        lo, hi = mean - sd, mean + sd
        valid_band = (lo > 0) & positive if log else positive
        ax.fill_between(x, lo, hi, where=valid_band, color=color, alpha=0.14,
                        linewidth=0, interpolate=False, zorder=2)
    ax.plot(x, shown_mean, color=color, label=label, linestyle=linestyle,
            linewidth=1.8 if not reference else 1.55, zorder=3 if not reference else 4)


def render(plt, aggregates, refs, kl_target, diagnostic):
    if diagnostic:
        name = "diagnostics"
        title = "Stage 3: update diagnostics for five completed conditions"
        metrics = ("kl_true_per_action", "theta_dist", "grad_cos_prev")
        labels = ("Post-update KL statistic (log scale)", "Distance from pretrained actor", "Consecutive gradient cosine")
        ylabels = ("KL per action-chunk diagnostic", r"$\|\theta-\theta_0\|_2$", "Cosine")
    else:
        name = "performance"
        title = "Stage 3: performance of five completed conditions"
        metrics = ("eval_episode_reward", "train_episode_reward", "J_disc_train")
        labels = ("Evaluation sampler", "Training sampler", "Discounted training objective")
        ylabels = ("Undiscounted episode return", "Undiscounted episode return", r"$J_{\rm disc}$ (raw reward)")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.3), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.81,
                        hspace=0.65, wspace=0.30)
    fig.suptitle(title, x=0.075, y=0.982, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.075, 0.945, "Seeds 0–2 · mean ± sample SD · faint individual seeds · no smoothing",
             fontsize=11, color="#444444")
    for row_index, (row_title, conditions) in enumerate(PANELS):
        row_y = 0.898 if row_index == 0 else 0.470
        fig.text(0.075, row_y, row_title, fontsize=12, fontweight="bold")
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            log = metric == "kl_true_per_action"
            for condition in conditions:
                plot_one(ax, aggregates[condition][metric], COLORS[condition], condition, log=log)
            for ref_name, by_metric in refs.items():
                color, style = REF_STYLES[ref_name]
                plot_one(ax, by_metric[metric], color, ref_name, log=log,
                         reference=True, linestyle=style)
            if log:
                ax.set_yscale("log")
                ax.axhline(kl_target, color="#333333", linewidth=1.2,
                           linestyle=(0, (5, 2, 1, 2)), label=r"Calibration $\mathrm{KL}^*$", zorder=4)
            if metric == "grad_cos_prev":
                ax.axhline(0, color="#999999", linewidth=0.7, zorder=0)
            ax.set_title(labels[col_index], fontsize=11.5, pad=9)
            ax.set_ylabel(ylabels[col_index], fontsize=10.5)
            ax.set_xlim(0, 10.15)
            ax.set_xticks(np.arange(0, 11, 2))
            ax.set_xlabel("Training environment steps (million)", fontsize=10)
            ax.tick_params(labelsize=10, labelbottom=True)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, which="major")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, legend_labels = axes[row_index, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.068, row_y - 0.008),
                   ncol=len(legend_labels), frameon=False, fontsize=10, handlelength=2.3,
                   columnspacing=1.45, borderpad=0)
    if diagnostic:
        footnote = "First-update gradient cosine is undefined and omitted. Gray reference: NC4 (KL only)."
    else:
        footnote = "Evaluation ends at 9.36M steps; training ends at 10.08M. Earlier stages did not log the discounted objective."
    fig.text(0.075, 0.025, footnote, fontsize=10, color="#444444")
    paths = []
    for extension in ("png", "pdf"):
        path = CANDIDATES / (name + "." + extension)
        fig.savefig(str(path), dpi=180, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    owned = [CANDIDATES / ("%s.%s" % (kind, ext))
             for kind in ("performance", "diagnostics") for ext in ("png", "pdf")]
    owned += [CANDIDATES / "captions.md", OUT / "plot_record.json"]
    if any(path.exists() for path in owned):
        raise FileExistsError("This script preserves earlier artifacts; archive or select a new output path before rerunning.")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    runs, source_records = load_selected()
    references = reference_runs(ROOT)
    for name, selected in references.items():
        if len(selected) != 3 or any(len(r["rows"]) != 140 for r in selected):
            raise ValueError("incomplete reference: " + name)
        for run in selected:
            if numeric_bad(run["rows"]):
                raise ValueError("nonfinite reference: " + name)
            source_records.append({"reference": name, "seed": run["seed"], "rows": len(run["rows"]),
                                   "result_path": run["path"]})
    calibration_path = ROOT / "cal_seed3000/calibration.json"
    calibration = read_json(calibration_path)
    kl_target = float(calibration["kl_target"])
    aggregates = {name: {metric: series(selected, metric) for metric in METRICS}
                  for name, selected in runs.items()}
    refs = {name: {metric: series(selected, metric) for metric in METRICS}
            for name, selected in references.items()}
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    files = render(plt, aggregates, refs, kl_target, diagnostic=False)
    files += render(plt, aggregates, refs, kl_target, diagnostic=True)
    captions = """# Captions for the five completed Stage-3 conditions

## Performance — performance.png / .pdf

The top row compares the AdamW estimator ladder (E0: GAE control; E1: Monte Carlo returns with a leave-one-out baseline; E2: E1 plus decision discounting and the prescribed estimator rescalings), and the bottom row compares E2 with plain SGD at calibrated step sizes η* and 3η* (SGD1 and SGD3). Columns show undiscounted 1000-step episode returns under the evaluation and training samplers, then the training objective J_disc = mean over 80 episodes of Σₖ₌₀²⁴⁹ 0.99ᵏ Rₖ, with raw reward Rₖ summed over each four-step decision; the horizontal axis is cumulative training environment steps in millions. Colored lines are arithmetic means over independent training seeds 0–2, shaded bands are ± one sample standard deviation (ddof=1), and faint lines are the individual seeds, with no smoothing; gray dashed/dotted lines give three-seed means from Stage 1 DPPO and Stage 2 NC4 at 1e-4 only where those metrics were recorded. E1 has the highest endpoint mean return and late-training discounted objective among these five conditions, while SGD1 and SGD3 make less progress than E2 over this sample budget; the comparisons do not isolate discounting from E2's simultaneous rescalings. Evaluation ends at iteration 130 (9.36M steps) and training at iteration 139 (10.08M); these plots do not include the separate final-checkpoint evaluation, and three seeds give limited precision.

## Update diagnostics — diagnostics.png / .pdf

Rows compare the same five completed conditions, with columns showing the sampled post-update KL statistic on a logarithmic vertical axis, Euclidean actor-parameter distance from the pretrained policy, and cosine similarity of consecutive actor gradients used by training. The KL statistic is 10 × mean[expm1(d) − d], where d is the unclamped coordinate-summed change in denoising-transition log-probability on a 20,000-pair batch subsample; it is not the exact marginal KL of the executed action, and the horizontal calibration target is KL* = 0.0005845173361. Colored means and ± one sample-SD bands (ddof=1) use seeds 0–2 at each recorded step, faint lines show each seed, and the dotted gray curve is the earlier three-seed NC4 mean for KL only, without reference uncertainty bands or smoothing. SGD moves much less in parameter space than AdamW here, SGD3 often exceeds the calibration KL target, and consecutive gradient cosines fluctuate around zero; these diagnostics alone do not establish the cause of return differences. The first training update has no preceding gradient, so its cosine is omitted; no recorded KL mean, individual KL, or lower SD bound is nonpositive in this subset.

Sources and reproduction command: [plot_record.json](../plot_record.json); plot source: [plot_completed_five.py](../../../code/plot_completed_five.py).
"""
    (CANDIDATES / "captions.md").write_text(captions)
    exclusions = {}
    for name, by_metric in aggregates.items():
        kl = by_metric["kl_true_per_action"]
        exclusions[name] = {"undefined_first_gradient_cosines": 3,
                            "nonpositive_kl_individual": int(np.sum(kl["values"] <= 0)),
                            "nonpositive_kl_mean": int(np.sum(kl["mean"] <= 0)),
                            "nonpositive_kl_lower_sd": int(np.sum(kl["mean"] - kl["sd"] <= 0))}
    # Keep masks explicit: log plots never substitute an arbitrary epsilon.
    nonpositive = sum(entry["nonpositive_kl_lower_sd"] + entry["nonpositive_kl_individual"]
                      + entry["nonpositive_kl_mean"] for entry in exclusions.values())
    if nonpositive:
        captions = captions.replace("no recorded KL mean, individual KL, or lower SD bound is nonpositive in this subset.",
                                    "nonpositive KL values or lower SD bounds cannot be drawn on a log axis and are masked, with counts recorded in plot_record.json; no epsilon is substituted.")
        (CANDIDATES / "captions.md").write_text(captions)
    record = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "rendered_pending_visual_review",
              "scope": list(CONDITIONS), "seeds": [0, 1, 2], "scientific_source_commit": SOURCE_COMMIT,
              "script": str(Path(__file__).resolve()),
              "command": " ".join(shlex.quote(v) for v in [sys.executable, "-B", str(Path(__file__).resolve())]),
              "working_directory": str(Path.cwd()), "sources": source_records,
              "calibration_path": str(calibration_path), "kl_target": kl_target,
              "eta_star": calibration["eta_star"], "aggregation": "arithmetic seed mean; sample SD ddof=1; n=3; no smoothing",
              "uncertainty_unit": "independent training seed, not episodes or iterations",
              "log_axis_policy": "include zeros in arithmetic aggregation; mask nonpositive rendered values/band; no epsilon",
              "exclusions_unique_conditions": exclusions, "imputed_values": 0,
              "reference_metrics": {name: [key for key, values in metrics.items() if values is not None]
                                    for name, metrics in refs.items()},
              "versions": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
              "outputs": files + [str(CANDIDATES / "captions.md")],
              "figure_size_inches": [13.5, 8.3], "png_dpi": 180,
              "aggregation_rows": {name: {metric: len(value["steps"]) if value is not None else 0
                                            for metric, value in by_metric.items()}
                                   for name, by_metric in aggregates.items()}}
    (OUT / "plot_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": record["outputs"], "log_omissions": exclusions}, indent=2))


if __name__ == "__main__":
    main()
