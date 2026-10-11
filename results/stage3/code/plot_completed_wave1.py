#!/usr/bin/env python
"""Plot all nine completed Stage-3 wave-1 conditions without changing runs."""

import datetime as dt
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np

from analyze_stage3 import COLORS, ROOT, finite, numeric_bad, read_json, read_rows, reference_runs, window_noise
from plot_completed_five import REF_STYLES, SOURCE_COMMIT, plot_one, series


CONDITIONS = ("E0", "E1", "E2", "E2_beta0", "E2_nobase", "SGD03", "SGD1", "SGD3", "SGD10")
GROUPS = (("Estimator ladder under AdamW", ("E0", "E1", "E2")),
          ("E2 under AdamW: momentum and baseline", ("E2", "E2_beta0", "E2_nobase")),
          ("E2 estimator: AdamW and the SGD step-size sweep", ("E2", "SGD03", "SGD1", "SGD3", "SGD10")))
METRICS = ("eval_episode_reward", "train_episode_reward", "J_disc_train", "kl_true_per_action",
           "theta_dist", "grad_cos_prev", "x0_sat_frac_mean", "action_oor_frac")
OUT = ROOT / "runs/completed_wave1_analysis"
CANDIDATES = OUT / "candidates"


def load_inputs():
    runs, sources = {}, []
    for condition in CONDITIONS:
        runs[condition] = []
        for seed in range(3):
            run_id = "%s_seed%d" % (condition, seed)
            manifest_path = ROOT / "runs" / run_id / "manifest.json"
            manifest = read_json(manifest_path)
            if not manifest or manifest.get("status") != "complete" or manifest.get("source_commit") != SOURCE_COMMIT:
                raise ValueError("incomplete run or unexpected source: " + run_id)
            path = Path(manifest["logdir"]) / "result.pkl"
            rows = read_rows(path)
            if len(rows) != 140 or numeric_bad(rows):
                raise ValueError("incomplete/nonfinite saved values: " + run_id)
            for row in rows:
                itr = row["itr"]
                if row["step"] != (itr - itr // 10) * 80000:
                    raise ValueError("incorrect step grid: " + run_id)
                expected = ("eval_episode_reward",) if itr % 10 == 0 else METRICS[1:] + ("split_dot", "split_diff_sq")
                for key in expected:
                    if key == "grad_cos_prev" and itr == 1 and row.get(key) is None:
                        continue
                    if not finite(row.get(key)):
                        raise ValueError("undefined %s at %s/%d" % (key, run_id, itr))
            runs[condition].append({"seed": seed, "rows": rows, "path": str(path)})
            sources.append({"run": run_id, "condition": condition, "seed": seed,
                            "manifest_path": str(manifest_path), "result_path": str(path),
                            "source_commit": manifest["source_commit"], "rows": len(rows), "status": "complete"})
    references = reference_runs(ROOT)
    for name, selected in references.items():
        if len(selected) != 3 or any(len(r["rows"]) != 140 or numeric_bad(r["rows"]) for r in selected):
            raise ValueError("incomplete or nonfinite reference: " + name)
        sources += [{"reference": name, "seed": run["seed"], "result_path": run["path"], "rows": len(run["rows"])}
                    for run in selected]
    return runs, references, sources


def noise_series(runs):
    """Preserve all fourteen window positions and require all three seeds for a mean."""
    windows = [window_noise(run["rows"]) for run in runs]
    steps = np.asarray([row["step"] for row in windows[0]], dtype=np.int64)
    values = np.asarray([[row["B_env"] if row["B_env"] is not None else np.nan for row in ws]
                         for ws in windows], dtype=np.float64)
    valid = np.isfinite(values).all(axis=0)
    mean, sd = np.full(14, np.nan), np.full(14, np.nan)
    mean[valid] = values[:, valid].mean(axis=0)
    sd[valid] = values[:, valid].std(axis=0, ddof=1)
    return {"steps": steps, "values": values, "mean": mean, "sd": sd,
            "complete_seed_mean": valid, "windows": windows}


def metric_series(runs, metric):
    # action_oor_frac is also present at eval iterations, whose sampler differs.
    # Restrict the training diagnostic rather than mixing two samplers at equal steps.
    if metric == "action_oor_frac":
        runs = [dict(run, rows=[r for r in run["rows"] if "train_episode_reward" in r]) for run in runs]
    return series(runs, metric)


def plot_noise(ax, aggregate, color, label, missing_level):
    x = aggregate["steps"] / 1e6
    for values in aggregate["values"]:
        ax.plot(x, values, color=color, linewidth=0.7, alpha=0.25, marker=".", markersize=3.5)
    mean, sd = aggregate["mean"], aggregate["sd"]
    valid = np.isfinite(mean)
    lo, hi = mean - sd, mean + sd
    ax.fill_between(x, lo, hi, where=valid & (lo > 0), alpha=0.14, color=color, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.8, marker="o", markersize=3.5,
            label="%s (%d/14)" % (label, valid.sum()))
    # Most means are isolated. Whiskers make their SD visible without connecting gaps.
    ax.vlines(x[valid], np.where(lo[valid] > 0, lo[valid], mean[valid]), hi[valid],
              color=color, linewidth=1.2, alpha=0.7)
    # Crosses denote an undefined condition mean, not measured values on the log axis.
    ax.plot(x[~valid], np.full((~valid).sum(), missing_level), marker="x", markersize=4,
            linewidth=0, color=color, alpha=0.8, transform=ax.get_xaxis_transform())


def render(plt, aggregates, references, kl_target, name):
    definitions = {
        "performance": ("Performance", ("eval_episode_reward", "train_episode_reward", "J_disc_train"),
                        ("Evaluation sampler", "Training sampler", "Discounted training objective"),
                        ("Undiscounted episode return", "Undiscounted episode return", r"$J_{\rm disc}$ (raw reward)")),
        "diagnostics": ("Update diagnostics", ("kl_true_per_action", "theta_dist", "grad_cos_prev"),
                        ("Post-update KL statistic (log)", "Distance from pretrained actor", "Consecutive gradient cosine"),
                        ("KL per action-chunk diagnostic", r"$\|\theta-\theta_0\|_2$", "Cosine")),
        "noise_saturation": ("Gradient noise and saturation", ("B_env", "x0_sat_frac_mean", "action_oor_frac"),
                             ("Window gradient-noise estimate (log)", "Denoised prediction saturation", "Training actions outside [−1, 1]"),
                             (r"$B_{\rm env}$ (environment rollouts)", "Fraction of predicted coordinates", "Fraction of executed coordinates"))}
    heading, metrics, titles, ylabels = definitions[name]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.855,
                        hspace=0.72, wspace=0.31)
    fig.suptitle("Stage 3: " + heading.lower() + " across completed wave 1", x=0.075, y=0.985,
                 ha="left", fontsize=15.5, fontweight="bold")
    fig.text(0.075, 0.955, "Nine conditions · seeds 0–2 · mean ± sample SD · faint individual seeds · no smoothing",
             fontsize=10.5, color="#444444")
    for row_index, (row_title, conditions) in enumerate(GROUPS):
        row_y = axes[row_index, 0].get_position().y1 + 0.061
        fig.text(0.075, row_y, row_title, fontsize=11.5, fontweight="bold")
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            log = metric in ("kl_true_per_action", "B_env")
            for index, condition in enumerate(conditions):
                if metric == "B_env":
                    plot_noise(ax, aggregates[condition][metric], COLORS[condition], condition, 0.025 + index * 0.032)
                else:
                    plot_one(ax, aggregates[condition][metric], COLORS[condition], condition, log=log)
            if metric != "B_env":
                for ref_name, by_metric in references.items():
                    color, style = REF_STYLES[ref_name]
                    plot_one(ax, by_metric[metric], color, ref_name, log=log, reference=True, linestyle=style)
            if log:
                ax.set_yscale("log")
            if metric == "kl_true_per_action":
                ax.axhline(kl_target, color="#333333", linewidth=1.1, linestyle=(0, (5, 2, 1, 2)),
                           label=r"Calibration $\mathrm{KL}^*$", zorder=4)
            if metric == "B_env":
                available = np.concatenate([aggregates[c][metric]["values"].ravel() for c in conditions])
                lower_bounds = np.concatenate([aggregates[c][metric]["mean"] - aggregates[c][metric]["sd"] for c in conditions])
                available = np.concatenate([available, lower_bounds])
                available = available[np.isfinite(available) & (available > 0)]
                ax.set_ylim(available.min() / 8, available.max() * 1.5)
            if metric == "grad_cos_prev":
                ax.axhline(0, color="#999999", linewidth=0.7, zorder=0)
            ax.set_title(titles[col_index], fontsize=10.5, pad=8)
            ax.set_ylabel(ylabels[col_index], fontsize=10)
            ax.set_xlim(0, 10.15)
            ax.set_xticks(np.arange(0, 11, 2))
            ax.set_xlabel("Training environment steps (million)", fontsize=9.5)
            ax.tick_params(labelsize=9.5, labelbottom=True)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, which="major")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, labels = axes[row_index, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.068, row_y - 0.007),
                   ncol=len(labels), frameon=False, fontsize=9.5, handlelength=2.1,
                   columnspacing=1.05, borderpad=0)
    footnotes = {
        "performance": "Evaluation ends at 9.36M steps; training at 10.08M. Gray curves: earlier-stage means where recorded.",
        "diagnostics": "First-update gradient cosines are undefined. Gray reference: NC4 KL only; dash-dot: calibration target.",
        "noise_saturation": "B_env crosses: no three-seed mean; legend counts: defined means / 14 windows. Cross height has no data meaning."}
    fig.text(0.075, 0.036, footnotes[name], fontsize=9.5, color="#444444")
    if name == "noise_saturation":
        fig.text(0.075, 0.018, "Dots/lines retain valid individual estimates and gaps; no averaging across only the remaining positive-denominator seeds.",
                 fontsize=9.5, color="#444444")
    files = []
    for extension in ("png", "pdf"):
        path = CANDIDATES / (name + "." + extension)
        fig.savefig(str(path), dpi=180, facecolor="white")
        files.append(str(path))
    plt.close(fig)
    return files


def main():
    names = ("performance", "diagnostics", "noise_saturation")
    owned = [CANDIDATES / (name + "." + ext) for name in names for ext in ("png", "pdf")]
    owned += [CANDIDATES / "captions.md", OUT / "plot_record.json"]
    if any(path.exists() for path in owned):
        raise FileExistsError("Refusing to overwrite earlier plot artifacts.")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    runs, references, sources = load_inputs()
    aggregate = {name: {metric: metric_series(selected, metric) for metric in METRICS}
                 for name, selected in runs.items()}
    for name, selected in runs.items():
        aggregate[name]["B_env"] = noise_series(selected)
    refs = {name: {metric: metric_series(selected, metric) for metric in METRICS} for name, selected in references.items()}
    calibration_path = ROOT / "cal_seed3000/calibration.json"
    calibration = read_json(calibration_path)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    outputs = []
    for name in names:
        outputs.extend(render(plt, aggregate, refs, calibration["kl_target"], name))
    exclusions = {}
    for condition, metrics in aggregate.items():
        kl, noise = metrics["kl_true_per_action"], metrics["B_env"]
        exclusions[condition] = {
            "undefined_first_gradient_cosines": 3,
            "nonpositive_kl_individual": int((kl["values"] <= 0).sum()),
            "nonpositive_kl_mean": int((kl["mean"] <= 0).sum()),
            "nonpositive_kl_lower_sd": int((kl["mean"] - kl["sd"] <= 0).sum()),
            "noise_defined_windows_per_seed": np.isfinite(noise["values"]).sum(axis=1).tolist(),
            "noise_defined_three_seed_means": int(noise["complete_seed_mean"].sum()),
            "noise_undefined_three_seed_means": int((~noise["complete_seed_mean"]).sum()),
            "noise_nonpositive_lower_sd": int((noise["mean"] - noise["sd"] <= 0).sum()),
            "noise_window_steps": noise["steps"].tolist(),
            "noise_three_seed_mean_defined": noise["complete_seed_mean"].tolist()}
    captions = """# Completed wave-1 figure captions

## Performance — performance.png / .pdf

Rows compare the AdamW estimator ladder (E0/E1/E2), E2's momentum and baseline variants, and E2 against SGD at η*/3, η*, 3η* and 10η*; η* = 0.0001210723567, while all AdamW runs use lr=1e-4. Columns show 1000-step undiscounted episode returns under the eval/train samplers and J_disc, the mean raw discounted return Σₖ₌₀²⁴⁹ 0.99ᵏ Rₖ over 80 episodes, where each decision reward Rₖ sums four environment-step rewards. Colored lines/bands are arithmetic means ± one sample SD (ddof=1) over independent training seeds 0–2, faint lines are individual seeds, and gray Stage-1 DPPO/Stage-2 NC4 curves are three-seed means where recorded; there is no smoothing or reference uncertainty band. E1 has the highest final scheduled mean eval and late-training J_disc among these conditions, removing E2's LOO baseline reduces progress, and larger SGD steps improve training return here; endpoint eval differences remain subject to three-seed uncertainty. Eval ends at 9.36M training steps and training at 10.08M; BATCH4 and final-checkpoint evaluations are excluded because they were unfinished at this snapshot, and E2 changes discounting together with estimator rescalings.

## Update diagnostics — diagnostics.png / .pdf

The same rows show the post-update denoising-transition KL statistic (log axis), actor distance from the pretrained parameters, and cosine between consecutive actor gradients used by training, against cumulative training environment steps in millions. KL is 10 × mean[expm1(d) − d] on 20,000 stored decision/denoising-step pairs, with unclamped coordinate-summed log-probability change d; it is not the exact marginal executed-action KL, and the dash-dot target is KL* = 0.0005845173361. Means and ± one sample-SD bands use three independent seeds (ddof=1), faint curves retain individual seeds, and the dotted gray NC4 line shows only its recorded KL mean. SGD3/SGD10 often exceed the initial KL target while remaining much closer to the pretrained actor than AdamW, and gradient cosines fluctuate around zero, so these measurements alone do not explain return differences. The first gradient cosine is undefined; nonpositive lower SD bounds are masked on log axes without replacing values by an epsilon, with all omission counts in plot_record.json.

## Noise and saturation — noise_saturation.png / .pdf

For each nonoverlapping ten-iteration window (nine training updates), B_env = 10 × Σ‖g_A−g_B‖² / Σ⟨g_A,g_B⟩ estimates gradient noise in units of environment rollouts, with each plotted window placed at its final cumulative training step. Left panels use a log axis: faint dots/lines show each seed's positive-denominator estimates with gaps, colored means/SD whiskers require all three seeds, and bottom crosses mark undefined condition means whose vertical positions carry no data value; legend counts give defined three-seed means out of 14 windows, and nonpositive lower SD bounds are omitted. Middle panels show the fraction of pre-clamp denoised predictions outside ±1, averaged across ten fine-tuned denoising steps on a 20,000-pair subsample, and right panels show the fraction of actual executed training-sampler action coordinates outside ±1; both use training iterations only, with means ± one sample SD across seeds 0–2 (ddof=1), faint seeds and no smoothing. Noise means are unresolved in most windows (E2: 0/14), so these curves cannot reliably rank conditions by noise scale, whereas saturation and action-range fractions are measurable and differ across conditions. No nonpositive-denominator estimate is treated as zero or included in a survivor-only mean, no Phase-1 bootstrap confidence interval is used, and these observational diagnostics do not establish a saturation mechanism.

Sources, masks and reproduction: [plot_record.json](../plot_record.json); [plot source](../../../code/plot_completed_wave1.py).
"""
    (CANDIDATES / "captions.md").write_text(captions)
    record = {
        "status": "rendered_pending_visual_review", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "conditions": list(CONDITIONS), "seeds": [0, 1, 2], "scientific_source_commit": SOURCE_COMMIT,
        "source_script": str(Path(__file__).resolve()), "reused_helpers": [str(ROOT / "code/plot_completed_five.py"), str(ROOT / "code/analyze_stage3.py")],
        "command": " ".join(shlex.quote(v) for v in [sys.executable, "-B", str(Path(__file__).resolve())]),
        "working_directory": str(Path.cwd()), "source_inputs": sources,
        "verified": {"completed_runs": 27, "rows_per_run": 140, "expected_step_grid": True, "finite_raw_numeric_values": True},
        "excluded_conditions": {"BATCH4": "running at requested snapshot", "PROJ3": "prespecified skip", "PROJ10": "prespecified skip"},
        "checkpoint_evaluations": "not included; pending at snapshot",
        "aggregation": "arithmetic mean and sample SD ddof=1 over exactly three seeds; no smoothing or interpolation",
        "action_oor_sampler": "training rows only, 126 observations per seed; eval-sampler values are excluded deliberately",
        "noise_windows": "[0,9], [10,19], ..., [130,139]; nine training iterations each; undefined unless denominator positive in all three seeds",
        "log_policy": "No epsilon; preserve raw data, mask nonpositive plot bounds; per-condition counts below",
        "exclusions_unique_conditions": exclusions,
        "reference_metrics": {name: [key for key, val in vals.items() if val is not None] for name, vals in refs.items()},
        "kl_target": calibration["kl_target"], "eta_star": calibration["eta_star"], "calibration_path": str(calibration_path),
        "versions": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "figure_size_inches": [13.5, 12.5], "png_dpi": 180,
        "outputs": outputs + [str(CANDIDATES / "captions.md")]}
    (OUT / "plot_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": record["outputs"], "defined_noise_means": {c: v["noise_defined_three_seed_means"] for c, v in exclusions.items()}}, indent=2))


if __name__ == "__main__":
    main()
