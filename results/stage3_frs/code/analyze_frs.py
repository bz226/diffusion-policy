#!/usr/bin/env python
"""Read-only FRS analysis; outputs require terminal runs and a passed gate.

No training, new sampling, source mutation, or automatic curation is performed.
Existing Stage-3 summaries are reused from their native evidence, not copied CSVs.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("FRS1", "E2", "E2_beta0", "SGD10")
COLORS = {"FRS1": "#0072B2", "E2": "#009E73", "E2_beta0": "#CC79A7", "SGD10": "#D55E00"}
TERMINAL = ("complete", "failed", "skipped")
SCHEDULER_TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                      "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED")
PANELS = (("eval_episode_reward", "Eval return", False),
          ("train_episode_reward", "Train return", False),
          ("J_disc_train", "Discounted training objective J", False),
          ("kl_true_per_action", "Sampled transition KL", True),
          ("theta_dist", "Distance from pretrained actor", False),
          ("B_env", "Window noise scale B_env", True),
          ("grad_cos_prev", "Consecutive gradient cosine", False),
          ("x0_sat_frac_mean", "Saturation / action violation", False))


def require(value, message):
    if not value:
        raise ValueError(message)


def helper(reference):
    path = reference / "code/analyze_stage3.py"
    # Compile the read-only helper directly, avoiding a reference-tree pycache.
    module = types.ModuleType("frs_stage3_analysis_reference")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def manifest_set(a, root, specs):
    result = []
    for spec in specs:
        path = root / "runs" / spec["run_id"] / "manifest.json"
        value = a.read_json(path, {})
        require(value.get("status") in TERMINAL, "Nonterminal or missing manifest: " + str(path))
        result.append({"run": spec["run_id"], "manifest": str(path), "status": value["status"],
                       "command": value.get("command"), "overrides": spec.get("overrides"),
                       "environment": value.get("environment"), "hardware": value.get("hardware"),
                       "source_before": value.get("source_before"), "source_after": value.get("source_after"),
                       "source_commit": value.get("source_commit"), "parent_commit": value.get("parent_commit"),
                       "console_log": spec.get("console_log"), "failure_reason": value.get("failure_reason")})
    return result


def validate_runs(a, runs):
    metrics = ("train_episode_reward", "J_disc_train", "kl_true_per_action", "theta_dist",
               "update_norm", "x0_sat_frac_mean", "x0_sat_frac_last", "action_oor_frac",
               "split_dot", "split_diff_sq", "time", "actor_lr")
    for run in runs:
        rows, complete = run["rows"], run["status"] == "complete"
        require(len(rows) <= 140, "Run exceeds approved 140 iterations: " + run["spec"]["run_id"])
        for row in rows:
            evaluation = row["itr"] % 10 == 0
            require(("eval_episode_reward" in row) == evaluation and
                    ("train_episode_reward" in row) != evaluation,
                    "Evaluation/training schedule mismatch: " + run["result_path"])
        if not complete:
            continue  # Failed prefixes and nonfinite evidence remain visible.
        require(not a.numeric_bad(rows), "Completed run contains nonfinite evidence: " + run["result_path"])
        require(sum("train_episode_reward" in row for row in rows) == 126 and rows[-1]["step"] == 10080000,
                "Completed run has wrong training count/endpoint: " + run["result_path"])
        needed = metrics + (("max_ratio", "median_ratio") if run["spec"]["condition"] == "FRS1" else ())
        for row in rows:
            fields = ("eval_episode_reward",) if row["itr"] % 10 == 0 else needed
            require(all(a.finite(row.get(key)) for key in fields),
                    "Completed run lacks finite required metrics at itr %s: %s" % (row["itr"], run["result_path"]))
        evaluation = run["checkpoint_eval"]
        if run["checkpoint_eval_status"] == "complete":
            require(evaluation.get("status") == "completed", "Checkpoint evaluation payload is not completed")
            for sampler in ("train", "eval"):
                result = evaluation.get("samplers", {}).get(sampler, {})
                require(all(a.finite(result.get(key, {}).get("mean")) for key in ("J_disc", "undiscounted_return")),
                        "Completed checkpoint evaluation lacks finite return/J: " + run["checkpoint_eval_path"])


def aggregate(a, summaries, metrics):
    grouped = {}
    for condition in CONDITIONS:
        selected = {row["seed"]: row for row in summaries if row["condition"] == condition}
        require(set(selected) == {0, 1, 2}, "Missing planned condition/seed slot: " + condition)
        grouped[condition] = {
            "statuses": [selected[s]["status"] for s in range(3)],
            "checkpoint_eval_statuses": [selected[s]["checkpoint_eval_status"] for s in range(3)],
            "collapse_per_seed": [selected[s]["collapse"] for s in range(3)],
            "collapse_count": sum(selected[s]["collapse"] is True for s in range(3)),
            "metrics": {key: a.mean_sd([selected[s].get(key) for s in range(3)]) for key in metrics},
        }
    return grouped


def comparisons(grouped, metrics):
    rows = []
    for metric in metrics:
        x, y = (grouped[name]["metrics"][metric] for name in ("E2_beta0", "FRS1"))
        difference = threshold = None
        interpretation = "unresolved: all three comparable seed values are required"
        if x["mean"] is not None and y["mean"] is not None:
            difference = y["mean"] - x["mean"]
            threshold = 2 * math.sqrt((x["sd"] ** 2 + y["sd"] ** 2) / 2)
            interpretation = ("equal observed means" if difference == 0 else
                              "unresolved: below two pooled seed SD" if abs(difference) < threshold else
                              "at least two pooled seed SD; descriptive, not a significance test")
        rows.append({"metric": metric, "FRS1_mean": y["mean"], "FRS1_sample_sd": y["sd"],
                     "E2_beta0_mean": x["mean"], "E2_beta0_sample_sd": x["sd"],
                     "difference_FRS1_minus_E2_beta0": difference, "two_pooled_seed_sd": threshold,
                     "interpretation": interpretation})
    return rows


def ratio_summary(a, run):
    train = [r for r in run["rows"] if "train_episode_reward" in r]
    valid = [r for r in train if all(a.finite(r.get(k)) for k in ("max_ratio", "median_ratio", "kl_true_per_action"))]
    record = {"run": run["spec"]["run_id"], "seed": run["spec"]["seed"], "status": run["status"],
              "scope": "observed training iterations", "observations": len(valid),
              "definitions": "max_ratio/median_ratio summarize absolute gradient/denominator over actor coordinates; no spike threshold"}
    for key in ("max_ratio", "median_ratio", "kl_true_per_action"):
        peak = max(valid, key=lambda row: row[key]) if valid else {}
        record.update({key + "_maximum": peak.get(key), key + "_maximum_itr": peak.get("itr"),
                       key + "_time_median": float(np.median([r[key] for r in valid])) if valid else None})
    peak = max(valid, key=lambda row: row["kl_true_per_action"]) if valid else {}
    record.update({"max_ratio_at_peak_KL": peak.get("max_ratio"), "median_ratio_at_peak_KL": peak.get("median_ratio")})
    return record


def draw(a, runs, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    omissions = {"nonpositive_log_points": {}, "undefined_noise_windows": {}}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (metric, title, logscale) in zip(axes.flat, PANELS):
        for condition in CONDITIONS:
            selected = [r for r in runs if r["spec"]["condition"] == condition]
            a.plot_group(ax, selected, metric, "step", condition, COLORS[condition], logscale)
            if metric == "x0_sat_frac_mean":
                a.plot_group(ax, selected, "action_oor_frac", "step", condition + " action", COLORS[condition], linestyle="--")
            if logscale:
                omissions["nonpositive_log_points"][condition + ":" + metric] = sum(
                    a.finite(x) and x <= 0 for r in selected for x in a.series(r, metric, "step").values())
            if metric == "B_env":
                omissions["undefined_noise_windows"][condition] = sum(
                    w["reason"] is not None for r in selected for w in r["windows"])
                for run in selected:
                    x = [w["step"] / 1e6 for w in run["windows"]
                         if w["reason"] == "nonpositive_split_dot" and a.finite(w.get("step"))]
                    ax.scatter(x, [.025] * len(x), marker="x", s=14, alpha=.45,
                               color=COLORS[condition], transform=ax.get_xaxis_transform())
        if logscale:
            ax.set_yscale("log")
        if metric == "B_env":
            ax.axhline(40, color="gray", linestyle=":", linewidth=.8)
        if metric == "grad_cos_prev":
            ax.axhline(0, color="gray", linewidth=.5)
        ax.set_title(title)
        ax.set_xlabel("Training environment steps (millions)")
        ax.grid(alpha=.18)
    axes.flat[0].legend(fontsize=9, frameon=False)
    axes.flat[7].text(.02, .98, "solid: x0 saturation\ndashed: executed |a| > 1", transform=axes.flat[7].transAxes, va="top", fontsize=8)
    fig.tight_layout()
    caption = ("FRS1 versus completed E2, E2_beta0 and SGD10: the eight panels use raw saved metrics against training environment steps. "
               "Thin curves are seeds 0/1/2; bold means and bands use sample SD (ddof=1) only where all three finite values exist, without smoothing or extrapolation. "
               "B_env is 10 times summed split-gradient squared difference divided by summed split dot over complete nine-training-iteration windows; undefined windows are gaps, crosses denote nonpositive denominators, and B_env=40 is a descriptive guide under the approximate independent-environment interpretation. "
               "Log axes omit nonpositive points/lower SD bounds without epsilon substitution; saturation and action violation use training rows only. "
               "Final in-training evaluation is itr130 at9.36M steps, whereas training ends at10.08M; split estimates are batch-coupled and three runs do not establish a general ordering.")
    for extension in ("png", "pdf"):
        fig.savefig(out / ("frs_comparison." + extension), dpi=180,
                    metadata={"Description": caption} if extension == "png" else {"Subject": caption})
    plt.close(fig)
    selected = [r for r in runs if r["spec"]["condition"] == "FRS1"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, metric in zip(axes, ("max_ratio", "median_ratio", "kl_true_per_action")):
        a.plot_group(ax, selected, metric, "itr", "FRS1", COLORS["FRS1"], logscale=True)
        ax.set_yscale("log")
        ax.set_ylabel(metric)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel("Iteration (training updates only)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    ratio_caption = ("FRS1 coordinate-gradient ratios and sampled transition KL versus training iteration, with individual seeds and mean ± sample SD across all three seeds. "
                     "max_ratio and median_ratio are the maximum and global midpoint median of |loss gradient|/frozen denominator before SGD; the median is over coordinates, not seeds. "
                     "The aligned panels expose temporal coincidence with KL excursions without defining a new spike threshold or a causal test; raw per-seed maxima, temporal medians and the ratios at each seed's maximum-KL iteration are in ratio_extrema.csv. "
                     "Nonpositive log values and lower SD bounds are omitted without replacement, failed prefixes remain visible, and missing values never receive survivor-only means.")
    for extension in ("png", "pdf"):
        fig.savefig(out / ("frs_ratios_kl." + extension), dpi=180,
                    metadata={"Description": ratio_caption} if extension == "png" else {"Subject": ratio_caption})
    plt.close(fig)
    return omissions, caption, ratio_caption


def short_report(a, grouped, comparison, ratios, stats, all_complete):
    def estimate(condition, metric):
        value = grouped[condition]["metrics"][metric]
        return a.fmt(value["mean"]) + " ± " + a.fmt(value["sd"])
    selected = ("eval_itr130", "train_last10", "J_disc_last10", "median_kl", "theta_dist_final")
    lookup = {row["metric"]: row for row in comparison}
    primary = lookup["J_disc_last10"]
    effect = primary["difference_FRS1_minus_E2_beta0"]
    threshold = primary["two_pooled_seed_sd"]
    primary_answer = ("The last-10 discounted-objective comparison is unavailable."
                      if effect is None else
                      "The last-10 discounted-objective difference remains unresolved under the two-pooled-SD rule."
                      if effect == 0 or abs(effect) < threshold else
                      "FRS1's last-10 discounted objective is " + ("higher" if effect > 0 else "lower") +
                      " by at least two pooled seed SD; this is a descriptive comparison.")
    table = a.md_table(["Metric", "FRS1 mean ± SD", "E2_beta0 mean ± SD", "Difference / 2 pooled SD"],
                       [[key, estimate("FRS1", key), estimate("E2_beta0", key),
                         a.fmt(lookup[key]["difference_FRS1_minus_E2_beta0"]) + " / " + a.fmt(lookup[key]["two_pooled_seed_sd"])] for key in selected])
    peaks = "; ".join("seed%d KL max %s at itr%s (max/median ratio %s/%s)" %
                       (r["seed"], a.fmt(r["kl_true_per_action_maximum"]), r["kl_true_per_action_maximum_itr"],
                        a.fmt(r["max_ratio_at_peak_KL"]), a.fmt(r["median_ratio_at_peak_KL"])) for r in ratios)
    status = "All three FRS1 training runs and final-checkpoint evaluations completed." if all_complete else "The campaign terminated with incomplete or failed FRS1 evidence; affected endpoint comparisons are unavailable."
    return ("# Frozen-rescaling results\n\n"
            "**Question.** Does frozen pretrained-gradient RMS rescaling improve E2 return or stability relative to E2_beta0?\n\n"
            "**Answer.** " + status + " " + primary_answer + " These three-seed comparisons do not establish equivalence to adaptive Adam or a convergence theorem.\n\n"
            + table + "\n\n![FRS comparison](frs_comparison.png)\n\n"
            "Thin seeds and mean ± sample SD use native results; [captions](captions.md) define all panels. In-training evaluation ends at itr130/9.36M steps, before the state139 checkpoint at10.08M. The [per-seed](seed_results.csv), [condition](condition_results.csv) and [comparison](comparisons.csv) tables include final-checkpoint returns/J under both samplers, delta-J, noise, collapse and timing. No survivor-only means are used.\n\n"
            "**Ratio excursions.** " + peaks + ". The [ratio/KL figure](frs_ratios_kl.png) and [raw extrema](ratio_extrema.csv) show temporal coincidence without a new spike threshold or causal claim.\n\n"
            "**Frozen scales.** Median r=" + a.fmt(stats["global_median_r"]) + "; exact zeros=" + str(stats["r_zero_count"]) + "; strictly floored coordinates=" + str(stats["floored_count"]) + ". The largest " + str(stats["top1pct_coordinate_count"]) + " coordinates carry " + a.fmt(stats["top1pct_squared_mass_share"]) + " of total r². [All seven quantiles and definitions](rescale_statistics.csv) come from the passed cached-input gate.\n\n"
            "**Setup and limits.** FRS1 uses fixed LR1e-4 and the unchanged E2 estimator, frozen float64 RMS of25 saved V4 full-batch gradients (source seed1000; processing seed4000), and denominator max(r,0.001×global median)+1e-8. Saved per-environment float32 gradients summed in float64 need not reproduce shuffled training accumulation bitwise. Gym initial states are not paired; the old theta0 evaluation has three repeats, while FRS final evaluations have one. Batch-coupled noise windows are approximate. Separately, the prior Phase1 signal-norm percentile CI limitation remains; r statistics are RMS heterogeneity, not signal CIs.\n\n"
            "**Decision.** Review these candidates for keep, revise or leave uncurated. No extra runs, rescue changes or automatic curation are authorized.\n")


def generate(root, reference, output, publish_report=False):
    root, reference, output = root.resolve(), reference.resolve(), output.resolve()
    require(root / "runs" in output.parents, "Analysis output must be a new directory inside this study's runs")
    require(not output.exists(), "Preserve existing analysis; output directory already exists")
    require(not publish_report or not (root / "REPORT.md").exists(), "Refusing to overwrite study REPORT.md")
    a = helper(reference)
    campaign = a.read_json(root / "runs/campaign.json", {})
    plan = a.read_json(root / "runs/queue_plan.json", {})
    require(campaign.get("status") in ("complete", "complete_with_failures"), "FRS campaign is not terminal")
    for submitted in campaign.get("submitted", []):
        require(str(submitted.get("scheduler_state", "")).split(" ")[0] in SCHEDULER_TERMINAL,
                "Submitted allocation has unknown/nonterminal scheduler state")
    specs = plan.get("runs", campaign.get("runs", []))
    require({s["run_id"] for s in specs} == {name for seed in range(3) for name in
            ("FRS1_seed%d" % seed, "ckpt_FRS1_seed%d_seed5000" % seed)}, "Missing or additional approved run slots")
    require({(s["condition"], s["seed"]) for s in specs if s["mode"] == "train"} ==
            {("FRS1", s) for s in range(3)}, "Invalid FRS training conditions/seeds")
    commands = manifest_set(a, root, specs)
    gate_path = root / "provenance/committed_gate.json"
    gate = a.read_json(gate_path, {})
    require(gate.get("passed") is True, "Committed FRS numerical gate has not passed")
    scale_path = Path(gate["rescale_path"]).resolve()
    stats_path = scale_path.parent / "rescale_stats.json"
    stats = a.read_json(stats_path, {})
    require(stats.get("status") == "completed" and stats.get("processing_seed") == 4000 and
            stats.get("parameter_count") == 568392 and stats.get("floor") == .001 and stats.get("epsilon") == 1e-8,
            "Frozen scale statistics are missing or incompatible")
    require(set(stats.get("r_quantiles", {})) == {"0", "1", "10", "50", "90", "99", "100"}, "Required r quantiles missing")
    require(all(key in stats for key in ("r_zero_count", "floored_count", "top1pct_coordinate_count",
                                        "top1pct_squared_mass_share")) and
            (stats_path.parent / "rescale_stats.csv").is_file(), "Required frozen-statistics evidence missing")
    reference_campaign = a.read_json(reference / "runs/campaign.json", {})
    require(reference_campaign.get("status") == "complete", "Stage3 reference campaign is not completed")
    reference_specs = [s for s in reference_campaign["runs"] if
                       s.get("mode") == "train" and s.get("condition") in CONDITIONS[1:]]
    require({(s["condition"], s["seed"]) for s in reference_specs} ==
            {(c, s) for c in CONDITIONS[1:] for s in range(3)}, "Missing reference condition/seed slots")
    reference_ids = {s["run_id"] for s in reference_specs} | {"ckpt_" + s["run_id"] + "_seed5000" for s in reference_specs} | {"ckpt_theta0_seed5000"}
    commands += manifest_set(a, reference, [s for s in reference_campaign["runs"] if s["run_id"] in reference_ids])
    theta0_path = reference / "ckpt_theta0_seed5000/ckpt_eval.json"
    theta0 = a.read_json(theta0_path, {})
    require(theta0.get("status") == "completed", "Completed theta0 evaluation is required")
    runs = a.load_runs(root, {"runs": specs}, False) + a.load_runs(reference, {"runs": reference_specs}, False)
    require(all(r["status"] == r["checkpoint_eval_status"] == "complete" for r in runs if r["spec"]["condition"] != "FRS1"),
            "Requested completed Stage3 reference is incomplete")
    validate_runs(a, runs)
    summaries = [a.summarize(run, theta0) for run in runs]
    metrics = list(a.METRICS) + ["wall_seconds", "policy_action_oor_mean", "longest_zero_update_streak"]
    grouped = aggregate(a, summaries, metrics)
    comparison = comparisons(grouped, metrics)
    ratios = [ratio_summary(a, r) for r in runs if r["spec"]["condition"] == "FRS1"]
    all_complete = all(r["status"] == r["checkpoint_eval_status"] == "complete" for r in runs)
    output.mkdir(parents=True)
    candidates = output / "candidates"
    candidates.mkdir()
    record = {"status": "generating", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "study": str(root),
              "reference": str(reference), "campaign": str(root / "runs/campaign.json"),
              "queue_plan": str(root / "runs/queue_plan.json"), "gate": str(gate_path),
              "rescale_statistics_source": str(stats_path), "theta0_evaluation": str(theta0_path),
              "sample_sd_ddof": 1, "mean_policy": "three finite seed values required; no survivor means",
              "all_training_and_evaluation_complete": all_complete, "curation_status": "unreviewed",
              "phase1_uncertainty_review": a.noise_uncertainty_review(reference) or None,
              "commands": commands, "provenance_commands": []}
    a.write_json(output / "analysis_record.json", record)
    try:
        source = output / "analysis_source"
        source.mkdir()
        shutil.copy2(__file__, source / "analyze_frs.py")
        shutil.copy2(reference / "code/analyze_stage3.py", source / "analyze_stage3_reference.py")
        parent = root.parents[1]
        for name, argv in (("parent_commit", ["git", "-C", str(parent), "rev-parse", "HEAD"]),
                           ("analysis_source_diff", ["git", "-C", str(parent), "diff", "HEAD", "--", str(Path(__file__).resolve()), str(reference / "code/analyze_stage3.py")])):
            value = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            record["provenance_commands"].append({"command": argv, "returncode": value.returncode})
            if name == "parent_commit":
                record[name] = value.stdout.strip()
            else:
                (source / "parent_analysis.diff").write_text(value.stdout)
        a.write_csv(candidates / "seed_results.csv", summaries)
        flat = []
        for condition, entry in grouped.items():
            row = {"condition": condition, **{k: v for k, v in entry.items() if k != "metrics"}}
            for metric, values in entry["metrics"].items():
                row.update({metric + "_mean": values["mean"], metric + "_sample_sd": values["sd"],
                            metric + "_n": values["n_available"], **{metric + "_seed%d" % s: values["per_seed"][s] for s in range(3)}})
            flat.append(row)
        a.write_csv(candidates / "condition_results.csv", flat)
        a.write_csv(candidates / "comparisons.csv", comparison)
        a.write_csv(candidates / "ratio_extrema.csv", ratios)
        a.write_csv(candidates / "all_runs_long.csv", [{"run": r["spec"]["run_id"], "condition": r["spec"]["condition"],
                    "seed": r["spec"]["seed"], **row} for r in runs for row in r["rows"]])
        a.write_csv(candidates / "noise_windows.csv", [{"run": r["spec"]["run_id"], "condition": r["spec"]["condition"],
                    "seed": r["spec"]["seed"], **row} for r in runs for row in r["windows"]])
        shutil.copy2(stats_path, candidates / "rescale_statistics.json")
        shutil.copy2(stats_path.parent / "rescale_stats.csv", candidates / "rescale_statistics.csv")
        omissions, caption, ratio_caption = draw(a, runs, candidates)
        table_caption = ("Per-seed and condition tables mirror Stage3 definitions; all means/sample SD require three finite seed summaries, and failed runs retain finite prefixes with unavailable endpoints. "
                         "Last10 uses the final ten training iterations only for completed140-row runs; final checkpoint evaluations use both samplers and delta-J subtracts saved theta0 means. "
                         "Collapse is an observed evaluation below half that seed's initial evaluation or a numerical nonfinite failure; undefined diagnostics alone are not collapse. "
                         "Timing includes native mean training-iteration time and process wall time; repeated diagnostic observations are not independent seeds. "
                         "Comparisons use twice sqrt((s_FRS1²+s_E2_beta0²)/2), not an SE or significance test; old theta0 has three evaluation repeats, while FRS final evaluation has one, and Gym initial states are not paired.")
        (candidates / "captions.md").write_text("# Candidate captions\n\n## Eight metrics\n\n" + caption + "\n\n## Ratios and KL\n\n" + ratio_caption +
                    "\n\n## Tables\n\n" + table_caption + "\n\n## Frozen RMS statistics\n\nStatistics are copied from the passed gate's deterministic conversion of25 saved V4 batches, without new sampling. Quantiles are raw coordinate RMS; floor counts use strict pre-epsilon comparison and the top1% share uses ceil(.01P) coordinates. They do not repair or replace the prior Phase1 signal-norm confidence intervals.\n")
        report = short_report(a, grouped, comparison, ratios, stats, all_complete)
        (candidates / "REPORT.md").write_text(report)
        if publish_report:
            with (root / "REPORT.md").open("x") as stream:
                prefix = str(candidates.relative_to(root)) + "/"
                stream.write(re.sub(r"\]\(([^)]+)\)", lambda match: "](" + prefix + match.group(1) + ")", report))
            record["published_report"] = str(root / "REPORT.md")
        record.update({"status": "completed_candidates" if all_complete else "terminal_candidates_with_failures",
                       "summaries": summaries, "conditions": grouped, "comparisons": comparison,
                       "ratio_extrema": ratios, "log_omissions": omissions,
                       "artifacts": [str(p.relative_to(output)) for p in sorted(candidates.iterdir()) if p.is_file()],
                       "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    except BaseException as error:
        record.update({"status": "analysis_failed", "error": type(error).__name__ + ": " + str(error)})
        raise
    finally:
        a.write_json(output / "analysis_record.json", record)
    print(json.dumps({"status": record["status"], "candidates": str(candidates),
                      "all_runs_complete": all_complete, "curation": "not performed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "stage3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/final_analysis")
    parser.add_argument("--publish-report", action="store_true", help="Opt in to a new study REPORT.md; never overwrite")
    args = parser.parse_args()
    generate(args.root, args.reference, args.output, args.publish_report)
