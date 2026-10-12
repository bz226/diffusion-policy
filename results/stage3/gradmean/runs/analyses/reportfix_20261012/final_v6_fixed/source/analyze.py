#!/usr/bin/env python
"""Analyze approved Addendum G evidence into a NEW partial/final candidate tree.

Run on an allocated CPU node. Reads completed groups and existing references;
does not launch training, change raw evidence, or curate outputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from analysis_core import (POINTS, ROUND_BATCHES, VARIANTS, cross_point_cosines,
                           discover, path_rows, score_and_lemma_rows)
from analysis_references import drift_comparisons, phase1_comparisons
from gradmean_math import summarize_gram


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT
STAGE3 = ROOT.parent
PARENT = ROOT.parents[2]


def serializable(value):
    if isinstance(value, np.ndarray):
        return [serializable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite analysis output must have an explicit unavailable reason")
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(serializable(value), allow_nan=False) if isinstance(value, (list, tuple, dict)) else serializable(value)
                             for key, value in row.items()})


def convergence(points):
    rows = []
    for name, point in points.items():
        for groups in range(1, point.n + 1):
            summary = summarize_gram(point.grams[3, :groups, :groups], point.q[:groups, 3, 3])
            rho, ci = summary["estimate"]["rho"], summary["jackknife90"]["rho"]
            rows.append({"point": name, "K": groups * 50, "groups": groups,
                         "rho": rho, "rho_se": ci["se"], "rho_low": ci["lower"], "rho_high": ci["upper"],
                         "c": summary["estimate"]["cosine_reliability"],
                         "split_count": summary["metadata"]["split_count"],
                         "split_seed": summary["metadata"]["split_seed"],
                         "round_boundary": groups * 50 in ROUND_BATCHES,
                         "threshold_met": rho is not None and rho >= .68,
                         "uncertainty_reason": ci["reason"]})
    return rows


def check_final(points, campaign):
    if set(points) != set(POINTS):
        raise ValueError("Final analysis requires all five policies; use --partial for available evidence")
    fixed = campaign.get("fixed_v6_plan")
    if fixed:
        required = dict(approved=True, new_groups_per_point=10, batches_per_group=50,
                        legacy_admissions_closed=True, adaptive_stopping=False)
        if any(fixed.get(key) != value for key, value in required.items()):
            raise ValueError("Fixed V6 plan does not match the approved compute stop")
        expected = set(fixed.get("group_indices", []))
        if expected != set(list(range(82, 90)) + [91, 92]):
            raise ValueError("Fixed V6 logical group set differs from the approved plan")
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                    "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}
        if any(not entry.get("terminal_recorded") or entry.get("scheduler_state") not in terminal
               for entry in campaign.get("submitted", [])):
            raise ValueError("All already admitted legacy/new jobs must finish before final analysis")
        for name, point in points.items():
            v6_paths = [path for path, meta in zip(point.paths, point.metas) if "V6" in meta.get("variant_ids", ())]
            groups = {int(path.name.split("_j", 1)[1].split("_", 1)[0]) for path in v6_paths}
            if len(v6_paths) != 10 or groups != expected:
                raise ValueError("Exactly ten approved six-variant groups required at " + name)
        return
    for name, point in points.items():
        status = campaign.get("points", {}).get(name, {}).get("status")
        rounds = campaign.get("points", {}).get(name, {}).get("rounds", [])
        if not rounds:
            raise ValueError("Final campaign point has no recorded closure round: " + name)
        closure = rounds[-1]
        closure_paths = set(closure["groups"])
        ids = np.asarray([i for i, path in enumerate(point.paths) if path.name in closure_paths], dtype=int)
        target = closure["target_batches"]
        if target not in ROUND_BATCHES or len(ids) * 50 != target or len(ids) != len(closure_paths):
            raise ValueError("Original closure does not identify a complete approved round: " + name)
        summary = summarize_gram(point.grams[3][np.ix_(ids, ids)], point.q[ids, 3, 3])
        rho = summary["estimate"]["rho"]
        if status not in ("closed_rho", "closed_cap"):
            raise ValueError("Final campaign point is not closed: " + name)
        if status == "closed_rho" and not (rho is not None and rho >= .68):
            raise ValueError("Campaign closure conflicts with original closing-round data")
        if status == "closed_cap" and target != 4100:
            raise ValueError("Campaign cap closure lacks its original 4100 batches")


def display(value, precision=3):
    if value is None:
        return "—"
    if abs(value) >= 1e5 or (0 < abs(value) < .001):
        return ("%." + str(precision) + "g") % value
    return ("{:,." + str(precision) + "f}").format(value)


def write_summary(path, payload, partial):
    main = {row["point"]: row for row in payload["main"] if row["variant"] == "V4"}
    unresolved = [point for point, row in main.items() if row["rho"] is None or row["rho"] < .68]
    lower_bounds = [point for point, row in main.items() if row["B_ep_type"] == "lower_bound"]
    unavailable = [point for point, row in main.items() if row["B_ep_type"] == "unavailable"]
    kind = "Partial" if partial else "Final"
    lines = ["# %s fixed-policy gradient measurements" % kind, "",
             "**Question.** How accurately can the mean gradient and its noise be measured along the DPPO and NC4 checkpoint paths?", "",
             "**Answer.** Completed measurements cover %d of five policies. %d measured policies meet the prescribed V4 split-correlation threshold of 0.68; %d remain below it or unresolved at the current sample count. %s" % (
                 len(main), len(main) - len(unresolved), len(unresolved),
                 "This is an interim analysis; missing policies and uncertainty remain visible." if partial else
                 "The user replaced adaptive collection with a fixed compute plan; completion does not establish a resolved direction." if payload.get("fixed_v6_plan") else
                 "A policy closed at the sample cap can still have an unresolved direction."), "",
             "| Policy | Independent batches | V4 squared signal | V4 B in episodes | Split correlation |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in POINTS:
        row = main.get(name)
        if row is None:
            lines.append("| %s | — | — | Not measured | — |" % name)
            continue
        noise = ("≥ " + display(row["B_ep_lower_bound"], 0)) if row["B_ep_type"] == "lower_bound" else display(row["B_ep"], 0)
        lines.append("| %s | %d | %s | %s | %s |" % (name, row["K"], display(row["u_sq"]), noise, display(row["rho"])))
    lines.extend(["", "The signed squared signal uses V4 in paper gradient units. The ≥ entries use the trace estimate divided by the upper endpoint of the signal's 90% interval; they are plug-in lower bounds, without a joint confidence guarantee. Undefined values are preserved rather than set to zero.", "",
                  "**Uncertainty.** Every measurement interval deletes one independent group of 50 whole batches, recomputes the statistic, and uses ±1.645 standard errors. Each batch contains 80 episodes. Groups are paired across variants at one policy and independent across policies; the 20 random half-splits are not 20 independent experiments.", "",
                  "**Comparisons.** [Path alignment](path_alignment.csv) uses endpoint gradients and checkpoint displacements; its trapezoid approximates a chord integral, not the complete training trajectory. [Existing SGD drift](drift_summary.csv) uses all 126 training returns per seed and is a path-averaged proxy, particularly approximate for SGD10. [Earlier Phase-1 estimates](phase1_comparison.csv) remain provisional because their uncertainty was not valid.", "",
                  "**Boundary.** V1–V3 publish noise scales and cosines only; V4/V5 norms and covariance traces use paper units. The [variance-lemma comparison](score_variance_lemma.csv) uses an observed reward maximum and 200 decision-score samples from one group, so it has no group-jackknife interval and is not a proven bound.", "",
                  "[V4 path figure](figures/v4_path.png) · [All results and intervals](main_results.csv) · [Per-policy returns and diagnostics](per_point.csv) · [Table definitions](table_captions.md) · [Analysis record](../analysis_status.json).", "",
                  "These candidates are uncurated. No training or additional scientific sampling was performed by this analysis."])
    # Plot filenames are supplied by the rendering helper, then patched to its
    # actual V4 artifact before this file is delivered.
    text = "\n".join(lines) + "\n"
    if len(text.split()) > 600:
        raise ValueError("Short analysis summary exceeded the research-report word budget")
    path.write_text(text)


TABLE_CAPTIONS = """# Table definitions

`main_results.csv` reports each policy and estimator with its number of independent whole batches. V4/V5 squared signals, norms and episode-equivalent covariance traces use paper ascent-gradient units; V1–V3 dimensioned quantities and effective ranks are omitted by protocol. B is the episode-equivalent trace divided by squared signal; unresolved signals produce the explicitly typed upper-signal plug-in lower bound, and its sampling forecast is also a lower bound. Expected training-batch cosine uses 80 episodes, whereas c and c_U concern the entire measurement mean. Every available interval is a delete-one-group 90% normal interval; raw internal ratios are retained separately for audit, not substituted for resolved point estimates.

`per_point.csv` gives raw discounted and undiscounted return means, group-jackknife standard errors and intervals, fixed checkpoint distance, saturation, action-range fractions and observed reward maxima. One group contains 50 fresh batches, each with 80 complete episodes. Checkpoint distances are deterministic functions of saved weights and therefore have no sampling interval.

`same_point_cosines.csv` excludes same-batch cross products before forming mean-gradient cosines and deletes groups jointly across estimators. `cross_point_cosines.csv` uses independent products of policy means and sums the separate policies' jackknife variance contributions. Negative signals give unavailable cosines; estimated cosines outside [−1, 1] remain visible and indicate unresolved estimation error rather than clipped population values.

`path_alignment.csv` reports the six prescribed checkpoint segments: displacement norm, V4 start-gradient dot product and cosine, endpoint-trapezoid prediction, and the measured difference in raw discounted returns. Policy groups are independent for uncertainty, while quantities measured together at a policy remain paired. The endpoint trapezoid neglects curvature between checkpoints and is not an exact training-path integral.

`contrasts.csv` contains paired-group V5/V4 noise-trace ratios and V2/V3 noise-scale ratios. A noise-scale ratio is unavailable if either denominator signal is unresolved. `split_convergence.csv` recomputes the V4 split correlation and group-jackknife interval after each completed group; its repeated prefixes are dependent. The original round markers remain descriptive after the user replaced adaptive continuation with the fixed ten-new-group V6 plan.

`score_variance_lemma.csv` uses the mean squared norm of an unweighted decision score summed over ten denoising transitions and 24 coordinates, along with its decision-index decile means and 99th percentile. The 200 decisions come from the first batch of one successful group per policy. Their uncertainty cannot be estimated by deleting groups, and an observed reward maximum is not a global reward bound; the lemma ratio is an empirical scale comparison only.

`drift_per_seed.csv` fits discounted training return against the number of completed actor updates using all 126 observations. `drift_summary.csv` aggregates the three independent seeds; its SE equals sample SD divided by √3, with a 1.645 normal multiplier. Division by actor_lr/2500 yields a path-averaged squared-gradient proxy, not a local measurement; comparison with P0 combines independent seed and measurement-group uncertainty, and SGD10 is explicitly flagged as a large-step approximation.

`phase1_comparison.csv` retains the earlier P0/N2 point estimates and compares them descriptively with new intervals or bounds. Earlier uncertainty is not reused, within-batch coupling affected those estimates, and the critic/scaler preparation differs. Whether an old point falls inside a new interval is not a formal consistency test.
"""


def main():
    global ROOT, STAGE3, PARENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--root", type=Path, default=CODE_ROOT, help="Absolute measurement/output study root")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    STAGE3, PARENT = ROOT.parent, ROOT.parents[2]
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists() or "curated" in output.relative_to(ROOT).parts:
        raise ValueError("Use a NEW uncurated output directory inside this study")
    output.mkdir(parents=True)
    candidates, internal = output / "candidates", output / "internal"
    candidates.mkdir()
    internal.mkdir()
    started = time.monotonic()
    status = {"status": "running", "partial": args.partial, "command": sys.argv,
              "started_utc": datetime.now(timezone.utc).isoformat(), "outputs": [],
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "new_scientific_samples": 0,
              "python": sys.version, "numpy": np.__version__}
    write_json(output / "analysis_status.json", status)
    try:
        source = output / "source"
        source.mkdir()
        for name in ("analyze.py", "analysis_core.py", "analysis_references.py", "gradmean_math.py", "analysis_v6.py"):
            shutil.copy2(CODE_ROOT / name, source / name)
        shutil.copy2(CODE_ROOT / "code/analysis_plots.py", source / "analysis_plots.py")
        shutil.copy2(CODE_ROOT / "code/checkpoint_distances.py", source / "checkpoint_distances.py")
        shutil.copy2(CODE_ROOT / "code/analysis_report.py", source / "analysis_report.py")
        status["analysis_source_root"] = str(CODE_ROOT)
        status["analysis_source_parent_commit"] = subprocess.check_output(["git", "-C", str(CODE_ROOT.parents[2]), "rev-parse", "HEAD"], text=True).strip()
        status["parent_commit"] = subprocess.check_output(["git", "-C", str(PARENT), "rev-parse", "HEAD"], text=True).strip()
        points, inventory = discover(ROOT)
        campaign_path = ROOT / "runs/campaign.json"
        campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
        if not args.partial:
            check_final(points, campaign)
        print("Loaded completed measurement groups: " + repr({name: point.k for name, point in points.items()}), flush=True)
        sys.path.insert(0, str(CODE_ROOT / "code"))
        from checkpoint_distances import distance_summary, load_checkpoint_vectors
        first_meta = next(iter(points.values())).paths[0] / "meta.json" if points else None
        vectors, checkpoint_meta = load_checkpoint_vectors(mapping_path=ROOT / "provenance/checkpoints.json",
            source_path=CODE_ROOT.parents[2] / "results/stage3_gradmean/dppo", gradient_meta=first_meta)
        for point in points.values():
            if point.specs != checkpoint_meta["parameter_specs"]:
                raise ValueError("Checkpoint and gradient parameter order differ")
        distances = distance_summary(vectors)
        distance = distances["distance_from_P0"]
        write_json(internal / "point_statistics.json", {name: point.summaries for name, point in points.items()})
        write_json(internal / "checkpoint_metadata.json", {**checkpoint_meta, **distances})
        main_rows = [row for name, point in points.items() for row in point.public_rows(distance[name])]
        point_rows = [point.point_row(distance[name]) for name, point in points.items()]
        same_cos = [row for point in points.values() for row in point.same_cosines()]
        cross_cos = cross_point_cosines(points)
        paths = path_rows(points, vectors)
        contrasts = [row for point in points.values() for row in point.contrasts()]
        scores = score_and_lemma_rows(points)
        drift_seeds, drift_summary = drift_comparisons(STAGE3, points.get("P0"))
        phase1 = phase1_comparisons(STAGE3, points)
        conv = convergence(points)
        tables = {"main_results": main_rows, "per_point": point_rows, "same_point_cosines": same_cos,
                  "cross_point_cosines": cross_cos, "path_alignment": paths, "contrasts": contrasts,
                  "score_variance_lemma": scores, "drift_per_seed": drift_seeds,
                  "drift_summary": drift_summary, "phase1_comparison": phase1, "split_convergence": conv}
        from analysis_v6 import analyze_v6, write_note
        v6 = analyze_v6(points, campaign)
        for name, rows in v6["tables"].items():
            write_csv(candidates / (name + ".csv"), rows)
        write_json(candidates / "v6_results.json", v6)
        write_note(candidates / "V6_NOTE.md", v6)
        status["v6_consistency_failure"] = v6["consistency_failure"]
        status["v6_consistency_stop"] = v6["consistency_failure"]
        status["v6_consistency_points"] = [row["point"] for row in v6["tables"]["v6_consistency"] if row["consistency_failure"]]
        status["v6_counts"] = v6["counts"]
        for name, rows in tables.items():
            write_csv(candidates / (name + ".csv"), rows)
        public = {"scope": "partial" if args.partial else "final", "main": main_rows, "v6": v6,
                  "fixed_v6_plan": campaign.get("fixed_v6_plan"),
                  "tables": tables, "group_inventory": inventory,
                  "failed_or_incomplete_groups": [entry for entry in inventory if not entry["included"]],
                  "unavailable_points": [point for point in POINTS if point not in points],
                  "curation": "unreviewed; no automatic curation",
                  "uncertainty": "Delete one group of 50 batches; independent-policy variance sums; normal 90% intervals using 1.645."}
        write_json(candidates / "results.json", public)
        (candidates / "table_captions.md").write_text(TABLE_CAPTIONS)
        from analysis_plots import render
        figures = render(candidates / "figures", main_rows, conv, same_cos, cross_cos)
        write_summary(candidates / "SUMMARY.md", public, args.partial)
        if figures:
            summary_path = candidates / "SUMMARY.md"
            actual_v4 = os.path.relpath(figures[0]["png"], candidates)
            summary_path.write_text(summary_path.read_text().replace("figures/v4_path.png", actual_v4))
        from analysis_report import write_reports
        status["elapsed_seconds_at_report"] = time.monotonic() - started
        report_paths = write_reports(ROOT, candidates, public, figures,
                                     {**checkpoint_meta, **distances}, campaign, status,
                                     final=output.name == "final")
        if output.name == "final_v6_fixed":
            from analysis_v6 import publish_note
            status["published_V6_note"] = publish_note(ROOT, output, v6)
        status.update(status="complete", group_inventory=inventory,
                      completed_batches={name: point.k for name, point in points.items()},
                      campaign_status=campaign.get("status"), figures=figures,
                      summary=str(candidates / "SUMMARY.md"),
                      **report_paths,
                      outputs=[str(path.resolve()) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "analysis_status.json"])
    except BaseException as error:
        status.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "analysis_status.json", status)
    print(json.dumps({"status": status["status"], "output": str(output), "elapsed_seconds": status["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
