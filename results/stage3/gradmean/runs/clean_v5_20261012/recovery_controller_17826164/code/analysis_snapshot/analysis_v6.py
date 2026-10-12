"""V6-only reporting with paired new groups and all-group V4 signal.

Old V1-V5 measurements are never padded or discarded. Independent cohorts are
jackknife strata; deleting a new group removes its V4 and V6 measurements jointly.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import re
import numpy as np

from analysis_core import (POINTS, PointData, THEORY_SCALE, EPISODES, GROUP_SIZE,
                           group_mean, independent_result, interval_fields, jk_scalar)
from gradmean_math import VARIANTS, same_point_cross

ALL_VARIANTS = VARIANTS + ("V6",)
BINS = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250))


def _summary_scalar(summary, field):
    return {"value": summary["estimate"][field], **summary["jackknife90"][field]}


def _ratio(point, numerator, denominator):
    return point.scalar(lambda moment: float(moment["trace"][numerator] / moment["trace"][denominator])
                        if moment["trace"][denominator] > 0 else None)


def shared_signal_noise_scale(point, numerator_variant, signal_variant="V4"):
    """Same-cohort B with common V4 signal, useful for V4/V5 reference displays."""
    a, b = point.variants.index(numerator_variant), point.variants.index(signal_variant)
    value = point.scalar(lambda m: float(EPISODES * m["trace"][a] / m["signal"][b])
                         if m["signal"][b] > 0 and m["trace"][a] >= 0 else None)
    summary = point.summaries[signal_variant]
    bound = None
    if summary["scale_reporting"]["kind"] != "point":
        upper = summary["jackknife90"]["signal_sq"].get("upper")
        trace = point.full_moments["trace"][a]
        if upper is not None and upper > 0 and trace >= 0:
            bound = float(EPISODES * trace / upper)
        value = {"value": None, "se": None, "lower": None, "upper": None,
                 "reason": "common_V4_signal_unresolved"}
    return dict(value, lower_bound=bound, signal_variant=signal_variant,
                numerator_variant=numerator_variant, K=point.k,
                kind="point" if value["value"] is not None else "lower_bound" if bound is not None else "unavailable")


def _mixed_noise_scale(all_data, paired):
    """Two independent cohort strata, paired deletion inside the new stratum."""
    paired_lookup = {path.resolve(): i for i, path in enumerate(paired.paths)}
    new_global = np.asarray([i for i, path in enumerate(all_data.paths) if path.resolve() in paired_lookup], dtype=int)
    old_global = np.asarray([i for i, path in enumerate(all_data.paths) if path.resolve() not in paired_lookup], dtype=int)
    # PointData discovery preserves chronology; explicitly map instead of relying on it.
    new_to_paired = np.asarray([paired_lookup[all_data.paths[i].resolve()] for i in new_global], dtype=int)
    counts = {"new_paired": len(new_global)}
    if len(old_global):
        counts["legacy"] = len(old_global)
    def statistic(ids):
        old = old_global[ids["legacy"]] if len(old_global) else np.empty(0, dtype=int)
        selected_new = ids["new_paired"]
        all_ids = np.sort(np.concatenate((old, new_global[selected_new])))
        a = all_data.moments(all_ids)
        p = paired.moments(new_to_paired[selected_new])
        signal = a["signal"][3] if a is not None else None
        trace = p["trace"][5] if p is not None else None
        return {"B_ep": float(EPISODES * trace / signal) if signal is not None and signal > 0 and trace is not None and trace >= 0 else None}
    result = independent_result(statistic, counts)
    scale = all_data.summaries["V4"]["scale_reporting"]
    full = result["estimate"]["B_ep"]
    value = {"value": full, **result["jackknife90"]["B_ep"]}
    bound = None
    if scale["kind"] != "point":
        signal_upper = all_data.summaries["V4"]["jackknife90"]["signal_sq"].get("upper")
        trace = paired.full_moments["trace"][5]
        if signal_upper is not None and signal_upper > 0 and trace >= 0:
            bound = float(EPISODES * trace / signal_upper)
        value = {"value": None, "se": None, "lower": None, "upper": None,
                 "reason": "all_V4_signal_unresolved; trace/upper_signal is a plug-in lower bound"}
    return value, bound, result, counts


def _baseline_rows(point):
    rows = []
    for lo, hi in BINS:
        suffix = "_k%d_%d" % (lo, hi)
        fields = ["baseline_" + name + quantity + suffix for name in ("critic", "loo")
                  for quantity in ("_energy", "_residual_mean")]
        fields += ["frozen_reward_divisor"]
        if not all(field in row and row[field] not in (None, "")
                   for group in point.batches for row in group for field in fields):
            raise ValueError("V6 collection lacks required baseline-bin/scaler diagnostics")
        means = {field: np.asarray([np.mean([float(row[field]) for row in group])
                                   for group in point.batches]) for field in fields}
        record = {"point": point.point, "k_start": lo, "k_end_exclusive": hi,
                  "K_V6": point.k, "groups_V6": point.n,
                  "energy_definition": "mean_episode sum_bin gamma^(2k)*(G-b)^2",
                  "residual_mean_definition": "mean over episodes and k cells in bin, without gamma weights"}
        for field in fields:
            label = field.replace(suffix, "")
            record.update(interval_fields(label, group_mean(means[field])))
        a, b = means["baseline_critic_energy" + suffix], means["baseline_loo_energy" + suffix]
        def ratio(ids):
            return float(a[ids].mean() / b[ids].mean()) if b[ids].mean() > 0 else None
        ratio_stat = jk_scalar(ratio(np.arange(point.n)), [ratio(np.delete(np.arange(point.n), i))
                                                        for i in range(point.n)] if point.n > 1 else [])
        record.update(interval_fields("critic_over_loo_energy", ratio_stat))
        rows.append(record)
    return rows


def analyze_v6(points, campaign=None):
    campaign = campaign or {}
    main, contrasts, baselines, internals, counts, consistency = [], [], [], {}, {}, []
    for name in POINTS:
        if name not in points:
            counts[name] = {"K_all": 0, "K_V6": 0}
            continue
        all_data = points[name]
        paths = [path for path, meta in zip(all_data.paths, all_data.metas) if "V6" in meta.get("variant_ids", ())]
        counts[name] = {"K_all": all_data.k, "K_V6": len(paths) * GROUP_SIZE}
        if not paths:
            continue
        paired = PointData(name, paths, variants=ALL_VARIANTS)
        summary = paired.summaries["V6"]
        scalesq = THEORY_SCALE ** 2
        row = {"point": name, "K_all_V4": all_data.k, "K_V6": paired.k,
               "groups_all_V4": all_data.n, "groups_V6": paired.n,
               "episodes_V6": paired.k * EPISODES,
               "gradient_units": "paper episode-ascent; code vectors multiplied by2500",
               "noise_scale_denominator": "V4 signed U-statistic from all independent batches",
               "minimum_V6_batches_met": paired.k >= 500}
        row.update(interval_fields("V6_trace_ep", _summary_scalar(summary, "covariance_trace"), EPISODES * scalesq))
        row.update(interval_fields("V6_own_u_sq", _summary_scalar(summary, "signal_sq"), scalesq))
        row.update(interval_fields("V4_all_u_sq", _summary_scalar(all_data.summaries["V4"], "signal_sq"), scalesq))
        noise, bound, mixed, strata = _mixed_noise_scale(all_data, paired)
        row.update(interval_fields("V6_B_ep", noise))
        row.update(V6_B_ep_lower_bound=bound,
                   V6_B_ep_type="point" if noise["value"] is not None else "lower_bound" if bound is not None else "unavailable",
                   noise_uncertainty="independent legacy/new cohort variance contributions; new-group deletion paired across V4/V6",
                   jackknife_strata=strata)
        cross = same_point_cross(paired.grams[5], paired.grams[3],
                                 paired.sums[5] @ paired.sums[3].T,
                                 paired.q[:, 5, 5], paired.q[:, 3, 3], paired.q[:, 5, 3])
        ci = cross["jackknife90"]["cosine"]
        row.update(interval_fields("V6_V4_paired_cosine", {"value": cross["estimate"]["cosine"], **ci}))
        row.update(interval_fields("V6_V4_paired_mean_inner_product",
                                   {"value": cross["estimate"]["mean_inner_product"], **cross["jackknife90"]["mean_inner_product"]}, scalesq))
        signals_resolved = all(paired.summaries[v]["scale_reporting"]["kind"] == "point" for v in ("V4", "V6"))
        fails = bool(signals_resolved and ci.get("valid") and ci.get("upper") is not None and ci["upper"] < 1.)
        row.update(consistency_signals_resolved=signals_resolved,
                   consistency_status="failure_below_one" if fails else "unresolved" if not signals_resolved or not ci.get("valid") else "not_rejected",
                   consistency_failure=fails,
                   consistency_rule="at scheduled analysis: both paired signals resolved and valid 90%cosine upper<1; N/A never fails; repeated diagnostic not global test")
        main.append(row)
        consistency.append({key: row[key] for key in ("point", "K_V6", "consistency_status", "consistency_failure",
                                                     "consistency_signals_resolved", "V6_V4_paired_cosine", "V6_V4_paired_cosine_low", "V6_V4_paired_cosine_high")})
        for numerator, denominator, label in ((5, 3, "V6_trace_over_V4_trace"), (4, 5, "V5_trace_over_V6_trace")):
            contrasts.append({"point": name, "contrast": label, "K_paired": paired.k, "groups_paired": paired.n,
                              **interval_fields("ratio", _ratio(paired, numerator, denominator)),
                              "uncertainty": "paired delete-one-new-50-batch-group; both traces use identical cohort"})
        baselines.extend(_baseline_rows(paired))
        internals[name] = {"V6_summary": summary, "mixed_noise_scale": mixed,
                           "paired_mean_direction": cross, "V6_group_paths": [str(path) for path in paths],
                           "legacy_groups": all_data.n - paired.n,
                           "original_closure_status": campaign.get("points", {}).get(name, {}).get("status"),
                           "original_closure_rounds": campaign.get("points", {}).get(name, {}).get("rounds", [])}
    return {"counts": counts, "tables": {"v6_main_results": main, "v6_variance_ratios": contrasts,
                                         "v6_baseline_bins": baselines, "v6_consistency": consistency},
            "internal": internals,
            "consistency_failure": any(row["consistency_failure"] for row in consistency),
            "minimum_batches_all_points_met": all(counts[p]["K_V6"] >= 500 for p in POINTS),
            "missing_points": [p for p in POINTS if counts[p]["K_V6"] == 0],
            "uncertainty": "90%normal intervals,1.645SE. All calculations float64; group sums were storedfloat32.",
            "legacy_outputs_unchanged": True, "fixed_v6_plan": campaign.get("fixed_v6_plan")}


def _fmt(value, digits=3):
    return "N/A" if value is None else ("%." + str(digits) + "g") % value


def _ratio_interval(row):
    return "%s [%s,%s]" % (_fmt(row.get("ratio")), _fmt(row.get("ratio_low")), _fmt(row.get("ratio_high")))


def publish_note(root, output, result):
    """Publish the requested current report, preserving the candidate and prior note."""
    root, output = Path(root), Path(output)
    candidate = output / "candidates/V6_NOTE.md"
    gate_path = root / "runs/gate_v6/gate_summary.json"
    switch_path = root / "provenance/v6_source_switch.json"
    decision_path = root / "provenance/fixed_v6_plan_20261012.json"
    for path in (candidate, gate_path, switch_path, decision_path):
        if not path.is_file():
            raise ValueError("Final V6 report lacks required evidence: " + str(path))
    gate = json.loads(gate_path.read_text())
    switch = json.loads(switch_path.read_text())
    if gate.get("passed") is not True:
        raise ValueError("Do not publish successful V6 gate provenance without a passed gate")
    required = ("switched_utc", "old_source_commit", "new_source_commit")
    if any(not switch.get(key) for key in required):
        raise ValueError("Source switch must record timestamp and both revisions")
    checks = gate.get("checks", gate)
    values = []
    for key in ("H1", "H2", "H3", "H4"):
        value = gate.get(key, checks.get(key))
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise ValueError("Missing passed V6 gate result " + key)
        values.append(key + " passed")
    text = candidate.read_text()
    def destination(match):
        target = match.group(1)
        if "://" in target or target.startswith("#"):
            return match.group(0)
        return "](" + os.path.relpath(str((candidate.parent / target).resolve()), str(root.resolve())) + ")"
    text = re.sub(r"\]\(([^)]+)\)", destination, text)
    text += ("\n**Execution provenance.** " + ", ".join(values) +
             " ([gate](runs/gate_v6/gate_summary.json)). Source switched at " + switch["switched_utc"] +
             "; old revision `" + switch["old_source_commit"] + "`, new revision `" + switch["new_source_commit"] +
             "`. [Switch record](provenance/v6_source_switch.json). The [fixed compute decision](provenance/fixed_v6_plan_20261012.json) "
             "preceded V6 data; exact included counts appear above. Earlier batches and candidates are preserved.\n")
    target = root / "V6_NOTE.md"
    if target.exists():
        archive = output / "internal/V6_NOTE_before_final.md"
        if archive.exists():
            raise ValueError("Previous V6 report archive already exists")
        archive.write_bytes(target.read_bytes())
    target.write_text(text)
    return str(target)


def write_note(path, result):
    tables = result["tables"]
    ratios = {row["point"]: row for row in tables["v6_variance_ratios"] if row["contrast"] == "V6_trace_over_V4_trace"}
    main = {row["point"]: row for row in tables["v6_main_results"]}
    bins = tables["v6_baseline_bins"]
    fixed = bool(result.get("fixed_v6_plan"))
    lines = ["# V6: the frozen critic baseline", "",
             "**Question.** How does the warmed state-only critic change gradient variance relative to LOO along the fixed DPPO and without_PPO paths?", "",
             "**Answer.** " + ("The prescribed mean-direction consistency diagnostic failed; further collection must stop for review. " if result["consistency_failure"] else
                               "No resolved mean-direction inconsistency was detected at this scheduled analysis. ") +
             ("All five policies have at least 500 V6 batches." if result["minimum_batches_all_points_met"] else
              "V6 coverage is incomplete; absent policies and unresolved comparisons remain unavailable."), "",
             "| Policy | V6 / all-V4 batches | Critic/LOO variance [90% interval] | V6 noise scale, episodes | Paired direction cosine |", "|---|---:|---:|---:|---:|"]
    for point in POINTS:
        row, ratio = main.get(point, {}), ratios.get(point, {})
        count = result["counts"][point]
        ratio_text = "%s [%s,%s]" % (_fmt(ratio.get("ratio")), _fmt(ratio.get("ratio_low")), _fmt(ratio.get("ratio_high")))
        scale = ("≥" + _fmt(row["V6_B_ep_lower_bound"])) if row.get("V6_B_ep_type") == "lower_bound" else _fmt(row.get("V6_B_ep"))
        lines.append("| %s | %d / %d | %s | %s | %s |" % (point, count["K_V6"], count["K_all"], ratio_text, scale, _fmt(row.get("V6_V4_paired_cosine"))))
    lines.extend(["", "**Q1—critic versus LOO.** A variance ratio above 1 favors LOO; an interval crossing 1 is unresolved. " +
                  "; ".join(p + ": " + ("critic higher" if r.get("ratio_low") is not None and r["ratio_low"] > 1 else
                  "critic lower" if r.get("ratio_high") is not None and r["ratio_high"] < 1 else "unresolved") for p, r in ratios.items()) + ".", "",
                  "**Q2—decision bins.** The weighted residual-energy ratios are descriptive baseline-quality diagnostics, not measured gradient-variance ratios. " +
                  "; ".join(p + ": " + ", ".join("%d–%d:%s" % (r["k_start"], r["k_end_exclusive"] - 1, _fmt(r.get("critic_over_loo_energy")))
                           for r in bins if r["point"] == p) for p in POINTS if p in main) +
                  ". [Bin table](v6_baseline_bins.csv) includes raw signed critic residual means and their uncertainty; mean residual is baseline error, not policy-gradient bias.", "",
                  "**Q3—middle versus final.** " + "; ".join(label + " critic/LOO variance: " +
                           _ratio_interval(ratios.get(a, {})) + " → " + _ratio_interval(ratios.get(b, {}))
                           for label, a, b in (("DPPO", "D1", "D2"), ("without_PPO", "N1", "N2"))) +
                  ". These are fixed seed-0 policy/critic snapshots; no independent training-seed uncertainty is measured.", "",
                  "**Units and uncertainty.** V6 trace and own squared signal use paper units (code gradients × 2500). Noise scale divides the V6 episode-equivalent trace by V4's squared signal using all batches. Its jackknife sums independent legacy/new-cohort variance contributions, deleting new groups jointly from numerator and denominator. Ratios and cosines use only paired new groups, exclude same-batch cross products for mean inner products, and delete each 50-batch group jointly. Bounds for unresolved all-V4 signal are plug-in lower bounds, not joint confidence limits.", "",
                  "**Consistency boundary.** Stop only when both paired signal intervals are positive and a valid 90% cosine interval lies below 1. N/A does not fail; estimates are never clipped. This repeated diagnostic is not a globally calibrated hypothesis test. The compute-driven plan adds exactly 500 V6 batches at each point, then stops; any unresolved directions remain unresolved. Earlier V4 closure records are preserved as historical evidence.", "",
                  "[Full estimates](v6_main_results.csv) · [Paired variance ratios](v6_variance_ratios.csv) · [Consistency](v6_consistency.csv) · [Reproducible statistics](v6_results.json). Candidates only; no automatic curation."])
    path.write_text("\n".join(lines) + "\n")
