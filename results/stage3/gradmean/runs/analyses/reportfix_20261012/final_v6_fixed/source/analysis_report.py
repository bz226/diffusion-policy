"""Format the approved full Addendum G report from already computed tables.

No measurements, statistical estimators, checkpoint loads, or scheduler calls.
Read-only run metadata aggregation supplies provenance and timing inventories.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shlex


POINTS = ("P0", "D1", "D2", "N1", "N2")


def number(value):
    if value is None:
        return "N/A"
    if not math.isfinite(float(value)):
        raise ValueError("A report number must be finite or explicitly unavailable")
    return format(float(value), ".4g")


def ci(row, key):
    value = row.get(key)
    if value is None:
        return "N/A"
    lo, hi = row.get(key + "_low"), row.get(key + "_high")
    if lo is None or hi is None:
        return number(value) + " (CI N/A)"
    return "%s [%s, %s]" % (number(value), number(lo), number(hi))


def noise(row):
    if row.get("B_ep_type") == "lower_bound":
        return "≥ " + number(row.get("B_ep_lower_bound")) + " (plug-in)"
    return ci(row, "B_ep")


def batch_cos(row):
    if row.get("expected_batch_cos_upper_bound") is not None:
        return "≤ " + number(row["expected_batch_cos_upper_bound"]) + " (plug-in)"
    return ci(row, "expected_batch_cos")


def table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def metadata_inventory(root, campaign):
    """Copy selected existing metadata; never query a live scheduler here."""
    gates = []
    for path in sorted((root / "runs").glob("gate*/gate_summary.json")):
        record = read(path)
        gates.append({"path": str(path.resolve()), "record": record,
                      "scheduler_completion": read(path.parent / "scheduler_completion.json")})
    jobs = []
    for entry in campaign.get("submitted", []) + campaign.get("external", []):
        run_id = entry.get("run_id", "external_" + str(entry.get("job_id", "unknown")))
        manifest_path = root / "runs" / run_id / "manifest.json"
        manifest = read(manifest_path)
        logdir = Path(manifest["logdir"]) if manifest.get("logdir") else None
        native_path = (logdir / ("grad_mean.json" if manifest.get("step") == "prepare" else "meta.json")) if logdir else None
        native = read(native_path) if native_path else {}
        batches = native.get("completed_batches", native.get("completed_warmup_batches"))
        native_seconds = native.get("elapsed_seconds")
        jobs.append({"run_id": run_id, "job_id": entry.get("job_id"), "point": manifest.get("point"),
                     "step": manifest.get("step", "gate"), "seed": manifest.get("seed"),
                     "purpose": entry.get("purpose"),
                     "status": entry.get("run_status", manifest.get("status")),
                     "scheduler_state": entry.get("scheduler_state"), "exit_code": entry.get("exit_code"),
                     "worker_wall_seconds": manifest.get("elapsed_seconds"),
                     "native_wall_seconds": native.get("elapsed_seconds"),
                     "allocation_wall_seconds": entry.get("elapsed_seconds"),
                     "allocated_gpu_seconds": entry.get("allocated_gpu_seconds"),
                     "completed_batches": batches,
                     "native_seconds_per_completed_batch": native_seconds / batches
                         if native_seconds is not None and batches else None,
                     "parent_commit": manifest.get("parent_commit"),
                     "manifest": str(manifest_path) if manifest_path.is_file() else None,
                     "native_metadata": str(native_path) if native_path and native_path.is_file() else None,
                     "command": manifest.get("command"), "submission_command": entry.get("command"),
                     "failure_reason": manifest.get("terminal_failure_reason", manifest.get("failure_reason")),
                     "started_utc": manifest.get("started_utc"), "ended_utc": manifest.get("ended_utc"),
                     "environment": manifest.get("environment"), "gpu_inventory": manifest.get("gpu_inventory"),
                     "source_provenance": manifest.get("source_provenance", entry.get("source_provenance"))})
    return {"gates": gates, "jobs": jobs, "campaign_ledger": campaign.get("ledger", {}),
            "campaign_status": campaign.get("status"), "campaign_created_utc": campaign.get("created_utc"),
            "campaign_updated_utc": campaign.get("updated_utc"),
            "revisions": read(root / "provenance/revisions.json"),
            "checkpoints": read(root / "provenance/checkpoints.json"),
            "amendment": read(root / "provenance/amendment_G1.json")}


def build_report(destination, root, candidates, payload, figures, checkpoint_meta, campaign, status, inventory):
    """Return Markdown with links resolved relative to its actual destination."""
    def link(label, path):
        path = Path(path).resolve()
        if not path.exists():
            return label + " (record unavailable)"
        target = os.path.relpath(str(path), str(destination.parent))
        if " " in target:
            target = "<" + target + ">"
        return "[%s](%s)" % (label, target)
    def source(name):
        return link(name.replace("_", " "), candidates / (name + ".csv"))
    def lookup(rows, **keys):
        return next((row for row in rows if all(row.get(k) == v for k, v in keys.items())), {})
    tables = payload["tables"]
    mains = {(row["point"], row["variant"]): row for row in payload["main"]}
    v4 = {point: mains.get((point, "V4"), {}) for point in POINTS}
    per_point = {row["point"]: row for row in tables["per_point"]}
    measured = [p for p in POINTS if v4[p]]
    unresolved = [p for p in measured if v4[p].get("B_ep_type") != "point"]
    reliable = [p for p in measured if v4[p].get("rho") is not None and v4[p]["rho"] >= .68]
    noise_larger = [p for p in measured if v4[p].get("B_ep_type") == "point" and v4[p].get("B_ep", 0) > 80]
    fixed_plan = campaign.get("fixed_v6_plan")
    lines = ["# Addendum G: fixed-policy gradient signal and noise", "",
             "**Question.** At five pretrained, DPPO and NC4 checkpoints, how large and reliable is the mean discounted-return gradient, and how does it relate to the observed policy displacements?", "",
             "**Answer.** This %s analysis contains verified measurements at %d of five checkpoints; %d meet the prescribed V4 split-correlation threshold of 0.68. Resolved V4 point estimates have noise scale above the 80-episode training batch at %d of %d measured checkpoints. %s" % (
                 payload["scope"], len(measured), len(reliable), len(noise_larger), len(measured),
                 "The V4 signal/noise ratio remains unresolved at " + ", ".join(unresolved) + "." if unresolved else
                 "No measured V4 signal/noise ratio is currently classified as unresolved." if measured else "No checkpoint has complete measurement groups yet."), "",
             ("**Stopping plan.** Before any V6 collection, the user replaced adaptive continuation with a compute-driven fixed plan: all already admitted legacy jobs finish, then exactly ten new 50-batch six-variant groups are collected at each of five policies, followed by a stop. Thus V6 has 500 batches per policy; V1–V5 also retain every verified legacy group. Reaching the original 0.68 reliability threshold is descriptive, not required for completion. " + link("V6 analysis", candidates / "V6_NOTE.md") if fixed_plan else
              "**Stopping plan.** The original round-boundary reliability/cap rule applies."), "",
             "**Main limitation.** These are measurements at fixed checkpoints from one training seed per path. They do not show that gradient noise caused a training outcome. Unresolved signals, missing data and failed groups remain visible; a sample-cap closure does not establish a reliable direction. " +
             ("This interim report cannot support a final five-checkpoint conclusion." if payload["scope"] == "partial" else "The final scope is limited to the approved checkpoints and sample counts."), "",
             "Numbers in square brackets are 90% delete-one-group normal intervals (±1.645 SE), unless stated otherwise. One independent group contains 50 whole batches of 80 episodes. V4/V5 use paper ascent-gradient units; V1–V3 publish noise scales and cosines only. A ≥ or ≤ plug-in bound uses the estimated trace and the upper signal interval endpoint and has **no joint confidence guarantee**. N/A is retained when a statistic or its uncertainty is undefined.", "",
             "## Q1. Is the gradient noise-dominated at 80 episodes?", "",
             table(["Point", "Batches K", "V4 B (episodes)", "Expected 80-episode batch cosine"],
                   [[p, v4[p].get("K", "N/A"), noise(v4[p]), batch_cos(v4[p])] for p in POINTS]), "",
             "B above 80 means the estimated gradient-noise trace exceeds squared signal at this batch size; it is a descriptive scale comparison, not a hypothesis test. The expected cosine is the prescribed moment approximation, not a directly observed cosine. " + source("main_results") + " includes all five estimators and their uncertainty.", "",
             "## Q2. How do signal, noise and performance change along each path?", "",
             table(["Point / path position", "Discounted J", "V4 gradient norm", "V4 episode trace", "V4 B (episodes)"],
                   [[p, ci(per_point.get(p, {}), "J_disc"), ci(v4[p], "u_norm"), ci(v4[p], "trace_ep"), noise(v4[p])] for p in POINTS]), "",
             "Read the DPPO path as P0 → D1 → D2 and the NC4 path as P0 → N1 → N2. Unresolved norms and interval width limit descriptive ordering; marginal interval overlap is not used as a significance test. A higher return at these points does not by itself establish a general relation between policy quality and gradient norm.", "",
             "## Q3. How do DPPO and NC4 compare at matched iterations?", "",
             table(["Saved iteration", "DPPO / NC4", "Discounted J", "V4 gradient norm", "V4 B (episodes)"],
                   [[itr, p, ci(per_point.get(p, {}), "J_disc"), ci(v4[p], "u_norm"), noise(v4[p])]
                    for itr, pair in ((70, ("D1", "N1")), (139, ("D2", "N2"))) for p in pair]), "",
             "The paired rows match saved iteration and training seed 0, not independently replicated training paths. No additional significance test or derived DPPO/NC4 ratio is introduced.", "",
             "## Q4. How much does the baseline choice change variance or noise scale?", "",
             table(["Point", "No baseline / LOO trace (V5/V4)", "Critic / LOO B with MC (V2/V3)"],
                   [[p, ci(lookup(tables["contrasts"], point=p, contrast="V5_trace_over_V4_trace"), "ratio"),
                     ci(lookup(tables["contrasts"], point=p, contrast="V2_B_ep_over_V3_B_ep"), "ratio")] for p in POINTS]), "",
             "A V5/V4 ratio above one gives the measured multiplicative variance reduction from LOO. V2/V3 compares noise scales, not pure variances, because their normalized estimators can have different mean signals. These contrasts delete the same groups jointly across variants; an unresolved denominator remains N/A. " + source("contrasts") + ".", "",
             "## Q5. Are the DPPO-style and E1-style measurement estimators aligned with the discounted-return gradient?", "",
             table(["Point", "cos(V1, V4)", "cos(V3, V4)"],
                   [[p, ci(lookup(tables["same_point_cosines"], point=p, from_variant="V1", to_variant="V4"), "cosine"),
                     ci(lookup(tables["same_point_cosines"], point=p, from_variant="V3", to_variant="V4"), "cosine")] for p in POINTS]), "",
             "V1 is the unclamped NC4/DPPO-style measurement proxy; V3 is the E1-style MC/LOO measurement variant. Both use full-batch advantage normalization, whereas training normalizes per chunk and original DPPO also retains coordinate log-probability clamps, so these are not exact training-update gradients. V4 estimates the raw discounted-return gradient. Same-batch products are excluded before forming the cosines; estimates outside [−1, 1] remain visible as estimation error, and N/A does not imply orthogonality. " + source("same_point_cosines") + ".", "",
             "## Q6. How much does the discounted-return gradient rotate?", "",
             table(["Checkpoint pair", "V4 mean-gradient cosine"],
                   [[a + " / " + b, ci(lookup(tables["cross_point_cosines"], variant="V4", from_point=a, to_point=b), "cosine")]
                    for i, a in enumerate(POINTS) for b in POINTS[i + 1:]]), "",
             "A lower cosine indicates a larger directional change only where signals and intervals are resolved. Different policies use independent measurement groups; this is not a paired-rollout comparison. The complete V1 and V4 matrices are in " + source("cross_point_cosines") + ".", "",
             "## Q7. Do checkpoint displacements follow the starting gradient?", "",
             table(["Segment", "Distance", "Start cosine", "First-order ΔJ", "Trapezoid ΔJ", "Measured ΔJ"],
                   [[row["start"] + " → " + row["end"], number(row.get("displacement_norm")),
                     ci(row, "start_cos_displacement"), ci(row, "start_dot_displacement"),
                     ci(row, "trapezoid_delta_J"), ci(row, "measured_delta_J")] for row in tables["path_alignment"]]), "",
             "The first-order prediction is the starting V4 gradient dotted with the checkpoint displacement; the trapezoid averages the two endpoint gradients. Numerical agreement or disagreement is shown directly without a new residual test. Curvature between endpoints and the difference between a checkpoint chord and the actual training path limit this approximation. " + source("path_alignment") + ".", "",
             "## Q8. How loose is the empirical variance-lemma scale?", "",
             table(["Point", "Mean score norm²", "Max decision-decile mean", "Score norm² p99", "Lemma / V5 episode trace"],
                   [[p, number(lookup(tables["score_variance_lemma"], point=p).get("score_squared_norm_mean")),
                     number(lookup(tables["score_variance_lemma"], point=p).get("score_squared_norm_max_k_decile_mean")),
                     number(lookup(tables["score_variance_lemma"], point=p).get("score_squared_norm_p99")),
                     number(lookup(tables["score_variance_lemma"], point=p).get("lemma_over_observed_trace"))] for p in POINTS]), "",
             "The scale is observed r_max² × mean score norm² / (1 − 0.99)³. Its ratio to V5 trace reports empirical looseness, not a verified global bound: only 200 random decision scores from one group and an observed reward maximum are available. Group-jackknife uncertainty is N/A by the approved exception. Raw maxima, scale values and decile means are in " + source("score_variance_lemma") + ".", "",
             "## Q9. Does training drift agree with the direct P0 signal measurement?", "",
             "The direct P0 squared-gradient signal in paper units is **" + ci(v4["P0"], "u_sq") + "**.", "",
             table(["Existing condition", "J slope per actor update", "Squared-gradient drift proxy", "Proxy / P0 signal"],
                   [[row["condition"], ci(row, "J_disc_slope_per_update"), ci(row, "squared_gradient_drift_proxy"),
                     ci(row, "proxy_over_P0_signal")] for row in tables["drift_summary"]]), "",
             "Each seed uses OLS over all 126 pre-update training returns against completed actor updates 0–125; dividing by actor_lr/2500 forms the proxy. Intervals aggregate three independent training seeds, and the ratio combines that uncertainty with independent P0 groups. This is a path-average proxy, not a local θ0 gradient measurement; SGD10 is especially sensitive to its larger step. Agreement is descriptive and cannot validate a local drift identity. " + source("drift_per_seed") + " and " + source("drift_summary") + ".", "",
             "## Q10. Do the old Phase-1 estimates fall inside the new intervals?", ""]
    phase_rows = []
    for row in tables["phase1_comparison"]:
        if "metric" not in row:
            phase_rows.append([row["point"], "N/A", "N/A", "N/A", "New or reference data unavailable"])
            continue
        containment = row.get("new_interval_contains_old_point_descriptive_only")
        new = ("≥ " + number(row.get("new_B_ep_lower_bound")) + " (plug-in; no point CI)") if row.get("new_B_ep_type") == "lower_bound" else (
            "%s [%s, %s]" % (number(row.get("new_point")), number(row.get("new_ci90_low")), number(row.get("new_ci90_high"))))
        phase_rows.append([row["point"] + " / " + row["variant"], row["metric"], number(row.get("phase1_point")), new,
                           "Yes" if containment is True else "No" if containment is False else "N/A"])
    lines += [table(["Point / variant", "Metric", "Old point", "New estimate / 90% interval", "Old point inside?"], phase_rows), "",
              "Old intervals are not reused: within-batch dependence made their uncertainty unreliable, and critic/scaler preparation differs for V1/V2. Containment of one provisional point in a new interval is not a formal consistency test. Dimensioned comparisons are shown only for V4/V5 after the documented unit conversion. " + source("phase1_comparison") + ".", "",
              "## Measurement reliability and checkpoint inventory", "",
              table(["Point", "Groups / batches", "ρ", "Mean-direction c", "Moment c_U", "Effective rank", "Forecast batches"],
                    [[p, "%s / %s" % (v4[p].get("groups", "N/A"), v4[p].get("K", "N/A")), ci(v4[p], "rho"),
                      ci(v4[p], "c"), ci(v4[p], "c_U"), ci(v4[p], "effective_rank"),
                      "≥ " + number(v4[p].get("K_needed")) + " (plug-in)" if v4[p].get("K_needed_type") == "lower_bound"
                      else ci(v4[p], "K_needed")] for p in POINTS]), "",
              "ρ averages 20 balanced group half-splits; c describes the full measurement mean, unlike the 80-episode cosine in Q1. Effective rank uses the approved mean-free four-group difference statistic. " +
              ("The user replaced adaptive stopping with exactly 500 new V6 batches per point; forecasts and the original 0.68 threshold are descriptive and do not authorize more collection. " if fixed_plan else
               "Forecasts are descriptive: closure occurs only at approved round boundaries when ρ ≥ 0.68 or at 4,100 batches. ") +
              "Repeated convergence prefixes are dependent.", ""]
    mapping = inventory["checkpoints"]
    distances = checkpoint_meta.get("distance_from_P0", {})
    lines += [table(["Point", "Original actor checkpoint", "Distance from P0", "Frozen-preparation batches"],
                    [[p, link(Path(mapping[p]["checkpoint"]).name, mapping[p]["checkpoint"]) if p in mapping else "N/A",
                      number(distances.get(p)), mapping.get(p, {}).get("warmup_batches", "N/A")] for p in POINTS]), "",
              "Distances use the registered actor parameter order, float64 differences of saved float32 values, and no coordinate scaling. P0 is the released pretrained EMA policy; the four RL checkpoints are from training seed 0. " +
              link("Checkpoint mapping", root / "provenance/checkpoints.json") + " · " +
              link("Distance and parameter-order record", candidates.parent / "internal/checkpoint_metadata.json") + " · " +
              source("per_point") + " (discounted and undiscounted returns, SEs, saturation, out-of-range actions and explained variance).", ""]
    lines += gate_section(inventory, link, fixed_plan=fixed_plan)
    lines += provenance_section(root, candidates, payload, campaign, status, inventory, link)
    lines += ["## Figures and complete numerical evidence", "",
              "Each figure has an adjacent caption specifying units, uncertainty and limitations. The CSVs retain all approved per-point and per-variant estimates, bounds and missing-value reasons.", "",
              table(["Figure", "PNG", "PDF", "Caption"], [[f["id"], link("View", f["png"]), link("PDF", f["pdf"]),
                     link("Caption", f["caption"])] for f in figures]), "",
              " · ".join(source(name) for name in tables), "",
              link("Table definitions", candidates / "table_captions.md") + " · " + link("Machine-readable results", candidates / "results.json") +
              " · " + link("Methods", root / "methods.md") + " · " + link("Analysis execution record", candidates.parent / "analysis_status.json"), "",
              "**Decision and boundary.** Review these uncurated candidates for keep, revise, or leave uncurated. No new trajectories, training updates, extrapolated stopping target or additional scientific analysis is authorized by this report. Earlier failures and excluded records remain in the run inventory."]
    return "\n".join(lines) + "\n"


def gate_section(inventory, link, fixed_plan=False):
    records = inventory["gates"]
    offline = next((r for r in reversed(records) if r["record"].get("stage") == "offline"), None)
    live = next((r for r in reversed(records) if r["record"].get("stage") == "full"), None)
    synthetic = next((r for r in records if r["record"].get("gate") == "G2-G1"), None)
    original = next((r for r in records if Path(r["path"]).parent.name == "gate_G2"), None)
    lines = ["## Verification gates and retained failures", ""]
    if offline:
        record = offline["record"]
        g1 = record.get("G1", {})
        lines += ["G1 fixed-batch gradient comparison: **%s**. %s." % (
            "passed" if g1.get("passed") else "not passed", link("Offline gate record", offline["path"])), "",
            table(["Variant / check", "Maximum absolute difference", "Relative L2 difference"],
                  [[row["variant"], number(row.get("max_abs")), number(row.get("relative_l2"))] for row in g1.get("per_variant", [])] +
                  [["V4 vs negative E2 loss gradient", number(g1.get("V4_vs_negative_E2_loss_gradient", {}).get("max_abs")),
                    number(g1.get("V4_vs_negative_E2_loss_gradient", {}).get("relative_l2"))]]), ""]
        replay = []
        for name, result in record.get("G5", {}).items():
            comparisons = list(result.get("comparisons", {}).values())
            replay.append([name, "Passed" if result.get("passed") else "Not passed",
                           all(row.get("bitwise_equal") is True for row in comparisons) if comparisons else "N/A",
                           number(max((row.get("max_abs", 0) for row in comparisons), default=None)),
                           number(max((row.get("max_rel", 0) for row in comparisons), default=None))])
        lines += ["G5 default/NC4 fixed-batch replay compares losses, actor/critic gradients, updated parameters, optimizer states, sampling and RNG states. The table summarizes existing recorded comparisons without rerunning them.", "",
                  table(["Replay", "Status", "All comparisons bitwise equal", "Max abs", "Max relative"], replay), ""]
    else:
        lines += ["G1/G5 offline verification record is unavailable in this analysis snapshot; no pass is inferred.", ""]
    if synthetic:
        record = synthetic["record"]
        rows = []
        for name, regime in record.get("regimes", {}).items():
            checks = regime.get("checks", {})
            bias = checks.get("signal_sq_mean_bias", {})
            trace = checks.get("covariance_trace_mean_bias", {})
            rank = checks.get("mean_effective_rank", {})
            rows.append([name, "Passed" if regime.get("passed") else "Not passed",
                         "%s / %s; SE %s" % (number(bias.get("mean")), number(bias.get("truth")), number(bias.get("mc_se"))),
                         "%s / %s; SE %s" % (number(trace.get("mean")), number(trace.get("truth")), number(trace.get("mc_se"))),
                         number(checks.get("signal_sq_90_percent_coverage", {}).get("coverage_of_all_replicates")),
                         number(checks.get("covariance_trace_90_percent_coverage", {}).get("coverage_of_all_replicates")),
                         "%s / %s" % (number(rank.get("mean")), number(rank.get("truth")))])
        lines += ["Amended G2 uses 200 predeclared replicas in each regime, P = 20,000, K = 2,000, groups of 50, seed 8675309 and the same generator streams. Formal interval-coverage criteria apply to signal and trace only. " + link("G2-G1 record", synthetic["path"]) + ".", "",
                  table(["Regime", "Status", "Signal mean / truth; MC SE", "Trace mean / truth; MC SE", "Signal coverage", "Trace coverage", "Rank mean / truth"], rows), ""]
        low = record.get("regimes", {}).get("L", {}).get("direction_diagnostics", {})
        target = record.get("regimes", {}).get("T", {}).get("checks", {})
        shutdown = record.get("executor_shutdown", {})
        lines += ["Low-signal L has %s/%s nonpositive split correlations; its conditional signed cosine error is %s over %s defined replicas and is descriptive. Target T has mean absolute cosine error %s, with %s/%s replicas within absolute error 0.06 (required fraction ≥ 0.95). Pool shutdown success: %s; surviving child PIDs: %s; expected process exit: %s." % (
            low.get("nonpositive_rho_count", "N/A"), low.get("replicates", "N/A"), number(low.get("conditional_mean_signed_error")),
            low.get("conditional_error_denominator", "N/A"), number(target.get("mean_absolute_cosine_error", {}).get("mean_absolute_error")),
            target.get("cosine_absolute_error_tail", {}).get("within_threshold_count", "N/A"),
            target.get("cosine_absolute_error_tail", {}).get("denominator", "N/A"),
            shutdown.get("shutdown_success", "N/A"), shutdown.get("active_child_pids", "N/A"), record.get("expected_exit_code", "N/A")), ""]
        scheduler = synthetic.get("scheduler_completion", {})
        lines += ["G2 scheduler job %s: state %s, exit %s, elapsed %s seconds, %s GPUs. %s." % (
            scheduler.get("job_id", "N/A"), scheduler.get("state", "N/A"), scheduler.get("exit_code", "N/A"),
            number(scheduler.get("elapsed_seconds")), scheduler.get("gpu_count", "N/A"),
            link("Terminal scheduler record", Path(synthetic["path"]).parent / "scheduler_completion.json")), ""]
    else:
        lines += ["The amended G2 record is unavailable in this snapshot; no synthetic-gate pass is inferred.", ""]
    if live:
        record = live["record"]
        g3, g4 = record.get("G3", {}), record.get("G4", {})
        lines += ["G3 independent collection/cleanup: %s; seeds %s; different initial observations %s and returns %s; process exit codes %s; surviving children %s. G4 frozen actor: %s. %s." % (
            "passed" if g3.get("passed") else "not passed", g3.get("seeds", "N/A"), g3.get("first_observations_differ", "N/A"),
            g3.get("first_returns_differ", "N/A"), g3.get("process_exit_codes", "N/A"), g3.get("surviving_children", "N/A"),
            "passed" if g4.get("passed") else "not passed", link("Live gate record", live["path"])), ""]
        pilot = record.get("pilot_review", {})
        if pilot:
            lines += ["Automatic pilot timing review: **%s**. %s. %s" % (
                pilot.get("status", "unavailable"),
                link("Pilot forecast report", pilot["report"]) if pilot.get("report") else "Report path unavailable",
                "Reporting error: " + pilot["error"] if pilot.get("error") else
                "The historical pilot forecast preceded the user’s fixed V6 compute stop." if fixed_plan else
                "The forecast is descriptive; the GPU-hour cap and admission policy remain unchanged."), ""]
    else:
        lines += ["G3/G4 live verification is not present in this snapshot; it is not marked passed.", ""]
    if original:
        lines += ["The original G2 **failed** because one of 200 low-signal replicas had undefined direction reliability. Its saved summary and subsequent Python 3.8 cleanup traceback are preserved; the user approved Amendment G-1 to treat this case as unresolved and add the target regime. This is a protocol amendment after a retained failure, not an unchanged-gate success. " + link("Original failed gate", original["path"]) + ".", ""]
    return lines


def provenance_section(root, candidates, payload, campaign, status, inventory, link):
    jobs = inventory["jobs"]
    ledger = inventory["campaign_ledger"]
    lines = ["## Execution, resources and deviations", "",
             "Analysis parent commit: `%s`. Training/collection source revisions are recorded separately per run; a later analysis commit does not replace the source snapshot used for collection. %s." % (
                 status.get("parent_commit", "unavailable"), link("Revision history", root / "provenance/revisions.json")), "",
             "Collection uses one A6000 GPU, 40 CPUs and 64 GiB per job, at most nine concurrent GPU jobs, and a 125 GPU-hour total cap. Each preparation or 50-batch collection job is capped at one hour. CPU analyses use four CPUs, 64 GiB and a 30-minute allocation; gate allocations are recorded separately.", "",
             "Recorded used GPU hours: %s; outstanding reserved GPU hours: %s; cap: %s. These are scheduler allocation totals, including failed allocations; parallel job times must not be added to estimate campaign wall-clock duration." % (
                 number(ledger["actual_gpu_seconds"] / 3600) if "actual_gpu_seconds" in ledger else "N/A",
                 number(ledger["outstanding_reserved_gpu_seconds"] / 3600) if "outstanding_reserved_gpu_seconds" in ledger else "N/A",
                 number(ledger["budget_gpu_seconds"] / 3600) if "budget_gpu_seconds" in ledger else "N/A"), "",
             "Campaign record: %s through %s, status `%s`. Analysis started %s; elapsed wall time through report generation %s seconds. %s." % (
                 inventory.get("campaign_created_utc", "unavailable"), inventory.get("campaign_updated_utc", "unavailable"),
                 inventory.get("campaign_status", "unavailable"), status.get("started_utc", "unavailable"),
                 number(status.get("elapsed_seconds_at_report")), link("Campaign ledger", root / "runs/campaign.json")), ""]
    timing = []
    for point in POINTS:
        measured = [row for row in jobs if row.get("point") == point and row.get("step") == "collect"]
        walls = [row["worker_wall_seconds"] for row in measured if row.get("worker_wall_seconds") is not None]
        timing.append([point, len(measured), sum(row.get("status") == "complete" for row in measured),
                       number(min(walls)) if walls else "N/A", number(max(walls)) if walls else "N/A",
                       number(sum(walls)) if walls else "N/A"])
    lines += [table(["Point", "Collection attempts", "Complete workers", "Min job seconds", "Max job seconds", "Sum job seconds"], timing), "",
              "Times above are observed worker wall times, not uncertainty estimates; the per-run inventory preserves scheduler allocation time, native computation time, job IDs, failures, source revisions and exact commands. " +
              link("Run and timing inventory", candidates / "run_inventory.csv") + " · " + link("Metadata snapshot", candidates / "report_provenance.json") + ".", "",
              table(["Preparation / gate", "Job", "Recorded state", "Allocation seconds", "Allocated GPU seconds"],
                    [[row.get("purpose") or row["run_id"], row.get("job_id", "N/A"), row.get("scheduler_state", "N/A"),
                      number(row.get("allocation_wall_seconds")), number(row.get("allocated_gpu_seconds"))]
                     for row in jobs if row.get("step") != "collect"]), "",
              "Preparation timing is kept separate from 50-batch measurement timing. Gate allocation times include process startup and verification overhead; they are not training iteration times.", "",
              "Original collection seeds follow 100000 + 10000 × point_index + 100 × group_index; authorized replacement seeds start at 1000000. Preparation uses group_index 90. Point indices are P0 = 0, D1 = 1, D2 = 2, N1 = 3, N2 = 4; each run seeds its 40 environments with seed + 0…39. Exact seeds and native commands are in the inventory; no global RNG is used for the diagnostic splits.", "",
              "Analysis command: `" + shlex.join([str(arg) for arg in status.get("command", [])]) + "`. Python: `" + str(status.get("python", "unavailable")).splitlines()[0] + "`; NumPy: `" + str(status.get("numpy", "unavailable")) + "`. The metadata snapshot preserves existing worker environment and GPU inventories.", ""]
    excluded = payload.get("failed_or_incomplete_groups", [])
    lines += ["**Exclusions and deviations.** %d group records were excluded from the verified measurement set. Native, worker, cleanup and terminal scheduler success are normally required. The explicit administrative recovery admits seven fully completed measurements whose wrappers failed only at the obsolete branch-name check; their original FAILED scheduler states, failed manifests, zero native exit codes and successful cleanup records remain preserved. Other failed jobs remain excluded. At most one approved independent-seed replacement is selected for a group; original failures remain stored." % len(excluded), ""]
    if excluded:
        lines += [table(["Point / group", "Recorded state", "Exclusion reason", "Native record"],
                        [["%s / %s%s" % (row["point"], row["group"], " retry" if row.get("retry") else ""), row.get("status"),
                          "; ".join(row.get("exclusion_reasons", [])), link("Metadata", Path(row["path"]) / "meta.json")]
                         for row in excluded]), ""]
    lines += ["Approved clarifications include the mean-free four-group effective-rank denominator; counting both P0 live-gate groups within round 1; public V1–V3 noise scales/cosines only; score uncertainty unavailable from one group; and the all-126-update drift proxy. Amendment G-1 additionally defines unresolved directions and upper-signal plug-in bounds, adds the target synthetic regime, and corrects process cleanup. " +
              link("Approval record", root / "provenance/approval.json") + " · " + link("Amendment G-1", root / "provenance/amendment_G1.json") + ".", ""]
    return lines


def write_reports(root, candidates, payload, figures, checkpoint_meta, campaign, status, final=False):
    """Create new candidate report; mirror only the authorized final report."""
    root, candidates = Path(root), Path(candidates)
    path = candidates / "REPORT.md"
    if path.exists():
        raise ValueError("Refusing to overwrite a candidate report")
    inventory = metadata_inventory(root, campaign)
    provenance = candidates / "report_provenance.json"
    provenance.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    write_csv(candidates / "run_inventory.csv", inventory["jobs"])
    path.write_text(build_report(path, root, candidates, payload, figures, checkpoint_meta, campaign, status, inventory))
    result = {"report": str(path.resolve()), "report_provenance": str(provenance.resolve())}
    if final:
        target = root / "REPORT.md"
        # The current reader-facing report is an approved derived document.
        # Preserve its prior body as evidence before replacing it, and leave
        # runtime status updates to the existing controller.
        if target.exists():
            (candidates.parent / "internal/report_before_final_analysis.md").write_bytes(target.read_bytes())
        temporary = root / ".REPORT.analysis.tmp"
        temporary.write_text(build_report(target, root, candidates, payload, figures, checkpoint_meta, campaign, status, inventory))
        os.replace(str(temporary), str(target))
        result["current_report"] = str(target.resolve())
    return result
