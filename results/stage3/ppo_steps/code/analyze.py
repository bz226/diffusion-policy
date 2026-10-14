#!/usr/bin/env python3
"""Read-only aggregation of DPPO per-step gradient measurements (Python 3.8).

Only completed 50-batch group outputs are read. This program never launches jobs,
updates a policy, modifies measurements, or changes existing reward figures.
An output directory must be empty unless --overwrite is explicitly supplied.
"""

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


EPISODES_PER_BATCH = 80
EXPECTED_GROUPS = 10
BATCHES_PER_GROUP = 50
PARAMETER_COUNT = 568392
SCALARS = ("clipfrac", "approx_kl", "switched_off_frac",
           "mean_abs_ratio_minus_one", "grad_norm", "cos_g1")
POINTS = ((35, 32), (70, 63), (105, 95))


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value) if value not in (None, "", "None") else float("nan")


def clean_json(value):
    """JSON uses null for non-finite numbers; CSV retains explicit inf/nan."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, sort_keys=True) + "\n")


def moments(total, squared_sum, count):
    """Pairwise squared-mean and ordinary sample covariance trace, float64."""
    if count < 2:
        return float("nan"), float("nan")
    squared_total = float(np.dot(total, total))
    signal = (squared_total - squared_sum) / (count * (count - 1.0))
    variance = (squared_sum - squared_total / count) / (count - 1.0)
    if variance < -1e-12 * max(squared_sum, 1e-300):
        raise ValueError("Negative covariance trace; group sums and squared norms disagree")
    return signal, max(0.0, variance)


def jackknife_se(replicates):
    values = np.asarray(replicates, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.sqrt((len(values) - 1.0) / len(values) *
                         np.sum((values - values.mean()) ** 2)))


def summarize_moments(group_sums, group_sq, group_counts):
    """Delete whole jobs jointly for numerator and denominator of B.

    Counts may differ only when KL stopping prevents a later PPO step. In that
    case the result describes gradients conditional on reaching that step.
    Every original group remains a delete-group unit, including zero-count ones.
    """
    group_sums = np.asarray(group_sums, dtype=np.float64)
    group_sq = np.asarray(group_sq, dtype=np.float64)
    counts = np.asarray(group_counts, dtype=np.int64)
    if group_sums.ndim != 2 or len(group_sums) != len(counts):
        raise ValueError("Invalid grouped moments")
    if not np.all(np.isfinite(group_sums)) or not np.all(np.isfinite(group_sq)):
        raise ValueError("Non-finite gradient sums or squared norms")
    total = np.sum(group_sums, axis=0, dtype=np.float64)
    squared_sum = math.fsum(group_sq.tolist())
    count = int(counts.sum())
    signal, variance = moments(total, squared_sum, count)
    leave = [moments(total - group_sums[g], squared_sum - group_sq[g],
                     count - int(counts[g])) for g in range(len(counts))]
    sig_reps = [item[0] for item in leave]
    tr_reps = [EPISODES_PER_BATCH * item[1] for item in leave]
    # Signed denominator replicas preserve the ordinary joint ratio jackknife.
    # A nonpositive squared-mean CI makes the reported B interval unbounded.
    ratio_reps = [EPISODES_PER_BATCH * var / sig if sig != 0 else float("nan")
                  for sig, var in leave]
    critical = float(student_t.ppf(0.95, len(counts) - 1)) if len(counts) > 1 else float("nan")
    trace = EPISODES_PER_BATCH * variance
    out = {"K": count, "groups": len(counts), "u_sq": signal,
           "u_norm": math.sqrt(signal) if signal > 0 else float("nan"),
           "trace_ep": trace,
           "B": trace / signal if signal > 0 else float("nan"),
           "t_critical": critical, "group_counts": counts.tolist()}
    for key, replicas in (("u_sq", sig_reps), ("trace_ep", tr_reps), ("B", ratio_reps)):
        se = jackknife_se(replicas)
        out[key + "_se"] = se
        out[key + "_lo"] = out[key] - critical * se
        out[key + "_hi"] = out[key] + critical * se
    # A monotone transform of the signed signal interval, not a jackknife of
    # sqrt applied to possibly negative leave-group-out squared signals.
    out["u_norm_lo"] = math.sqrt(max(0.0, out["u_sq_lo"])) if math.isfinite(out["u_sq_lo"]) else float("nan")
    out["u_norm_hi"] = math.sqrt(max(0.0, out["u_sq_hi"])) if math.isfinite(out["u_sq_hi"]) else float("nan")
    out["signal_resolved"] = bool(out["u_sq_lo"] > 0)
    out["B_lo"] = max(0.0, out["B_lo"]) if math.isfinite(out["B_lo"]) else 0.0
    if not out["signal_resolved"] or not math.isfinite(out["B_hi"]):
        out["B_hi"] = float("inf")
    out["B_status"] = "point" if out["signal_resolved"] else "unbounded_above"
    if signal <= 0:
        out["B_status"] = "nonpositive_signal"
    return out


def step_key(value):
    if str(value).lower() == "full":
        return "full"
    return int(value)


def discover_groups(root):
    groups, ignored = {}, []
    seen = set()
    for path in sorted(Path(root).rglob("meta.json")):
        meta = read_json(path)
        part = str(meta.get("part", "")).upper().replace("PART", "").strip()
        if part not in ("A", "C"):
            continue
        if str(meta.get("status", "")).lower() not in ("completed", "complete"):
            ignored.append({"path": str(path), "reason": "not completed"})
            continue
        itr = int(meta["checkpoint_itr"])
        updates = int(meta["updates"])
        if (itr, updates) not in POINTS:
            raise ValueError("Unexpected checkpoint/update mapping in %s" % path)
        if int(meta["K"]) != BATCHES_PER_GROUP:
            raise ValueError("A completed job must contain exactly 50 batches: %s" % path)
        if int(meta["parameter_count"]) != PARAMETER_COUNT:
            raise ValueError("Unexpected actor parameter count in %s" % path)
        if meta.get("max_grad_norm") is not None or meta.get("reset_at_iteration") is not False:
            raise ValueError("Unexpected clipping/reset convention in %s" % path)
        if meta.get("group_vector_kind") != "sum" or meta.get("stored_dtype") != "float64":
            raise ValueError("Group vector storage convention differs in %s" % path)
        identity = (part, itr, int(meta["seed"]))
        if identity in seen:
            raise ValueError("Duplicate seed/group at one measurement point: %s" % (identity,))
        seen.add(identity)
        rows = read_csv(path.parent / "batches.csv")
        by_step = {}
        keys = set()
        for row in rows:
            step = step_key(row["step"])
            batch = int(row["batch"])
            key = (batch, step)
            if key in keys:
                raise ValueError("Duplicate per-step batch record in %s" % path.parent)
            keys.add(key)
            by_step.setdefault(step, []).append(row)
        if len(by_step.get(1, [])) != BATCHES_PER_GROUP:
            raise ValueError("Every batch must reach PPO step 1: %s" % path.parent)
        if len(meta.get("batches", [])) != BATCHES_PER_GROUP:
            raise ValueError("Missing per-batch completion/early-stop metadata: %s" % path)
        if part == "A" and "full" not in by_step:
            # Alternative compact collector format: full_grad_sq accompanies step 1.
            if all("full_grad_sq" in r for r in by_step[1]):
                by_step["full"] = [dict(r, grad_sq=r["full_grad_sq"]) for r in by_step[1]]
            else:
                raise ValueError("Part A needs full-gradient squared norms: %s" % path.parent)
        group = {"path": path.parent, "meta": meta, "by_step": by_step}
        groups.setdefault((part, itr, updates), []).append(group)
    for key, items in groups.items():
        if len(items) > EXPECTED_GROUPS:
            raise ValueError("More than ten completed groups at %s; refusing silent subsetting" % (key,))
        specs = [item["meta"].get("parameter_specs") for item in items]
        if any(spec != specs[0] for spec in specs):
            raise ValueError("Actor parameter order differs between groups at %s" % (key,))
    return groups, ignored


def summarize_step(items, step):
    sums, squared, counts, flat_rows = [], [], [], []
    for item in items:
        rows = item["by_step"].get(step, [])
        counts.append(len(rows))
        flat_rows.extend(rows)
        squared.append(math.fsum(number(row["grad_sq"]) for row in rows))
        name = "sums_full.npy" if step == "full" else "sums_step%02d.npy" % step
        path = item["path"] / name
        if not rows:
            # Not an observed zero gradient: no gradients contribute to this sum.
            arr = np.zeros(PARAMETER_COUNT, dtype=np.float64)
            if path.exists() and np.any(np.load(str(path), allow_pickle=False)):
                raise ValueError("Nonzero gradient sum without any measured rows: %s" % path)
        else:
            arr = np.load(str(path), allow_pickle=False)
            if arr.shape != (PARAMETER_COUNT,) or arr.dtype != np.float64:
                raise ValueError("Group sums must be float64 actor vectors: %s" % path)
        sums.append(arr)
    if sum(counts) < 2:
        return None
    stats = summarize_moments(sums, squared, counts)
    for scalar in SCALARS:
        values = [number(row.get(scalar)) for row in flat_rows]
        finite = [value for value in values if math.isfinite(value)]
        if scalar != "cos_g1" and len(finite) != len(values) and step != "full":
            raise ValueError("Missing/non-finite %s at step %s" % (scalar, step))
        stats[scalar] = float(np.mean(finite)) if finite else float("nan")
        stats[scalar + "_finite_K"] = len(finite)
    cs = [number(row.get("c", item["meta"].get("c")))
          for item in items for row in item["by_step"].get(step, [])]
    finite_cs = [value for value in cs if math.isfinite(value)]
    stats["c"] = float(np.mean(finite_cs)) if finite_cs else float("nan")
    stats["c_min"] = min(finite_cs) if finite_cs else float("nan")
    stats["c_max"] = max(finite_cs) if finite_cs else float("nan")
    stats["c_source"] = "; ".join(sorted(set(str(row.get("c_source", item["meta"].get("c_source", "not recorded")))
                                          for item in items for row in item["by_step"].get(step, []))))
    return stats


def build_rows(groups):
    rows, missing = [], []
    for (part, itr, updates), items in sorted(groups.items()):
        full = summarize_step(items, "full") if part == "A" else None
        reached = {s: sum(len(item["by_step"].get(s, [])) for item in items)
                   for s in range(1, 21)}
        for step in ([1] if part == "A" else range(1, 21)):
            stats = summarize_step(items, step)
            if stats is None:
                missing.append({"part": part, "checkpoint_itr": itr, "step": step,
                                "reason": "fewer than two observed gradients"})
                continue
            row = {"method": "DPPO", "part": part, "checkpoint_itr": itr,
                   "updates": updates, "step": step, "scheduled_K": len(items) * 50,
                   "expected_K": 500, "status": "complete", "checkpoint_file": "state_%d.pt" % itr}
            row.update(stats)
            if len(items) < 10:
                row["status"] = "partial_collection"
            if stats["K"] < row["scheduled_K"]:
                row["status"] += "; conditional_on_reaching_step"
            row["early_stop_batches"] = sum(bool(batch["target_kl_fired"])
                                             for item in items for batch in item["meta"]["batches"]) if part == "C" else 0
            row["batches_not_reaching_step20"] = (reached[1] - reached[20]) if part == "C" else 0
            starts = [float(batch["c_start"]) for item in items for batch in item["meta"]["batches"]]
            row["c_start_min"] = min(starts)
            row["c_start_max"] = max(starts)
            row["critic_updates_skipped"] = all(item["meta"].get("critic_optimizer_steps_skipped", False) for item in items)
            if full is not None:
                for key in ("K", "u_norm", "u_sq", "trace_ep", "B", "B_lo", "B_hi", "B_status"):
                    row["full_" + key] = full[key]
            row["jobs"] = ";".join(item["path"].name for item in items)
            row["seeds"] = ";".join(str(item["meta"]["seed"]) for item in items)
            row["source_commits"] = ";".join(sorted(set(str(item["meta"].get("commit", item["meta"].get("sourcecommit", item["meta"].get("source_commit", "not recorded")))) for item in items)))
            rows.append(row)
    return rows, missing


def format_number(value):
    value = number(value)
    if math.isnan(value):
        return "N/A"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    return "%.5g" % value


def estimate(row, key):
    close = ")" if math.isinf(number(row.get(key + "_hi"))) else "]"
    return "%s [%s, %s%s" % (format_number(row.get(key)),
                             format_number(row.get(key + "_lo")),
                             format_number(row.get(key + "_hi")), close)


def write_table(rows, out):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and key != "group_counts":
                keys.append(key)
    with (out / "ppo_steps_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    lines = ["# DPPO per-step gradient noise", "",
             "All estimates use training-loss units. A group is one independent 50-batch job; "
             "each batch contains 80 episodes. Brackets are 90% delete-one-group jackknife intervals "
             "with Student t quantiles on groups − 1 degrees of freedom (t(0.95,9) = 1.833113 at ten groups). "
             "B uses a joint numerator/denominator jackknife. A squared-mean interval reaching zero gives "
             "an unbounded B interval. Norm intervals are the nonnegative square-root transform of the "
             "signed squared-mean interval, also reported below. Per-episode variance at later PPO steps "
             "is the requested 80× batch-variance convention.", "",
             "| Part | Updates (checkpoint) | Step | K / groups | Mean norm | Signed squared mean | Variance/episode | B (episodes) | B_full | clipfrac | Switched off | Mean abs ratio − 1 | approx_kl | cos(g_s,g_1) | c / source | Status |",
             "|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        full = estimate(row, "full_B") if row["part"] == "A" else "—"
        cells = [row["part"], "%d (%s)" % (row["updates"], row["checkpoint_file"]), str(row["step"]),
                 "%d / %d" % (row["K"], row["groups"]), estimate(row, "u_norm"), estimate(row, "u_sq"),
                 estimate(row, "trace_ep"), estimate(row, "B"), full]
        cells += [format_number(row[key]) for key in ("clipfrac", "switched_off_frac", "mean_abs_ratio_minus_one", "approx_kl", "cos_g1")]
        cells += ["%s / %s" % (format_number(row["c"]), row["c_source"]), row["status"]]
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in cells) + " |")
    lines.extend(["", "An omitted later step is not a zero gradient. Rows with K below the number of "
                  "collected batches describe gradients conditional on reaching that step; such rows "
                  "do not establish the unconditional 500-batch statistic. No intervals are drawn in figures.", ""])
    (out / "ppo_steps_table.md").write_text("\n".join(lines))


def equal_log_span(axes, columns):
    span = max(2.0, math.ceil(max(math.log10(max(v)) - math.log10(min(v))
                                  for v in columns) * 2.0 + 0.2) / 2.0)
    for ax, values in zip(axes, columns):
        center = (math.log10(min(values)) + math.log10(max(values))) / 2.0
        ax.set_ylim(10 ** (center - span / 2.0), 10 ** (center + span / 2.0))
        ax.set_yscale("log")
    return span


def render_figures(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figures = out / "figs"
    figures.mkdir(exist_ok=True)
    record = {"created": [], "omitted": [], "spans_decades": {}}
    keys = ("u_norm", "trace_ep", "B")
    titles = ("Mean gradient norm", "Gradient variance\n(per episode)", "Relative variance\n(per episode)")
    ylabels = ("Gradient norm (training-loss units)", "Variance (squared training-loss units)", "Relative variance = B (episodes)")
    captions = {
        "A": "DPPO's first gradient step on each batch (50,000 pairs at θ_old), in training-loss units. "
             "Points are estimates from one training seed; 90% intervals are in the table.",
        "C": "Each gradient DPPO passes to AdamW during one batch's 20 steps, from the run's own state at "
             "each checkpoint, in training-loss units. Points are estimates from one training seed; "
             "90% intervals are in the table. Per-episode variance is 80 times the batch-gradient variance; "
             "for steps after the first this is a scaling convention."
    }
    for part, stem in (("A", "dppo_step1_noise"), ("C", "dppo_per_step_noise")):
        selected = [row for row in rows if row["part"] == part]
        expected = 3 if part == "A" else 60
        invalid = [row for row in selected if any(not math.isfinite(row[k]) or row[k] <= 0 for k in keys)]
        partial = [row for row in selected if row["groups"] != 10 or row["K"] != 500]
        if len(selected) != expected or invalid or partial:
            record["omitted"].append({"figure": stem,
                                      "reason": "Requires all three points, 500 observed gradients per step, and positive finite point estimates",
                                      "available_rows": len(selected),
                                      "invalid": [(r["updates"], r["step"]) for r in invalid],
                                      "partial": [(r["updates"], r["step"], r["K"]) for r in partial]})
            continue
        # Give the three-entry legend its own row above the two-line titles.
        # Point values, axis spans and the Part-A layout are unchanged.
        fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.9 if part == "C" else 3.4))
        fig.subplots_adjust(left=0.075, right=0.985,
                            bottom=0.18 if part == "C" else 0.21,
                            top=0.72 if part == "C" else 0.78, wspace=0.52)
        for idx, (ax, key, title, label) in enumerate(zip(axes, keys, titles, ylabels)):
            if part == "A":
                ordered = sorted(selected, key=lambda r: r["updates"])
                ax.plot([r["updates"] for r in ordered], [r[key] for r in ordered],
                        "o-", color="#0072B2", markersize=4, linewidth=1.3, label="DPPO")
                ax.set_xticks([32, 63, 95])
                ax.set_xlabel("Updates (one update = 80 episodes)")
            else:
                for updates, color in zip((32, 63, 95), ("#7eb6d6", "#3185b6", "#004d7a")):
                    ordered = sorted([r for r in selected if r["updates"] == updates], key=lambda r: r["step"])
                    ax.plot([r["step"] for r in ordered], [r[key] for r in ordered],
                            "o-", color=color, markersize=3, linewidth=1.3,
                            label="DPPO, %d updates" % updates)
                ax.set_xticks([1, 4, 8, 12, 16, 20])
                ax.set_xlabel("PPO step within the batch")
            ax.set_title(title)
            ax.set_ylabel(label)
            ax.grid(axis="y", color="0.90", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        span = equal_log_span(axes, [[r[k] for r in selected] for k in keys])
        fig.suptitle("Each checkpoint: 500 batches of 80 episodes (10 groups)", y=0.98, fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=8)
        for suffix in ("png", "pdf"):
            fig.savefig(str(figures / (stem + "." + suffix)), dpi=300)
        plt.close(fig)
        (figures / (stem + ".caption.md")).write_text(captions[part] + "\n")
        record["created"].append(stem)
        record["spans_decades"][stem] = span
    crows = [r for r in rows if r["part"] == "C"]
    if len(crows) == 60 and all(r["groups"] == 10 and r["K"] == 500 for r in crows):
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for updates, color in zip((32, 63, 95), ("#7eb6d6", "#3185b6", "#004d7a")):
            ordered = sorted([r for r in crows if r["updates"] == updates], key=lambda r: r["step"])
            ax.plot([r["step"] for r in ordered], [r["switched_off_frac"] for r in ordered],
                    "o-", color=color, markersize=3, label="DPPO, %d updates" % updates)
        ax.set_xlabel("PPO step within the batch")
        ax.set_ylabel("Pairs switched off by the clip (fraction)")
        ax.set_ylim(bottom=0)
        ax.set_xticks([1, 4, 8, 12, 16, 20])
        ax.grid(axis="y", color="0.90", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(str(figures / ("dppo_per_step_inactive." + suffix)), dpi=300)
        plt.close(fig)
        (figures / "dppo_per_step_inactive.caption.md").write_text(
            "Fraction of DPPO minibatch pairs whose actor contribution is switched off by PPO clipping: "
            "positive adjusted advantage with ratio above 1 + ε, or negative adjusted advantage below 1 − ε. "
            "Each point averages 500 batches at one checkpoint of one training seed.\n")
        record["created"].append("dppo_per_step_inactive")
    return record


def rerun_sanity(rerun_path, originals, out):
    if rerun_path is None:
        return {"status": "not supplied"}
    def evaluation(path):
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)
        return [(int(r["itr"]), float(r["eval_episode_reward"])) for r in data if "eval_episode_reward" in r]
    rerun = evaluation(rerun_path)
    refs = [evaluation(path) for path in originals]
    final = [dict(items).get(130) for items in refs]
    rerun_final = dict(rerun).get(130)
    status = "pending final evaluation"
    if len(final) == 3 and all(v is not None for v in final) and rerun_final is not None:
        status = "red flag" if rerun_final < min(final) - 150 or rerun_final > max(final) + 150 else "within requested tolerance"
    rows = [{"itr": itr, "DPPO_rerun": val} for itr, val in rerun]
    for row in rows:
        for idx, curve in enumerate(refs):
            row["DPPO_original_seed%d" % idx] = dict(curve).get(row["itr"], "")
    if rows:
        with (out / "rerun_evaluation_curve.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for idx, curve in enumerate(refs):
            ax.plot([itr * 0.072 for itr, unused in curve], [value for unused, value in curve],
                    linewidth=0.9, alpha=0.5, label="DPPO original seed %d" % idx)
        ax.plot([itr * 0.072 for itr, unused in rerun], [value for unused, value in rerun],
                "o-", color="#0072B2", markersize=3, label="DPPO rerun seed 0")
        ax.set_xlabel("Training env steps (millions)")
        ax.set_ylabel("Evaluation reward (episode return)")
        ax.grid(axis="y", color="0.90", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(str(out / "figs" / ("dppo_rerun_sanity." + suffix)), dpi=300)
        plt.close(fig)
        (out / "figs" / "dppo_rerun_sanity.caption.md").write_text(
            "DPPO rerun seed 0 with full-state saving, compared with the three original DPPO seeds. "
            "Each point is an 80-episode evaluation mean. A final evaluation more than 150 reward "
            "units outside the original seeds' range is flagged; this is a sanity check, not an exact-replay gate.\n")
    return {"status": status, "final_rerun": rerun_final, "final_originals": final,
            "rerun_result": str(rerun_path), "original_results": [str(p) for p in originals]}


def write_note(rows, groups, figures, args, out, sanity, ignored, missing):
    ref = {}
    if args.reference_table and Path(args.reference_table).exists():
        ref = {int(row["update_number"]): row for row in read_csv(args.reference_table)}
    lines = ["# DPPO per-step gradient noise", "",
             "This measurement asks how much noise DPPO's AdamW receives at each minibatch step. "
             "The table and figures use the gradient of the actual PPO actor loss in training-loss units; "
             "later steps depend on the saved optimizer moments and within-batch updates.", "",
             "Checkpoint mapping recorded in the collection metadata: state_35 → 32 updates, "
             "state_70 → 63, state_105 → 95. Part A uses the original checkpoint critic and θ_old; "
             "Part C restores the rerun's full state before each batch. "
             "No gradient-norm clipping is configured, and reset_at_iteration is false. "
             "The sample is one training seed per path, not independent training seeds.", "",
             "| Updates | Part A B₁ | Part A B_full | Part C B₁ | without_PPO_and_GAE as-run B |",
             "|---:|---:|---:|---:|---:|"]
    for unused_itr, updates in POINTS:
        a = next((r for r in rows if r["part"] == "A" and r["updates"] == updates), {})
        c = next((r for r in rows if r["part"] == "C" and r["updates"] == updates and r["step"] == 1), {})
        reference = ref.get(updates, {})
        lines.append("| %d | %s | %s | %s | %s |" %
                     (updates, estimate(a, "B"), estimate(a, "full_B"), estimate(c, "B"),
                      estimate(reference, "as_run_B_ep")))
    lines.extend(["", "B is in episodes; one 80-episode update has relative variance B/80. "
                  "The reference uses Monte Carlo-minus-critic advantages and a full-batch actor step, "
                  "so its difference from DPPO is not solely a minibatch-size effect. "
                  "Within Part A, B₁ versus B_full isolates first-minibatch sampling at the same θ_old.", "",
                  "| Part | Updates | Groups / batches | c (range if changing) | Source | B interval reaches zero signal? |",
                  "|---|---:|---|---|---|---|"])
    for (part, itr, updates), items in sorted(groups.items()):
        subset = [r for r in rows if r["part"] == part and r["updates"] == updates]
        first = next((r for r in subset if r["step"] == 1), {})
        unresolved = ", ".join(str(r["step"]) for r in subset if not r["signal_resolved"]) or "none"
        lines.append("| %s | %d | %d / %d | %s [%s, %s] | %s | steps: %s |" %
                     (part, updates, len(items), len(items) * 50, format_number(first.get("c")),
                      format_number(first.get("c_min")), format_number(first.get("c_max")),
                      first.get("c_source", "not recorded"), unresolved))
    stop_counts = [(r["updates"], r["early_stop_batches"]) for r in rows if r["part"] == "C" and r["step"] == 1]
    lines.extend(["", "KL early-stop batches by checkpoint: %s. Missing later steps are never filled with zeros." %
                  (", ".join("%d updates: %d" % item for item in stop_counts) or "Part C pending"),
                  "Part C skips critic optimizer updates; actor advantages were already computed before the update loop. "
                  "The tabulated c is the batch's used divisor (with its observed range); restored starting divisors "
                  "are retained separately in the CSV.",
                  "", "R3 rerun sanity: %s; final rerun = %s; original final evaluations = %s." %
                  (sanity["status"], format_number(sanity.get("final_rerun")),
                   ", ".join(format_number(v) for v in sanity.get("final_originals", [])) or "not supplied")])
    if args.gate_record:
        lines.extend(["", "Gate records (the analysis does not rerun or infer gates):"])
        for path in args.gate_record:
            lines.append("- [%s](%s)" % (Path(path).name, os.path.relpath(str(Path(path).resolve()), str(out))))
    else:
        lines.extend(["", "Critic equality, R1/R2 and S1 gate evidence was not supplied to this analysis; "
                      "do not treat aggregation success as a passed scientific gate."])
    if figures["omitted"]:
        lines.extend(["", "Figures not yet drawable:"])
        for item in figures["omitted"]:
            lines.append("- %s: %s; invalid points %s; partial points %s." %
                         (item["figure"], item["reason"], item["invalid"], item["partial"]))
    expanded = {k: v for k, v in figures["spans_decades"].items() if v > 2}
    if expanded:
        lines.extend(["", "Data require more than two decades; all panels still share the same span: %s." % expanded])
    lines.extend(["", "Full precision, signed squared-mean intervals and per-step counts are in "
                  "[ppo_steps_table.csv](ppo_steps_table.csv); a readable table is in "
                  "[ppo_steps_table.md](ppo_steps_table.md). Group paths, seeds and source commits are in "
                  "[analysis_record.json](analysis_record.json). No scientific jobs were launched by this analysis.", ""])
    (out / "NOTE.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--gate-record", action="append", default=[], type=Path)
    parser.add_argument("--rerun-result", type=Path)
    parser.add_argument("--original-result", action="append", default=[], type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error("Output directory is not empty; use a new candidate directory or explicitly --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    groups, ignored = discover_groups(args.groups_root)
    rows, missing = build_rows(groups)
    write_table(rows, out)
    figures = render_figures(rows, out)
    sanity = rerun_sanity(args.rerun_result, args.original_result, out)
    write_note(rows, groups, figures, args, out, sanity, ignored, missing)
    status = "complete" if len(rows) == 63 and all(r["groups"] == 10 and r["K"] == 500 for r in rows) else "partial"
    if any(row["u_sq"] <= 0 for row in rows):
        status = "blocked_nonpositive_signal"
    record = {"status": status,
              "groups_root": str(args.groups_root.resolve()), "parameter_count": PARAMETER_COUNT,
              "independent_unit": "50-batch job", "expected_groups_per_point": 10,
              "episodes_per_batch": 80, "figures": figures, "rerun_sanity": sanity,
              "ignored_incomplete_jobs": ignored, "missing_steps": missing,
              "selected_groups": [{"path": str(item["path"].resolve()), "meta": item["meta"]}
                                  for items in groups.values() for item in items],
              "rows": rows}
    save_json(out / "analysis_record.json", record)
    print(json.dumps({"status": record["status"], "groups": sum(map(len, groups.values())),
                      "table_rows": len(rows), "figures": figures["created"],
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
