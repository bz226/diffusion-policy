#!/usr/bin/env python3
"""Analyze paired optimizer gradients from stored group sums (Python 3.8).

No rollout, gradient computation, training, or modification of input files occurs.
Outputs go to a new candidate directory; an existing directory is never replaced.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
ITR_UPDATES = {0: 0, 35: 32, 70: 63, 105: 95, 139: 126}
ESTIMATORS = ("V1", "V2")
LABELS = {"V1": "without_PPO", "V2": "without_PPO_and_GAE"}
ORANGE = "#D55E00"
GREEN = "#009E73"
KEYS = ("mean_norm", "variance", "relative_variance")
OLD_LIMITS = {"mean_norm": [1e-5, 1e-2], "variance": [1e-5, 1e-3],
              "relative_variance": [1e1, 1e6]}


def number(value):
    if value is None or value == "":
        return None
    return float(value)


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unresolved"
    if math.isinf(float(value)):
        return "∞" if value > 0 else "−∞"
    return "%.6g" % value


def relative(path, base=ROOT):
    return os.path.relpath(str(path), str(base))


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value == math.inf else "-inf" if value == -math.inf else None
    return value


def preserved_figures(study, output=None):
    """Check only the explicit user-requested pre-existing figure hashes."""
    before_path = study / "runs/figure_hashes_before.json"
    before = json.loads(before_path.read_text())
    after = {}
    for name in before:
        path = ROOT / name
        after[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("A pre-existing figure changed; stop before publishing analysis")
    if output is not None:
        dump_json(output / "figure_hashes_after.json", after)
    return {"files": len(before), "unchanged": True, "before": str(before_path)}


def read_groups(collect_root):
    """Load completed fifty-batch groups; incomplete jobs are only recorded."""
    points, unfinished = {}, []
    seed_owners, return_owners, audited = {}, {}, []
    for meta_path in sorted(collect_root.rglob("meta.json")):
        meta = json.loads(meta_path.read_text())
        if not all(key in meta for key in ("checkpoint_itr", "updates", "K")):
            continue
        if meta.get("status") not in ("complete", "completed"):
            unfinished.append({"directory": str(meta_path.parent), "status": meta.get("status"),
                               "K": meta.get("K")})
            continue
        if not meta.get("clean_exit") or not meta.get("actor_and_critic_unchanged"):
            raise ValueError("Completed job lacks clean-exit/model-invariance evidence: " + str(meta_path))
        if meta.get("critic_checkpoint_sha256") != meta.get("critic_after_sha256"):
            raise ValueError("Collected critic differs from the checkpoint: " + str(meta_path))
        itr, update, count = int(meta["checkpoint_itr"]), int(meta["updates"]), int(meta["K"])
        if itr not in ITR_UPDATES or ITR_UPDATES[itr] != update:
            raise ValueError("Checkpoint/update mapping differs: " + str(meta_path))
        if count != 50 or int(meta.get("target_batches", 50)) != 50:
            raise ValueError("A completed group must contain fifty batches: " + str(meta_path))
        directory = meta_path.parent
        cleanup = json.loads((directory / "cleanup.json").read_text())
        if cleanup.get("status") != "passed" or not cleanup.get("all_children_exited"):
            raise ValueError("Job did not verify explicit worker cleanup: " + str(directory))
        vectors = [np.load(str(directory / ("sums_%s.npy" % v))) for v in ESTIMATORS]
        if any(a.dtype != np.dtype("float64") or a.ndim != 1 or not np.isfinite(a).all()
               for a in vectors):
            raise ValueError("Group sums must be finite float64 vectors: " + str(directory))
        if any(vector.shape != (568392,) for vector in vectors):
            raise ValueError("Expected the 568,392 actor-parameter coordinates")
        batches = list(csv.DictReader((directory / "batches.csv").open()))
        if len(batches) != count or len({int(r["batch"]) for r in batches}) != count:
            raise ValueError("Batch count/index mismatch: " + str(directory))
        batches.sort(key=lambda row: int(row["batch"]))
        if [int(row["batch"]) for row in batches] != list(range(50)):
            raise ValueError("Expected production batch indices 0–49")
        spec = meta["spec"]
        if meta["parent_revision"] != spec["parent_revision"]:
            raise ValueError("Job revision differs from its saved specification")
        if len(meta["seed_records"]) != 50:
            raise ValueError("Need fifty actual seed records")
        for index, row in enumerate(batches):
            base = (int(spec.get("seed_base", 50000000)) + int(spec["point_index"]) * 1000000
                    + int(spec["job_index"]) * 10000 + index * 100)
            expected = {"environment_seeds": list(range(base, base + 40)),
                        "diffusion_seed": base + 50, "permutation_seed": base + 51,
                        "python_seed": base + 52, "numpy_seed": base + 53}
            actual = json.loads(row["seeds"])
            if actual != expected or actual != meta["seed_records"][index]:
                raise ValueError("Actual seeds differ from metadata/specification: %s batch %d" % (directory, index))
            for seed in actual["environment_seeds"] + [actual[key] for key in
                    ("diffusion_seed", "permutation_seed", "python_seed", "numpy_seed")]:
                if seed in seed_owners:
                    raise ValueError("G4 reused actual seed %d in %s and %s" % (seed, seed_owners[seed], directory))
                seed_owners[seed] = str(directory)
            before = float(row["prebatch_scaler_c"])
            if not math.isclose(before, float(meta["scaler_c"]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Batch's pre-update scaler differs from frozen state")
            if itr != 0 and not math.isclose(float(row["postbatch_scaler_c"]), before, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Fitted later scaler changed during collection")
        first_returns = np.asarray(json.loads(batches[0]["episode_returns"]), dtype=np.float64)
        if first_returns.shape != (80,) or not np.isfinite(first_returns).all():
            raise ValueError("G4 requires eighty finite first-batch episode returns per job")
        for value in first_returns:
            owner = return_owners.get(float(value))
            if owner is not None and owner != str(directory):
                raise ValueError("G4 two production jobs share a first-batch episode return: %s and %s" % (owner, directory))
            return_owners[float(value)] = str(directory)
        audited.append({"directory": str(directory), "batches": 50, "actual_seed_values": 2200,
                        "first_batch_episode_returns_checked": 80,
                        "first_batch_return_mean": float(first_returns.mean())})
        norms = np.array([[float(r["norm_sq_" + v]) for v in ESTIMATORS] for r in batches])
        cross = np.array([float(r["cross_V1_V2"]) for r in batches])
        if not np.isfinite(norms).all() or not np.isfinite(cross).all() or np.any(norms <= 0):
            raise ValueError("Invalid stored batch scalars: " + str(directory))
        within = cross / np.sqrt(norms[:, 0] * norms[:, 1])
        if np.any(np.abs(within) > 1 + 1e-10):
            raise ValueError("Invalid within-batch cosine")
        if all("within_batch_cosine" in row for row in batches):
            recorded = np.array([float(r["within_batch_cosine"]) for r in batches])
            if not np.allclose(recorded, within, rtol=1e-11, atol=1e-13):
                raise ValueError("Stored within-batch cosine disagrees with squared norms/cross product")
        group = {"directory": str(directory), "meta": meta, "vectors": vectors,
                 "count": count, "Q": norms.sum(axis=0, dtype=np.float64),
                 "cross_sum": float(cross.sum(dtype=np.float64)),
                 "within_sum": float(within.sum(dtype=np.float64))}
        points.setdefault(update, []).append(group)
    for update, groups in points.items():
        if len(groups) > 10:
            raise ValueError("More than the approved ten complete groups at update %d" % update)
        checkpoints = {g["meta"]["source_checkpoint"] for g in groups}
        scales = {(float(g["meta"]["scaler_c"]), g["meta"]["scaler_c_source"]) for g in groups}
        if len(checkpoints) != 1 or len(scales) != 1:
            raise ValueError("Groups mix checkpoint/scaler states at update %d" % update)
    audit = {"passed": True, "jobs_checked": len(audited), "batches_checked": len(audited) * 50,
             "distinct_actual_seed_values": len(seed_owners),
             "first_batch_episode_returns_checked": len(audited) * 80,
             "duplicate_cross_job_first_batch_episode_returns": 0, "jobs": audited,
             "scope": "Every completed production job, not only the two live gate probes"}
    return points, unfinished, audit


def interval(value, leave_one_out, critical):
    values = np.asarray(leave_one_out, dtype=np.float64)
    if value is not None and not math.isfinite(float(value)):
        value = None
    if value is None or values.size < 2 or not np.isfinite(values).all():
        return {"value": value, "lo": None, "hi": None, "se": None}
    variance = (values.size - 1.0) / values.size * np.square(values - values.mean()).sum()
    se = math.sqrt(float(variance))
    return {"value": float(value), "lo": float(value - critical * se),
            "hi": float(value + critical * se), "se": se}


def summarize_point(groups, update):
    """Pairwise batch means and joint delete-one-job jackknife estimates."""
    count = len(groups)
    if count < 2:
        return None
    vectors = np.stack([np.stack(g["vectors"]) for g in groups])  # group, estimator, parameter
    gram = np.einsum("gip,hjp->ijgh", vectors, vectors, optimize=True, dtype=np.float64)
    del vectors
    if not np.isfinite(gram).all():
        raise ValueError("Nonfinite group inner products at update %d" % update)
    sizes = np.array([g["count"] for g in groups], dtype=np.float64)
    q = np.stack([g["Q"] for g in groups])
    within = np.array([g["within_sum"] for g in groups], dtype=np.float64)
    critical = float(student_t.ppf(.95, count - 1))

    def estimates(indices):
        n = float(sizes[indices].sum())
        n2 = float(np.square(sizes[indices]).sum())
        selected = gram[:, :, indices][:, :, :, indices]
        total = selected.sum(axis=(2, 3), dtype=np.float64)
        qsum = q[indices].sum(axis=0, dtype=np.float64)
        norm2 = np.diag(total)
        signal = (norm2 - qsum) / (n * (n - 1))
        variance = (qsum - norm2 / n) / (n - 1)
        denominator = n * n - n2
        group_u = ((total - np.trace(selected, axis1=2, axis2=3)) / denominator
                   if denominator > 0 else np.full((2, 2), np.nan))
        if np.any(variance <= 0):
            raise ValueError("Nonpositive covariance trace at update %d" % update)
        ratio = np.divide(variance, signal, out=np.full(2, np.nan), where=signal != 0)
        group_cos = (float(group_u[0, 1] / math.sqrt(group_u[0, 0] * group_u[1, 1]))
                     if np.all(np.diag(group_u) > 0) else None)
        return {"K": int(n), "signal": signal, "variance": variance, "ratio": ratio,
                "group_u": group_u, "group_cos": group_cos,
                "trace_ratio": float(variance[1] / variance[0]),
                "R_ratio": float(ratio[1] / ratio[0]) if np.isfinite(ratio).all() and ratio[0] != 0 else math.nan,
                "within": float(within[indices].sum() / n)}

    full = estimates(np.arange(count))
    jack = [estimates(np.delete(np.arange(count), leave)) for leave in range(count)]
    records = []
    for v, label in enumerate(ESTIMATORS):
        signal = interval(full["signal"][v], [j["signal"][v] for j in jack], critical)
        variance = interval(full["variance"][v], [j["variance"][v] for j in jack], critical)
        resolved = bool(signal["lo"] is not None and signal["lo"] > 0)
        positive = bool(signal["value"] > 0)
        ratio = interval(full["ratio"][v] if positive else None,
                         [j["ratio"][v] if j["signal"][v] > 0 else math.nan for j in jack], critical)
        lower_bound = (variance["value"] / signal["hi"] if not resolved and signal["hi"] > 0 else None)
        if not resolved or ratio["lo"] is None:
            # The plug-in bound below holds variance fixed. It is not a joint
            # confidence endpoint; keep it separate from the unbounded interval.
            ratio["lo"], ratio["hi"] = 0., math.inf
            ratio["se"] = None
        expected = ((1 + ratio["value"]) ** -.5 if resolved else
                    (1 + lower_bound) ** -.5 if lower_bound is not None else None)
        norm = {"value": math.sqrt(signal["value"]) if positive else None,
                "lo": math.sqrt(max(0., signal["lo"])),
                "hi": math.sqrt(max(0., signal["hi"])), "se": None}
        meta = groups[0]["meta"]
        record = {"method": LABELS[label], "estimator": label, "update": update,
                  "checkpoint_itr": int(meta["checkpoint_itr"]),
                  "checkpoint": Path(meta["source_checkpoint"]).name, "K": full["K"], "groups": count,
                  "episodes": 80 * full["K"], "mean_resolved": resolved,
                  "relative_variance_lower_bound": lower_bound,
                  "expected_cosine": expected, "expected_cosine_relation": "=" if resolved else "<=",
                  "scale_c": float(meta["scaler_c"]), "scale_source": meta["scaler_c_source"],
                  "scale_approximation": bool(meta.get("scaler_approximation", update > 0)),
                  "t_critical": critical, "df": count - 1}
        for name, stat in (("signal_sq", signal), ("mean_norm", norm),
                           ("variance", variance), ("relative_variance", ratio)):
            for suffix, source in (("", "value"), ("_lo", "lo"), ("_hi", "hi"), ("_se", "se")):
                record[name + suffix] = stat[source]
        for suffix in ("", "_lo", "_hi"):
            value = record["relative_variance" + suffix]
            record["B" + suffix] = 80 * value if value is not None else None
        records.append(record)
    both_resolved = all(row["mean_resolved"] for row in records)
    cosine_defined = both_resolved and full["group_cos"] is not None and all(j["group_cos"] is not None for j in jack)
    ratio_defined = both_resolved and all(np.all(j["signal"] > 0) for j in jack)
    paired = {"update": update, "K": full["K"], "groups": count,
              "trace_ratio": interval(full["trace_ratio"], [j["trace_ratio"] for j in jack], critical),
              "R_ratio": interval(full["R_ratio"], [j["R_ratio"] for j in jack], critical) if ratio_defined else None,
              "mean_dot": interval(float(full["group_u"][0, 1]),
                                   [float(j["group_u"][0, 1]) for j in jack], critical),
              "mean_cosine": interval(full["group_cos"], [j["group_cos"] for j in jack], critical) if cosine_defined else None,
              "within_batch_cosine": interval(full["within"], [j["within"] for j in jack], critical),
              "group_signal_sq_V1": float(full["group_u"][0, 0]),
              "group_signal_sq_V2": float(full["group_u"][1, 1]),
              "mean_cosine_resolved": cosine_defined,
              "mean_dot_formula": "between-group U statistic; same formula for both cosine denominators",
              "directory_names": [g["directory"] for g in groups]}
    return {"rows": records, "paired": paired,
            "groups": [{"directory": g["directory"], "meta": g["meta"]} for g in groups]}


def read_references(path):
    rows = []
    for source in csv.DictReader(path.open()):
        row = dict(source)
        row["update"], row["K"], row["groups"] = int(row["update"]), int(row["K"]), int(row["groups"])
        row["step"] = None if row["step"] in ("", "–") else int(row["step"])
        row["mean_resolved"] = row["mean_resolved"] == "True"
        for key in KEYS + ("signal_sq", "relative_variance_lower_bound", "expected_cosine"):
            for suffix in ("", "_lo", "_hi"):
                if key + suffix in row:
                    row[key + suffix] = number(row[key + suffix])
        if row["method"] not in ("DPPO", "without_PPO_and_GAE"):
            raise ValueError("Unexpected reference method")
        rows.append(row)
    if len(rows) != 65:
        raise ValueError("Expected the 65 saved reference rows")
    return rows


def shared_limits(rows, references):
    limits = {key: list(bounds) for key, bounds in OLD_LIMITS.items()}
    widened = False
    for key in KEYS:
        values = [row[key] for row in rows + references if row[key] is not None and row[key] > 0]
        if min(values) < limits[key][0]:
            limits[key][0] = 10. ** math.floor(math.log10(min(values)) - .25)
            widened = True
        if max(values) > limits[key][1]:
            limits[key][1] = 10. ** math.ceil(math.log10(max(values)) + .25)
            widened = True
    return limits, widened


def unresolved_caption(rows):
    entries = []
    for row in rows:
        if not row["mean_resolved"]:
            bound = row["relative_variance_lower_bound"]
            entries.append("%s at %d updates: %s" % (
                row["method"], row["update"], "R ≥ " + fmt(bound) if bound is not None else "no finite R bound"))
    text = (" Means not resolved: " + "; ".join(entries) +
            ". Ratios drawn are point estimates; bounds condition on the variance estimate.") if entries else " All displayed means are resolved."
    nonpositive = ["%s at %d updates" % (row["method"], row["update"])
                   for row in rows if row.get("signal_sq") is not None and row["signal_sq"] <= 0]
    if nonpositive:
        text += (" Nonpositive squared-mean estimates at " + "; ".join(nonpositive) +
                 " cannot define a norm or positive R on a log axis; those points are absent from those panels.")
    return text


def captions(rows, references, complete):
    own = [row for row in rows if row["estimator"] == "V1"]
    reference = [row for row in references if row["method"] == "without_PPO_and_GAE"]
    sample = ("500 batches of 80 episodes per point" if complete else
              "Partial collection; the table gives completed batches and groups at each point")
    essential = (sample + ", in training-loss units. Points are estimates; 90% intervals are in noise_without_ppo_table. "
                 "The native 50,000-pair chunk normalization is retained. Later checkpoints use a fixed fitted reward "
                 "scale, so their gradients approximate the historical scaler state; intervals condition on this fit.")
    first = ("The gradient without_PPO passed to AdamW, at its own checkpoints: its mean norm, the trace of its "
             "covariance across batches, and their ratio, the relative variance of one 80-episode update "
             "(B/80 in the per-episode convention). " + essential + unresolved_caption(own))
    second = (first + " Each run is measured at its own checkpoints, so the parameters differ between the two lines "
              "at the same update count. The stored without_PPO_and_GAE reference used full-batch normalization, "
              "previously checked approximately against its native chunks, and a fitted scale at update 126." +
              unresolved_caption(reference))
    third = ("Both gradients computed on the same batches at without_PPO's parameters, differing only in the advantage: "
             "GAE (λ = 0.95) with the critic vs the Monte Carlo return minus the same critic. " +
             essential + unresolved_caption(rows))
    return {"without_ppo_noise_actual": first, "one_step_noise_actual_compare": second,
            "gae_vs_mc_paired_noise": third}


def render_figure(name, series, limits, output, title, subtitle, caption):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
    settings = {"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": .6,
                "pdf.fonttype": 42, "ps.fonttype": 42}
    titles = ("Mean gradient\nnorm", "Gradient\nvariance\n(one update)", "Relative\nvariance\n(one update)")
    labels = (r"$\|\mathrm{E}\,g\|$" + "\n(training-loss units)",
              r"$\mathrm{tr}\,\mathrm{Cov}(g)$" + "\n(training-loss units²)",
              r"$\mathrm{tr}\,\mathrm{Cov}(g)/\|\mathrm{E}\,g\|^2$" + "\n(no units)")
    checks, omitted = [], []
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 3, figsize=(3.6, 3.1))
        # Reserve explicit space for long method names and three-line titles.
        # Unlike an axes-anchored legend, this cannot shrink the data axes when
        # the title or label widths change between the three approved figures.
        fig.subplots_adjust(left=.155, right=.985, bottom=.24,
                            top=.63 if len(series) > 1 else .74, wspace=1.0)
        for ax, key, panel, label in zip(axes, KEYS, titles, labels):
            for method, color, data in series:
                ordered = sorted(data, key=lambda row: row["update"])
                x = [row["update"] for row in ordered]
                y = [row[key] if row[key] is not None and row[key] > 0 else math.nan for row in ordered]
                omitted += [{"figure": name, "method": method, "update": row["update"], "quantity": key}
                            for row in ordered if row[key] is None or row[key] <= 0]
                line, = ax.plot(x, y, "o-", color=color, label=method, markersize=2.3, linewidth=.9,
                                markerfacecolor=color, markeredgecolor=color)
                assert np.array_equal(line.get_xdata(), x)
                assert np.array_equal(line.get_ydata(), y, equal_nan=True)
                assert line.get_marker() == "o" and line.get_markerfacecolor() == color
                assert line.get_markeredgecolor() == color and line.get_linestyle() == "-"
                checks.append({"quantity": key, "method": method, "updates": x, "values": y})
            ax.set_yscale("log")
            ax.set_ylim(limits[key])
            ax.set_xlim(-5, 131)
            ax.set_xticks([0, 32, 63, 95, 126])
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")
            ax.tick_params(axis="both", labelsize=6.2, length=2.5, pad=1.5)
            ax.tick_params(axis="y", which="minor", length=0)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(panel, fontsize=7, pad=5)
            ax.set_ylabel(label, fontsize=6.4, labelpad=1.5)
            ax.grid(axis="y", which="major", color=".9", linewidth=.5)
            ax.grid(axis="x", visible=False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            assert len(ax.collections) == 0
            assert len(ax.lines) == len(series)
            assert list(ax.get_ylim()) == limits[key]
            assert list(ax.get_xticks()) == [0, 32, 63, 95, 126]
            assert ax.get_yscale() == "log"
            assert not any(line.get_visible() for line in ax.get_xgridlines())
        if len(series) > 1:
            handles, names = axes[0].get_legend_handles_labels()
            fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .88),
                       frameon=False, fontsize=6.2, ncol=1, handlelength=1.5)
        fig.suptitle(title + "\n" + subtitle, fontsize=7.8, y=.985)
        fig.supxlabel("Updates (one update = 80 episodes)", fontsize=7, y=.02)
        fig.set_dpi(300)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = list(fig.texts)
        for ax in axes:
            low, high = ax.get_ylim()
            visible_ylabels = [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
                               if low <= value <= high]
            artists += [ax.title, ax.yaxis.label] + ax.get_xticklabels() + visible_ylabels
        for legend in fig.legends:
            artists += legend.get_texts()
        for artist in artists:
            if not artist.get_visible() or not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -.5 and box.y0 >= -.5, "Text clipped at lower/left edge: " + artist.get_text()
            assert box.x1 <= fig.bbox.width + .5 and box.y1 <= fig.bbox.height + .5, "Text clipped at upper/right edge: " + artist.get_text()
        for legend in fig.legends:
            box = legend.get_window_extent(renderer)
            for artist in list(fig.texts) + [ax.title for ax in axes]:
                assert not box.overlaps(artist.get_window_extent(renderer)), "Legend overlaps a figure or panel title"
        for extension in ("png", "pdf"):
            metadata = {"Description": caption} if extension == "png" else {"Subject": caption}
            fig.savefig(str(output / (name + "." + extension)), dpi=300, metadata=metadata)
        (output / (name + ".caption.md")).write_text(caption + "\n", encoding="utf-8")
        plt.close(fig)
    return {"figure": name, "checks": checks, "omitted_nonpositive_points": omitted,
            "render_verified": True, "layout_verified": True,
            "filled_circles_only": True, "shared_limits_verified": True,
            "horizontal_grid_only": True, "dpi": 300}


def write_tables(rows, paired, output):
    paired_rows = []
    for item in paired:
        row = {"update": item["update"], "K": item["K"], "groups": item["groups"]}
        for key in ("trace_ratio", "R_ratio", "mean_dot", "mean_cosine", "within_batch_cosine"):
            for suffix, source in (("", "value"), ("_lo", "lo"), ("_hi", "hi")):
                row[key + suffix] = item[key][source] if item[key] is not None else None
        paired_rows.append(row)
    # Repeat the checkpoint's paired fields on its two estimator rows so the
    # requested single CSV contains every per-estimator and paired quantity.
    combined = []
    for source in rows:
        row = dict(source)
        paired_row = next(item for item in paired_rows if item["update"] == source["update"])
        row.update({"paired_" + key: value for key, value in paired_row.items()
                    if key not in ("update", "K", "groups")})
        combined.append(row)
    with (output / "noise_without_ppo_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    def cell(row, key):
        return "%s [%s, %s]" % (fmt(row[key]), fmt(row[key + "_lo"]), fmt(row[key + "_hi"]))
    lines = ["# Paired gradient noise at without_PPO's parameters", "",
             "Square brackets are 90% delete-one-group jackknife intervals with t critical values on groups − 1 "
             "degrees of freedom. Each independent group contains 50 fresh 80-episode batches. "
             "Norm intervals monotonically transform the signed squared-mean interval (CSV); variance and R use "
             "the joint jackknife. R is unitless and B = 80R is in episodes.", "",
             "| Estimator | Updates | K / groups | Mean norm [90%] | tr Cov(g) [90%] | R [90%] | Mean resolved? | R bound | Expected cosine | B [90%] |",
             "|---|---:|---:|---|---|---|---|---|---|---|"]
    for row in rows:
        cosine = ("≤ " if row["expected_cosine_relation"] == "<=" else "") + fmt(row["expected_cosine"])
        lines.append("| %s | %d | %d / %d | %s | %s | %s | %s | %s | %s | %s |" % (
            row["method"], row["update"], row["K"], row["groups"], cell(row, "mean_norm"),
            cell(row, "variance"), cell(row, "relative_variance"), "yes" if row["mean_resolved"] else "no",
            "≥ " + fmt(row["relative_variance_lower_bound"]) if row["relative_variance_lower_bound"] else "—",
            cosine, cell(row, "B")))
    lines += ["", "## Paired Monte Carlo / GAE comparison", "",
              "Both estimators use each batch's identical parameters, critic and native chunk normalization. "
              "The mean dot product excludes same-group pairs. Its cosine uses that same group U statistic "
              "for both denominator squared norms; it is unresolved if either full or deleted-group squared "
              "norm is nonpositive. R ratios and direction cosines are shown only when both individual means resolve.", "",
              "| Updates | Trace ratio [90%] | R ratio [90%] | Mean dot product [90%] | Mean-direction cosine [90%] | Within-batch cosine [90%] |",
              "|---:|---|---|---|---|---|"]
    for row in paired_rows:
        lines.append("| %d | %s | %s | %s | %s | %s |" % (row["update"], cell(row, "trace_ratio"),
                     cell(row, "R_ratio"), cell(row, "mean_dot"), cell(row, "mean_cosine"), cell(row, "within_batch_cosine")))
    lines += ["", "## State and resolution limits", "",
              "A mean is resolved only when its squared-mean interval is entirely positive. If it reaches zero, "
              "the requested bound R ≥ tr Cov(g) / upper(‖E g‖²) holds the variance at its point estimate; "
              "it is not a simultaneous confidence bound for numerator and denominator. The implied expected-cosine "
              "bound is an upper bound. The joint R interval is reported as [0, ∞) when the mean is unresolved "
              "or a deleted-group denominator is nonpositive; the separate plug-in bound is never its lower endpoint. "
              "Undefined nonpositive point estimates remain unresolved, never clipped to positive values.", "",
              "| Updates | Reward divisor | Source | Approximation? |", "|---:|---:|---|---|"]
    for row in rows[::2]:
        lines.append("| %d | %s | %s | %s |" % (row["update"], fmt(row["scale_c"]), row["scale_source"],
                                               "yes" if row["scale_approximation"] else "no"))
    lines += ["", "Later fitted divisors do not reconstruct the historical running scaler. Their figures and intervals "
              "condition on this approximation. CSV estimator codes identify the formulas, not additional training runs.", ""]
    (output / "noise_without_ppo_table.md").write_text("\n".join(lines), encoding="utf-8")


def report_text(rows, paired, references, complete, limits, widened, record):
    lines = ["# Gradient noise of without_PPO", "",
             "Does replacing GAE by Monte Carlo change the noise of a one-step actor gradient, at identical parameters?", "",
             ("The approved collection is complete." if complete else "The collection is incomplete; these are provisional estimates.") +
             " The paired comparison uses native chunk normalization. Later checkpoints use fitted, fixed reward divisors, "
             "so those points approximate the historical training scaler; all intervals condition on the fits.", "",
             "## Q1. Noise of the optimizer gradient", "",
             "R = tr Cov(g) / ‖E g‖² is unitless. The two own-run references compare different parameters at equal updates.", "",
             "| Updates | without_PPO R | without_PPO_and_GAE R, own run | DPPO step-1 R, own run |",
             "|---:|---:|---:|---:|"]
    if record.get("plot_status") == "nonpositive_points_omitted":
        lines[4:4] = ["**Figure completion is blocked:** at least one squared-mean point estimate is nonpositive. "
                      "It cannot define the requested mean norm or R on a log axis. Signed estimates and intervals "
                      "are preserved in the table; the incomplete candidate figures are not completed deliverables.", ""]
    for row in rows:
        if row["estimator"] != "V1":
            continue
        own = next((r for r in references if r["method"] == "without_PPO_and_GAE" and r["update"] == row["update"]), None)
        dppo = next((r for r in references if r["method"] == "DPPO" and r["update"] == row["update"] and r["step"] == 1), None)
        lines.append("| %d | %s | %s | %s |" % (row["update"], fmt(row["relative_variance"]),
                     fmt(own["relative_variance"]) if own else "—", fmt(dppo["relative_variance"]) if dppo else "—"))
    lines += ["", "[Own-run comparison](one_step_noise_actual_compare.png). The older green reference uses full-batch "
              "normalization, previously checked approximately against its native chunks; the new paired estimates use "
              "the exact chunk path. This limits a strict panel-by-panel cross-study interpretation.", "",
              "## Q2. Paired advantage comparison", "",
              "| Updates | Monte Carlo / GAE covariance trace | Monte Carlo / GAE R | Within-batch cosine |",
              "|---:|---:|---:|---:|"]
    for item in paired:
        lines.append("| %d | %s | %s | %s |" % (item["update"], fmt(item["trace_ratio"]["value"]),
                     fmt(item["R_ratio"]["value"]) if item["R_ratio"] else "unresolved",
                     fmt(item["within_batch_cosine"]["value"])))
    lines += ["", "[Paired figure](gae_vs_mc_paired_noise.png) · [All intervals and mean-direction comparisons](noise_without_ppo_table.md).", "",
              "## Q3. Resolution", "", unresolved_caption(rows).strip(), "",
              "## Reproducibility and limits", "",
              "Checkpoint iterations 0, 35, 70, 105 and 139 map to 0, 32, 63, 95 and 126 completed actor updates. "
              "Each point is capped at ten independent 50-batch jobs. The table records completed sample counts, "
              "scaler values and sources. Group sums and statistics are float64; 90% intervals use a delete-one-job "
              "jackknife and t quantiles (9 degrees of freedom for complete points). No gradients were recomputed for analysis.", "",
              "Shared y-limits: norm %s; covariance %s; R %s." % tuple(str(limits[key]) for key in KEYS), "",
              ("New values require wider limits. Comparison copies use these limits in this study only; "
               "all original figures remain unchanged." if widened else "All new figures retain the existing per-gradient y-limits."), "",
              "[Analysis record](analysis_record.json) records exact input directories, revisions, seeds, command, "
              "sample counts and the existing-figure preservation check. See the study's methods and gate records "
              "for native gradient identity, state checks and independence verification. This analysis does not establish "
              "that GAE versus Monte Carlo alone explains the performance difference between trained policies.", ""]
    gates = record.get("gate_results")
    if gates is not None:
        checks = gates["checks"]
        errors = [checks[key]["gradient_comparison"]["relative_l2"] for key in ("G1", "G2")]
        lines += ["## Verification", "",
                  "Native-gradient identity passed: GAE relative L2 %s; Monte Carlo relative L2 %s (limit 1e−6). "
                  "The critic/state check, independent-stream check and default-path CPU replay all passed. "
                  "The final production audit checked %d jobs, %d actual batches and %d first-batch episode returns; "
                  "no seed or cross-job first-return duplicate was found. "
                  "All %d pre-existing figures retain their requested hashes." % (
                      fmt(errors[0]), fmt(errors[1]), record["G4_actual_production_audit"]["jobs_checked"],
                      record["G4_actual_production_audit"]["batches_checked"],
                      record["G4_actual_production_audit"]["first_batch_episode_returns_checked"],
                      record["existing_figures"]["files"]), "",
                  "[Exact gate evidence](gate_summary.json) · [Analysis/source provenance](analysis_record.json). "
                  "No new gates were run during this analysis.", ""]
    return "\n".join(lines)


def report_links(text, source_directory, destination_directory, published=None):
    """Relocate relative Markdown links without rewriting the saved candidate."""
    published = published or {}
    def replace(match):
        target = match.group(1)
        if target.startswith(("http://", "https://", "#", "/")):
            return match.group(0)
        source = source_directory / target
        destination = published.get(str(source), source)
        return "](" + os.path.relpath(str(destination), str(destination_directory)) + ")"
    return re.sub(r"\]\(([^)]+)\)", replace, text)


def new_copy(source, destination):
    """Copy a new deliverable exclusively, never replacing an existing file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if source.read_bytes() != destination.read_bytes():
        raise RuntimeError("Published copy differs from its candidate: " + str(destination))


def publish_final(study, output, record, note):
    """Publish requested new outputs after actual render and preservation checks."""
    if record["status"] != "complete" or record["plot_status"] != "complete":
        raise RuntimeError("Incomplete analysis cannot publish final figures")
    names = ("without_ppo_noise_actual", "one_step_noise_actual_compare", "gae_vs_mc_paired_noise")
    plotted = {item["figure"]: item for item in record["figure_checks"] if "figure" in item}
    from PIL import Image
    mapping = {}
    verification = []
    for name in names:
        check = plotted.get(name, {})
        required = ("render_verified", "layout_verified", "filled_circles_only",
                    "shared_limits_verified", "horizontal_grid_only")
        if not all(check.get(key) for key in required) or check.get("omitted_nonpositive_points"):
            raise RuntimeError("Figure lacks required actual-render checks: " + name)
        for panel in check["checks"]:
            if panel["updates"] != [0, 32, 63, 95, 126] or any(
                    value is None or not math.isfinite(value) or value <= 0 for value in panel["values"]):
                raise RuntimeError("Final figure lacks a requested positive point: " + name)
        for extension in ("png", "pdf", "caption.md"):
            source = output / (name + "." + extension)
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError("Missing candidate deliverable: " + str(source))
            mapping[str(source)] = study / "figs" / source.name
        with Image.open(str(output / (name + ".png"))) as raster:
            dpi = raster.info.get("dpi")
            if dpi is None or any(abs(value - 300) > .01 for value in dpi):
                raise RuntimeError("PNG does not record 300 dpi: " + name)
            raster.verify()
        with (output / (name + ".pdf")).open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise RuntimeError("Invalid PDF candidate: " + name)
        verification.append({"figure": name, "png_verified": True, "png_dpi": list(dpi),
                             "pdf_header_verified": True, "render_object_checks_passed": True})
    for extension in ("csv", "md"):
        source = output / ("noise_without_ppo_table." + extension)
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("Missing table candidate")
        mapping[str(source)] = study / "figs" / source.name
    references = output / "widened_reference_copies"
    if references.exists():
        for source in sorted(references.rglob("*")):
            if source.is_file():
                mapping[str(source)] = study / "figs/widened_reference_copies" / source.relative_to(references)
    report = study / "REPORT.md"
    conflicts = [str(path) for path in list(mapping.values()) + [report] if path.exists()]
    if conflicts:
        raise FileExistsError("Refuse to replace existing deliverables: " + ", ".join(conflicts))
    preserved_figures(study, output)
    for source, destination in mapping.items():
        new_copy(Path(source), destination)
    preserved = preserved_figures(study, output)
    text = report_links(note, output, study, mapping)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {"status": "published", "paths": [str(path) for path in mapping.values()] + [str(report)],
            "candidate_files_retained": True, "curated": False,
            "render_verification": verification, "existing_figures": preserved}


def publish_blocked_report(study, output, note):
    report = study / "REPORT.md"
    if report.exists():
        raise FileExistsError("Refuse to replace existing report: " + str(report))
    text = report_links(note, output, study)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {"status": "blocked_report_only", "paths": [str(report)],
            "final_figures_published": False, "candidate_evidence_retained": True, "curated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--collect-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--write-report", action="store_true",
                        help="Finalize candidates and publish new study figs/table/REPORT.md after verification")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    if args.write_report and args.tables_only:
        raise ValueError("Final delivery requires the requested figures; do not combine --write-report with --tables-only")
    started = time.monotonic()
    study, output = args.study.resolve(), args.output.resolve()
    collect_root = (args.collect_root or study / "collect").resolve()
    if output.exists():
        raise FileExistsError("Refuse to replace existing analysis: " + str(output))
    preserved = preserved_figures(study)
    groups, unfinished, independence = read_groups(collect_root)
    complete = set(groups) == set(ITR_UPDATES.values()) and all(len(v) == 10 for v in groups.values())
    if args.require_complete and not complete:
        raise RuntimeError("The approved 5 × 10 complete groups are not all present")
    if args.write_report and not complete:
        raise RuntimeError("Do not finalize REPORT.md before all groups complete")
    gate_path = study / "runs/gates/complete/gate_summary.json"
    gates = json.loads(gate_path.read_text()) if gate_path.exists() else None
    if args.require_complete or args.write_report:
        if gates is None or not gates.get("passed") or set(gates.get("checks", {})) != {"G1", "G2", "G3", "G4", "G5"}:
            raise RuntimeError("Final analysis requires the saved passing G1–G5 gate record")
        if not all(check.get("passed") for check in gates["checks"].values()):
            raise RuntimeError("A required saved gate is not passed")
    stats = {update: summarize_point(value, update) for update, value in sorted(groups.items()) if len(value) >= 2}
    if not stats:
        raise RuntimeError("Need at least two completed groups at one point for a jackknife analysis")
    rows = [row for point in stats.values() for row in point["rows"]]
    paired = [point["paired"] for point in stats.values()]
    reference_path = ROOT / "results/stage3/gradmean/figs_clean/noise_actual_table.csv"
    references = read_references(reference_path)
    limits, widened = shared_limits(rows, references)
    output.mkdir(parents=True)
    write_tables(rows, paired, output)
    cap = captions(rows, references, complete)
    checks = []
    if not args.tables_only:
        own = [row for row in rows if row["estimator"] == "V1"]
        alternative = [row for row in rows if row["estimator"] == "V2"]
        ref = [row for row in references if row["method"] == "without_PPO_and_GAE"]
        subtitle = "each point: 500 batches of 80 episodes" if complete else "partial collection; completed counts in table"
        setups = [
            ("without_ppo_noise_actual", [("without_PPO", ORANGE, own)], "without_PPO, as run"),
            ("one_step_noise_actual_compare", [("without_PPO", ORANGE, own), ("without_PPO_and_GAE", GREEN, ref)], "One-step optimizer gradients"),
            ("gae_vs_mc_paired_noise", [("GAE + critic", ORANGE, own), ("Monte Carlo + critic", GREEN, alternative)], "At without_PPO's parameters"),
        ]
        for name, series, title in setups:
            checks.append(render_figure(name, series, limits, output, title, subtitle, cap[name]))
        if widened:
            # These renderers only read the supplied saved estimates. Copies remain
            # inside this study, preserving all old artifacts byte for byte.
            ref_out = output / "widened_reference_copies"
            ref_out.mkdir()
            sys.path.insert(0, str(ROOT / "results/stage3/gradmean"))
            import noise_actual_data
            from plot_noise_actual import render_as_run
            old_caps = noise_actual_data.captions(references)
            checks.append(render_as_run(references, limits, ref_out, old_caps["without_ppo_and_gae_noise_actual"]))
            source = ROOT / "results/stage3/ppo_steps/code/plot_noise_actual.py"
            spec = importlib.util.spec_from_file_location("saved_dppo_noise_renderer", str(source))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            checks.extend(module.render(references, limits, ref_out, old_caps))
            for extension in ("csv", "md"):
                shutil.copy2(str(reference_path.with_suffix("." + extension)),
                             str(ref_out / ("noise_actual_table." + extension)))
    record = {"status": "complete" if complete else "partial", "command": [sys.executable] + sys.argv,
              "input_root": str(collect_root), "reference_table": str(reference_path),
              "statistics": stats, "unfinished_jobs": unfinished, "limits": limits,
              "G4_actual_production_audit": independence,
              "gate_results": gates, "gate_source": str(gate_path),
              "campaign_record": str(study / "runs/campaign/state.json"),
              "widened_reference_copies": widened, "figure_checks": checks,
              "existing_figures": preserved_figures(study, output),
              "sampling_unit": "one independent 50-batch job", "gradient_units": "training loss",
              "normalization": "native 50,000-pair chunks", "new_sampling": False,
              "elapsed_seconds": time.monotonic() - started}
    source_dir = output / "source"
    source_dir.mkdir()
    shutil.copy2(str(Path(__file__).resolve()), str(source_dir / "analyze.py"))
    record["analysis_source_snapshot"] = str(source_dir / "analyze.py")
    if gates is not None:
        shutil.copy2(str(gate_path), str(output / "gate_summary.json"))
    record["parent_revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], universal_newlines=True).strip()
    record["plot_status"] = ("nonpositive_points_omitted" if any(
        check.get("omitted_nonpositive_points") for check in checks) else
        "not_requested" if args.tables_only else "complete")
    if record["plot_status"] == "nonpositive_points_omitted":
        record["status"] = "blocked_nonpositive_signal"
    dump_json(output / "analysis_record.json", json_safe(record))
    note = report_text(rows, paired, references, complete, limits, widened, record)
    (output / ("REPORT.md" if args.write_report else "ANALYSIS_NOTE.md")).write_text(note, encoding="utf-8")
    if args.write_report:
        if record["status"] == "blocked_nonpositive_signal":
            record["publication"] = publish_blocked_report(study, output, note)
        else:
            record["publication"] = publish_final(study, output, record, note)
        dump_json(output / "analysis_record.json", json_safe(record))
    print(json.dumps({"output": str(output), "status": record["status"], "points": len(stats),
                      "complete_groups": {str(k): len(v) for k, v in groups.items()},
                      "preserved_figures": preserved["files"], "widened_limits": widened}))
    if record["status"] == "blocked_nonpositive_signal" and (args.require_complete or args.write_report):
        raise RuntimeError("Requested log-axis figures incomplete: nonpositive squared-mean estimates; see saved analysis record and table")


if __name__ == "__main__":
    main()
