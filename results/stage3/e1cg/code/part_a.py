#!/usr/bin/env python
"""Part A: paired existing-gradient alignment, without sampling or training.

Python 3.8. The existing V6-aware PointData implementation supplies float64
same-batch-excluded U moments; this module applies the requested strict
jackknife validity and common-mean checks without changing older analyses.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t


VARIANTS = ("V2", "V4", "V5", "V6")
PAIRS = (("V2", "V6"), ("V2", "V4"), ("V2", "V5"),
         ("V4", "V6"), ("V5", "V6"))
OWN = (("W0_000", 0), ("W0_035", 32), ("W0_070", 63),
       ("W0_105", 95), ("W0_139", 126))
OTHER = (("P0", 0, "pretrained policy"), ("D1", 63, "DPPO"),
         ("D2", 126, "DPPO"), ("N1", 63, "without_PPO"),
         ("N2", 126, "without_PPO"), ("C1", 63, "without_PPO_and_GAE"))


def read(path):
    return json.loads(Path(path).read_text())


def completed_groups(root, point):
    groups = []
    for path in sorted(root.glob(point + "_j*/meta.json")):
        meta = read(path)
        if (meta.get("status") == "completed" and meta.get("completed_batches") == 50
                and all(v in meta.get("variant_ids", []) for v in VARIANTS)):
            groups.append(path.parent)
    return groups


def strict_cosine(data, first, second):
    """A failed full OR leave-one-group denominator invalidates the cosine."""
    a, b = data.variants.index(first), data.variants.index(second)
    moments = [data.full_moments] + data.deleted_moments
    values, failures = [], []
    for index, moment in enumerate(moments):
        aa, bb = float(moment["signal"][a]), float(moment["signal"][b])
        ab = float(moment["cross_signal"][a, b])
        if not all(math.isfinite(x) for x in (aa, bb, ab)) or aa <= 0 or bb <= 0:
            failures.append({"replicate": "full" if index == 0 else data.paths[index - 1].name,
                             "U_first_first": aa, "U_second_second": bb})
            values.append(None)
        else:
            values.append(ab / math.sqrt(aa * bb))
    result = {"pair": first + "_" + second, "first": first, "second": second,
              "groups": data.n, "K": data.k, "df": data.n - 1,
              "critical": float(student_t.ppf(.95, data.n - 1)),
              "U_first_first": float(data.full_moments["signal"][a]),
              "U_second_second": float(data.full_moments["signal"][b]),
              "U_first_second": float(data.full_moments["cross_signal"][a, b]),
              "status": "unresolved" if failures else "resolved", "failures": failures,
              "value": None, "se": None, "low": None, "high": None}
    if not failures:
        full, deleted = values[0], np.asarray(values[1:], dtype=np.float64)
        se = math.sqrt((data.n - 1.0) / data.n *
                       np.square(deleted - deleted.mean()).sum(dtype=np.float64))
        result.update(value=full, se=se, low=full - result["critical"] * se,
                      high=full + result["critical"] * se)
    result["sanity_reaches_one"] = (None if failures else result["low"] <= 1.0 <= result["high"])
    return result


def verify_cross_products(data):
    """Cross-check metadata sufficient statistics against saved batch rows."""
    max_difference = 0.0
    for group, rows in enumerate(data.batches):
        for a, first in enumerate(data.variants):
            for b, second in enumerate(data.variants):
                key = first + "_" + second + "_dot"
                value = math.fsum(float(row[key]) for row in rows)
                reference = float(data.q[group, a, b])
                difference = abs(value - reference)
                max_difference = max(max_difference, difference)
                if not math.isclose(value, reference, rel_tol=2e-12, abs_tol=1e-12):
                    raise ValueError("CSV/metadata cross moment differs: %s %s" %
                                     (data.paths[group], key))
    return max_difference


def describe_cohort(project, key, update, method, own, paths):
    meta = read(paths[0] / "meta.json")
    settings = meta["settings"]
    row = {"point": key, "path_method": method, "updates": update, "own_critic": own,
           "native_point": meta["point"], "source_checkpoint": meta["source_checkpoint"],
           "source_commit": meta["parent_commit"], "group_paths": [str(p) for p in paths],
           "group_ids": [p.name for p in paths], "measurement_seeds": [read(p / "meta.json")["seed"] for p in paths],
           "groups": len(paths), "K": 50 * len(paths), "m": 4000 * len(paths),
           "normalization": "full-batch proxy; training uses 50000-pair chunks",
           "V2_definition": "raw MC G minus cV" if own else "MC of scaled/clipped rewards minus V",
           "warmup_batches": 0 if own else settings["gm_warmup_batches"],
           "same_batch_products_available": True}
    if own:
        prepared = read(project / "results/stage3/gradmean/runs/clean_v5_20261012/prepare" / key / "grad_mean.json")
        row.update(c=prepared["effective_scaler_c"], c_source=prepared["scaler_c_source"])
    else:
        row.update(c=None, c_source="frozen reward scaler after critic warm-up")
    return row


def analyze(project, source):
    sys.path.insert(0, str(source))
    core = importlib.import_module("analysis_core")
    if Path(core.__file__).resolve().parent != source.resolve():
        raise ValueError("Wrong analysis_core loaded")
    own_root = project / "results/stage3/gradmean/runs/clean_v5_20261012/collect"
    old_root = project / "results/stage3/gradmean/collect"
    specs = [(point, update, "without_PPO_and_GAE", True, own_root) for point, update in OWN]
    specs += [(point, update, method, False, old_root) for point, update, method in OTHER]
    result = []
    for point, update, method, own, root in specs:
        paths = completed_groups(root, point)
        expected = 9 if point == "C1" else 10
        if len(paths) != expected:
            raise ValueError("Expected %d approved groups at %s, found %d" % (expected, point, len(paths)))
        cohort = describe_cohort(project, point, update, method, own, paths)
        data = core.PointData(cohort["native_point"], paths, variants=VARIANTS)
        cohort["cross_product_csv_check_max_abs"] = verify_cross_products(data)
        cohort["pairs"] = {a + "_" + b: strict_cosine(data, a, b) for a, b in PAIRS}
        checks = [cohort["pairs"][name] for name in ("V4_V6", "V5_V6")]
        failed = [x["pair"] for x in checks if x["status"] == "resolved" and not x["sanity_reaches_one"]]
        unresolved = [x["pair"] for x in checks if x["status"] != "resolved"]
        main = cohort["pairs"]["V2_V6"]
        cohort["sanity_status"] = "failed" if failed else "unresolved" if unresolved else "passed"
        cohort["failed_sanity_pairs"] = failed
        cohort["unresolved_sanity_pairs"] = unresolved
        cohort["interpretation"] = ("do_not_interpret_failed_sanity" if failed else
                                    "main_cosine_unresolved" if main["status"] != "resolved" else
                                    "conditional_sanity_unresolved" if unresolved else "sanity_checked")
        result.append(cohort)
        print(json.dumps({"point": point, "K": data.k, "main": main,
                          "sanity_status": cohort["sanity_status"], "failed": failed,
                          "unresolved": unresolved}, sort_keys=True), flush=True)
        del data
    return result


def fmt(value, digits=4):
    return "N/A" if value is None else ("%." + str(digits) + "g") % value


def cell(metric):
    if metric["status"] != "resolved":
        return "unresolved"
    return "%s [%s, %s]" % (fmt(metric["value"]), fmt(metric["low"]), fmt(metric["high"]))


def write_tables(cohorts, output):
    rows = []
    for point in cohorts:
        for a, b in PAIRS:
            metric = point["pairs"][a + "_" + b]
            row = {key: point[key] for key in ("point", "path_method", "updates", "own_critic", "K", "m", "groups",
                                               "source_checkpoint", "source_commit", "V2_definition", "warmup_batches",
                                               "normalization", "c", "c_source", "sanity_status", "interpretation")}
            row.update({key: metric[key] for key in ("pair", "status", "value", "se", "low", "high", "df", "critical",
                                                    "U_first_first", "U_second_second", "U_first_second")})
            row["invalid_replicates"] = ";".join(x["replicate"] for x in metric["failures"])
            row["group_ids"] = ";".join(point["group_ids"])
            row["group_paths"] = ";".join(point["group_paths"])
            rows.append(row)
    with (output / "alignment_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Existing-gradient alignment", "",
             "Cosines compare Euclidean mean ascent directions. Brackets are 90% delete-one-group intervals: "
             "ten independent 50-batch groups and t₉, except the preserved nine-group warmed checkpoint with t₈. "
             "Numerators and both squared-norm denominators use the same paired-batch U-statistic, excluding same-batch products. "
             "A nonpositive denominator in any full/deleted sample makes the cosine unresolved; values above 1 are retained as noisy estimates.", "",
             "| Point | Path | Update | K | Critic | c(V2,V6) | c(V2,V4) | c(V2,V5) | c(V4,V6) | c(V5,V6) | Common-mean checks |",
             "|---|---|---:|---:|---|---|---|---|---|---|---|"]
    for point in cohorts:
        cells = [point["point"], point["path_method"], str(point["updates"]), str(point["K"]),
                 "own checkpoint" if point["own_critic"] else "warmed %d batches" % point["warmup_batches"]]
        cells += [cell(point["pairs"][a + "_" + b]) for a, b in PAIRS]
        cells += [point["sanity_status"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "**Limits.** V2 at the five own-critic points is raw MC return minus cV, normalized over the full batch. "
              "The older warmed cohorts use MC of scaled/clipped rewards minus V; those are distinct proxies and are not pooled. "
              "Training normalized each 50,000-pair chunk; the authorized two-batch proxy check had cosines 0.999991 and 0.999994, "
              "and quarter/full norm ratios 1.000176 and 0.999930. The final own-critic point uses a five-batch fitted c = 438.775862; "
              "its interval conditions on that scale. AdamW preconditions gradients, so Euclidean alignment is an indicator, not an exact trajectory prediction.", "",
              "The CSV lists checkpoint paths, source revisions, every selected group and invalid deletion. "
              "A failed common-mean interval excludes 1 and blocks interpretation of the main cosine; an unresolved check does not establish agreement."]
    (output / "alignment_table.md").write_text("\n".join(lines) + "\n")


def plot(cohorts, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    own = [row for row in cohorts if row["own_critic"]]
    values = [row["pairs"]["V2_V6"]["value"] if row["sanity_status"] != "failed" else None for row in own]
    finite = [x for x in values if x is not None]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot([row["updates"] for row in own], [np.nan if x is None else x for x in values],
            color="#009E73", marker="o", ms=4, lw=1.1)
    ax.set_xlabel("Updates (one update = 80 episodes)")
    ax.set_ylabel("Cosine of mean gradients (unitless)")
    ax.set_ylim(0, max(1.0, max(finite, default=1.0) * 1.06))
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    caption = ("Cosine between the mean gradient of without_PPO_and_GAE (no γ^k) and ∇J_H "
               "(the paper's estimator with the same critic baseline), at the run's own checkpoints, 500 batches per point. "
               "Points are estimates; 90% intervals are in alignment_table. Estimates above 1 are retained, so the approved "
               "vertical range extends above 1; they reflect estimation uncertainty, not population cosines. "
               "Unresolved cosines or failed common-mean checks are omitted. The final scale was fitted; full-batch normalization is a checked training proxy.")
    fig.savefig(output / "e1cg_alignment.png", dpi=300, bbox_inches="tight", metadata={"Description": caption})
    fig.savefig(output / "e1cg_alignment.pdf", bbox_inches="tight", metadata={"Subject": caption, "Title": "Mean-gradient alignment"})
    plt.close(fig)
    (output / "e1cg_alignment.caption.md").write_text(caption + "\n")


def interim(cohorts, output):
    own = [row for row in cohorts if row["own_critic"]]
    numbers = "; ".join("update %d: %s" % (row["updates"], cell(row["pairs"]["V2_V6"])) for row in own)
    unresolved = [row["point"] for row in cohorts if row["pairs"]["V2_V6"]["status"] != "resolved"]
    failed = [row["point"] for row in cohorts if row["sanity_status"] == "failed"]
    unverified = [row["point"] for row in cohorts if row["sanity_status"] == "unresolved"]
    note = ["# Part A: mean-gradient alignment", "",
            "The existing data do not establish the ≥0.9 alignment premise at every own-critic checkpoint. "
            "The update-0 point is below that threshold, and all five intervals are wide enough to include 1. "
            "Thus this measurement gives no firm prediction that the two learning curves will retrace each other.", "",
            numbers + ".", "",
            "**Checks:** unresolved main cosines: %s. Common-mean intervals excluding 1: %s. "
            "Unresolved common-mean checks: %s. A failed check blocks interpretation at that point; unresolved checks leave the sanity test inconclusive." %
            (", ".join(unresolved) or "none", ", ".join(failed) or "none", ", ".join(unverified) or "none"), "",
            "The [table](alignment_table.md) separates all eleven paired cohorts and records estimator differences, "
            "normalization and fitted-scale limitations. [Figure and caption](e1cg_alignment.caption.md). "
            "No new sampling or training was performed. The approved display retains estimates above 1; the nine-group cohort uses t₈."]
    (output / "INTERIM.md").write_text("\n".join(note) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-source", type=Path)
    parser.add_argument("--statistics-only", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    source = (args.analysis_source or project / ".codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.render_existing:
        record = read(output / "alignment_record.json")
        cohorts = record["cohorts"]
    else:
        cohorts = analyze(project, source)
        revision = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], universal_newlines=True).strip()
        record = {"created_utc": datetime.now(timezone.utc).isoformat(), "project_revision": revision,
                  "command": sys.argv, "analysis_source": str(source), "code": str(Path(__file__).resolve()),
                  "estimator": "(sum-gradient cross product minus sum same-batch cross products)/(K*(K-1)); same formula for numerator and both denominators",
                  "interval": "paired delete-one-50-batch-group jackknife; Student t(0.95,G-1)",
                  "cohorts": cohorts}
        (output / "alignment_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    write_tables(cohorts, output)
    if not args.statistics_only:
        plot(cohorts, output)
        interim(cohorts, output)


if __name__ == "__main__":
    main()
