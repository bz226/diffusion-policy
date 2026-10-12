"""Read-only, equal-sample statistics for clean figures version 3.

This module does not change the collector or its historical analysis. It uses
the verified V6 PointData reader, then recomputes every displayed statistic on
the same earliest-launched complete V6 groups with Student-t jackknife limits.
Python 3.8; no simulations, checkpoint writes, or global RNG draws.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import t as student_t


POINTS = ("P0", "D1", "D2", "N1", "N2", "C1", "C2")
VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6")
NATIVE_POINT = {"C1": "D1", "C2": "D2"}
EPISODES = 80
GROUP_SIZE = 50
THEORY = 2500.0


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _load_core(root, analysis_source=None):
    project = Path(root).resolve().parents[2]
    source = Path(analysis_source).resolve() if analysis_source else (
        project / ".codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean")
    if not (source / "analysis_v6.py").is_file():
        raise ValueError("The approved V6 analysis source is missing: " + str(source))
    sys.path.insert(0, str(source))
    core = importlib.import_module("analysis_core")
    if Path(core.__file__).resolve().parent != source:
        raise ValueError("analysis_core was already imported from a different source; use a fresh process")
    return core, source


def jackknife(full, deleted):
    """Literal delete-one-group t interval, without clipping/bias correction."""
    values = [float(x) if _finite(x) else None for x in deleted]
    result = {"value": float(full) if _finite(full) else None,
              "lo": None, "hi": None, "se": None, "status": "unavailable",
              "groups": len(values), "df": len(values) - 1 if values else None,
              "critical": float(student_t.ppf(.95, len(values) - 1)) if len(values) >= 2 else None,
              "reason": None}
    if result["value"] is None:
        result["reason"] = "full_sample_statistic_undefined"
    elif len(values) < 2 or any(x is None for x in values):
        result["status"] = "estimate_without_interval"
        result["reason"] = "insufficient_or_undefined_delete_group_replicate"
    else:
        replicas = np.asarray(values, dtype=np.float64)
        se = math.sqrt((len(values) - 1.0) / len(values) *
                       np.square(replicas - replicas.mean()).sum(dtype=np.float64))
        result.update(se=se, lo=result["value"] - result["critical"] * se,
                      hi=result["value"] + result["critical"] * se, status="point")
    return result


def _scaled(result, factor):
    result = dict(result)
    for field in ("value", "lo", "hi", "se", "lower_bound", "raw_value"):
        if _finite(result.get(field)):
            result[field] = float(result[field]) * factor
    return result


def _scalar(data, operation):
    return jackknife(operation(data.full_moments),
                     [operation(moment) for moment in data.deleted_moments])


def _signal(data, variant):
    index = data.variants.index(variant)
    return _scalar(data, lambda m: float(m["signal"][index]))


def _norm(data, variant):
    index = data.variants.index(variant)
    return _scalar(data, lambda m: math.sqrt(m["signal"][index])
                   if m["signal"][index] > 0 else None)


def _noise_scale(data, variant, signal_variant):
    numerator = data.variants.index(variant)
    denominator = data.variants.index(signal_variant)
    signal = _signal(data, signal_variant)
    result = _scalar(data, lambda m: float(EPISODES * m["trace"][numerator] /
                                          m["signal"][denominator])
                     if m["signal"][denominator] > 0 and m["trace"][numerator] >= 0 else None)
    result.update(signal_estimator=signal_variant, variance_estimator=variant,
                  lower_bound=None, bound_is_joint_confidence_limit=False)
    if signal["status"] != "point":
        result.update(raw_value=result["value"], value=None, lo=None, hi=None, se=None,
                      status="unavailable", reason="signal_jackknife_interval_unavailable")
    elif signal["value"] <= 0 or signal["lo"] <= 0:
        upper = signal["hi"]
        trace = float(data.full_moments["trace"][numerator])
        bound = EPISODES * trace / upper if upper > 0 and trace >= 0 else None
        result.update(raw_value=result["value"], value=None, lo=None, hi=None, se=None,
                      status="lower_bound" if bound is not None else "unavailable",
                      lower_bound=bound, reason="signal_nonpositive_or_interval_includes_zero")
    return result


def _cosine(data, first, second):
    a, b = data.variants.index(first), data.variants.index(second)
    def operation(moment):
        left, right = moment["signal"][a], moment["signal"][b]
        return float(moment["cross_signal"][a, b] / math.sqrt(left * right)) if left > 0 and right > 0 else None
    result = _scalar(data, operation)
    result["same_batch_pairs_excluded"] = True
    result["clipped"] = False
    return result


def _reward(data, field):
    values = np.asarray(data.group_rewards[field], dtype=np.float64)
    return jackknife(float(values.mean()),
                     [float(np.delete(values, i).mean()) for i in range(len(values))]
                     if len(values) > 1 else [])


def statistics_from_data(data):
    """Public same-cohort metrics; useful independently for validation."""
    result = {"K": data.k, "m": EPISODES * data.k, "groups": data.n,
              "metrics": {"J_disc": _reward(data, "J_disc"), "return": _reward(data, "return"),
                          "u_sq": _scaled(_signal(data, "V4"), THEORY ** 2),
                          "u_norm": _scaled(_norm(data, "V4"), THEORY)},
              "estimators": {}, "trained": {}, "variance_ratios": {}}
    for variant in ("V4", "V5", "V6"):
        index = data.variants.index(variant)
        scale = _noise_scale(data, variant, "V4")
        result["estimators"][variant] = {
            "trace_ep": _scaled(_scalar(data, lambda m, i=index: float(m["trace"][i])), EPISODES * THEORY ** 2),
            "B_ep": scale, "B_update": _scaled(scale, 1.0 / EPISODES)}
    for variant in ("V1", "V2"):
        result["trained"][variant] = {"B_ep": _noise_scale(data, variant, variant),
                                      "cos_V4": _cosine(data, variant, "V4")}
    for name, first, second in (("V6_over_V4", "V6", "V4"), ("V5_over_V6", "V5", "V6")):
        a, b = data.variants.index(first), data.variants.index(second)
        result["variance_ratios"][name] = _scalar(data, lambda m, a=a, b=b:
            float(m["trace"][a] / m["trace"][b]) if m["trace"][b] > 0 else None)
    return result


def _clean_exit(record):
    workers = record.get("workers", [])
    return (record.get("status") == "passed" and record.get("all_children_exited") is True
            and record.get("remaining_active_children") == [] and record.get("remaining_direct_children") == []
            and record.get("errors") == [] and record.get("primary_error") is None
            and len(workers) == 40 and all(w.get("exitcode") == 0 and w.get("alive") is False for w in workers))


def _launch_key(record):
    value = record.get("started_utc")
    if not value:
        raise ValueError("Missing actual launch timestamp for " + record["run_id"])
    return (datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp(), record["job_id"], record["run_id"])


def discover_groups(root, core, c_root=None):
    """Verify native, worker, scheduler and cleanup records before selection."""
    root = Path(root)
    c_root = Path(c_root) if c_root else root / "runs/clean_v3_collection_20261012"
    campaigns = {False: _read(root / "runs/campaign.json"), True: _read(c_root / "campaign.json")}
    entries = {}
    for is_c, campaign in campaigns.items():
        records = campaign.get("submitted", campaign.get("entries", []))
        if isinstance(records, dict):
            records = list(records.values())
        entries[is_c] = {entry["run_id"]: entry for entry in records}
    inventory, included = [], {point: [] for point in POINTS}
    for directory in sorted((root / "collect").glob("*")):
        match = re.fullmatch(r"(P0|D1|D2|N1|N2|C1|C2)_j(\d+)(_retry1)?", directory.name)
        if not match or not directory.is_dir():
            continue
        point, group = match[1], int(match[2])
        is_c = point.startswith("C")
        meta = _read(directory / "meta.json")
        entry = entries[is_c].get(directory.name, {})
        manifest_path = ((c_root / "workers") if is_c else root / "runs") / directory.name / "manifest.json"
        manifest = _read(manifest_path)
        cleanup = _read(directory / "cleanup.json")
        recovery = (not is_c and core.verified_administrative_recovery(root, directory.name, entry, manifest))
        worker = (manifest.get("status") == "complete" and manifest.get("subprocess_exitcode") == 0
                  and manifest.get("remaining_process_group_members") == []) or recovery
        terminal_status = (entry.get("status") == "complete" if is_c else
                           entry.get("terminal_recorded") is True and entry.get("run_status") == "complete")
        terminal = (terminal_status and entry.get("scheduler_state") == "COMPLETED"
                    and entry.get("exit_code") == "0:0") or recovery
        native = (meta.get("status") == "completed" and meta.get("completed_batches") == GROUP_SIZE
                  and meta.get("point") == NATIVE_POINT.get(point, point))
        alias_valid = (not is_c or (manifest.get("external_point_id") == point and
                                   manifest.get("collector_point") == NATIVE_POINT[point]))
        reasons = [reason for reason, passed in (("native_not_complete", native), ("worker_not_complete", worker),
                   ("scheduler_not_complete", terminal), ("cleanup_not_complete", _clean_exit(cleanup)),
                   ("collector_alias_mismatch", alias_valid)) if not passed]
        record = {"point": point, "collector_point": NATIVE_POINT.get(point, point), "group": group,
                  "run_id": directory.name, "job_id": str(entry.get("job_id", "")),
                  "seed": meta.get("seed", manifest.get("seed")), "started_utc": manifest.get("started_utc"),
                  "submitted_utc": entry.get("submitted_utc"), "path": str(directory.resolve()),
                  "manifest": str(manifest_path.resolve()), "variants": meta.get("variant_ids", []),
                  "included": not reasons, "exclusion_reasons": reasons,
                  "source_commit": meta.get("parent_commit"), "frozen_sha256": meta.get("frozen_sha256")}
        inventory.append(record)
        if not reasons:
            if any(item["group"] == group for item in included[point]):
                raise ValueError("Duplicate successful group/replacement for " + point + " group " + str(group))
            included[point].append(record)
    for point, records in included.items():
        records.sort(key=_launch_key)
    return included, inventory


def build_statistics(root, c_root=None, analysis_source=None, batches=500):
    """Return JSON-safe point records; incomplete C points are omitted.

    Each point contains metrics, estimators[V4/V5/V6], trained[V1/V2],
    variance_ratios, selected_jobs, and separate reference_V4 from all complete groups.
    Metric fields are value/lo/hi/se/status; unresolved B has lower_bound.
    """
    root = Path(root).resolve()
    if batches != 500:
        raise ValueError("Clean figures v3 specifies a fixed 500-batch cohort")
    core, source = _load_core(root, analysis_source)
    groups, inventory = discover_groups(root, core, c_root)
    output = {"points": {}, "availability": {}, "inventory": inventory,
              "selection": "earliest actual worker launches; completed V6-enabled 50-batch groups only",
              "analysis_source": str(source), "K": batches, "m": EPISODES * batches,
              "interval": {"level": .90, "method": "delete-one-group Student-t", "groups": batches // GROUP_SIZE,
                           "df": batches // GROUP_SIZE - 1,
                           "critical": float(student_t.ppf(.95, batches // GROUP_SIZE - 1)),
                           "bound_note": "Unresolved B is trace divided by the upper signal limit; this plug-in lower bound is not a joint confidence limit."}}
    for point in POINTS:
        records = groups[point]
        eligible = [record for record in records if tuple(record["variants"]) == VARIANTS]
        output["availability"][point] = {"complete_groups": len(records), "V6_complete_groups": len(eligible),
                                           "V6_complete_batches": len(eligible) * GROUP_SIZE,
                                           "status": "complete" if len(eligible) >= batches // GROUP_SIZE else "pending"}
        if len(eligible) < batches // GROUP_SIZE:
            if not point.startswith("C"):
                raise ValueError(point + " lacks the already-completed 500 V6 batches required by v3")
            continue
        selected = eligible[:batches // GROUP_SIZE]
        native = NATIVE_POINT.get(point, point)
        data = core.PointData(native, [record["path"] for record in selected], variants=VARIANTS)
        result = statistics_from_data(data)
        result.update(point=point, collector_point=native, selected_jobs=selected,
                      trained_estimator="V2" if point.startswith("C") else "V1",
                      source_checkpoint=data.metas[0].get("source_checkpoint"), frozen_sha256=data.metas[0]["frozen_sha256"])
        del data
        # Load only V4 for the all-data reference, keeping peak memory bounded.
        reference = core.PointData(native, [record["path"] for record in records], variants=("V4",))
        result["reference_V4"] = {"K": reference.k, "m": reference.k * EPISODES, "groups": reference.n,
                                  "u_norm": _scaled(_norm(reference, "V4"), THEORY),
                                  "u_sq": _scaled(_signal(reference, "V4"), THEORY ** 2),
                                  "B_ep": _noise_scale(reference, "V4", "V4"),
                                  "jobs": [record["run_id"] for record in records],
                                  "used_in_main_figures": False}
        del reference
        output["points"][point] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--c-root", type=Path)
    parser.add_argument("--analysis-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_statistics(args.root, args.c_root, args.analysis_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"points": list(result["points"]), "availability": result["availability"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
