"""Own-critic, equal-cohort clean-v5 statistics; Python 3.8, read-only.

The measured V2 and V6 each use their own pairwise signal estimate. Every
reported point uses ten verified 50-batch groups; incomplete points are omitted.
No rollouts, fitted baselines, parameter changes, or changes to older analyses.
"""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from clean_v3_stats import (_clean_exit, _cosine, _finite, _launch_key, _load_core,
                            _noise_scale, _norm, _read, _reward, _scalar, _signal,
                            jackknife)


POINTS = ("W0_000", "W0_035", "W0_070", "W0_105", "W0_139")
NATIVE_POINT = dict(zip(POINTS, ("P0", "D1", "D2", "N1", "N2")))
CHECKPOINT_ITR = dict(zip(POINTS, (0, 35, 70, 105, 139)))
VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6")
EPISODES = 80
GROUP_SIZE = 50
BATCHES = 500
THEORY = 2500.0


def _scaled(metric, factor):
    output = dict(metric)
    for key in ("value", "lo", "hi", "se", "lower_bound", "upper_bound", "raw_value"):
        if _finite(output.get(key)):
            output[key] = float(output[key]) * factor
    return output


def _point_norm(data, variant):
    signal, norm = _signal(data, variant), _norm(data, variant)
    norm["signal_resolved"] = (signal.get("lo") is not None and signal["lo"] > 0)
    if norm["value"] is None and _finite(signal.get("hi")) and signal["hi"] > 0:
        norm["upper_bound"] = math.sqrt(signal["hi"])
        norm["status"] = "upper_bound"
        norm["reason"] = "nonpositive_own_signal; square root of upper signal limit"
    return norm


def predicted_consecutive_cos(data, variant="V2"):
    """1/(1+B_ep/80), with joint deletions rather than transformed error bars."""
    index = data.variants.index(variant)
    scale = _noise_scale(data, variant, variant)
    def operation(moment):
        signal, trace = float(moment["signal"][index]), float(moment["trace"][index])
        return signal / (signal + trace) if signal > 0 and trace >= 0 else None
    result = _scalar(data, operation)
    result.update(definition="1 / (1 + B_V2 / 80)", estimator=variant,
                  bound_is_joint_confidence_limit=False)
    if scale["status"] == "lower_bound":
        result.update(raw_value=result["value"], value=None, lo=None, hi=None, se=None,
                      status="upper_bound", upper_bound=1.0 / (1.0 + scale["lower_bound"] / EPISODES),
                      reason="own signal unresolved; transform of plug-in lower bound on B")
    elif scale["value"] is None:
        result.update(raw_value=result["value"], value=None, lo=None, hi=None, se=None,
                      status="unavailable", reason=scale.get("reason"))
    return result


def _group_scalar(data, field):
    means = []
    for rows in data.batches:
        values = [float(row[field]) for row in rows if field in row and row[field] not in (None, "")]
        if len(values) != GROUP_SIZE or not np.isfinite(values).all():
            return {"value": None, "lo": None, "hi": None, "se": None, "status": "unavailable",
                    "reason": "missing_or_nonfinite_batch_" + field}
        means.append(math.fsum(values) / len(values))
    return jackknife(math.fsum(means) / len(means),
                     [math.fsum(means[:i] + means[i + 1:]) / (len(means) - 1)
                      for i in range(len(means))] if len(means) > 1 else [])


def statistics_from_data(data):
    """Public pure-cohort calculation; data is the existing PointData reader."""
    result = {"K": data.k, "m": data.k * EPISODES, "groups": data.n,
              "estimators": {}, "cos_V2_V6": _cosine(data, "V2", "V6"),
              "explained_var": _group_scalar(data, "explained_variance"),
              "J_disc": _reward(data, "J_disc"), "return": _reward(data, "return"),
              "predicted_consecutive_cos": predicted_consecutive_cos(data)}
    for variant, multiplier in (("V2", 1.0), ("V6", THEORY)):
        index = data.variants.index(variant)
        scale = _noise_scale(data, variant, variant)
        result["estimators"][variant] = {
            "u_sq": _scaled(_signal(data, variant), multiplier ** 2),
            "u_norm": _scaled(_point_norm(data, variant), multiplier),
            "trace_ep": _scaled(_scalar(data, lambda moment, i=index: float(moment["trace"][i])),
                                EPISODES * multiplier ** 2),
            "B_ep": scale, "B_update": _scaled(scale, 1.0 / EPISODES),
            "signal_estimator": variant, "gradient_multiplier": multiplier,
            "gradient_units": "V2 code units with selected advantage normalization" if variant == "V2" else "paper units",
        }
    result["explained_var"]["definition"] = "mean of collection batches' explained_variance, jackknifed over whole50-batch groups"
    return result


def _completed_record(entry, manifest, cleanup):
    return (entry.get("status") == "complete" and entry.get("scheduler_state") == "COMPLETED"
            and entry.get("exit_code") == "0:0" and manifest.get("status") == "complete"
            and manifest.get("subprocess_exitcode") == 0
            and manifest.get("remaining_process_group_members") == [] and _clean_exit(cleanup))


def _entries(campaign):
    records = campaign.get("entries", [])
    if isinstance(records, dict):
        records = list(records.values())
    return {entry["run_id"]: entry for entry in records}


def _validate_spec_metadata(run, run_id, meta, checkpoint, point, normalization):
    spec = _read(run / "specs" / (run_id + ".json"))
    settings = meta.get("settings", {})
    expected = spec.get("expected_settings", {})
    if (not expected or any(settings.get(key) != value for key, value in expected.items())
            or spec.get("point") != point or spec.get("collector_point") != NATIVE_POINT[point]
            or spec.get("checkpoint") != checkpoint["checkpoint_path"]
            or meta.get("source_checkpoint") != checkpoint["checkpoint_path"]
            or meta.get("point") != NATIVE_POINT[point]):
        raise ValueError("Wrong own-checkpoint source or settings: " + run_id)
    if (type(settings.get("gm_warmup")) is not int or settings["gm_warmup"] != 0
            or settings.get("gm_v2_raw") is not True or settings.get("gm_include_critic") is not True
            or settings.get("gm_normalization") != normalization
            or settings.get("gm_scaler_c") != checkpoint["divisor"]):
        raise ValueError("W=0/raw-return V2/normalization/divisor contract differs: " + run_id)
    if (not spec.get("parent_commit") or meta.get("parent_commit") != spec["parent_commit"]
            or settings.get("gm_parent_commit") != spec["parent_commit"]):
        raise ValueError("Missing or inconsistent scientific source revision: " + run_id)
    if meta.get("actor_optimizer_steps") != 0 or meta.get("critic_optimizer_steps") != 0:
        raise ValueError("Own-critic measurements must not perform optimizer steps: " + run_id)
    return spec


def _prepare_metadata(run, point, checkpoint, entries):
    name = "prepare_" + point
    directory = run / "prepare" / point
    meta = _read(directory / "grad_mean.json")
    manifest = _read(run / "workers" / name / "manifest.json")
    if not _completed_record(entries.get(name, {}), manifest, _read(directory / "cleanup.json")):
        raise ValueError("Complete groups lack a terminal, clean own-critic preparation: " + point)
    spec = _validate_spec_metadata(run, name, meta, checkpoint, point, "full")
    if meta.get("status") != "completed" or meta.get("completed_warmup_batches") != 0 or meta.get("warmup") != []:
        raise ValueError("Critic warmup occurred or preparation is incomplete: " + point)
    for key in ("actor", "critic"):
        if not meta.get(key + "_before_sha256") or meta[key + "_before_sha256"] != meta.get(key + "_after_sha256"):
            raise ValueError("Own checkpoint " + key + " changed during preparation")
    frozen_path = directory / ("frozen_" + NATIVE_POINT[point] + ".pt")
    if (meta.get("frozen_path") != str(frozen_path) or not frozen_path.is_file()
            or not meta.get("frozen_sha256") or not meta.get("scaler_after_sha256")):
        raise ValueError("Frozen artifact is missing or aliases another preparation")
    effective = meta.get("effective_scaler_c")
    if not _finite(effective) or effective <= 0:
        raise ValueError("Invalid frozen raw-reward divisor")
    fallback = checkpoint["divisor"] is None
    if (meta.get("scaler_c_source") != ("least_squares_five_batches" if fallback else "provided")
            or meta.get("scaler_c_fallback_batches") != (5 if fallback else 0)):
        raise ValueError("Divisor source differs from the approved checkpoint-specific rule")
    fit = meta.get("scaler_fit_batches", [])
    if fallback:
        if len(fit) != 5:
            raise ValueError("Final divisor fit must use exactly five saved batch records")
        numerator = math.fsum(float(row["sum_VG"]) for row in fit)
        denominator = math.fsum(float(row["sum_V2"]) for row in fit)
        if denominator <= 0 or not math.isclose(effective, numerator / denominator, rel_tol=1e-12):
            raise ValueError("Final fitted divisor does not match its saved sufficient statistics")
    elif fit or not math.isclose(effective, checkpoint["divisor"], rel_tol=2e-15):
        raise ValueError("Provided divisor differs from the exact logged value")
    return meta, spec


def _validate_cohort(run, point, records, checkpoint, entries, normalization):
    prepared, preparation_spec = _prepare_metadata(run, point, checkpoint, entries)
    for record in records:
        meta = _read(Path(record["path"]) / "meta.json")
        spec = _validate_spec_metadata(run, record["run_id"], meta, checkpoint, point,
                                       normalization["selected_normalization"])
        if (spec["parent_commit"] != normalization["source_commit"]
                or spec["parent_commit"] != preparation_spec["parent_commit"]
                or meta.get("frozen_sha256") != prepared["frozen_sha256"]
                or meta.get("frozen_path") != prepared["frozen_path"]):
            raise ValueError("Collect source/frozen state differs from its approved preparation")
        for key in ("actor", "critic", "scaler"):
            before, after = meta.get(key + "_before_sha256"), meta.get(key + "_after_sha256")
            if not before or before != after or after != prepared.get(key + "_after_sha256"):
                raise ValueError("Collect " + key + " differs from its frozen preparation")
        record.update(source_commit=spec["parent_commit"], frozen_sha256=meta["frozen_sha256"],
                      normalization=normalization["selected_normalization"])
    return prepared


def _validate_gate_provenance(run, campaign, normalization):
    provenance = campaign.get("source_provenance", {})
    gate_path = run / "gate_cpu/gate_summary.json"
    gate = _read(gate_path)
    if (provenance.get("gate_path") != str(gate_path) or gate.get("passed") is not True
            or gate.get("device") != "cpu"
            or any(gate.get(name, {}).get("passed") is not True for name in ("F1", "F2", "F3"))
            or str(provenance.get("cpu_gate_job_id")) != str(gate.get("slurm_job_id"))
            or provenance.get("source_commit") != normalization.get("source_commit")):
        raise ValueError("Completed collection lacks the approved CPU gate/source provenance")


def build_statistics(root, analysis_source=None):
    """Read verified clean-v5 outputs; each pending point stays omitted.

    See return keys points, availability, inventory and runmetadata. A native
    completed file alone never admits an allocation without terminal records.
    """
    root = Path(root).resolve()
    run = root / "runs/clean_v5_20261012"
    core, source = _load_core(root, analysis_source)
    audit = _read(run / "log_audit.json")
    if audit.get("version") != 5:
        raise ValueError("Verified original-run clean-v5 log audit is required")
    campaign = _read(run / "campaign.json")
    entries = _entries(campaign)
    normalization = _read(run / "normalization_selection.json")
    groups, inventory = {point: [] for point in POINTS}, []
    for path in sorted((run / "collect").glob("*/meta.json")):
        match = re.fullmatch(r"(W0_000|W0_035|W0_070|W0_105|W0_139)_j(\d+)(?:_retry1)?", path.parent.name)
        if match is None:
            continue
        point, group = match.group(1), int(match.group(2))
        meta, entry = _read(path), entries.get(path.parent.name, {})
        manifest_path = run / "workers" / path.parent.name / "manifest.json"
        manifest, cleanup = _read(manifest_path), _read(path.parent / "cleanup.json")
        native = (meta.get("status") == "completed" and meta.get("completed_batches") == GROUP_SIZE
                  and meta.get("point") == NATIVE_POINT[point]
                  and tuple(meta.get("variant_ids", [])) == VARIANTS)
        record = {"point": point, "collector_point": NATIVE_POINT[point], "group": group,
                  "run_id": path.parent.name, "job_id": str(entry.get("job_id", "")),
                  "seed": meta.get("seed"), "started_utc": manifest.get("started_utc"),
                  "path": str(path.parent), "manifest": str(manifest_path)}
        reasons = []
        if not native:
            reasons.append("native_not_complete_or_wrong_variants")
        if not _completed_record(entry, manifest, cleanup):
            reasons.append("worker_scheduler_or_cleanup_not_verified_complete")
        if manifest.get("external_point_id") != point or manifest.get("collector_point") != NATIVE_POINT[point]:
            reasons.append("collector_alias_mismatch")
        record.update(included=not reasons, exclusion_reasons=reasons)
        inventory.append(record)
        if not reasons:
            groups[point].append(record)
    output = {"points": {}, "availability": {}, "inventory": inventory,
              "K": BATCHES, "m": BATCHES * EPISODES,
              "normalization": normalization.get("selected_normalization"),
              "interval": {"level": .90, "method": "delete-one-50-batch-group Student-t",
                           "groups": 10, "df": 9, "critical": jackknife(1.0, [1.0] * 10)["critical"],
                           "bound_note": "Unresolved B uses trace divided by the upper own-signal limit; a plug-in bound, not a joint confidence limit."},
              "runmetadata": {"campaign": str(run / "campaign.json"), "status": campaign.get("status", "not_started"),
                              "analysis_source": str(source), "log_audit": str(run / "log_audit.json"),
                              "normalization_selection": normalization,
                              "normalization": normalization.get("selected_normalization"),
                              "source_provenance": campaign.get("source_provenance", {}),
                              "scientific_sampling_performed_by_analysis": False}}
    for point in POINTS:
        records = sorted(groups[point], key=_launch_key)
        if len({record["group"] for record in records}) != len(records):
            raise ValueError("Duplicate successful group: " + point)
        if len(records) > 10:
            raise ValueError("Clean v5 permits exactly ten successful collect groups per point")
        output["availability"][point] = {"complete_groups": len(records), "complete_batches": len(records) * GROUP_SIZE,
                                         "status": "complete" if len(records) == 10 else "pending"}
        if len(records) != 10:
            continue
        if (normalization.get("selected_normalization") not in ("full", "quarter")
                or normalization.get("frozen_before_first_collect") is not True):
            raise ValueError("The production normalization selection was not frozen")
        _validate_gate_provenance(run, campaign, normalization)
        checkpoint = audit["checkpoints"][str(CHECKPOINT_ITR[point])]
        prepared = _validate_cohort(run, point, records, checkpoint, entries, normalization)
        data = core.PointData(NATIVE_POINT[point], [record["path"] for record in records], variants=VARIANTS)
        if any(float(row.get("frozen_reward_divisor", "nan")) != prepared["effective_scaler_c"]
               for batch in data.batches for row in batch):
            raise ValueError("A measured batch used a different frozen reward divisor")
        result = statistics_from_data(data)
        result.update(point=point, collector_point=NATIVE_POINT[point], selected_jobs=records,
                      checkpoint_itr=checkpoint["checkpoint_itr"], update_number=checkpoint["update_number"],
                      training_env_steps=checkpoint["training_env_steps"], checkpoint_path=checkpoint["checkpoint_path"],
                      logged_consecutive_cos=dict(checkpoint["grad_cos_prev_window"]),
                      logged_explained_var=dict(checkpoint["logged_explained_var"], lo=None, hi=None, se=None),
                      divisor_source={"effective_divisor": prepared["effective_scaler_c"],
                                      "source": prepared["scaler_c_source"],
                                      "fallback_batches": prepared["scaler_c_fallback_batches"],
                                      "prepare_metadata": str(run / "prepare" / point / "grad_mean.json"),
                                      "original_log_source": checkpoint["divisor_source"]},
                      divisor={"value": prepared["effective_scaler_c"], "lo": None, "hi": None, "se": None,
                               "status": "fixed_fitted" if prepared["scaler_c_fallback_batches"] else "fixed_provided",
                               "reason": "Measurement intervals condition on this fixed scale; its fitting uncertainty is not included."},
                      frozen_sha256=data.metas[0]["frozen_sha256"],
                      normalization=normalization["selected_normalization"])
        logged = result["logged_consecutive_cos"]
        logged.update(value=logged["mean"], se=logged["sample_se"], lo=None, hi=None, status="logged_window_mean")
        output["points"][point] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_statistics(args.root, args.analysis_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"points": list(output["points"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
