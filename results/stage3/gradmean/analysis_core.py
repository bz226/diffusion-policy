"""Read-only analysis of approved Addendum G group records.

The statistical unit is an independent whole batch; uncertainty deletes entire
50-batch groups. No trajectory generation, optimizer, or checkpoint mutation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import numpy as np

from gradmean_math import (VARIANTS, independent_group_jackknife,
                           jackknife_interval, summarize_gram)


POINTS = ("P0", "D1", "D2", "N1", "N2")
THEORY_SCALE = 2500.
GROUP_SIZE = 50
EPISODES = 80
ROUND_BATCHES = (500, 1400, 2300, 3200, 4100)


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value):
    return value is not None and math.isfinite(float(value))


def cosine(dot, first_sq, second_sq):
    return float(dot / math.sqrt(first_sq * second_sq)) if first_sq > 0 and second_sq > 0 else None


def jk_scalar(full, deleted):
    return {"value": full, **jackknife_interval(full, deleted)}


def group_mean(values):
    values = np.asarray(values, dtype=np.float64)
    full = float(values.mean())
    deleted = [float(np.delete(values, i).mean()) for i in range(len(values))] if len(values) > 1 else []
    return jk_scalar(full, deleted)


def independent_result(estimator, counts):
    if all(count >= 2 for count in counts.values()):
        return independent_group_jackknife(estimator, counts)
    full = estimator({name: np.arange(count) for name, count in counts.items()})
    return {"estimate": full, "jackknife90": {
        field: {"valid": False, "se": None, "lower": None, "upper": None,
                "reason": "At least two groups per independent policy are required."}
        for field in full}}


def interval_fields(prefix, result, scale=1.):
    return {prefix: None if result.get("value") is None else result["value"] * scale,
            prefix + "_se": None if result.get("se") is None else result["se"] * scale,
            prefix + "_low": None if result.get("lower") is None else result["lower"] * scale,
            prefix + "_high": None if result.get("upper") is None else result["upper"] * scale,
            prefix + "_uncertainty_reason": result.get("reason")}


class PointData:
    def __init__(self, point, directories):
        self.point = point
        self.paths = [Path(path) for path in directories]
        self.metas = [read_json(path / "meta.json") for path in self.paths]
        self.n = len(self.paths)
        self.k = self.n * GROUP_SIZE
        self.ids = np.arange(self.n)
        if not self.n:
            raise ValueError("PointData requires at least one complete group")
        for meta in self.metas:
            if meta["status"] != "completed" or meta["completed_batches"] != GROUP_SIZE or meta["point"] != point:
                raise ValueError("Incomplete/wrong-policy group included in " + point)
            if meta["group_vector_kind"] != "sum" or meta["theory_multiplier_V4_V5"] != 2500:
                raise ValueError("Gradient units/group-sum convention changed")
            if meta["episodes_per_batch"] != EPISODES or meta["pairs_per_batch"] != 200000:
                raise ValueError("Batch geometry changed")
            for name in ("actor", "critic", "scaler"):
                if meta[name + "_before_sha256"] != meta[name + "_after_sha256"]:
                    raise ValueError("Frozen " + name + " changed during collection")
        if len({meta["seed"] for meta in self.metas}) != self.n:
            raise ValueError("Duplicate measurement seed")
        if len({meta["frozen_sha256"] for meta in self.metas}) != 1:
            raise ValueError("Point groups used different frozen preparations")
        self.specs = self.metas[0]["parameter_specs"]
        if any(meta["parameter_specs"] != self.specs for meta in self.metas):
            raise ValueError("Actor parameter order differs across groups")
        self.p = int(self.metas[0]["parameter_count"])
        sums = []
        for variant in VARIANTS:
            vectors = [np.load(path / ("sums_" + variant + ".npy"), allow_pickle=False) for path in self.paths]
            if any(vector.shape != (self.p,) or vector.dtype != np.float32 or not np.isfinite(vector).all() for vector in vectors):
                raise ValueError("Invalid saved float32 group sum")
            sums.append(np.stack(vectors).astype(np.float64))
        self.sums = np.stack(sums)  # variant, group, coordinate; no hidden scale/sign change.
        self.q = np.asarray([meta["gradient_cross_sum"] for meta in self.metas], dtype=np.float64)
        if self.q.shape != (self.n, 5, 5) or not np.isfinite(self.q).all():
            raise ValueError("Invalid float64 within-batch cross moments")
        self.grams = np.stack([value @ value.T for value in self.sums])
        self.total = self.sums.sum(axis=1, dtype=np.float64)
        self.total_cross = self.total @ self.total.T
        self.group_total_cross = np.stack([value @ self.total.T for value in self.sums], axis=1)
        self.group_self_cross = np.einsum("vgp,wgp->gvw", self.sums, self.sums, optimize=True)
        self.q_total = self.q.sum(axis=0, dtype=np.float64)
        self.batches = [csv_rows(path / "batches.csv") for path in self.paths]
        if any(len(rows) != GROUP_SIZE for rows in self.batches):
            raise ValueError("Completed group does not contain 50 scalar batch rows")
        self.group_rewards = {}
        for field in ("J_disc", "return"):
            self.group_rewards[field] = np.array([
                sum(float(row[field + "_sum"]) for row in rows) /
                sum(int(row["n_episodes"]) for row in rows) for rows in self.batches])
        self.summaries = {variant: summarize_gram(self.grams[i], self.q[:, i, i])
                          for i, variant in enumerate(VARIANTS)}
        self.full_moments = self.moments(self.ids)
        self.deleted_moments = [self.moments(np.delete(self.ids, i)) for i in range(self.n)] if self.n > 1 else []

    def moments(self, ids):
        ids = np.asarray(ids, dtype=int)
        count = len(ids) * GROUP_SIZE
        if not len(ids):
            return None
        omitted = np.setdiff1d(self.ids, ids, assume_unique=True)
        if not len(omitted):
            total_cross, q = self.total_cross, self.q_total
        elif len(omitted) == 1:
            g = int(omitted[0])
            total_cross = self.total_cross - self.group_total_cross[g] - self.group_total_cross[g].T + self.group_self_cross[g]
            q = self.q_total - self.q[g]
        else:
            total = self.sums[:, ids].sum(axis=1, dtype=np.float64)
            total_cross, q = total @ total.T, self.q[ids].sum(axis=0, dtype=np.float64)
        cross_signal = (total_cross - q) / (count * (count - 1))
        traces = (np.diag(q) - np.diag(total_cross) / count) / (count - 1)
        return {"cross_signal": cross_signal, "signal": np.diag(cross_signal), "trace": traces, "K": count}

    def mean(self, variant, ids=None):
        v = VARIANTS.index(variant)
        if ids is None or len(ids) == self.n:
            return self.total[v] / self.k
        ids = np.asarray(ids, dtype=int)
        omitted = np.setdiff1d(self.ids, ids, assume_unique=True)
        total = self.total[v] - self.sums[v, omitted[0]] if len(omitted) == 1 else self.sums[v, ids].sum(axis=0)
        return total / (len(ids) * GROUP_SIZE)

    def reward(self, ids, field="J_disc"):
        return float(self.group_rewards[field][ids].mean())

    def scalar(self, estimator):
        full = estimator(self.full_moments)
        return jk_scalar(full, [estimator(moments) for moments in self.deleted_moments])

    def same_cosines(self):
        rows = []
        for v, first in enumerate(VARIANTS):
            for w, second in enumerate(VARIANTS):
                estimate = self.scalar(lambda moment: cosine(moment["cross_signal"][v, w], moment["signal"][v], moment["signal"][w]))
                rows.append({"point": self.point, "from_variant": first, "to_variant": second,
                             **interval_fields("cosine", estimate),
                             "status": "unresolved_signal" if estimate["value"] is None else
                             "outside_population_cosine_range" if abs(estimate["value"]) > 1 + 1e-12 else "estimated",
                             "uncertainty": "paired delete-one-50-batch-group"})
        return rows

    def contrasts(self):
        variance = self.scalar(lambda m: float(m["trace"][4] / m["trace"][3]) if m["trace"][3] > 0 else None)
        def noise_ratio(m):
            a, b = m["signal"][1:3]
            ta, tb = m["trace"][1:3]
            return float((ta / a) / (tb / b)) if a > 0 and b > 0 and tb > 0 else None
        noise = self.scalar(noise_ratio)
        resolved = all(self.summaries[v]["scale_reporting"]["kind"] == "point" for v in ("V2", "V3"))
        if not resolved:
            noise = {"value": None, "se": None, "lower": None, "upper": None,
                     "reason": "V2_or_V3_noise_scale_is_not_a_resolved_point"}
        return [{"point": self.point, "contrast": "V5_trace_over_V4_trace", **interval_fields("ratio", variance)},
                {"point": self.point, "contrast": "V2_B_ep_over_V3_B_ep", **interval_fields("ratio", noise)}]

    def public_rows(self, theta_distance):
        rows = []
        for v, variant in enumerate(VARIANTS):
            summary = self.summaries[variant]
            estimate, intervals, scale = summary["estimate"], summary["jackknife90"], summary["scale_reporting"]
            row = {"point": self.point, "variant": variant, "K": self.k, "groups": self.n,
                   "episodes": self.k * EPISODES, "theta_distance": theta_distance,
                   "B_ep_type": scale["kind"], "B_ep_reason": scale["reason"],
                   "B_ep_lower_bound": estimate["B_ep_lower_bound"],
                   "K_needed_type": scale["kind"], "K_needed": scale["K_needed"],
                   "K_needed_is_stopping_rule": False, "gradient_dimensioned_units": "paper episode-ascent" if v >= 3 else "not_public_by_protocol"}
            for metric, label, factor in (("signal_sq", "u_sq", THEORY_SCALE ** 2),
                                          ("signal_norm", "u_norm", THEORY_SCALE),
                                          ("covariance_trace", "trace_ep", EPISODES * THEORY_SCALE ** 2)):
                value = {"value": estimate[metric], **intervals[metric]}
                row.update(interval_fields(label, value, factor) if v >= 3 else {
                    key: None for key in (label, label + "_se", label + "_low", label + "_high", label + "_uncertainty_reason")})
            for metric, label in (("B_ep", "B_ep"), ("rho", "rho"), ("cosine_reliability", "c"),
                                  ("cosine_moment_crosscheck", "c_U"), ("effective_rank", "effective_rank")):
                value = {"value": estimate[metric], **intervals[metric]}
                if metric == "B_ep" and scale["kind"] != "point":
                    value = {"value": None, "se": None, "lower": None, "upper": None, "reason": scale["reason"]}
                if metric == "effective_rank" and v < 3:
                    value = {"value": None, "se": None, "lower": None, "upper": None,
                             "reason": "Public V1-V3 scope is noise scales and cosines only."}
                row.update(interval_fields(label, value))
            forecast = {"value": estimate["forecast_batches"], **intervals["forecast_batches"]}
            if scale["kind"] != "point":
                forecast = {"value": scale["K_needed"], "se": None, "lower": None, "upper": None,
                            "reason": "Plug-in lower bound, not a point confidence interval." if scale["kind"] == "lower_bound" else scale["reason"]}
            row.update(interval_fields("K_needed", forecast))
            def expected_batch_cos(moment):
                signal, trace = moment["signal"][v], moment["trace"][v]
                return float((1 + trace / signal) ** -.5) if signal > 0 and trace >= 0 else None
            batch_cos = self.scalar(expected_batch_cos)
            if scale["kind"] != "point":
                batch_cos = {"value": None, "se": None, "lower": None, "upper": None, "reason": scale["reason"]}
            row.update(interval_fields("expected_batch_cos", batch_cos))
            row["expected_batch_cos_upper_bound"] = ((1 + scale["B_ep"] / EPISODES) ** -.5
                if scale["kind"] == "lower_bound" else None)
            rows.append(row)
        return rows

    def point_row(self, theta_distance):
        row = {"point": self.point, "K": self.k, "groups": self.n, "episodes": self.k * EPISODES,
               "theta_distance": theta_distance, "uncertainty": "delete-one-50-batch-group; fixed checkpoint distance has no sampling interval"}
        for field in ("J_disc", "return"):
            row.update(interval_fields(field, group_mean(self.group_rewards[field])))
        scalar_fields = ("explained_variance", "x0_sat_frac_mean", "x0_sat_frac_last",
                         "action_oor_frac", "policy_action_oor_frac")
        for field in scalar_fields:
            if not all(field in batch and batch[field] not in (None, "") for rows in self.batches for batch in rows):
                continue
            means = [np.mean([float(batch[field]) for batch in rows]) for rows in self.batches]
            row.update(interval_fields(field, group_mean(means)))
        maxima = [max(float(batch["max_abs_reward"]) for batch in rows) for rows in self.batches]
        max_stat = jk_scalar(max(maxima), [max(np.delete(maxima, i)) for i in range(self.n)] if self.n > 1 else [])
        row.update(interval_fields("observed_max_abs_decision_reward", max_stat))
        return row


def verified_administrative_recovery(root, run_id, entry, manifest):
    """Narrow exception for seven successful measurements rejected only by the old branch guard."""
    allowed_ids = {'D2_j62', 'D2_j63', 'N2_j46', 'N2_j47', 'N2_j48', 'N2_j49', 'N2_j50'}
    record = entry.get('administrative_recovery', {})
    if (run_id not in allowed_ids or not record or manifest.get('administrative_recovery') != record or
            record.get('kind') != 'completed_native_branch_guard_failure' or
            record.get('scientific_status') != 'complete' or record.get('run_id') != run_id or
            record.get('job_id') != str(entry.get('job_id')) or
            record.get('scheduler_state') != 'FAILED' or record.get('scheduler_exit_code') != '1:0' or
            record.get('native_subprocess_exitcode') != 0 or
            record.get('original_worker_status') != 'failed' or
            record.get('original_failure_reason') != 'ValueError: Parent branch must be stage3-gradmean' or
            entry.get('scheduler_state') != 'FAILED' or entry.get('exit_code') != '1:0' or
            entry.get('terminal_recorded') is not True or entry.get('run_status') != 'complete' or
            manifest.get('status') != 'failed' or manifest.get('scientific_status') != 'complete' or
            manifest.get('subprocess_exitcode') != 0 or manifest.get('remaining_process_group_members') != [] or
            not record.get('verification')):
        return False
    config_path = Path(record.get('config_path', '')).resolve()
    original_path = Path(record.get('original_manifest', '')).resolve()
    if (Path(root).resolve() / 'runs' not in config_path.parents or
            config_path.parent not in original_path.parents or not config_path.is_file() or not original_path.is_file()):
        return False
    config, original = read_json(config_path), read_json(original_path)
    approved = config.get('runs', {}).get(run_id, {})
    return (config.get('approved') is True and config.get('kind') == record['kind'] and
            set(config.get('runs', {})) == allowed_ids and approved.get('job_id') == record['job_id'] and
            approved.get('original_manifest') == str(original_path) and
            original.get('failure_reason') == record['original_failure_reason'] and
            original.get('subprocess_exitcode') == 0 and original.get('failure_cleanup_survivors') == [] and
            all(manifest.get(key) == value for key, value in original.items()))


def discover(root):
    root = Path(root)
    campaign_path = root / "runs/campaign.json"
    campaign = read_json(campaign_path) if campaign_path.exists() else {}
    admitted = {entry["run_id"]: entry for entry in campaign.get("submitted", [])}
    inventory, found = [], {point: {} for point in POINTS}
    for path in sorted((root / "collect").glob("*/meta.json")):
        match = re.fullmatch(r"(P0|D1|D2|N1|N2)_j(\d{2})(_retry1)?", path.parent.name)
        if not match:
            continue
        meta = read_json(path)
        run_id = path.parent.name
        entry = admitted.get(run_id, {})
        manifest_path = root / "runs" / run_id / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        cleanup_path = path.parent / "cleanup.json"
        cleanup = read_json(cleanup_path) if cleanup_path.exists() else {}
        worker_complete = (manifest.get("status") == "complete" and manifest.get("subprocess_exitcode") == 0
                           and manifest.get("remaining_process_group_members") == [])
        terminal_complete = (entry.get("terminal_recorded") is True and entry.get("run_status") == "complete"
                             and entry.get("scheduler_state") == "COMPLETED" and entry.get("exit_code") == "0:0")
        recovered = verified_administrative_recovery(root, run_id, entry, manifest)
        worker_complete, terminal_complete = worker_complete or recovered, terminal_complete or recovered
        cleanup_complete = (cleanup.get("status") == "passed" and cleanup.get("all_children_exited") is True
                            and cleanup.get("remaining_active_children") == [] and cleanup.get("remaining_direct_children") == []
                            and cleanup.get("errors") == [] and cleanup.get("primary_error") is None
                            and len(cleanup.get("workers", [])) == 40
                            and all(worker.get("exitcode") == 0 and worker.get("alive") is False
                                    for worker in cleanup.get("workers", [])))
        native_complete = meta.get("status") == "completed" and meta.get("completed_batches") == GROUP_SIZE
        reasons = [name for name, passed in (("native_not_complete", native_complete),
                   ("worker_not_verified_complete", worker_complete), ("scheduler_not_verified_complete", terminal_complete),
                   ("cleanup_not_verified_complete", cleanup_complete)) if not passed]
        record = {"point": match[1], "group": int(match[2]), "retry": bool(match[3]),
                  "path": str(path.parent.resolve()), "status": meta.get("status"),
                  "completed_batches": meta.get("completed_batches"), "seed": meta.get("seed"),
                  "error": meta.get("error"), "worker_status": manifest.get("status"),
                  "scheduler_state": entry.get("scheduler_state"), "scheduler_exit_code": entry.get("exit_code"),
                  "administrative_recovery": entry.get("administrative_recovery") if recovered else None,
                  "manifest": str(manifest_path), "cleanup": str(cleanup_path),
                  "exclusion_reasons": reasons, "included": not reasons}
        inventory.append(record)
        if record["included"]:
            if record["group"] in found[record["point"]]:
                raise ValueError("Two successful originals/replacements for the same group slot")
            found[record["point"]][record["group"]] = path.parent
    points = {point: PointData(point, [paths[group] for group in sorted(paths)]) for point, paths in found.items() if paths}
    specs = [point.specs for point in points.values()]
    if specs and any(spec != specs[0] for spec in specs):
        raise ValueError("Checkpoint parameter order differs across measured policies")
    return points, inventory


def cross_point_cosines(points):
    rows = []
    for variant in ("V1", "V4"):
        v = VARIANTS.index(variant)
        for a in POINTS:
            for b in POINTS:
                if a not in points or b not in points:
                    rows.append({"variant": variant, "from_point": a, "to_point": b,
                                 "cosine": None, "cosine_low": None, "cosine_high": None,
                                 "reason": "point_not_collected"})
                    continue
                if a == b:
                    estimate = points[a].scalar(lambda m: 1. if m["signal"][v] > 0 else None)
                else:
                    left, right = points[a], points[b]
                    def statistic(ids):
                        lm, rm = left.moments(ids[a]), right.moments(ids[b])
                        inner = float(left.mean(variant, ids[a]) @ right.mean(variant, ids[b]))
                        return {"cosine": cosine(inner, lm["signal"][v], rm["signal"][v])}
                    result = independent_result(statistic, {a: left.n, b: right.n})
                    estimate = {"value": result["estimate"]["cosine"], **result["jackknife90"]["cosine"]}
                rows.append({"variant": variant, "from_point": a, "to_point": b,
                             **interval_fields("cosine", estimate),
                             "uncertainty": "independent-policy group jackknife; not paired between policies"})
    return rows


def path_rows(points, vectors):
    rows = []
    pairs = (("P0", "D1"), ("D1", "D2"), ("P0", "D2"),
             ("P0", "N1"), ("N1", "N2"), ("P0", "N2"))
    for start, end in pairs:
        if start not in points or end not in points:
            rows.append({"start": start, "end": end, "status": "point_not_collected"})
            continue
        a, b = points[start], points[end]
        d = np.asarray(vectors[end], dtype=np.float64) - np.asarray(vectors[start], dtype=np.float64)
        d_norm = float(np.linalg.norm(d))
        projections = {start: a.sums[3] @ d, end: b.sums[3] @ d}
        def statistic(ids):
            am, bm = a.moments(ids[start]), b.moments(ids[end])
            start_dot = THEORY_SCALE * projections[start][ids[start]].sum() / am["K"]
            end_dot = THEORY_SCALE * projections[end][ids[end]].sum() / bm["K"]
            return {"start_dot_displacement": float(start_dot),
                    "start_cos_displacement": cosine(start_dot, THEORY_SCALE ** 2 * am["signal"][3], d_norm ** 2),
                    "trapezoid_delta_J": float(.5 * (start_dot + end_dot)),
                    "measured_delta_J": b.reward(ids[end]) - a.reward(ids[start])}
        result = independent_result(statistic, {start: a.n, end: b.n})
        row = {"start": start, "end": end, "status": "estimated", "displacement_norm": d_norm,
               "units": "raw discounted decision rewards; V4 theory ascent gradients",
               "limitation": "Endpoint trapezoid along checkpoint chord; not an exact training-path integral."}
        for field, value in result["estimate"].items():
            row.update(interval_fields(field, {"value": value, **result["jackknife90"][field]}))
        rows.append(row)
    return rows


def score_and_lemma_rows(points):
    rows = []
    for name, point in points.items():
        paths = [path / "decision_scores.csv" for path in point.paths if (path / "decision_scores.csv").is_file()]
        if not paths:
            rows.append({"point": name, "status": "score_sample_unavailable"})
            continue
        if len(paths) != 1:
            raise ValueError("More than one successful first-group score sample")
        records = csv_rows(paths[0])
        if len(records) != 200:
            raise ValueError("Score diagnostic must contain exactly200 decisions")
        values = np.asarray([float(record["score_norm_sq"]) for record in records])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite stored score norm")
        bins = np.asarray([int(record["k"]) // 25 for record in records])
        deciles = [float(values[bins == i].mean()) if np.any(bins == i) else None for i in range(10)]
        maximum = max(deciles) if all(value is not None for value in deciles) else None
        rmax = max(float(batch["max_abs_reward"]) for batch_rows in point.batches for batch in batch_rows)
        mean = float(values.mean())
        rhs = rmax ** 2 * mean / (.01 ** 3)
        v5_trace = EPISODES * THEORY_SCALE ** 2 * point.full_moments["trace"][4]
        rows.append({"point": name, "status": "empirical_plugin_only", "decisions": 200,
                     "score_sample_path": str(paths[0]), "score_squared_norm_mean": mean,
                     "score_squared_norm_max_k_decile_mean": maximum,
                     "score_squared_norm_p99": float(np.quantile(values, .99)),
                     "score_squared_norm_decile_means": deciles,
                     "observed_rmax": rmax, "gamma_per_decision": .99,
                     "lemma_plugin_scale": rhs, "V5_per_episode_trace_theory": float(v5_trace),
                     "lemma_over_observed_trace": rhs / v5_trace if v5_trace > 0 else None,
                     "interval": None, "uncertainty_reason": "Approved exception: score samples occur in only one group; group jackknife unavailable.",
                     "limitation": "Observed reward maximum and200 sampled decision scores do not establish global bounds or a theorem."})
    return rows
