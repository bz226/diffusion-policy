"""Read-only comparisons with existing Stage3 training and Phase1 records."""
from __future__ import annotations

import math
from pathlib import Path
import pickle

import numpy as np

from analysis_core import (THEORY_SCALE, VARIANTS, group_mean, interval_fields,
                           read_json)
from gradmean_math import independent_group_jackknife


def drift_comparisons(stage3, p0):
    rows, grouped = [], {}
    for condition in ("SGD1", "SGD3", "SGD10", "BATCH4"):
        seed_rows = []
        for seed in range(3):
            manifest_path = Path(stage3) / "runs" / (condition + "_seed%d" % seed) / "manifest.json"
            if not manifest_path.exists():
                rows.append({"condition": condition, "seed": seed, "status": "reference_unavailable"})
                continue
            manifest = read_json(manifest_path)
            if manifest.get("status") != "complete":
                rows.append({"condition": condition, "seed": seed, "status": "reference_not_complete"})
                continue
            path = Path(manifest["logdir"]) / "result.pkl"
            with path.open("rb") as stream:
                native = pickle.load(stream)
            training = [record for record in native if "train_episode_reward" in record]
            if len(native) != 140 or len(training) != 126:
                raise ValueError("Completed drift reference lacks140/126 rows")
            if any(record["actor_optimizer_steps"] != 1 for record in training):
                raise ValueError("Drift assumes exactly one actor update per training batch")
            learning_rates = np.asarray([float(record["actor_lr"]) for record in training])
            if not np.isfinite(learning_rates).all() or np.any(learning_rates != learning_rates[0]) or learning_rates[0] <= 0:
                raise ValueError("Drift reference does not have a positive constant actor LR")
            # These returns precede their iteration's update: the first x is zero.
            updates = np.arange(len(training), dtype=np.float64)
            rewards = np.asarray([float(record["J_disc_train"]) for record in training])
            if not np.isfinite(rewards).all():
                raise ValueError("Nonfinite drift reference")
            centered = updates - updates.mean()
            slope = float(centered @ (rewards - rewards.mean()) / (centered @ centered))
            eta = float(learning_rates[0])
            value = {"condition": condition, "seed": seed, "status": "complete_reference",
                     "observations": 126, "completed_update_min": 0, "completed_update_max": 125,
                     "actor_lr_code": eta, "actor_lr_theory": eta / THEORY_SCALE,
                     "J_disc_slope_per_update": slope,
                     "squared_gradient_drift_proxy": slope / (eta / THEORY_SCALE),
                     "result_path": str(path), "manifest_path": str(manifest_path),
                     "large_step_caveat": condition == "SGD10",
                     "interpretation": "OLS over the entire path, before each update; not a local P0 gradient estimate.",
                     "per_seed_uncertainty": "No independent repeated trajectory within a seed; uncertainty is aggregated across three independent seeds."}
            rows.append(value)
            seed_rows.append(value)
        if len(seed_rows) != 3:
            grouped[condition] = {"condition": condition, "status": "incomplete_three_seed_reference"}
            continue
        proxies = np.asarray([row["squared_gradient_drift_proxy"] for row in seed_rows])
        summary = {"condition": condition, "status": "three_seed_path_proxy", "seeds": 3,
                   **interval_fields("squared_gradient_drift_proxy", group_mean(proxies)),
                   **interval_fields("J_disc_slope_per_update", group_mean([row["J_disc_slope_per_update"] for row in seed_rows])),
                   "uncertainty": "Delete-one-independent-seed SE (= sample SD/sqrt3); normal90% interval with1.645, not a t interval.",
                   "large_step_caveat": condition == "SGD10"}
        if p0 is not None and p0.summaries["V4"]["scale_reporting"]["kind"] == "point":
            def statistic(ids):
                mean_proxy = float(proxies[ids["training_seeds"]].mean())
                signal = THEORY_SCALE ** 2 * p0.moments(ids["P0"])["signal"][3]
                return {"proxy_over_P0_signal": mean_proxy / signal if signal > 0 else None}
            result = independent_group_jackknife(statistic, {"training_seeds": 3, "P0": p0.n})
            summary.update(interval_fields("proxy_over_P0_signal", {
                "value": result["estimate"]["proxy_over_P0_signal"], **result["jackknife90"]["proxy_over_P0_signal"]}))
        else:
            summary.update(proxy_over_P0_signal=None, proxy_over_P0_signal_low=None,
                           proxy_over_P0_signal_high=None, proxy_over_P0_signal_uncertainty_reason="P0_signal_unresolved_or_unavailable")
        grouped[condition] = summary
    return rows, list(grouped.values())


def phase1_comparisons(stage3, points):
    rows = []
    mapping = {"P0": "ns_theta0_seed1000", "N2": "ns_ckpt_seed2000"}
    for point, run in mapping.items():
        path = Path(stage3) / run / "noise_scale.json"
        if point not in points or not path.exists():
            rows.append({"point": point, "status": "reference_or_new_point_unavailable"})
            continue
        old = read_json(path)
        if old.get("status") != "completed" or old.get("measurement_batches") != 25:
            raise ValueError("Phase1 reference is not the completed25-batch record")
        current = points[point]
        for v, variant in enumerate(VARIANTS):
            old_row, new = old["variants"][v], current.summaries[variant]
            scale = new["scale_reporting"]
            definitions = [("B_ep", old_row["B_ep"], scale["B_ep"] if scale["kind"] == "point" else None,
                            new["jackknife90"]["B_ep"] if scale["kind"] == "point" else None)]
            if v >= 3:
                definitions.extend([
                    ("u_sq_theory", old_row["u_sq"] * THEORY_SCALE ** 2,
                     new["estimate"]["signal_sq"] * THEORY_SCALE ** 2,
                     {key: None if value is None else value * THEORY_SCALE ** 2 for key, value in
                      new["jackknife90"]["signal_sq"].items() if key in ("lower", "upper")}),
                    ("trace_ep_theory", old_row["tr_sigma"] * 2 * THEORY_SCALE ** 2,
                     new["estimate"]["covariance_trace"] * 80 * THEORY_SCALE ** 2,
                     {key: None if value is None else value * 80 * THEORY_SCALE ** 2 for key, value in
                      new["jackknife90"]["covariance_trace"].items() if key in ("lower", "upper")})])
            for metric, earlier, now, interval in definitions:
                lower, upper = (interval.get("lower"), interval.get("upper")) if interval else (None, None)
                contains = None if earlier is None or lower is None or upper is None else lower <= earlier <= upper
                rows.append({"point": point, "variant": variant, "metric": metric,
                             "phase1_point": earlier, "new_point": now,
                             "new_ci90_low": lower, "new_ci90_high": upper,
                             "new_B_ep_type": scale["kind"] if metric == "B_ep" else None,
                             "new_B_ep_lower_bound": scale["B_ep"] if metric == "B_ep" and scale["kind"] == "lower_bound" else None,
                             "new_interval_contains_old_point_descriptive_only": contains,
                             "phase1_path": str(path), "status": "descriptive_comparison_only",
                             "limitation": "Old within-batch-coupled estimates and intervals were unreliable; old uncertainty is not reused. New critic/scaler warmup differs for V1/V2. Containment is not a consistency test."})
    return rows
