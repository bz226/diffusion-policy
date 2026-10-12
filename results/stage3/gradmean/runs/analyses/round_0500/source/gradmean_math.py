"""Float64 statistics for independent whole batches, grouped only for uncertainty.

No training imports, global RNG use, floors, clipping, or bias-offset corrections.
Input vectors retain their declared units. File summaries use the collector's
code-ascent units; multiply V4/V5 vectors by 2500 separately for paper units.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


VARIANTS = ("V1", "V2", "V3", "V4", "V5")
CRITICAL_90 = 1.645


def finite_float(value):
    """JSON uses null, with separate validity metadata, for undefined values."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def split_orders(groups: int, seed: int = 8675310, splits: int = 20):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(groups) for _ in range(splits)])


def four_group_covariance_square(gram):
    """Estimate tr(C^2), C=Cov(group mean), via four DISTINCT groups.

    The symmetric U-statistic is mean[((Xa-Xb).(Xc-Xd))^2]/4.
    Its Gram expansion is A - 2 B + C: squared edges, wedges, and
    disjoint edges respectively. Diagonal entries never enter it.
    This exact algebra is verified against ordered quadruple enumeration.
    """
    gram = np.asarray(gram, dtype=np.float64)
    n = len(gram)
    if n < 4:
        return None
    # Translation does not change differences. Centering the Gram matrix first
    # reduces cancellation and does not change the four-group estimator.
    row_mean = gram.mean(axis=1)
    h = gram - row_mean[:, None] - row_mean[None, :] + gram.mean()
    h = h.copy()
    np.fill_diagonal(h, 0.0)
    edges_squared = np.einsum("ij,ij->", h, h, dtype=np.float64)
    row_sums = h.sum(axis=1, dtype=np.float64)
    wedges_plus_edges = np.dot(row_sums, row_sums)
    edges_sum = row_sums.sum(dtype=np.float64)
    a = edges_squared / (n * (n - 1))
    b = (wedges_plus_edges - edges_squared) / (n * (n - 1) * (n - 2))
    c = (edges_sum ** 2 - 4 * wedges_plus_edges + 2 * edges_squared) / (
        n * (n - 1) * (n - 2) * (n - 3)
    )
    return float(a - 2 * b + c)


def _cosine(dot, norm_a_sq, norm_b_sq):
    if norm_a_sq <= 0 or norm_b_sq <= 0:
        return None
    return finite_float(dot / math.sqrt(norm_a_sq * norm_b_sq))


def scalar_statistics(gram, norm_sq_sums, group_size=50, orders=None,
                      group_ids=None, episodes_per_batch=80):
    """Compute scalar statistics from the Gram matrix of GROUP SUM vectors.

    norm_sq_sums[g] is sum over independent batches of ||s_b||^2, not
    ||S_g||^2. group_ids connects deletion replicates to fixed split orders.
    """
    gram = np.asarray(gram, dtype=np.float64)
    norm_sq_sums = np.asarray(norm_sq_sums, dtype=np.float64)
    n = len(gram)
    if gram.shape != (n, n) or norm_sq_sums.shape != (n,):
        raise ValueError("Gram and within-group squared norm shapes differ")
    if n == 0 or group_size < 1 or not np.isfinite(gram).all() or not np.isfinite(norm_sq_sums).all():
        raise ValueError("Require nonempty, finite data and a positive group size")
    k = n * int(group_size)
    q = norm_sq_sums.sum(dtype=np.float64)
    total_norm_sq = gram.sum(dtype=np.float64)
    signal = (total_norm_sq - q) / (k * (k - 1)) if k > 1 else None
    trace = (q - total_norm_sq / k) / (k - 1) if k > 1 else None
    group_cov_sq = four_group_covariance_square(gram / float(group_size ** 2))
    covariance_sq = None if group_cov_sq is None else group_size ** 2 * group_cov_sq
    rho_values = []
    if orders is not None and n >= 2:
        ids = np.arange(n) if group_ids is None else np.asarray(group_ids)
        lookup = {int(original): local for local, original in enumerate(ids)}
        for order in np.asarray(orders):
            kept = np.array([lookup[int(i)] for i in order if int(i) in lookup], dtype=int)
            if len(kept) != n:
                raise ValueError("Split order does not cover retained groups exactly")
            a, b = kept[:n // 2], kept[n // 2:]
            rho_values.append(_cosine(gram[np.ix_(a, b)].sum(),
                                      gram[np.ix_(a, a)].sum(), gram[np.ix_(b, b)].sum()))
    rho = float(np.mean(rho_values)) if rho_values and all(x is not None for x in rho_values) else None
    c = math.sqrt(2 * rho / (1 + rho)) if rho is not None and rho > 0 else None
    positive_signal = signal is not None and signal > 0
    valid_trace = trace is not None and trace >= 0
    values = {
        "signal_sq": signal,
        "signal_norm": math.sqrt(signal) if positive_signal else None,
        "covariance_trace": trace,
        "covariance_trace_squared": covariance_sq,
        "mean_gradient_norm_sq": total_norm_sq / k ** 2,
        "B_batch": trace / signal if positive_signal else None,
        "B_ep": episodes_per_batch * trace / signal if positive_signal else None,
        "effective_rank": trace ** 2 / covariance_sq if covariance_sq is not None and covariance_sq > 0 else None,
        "rho": rho,
        "cosine_reliability": c,
        "cosine_moment_crosscheck": (1 + trace / (k * signal)) ** -.5 if positive_signal and valid_trace else None,
        "forecast_batches": 4.26 * trace / signal if positive_signal else None,
    }
    return {name: finite_float(value) for name, value in values.items()}


def jackknife_interval(estimate, replicates, critical=CRITICAL_90):
    """The literal delete-one-group normal interval; no bias correction."""
    values = [finite_float(x) for x in replicates]
    if estimate is None:
        return {"se": None, "lower": None, "upper": None, "valid": False,
                "reason": "full_sample_statistic_undefined", "replicates": values}
    if len(values) < 2 or any(x is None for x in values):
        return {"se": None, "lower": None, "upper": None, "valid": False,
                "reason": "insufficient_or_undefined_delete_group_replicate", "replicates": values}
    x = np.asarray(values, dtype=np.float64)
    se = math.sqrt((len(x) - 1) / len(x) * np.square(x - x.mean()).sum())
    return {"se": se, "lower": estimate - critical * se, "upper": estimate + critical * se,
            "valid": True, "reason": None, "replicates": values}


def signal_scale_reporting(estimate, signal_interval, episodes_per_batch=80):
    """G-1 display semantics, after JK, without altering internal U-statistics.

    This is the approved plug-in lower bound using the upper signal limit; it
    does not incorporate uncertainty of the trace numerator or claim joint
    confidence coverage. A missing bound never triggers sampling closure.
    """
    signal = estimate["signal_sq"]
    trace = estimate["covariance_trace"]
    lower = signal_interval.get("lower")
    upper = signal_interval.get("upper")
    ci_valid = signal_interval.get("valid", False) and lower is not None and upper is not None
    unresolved = signal is None or signal <= 0 or (ci_valid and lower <= 0 <= upper)
    base = {"kind": None, "B_ep": None, "K_needed": None, "reason": None,
            "signal_ci_lower": lower, "signal_ci_upper": upper,
            "finite_bound_available": False, "bound_is_stopping_criterion": False,
            "definition": "Approved plug-in bound; trace uncertainty is not propagated into its numerator."}
    if not ci_valid:
        base.update(kind="unavailable", reason="signal_jackknife_interval_unavailable")
    elif unresolved:
        if upper <= 0:
            base.update(kind="unavailable", reason="signal_ci_upper_nonpositive")
        elif trace is None or trace < 0:
            base.update(kind="unavailable", reason="covariance_trace_unavailable_or_negative")
        else:
            bound = episodes_per_batch * trace / upper
            base.update(kind="lower_bound", B_ep=bound,
                        K_needed=4.26 * bound / episodes_per_batch,
                        finite_bound_available=True, reason="signal_nonpositive_or_interval_includes_zero")
    else:
        base.update(kind="point", B_ep=estimate["B_ep"], K_needed=estimate["forecast_batches"],
                    reason="positive_signal_interval")
    return base


def summarize_gram(gram, norm_sq_sums, group_size=50, orders=None,
                   split_seed=8675310, episodes_per_batch=80):
    gram = np.asarray(gram, dtype=np.float64)
    q = np.asarray(norm_sq_sums, dtype=np.float64)
    n = len(gram)
    orders = split_orders(n, split_seed) if orders is None and n >= 2 else orders
    full = scalar_statistics(gram, q, group_size, orders, episodes_per_batch=episodes_per_batch)
    replicas = []
    if n > 1:
        for deleted in range(n):
            ids = np.delete(np.arange(n), deleted)
            replicas.append(scalar_statistics(gram[np.ix_(ids, ids)], q[ids], group_size,
                                              orders, ids, episodes_per_batch))
    intervals = {key: jackknife_interval(value, [r[key] for r in replicas]) for key, value in full.items()}
    raw = dict(full)
    reporting = signal_scale_reporting(raw, intervals["signal_sq"], episodes_per_batch)
    # Raw ratios remain available for mathematical audit, never mislabel a bound
    # as a point estimate in the public summary.
    if reporting["kind"] != "point":
        for key in ("B_batch", "B_ep", "forecast_batches"):
            full[key] = None
    full["B_ep_lower_bound"] = reporting["B_ep"] if reporting["kind"] == "lower_bound" else None
    full["forecast_batches_lower_bound"] = reporting["K_needed"] if reporting["kind"] == "lower_bound" else None
    return {"estimate": full, "raw_estimate": raw, "scale_reporting": reporting,
            "jackknife90": intervals,
            "metadata": {"batch_count": n * group_size, "group_count": n,
                         "group_size": group_size, "episodes_per_batch": episodes_per_batch,
                         "split_count": 0 if orders is None else len(orders),
                         "split_seed": split_seed, "normal_critical_value": CRITICAL_90,
                         "full_split_group_sizes": [n // 2, n - n // 2],
                         "deletion_split_group_sizes": [(n - 1) // 2, n - 1 - (n - 1) // 2],
                         "clipped_or_bias_corrected": False}}


def summarize_groups(group_sums, norm_sq_sums, group_size=50, **kwargs):
    sums = np.asarray(group_sums, dtype=np.float64)
    if sums.ndim != 2:
        raise ValueError("group_sums must be (groups, coordinates)")
    return summarize_gram(sums @ sums.T, norm_sq_sums, group_size, **kwargs)


def same_point_cross(gram_v, gram_w, cross_gram, q_v, q_w, q_cross, group_size=50):
    """Paired group deletion, excluding SAME-BATCH pairs in the numerator."""
    gram_v, gram_w, cross_gram = [np.asarray(x, dtype=np.float64) for x in (gram_v, gram_w, cross_gram)]
    q_v, q_w, q_cross = [np.asarray(x, dtype=np.float64) for x in (q_v, q_w, q_cross)]
    n = len(gram_v)

    def estimate(ids):
        k = len(ids) * group_size
        if k <= 1:
            return {"mean_inner_product": None, "cosine": None}
        vv = (gram_v[np.ix_(ids, ids)].sum() - q_v[ids].sum()) / (k * (k - 1))
        ww = (gram_w[np.ix_(ids, ids)].sum() - q_w[ids].sum()) / (k * (k - 1))
        vw = (cross_gram[np.ix_(ids, ids)].sum() - q_cross[ids].sum()) / (k * (k - 1))
        return {"mean_inner_product": finite_float(vw), "cosine": _cosine(vw, vv, ww)}

    full = estimate(np.arange(n))
    replicas = [estimate(np.delete(np.arange(n), i)) for i in range(n)] if n > 1 else []
    return {"estimate": full,
            "jackknife90": {key: jackknife_interval(value, [r[key] for r in replicas]) for key, value in full.items()}}


def independent_group_jackknife(estimator: Callable, groups_by_point):
    """Sum independent stratum contributions; do not pair unrelated groups.

    estimator receives {point: retained_group_index_array} and returns scalars.
    It must recompute all ratios/cosines from those retained measurements.
    """
    ids = {point: np.arange(count) for point, count in groups_by_point.items()}
    full = estimator(ids)
    strata = {}
    for point, original in ids.items():
        replicas = []
        for deleted in range(len(original)):
            kept = dict(ids)
            kept[point] = np.delete(original, deleted)
            replicas.append(estimator(kept))
        strata[point] = {key: jackknife_interval(value, [r[key] for r in replicas]) for key, value in full.items()}
    intervals = {}
    for key, value in full.items():
        if value is None or any(not strata[point][key]["valid"] for point in strata):
            intervals[key] = {"valid": False, "se": None, "lower": None, "upper": None,
                              "reason": "undefined_full_or_independent_stratum_replicate"}
        else:
            se = math.sqrt(sum(strata[point][key]["se"] ** 2 for point in strata))
            intervals[key] = {"valid": True, "se": se, "lower": value - CRITICAL_90 * se,
                              "upper": value + CRITICAL_90 * se, "reason": None}
    return {"estimate": full, "jackknife90": intervals, "strata": strata}


def summarize_group_files(group_dirs, split_seed=8675310):
    """Read collector records; native sum arrays remain untouched."""
    paths = [Path(path).resolve() for path in group_dirs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Require unique group directories")
    metas = [json.loads((path / "meta.json").read_text()) for path in paths]
    sizes = [int(meta["completed_batches"]) for meta in metas]
    if len(set(sizes)) != 1 or sizes[0] != 50:
        raise ValueError("Every completed group must contain exactly 50 batches")
    for meta in metas:
        if meta.get("status") not in ("completed", "complete"):
            raise ValueError("Do not summarize incomplete or failed groups")
    points = {meta["point"] for meta in metas}
    if len(points) != 1:
        raise ValueError("Point summary cannot mix policies")
    seeds = [meta["seed"] for meta in metas]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate measurement seeds are not independent groups")
    crosses = np.asarray([meta["gradient_cross_sum"] for meta in metas], dtype=np.float64)
    if crosses.shape != (len(paths), 5, 5) or not np.isfinite(crosses).all():
        raise ValueError("Invalid within-batch cross-moment records")
    variants = {}
    # One variant at a time limits memory when there are 82 groups of 568392 coordinates.
    for i, variant in enumerate(VARIANTS):
        sums = np.stack([np.load(path / ("sums_" + variant + ".npy"), allow_pickle=False)
                         for path in paths]).astype(np.float64)
        variants[variant] = summarize_groups(sums, crosses[:, i, i], sizes[0], split_seed=split_seed)
    return {"point": next(iter(points)), "batch_count": sum(sizes), "group_count": len(paths),
            "group_dirs": [str(path) for path in paths], "measurement_seeds": seeds,
            "gradient_units": "collector code-ascent units; no implicit 2500 conversion",
            "variants": variants,
            "scope": "Internal moments retained for computation; public V1-V3 results are B and cosines only."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=8675310)
    args = parser.parse_args()
    result = summarize_group_files(args.groups, args.split_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("Preserve prior summaries; choose a new output: " + str(args.out))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
