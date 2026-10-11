"""Predeclared G2 synthetic gate. Full execution is reserved for Slurm.

`--self-check-only` runs deterministic algebra/software checks, not a scientific
synthetic replicate. The full gate always uses all 200 predeclared replicates.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gradmean_math import (four_group_covariance_square, independent_group_jackknife,
                           jackknife_interval, same_point_cross, scalar_statistics,
                           split_orders, summarize_groups)


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "g2_predeclared.json"


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _assert_close(checks, name, actual, expected, rtol=1e-12, atol=1e-12):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    maximum = float(np.max(np.abs(actual - expected)))
    passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    checks[name] = {"passed": passed, "max_absolute_difference": maximum, "rtol": rtol, "atol": atol}
    if not passed:
        raise AssertionError(name + ": " + str(checks[name]))


def software_checks():
    """Independent scalar references use only fixed small rational arrays."""
    checks = {}
    samples = (np.arange(48, dtype=np.float64).reshape(12, 4) % 13 - 4) / 7
    groups = samples.reshape(6, 2, 4).sum(axis=1)
    q = np.square(samples).sum(axis=1).reshape(6, 2).sum(axis=1)
    gram = groups @ groups.T
    orders = split_orders(6, 17)
    result = scalar_statistics(gram, q, 2, orders)
    brute_signal = sum(float(samples[a] @ samples[b]) for a in range(12) for b in range(12) if a != b) / (12 * 11)
    brute_trace = sum(float((row - samples.mean(axis=0)) @ (row - samples.mean(axis=0))) for row in samples) / 11
    _assert_close(checks, "signal_excludes_same_batch_pairs", result["signal_sq"], brute_signal)
    _assert_close(checks, "unbiased_batch_covariance_trace", result["covariance_trace"], brute_trace)
    means = groups / 2
    rank_ref_terms = []
    for a, b, c, d in itertools.permutations(range(6), 4):
        rank_ref_terms.append(float((means[a] - means[b]) @ (means[c] - means[d])) ** 2 / 4)
    rank_ref = float(np.mean(rank_ref_terms))
    _assert_close(checks, "four_distinct_groups_vs_enumerated_reference",
                  four_group_covariance_square(means @ means.T), rank_ref)
    shifted = means + np.array([3., -7., .5, 2.])
    _assert_close(checks, "four_group_translation_invariance",
                  four_group_covariance_square(shifted @ shifted.T), rank_ref)
    _assert_close(checks, "group_to_batch_covariance_square_factor",
                  result["covariance_trace_squared"], 4 * rank_ref)

    other = samples[:, ::-1] * np.array([1., -2., .5, 3.]) + .7
    other_groups = other.reshape(6, 2, 4).sum(axis=1)
    q_other = np.square(other).sum(axis=1).reshape(6, 2).sum(axis=1)
    q_cross = (samples * other).sum(axis=1).reshape(6, 2).sum(axis=1)
    cross = same_point_cross(gram, other_groups @ other_groups.T, groups @ other_groups.T,
                             q, q_other, q_cross, 2)
    brute_cross = sum(float(samples[a] @ other[b]) for a in range(12) for b in range(12) if a != b) / (12 * 11)
    _assert_close(checks, "paired_cross_excludes_same_batch",
                  cross["estimate"]["mean_inner_product"], brute_cross)

    summary = summarize_groups(groups, q, 2, orders=orders)
    loo_signal = []
    for deleted in range(6):
        retained = np.delete(np.arange(12), [2 * deleted, 2 * deleted + 1])
        vectors = samples[retained]
        loo_signal.append(sum(float(vectors[a] @ vectors[b]) for a in range(10) for b in range(10) if a != b) / 90)
    ref_se = math.sqrt(5 / 6 * sum((x - np.mean(loo_signal)) ** 2 for x in loo_signal))
    _assert_close(checks, "delete_group_signal_replicates", summary["jackknife90"]["signal_sq"]["replicates"], loo_signal)
    _assert_close(checks, "delete_group_signal_standard_error", summary["jackknife90"]["signal_sq"]["se"], ref_se)
    _assert_close(checks, "literal_90_percent_normal_interval", summary["jackknife90"]["signal_sq"]["upper"], brute_signal + 1.645 * ref_se)

    x, y = np.array([1., 3., 5.]), np.array([2., 4., 8., 10.])
    joint = independent_group_jackknife(lambda ids: {"dot": float(x[ids["x"]].mean() * y[ids["y"]].mean())}, {"x": 3, "y": 4})
    se_x = jackknife_interval(float(x.mean() * y.mean()), [float(np.delete(x, i).mean() * y.mean()) for i in range(3)])["se"]
    se_y = jackknife_interval(float(x.mean() * y.mean()), [float(x.mean() * np.delete(y, i).mean()) for i in range(4)])["se"]
    _assert_close(checks, "independent_stratum_variances_sum", joint["jackknife90"]["dot"]["se"], math.sqrt(se_x ** 2 + se_y ** 2))

    anti = np.array([[1., 0.], [-1., 0.]])
    unavailable = summarize_groups(anti, [1., 1.], 1, orders=np.array([[0, 1]]))
    assert unavailable["estimate"]["signal_sq"] == -1
    assert unavailable["estimate"]["rho"] == -1
    assert unavailable["estimate"]["B_ep"] is None
    assert unavailable["estimate"]["cosine_reliability"] is None
    assert not jackknife_interval(1., [None, 1., 2.])["valid"]
    checks["signed_estimates_preserved_and_undefined_not_dropped"] = {"passed": True}
    python_before, numpy_before = random.getstate(), np.random.get_state()
    split_orders(6, 19)
    numpy_after = np.random.get_state()
    assert python_before == random.getstate()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1]) and numpy_before[2:] == numpy_after[2:]
    checks["dedicated_split_rng_leaves_global_states_unchanged"] = {"passed": True}
    return {"passed": all(value["passed"] for value in checks.values()), "checks": checks,
            "scope": "Deterministic small-array scalar/identity/software checks only; no exploratory synthetic trials."}


def _validate_config(config):
    fixed = {"dimension": 20000, "batches": 2000, "group_size": 50,
             "replicates": 200, "seed": 8675309, "split_seed": 8675310,
             "half_splits": 20, "episode_count_per_batch": 80}
    for key, expected in fixed.items():
        if config[key] != expected:
            raise ValueError("Predeclared G2 setting changed: " + key)
    assert config["mean"] == {"basis_coordinate": 0, "value": 1.0, "signal_squared": 1.0}
    assert config["covariance"]["rank_component"] == 50
    assert config["covariance"]["rank_eigenvalue_increment"] == 72.0
    assert config["covariance"]["isotropic_eigenvalue"] == .02
    assert config["criteria"]["required_coverage_metrics"] == ["signal_sq", "covariance_trace"]


def one_replicate(arguments):
    index, config = arguments
    # Index-based SeedSequence makes serial/parallel scheduling scientifically identical.
    rng = np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=(index,)))
    p, k, m = config["dimension"], config["batches"], config["group_size"]
    variances = np.full(p, .02, dtype=np.float64)
    variances[:50] += 72.
    standard_deviations = np.sqrt(variances)
    sums, qs = [], []
    for _ in range(k // m):
        samples = rng.standard_normal((m, p), dtype=np.float64)
        samples *= standard_deviations
        samples[:, 0] += 1.
        sums.append(samples.sum(axis=0, dtype=np.float64))
        qs.append(float(np.einsum("ij,ij->", samples, samples, dtype=np.float64)))
    sums = np.stack(sums)
    summary = summarize_groups(sums, qs, m, split_seed=config["split_seed"] + index,
                               episodes_per_batch=config["episode_count_per_batch"])
    sample_mean = sums.sum(axis=0, dtype=np.float64) / k
    realized_cosine = float(sample_mean[0] / np.linalg.norm(sample_mean))
    return {"replicate": index, "seed_entropy": config["seed"], "seed_spawn_key": [index],
            "split_seed": config["split_seed"] + index, "realized_cosine": realized_cosine,
            "estimate": summary["estimate"], "jackknife90": summary["jackknife90"],
            "metadata": summary["metadata"]}


def _mean_and_mcse(values):
    if any(value is None or not math.isfinite(value) for value in values):
        return {"mean": None, "mc_se": None, "valid_count": sum(value is not None and math.isfinite(value) for value in values), "count": len(values)}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "mc_se": float(array.std(ddof=1) / math.sqrt(len(array))), "valid_count": len(array), "count": len(array)}


def assess(replicates, config):
    criteria = config["criteria"]
    trace = config["covariance"]["trace"]
    trace_sq = 50 * (72.02 ** 2) + (20000 - 50) * (.02 ** 2)
    rank = trace ** 2 / trace_sq
    truth = {"signal_sq": 1., "signal_norm": 1., "covariance_trace": trace,
             "covariance_trace_squared": trace_sq, "B_batch": trace,
             "B_ep": 80 * trace, "effective_rank": rank,
             "forecast_batches": 4.26 * trace}
    checks = {"required_replicate_count": {"passed": len(replicates) == criteria["required_replicates"],
                                          "actual": len(replicates), "required": criteria["required_replicates"]}}
    means = {key: _mean_and_mcse([row["estimate"][key] for row in replicates]) for key in replicates[0]["estimate"]}
    for key in criteria["mean_bias_metrics"]:
        stats = means[key]
        bias = None if stats["mean"] is None else stats["mean"] - truth[key]
        passed = bias is not None and abs(bias) <= criteria["mean_bias_max_mc_standard_errors"] * stats["mc_se"]
        checks[key + "_mean_bias"] = {"passed": passed, "mean": stats["mean"], "truth": truth[key],
                                     "bias": bias, "mc_se": stats["mc_se"], "required_within_mc_se": 3.0}
    coverage = {}
    for key, reference in list(truth.items()) + [("cosine_reliability", "realized_cosine")]:
        covered, valid = 0, 0
        for row in replicates:
            interval = row["jackknife90"][key]
            value = row[reference] if isinstance(reference, str) else reference
            if interval["valid"]:
                valid += 1
                covered += interval["lower"] <= value <= interval["upper"]
        fraction = covered / len(replicates)
        coverage[key] = {"covered": int(covered), "valid_intervals": valid, "replicates": len(replicates),
                         "coverage_of_all_replicates": fraction,
                         "reference": reference, "formal_gate_metric": key in criteria["required_coverage_metrics"],
                         "undefined_intervals_excluded": False}
        if key in criteria["required_coverage_metrics"]:
            checks[key + "_90_percent_coverage"] = {"passed": valid == len(replicates) and criteria["coverage_min"] <= fraction <= criteria["coverage_max"],
                                                   **coverage[key], "minimum": .80, "maximum": .97}
    realized = _mean_and_mcse([row["realized_cosine"] for row in replicates])
    estimated_c = means["cosine_reliability"]["mean"]
    cosine_difference = None if estimated_c is None else estimated_c - realized["mean"]
    checks["mean_cosine_agrees_with_realized_cosine"] = {
        "passed": cosine_difference is not None and abs(cosine_difference) <= criteria["mean_cosine_max_absolute_error"],
        "mean_estimate": estimated_c, "mean_realized": realized["mean"],
        "difference": cosine_difference, "maximum_absolute_difference": .03,
        "valid_estimates": means["cosine_reliability"]["valid_count"], "required_estimates": len(replicates)}
    rank_mean = means["effective_rank"]["mean"]
    rank_error = None if rank_mean is None else (rank_mean - rank) / rank
    checks["mean_effective_rank"] = {"passed": rank_error is not None and abs(rank_error) <= criteria["mean_effective_rank_max_relative_error"],
                                     "mean": rank_mean, "truth": rank, "relative_error": rank_error,
                                     "maximum_relative_error": .20}
    return {"passed": all(value["passed"] for value in checks.values()), "checks": checks,
            "known_truth": truth, "mean_estimates": means, "interval_coverage": coverage,
            "coverage_warning": "Only signal and trace coverage are formal gates. Other intervals are descriptive, and reliability coverage has a random realized reference. No undefined intervals or replicates were silently dropped."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve evidence: output directory already exists: " + str(args.out))
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.out.mkdir(parents=True)
    started = time.time()
    config = json.loads(CONFIG_PATH.read_text())
    _validate_config(config)
    _json_write(args.out / "predeclared_config.json", config)
    software = software_checks()
    _json_write(args.out / "software_checks.json", software)
    if args.self_check_only:
        print(json.dumps({"passed": software["passed"], "scope": "software_checks_only", "output": str(args.out)}, allow_nan=False), flush=True)
        return
    _json_write(args.out / "execution.json", {"status": "running", "command": sys.argv, "started_unix": started,
                                            "python": platform.python_version(), "numpy": np.__version__,
                                            "workers": args.workers, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                            "config_source": str(CONFIG_PATH), "completed_replicates": 0})
    rows = []
    payloads = [(i, config) for i in range(config["replicates"])]
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        iterator = executor.map(one_replicate, payloads) if executor else map(one_replicate, payloads)
        with (args.out / "replicates.jsonl").open("x") as stream:
            for row in iterator:
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print("G2 replicate %d/%d saved; elapsed %.1fs" % (len(rows), config["replicates"], time.time() - started), flush=True)
        summary = assess(rows, config)
        summary.update({"gate": "G2", "software_checks_passed": software["passed"],
                        "predeclared_config": str(args.out / "predeclared_config.json"),
                        "replicates_path": str(args.out / "replicates.jsonl"),
                        "wall_seconds": time.time() - started, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "status": "passed" if summary["passed"] else "failed"})
        _json_write(args.out / "gate_summary.json", summary)
        print(json.dumps({"passed": summary["passed"], "output": str(args.out), "wall_seconds": summary["wall_seconds"]}), flush=True)
        if not summary["passed"]:
            raise SystemExit(2)
    except Exception as error:
        _json_write(args.out / "failure.json", {"status": "crashed", "error": repr(error),
                                              "completed_replicates": len(rows), "wall_seconds": time.time() - started})
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
