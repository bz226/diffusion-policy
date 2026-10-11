"""Predeclared G2-G1 synthetic gate. Full execution is reserved for Slurm.

`--self-check-only` runs deterministic algebra/software checks, not a scientific
synthetic replicate. The full gate uses 200 replicates in each fixed regime.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
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
                           signal_scale_reporting, split_orders, summarize_groups)


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "g2_g1_predeclared.json"


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
    zero = summarize_groups(np.eye(2), [1., 1.], 1, orders=np.array([[0, 1]]))
    assert zero["estimate"]["rho"] == 0 and zero["estimate"]["signal_sq"] == 0
    assert zero["estimate"]["cosine_reliability"] is None
    assert zero["estimate"]["cosine_moment_crosscheck"] is None
    checks["zero_rho_and_zero_signal_are_direction_unresolved"] = {"passed": True}
    raw = {"signal_sq": -.05, "covariance_trace": 4000., "B_ep": None, "forecast_batches": None}
    bound = signal_scale_reporting(raw, {"valid": True, "lower": -1., "upper": 1.})
    assert bound["kind"] == "lower_bound" and bound["B_ep"] == 320000.
    _assert_close(checks, "nonpositive_signal_lower_bound_forecast", bound["K_needed"], 17040.)
    raw.update(signal_sq=.2, B_ep=1600000., forecast_batches=85200.)
    bound = signal_scale_reporting(raw, {"valid": True, "lower": -.4, "upper": .8})
    assert bound["kind"] == "lower_bound" and bound["B_ep"] == 400000.
    _assert_close(checks, "interval_crossing_zero_uses_upper_signal_bound", bound["K_needed"], 21300.)
    raw.update(signal_sq=1., B_ep=320000., forecast_batches=17040.)
    point = signal_scale_reporting(raw, {"valid": True, "lower": .5, "upper": 1.5})
    assert point["kind"] == "point" and point["B_ep"] == 320000.
    raw["signal_sq"] = -.5
    for upper in (0., -.1):
        missing = signal_scale_reporting(raw, {"valid": True, "lower": -1., "upper": upper})
        assert missing["kind"] == "unavailable" and missing["B_ep"] is None
        assert missing["reason"] == "signal_ci_upper_nonpositive"
        assert not missing["bound_is_stopping_criterion"]
    checks["positive_signal_point_and_nonpositive_upper_bound_unavailable"] = {"passed": True}
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
    assert config["gate"] == "G2-G1" and config["amendment"] == "G-1"
    assert config["base_covariance"]["rank_component"] == 50
    assert config["base_covariance"]["rank_eigenvalue_increment"] == 72.0
    assert config["base_covariance"]["isotropic_eigenvalue"] == .02
    assert config["regimes"] == {
        "L": {"covariance_multiplier": 1., "covariance_trace": 4000., "direction_accuracy_is_gate": False},
        "T": {"covariance_multiplier": 470. / 4000., "covariance_trace": 470., "direction_accuracy_is_gate": True}}
    assert config["criteria"]["required_coverage_metrics"] == ["signal_sq", "covariance_trace"]


def one_replicate(arguments):
    regime, index, config = arguments
    # Index-based SeedSequence makes serial/parallel scheduling scientifically identical.
    rng = np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=(index,)))
    p, k, m = config["dimension"], config["batches"], config["group_size"]
    variances = np.full(p, .02, dtype=np.float64)
    variances[:50] += 72.
    variances *= config["regimes"][regime]["covariance_multiplier"]
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
    return {"regime": regime, "replicate": index, "seed_entropy": config["seed"], "seed_spawn_key": [index],
            "split_seed": config["split_seed"] + index, "realized_cosine": realized_cosine,
            "estimate": summary["estimate"], "raw_estimate": summary["raw_estimate"],
            "scale_reporting": summary["scale_reporting"], "jackknife90": summary["jackknife90"],
            "metadata": summary["metadata"]}


def _mean_and_mcse(values):
    if any(value is None or not math.isfinite(value) for value in values):
        return {"mean": None, "mc_se": None, "valid_count": sum(value is not None and math.isfinite(value) for value in values), "count": len(values)}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "mc_se": float(array.std(ddof=1) / math.sqrt(len(array))), "valid_count": len(array), "count": len(array)}


def assess_regime(replicates, config, regime):
    criteria = config["criteria"]
    trace = config["regimes"][regime]["covariance_trace"]
    multiplier = config["regimes"][regime]["covariance_multiplier"]
    trace_sq = (50 * (72.02 ** 2) + (20000 - 50) * (.02 ** 2)) * multiplier ** 2
    rank = trace ** 2 / trace_sq
    truth = {"signal_sq": 1., "signal_norm": 1., "covariance_trace": trace,
             "covariance_trace_squared": trace_sq, "B_batch": trace,
             "B_ep": 80 * trace, "effective_rank": rank,
             "forecast_batches": 4.26 * trace}
    required = criteria["required_replicates_per_regime"]
    checks = {"required_replicate_count": {"passed": len(replicates) == required and
                                          [row["replicate"] for row in replicates] == list(range(required)),
                                          "actual": len(replicates), "required": required}}
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
    valid_direction = [row for row in replicates if row["estimate"]["cosine_reliability"] is not None]
    signed_errors = [row["estimate"]["cosine_reliability"] - row["realized_cosine"] for row in valid_direction]
    nonpositive_rho = sum(row["estimate"]["rho"] is not None and row["estimate"]["rho"] <= 0 for row in replicates)
    direction = {
        "replicates": len(replicates), "defined_cosines": len(valid_direction),
        "nonpositive_rho_count": nonpositive_rho,
        "nonpositive_rho_fraction": nonpositive_rho / len(replicates),
        "unavailable_cosine_count": len(replicates) - len(valid_direction),
        "conditional_mean_signed_error": float(np.mean(signed_errors)) if signed_errors else None,
        "conditional_mean_absolute_error": float(np.mean(np.abs(signed_errors))) if signed_errors else None,
        "conditional_error_denominator": len(valid_direction),
        "conditional_scope": "Descriptive rho>0 subset; not substituted for all-replicate target-regime accuracy.",
        "mean_realized_cosine_all": float(np.mean([row["realized_cosine"] for row in replicates])),
        "direction_accuracy_is_gate": config["regimes"][regime]["direction_accuracy_is_gate"],
        "undefined_replicate_ids": [row["replicate"] for row in replicates if row["estimate"]["cosine_reliability"] is None]}
    if config["regimes"][regime]["direction_accuracy_is_gate"]:
        errors = [None if row["estimate"]["cosine_reliability"] is None else
                  abs(row["estimate"]["cosine_reliability"] - row["realized_cosine"]) for row in replicates]
        error_mean = _mean_and_mcse(errors)
        within = sum(error is not None and error <= criteria["T_absolute_cosine_error_threshold"] for error in errors)
        fraction = within / len(replicates)
        checks["mean_absolute_cosine_error"] = {
            "passed": error_mean["mean"] is not None and error_mean["mean"] <= criteria["T_mean_absolute_cosine_error_max"],
            "mean_absolute_error": error_mean["mean"], "mc_se": error_mean["mc_se"],
            "defined_count": error_mean["valid_count"], "denominator": len(replicates),
            "maximum_mean_absolute_error": criteria["T_mean_absolute_cosine_error_max"]}
        checks["cosine_absolute_error_tail"] = {
            "passed": fraction >= criteria["T_fraction_within_error_threshold_min"],
            "within_threshold_count": within, "denominator": len(replicates), "fraction": fraction,
            "absolute_error_threshold": criteria["T_absolute_cosine_error_threshold"],
            "minimum_fraction": criteria["T_fraction_within_error_threshold_min"],
            "unavailable_counted_as_failure": True}
    rank_mean = means["effective_rank"]["mean"]
    rank_error = None if rank_mean is None else (rank_mean - rank) / rank
    checks["mean_effective_rank"] = {"passed": rank_error is not None and abs(rank_error) <= criteria["mean_effective_rank_max_relative_error"],
                                     "mean": rank_mean, "truth": rank, "relative_error": rank_error,
                                     "maximum_relative_error": .20}
    return {"regime": regime, "passed": all(value["passed"] for value in checks.values()), "checks": checks,
            "known_truth": truth, "mean_estimates": means, "interval_coverage": coverage,
            "direction_diagnostics": direction,
            "coverage_warning": "Only signal and trace coverage are formal gates. Other intervals are descriptive raw-ratio intervals, and reliability coverage has a random realized reference. No undefined intervals or replicates were silently dropped."}


def assess(replicates, config):
    by_regime = {regime: assess_regime([row for row in replicates if row["regime"] == regime], config, regime)
                 for regime in config["regimes"]}
    return {"passed": all(value["passed"] for value in by_regime.values()), "regimes": by_regime,
            "replicate_count": len(replicates), "amendment": config["amendment"],
            "original_failure_preserved": "Original G2 declaration/source snapshot/run records remain separate and unchanged."}


def shutdown_executor(executor):
    """Python3.8-compatible close/join, with explicit liveness evidence."""
    processes = list(executor._processes.values()) if executor is not None else []
    if executor is not None:
        # shutdown(wait=True) closes the executor queues and joins its workers.
        # Capture Process objects first so joining/liveness checks are explicit.
        executor.shutdown(wait=True)
    records = []
    for process in processes:
        process.join()
        record = {"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()}
        records.append(record)
        if process.is_alive():
            raise RuntimeError("Executor worker remained alive after shutdown: " + str(process.pid))
        process.close()
        if record["exitcode"] != 0:
            raise RuntimeError("Executor worker exited unsuccessfully: " + repr(record))
    active = [process.pid for process in multiprocessing.active_children()]
    if active:
        raise RuntimeError("Child processes remain after G2-G1 cleanup: " + repr(active))
    return {"shutdown_success": True, "workers": records, "active_child_pids": active,
            "method": "executor.shutdown(wait=True); explicit process.join(), process.close(), active_children() check"}


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
    summary = None
    payloads = [(regime, i, config) for regime in config["regimes"] for i in range(config["replicates"])]
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        iterator = executor.map(one_replicate, payloads) if executor else map(one_replicate, payloads)
        with (args.out / "replicates.jsonl").open("x") as stream:
            for row in iterator:
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print("G2-G1 regime=%s replicate=%d/%d total=%d/%d saved; elapsed %.1fs" % (
                    row["regime"], row["replicate"] + 1, config["replicates"], len(rows), len(payloads), time.time() - started), flush=True)
        summary = assess(rows, config)
        scientific_passed = summary["passed"]
        summary.update({"gate": "G2-G1", "software_checks_passed": software["passed"],
                        "predeclared_config": str(args.out / "predeclared_config.json"),
                        "replicates_path": str(args.out / "replicates.jsonl"),
                        "wall_seconds": time.time() - started, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "scientific_passed": scientific_passed, "passed": False,
                        "status": "awaiting_executor_cleanup"})
        _json_write(args.out / "gate_summary.json", summary)
        print(json.dumps({"scientific_passed": scientific_passed, "awaiting_executor_cleanup": True,
                          "output": str(args.out), "wall_seconds": summary["wall_seconds"]}), flush=True)
        if not scientific_passed:
            raise SystemExit(2)
    except Exception as error:
        _json_write(args.out / "failure.json", {"status": "crashed", "error": repr(error),
                                              "completed_replicates": len(rows), "wall_seconds": time.time() - started})
        raise
    finally:
        # No unsupported cancel_futures keyword and no exception suppression.
        cleanup = shutdown_executor(executor)
        _json_write(args.out / "executor_shutdown.json", cleanup)
        print(json.dumps(cleanup, allow_nan=False), flush=True)
        if summary is not None:
            summary.update(passed=summary["scientific_passed"],
                           status="passed" if summary["scientific_passed"] else "failed",
                           executor_shutdown=cleanup, wall_seconds=time.time() - started,
                           expected_exit_code=0 if summary["scientific_passed"] else 2)
            _json_write(args.out / "gate_summary.json", summary)
            print(json.dumps({"passed": summary["passed"], "expected_exit_code": summary["expected_exit_code"],
                              "output": str(args.out), "wall_seconds": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
