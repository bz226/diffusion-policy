#!/usr/bin/env python3
"""Validate and plot the three approved, unmodified DPPO HalfCheetah runs.

Accepts explicit native log directories or result.pkl paths. No training, model
loading, smoothing, interpolation, extrapolation, or artifact curation occurs.
"""

import argparse
import csv
import json
import math
import numbers
import pickle
import sys
import textwrap
from pathlib import Path


N_ITERATIONS = 140
STEPS_PER_TRAIN_ITERATION = 80000
CHECKPOINT_ITERATIONS = (0, 35, 70, 105, 139)
TARGET_STEPS = (0, 2500000, 5000000, 7500000, 10000000)
ENDPOINT_NOTE = (
    "The prescribed 140 iterations finish at 10,080,000 training steps, but the "
    "last evaluation is iteration 130 at 9,360,000 steps. No evaluation at "
    "10,000,000 steps is available; the final evaluation comparison uses the "
    "last observed point without extrapolation."
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in range(3):
        parser.add_argument(
            "--seed%d" % seed,
            required=True,
            type=Path,
            help="Seed %d native log directory or result.pkl" % seed,
        )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New runs/<analysis-id>/candidates directory inside this study",
    )
    return parser.parse_args()


def finite_scalar(value, location):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("%s must be a real number" % location)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite value at %s" % location)
    return result


def check_all_finite(value, location, np):
    """Reject nonfinite measurements, including optional native result fields."""
    if isinstance(value, numbers.Number):
        if not bool(np.isfinite(value)):
            raise ValueError("Nonfinite value at %s" % location)
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            if not bool(np.isfinite(value).all()):
                raise ValueError("Nonfinite array at %s" % location)
        elif value.dtype == object:
            for index, item in enumerate(value.flat):
                check_all_finite(item, "%s[%d]" % (location, index), np)
    elif isinstance(value, dict):
        for key, item in value.items():
            check_all_finite(item, "%s.%s" % (location, key), np)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_all_finite(item, "%s[%d]" % (location, index), np)


def load_run(seed, supplied_path, np):
    path = supplied_path.expanduser().resolve()
    if path.is_dir():
        path = path / "result.pkl"
    if path.name != "result.pkl" or not path.is_file():
        raise ValueError("Seed %d: expected an existing result.pkl: %s" % (seed, path))
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or len(records) != N_ITERATIONS:
        raise ValueError("Seed %d: expected exactly 140 result records" % seed)
    eval_steps, eval_values, train_steps, train_values, times = [], [], [], [], []
    eval_times, train_times = [], []
    for itr, record in enumerate(records):
        location = "seed%d.itr%d" % (seed, itr)
        if not isinstance(record, dict):
            raise ValueError("%s is not a dict" % location)
        check_all_finite(record, location, np)
        for key in ("itr", "step"):
            if isinstance(record.get(key), bool) or not isinstance(
                record.get(key), numbers.Integral
            ):
                raise ValueError("%s.%s must be an integer" % (location, key))
        expected_step = (itr - itr // 10) * STEPS_PER_TRAIN_ITERATION
        if record["itr"] != itr or record["step"] != expected_step:
            raise ValueError("%s: incorrect native iteration or training step" % location)
        eval_mode = itr % 10 == 0
        wanted = "eval_episode_reward" if eval_mode else "train_episode_reward"
        forbidden = "train_episode_reward" if eval_mode else "eval_episode_reward"
        if wanted not in record or forbidden in record:
            raise ValueError("%s: invalid evaluation/training metric schedule" % location)
        reward = finite_scalar(record[wanted], location + "." + wanted)
        elapsed = finite_scalar(record.get("time"), location + ".time")
        if elapsed <= 0:
            raise ValueError("%s.time must be positive" % location)
        times.append(elapsed)
        if eval_mode:
            eval_steps.append(expected_step)
            eval_values.append(reward)
            eval_times.append(elapsed)
        else:
            train_steps.append(expected_step)
            train_values.append(reward)
            train_times.append(elapsed)

    checkpoints = []
    for itr in CHECKPOINT_ITERATIONS:
        checkpoint = path.parent / "checkpoint" / ("state_%d.pt" % itr)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError("Missing or empty checkpoint: %s" % checkpoint)
        checkpoints.append(str(checkpoint))
    return {
        "seed": seed,
        "result_path": str(path),
        "logdir": str(path.parent),
        "checkpoints": checkpoints,
        "eval_steps": np.asarray(eval_steps, dtype=np.int64),
        "eval_values": np.asarray(eval_values, dtype=float),
        "train_steps": np.asarray(train_steps, dtype=np.int64),
        "train_values": np.asarray(train_values, dtype=float),
        "timing": {
            "native_iteration_time_sum_seconds": float(np.sum(times)),
            "iteration_mean_seconds": float(np.mean(times)),
            "iteration_median_seconds": float(np.median(times)),
            "iteration_min_seconds": float(np.min(times)),
            "iteration_max_seconds": float(np.max(times)),
            "evaluation_iteration_mean_seconds": float(np.mean(eval_times)),
            "training_iteration_mean_seconds": float(np.mean(train_times)),
            "evaluation_iterations": len(eval_times),
            "training_iterations": len(train_times),
        },
    }


def curve(path, metric, steps, values, caption, np, plt):
    fig, ax = plt.subplots(figsize=(9, 6.4))
    colors = ("#0072B2", "#D55E00", "#009E73")
    x = steps / 1000000.0
    for seed, color in enumerate(colors):
        ax.plot(x, values[seed], color=color, linewidth=1.2, alpha=0.8,
                label="Seed %d" % seed)
    mean, std = np.mean(values, axis=0), np.std(values, axis=0, ddof=0)
    ax.fill_between(x, mean - std, mean + std, color="#777777", alpha=0.22,
                    label="Mean ± population std (3 seeds)")
    ax.plot(x, mean, color="#111111", linewidth=2, label="Mean")
    ax.set(xlabel="Training environment steps (millions)",
           ylabel="Undiscounted 1000-step episode return",
           title="DPPO HalfCheetah-v2: %s return" % metric,
           xlim=(0, 10.08))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.31)
    fig.text(0.05, 0.035, textwrap.fill(caption, width=122), fontsize=8,
             va="bottom", ha="left")
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180,
                    metadata={"Description": caption})
    plt.close(fig)


def main():
    args = arguments()
    study = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    runs_root = study / "runs"
    if output.name != "candidates" or runs_root not in output.parents:
        raise ValueError("--output-dir must be inside this study's runs/ and end in candidates")
    if output.exists():
        raise ValueError("Refusing existing output path: %s" % output)

    import numpy as np
    runs = [load_run(seed, getattr(args, "seed%d" % seed), np) for seed in range(3)]
    if len({run["result_path"] for run in runs}) != 3:
        raise ValueError("Three distinct result.pkl files are required")
    for metric in ("eval", "train"):
        if any(not np.array_equal(runs[0][metric + "_steps"], run[metric + "_steps"])
               for run in runs[1:]):
            raise ValueError("Seed step grids do not match")

    eval_steps = runs[0]["eval_steps"]
    eval_values = np.stack([run["eval_values"] for run in runs])
    train_values = np.stack([run["train_values"] for run in runs])
    nearest = [int(np.argmin(np.abs(eval_steps - target))) for target in TARGET_STEPS]
    if [int(eval_steps[index]) for index in nearest] != [0, 2160000, 5040000, 7200000, 9360000]:
        raise ValueError("Unexpected nearest evaluation step grid")

    # Every input invariant has passed before any output directory is created.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_mean = float(np.mean(eval_values[:, -1]))
    initial_checks = [bool(3850 <= value <= 4650) for value in eval_values[:, 0]]
    endpoint_check = bool(4550 <= final_mean <= 4900)
    numeric_pass = all(initial_checks) and endpoint_check
    captions = {
        "eval_return_vs_env_steps.png": (
            "Evaluation return during unmodified DPPO fine-tuning: each point is a native "
            "within-iteration mean of undiscounted 1000-step episode returns. The x-axis "
            "counts training environment steps in millions; colored lines show seeds 0, 1, "
            "and 2, and the black line and shaded band show their mean ± population "
            "standard deviation (ddof=0), not a confidence interval. The last observed "
            "across-seed mean is %.2f at 9.36 million steps. Evaluation ends before the "
            "10 million-step target; no smoothing or extrapolation is applied."
        ) % final_mean,
        "train_return_vs_env_steps.png": (
            "Training return during unmodified DPPO fine-tuning: each point is a native "
            "within-iteration mean of undiscounted 1000-step episode returns, collected "
            "using training exploration noise before that iteration's update. The x-axis "
            "counts training environment steps in millions; colored lines show seeds 0, 1, "
            "and 2, and the black line and shaded band show their mean ± population "
            "standard deviation (ddof=0), not a confidence interval. The final training "
            "mean is %.2f at 10.08 million steps. Training and evaluation returns use "
            "different sampling-noise settings and are not interchangeable; neither curve "
            "is smoothed or extrapolated."
        ) % float(np.mean(train_values[:, -1])),
        "returns.csv": (
            "Raw undiscounted episode returns are shown for each of three independent "
            "runs and their mean and population standard deviation (ddof=0), not a "
            "confidence interval. Evaluation rows use the nearest observed iteration to "
            "each requested step; both requested and actual steps are shown, without "
            "interpolation, and the final row lists training iteration 1 at 80,000 steps. "
            "The last observed evaluation mean is %.2f at 9.36 million steps, leaving "
            "the exact 10 million-step evaluation unmeasured."
        ) % final_mean,
    }
    summary = {
        "status": "validated_complete_runs",
        "endpoint_limitation": ENDPOINT_NOTE,
        "aggregation": {"independent_runs": 3, "seeds": [0, 1, 2],
                        "standard_deviation_ddof": 0,
                        "uncertainty": "Across-seed dispersion; not a confidence interval",
                        "smoothing": False, "interpolation": False, "extrapolation": False},
        "criteria": {
            "iteration_zero_interval": [3850, 4650],
            "iteration_zero_pass_by_seed": initial_checks,
            "last_observed_eval_mean_interval": [4550, 4900],
            "last_observed_eval_step": 9360000,
            "last_observed_eval_mean": final_mean,
            "last_observed_eval_mean_pass": endpoint_check,
            "numeric_interval_checks_pass": numeric_pass,
            "no_seed_collapse": "Requires human review; no numerical definition was supplied",
            "overall_status": "pending_collapse_review" if numeric_pass else "fail_numeric_intervals",
            "exact_10000000_step_evaluation": "not_available",
        },
        "timing_definition": (
            "Native result.pkl time is elapsed wall time since the previous iteration's "
            "timer read. Its sum is not end-to-end process runtime: initialization, "
            "downloads, and final logging/serialization are excluded."
        ),
        "runs": [{"seed": run["seed"], "logdir": run["logdir"],
                  "result_path": run["result_path"], "checkpoints": run["checkpoints"],
                  "iterations": 140, "final_training_step": 10080000,
                  "initial_eval_return": float(run["eval_values"][0]),
                  "last_eval_return": float(run["eval_values"][-1]),
                  "first_training_return": float(run["train_values"][0]),
                  "timing": run["timing"]} for run in runs],
        "artifacts": list(captions),
        "captions_file": "captions.txt",
    }

    output.mkdir(parents=True, exist_ok=False)
    for metric, title in (("eval", "evaluation"), ("train", "training")):
        name = metric + "_return_vs_env_steps.png"
        values = eval_values if metric == "eval" else train_values
        curve(output / name, title, runs[0][metric + "_steps"], values,
              captions[name], np, plt)
    with (output / "returns.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "requested_env_steps", "actual_env_steps", "itr",
                         "seed0", "seed1", "seed2", "mean", "population_std_ddof0"])
        for target, index in zip(TARGET_STEPS, nearest):
            values = eval_values[:, index]
            writer.writerow(["eval_episode_reward", target, int(eval_steps[index]), index * 10]
                            + [float(value) for value in values]
                            + [float(np.mean(values)), float(np.std(values, ddof=0))])
        values = train_values[:, 0]
        writer.writerow(["train_episode_reward", 80000, 80000, 1]
                        + [float(value) for value in values]
                        + [float(np.mean(values)), float(np.std(values, ddof=0))])
    with (output / "captions.txt").open("x") as stream:
        for name, caption in captions.items():
            stream.write(name + "\n" + caption + "\n\n")
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Validated all three runs; wrote two plots and one table to %s" % output)
    print(ENDPOINT_NOTE)
    print("Numeric interval checks: %s; no-collapse criterion requires review." %
          ("PASS" if numeric_pass else "FAIL"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Analysis stopped: %s" % error, file=sys.stderr)
        sys.exit(1)
