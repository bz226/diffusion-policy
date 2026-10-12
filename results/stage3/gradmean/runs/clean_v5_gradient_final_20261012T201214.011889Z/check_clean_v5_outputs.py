"""Independently check gradient-only V5 final artifacts and raw gradient moments.

Python 3.8. Reads evidence and recreates axes in memory only; no sampling or
figure publication. Exit0 means all checks pass; optional --require-complete
additionally requires all five 500-batch points.
"""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t


METHODS = ("DPPO", "without_PPO", "without_PPO_and_GAE")
POINTS = ("W0_000", "W0_035", "W0_070", "W0_105", "W0_139")
UPDATES = (0, 32, 63, 95, 126)


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, message):
    _assert(actual is not None and math.isfinite(float(actual)) and
            np.isclose(actual, expected, rtol=2e-7, atol=1e-10), message)


def _jk(value, deleted):
    if value is None:
        return dict(value=None, lo=None, hi=None, se=None)
    if any(item is None for item in deleted):
        return dict(value=value, lo=None, hi=None, se=None)
    samples = np.asarray(deleted, dtype=np.float64)
    n = len(samples)
    se = math.sqrt((n - 1.0) / n * np.square(samples - samples.mean()).sum())
    critical = student_t.ppf(.95, n - 1)
    return dict(value=value, se=se, lo=value - critical * se, hi=value + critical * se)


def _compare(actual, expected, label):
    for field in ("value", "lo", "hi", "se"):
        if expected[field] is None:
            _assert(actual.get(field) is None, label + " wrongly fills unavailable " + field)
        else:
            _close(actual.get(field), expected[field], label + " " + field)


def verify_point(point):
    """Recompute own V2/V6 U statistics directly, independently of PointData."""
    _assert(point["K"] == 500 and point["m"] == 40000 and point["groups"] == 10,
            "Wrong measurement cohort")
    jobs = point["selected_jobs"]
    _assert(len(jobs) == 10 and len({job["run_id"] for job in jobs}) == 10, "Repeated/missing group")
    groups, moments, ev = [], [], []
    for job in jobs:
        path = Path(job["path"])
        meta = json.loads((path / "meta.json").read_text())
        ids = meta["variant_ids"]
        indices = [ids.index(name) for name in ("V2", "V6")]
        groups.append(np.stack([np.load(str(path / ("sums_" + name + ".npy")), allow_pickle=False).astype(np.float64)
                                for name in ("V2", "V6")]))
        moments.append(np.asarray(meta["gradient_cross_sum"], dtype=np.float64)[np.ix_(indices, indices)])
        with (path / "batches.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        _assert(len(rows) == 50, "Incomplete scalar batch file")
        ev.append(np.mean([float(row["explained_variance"]) for row in rows]))
    groups, moments = np.stack(groups), np.stack(moments)
    def compute(indices):
        k = len(indices) * 50
        total = groups[indices].sum(axis=0, dtype=np.float64)
        q = moments[indices].sum(axis=0, dtype=np.float64)
        product = total @ total.T
        return ((product - q) / (k * (k - 1)),
                (np.diag(q) - np.diag(product) / k) / (k - 1))
    full = compute(np.arange(10))
    deleted = [compute(np.delete(np.arange(10), i)) for i in range(10)]
    for index, (variant, multiplier) in enumerate((("V2", 1.), ("V6", 2500.))):
        u = _jk(full[0][index, index], [m[0][index, index] for m in deleted])
        trace = _jk(full[1][index], [m[1][index] for m in deleted])
        metric = point["estimators"][variant]
        for source, name, scale in ((u, "u_sq", multiplier ** 2), (trace, "trace_ep", 80 * multiplier ** 2)):
            _compare(metric[name], {key: value * scale if value is not None else None for key, value in source.items()}, variant + " " + name)
        norm = _jk(math.sqrt(u["value"]) if u["value"] > 0 else None,
                   [math.sqrt(m[0][index, index]) if m[0][index, index] > 0 else None for m in deleted])
        _compare(metric["u_norm"], {key: value * multiplier if value is not None else None for key, value in norm.items()}, variant + " norm")
        if u["value"] > 0 and u["lo"] > 0:
            b = _jk(80 * trace["value"] / u["value"],
                    [80 * m[1][index] / m[0][index, index] if m[0][index, index] > 0 else None for m in deleted])
            _compare(metric["B_ep"], b, variant + " own-signal B")
            _compare(metric["B_update"], {key: value / 80 if value is not None else None for key, value in b.items()}, variant + " update B")
        else:
            _assert(metric["B_ep"]["value"] is None, "Unresolved own signal rendered as resolved B")
            if u["hi"] > 0:
                _close(metric["B_ep"]["lower_bound"], 80 * trace["value"] / u["hi"], variant + " B bound")
        if variant == "V2":
            predicted = point["predicted_consecutive_cos"]
            if u["value"] > 0 and u["lo"] > 0:
                prediction = _jk(u["value"] / (u["value"] + trace["value"]),
                    [m[0][index, index] / (m[0][index, index] + m[1][index]) if m[0][index, index] > 0 else None for m in deleted])
                _compare(predicted, prediction, "Predicted consecutive cosine")
            else:
                _assert(predicted["value"] is None, "Unresolved prediction rendered as point")
                if u["hi"] > 0:
                    _close(predicted["upper_bound"], 1 / (1 + trace["value"] / u["hi"]), "Predicted cosine bound")
    def cosine(m):
        u = m[0]
        return u[0, 1] / math.sqrt(u[0, 0] * u[1, 1]) if u[0, 0] > 0 and u[1, 1] > 0 else None
    _compare(point["cos_V2_V6"], _jk(cosine(full), [cosine(m) for m in deleted]), "Own-mean direction cosine")
    _compare(point["explained_var"], _jk(float(np.mean(ev)), [float(np.delete(ev, i).mean()) for i in range(10)]), "Explained variance")
    prepared = json.loads(Path(point["divisor_source"]["prepare_metadata"]).read_text())
    if prepared["scaler_c_fallback_batches"]:
        fit = prepared["scaler_fit_batches"]
        _assert(len(fit) == 5 and prepared["scaler_c_fallback_batches"] == 5, "Scale fit did not use five batches")
        n = np.array([row["sum_VG"] for row in fit], dtype=np.float64)
        d = np.array([row["sum_V2"] for row in fit], dtype=np.float64)
        values = [np.delete(n, i).sum() / np.delete(d, i).sum() if np.delete(d, i).sum() > 0 else None for i in range(5)]
        _compare(point["divisor"], _jk(n.sum() / d.sum(), values), "Fitted scale five-batch t4 interval")
    else:
        _compare(point["divisor"], dict(value=prepared["effective_scaler_c"], lo=None, hi=None, se=None), "Fixed logged scale")
    return {"groups": 10, "K": 500, "own_V2_and_V6_signals_verified": True}


def verify_axes(record, output):
    """Exercise the saved renderer using saved metrics, without writing figures."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection

    path = Path(record["run"]) / "plot_clean.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("clean_v5_checked_renderer", str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checked = []
    def capture(fig, unused_output, name, caption, unused_plt):
        _assert(caption.strip() == (output / (name + ".caption.md")).read_text().strip(), "Saved caption differs from rendering source")
        _assert(all(not axis.child_axes for axis in fig.axes), "Secondary/top axes unexpectedly present")
        if name in ("eval_reward", "train_reward"):
            _assert(len(fig.axes) == 1, "Reward figure must have one axis")
            axis = fig.axes[0]
            _assert("env steps" in axis.get_xlabel(), "Reward x axis is not environment steps")
            _assert(not any(isinstance(item, PolyCollection) for item in axis.collections), "Reward seed bands unexpectedly present")
            lines = {line.get_label(): line for line in axis.lines if line.get_label() in METHODS}
            _assert(set(lines) == set(METHODS), "Wrong reward methods")
            sampler = "eval" if name.startswith("eval") else "train"
            for method in METHODS:
                rows = record["rewards"]["curves"][method][sampler]
                _assert(np.array_equal(lines[method].get_xdata(), [row["env_steps"] / 1e6 for row in rows]), "Reward x values changed")
                _assert(np.array_equal(lines[method].get_ydata(), [record["rewards"]["pooled_start"][sampler]["mean"]] + [row["reward"] for row in rows[1:]]), "Reward y values changed")
            _assert(record["rewards"]["pooled_start"][sampler]["se"] is None, "Unavailable episode SE filled")
        else:
            _assert(len(fig.axes) == 3, "Noise figure needs exactly three panels")
            variant = "V2" if name == "gradient_noise_as_run" else "V6"
            for axis, field in zip(fig.axes, ("u_norm", "trace_ep", "B_ep")):
                _assert(axis.get_yscale() == "log" and "Updates" in axis.get_xlabel(), "Noise axes changed")
                _assert(axis.get_legend() is None, "An overlay/legend was added")
                lines = [line for line in axis.lines if len(line.get_xdata()) == 5]
                _assert(len(lines) == 1 and np.array_equal(lines[0].get_xdata(), UPDATES), "Extra estimator path/overlay or wrong noise x")
                expected = []
                for point in POINTS:
                    item = record["statistics"]["points"].get(point)
                    value = item["estimators"][variant][field]["value"] if item else None
                    expected.append(value if value is not None and value > 0 else np.nan)
                _assert(np.allclose(lines[0].get_ydata(), expected, equal_nan=True, rtol=0, atol=0), "Wrong noise metric rendered")
        checked.append(name); plt.close(fig)
    module.save_figure = capture
    # Reward figures are preserved, not rendered by this finalization.
    for variant in ("V2", "V6"):
        module.plot_noise(record["statistics"], variant, output, plt)
    return checked


def validate(output, require_complete=False):
    from PIL import Image
    output = Path(output).resolve()
    record = json.loads((output / "render_record.json").read_text())
    _assert(record["version"] == 5 and record["simulation_launched_by_renderer"] is False, "Wrong render version/type")
    points = record["statistics"]["points"]
    if require_complete:
        _assert(set(points) == set(POINTS) and record["status"] == "complete", "Final artifacts lack five complete points")
    _assert(record.get("artifacts_scope") == "gradient figures and table only" and record.get("reward_artifacts_rendered") is False, "Unexpected finalizer scope")
    names = ["gradient_noise_as_run", "gradient_noise_paper_form"]
    for name in names:
        for suffix in (".png", ".pdf", ".caption.md"):
            _assert((output / (name + suffix)).is_file(), "Missing artifact: " + name + suffix)
        caption = (output / (name + ".caption.md")).read_text().strip()
        _assert(all(label not in caption for label in ("NC4", "E1C", "noise floor 0.001", "deterministic sampler")), "Forbidden/misleading caption label")
        with Image.open(str(output / (name + ".png"))) as image:
            _assert(image.info.get("Description") == caption and all(abs(value - 300) < .1 for value in image.info.get("dpi", (0, 0))), "PNG caption/dpi mismatch")
        _assert((output / (name + ".pdf")).read_bytes().startswith(b"%PDF"), "Invalid PDF artifact")
    with (output / "points_table.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    _assert(len(rows) == 5 and [int(row["update_number"]) for row in rows] == list(UPDATES), "Table rows/update mapping wrong")
    for row, point in zip(rows, POINTS):
        _assert(row["method"] == "without_PPO_and_GAE", "Wrong table method")
        measured = points.get(point)
        _assert(int(row["K"]) == (500 if measured else 0) and int(row["m"]) == (40000 if measured else 0), "Table cohort size wrong")
        if measured:
            for variant, prefix in (("V2", "as_run"), ("V6", "paper_form")):
                for field in ("u_norm", "trace_ep", "B_ep", "B_update"):
                    expected = measured["estimators"][variant][field]["value"]
                    actual = row[prefix + "_" + field]
                    _assert(actual == "" if expected is None else float(actual) == expected, "CSV metric differs from render evidence")
            _close(float(row["c"]), measured["divisor"]["value"], "Table divisor")
            for field in ("lo", "hi", "se"):
                key = "c_" + field
                expected = measured["divisor"].get(field)
                _assert(row[key] == "" if expected is None else float(row[key]) == expected, "CSV divisor interval differs from saved metric")
    for name in ("points_table.md", "NOTE.md"):
        _assert((output / name).is_file(), "Missing table/note")
    numeric = {name: verify_point(point) for name, point in points.items()}
    axes = verify_axes(record, output)
    return dict(passed=True, version=5, complete=set(points) == set(POINTS), points=list(points),
                numeric_checks=numeric, figures_checked=axes, source=str(output),
                sampling_performed=False, figures_written=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = validate(args.output_dir, args.require_complete)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
