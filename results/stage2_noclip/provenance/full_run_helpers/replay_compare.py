#!/usr/bin/env python3
"""Exact Stage 2 replay gate. Read trusted local pickle files only.

The pristine probes must be complete. An R0 prefix may be supplied while training
continues; ``r0_passed`` remains null until all 20 iterations have been compared.
No floating-point tolerance, key intersection, or automatic log realignment is
used. New metrics have an explicit schema and never enter the probe equality mask.
"""

import argparse
import json
import math
from pathlib import Path
import pickle
import re
import statistics


TRAIN_KEYS = {"itr", "step", "time", "train_episode_reward"}
EVAL_KEYS = {"itr", "step", "time", "eval_success_rate", "eval_episode_reward", "eval_best_reward"}
REQUIRED_NEW_KEYS = {"kl_true_per_action", "logratio_p99", "clamp_hit_frac", "approx_kl", "clipfrac"}
OPTIONAL_NEW_KEYS = {"ratio", "actor_optimizer_steps", "critic_optimizer_steps", "actor_lr", "critic_lr", "actor_step_ratio", "actor_step_clipfrac"}
NEW_KEYS = REQUIRED_NEW_KEYS | OPTIONAL_NEW_KEYS
DIAGNOSTIC_LOG_KEYS = {"kl_true_per_action", "logratio_p99", "clamp_hit_frac"}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
NUM_RE = re.compile(r"(?<![\w.])(" + NUMBER + r")(?![\w.])", re.I)
TIMESTAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+\]")
LOG_PREFIX = re.compile(r"^(?:\[[^\]\n]+\])+ - ")
TRAIN_SUMMARY = re.compile(r"^(\d+): step ")
CHECKPOINT = re.compile(r"^Saved model to (.+)/checkpoint/state_(\d+)\.pt$")
CLOCK = re.compile(r" \| t:\s*(?:" + NUMBER + r")$", re.I)
NEW_TOKEN = re.compile(r"^([a-z_][a-z_0-9]*)(?:=|:\s*|\s+)(" + NUMBER + r")$", re.I)


class ReplayInputError(ValueError):
    """Malformed or incomplete reference evidence, not a scientific mismatch."""

    def __init__(self, message, first_difference=None):
        super().__init__(message)
        self.first_difference = first_difference


def _scalar(value):
    """Preserve exact scalar values, reject unapproved containers/shapes."""
    if hasattr(value, "shape") and value.shape != ():
        raise ReplayInputError("Expected scalar; found shape %r" % (value.shape,))
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayInputError("Expected numeric scalar; found %r" % (type(value).__name__,))
    if not math.isfinite(value):
        raise ReplayInputError("Nonfinite value: %r" % value)
    return value


def _payload(raw):
    without_time = TIMESTAMP.sub("", raw.rstrip("\r\n"))
    match = LOG_PREFIX.match(without_time)
    return (match.group(0), without_time[match.end():]) if match else ("", without_time)


def _event(line, line_number):
    """Split every numeric token so nondeterministic masks are per value."""
    tokens = NUM_RE.findall(line)
    for token in tokens:
        if not math.isfinite(float(token)):
            raise ReplayInputError("Nonfinite log token at line %d: %s" % (line_number, token))
    return {"template": NUM_RE.sub("<NUMBER>", line), "tokens": tokens, "line_number": line_number, "canonical": line}


def _load_rows(path, role):
    with open(path, "rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list):
        raise ReplayInputError("%s result is not a list" % role)
    if role != "R0" and len(rows) != 20:
        raise ReplayInputError("%s must have exactly 20 rows; found %d" % (role, len(rows)))
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("itr") != index:
            raise ReplayInputError("%s has missing/reordered iteration %d" % (role, index))
        legacy = EVAL_KEYS if index % 10 == 0 else TRAIN_KEYS
        missing = legacy - set(row)
        allowed = legacy | (NEW_KEYS if role == "R0" and index % 10 else set())
        unexpected = set(row) - allowed
        if missing or unexpected:
            raise ReplayInputError("%s itr %d missing keys %s; unexpected keys %s" % (role, index, sorted(missing), sorted(unexpected)), {"iteration": index, "source": "result_schema", "role": role, "missing": sorted(missing), "unexpected": sorted(unexpected)})
        if role == "R0" and index % 10 and REQUIRED_NEW_KEYS - set(row):
            raise ReplayInputError("R0 itr %d missing additive keys %s" % (index, sorted(REQUIRED_NEW_KEYS - set(row))))
        values = {}
        for key, value in row.items():
            if key == "time":
                continue
            try:
                values[key] = _scalar(value)
            except ReplayInputError as error:
                raise ReplayInputError(str(error), {"iteration": index, "source": "result", "role": role, "field": key}) from error
        expected_step = (index - index // 10) * 80000
        if values["step"] != expected_step:
            raise ReplayInputError("%s itr %d invalid cumulative step %r" % (role, index, values["step"]))
        if role == "R0" and index % 10:
            for fraction in ("clamp_hit_frac", "clipfrac", "actor_step_clipfrac"):
                if fraction in values and not 0 <= values[fraction] <= 1:
                    raise ReplayInputError("R0 itr %d invalid %s" % (index, fraction))
            for key in ("kl_true_per_action", "logratio_p99"):
                if values[key] < 0:
                    raise ReplayInputError("R0 itr %d negative %s" % (index, key))
        normalized.append(values)
    return normalized


def _load_log(path, role, row_count):
    """Parse unbuffered complete console output into strict iteration blocks."""
    text = Path(path).read_text(errors="strict")
    iterations = []
    current = []
    checkpoints = []
    exceptions = []
    startup_additions = []
    active = False
    for line_number, raw in enumerate(text.splitlines(), 1):
        prefix, payload = _payload(raw)
        if not active:
            if payload == "Processed step 0 of 500":
                active = True
            else:
                if payload.startswith("stage2_flags "):
                    startup_additions.append({"line_number": line_number, "payload": payload})
                continue
        iteration = len(iterations)
        if iteration >= min(row_count, 20):
            break
        match = CHECKPOINT.fullmatch(payload)
        if match:
            saved_itr = int(match.group(2))
            if saved_itr != iteration:
                raise ReplayInputError("%s checkpoint iteration disagrees with log position" % role)
            checkpoints.append(saved_itr)
            exceptions.append({"kind": "checkpoint_artifact_root", "iteration": iteration, "original": match.group(1), "line_number": line_number})
            if role != "R0" and saved_itr == 19:
                exceptions.append({"kind": "probe_terminal_checkpoint", "iteration": 19, "line_number": line_number, "payload": payload})
                continue
            payload = "Saved model to <RUN_ROOT>/checkpoint/state_%d.pt" % saved_itr
        is_eval = payload.startswith("eval: ")
        train_match = TRAIN_SUMMARY.match(payload)
        additions = {}
        if train_match:
            if int(train_match.group(1)) != iteration or iteration % 10 == 0:
                raise ReplayInputError("%s misaligned train summary at line %d" % (role, line_number))
            parts = payload.split(" | ")
            legacy_parts = []
            for part in parts:
                new_match = NEW_TOKEN.fullmatch(part.strip())
                if new_match and new_match.group(1) in NEW_KEYS:
                    key, token = new_match.groups()
                    if role != "R0" or key in additions:
                        raise ReplayInputError("Unexpected/duplicate additive log field %s at line %d" % (key, line_number))
                    if not math.isfinite(float(token)):
                        raise ReplayInputError("Nonfinite additive log field at line %d" % line_number)
                    additions[key] = token
                else:
                    legacy_parts.append(part)
            payload = " | ".join(legacy_parts)
            if not CLOCK.search(payload):
                raise ReplayInputError("%s missing terminal timing token at line %d" % (role, line_number))
            payload = CLOCK.sub("", payload)
            if role == "R0" and DIAGNOSTIC_LOG_KEYS - set(additions):
                raise ReplayInputError("R0 itr %d missing logged diagnostics %s" % (iteration, sorted(DIAGNOSTIC_LOG_KEYS - set(additions))))
            if additions:
                exceptions.append({"kind": "additive_iteration_metrics", "iteration": iteration, "values": additions, "line_number": line_number})
        if is_eval and iteration % 10:
            raise ReplayInputError("%s misaligned evaluation summary" % role)
        current.append(_event(prefix + payload, line_number))
        if train_match or is_eval:
            processed = [event["canonical"] for event in current if event["canonical"].startswith("Processed step ")]
            expected = ["Processed step %d of 500" % step for step in range(0, 500, 10)]
            if processed != expected:
                raise ReplayInputError("%s itr %d missing/reordered rollout progress lines" % (role, iteration))
            iterations.append(current)
            current = []
    if len(iterations) != min(row_count, 20):
        raise ReplayInputError("%s has %d result rows but only %d complete console iterations" % (role, min(row_count, 20), len(iterations)))
    expected_checkpoints = ([0, 19] if role != "R0" else [0]) if iterations else []
    if checkpoints != expected_checkpoints:
        raise ReplayInputError("%s checkpoints in replay interval %r, expected %r" % (role, checkpoints, expected_checkpoints))
    if role != "R0" and startup_additions:
        raise ReplayInputError("%s pristine probe printed new flags" % role)
    if role == "R0" and iterations:
        flag_lines = [item["payload"] for item in startup_additions]
        expected = "stage2_flags clamp_logprob=True logprob_reduce=mean actor_single_step=False"
        if flag_lines != [expected]:
            raise ReplayInputError("R0 default startup flags differ from the declared single printout: %r" % flag_lines)
    return {"iterations": iterations, "exceptions": exceptions, "startup_additions": startup_additions}


def _run(result_path, log_path, role):
    rows = _load_rows(result_path, role)
    logs = _load_log(log_path, role, len(rows))
    return {"rows": rows, "logs": logs, "result_path": str(result_path), "log_path": str(log_path)}


def _flatten(run, count):
    values = {}
    for iteration in range(count):
        row = run["rows"][iteration]
        for key in sorted((EVAL_KEYS if iteration % 10 == 0 else TRAIN_KEYS) - {"time"}):
            values[(iteration, "result", key)] = row[key]
        events = run["logs"]["iterations"][iteration]
        values[(iteration, "log_structure", "event_count")] = len(events)
        for ordinal, event in enumerate(events):
            values[(iteration, "log_structure", "%03d.template" % ordinal)] = event["template"]
            for token_index, token in enumerate(event["tokens"]):
                values[(iteration, "log_token", "%03d.token_%02d" % (ordinal, token_index))] = token
    return values


def _location(key, **values):
    output = {"iteration": key[0], "source": key[1], "field": key[2]}
    output.update(values)
    return output


def _same_keys(left, right, left_name, right_name):
    missing = set(left) ^ set(right)
    if missing:
        key = min(missing)
        detail = _location(key, **{left_name: left.get(key, "<missing>"), right_name: right.get(key, "<missing>")})
        raise ReplayInputError("%s/%s output structure differs; no invented alignment is permitted" % (left_name, right_name), detail)


def _gross_deviations(r0_rows, baseline_results):
    if len(baseline_results) != 3:
        raise ReplayInputError("Gross-deviation reference needs exactly three Stage 1 runs")
    references = []
    for path in baseline_results:
        with open(path, "rb") as stream:
            rows = pickle.load(stream)
        reference = {}
        for row in rows:
            if "eval_episode_reward" in row:
                step = _scalar(row["step"])
                if step in reference:
                    raise ReplayInputError("Duplicate Stage 1 evaluation step")
                reference[step] = _scalar(row["eval_episode_reward"])
        references.append(reference)
    checks = []
    for row in r0_rows:
        if "eval_episode_reward" not in row:
            continue
        step = row["step"]
        if any(step not in reference for reference in references):
            raise ReplayInputError("No three-seed Stage 1 evaluation at R0 step %d" % step)
        sample = [reference[step] for reference in references]
        mean, std = statistics.mean(sample), statistics.stdev(sample)
        lower, upper = mean - 5 * std, mean + 5 * std
        checks.append({"iteration": row["itr"], "step": step, "r0": row["eval_episode_reward"], "baseline_values": sample, "mean": mean, "sample_std_ddof1": std, "lower": lower, "upper": upper, "red_flag": not lower <= row["eval_episode_reward"] <= upper})
    return checks


def compare_replay(p1_result, p1_log, p2_result, p2_log, r0_result=None, r0_log=None, baseline_results=None):
    """Return JSON-safe exact comparisons; raises ReplayInputError for bad schema.

    Reinvoke on new saved R0 rows to enforce early matching-value failures. Call
    with all three Stage 1 result paths to continue nondeterministic gross checks.
    ``r0_passed`` concerns the exact replay gate; red flags are reported separately.
    """
    if (r0_result is None) != (r0_log is None):
        raise ReplayInputError("Supply both R0 result and log paths")
    p1 = _run(p1_result, p1_log, "P1")
    p2 = _run(p2_result, p2_log, "P2")
    left, right = _flatten(p1, 20), _flatten(p2, 20)
    _same_keys(left, right, "P1", "P2")
    masks = []
    probe_differences = []
    for key in sorted(left):
        same = left[key] == right[key]
        if key[1] == "log_structure" and not same:
            detail = _location(key, p1=left[key], p2=right[key])
            raise ReplayInputError("P1/P2 structural mismatch: %s" % detail, detail)
        masks.append(_location(key, equal=same))
        if not same:
            probe_differences.append(_location(key, p1=left[key], p2=right[key]))
    output = {"schema_version": 1, "deterministic_probe": not probe_differences,
              "comparison_status": "probes_compared", "r0_passed": None,
              "probe_comparison": {"iterations": 20, "value_count": len(left), "matching_value_count": len(left) - len(probe_differences), "differing_value_count": len(probe_differences), "first_difference": probe_differences[0] if probe_differences else None, "differences": probe_differences, "equality_mask": masks},
              "exceptions": {"P1": p1["logs"]["exceptions"], "P2": p2["logs"]["exceptions"]},
              "schemas": {"legacy_train": sorted(TRAIN_KEYS), "legacy_eval": sorted(EVAL_KEYS), "r0_required_additions": sorted(REQUIRED_NEW_KEYS), "r0_optional_additions": sorted(OPTIONAL_NEW_KEYS)},
              "excluded_timing": ["console timestamp prefixes", "training summary t: token", "result.pkl time field"],
              "gross_eval_checks": [], "gross_eval_red_flags": []}
    if r0_result is None:
        return output
    r0 = _run(r0_result, r0_log, "R0")
    count = min(len(r0["rows"]), 20)
    actual = _flatten(r0, count)
    expected = {key: value for key, value in left.items() if key[0] < count}
    _same_keys(expected, actual, "P1", "R0")
    differences = []
    enforced_count = 0
    unconstrained_count = 0
    for key in sorted(actual):
        if key[1] == "log_structure" and actual[key] != left[key]:
            detail = _location(key, p1=left[key], p2=right[key], r0=actual[key])
            raise ReplayInputError("R0 structural mismatch: %s" % detail, detail)
        if left[key] == right[key]:
            enforced_count += 1
            if actual[key] != left[key]:
                differences.append(_location(key, p1=left[key], p2=right[key], r0=actual[key]))
        else:
            unconstrained_count += 1
    output["r0_passed"] = False if differences else (True if count == 20 else None)
    output["comparison_status"] = "r0_failed" if differences else ("r0_passed" if count == 20 else "waiting_r0")
    output["r0_comparison"] = {"iterations": count, "last_compared_iteration": count - 1, "enforced_value_count": enforced_count, "unconstrained_value_count": unconstrained_count, "mismatch_count": len(differences), "first_difference": differences[0] if differences else None, "differences": differences}
    output["exceptions"]["R0"] = r0["logs"]["exceptions"]
    output["r0_startup_additions"] = r0["logs"]["startup_additions"]
    output["r0_new_metrics"] = [{"iteration": row["itr"], "values": {key: row[key] for key in sorted(NEW_KEYS & set(row))}} for row in r0["rows"] if row["itr"] % 10]
    if probe_differences and baseline_results is not None:
        output["gross_eval_checks"] = _gross_deviations(r0["rows"], baseline_results)
        output["gross_eval_red_flags"] = [check for check in output["gross_eval_checks"] if check["red_flag"]]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p1-result", "p1-log", "p2-result", "p2-log"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--r0-result")
    parser.add_argument("--r0-log")
    parser.add_argument("--baseline-result", action="append", dest="baseline_results")
    parser.add_argument("--output", help="JSON file; refuses to overwrite existing evidence")
    args = parser.parse_args()
    arguments = vars(args).copy()
    output_path = arguments.pop("output")
    try:
        report = compare_replay(**arguments)
        exit_code = 1 if report["r0_passed"] is False else 0
    except (ReplayInputError, OSError, EOFError, pickle.UnpicklingError) as error:
        report = {"schema_version": 1, "comparison_status": "invalid_input", "r0_passed": False, "error": str(error), "first_difference": getattr(error, "first_difference", None)}
        exit_code = 2
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output_path:
        with open(output_path, "x") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
