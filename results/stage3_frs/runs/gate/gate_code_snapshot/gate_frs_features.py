#!/usr/bin/env python
"""FRS algebra/state checks using the full saved-batch actor gradient."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from gate_common import compare, freeze, numeric_settings, require, rng_comparison, rng_state


def norm_errors(reference, observed):
    """Numerical reports have no denominator floor hiding small coordinates."""
    x = torch.cat([value.detach().cpu().double().reshape(-1) for value in reference])
    y = torch.cat([value.detach().cpu().double().reshape(-1) for value in observed])
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), "nonfinite formula evidence")
    delta = y - x
    xnorm = torch.linalg.vector_norm(x).item()
    dnorm = torch.linalg.vector_norm(delta).item()
    nonzero = x != 0
    max_relative = (delta[nonzero].abs() / x[nonzero].abs()).max().item() if bool(nonzero.any()) else 0.
    zero_mismatch = int(((~nonzero) & (delta != 0)).sum())
    if zero_mismatch:
        max_relative = "inf (nonzero error where reference equals zero)"
    relative_l2 = dnorm / xnorm if xnorm else (0. if dnorm == 0 else "inf")
    return {"max_abs": delta.abs().max().item(), "max_rel": max_relative,
            "relative_l2": relative_l2, "error_l2": dnorm, "reference_l2": xnorm,
            "zero_reference_mismatches": zero_mismatch, "coordinates": x.numel()}


def parameters_from(evidence, specs, device):
    return [(spec["name"], torch.nn.Parameter(
        evidence["initial_model"]["actor_ft." + spec["name"]].to(device).clone())) for spec in specs]


def set_gradients(named, raw):
    for name, parameter in named:
        value = raw[name]
        parameter.grad = value.to(parameter.device).clone() if value is not None else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", "--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--rescale-path", type=Path, required=True)
    parser.add_argument("--replay-evidence", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "feature output must be new")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "passed": False, "command": sys.argv,
              "device": args.device, "numeric_settings_before": numeric_settings(),
              "input_evidence": str(args.replay_evidence.resolve()),
              "rescale_path": str(args.rescale_path.resolve())}
    try:
        require(not torch.are_deterministic_algorithms_enabled(), "deterministic algorithms unexpectedly enabled")
        sys.path.insert(0, str(args.repo.resolve()))
        from agent.finetune.frs import FrozenRescaledSGD, build_rescale_artifact
        artifact = torch.load(args.rescale_path, map_location="cpu", weights_only=True)
        evidence = torch.load(args.replay_evidence, map_location="cpu", weights_only=False)
        specs = artifact["parameter_specs"]
        names = [spec["name"] for spec in specs]
        require(list(evidence["actor_parameters"]) == names, "artifact parameter order differs from actual actor")
        require(len(evidence["actor_gradients"]) == 1, "formula check requires exactly one actor step")
        raw = evidence["actor_gradients"][0]
        require(list(raw) == names, "saved gradient parameter order differs")
        group = evidence["initial_actor_optimizer"]["param_groups"][0]
        lr = float(group["lr"])
        require(group["momentum"] == 0 and group["weight_decay"] == 0,
                "formula requires the prescribed plain SGD settings")
        report["learning_rate"] = lr
        report["raw_gradient_source"] = "captured immediately before actual full-batch FRS optimizer step"

        # Reconstruct the floor/epsilon and native-dtype cast independently.
        flat_r = np.concatenate([artifact["r"][name].numpy().reshape(-1) for name in names])
        median_r = float(np.median(flat_r))
        require(median_r == artifact["global_median_r"], "global midpoint median mismatch")
        expected_denominators = {
            name: torch.from_numpy(np.maximum(artifact["r"][name].numpy(), 1e-3 * median_r) + 1e-8)
                       .to(artifact["r_tilde"][name].dtype) for name in names}
        report["denominator_formula"] = compare(expected_denominators, artifact["r_tilde"], True)
        require(report["denominator_formula"]["passed"], "floor/epsilon denominator mismatch")

        # Identity denominators exercise the full native actor geometry and saved
        # full-batch gradient, not a scalar toy optimizer.
        plain = parameters_from(evidence, specs, args.device)
        identity = parameters_from(evidence, specs, args.device)
        identity_artifact = build_rescale_artifact(
            identity, {name: torch.full(parameter.shape, 1. - 1e-8, dtype=torch.float64)
                       for name, parameter in identity}, floor=1e-3,
            metadata={"purpose": "gate exact denominator-one identity"})
        require(all(torch.equal(value, torch.ones_like(value))
                    for value in identity_artifact["r_tilde"].values()),
                "identity gate did not produce exact denominator one")
        identity_path = output / "denominator_one.pt"
        torch.save(identity_artifact, identity_path)
        opt_plain = torch.optim.SGD([p for _, p in plain], lr=lr,
                                    momentum=0, weight_decay=0)
        opt_identity = FrozenRescaledSGD(identity, lr=lr, rescale_path=identity_path, floor=1e-3)
        set_gradients(plain, raw)
        set_gradients(identity, raw)
        rng_before = rng_state()
        opt_plain.step()
        rng_middle = rng_state()
        opt_identity.step()
        rng_after = rng_state()
        report["denominator_one_plain_sgd"] = compare(
            dict(plain), dict(identity), args.device == "cpu", "all_actor_parameters")
        report["denominator_one_gradients"] = compare(
            [p.grad for _, p in plain], [p.grad for _, p in identity], args.device == "cpu", "actor_gradients")
        report["identity_rng_inert"] = {
            "plain": rng_comparison(rng_before, rng_middle),
            "rescaled": rng_comparison(rng_middle, rng_after)}
        require(report["denominator_one_plain_sgd"]["passed"] and
                report["denominator_one_gradients"]["passed"] and
                all(v["passed"] for v in report["identity_rng_inert"].values()),
                "denominator-one FRS does not reproduce plain SGD at the device's required tolerance")

        named = parameters_from(evidence, specs, args.device)
        optimizer = FrozenRescaledSGD(named, lr=lr, rescale_path=args.rescale_path, floor=1e-3)
        set_gradients(named, raw)
        before = {"parameters": freeze(dict(named)), "gradients": freeze([p.grad for _, p in named]),
                  "optimizer": freeze(optimizer.state_dict())}
        before_rng = rng_state()
        metrics = optimizer.ratio_metrics()
        after_rng = rng_state()
        after = {"parameters": freeze(dict(named)), "gradients": freeze([p.grad for _, p in named]),
                 "optimizer": freeze(optimizer.state_dict())}
        report["ratio_diagnostic_state_inert"] = compare(before, after, True)
        report["ratio_diagnostic_rng_inert"] = rng_comparison(before_rng, after_rng)
        require(report["ratio_diagnostic_state_inert"]["passed"] and
                report["ratio_diagnostic_rng_inert"]["passed"],
                "ratio diagnostic changes parameters/gradients/optimizer/global RNG")
        report["ratio_metrics"] = metrics
        expected_ratios = torch.cat([
            (raw[name].to(args.device) / artifact["r_tilde"][name].to(args.device)).abs().reshape(-1).double()
            if raw[name] is not None else torch.zeros(spec["numel"], device=args.device, dtype=torch.float64)
            for name, spec in zip(names, specs)])
        expected_metrics = {"max_ratio": expected_ratios.max().item(),
                            "median_ratio": float(np.median(expected_ratios.cpu().numpy()))}
        report["ratio_metrics_formula"] = compare(expected_metrics, metrics, True)
        require(report["ratio_metrics_formula"]["passed"], "ratio diagnostic formula mismatch")

        # Independent float64 reference specifies the mathematical update; do not
        # confuse parameter rounding with equality of subtracted displacements.
        origin = [evidence["initial_model"]["actor_ft." + name].double() for name in names]
        expected_delta = [(-lr * raw[name].double() / artifact["r_tilde"][name].double())
                          if raw[name] is not None else torch.zeros_like(x)
                          for name, x in zip(names, origin)]
        expected_theta = [x + d for x, d in zip(origin, expected_delta)]
        frozen_before = freeze(optimizer.state_dict()["frs"]["denominators"])
        rng_before = rng_state()
        optimizer.step()
        report["frs_step_rng_inert"] = rng_comparison(rng_before, rng_state())
        report["frozen_denominators"] = compare(
            frozen_before, freeze(optimizer.state_dict()["frs"]["denominators"]), True)
        report["step_ratio_metrics"] = compare(metrics, optimizer.last_ratio_metrics, True)
        observed_theta = [p.detach().cpu().double() for _, p in named]
        report["mathematical_update"] = norm_errors(expected_theta, observed_theta)
        report["mathematical_update"]["criterion_relative_l2"] = 1e-7
        actual_delta = [y - x for x, y in zip(origin, observed_theta)]
        report["displacement_roundoff"] = norm_errors(expected_delta, actual_delta)
        report["displacement_roundoff"]["interpretation"] = (
            "Report only: subtracting stored float32 parameters loses small update bits. "
            "The approved 1e-7 relative-L2 criterion applies to updated parameters, not these subtractions.")
        report["actual_training_update"] = compare(
            evidence["actor_parameters"], freeze(dict(named)), args.device == "cpu",
            "full_batch_training_vs_direct_optimizer")
        replay_theta = [evidence["actor_parameters"][name].double() for name in names]
        report["actual_training_formula"] = norm_errors(expected_theta, replay_theta)
        report["actual_training_formula"]["criterion_relative_l2"] = 1e-7
        report["training_frozen_denominators"] = compare(
            evidence["initial_actor_optimizer"]["frs"]["denominators"],
            evidence["actor_optimizer_state"]["frs"]["denominators"], True)
        report["training_logged_ratio_metrics"] = compare(
            expected_metrics, {key: evidence["logged_values"]["actor_step_metrics"][key]
                               for key in expected_metrics}, args.device == "cpu")
        require(report["mathematical_update"]["relative_l2"] <= 1e-7 and
                report["actual_training_formula"]["relative_l2"] <= 1e-7,
                "FRS update differs from theta - eta * raw_loss_gradient / denominator")
        require(all(report[key]["passed"] for key in (
            "frs_step_rng_inert", "frozen_denominators", "step_ratio_metrics", "actual_training_update",
            "training_frozen_denominators", "training_logged_ratio_metrics")),
            "FRS optimizer integration/immutable denominator/diagnostic mismatch")
        report["status"], report["passed"] = "passed", True
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["numeric_settings_after"] = numeric_settings()
        (output / "features.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
