#!/usr/bin/env python
"""Authorized precision-aware G3 amendment; no rollouts or source changes.

The native optimizer uses Tensor.add_(gradient, alpha=scale). For float32
parameters, ATen converts alpha to the float32 scalar type. The implicit update
is reconstructed from that scalar and the float32 gradient in float64; it is
not materialized through a separately rounded float32 multiply-then-add path.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from gate_adagrad_norm import EDITED, STUDY, compare, require, rng_state, write


def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    preliminary = json.loads((args.preliminary / 'gate_summary.json').read_text())
    previous = json.loads((args.previous / 'gate_summary.json').read_text())
    prior_seconds = args.prior_seconds
    report = {
        'status': 'running', 'passed': False, 'checks': {}, 'cpu_only': True,
        'source': str(args.source), 'scientific_source_revision': '2acb18e9629c577474ee7f1652c7cc6f5a3829ac',
        'preliminary': str(args.preliminary), 'previous_failed_G3': str(args.previous),
        'eta': args.eta, 'b0': args.b0, 'prior_gate_seconds': prior_seconds,
        'previous_internal_aggregate_seconds': previous['aggregate_gate_seconds'],
        'prior_accounting': 'Conservative prior wall time: 731.9507804671302 s preliminary plus 28 s complete prior G3 Slurm allocation.',
        'cap_seconds': 7200., 'commands': [sys.argv],
        'note': 'G1/G2/G4 and calibration reused; only authorized amended G3a/G3b are executed. G5 is not launched.'}
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['aggregate_gate_seconds'] = prior_seconds + report['elapsed_seconds']
        write(args.output / 'gate_summary.json', report)
    try:
        require(preliminary.get('preliminary_passed'), 'Accepted G1/G2/G4 evidence is required')
        require(prior_seconds < 7200., 'Aggregate CPU gate cap already exhausted')
        require(not torch.cuda.is_available(), 'Amended G3 must run with GPUs hidden')
        sys.path.insert(0, str(args.source))
        from agent.finetune.adagrad_norm import AdaGradNorm
        evidence_path = args.preliminary / 'adagrad_update/evidence.pt'
        evidence = torch.load(evidence_path, map_location='cpu', weights_only=False)
        names = list(evidence['actor_parameters'])
        initial = [evidence['initial_model']['actor_ft.' + name].clone() for name in names]
        gradients = [evidence['actor_gradients'][0][name].clone() for name in names]
        require(all(value.dtype == torch.float32 for value in initial + gradients),
                'The G2-certified native parameters/gradients are not float32')
        report['input'] = {'evidence': str(evidence_path), 'parameters': sum(p.numel() for p in initial),
                           'parameter_tensors': len(initial), 'native_parameter_dtype': 'float32',
                           'native_gradient_dtype': 'float32', 'actor_gradient_G2_bitwise_certified': True}
        full_gradient = torch.cat([g.reshape(-1) for g in gradients]).double()
        gradient_norm = full_gradient.norm().item()
        denominator = (args.b0 ** 2 + gradient_norm ** 2) ** .5
        formula = args.eta * gradient_norm / denominator
        report['gradient_norm'] = gradient_norm
        report['formula_displacement'] = formula

        # G3a: actual optimizer step on the same represented data in float64.
        parameters64 = [torch.nn.Parameter(p.double()) for p in initial]
        for parameter, gradient in zip(parameters64, gradients):
            parameter.grad = gradient.double()
        optimizer64 = AdaGradNorm(parameters64, lr=args.eta, b0=args.b0)
        before64 = torch.cat([p.detach().reshape(-1) for p in parameters64]).clone()
        before_rng = rng_state()
        optimizer64.step()
        after64 = torch.cat([p.detach().reshape(-1) for p in parameters64])
        observed64 = (after64 - before64).norm().item()
        relative64 = abs(observed64 - formula) / formula
        report['checks']['G3a'] = {
            'passed': relative64 <= 1e-10, 'parameter_dtype': 'float64',
            'gradient_dtype': 'float64', 'accumulator_dtype': str(optimizer64.b.dtype),
            'measured_displacement': observed64, 'formula_displacement': formula,
            'absolute_error': abs(observed64 - formula), 'relative_error': relative64,
            'relative_tolerance': 1e-10, 'b_after': optimizer64.b.item(),
            'rng_unchanged': compare(before_rng, rng_state(), True)}
        save()
        require(relative64 <= 1e-10, 'Amended G3a float64 step fails relative 1e-10')
        require(optimizer64.b.dtype == torch.float64, 'G3a accumulator is not float64')
        del parameters64, optimizer64, before64, after64

        # G3b: use the actual native add_ operation, including its alpha operand.
        parameters32 = [torch.nn.Parameter(p.clone()) for p in initial]
        for parameter, gradient in zip(parameters32, gradients):
            parameter.grad = gradient.clone()
        optimizer32 = AdaGradNorm(parameters32, lr=args.eta, b0=args.b0)
        squared = torch.zeros((), dtype=torch.float64)
        for gradient in gradients:
            squared.add_(torch.sum(gradient.double() * gradient.double()))
        predicted_b = torch.sqrt(optimizer32.b * optimizer32.b + squared)
        alpha_python64 = -args.eta / predicted_b.item()
        alpha_native32 = torch.tensor(alpha_python64, dtype=torch.float32).item()
        # Each float32 x float32 product is represented exactly in float64 here.
        implicit_update = full_gradient * alpha_native32
        implicit_length = implicit_update.norm().item()
        relative_update_length = abs(implicit_length - formula) / formula
        before32 = torch.cat([p.detach().reshape(-1) for p in parameters32]).double()
        before_rng = rng_state()
        optimizer32.step()
        after32 = torch.cat([p.detach().reshape(-1) for p in parameters32])
        actual_displacement = after32.double() - before32
        rounding_error = (actual_displacement - implicit_update).abs().numpy()
        spacing = np.abs(np.spacing(after32.numpy())).astype(np.float64)
        violations = rounding_error > spacing
        coordinate_ratio = rounding_error / spacing
        implicit_numpy = implicit_update.numpy()
        expected_relative_rms = float(np.sqrt(np.sum(
            implicit_numpy ** 2 * spacing ** 2 / 12., dtype=np.float64)) /
            np.sum(implicit_numpy ** 2, dtype=np.float64))
        actual_length = actual_displacement.norm().item()
        actual_relative = abs(actual_length - formula) / formula
        checks_passed = relative_update_length <= 1e-6 and not bool(violations.any())
        report['checks']['G3b'] = {
            'passed': checks_passed, 'parameter_dtype': 'float32', 'gradient_dtype': 'float32',
            'accumulator_dtype': str(optimizer32.b.dtype),
            'alpha_python_float64': alpha_python64, 'alpha_native_float32_as_float64': alpha_native32,
            'implicit_update_definition': 'float64 product of the native float32 alpha operand and native float32 gradient; no separately rounded float32 update tensor',
            'implicit_update_norm': implicit_length, 'formula_displacement': formula,
            'implicit_update_relative_error': relative_update_length, 'norm_relative_tolerance': 1e-6,
            'b_after': optimizer32.b.item(), 'predicted_b_equal': torch.equal(optimizer32.b, predicted_b),
            'coordinate_rounding_bound': 'abs(float64(theta_after)-float64(theta_before)-u) <= abs(np.spacing(theta_after.float32))',
            'coordinate_count': int(len(spacing)), 'coordinate_violations': int(violations.sum()),
            'max_absolute_coordinate_error': float(rounding_error.max()),
            'max_coordinate_error_in_ulps': float(coordinate_ratio.max()),
            'actual_float32_parameter_displacement': actual_length,
            'actual_norm_relative_error_diagnostic_only': actual_relative,
            'expected_relative_RMS_norm_rounding': expected_relative_rms,
            'RMS_formula': 'sqrt(sum(u_i^2 * abs(spacing(theta_after_i.float32))^2 / 12)) / sum(u_i^2)',
            'RMS_assumption': 'Independent zero-mean uniform coordinate rounding; an approximation, not a deterministic bound.',
            'previous_failed_relative_norm_error': previous['checks']['G3']['relative_error'],
            'previous_eta': previous['eta'], 'previous_b0': previous['b0'],
            'rng_unchanged': compare(before_rng, rng_state(), True)}
        save()
        require(relative_update_length <= 1e-6, 'Amended G3b implicit update norm fails relative 1e-6')
        require(not bool(violations.any()), 'Amended G3b displacement exceeds one actual float32 ULP')
        require(torch.equal(optimizer32.b, predicted_b), 'Amended G3b accumulator arithmetic differs')
        require(report['aggregate_gate_seconds'] <= 7200., 'Aggregate CPU gate cap exhausted')
        report['status'] = 'passed'
        report['passed'] = True
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
    g3a, g3b = report['checks']['G3a'], report['checks']['G3b']
    (args.output / 'REPORT.md').write_text(
        '# Amended first-step gate\n\n'
        'Both authorized precision checks passed on the saved gradient already certified in G2. '
        'The original failed G3 remains preserved; calibration, G1, G2 and G4 were not repeated. G5 has not run.\n\n'
        '| Check | Result | Requirement |\n|---|---:|---:|\n'
        '| Float64 actual step: relative norm error | %.9g | ≤ 1e-10 |\n'
        '| Native implicit update: relative norm error | %.9g | ≤ 1e-6 |\n'
        '| Largest coordinate rounding residual | %.9g ULP | ≤ 1 ULP |\n'
        '| Coordinates exceeding the bound | %d / %d | 0 |\n\n'
        'The expected relative RMS norm-rounding error is %.9g, compared with the original failed '
        'norm error %.9g. This RMS estimate assumes independent uniform coordinate rounding. '
        'The native update uses the float32 alpha and gradient operands of `add_`, multiplied in float64 '
        'for inspection; no separately rounded multiply-then-add implementation replaces it. '
        'The requested test values are η = %.9g and b₀ = %.9g. '
        'See [full evidence](gate_summary.json).\n' % (
            g3a['relative_error'], g3b['implicit_update_relative_error'],
            g3b['max_coordinate_error_in_ulps'], g3b['coordinate_violations'], g3b['coordinate_count'],
            g3b['expected_relative_RMS_norm_rounding'], g3b['previous_failed_relative_norm_error'],
            args.eta, args.b0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=EDITED)
    parser.add_argument('--preliminary', type=Path, default=STUDY / 'runs/gates/preliminary_retry')
    parser.add_argument('--previous', type=Path, default=STUDY / 'runs/gates/calibrated')
    parser.add_argument('--output', type=Path, default=STUDY / 'runs/gates/amended_G3')
    parser.add_argument('--eta', type=float, default=.00263955)
    parser.add_argument('--b0', type=float, default=.00000322129)
    parser.add_argument('--prior-seconds', type=float, default=759.9507804671302)
    args = parser.parse_args()
    torch.set_num_threads(2)
    main(args)
