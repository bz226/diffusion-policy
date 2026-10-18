#!/usr/bin/env python
"""Complete G3/G5 once calibration fixes eta and b0; no gate reruns."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch

from gate_adagrad_norm import EDITED, STUDY, compare, require, rng_state, write


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.preliminary / 'gate_summary.json').read_text())
    require(prior.get('preliminary_passed'), 'G1/G2/G4 must have passed before calibrated gates')
    prior_seconds = prior.get('aggregate_gate_seconds', prior['elapsed_seconds'])
    budget = args.cap_seconds - prior_seconds
    require(budget > 0, 'Aggregate CPU gate budget exhausted')
    started = time.monotonic()
    process = None
    def interrupted(signum, frame):
        raise RuntimeError('calibrated gate interrupted by signal %d' % signum)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    def stop_child():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    report = {'status': 'running', 'passed': False, 'checks': dict(prior['checks']),
              'preliminary': str(args.preliminary), 'prior_gate_seconds': prior_seconds,
              'eta': args.eta, 'b0': args.b0, 'source': str(args.source),
              'cap_seconds': args.cap_seconds, 'monitor_interval_seconds': 1800,
              'commands': [], 'monitor_history': []}
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['aggregate_gate_seconds'] = prior_seconds + report['elapsed_seconds']
        write(args.output / 'gate_summary.json', report)
    try:
        require(not torch.cuda.is_available(), 'Hide GPUs for calibrated CPU gates')
        sys.path.insert(0, str(args.source))
        from agent.finetune.adagrad_norm import AdaGradNorm
        evidence = torch.load(args.preliminary / 'adagrad_update/evidence.pt',
                              map_location='cpu', weights_only=False)
        actor_names = list(evidence['actor_parameters'])
        parameters = [torch.nn.Parameter(evidence['initial_model']['actor_ft.' + name].clone())
                      for name in actor_names]
        optimizer = AdaGradNorm(parameters, lr=args.eta, b0=args.b0)
        for parameter, name in zip(parameters, actor_names):
            parameter.grad = evidence['actor_gradients'][0][name].clone()
        before = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
        gradient = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).double()
        before_rng = rng_state()
        optimizer.step()
        after = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
        norm = gradient.norm().item()
        expected = args.eta * norm / (args.b0 ** 2 + norm ** 2) ** .5
        measured = (after - before).norm().item()
        relative = abs(measured - expected) / expected
        report['checks']['G3'] = {
            'passed': relative <= 1e-6, 'gradient_norm': norm, 'eta': args.eta, 'b0': args.b0,
            'measured_float32_parameter_displacement': measured, 'formula_displacement': expected,
            'relative_error': relative, 'rng_unchanged': compare(before_rng, rng_state(), True),
            'source': 'Same incoming native E1CG actor gradient certified bitwise in G2; calibrated eta and b0 in actual float32 optimizer step.'}
        save()
        require(relative <= 1e-6, 'G3 actual float32 parameter displacement fails strict relative 1e-6')
        require(report['checks']['G3']['rng_unchanged']['passed'], 'G3 step modified RNG')
        del evidence, parameters, optimizer, before, after, gradient
        target = args.output / 'live_smoke_12itr'
        command = [sys.executable, '-B', str(Path(__file__).with_name('gate_adagrad_norm.py').resolve()),
                   '--worker', 'smoke', '--source', str(args.source), '--output', str(target),
                   '--eta', str(args.eta), '--b0', str(args.b0), '--threads', str(args.threads)]
        report['commands'].append(command)
        report['active'] = 'G5_live_smoke'
        save()
        deadline = time.monotonic() + budget - (time.monotonic() - started)
        next_monitor = time.monotonic()
        previous_progress = None
        with (args.output / 'live_smoke_12itr.out').open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            report['smoke_pid'] = process.pid
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    stop_child()
                    raise RuntimeError('Aggregate two-hour CPU gate cap exhausted')
                if now >= next_monitor:
                    progress_path = target / 'smoke_progress.json'
                    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                    count = len(progress.get('iterations', []))
                    report['monitor_history'].append({
                        'elapsed_seconds': now - started, 'completed_iterations': count,
                        'suspected_stall': previous_progress == count if previous_progress is not None else False,
                        'pid': process.pid, 'running': True})
                    previous_progress = count
                    next_monitor = now + 1800
                    save()
                try:
                    process.wait(timeout=min(30, max(1, deadline - now)))
                except subprocess.TimeoutExpired:
                    pass
            require(process.returncode == 0, 'G5 live smoke worker failed; inspect output')
        report['checks']['G5'] = json.loads((target / 'smoke.json').read_text())
        report['passed'] = all(report['checks'][key]['passed'] for key in ('G1', 'G2', 'G3', 'G4', 'G5'))
        report['status'] = 'passed' if report['passed'] else 'failed'
        require(report['passed'], 'A required gate did not pass')
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        stop_child()
        signal.signal(signal.SIGTERM, previous_sigterm)
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=EDITED)
    parser.add_argument('--preliminary', type=Path, default=STUDY / 'runs/gates/preliminary_retry')
    parser.add_argument('--output', type=Path, default=STUDY / 'runs/gates/calibrated')
    parser.add_argument('--eta', type=float, required=True)
    parser.add_argument('--b0', type=float, required=True)
    parser.add_argument('--cap-seconds', type=float, default=7200.)
    parser.add_argument('--threads', type=int, default=2)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    main(arguments)
