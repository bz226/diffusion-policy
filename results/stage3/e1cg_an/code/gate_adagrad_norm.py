#!/usr/bin/env python
"""CPU-only AdaGrad-Norm gates; native fixed-batch updates and live smoke run.

Reuse the accepted critic/E1CG replay helpers without changing their files. The
update is extracted from the selected source AST, not a reimplemented loop.
"""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
STUDY = PARENT / 'results/stage3/e1cg_an'
REFERENCE = PARENT / '.codex-runtime/worktrees/stage3-e1cg/results/stage3/e1cg/dppo'
EDITED = PARENT / '.codex-runtime/worktrees/stage3-e1cg-an/results/stage3/e1cg_an/dppo'
FIXTURES = PARENT / 'results/stage3/runs/gate_capture'
CRITIC = PARENT / 'results/stage3/critic'
sys.path.insert(0, str(CRITIC / 'code'))
from gate_common import compare, freeze, require, rng_state, transfer
from gate_replay import make_agent, replay


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def flags(eta=None, b0=None):
    result = {'model.actor_loss': 'score', 'train.adv_estimator': 'mc_critic',
              'train.baseline_diag': True, 'train.stage3_diag': True,
              'train.decision_discount': True,
              'train.decision_discount_post_norm': True}
    if eta is not None:
        result.update({'train.actor_optimizer': 'adagrad_norm',
                       'train.actor_lr': eta, 'train.actor_lr_scheduler.min_lr': eta,
                       'train.adagrad_norm_b0': b0})
    return result


def strip_times(value):
    if isinstance(value, dict):
        return {key: strip_times(item) for key, item in value.items()
                if key not in ('seconds', 'elapsed_seconds') and not key.endswith('_seconds')}
    if isinstance(value, list):
        return [strip_times(item) for item in value]
    return value


def math_gate(source, out):
    sys.path.insert(0, str(source))
    from agent.finetune.adagrad_norm import AdaGradNorm
    generator = torch.Generator(device='cpu').manual_seed(20261019)
    initial = torch.randn(1000, generator=generator, dtype=torch.float32)
    gradients = torch.randn(20, 1000, generator=generator, dtype=torch.float32)
    result = {'passed': True, 'P': 1000, 'steps': 20, 'cases': []}
    for b0 in (1e-3, 10 * gradients[0].double().norm().item()):
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = AdaGradNorm([parameter], lr=.01, b0=b0)
        reference = initial.double().clone()
        b2 = torch.tensor(b0 * b0, dtype=torch.float64)
        rows = []
        for index, gradient in enumerate(gradients):
            parameter.grad = gradient.clone()
            reference_gradient = gradient.double()
            b2 += reference_gradient.square().sum()
            reference -= .01 * reference_gradient / b2.sqrt()
            before = rng_state()
            optimizer.step()
            difference = parameter.double() - reference
            relative = difference.norm().item() / reference.norm().item()
            b_relative = abs(optimizer.b.item() - b2.sqrt().item()) / b2.sqrt().item()
            row = {'step': index + 1, 'max_abs_parameter_error': difference.abs().max().item(),
                   'relative_L2_parameter_error': relative, 'relative_b_error': b_relative,
                   'gradient_unchanged': compare(parameter.grad, gradient, True)['passed'],
                   'rng_unchanged': compare(before, rng_state(), True)['passed']}
            rows.append(row)
            require(optimizer.b.dtype == torch.float64, 'G1 accumulator is not float64')
            require(relative <= 1e-6, 'G1 parameter reference error exceeds 1e-6')
            require(b_relative <= 1e-12, 'G1 accumulator differs from float64 reference')
            require(row['gradient_unchanged'] and row['rng_unchanged'], 'G1 step modifies gradient or RNG')
            require(optimizer.actor_updates == index + 1, 'G1 update counter differs')
        # State persistence must preserve the network-wide float64 accumulator.
        restored = AdaGradNorm([torch.nn.Parameter(initial.clone())], lr=.01, b0=1.)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        state_equal = compare(restored.state_dict(), optimizer.state_dict(), True)
        require(state_equal['passed'] and restored.b.dtype == torch.float64,
                'G1 optimizer serialization loses accumulator precision/state')
        result['cases'].append({'b0': b0, 'b0_exceeds_first_gradient_norm': b0 > gradients[0].double().norm().item(),
                                'steps': rows, 'state_roundtrip': state_equal})
    write(out / 'math.json', result)


def live_smoke(source, out, eta, b0):
    """Run all twelve actual iterations, including MuJoCo rollouts and diagnostics."""
    sys.path.insert(0, str(source))
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    from agent.finetune.grad_mean import cleanup_environment
    saved = torch.load(FIXTURES / 'nc4/training_batch.pt', map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(saved['config'])
    cfg.device = 'cpu'
    cfg.model.device = 'cpu'
    cfg.logdir = str(out / 'native')
    overrides = flags(eta, b0)
    overrides['train.n_train_itr'] = 12
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.save(cfg, out / 'smoke_config.yaml', resolve=True)
    require(cfg.env.n_envs == 40 and cfg.train.n_steps == 500 and cfg.train.batch_size == 50000
            and cfg.train.update_epochs == 5 and cfg.train.val_freq == 10,
            'G5 batch/update settings unexpectedly changed')
    cls = get_class(cfg._target_)
    agent = cls.__new__(cls)
    primary_error = None
    rows, steps, checkpoints = [], [], []
    cleanup = None
    try:
        cls.__init__(agent, cfg)
        require(agent.max_grad_norm is None, 'G5 gradient clipping is configured')
        original_start = agent._stage3_iteration_start
        original_record = agent._stage3_record
        original_step = agent.actor_optimizer.step
        start_state = {}
        def start():
            start_state['b'] = agent.actor_optimizer.b.item()
            start_state['updates'] = agent.actor_optimizer.actor_updates
            return original_start()
        def step(*args, **kwargs):
            before = agent.actor_optimizer.b.item()
            gradient_norm = torch.cat([p.grad.reshape(-1) for p in agent.model.actor_ft.parameters()]).double().norm().item()
            returned = original_step(*args, **kwargs)
            steps.append({'itr': agent.itr, 'b_before': before,
                          'b_after': agent.actor_optimizer.b.item(), 'grad_norm': gradient_norm,
                          'lr': agent.actor_optimizer.param_groups[0]['lr']})
            return returned
        def record(row, eval_mode):
            original_record(row, eval_mode)
            measured = {'itr': agent.itr, 'eval_mode': bool(eval_mode),
                        'lr': row['actor_lr'], 'b_start': start_state['b'],
                        'b_end': agent.actor_optimizer.b.item(),
                        'actor_updates_start': start_state['updates'],
                        'actor_updates_end': agent.actor_optimizer.actor_updates,
                        'logged_b': row.get('adagrad_b'),
                        'logged_effective_lr': row.get('adagrad_effective_lr')}
            rows.append(measured)
            write(out / 'smoke_progress.json', {'status': 'running', 'iterations': rows, 'steps': steps})
            require(row['actor_lr'] == eta, 'G5 logged actor lr differs from eta')
            require(measured['logged_b'] == measured['b_end'], 'G5 logged b differs from optimizer b')
            require(measured['logged_effective_lr'] == eta / measured['b_end'],
                    'G5 logged effective lr differs from eta / b')
            if eval_mode:
                require(measured['b_end'] == measured['b_start'] and
                        measured['actor_updates_end'] == measured['actor_updates_start'],
                        'G5 evaluation changed accumulator or actor update count')
            else:
                require(measured['actor_updates_end'] == measured['actor_updates_start'] + 1,
                        'G5 training iteration has other than one actor step')
        agent._stage3_iteration_start = start
        agent._stage3_record = record
        agent.actor_optimizer.step = step
        agent.run()
        require([row['itr'] for row in rows] == list(range(12)), 'G5 incomplete live twelve-iteration smoke')
        require([row['itr'] for row in rows if row['eval_mode']] == [0, 10], 'G5 evaluation schedule differs')
        require(len(steps) == 10, 'G5 expected ten actor updates in twelve iterations')
        with (Path(cfg.logdir) / 'result.pkl').open('rb') as stream:
            results = pickle.load(stream)
        require(len(results) == 12 and all(row['actor_lr'] == eta for row in results),
                'G5 saved result lacks twelve constant-lr records')
        for iteration in (0, 11):
            checkpoint = torch.load(Path(cfg.logdir) / 'checkpoint' / ('state_%d.pt' % iteration),
                                    map_location='cpu', weights_only=False)
            accumulator = checkpoint['adagrad_b']
            state = checkpoint['adagrad_norm_optimizer']['adagrad_norm']
            require(accumulator.dtype == torch.float64 and accumulator.item() == rows[iteration]['b_end'],
                    'G5 checkpoint accumulator differs from logged float64 b')
            require(torch.equal(state['b'], accumulator) and
                    state['actor_updates'] == rows[iteration]['actor_updates_end'],
                    'G5 optimizer checkpoint state differs from logged state')
            checkpoints.append({'itr': iteration, 'b': accumulator.item(), 'dtype': str(accumulator.dtype),
                                'actor_updates': state['actor_updates']})
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if hasattr(agent, 'venv'):
            if not hasattr(agent, 'logdir'):
                agent.logdir = str(cfg.logdir)
            cleanup = cleanup_environment(agent, primary_error=primary_error)
        write(out / 'smoke.json', {'passed': primary_error is None, 'iterations': rows, 'steps': steps,
                                 'checkpoints': checkpoints,
                                 'cleanup': cleanup, 'batch_settings_unchanged': True,
                                 'rollouts': 'actual CPU MuJoCo; no saved-rollout substitution'})
    require(cleanup['status'] == 'passed' and cleanup['all_children_exited'], 'G5 cleanup left children')


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'running', 'source': str(args.source), 'cpu_only': True}
    def interrupted(signum, frame):
        raise RuntimeError('gate worker interrupted by signal %d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        require(not torch.cuda.is_available(), 'CPU gates require hidden GPUs')
        if args.worker == 'math':
            math_gate(args.source, args.output)
        elif args.worker == 'smoke':
            live_smoke(args.source, args.output, args.eta, args.b0)
        else:
            overrides = json.loads(args.overrides)
            agent, saved, cfg = make_agent(args.source, FIXTURES / args.fixture / 'training_batch.pt',
                                          'cpu', args.output, overrides)
            batch = transfer(saved['batch'], 'cpu')
            batch['logprobs_k'] = torch.load(CRITIC / ('runs/gate/prepare_cpu_' + args.fixture + '/cpu_old_logprobs.pt'),
                                             map_location='cpu', weights_only=True)
            replay(agent, saved, batch, args.source, args.output, overrides.get('train.stage3_diag', False))
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        write(args.output / 'worker.json', report)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = time.time() + args.cap_seconds - args.prior_seconds
    report = {'status': 'running', 'passed': False, 'cpu_only': True, 'commands': [], 'checks': {},
              'reference_source': str(args.reference), 'edited_source': str(args.source),
              'reference_revision': '03f8261', 'eta_gate': args.eta, 'b0_gate': args.b0,
              'cap_seconds': args.cap_seconds, 'prior_gate_seconds': args.prior_seconds,
              'fixture': 'Existing full Stage-3 saved batch, pre-update model/optimizer/RNG state; same accepted CPU log-probs for both versions.'}
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['aggregate_gate_seconds'] = args.prior_seconds + report['elapsed_seconds']
        write(args.output / 'gate_summary.json', report)
    def run(name, mode, source, fixture='nc4', overrides=None):
        target = args.output / name
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', mode,
                   '--source', str(source), '--output', str(target), '--fixture', fixture,
                   '--overrides', json.dumps(overrides or {}), '--eta', str(args.eta),
                   '--b0', str(args.b0), '--threads', str(args.threads)]
        report['commands'].append(command)
        report['active'] = name
        save()
        print('START ' + name, flush=True)
        remaining = deadline - time.time()
        require(remaining > 0, 'CPU gate wall-clock cap exhausted')
        with (args.output / (name + '.out')).open('w') as stream:
            completed = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            try:
                returncode = completed.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # SIGTERM reaches the worker's finally block and explicit env cleanup.
                completed.terminate()
                try:
                    completed.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(completed.pid, signal.SIGKILL)
                    completed.wait()
                raise RuntimeError('CPU gate wall-clock cap exhausted during ' + name)
        require(returncode == 0, 'Gate worker failed: ' + name)
        return target
    def evidence(path):
        return torch.load(path / 'evidence.pt', map_location='cpu', weights_only=False)
    try:
        math_path = args.reuse_math or run('optimizer_math', 'math', args.source)
        report['checks']['G1'] = json.loads((math_path / 'math.json').read_text())
        require(report['checks']['G1']['passed'], 'G1 optimizer math did not pass')
        save()
        comparisons = {}
        keys = ('initial_rng', 'initial_model', 'initial_actor_optimizer', 'initial_critic_optimizer',
                'constructor_actor_optimizer', 'constructor_critic_optimizer', 'constructor_actor_scheduler',
                'constructor_critic_scheduler', 'input_batch', 'loss_components', 'actor_gradients',
                'critic_gradients', 'permutations', 'actor_parameters', 'critic_parameters', 'model_state',
                'actor_optimizer_state', 'critic_optimizer_state', 'actor_scheduler_state',
                'critic_scheduler_state', 'rng_after_update', 'diagnostic_generator_state',
                'logged_values', 'sampling', 'input_batch_after')
        for name, fixture, overrides in [('default', 'default', {}), ('nc4', 'nc4', {}),
                                          ('e1cg', 'nc4', flags())]:
            old = run(name + '_old', 'replay', args.reference, fixture, overrides)
            new = run(name + '_new', 'replay', args.source, fixture, overrides)
            first, second = evidence(old), evidence(new)
            checks = {key: compare(first[key], second[key], True, key) for key in keys}
            checks['scientific_stage3_metrics'] = compare(strip_times(first['stage3_metrics']),
                                                         strip_times(second['stage3_metrics']), True)
            passed = all(value['passed'] for value in checks.values())
            comparisons[name] = {'passed': passed, 'comparisons': checks}
            report['checks']['G4'] = {'passed': all(value['passed'] for value in comparisons.values()),
                                     'cases': comparisons}
            save()
            require(passed, 'G4 native CPU replay differs: ' + name)
            del first, second
        an_path = run('adagrad_update', 'replay', args.source, overrides=flags(args.eta, args.b0))
        original, changed = evidence(args.output / 'e1cg_old'), evidence(an_path)
        require(len(original['actor_gradients']) == len(changed['actor_gradients']) == 1,
                'G2 expected one actor update in each full batch')
        gradient_check = compare(original['actor_gradients'], changed['actor_gradients'], True)
        report['checks']['G2'] = {'passed': gradient_check['passed'], 'incoming_gradients': gradient_check,
                                 'scope': 'gradient captured immediately before each native optimizer.step'}
        save()
        require(gradient_check['passed'], 'G2 optimizer receives a different gradient')
        if args.preliminary:
            report['status'] = 'preliminary_passed'
            report['preliminary_passed'] = all(report['checks'][key]['passed'] for key in ('G1', 'G2', 'G4'))
            report['checks']['G3'] = {'passed': False, 'status': 'pending_calibrated_eta_b0'}
            report['checks']['G5'] = {'passed': False, 'status': 'pending_calibrated_eta_b0'}
            write(args.output / 'preliminary_gate.json', {
                'passed': report['preliminary_passed'], 'covered': ['G1', 'G2', 'G4'],
                'remaining': ['G3', 'G5'], 'seconds': time.monotonic() - started,
                'gate_summary': str(args.output / 'gate_summary.json')})
            return
        initial = changed['initial_model']
        before = torch.cat([initial['actor_ft.' + key].reshape(-1) for key in changed['actor_parameters']]).double()
        after = torch.cat([tensor.reshape(-1) for tensor in changed['actor_parameters'].values()]).double()
        gradient = torch.cat([tensor.reshape(-1) for tensor in changed['actor_gradients'][0].values()]).double()
        norm = gradient.norm().item()
        measured = (after - before).norm().item()
        expected = args.eta * norm / np.sqrt(args.b0 ** 2 + norm ** 2)
        relative = abs(measured - expected) / expected
        report['checks']['G3'] = {'passed': relative <= 1e-6, 'measured_float32_parameter_displacement': measured,
                                 'formula_displacement': expected, 'relative_error': relative,
                                 'gradient_norm': norm, 'eta': args.eta, 'b0': args.b0,
                                 'parameter_dtype': 'float32; displacement evaluated in float64'}
        save()
        require(relative <= 1e-6, 'G3 actual float32 parameter displacement differs from formula by more than 1e-6')
        del original, changed
        if not args.skip_smoke:
            smoke = run('live_smoke_12itr', 'smoke', args.source)
            report['checks']['G5'] = json.loads((smoke / 'smoke.json').read_text())
        else:
            report['checks']['G5'] = {'passed': False, 'status': 'not_run', 'reason': 'explicit --skip-smoke; never counts as gate pass'}
        report['passed'] = all(report['checks'][key]['passed'] for key in ('G1', 'G2', 'G3', 'G4', 'G5'))
        report['status'] = 'passed' if report['passed'] else 'incomplete'
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['math', 'replay', 'smoke'])
    parser.add_argument('--source', type=Path, default=EDITED)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path, default=STUDY / 'runs/gates')
    parser.add_argument('--fixture', default='nc4')
    parser.add_argument('--overrides', default='{}')
    parser.add_argument('--eta', type=float, default=.01)
    parser.add_argument('--b0', type=float, default=.001)
    parser.add_argument('--cap-seconds', type=float, default=7200.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--skip-smoke', action='store_true')
    parser.add_argument('--preliminary', action='store_true')
    parser.add_argument('--reuse-math', type=Path)
    parser.add_argument('--prior-seconds', type=float, default=0.)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    worker(arguments) if arguments.worker else main(arguments)
