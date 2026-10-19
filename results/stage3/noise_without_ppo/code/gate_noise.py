#!/usr/bin/env python
"""Approved measurement gates. This controller never submits scheduler jobs."""
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

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
STUDY = PROJECT / 'results/stage3/noise_without_ppo'
SOURCE = PROJECT / '.codex-runtime/worktrees/noise-without-ppo/results/stage3/noise_without_ppo/dppo'
PARENT_SOURCE = PROJECT / 'results/stage3_gradmean/dppo'
STAGE2_SOURCE = PROJECT / 'results/stage2_noclip/dppo'
MC_SOURCE = PROJECT / '.codex-runtime/worktrees/stage3-e1c/results/stage3_e1c/dppo'
sys.path.append(str(PROJECT / 'results/stage3/dl1/code'))
from gate_dl1 import compare_evidence
from gate_dl1_worker import write
from gate_common import numeric_settings, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('cpu', 'live', 'merge'), required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--parent-source', type=Path, default=PARENT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-result', type=Path)
    parser.add_argument('--live-result', type=Path)
    parser.add_argument('--study-spec', type=Path)
    parser.add_argument('--timeout-seconds', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Refuse to overwrite gate output')
    output.mkdir(parents=True)
    cap = {'cpu': 7140, 'live': 1740, 'merge': 120}[args.phase]
    timeout = args.timeout_seconds or cap
    require(0 < timeout <= cap, 'Gate timeout exceeds approved phase cap')
    started = time.monotonic()
    report = {'passed': False, 'status': 'running', 'phase': args.phase,
              'source': str(args.source.resolve()), 'checks': {}, 'commands': [],
              'parent_source': str(args.parent_source.resolve()),
              'stage2_reference': str(STAGE2_SOURCE), 'mc_reference': str(MC_SOURCE),
              'numeric_settings': numeric_settings(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}

    def save():
        write(output / 'gate_summary.json', report)

    def run(name, command, cwd, cpu=True):
        report['active_worker'] = name
        report['commands'].append(command)
        save()
        remaining = timeout - (time.monotonic() - started)
        require(remaining > 0, 'Gate exhausted its approved time cap')
        env = os.environ.copy()
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        env['DL1_GATE_THREADS'] = env.get('NOISE_GATE_THREADS', '8')
        print('GATE START ' + name, flush=True)
        with (output / (name + '.out')).open('x') as stream:
            child = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                exit_code = child.wait(timeout=remaining)
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                raise
        require(exit_code == 0, 'Gate worker failed: ' + name + '; read ' + str(output / (name + '.out')))
        target = output / name
        record = json.loads((target / 'worker.json').read_text())
        require(record.get('passed') and record.get('all_children_exited'),
                'Gate worker verification or cleanup failed: ' + name)
        report.setdefault('workers', []).append({'name': name, 'seconds': record['elapsed_seconds'],
                                                'exit_code': exit_code})
        save()
        return target

    def identity(name, source, kind, estimator, extra=()):
        command = [sys.executable, '-B', str(STUDY / 'code/gate_noise_worker.py'),
                   '--source', str(source), '--output', str(output / name),
                   '--kind', kind, '--estimator', estimator] + list(extra)
        return run(name, command, source)

    def replay(name, source):
        command = [sys.executable, '-B', str(PROJECT / 'results/stage3/dl1/code/gate_dl1_worker.py'),
                   '--source', str(source), '--output', str(output / name),
                   '--kind', 'replay', '--profile', 'default']
        return run(name, command, source)

    save()
    try:
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        if args.phase == 'cpu':
            require(not torch.cuda.is_available(), 'CPU phase requires CUDA_VISIBLE_DEVICES empty')
            native_v1 = identity('G1_native', STAGE2_SOURCE, 'reference', 'V1')
            measured_v1 = identity('G1_measured', args.source, 'measured', 'V1',
                                   ('--reference', str(native_v1 / 'identity.pt')))
            report['checks']['G1'] = json.loads((measured_v1 / 'identity.json').read_text())
            save()
            native_v2 = identity('G2_native', MC_SOURCE, 'reference', 'V2',
                                 ('--cpu-logprobs', str(native_v1 / 'cpu_old_logprobs.pt')))
            measured_v2 = identity('G2_measured', args.source, 'measured', 'V2',
                                   ('--reference', str(native_v2 / 'identity.pt')))
            report['checks']['G2'] = json.loads((measured_v2 / 'identity.json').read_text())
            save()
            before = replay('G5_parent', args.parent_source)
            after = replay('G5_measurement_source', args.source)
            old = torch.load(before / 'evidence.pt', map_location='cpu', weights_only=False)
            new = torch.load(after / 'evidence.pt', map_location='cpu', weights_only=False)
            report['checks']['G5'] = compare_evidence(old, new)
            report['checks']['G5']['scope'] = ('full saved 200000-pair native DPPO update at default flags; '
                '20 actor and critic optimizer steps, parameters, both optimizer/scheduler states, '
                'global RNG, logged values and fixed-noise sampling')
            save()
            require(report['checks']['G5']['passed'], 'G5 defaults no longer reproduce parent source bitwise')
        elif args.phase == 'live':
            require(args.cpu_result is not None and args.study_spec is not None,
                    'Live gate requires passed CPU summary and study spec')
            cpu = json.loads(args.cpu_result.read_text())
            require(cpu.get('passed') and cpu.get('phase') == 'cpu', 'CPU gates must pass before live gate')
            command = [sys.executable, '-B', str(STUDY / 'code/gate_noise_live.py'),
                       '--source', str(args.source), '--output', str(output / 'G3_G4_live'),
                       '--study-spec', str(args.study_spec)]
            target = run('G3_G4_live', command, args.source, cpu=False)
            result = json.loads((target / 'live_gates.json').read_text())
            report['checks'].update(result['checks'])
        else:
            require(args.cpu_result is not None and args.live_result is not None,
                    'Merge requires CPU and live summaries')
            cpu, live = json.loads(args.cpu_result.read_text()), json.loads(args.live_result.read_text())
            require(cpu.get('passed') and live.get('passed'), 'Cannot merge failed gates')
            require(cpu['source'] == live['source'] == str(args.source.resolve()), 'Gate sources differ')
            report['checks'] = dict(cpu['checks'], **live['checks'])
            require(set(report['checks']) == {'G1', 'G2', 'G3', 'G4', 'G5'}, 'Missing required gate')
            require(all(v['passed'] for v in report['checks'].values()), 'A required gate failed')
            report['phase_records'] = [str(args.cpu_result), str(args.live_result)]
        report.pop('active_worker', None)
        report.update(passed=True, status='passed')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        save()


if __name__ == '__main__':
    main()
