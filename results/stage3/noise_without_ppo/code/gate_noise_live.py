#!/usr/bin/env python
"""Frozen-state and independent-seed gates with exactly two live batches."""
import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
sys.path.append(str(PROJECT / 'results/stage3/critic/code'))
from gate_common import numeric_settings, require


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_spec(value):
    return json.loads(Path(value).read_text()) if isinstance(value, str) else dict(value)


def check_state(points):
    from agent.finetune import grad_mean as gm
    from agent.finetune.grad_mean_v5 import set_scaler_divisor
    from agent.finetune.noise_without_ppo import frozen_divisor
    from util.reward_scaling import RunningRewardScaler
    records = []
    for point in points:
        frozen = json.loads(Path(point['frozen_path']).read_text())
        require(frozen['status'] == 'completed' and frozen.get('clean_exit'),
                'Preparation not successfully completed: ' + point['point'])
        checkpoint = torch.load(point['checkpoint'], map_location='cpu', weights_only=True)
        critic = {name[len('critic.'):]: value for name, value in checkpoint['model'].items()
                  if name.startswith('critic.')}
        require(bool(critic), 'Checkpoint critic is missing')
        checksum = gm.state_sha256(critic)
        require(checksum == frozen['critic_checkpoint_sha256'] == frozen['critic_after_sha256'],
                'G3 prepared critic differs from checkpoint: ' + point['point'])
        require(frozen['warmup_batches'] == frozen['actor_optimizer_steps'] == frozen['critic_optimizer_steps'] == 0,
                'G3 preparation trained a model')
        cleanup = json.loads((Path(point['frozen_path']).parent / 'cleanup.json').read_text())
        require(cleanup.get('all_children_exited') and cleanup.get('status') == 'passed',
                'Preparation did not cleanly close its workers')
        agent = SimpleNamespace(running_reward_scaler=RunningRewardScaler(40))
        initial = int(point['checkpoint_itr']) == 0
        if initial:
            expected = frozen_divisor(agent.running_reward_scaler)
            require(frozen['fit_batches'] == [] and not frozen['scaler_approximation'],
                    'Initial scaler unexpectedly fitted or marked approximate')
        else:
            fits = frozen['fit_batches']
            require(len(fits) == 5 and frozen['scaler_approximation'],
                    'Later scaler needs exactly five explicitly approximate fit batches')
            expected = sum(row['sum_VG'] for row in fits) / sum(row['sum_V2'] for row in fits)
            set_scaler_divisor(agent, expected)
        effective = frozen_divisor(agent.running_reward_scaler)
        require(expected == frozen['scaler_c'] == effective, 'G3 fitted/provided/effective scaler divisor differs')
        records.append({'point': point['point'], 'checkpoint': point['checkpoint'],
                        'critic_checkpoint_sha256': checksum, 'critic_bitwise_equal': True,
                        'warmup_batches': 0, 'scaler_c': expected,
                        'scaler_c_source': frozen['scaler_c_source'],
                        'scaler_approximation': not initial,
                        'historical_running_state_available': initial,
                        'all_children_exited': True})
    require(len(records) == 5, 'G3 needs all five points')
    return {'passed': True, 'points': records,
            'limitation': 'Only update 0 has exact historical running-scaler state; later fitted divisors are frozen approximations.'}


def check_seed_namespace(study, live_specs):
    from agent.finetune.noise_without_ppo import seed_record
    specifications = [load_spec(value) for value in study['collect_specs']]
    require(len(specifications) == 50, 'Approved production plan needs 50 jobs')
    require(all(int(s.get('batches', 50)) == 50 for s in specifications),
            'Every production collect job must have 50 batches')
    records = []
    for spec in specifications:
        records.extend((spec['point'], int(spec['job_index']), i, seed_record(spec, i)) for i in range(50))
    for point in study['points']:
        frozen = json.loads(Path(point['frozen_path']).read_text())
        spec = frozen['spec']
        for row in frozen['fit_batches']:
            index = int(row['batch'])
            expected = seed_record(spec, index)
            require(row['seeds'] == expected, 'Preparation seed record disagrees with actual seed recipe')
            records.append((spec['point'], int(spec['job_index']), index, expected))
    for spec in live_specs:
        records.append((spec['point'], int(spec['job_index']), 0, seed_record(spec, 0)))
    seen = {}
    for point, job, batch, seeds in records:
        require(len(seeds['environment_seeds']) == 40, 'Need forty private environment seeds')
        values = seeds['environment_seeds'] + [seeds[key] for key in
            ('diffusion_seed', 'permutation_seed', 'python_seed', 'numpy_seed')]
        for seed in values:
            require(seed not in seen,
                    'G4 duplicate seed %d: %r and %r' % (seed, seen.get(seed), (point, job, batch)))
            seen[seed] = (point, job, batch)
    return {'passed': True, 'production_jobs': 50, 'production_batches': 2500,
            'preparation_batches': len(records) - 2500 - len(live_specs),
            'live_gate_batches': len(live_specs), 'distinct_seed_values': len(seen),
            'minimum_seed': min(seen), 'maximum_seed': max(seen),
            'seed_types': ['40 private environment/reset streams', 'diffusion', 'permutation', 'Python', 'NumPy']}


def live_batch(source, output, spec, remaining):
    destination = Path(spec['out'])
    spec_path = output / ('gate_j%d.json' % spec['job_index'])
    write(spec_path, spec)
    command = [sys.executable, '-B', '-m', 'agent.finetune.noise_without_ppo', '--spec', str(spec_path)]
    with (output / ('gate_j%d.out' % spec['job_index'])).open('x') as stream:
        process = subprocess.Popen(command, cwd=str(source), env=os.environ.copy(),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=remaining)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
    require(exit_code == 0, 'Live gate collection failed; inspect ' + str(spec_path.with_suffix('.out')))
    meta = json.loads((destination / 'meta.json').read_text())
    cleanup = json.loads((destination / 'cleanup.json').read_text())
    require(meta.get('status') == 'completed' and meta.get('clean_exit') and meta.get('K') == 1,
            'Live gate collection is incomplete')
    require(cleanup.get('all_children_exited') and cleanup.get('status') == 'passed',
            'Live gate left child processes or had a cleanup failure')
    require(meta['actor_optimizer_steps'] == meta['critic_optimizer_steps'] == 0,
            'Live gate unexpectedly updated a model')
    require(meta['critic_checkpoint_sha256'] == meta['critic_after_sha256'],
            'Live gate changed the checkpoint critic')
    with (destination / 'batches.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1, 'Gate job should contain exactly one batch')
    row = rows[0]
    returns = np.asarray(json.loads(row['episode_returns']), dtype=np.float64)
    observations = np.asarray(json.loads(row['initial_observations']), dtype=np.float64)
    seeds = json.loads(row['seeds'])
    require(returns.shape == (80,) and np.isfinite(returns).all(), 'Gate needs 80 finite episode returns')
    require(observations.shape[0] == 40 and np.isfinite(observations).all(),
            'Gate needs forty finite initial observations')
    return {'spec': spec, 'command': command, 'meta': meta, 'cleanup': cleanup,
            'episode_returns': returns.tolist(), 'initial_observations': observations.tolist(),
            'seeds': seeds, 'exit_code': exit_code, 'directory': str(destination)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study-spec', type=Path, required=True)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    require(not args.output.exists(), 'Refuse to overwrite live gate evidence')
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.source))
    from agent.finetune.noise_without_ppo import seed_record
    started = time.monotonic()
    record = {'passed': False, 'status': 'running', 'source': str(args.source),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'numeric_settings': numeric_settings(),
              'command': [sys.executable] + sys.argv}
    write(args.output / 'worker.json', record)
    try:
        require(torch.cuda.is_available(), 'Live gate needs an allocated GPU')
        require(not torch.are_deterministic_algorithms_enabled(), 'Do not enable deterministic algorithms')
        study = json.loads(args.study_spec.read_text())
        checks = {'G3': check_state(study['points'])}
        initial = next(point for point in study['points'] if int(point['checkpoint_itr']) == 0)
        template = next(load_spec(value) for value in study['collect_specs']
                        if load_spec(value)['point'] == initial['point'])
        live_specs = []
        for index in (97, 98):
            spec = dict(template)
            spec.update(operation='collect', batches=1, gate_only=True, job_index=index,
                        out=str(args.output / ('collect_gate_j%d' % index)), frozen_path=initial['frozen_path'])
            live_specs.append(spec)
        namespace = check_seed_namespace(study, live_specs)
        live = []
        for spec in live_specs:
            remaining = 1650 - (time.monotonic() - started)
            require(remaining > 0, 'Live gate exhausted approved time cap')
            item = live_batch(args.source, args.output, spec, remaining)
            require(item['seeds'] == seed_record(spec, 0), 'Actual live seeds do not match reserved namespace')
            live.append(item)
        left, right = [np.asarray(item['episode_returns'], dtype=np.float64) for item in live]
        obs_left, obs_right = [np.asarray(item['initial_observations'], dtype=np.float64) for item in live]
        duplicate_returns = int(np.count_nonzero(left[:, None] == right[None, :]))
        require(duplicate_returns == 0, 'G4 live jobs share an identical episode return')
        require(not np.array_equal(obs_left, obs_right), 'G4 different reset seeds produced identical initial observations')
        require(live[0]['meta']['critic_checkpoint_sha256'] == live[1]['meta']['critic_checkpoint_sha256'],
                'G4 jobs did not load identical frozen critic state')
        checks['G4'] = {'passed': True, 'namespace': namespace,
                        'live_directories': [item['directory'] for item in live],
                        'live_seeds': [item['seeds'] for item in live],
                        'live_first_batch_return_means': [float(left.mean()), float(right.mean())],
                        'identical_cross_job_episode_returns': duplicate_returns,
                        'initial_observations_differ': True, 'exit_codes': [item['exit_code'] for item in live],
                        'all_children_exited': True,
                        'environment_seeding': 'Production seeded_batch requires acknowledgments that private MuJoCo/reset and action-space RNGs were seeded before every batch.'}
        require(not mp.active_children(), 'Live gate controller left child processes')
        write(args.output / 'live_gates.json', {'passed': True, 'checks': checks})
        record.update(passed=True, status='passed', all_children_exited=True,
                      gpu=torch.cuda.get_device_name(0), checks=checks)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        write(args.output / 'worker.json', record)


if __name__ == '__main__':
    main()
