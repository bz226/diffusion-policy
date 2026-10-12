"""Bounded C1/C2 administration using the unchanged, gated V6 collector.

External C1/C2 names are mapped to collector labels D1/D2 solely because the
existing collector validates a fixed label allowlist. Checkpoints, frozen
states, seeds and output directories are separate from the original D points.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
G = PROJECT / 'results/stage3/gradmean'
HERE = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT / '.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/code/campaign.py'
module_spec = importlib.util.spec_from_file_location('v6_existing_runtime', str(RUNTIME))
old = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(old)
old.JOURNAL = HERE / 'commands.jsonl'
SOURCE_COMMIT = 'ce909d3b356fd7754963d4ec3f6bf38619022853'
GATE = G / 'runs/gate_v6/gate_summary.json'
CAP = 3600
MAX_PAR = 5
BASE_GPU_SECONDS = 244995
BUDGET_GPU_SECONDS = 125 * 3600
STATE = HERE / 'campaign.json'
POINTS = {'C1': {'collector_point': 'D1', 'point_index': 5, 'itr': 70, 'updates': 63},
          'C2': {'collector_point': 'D2', 'point_index': 6, 'itr': 139, 'updates': 126}}
RUN_IDS = ['prepare_C1', 'prepare_C2'] + [p + '_j%02d' % j for p in ('C1', 'C2') for j in range(10)]


def plan():
    return dict(version=3, source_commit=SOURCE_COMMIT, source_path=str(old.V6_SOURCE),
                scientific_source_unchanged=True, points=POINTS, run_ids=RUN_IDS,
                prepare_warmup_batches=5, collect_jobs_per_point=10, batches_per_job=50,
                episodes_per_batch=80, variants=['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                cap_seconds_per_job=CAP, max_parallel=MAX_PAR,
                base_gpu_seconds=BASE_GPU_SECONDS, aggregate_gpu_seconds_cap=BUDGET_GPU_SECONDS,
                maximum_initial_reserved_gpu_seconds=22 * CAP,
                automatic_retries=False, no_actor_updates=True,
                monitor_interval_seconds=1800, controller_timeout_seconds=10800,
                estimate_gpu_hours=[4.2, 4.5], estimate_wall_minutes=[60, 70],
                estimate_basis='50 completed V6 jobs: mean724.88s; five-batch preparations297-317s; five parallel GPUs',
                internal_alias_explanation='C1->D1 and C2->D2 reuse administrative labels accepted by the unchanged collector; original D checkpoint/data paths are never used.',
                authorization='Clean figures version3 section6: two new checkpoint points, five critic warmup batches each, ten50-batch V1-V6 groups per point; existing125GPU-hour aggregate cap retained.')


def identify(run_id):
    if run_id not in RUN_IDS:
        raise ValueError('Run outside the exact22 authorized jobs: ' + run_id)
    if run_id.startswith('prepare_'):
        return run_id[8:], None
    return run_id[:2], int(run_id[-2:])


def make_spec(run_id, frozen=None):
    point, group = identify(run_id)
    info = POINTS[point]
    step = 'prepare' if group is None else 'collect'
    seed = 100000 + 10000 * info['point_index'] + 100 * (90 if group is None else group)
    checkpoint = PROJECT / 'results/stage3/e1c/E1C_seed0/checkpoint' / ('state_%d.pt' % info['itr'])
    if not checkpoint.is_file():
        raise ValueError('Required checkpoint absent: ' + str(checkpoint))
    logdir = G / ('prepare' if group is None else 'collect') / (point if group is None else run_id)
    overrides = old.COMMON + ['+train.gm_include_critic=true', 'seed=' + str(seed),
        'logdir=' + str(logdir), 'base_policy_path=' + str(checkpoint),
        '+train.gm_step=' + step, '+train.gm_point=' + info['collector_point'],
        '+train.gm_parent_commit=' + SOURCE_COMMIT]
    expected = dict(mode='grad_mean', gm_step=step, gm_point=info['collector_point'],
                    seed=seed, gm_include_critic=True, environment_seeds=list(range(seed, seed + 40)),
                    gm_parent_commit=SOURCE_COMMIT)
    if group is None:
        expected['gm_warmup_batches'] = 5
        overrides.append('+train.gm_warmup_batches=5')
    else:
        if frozen is None:
            preparation = old.read(HERE / 'workers' / ('prepare_' + point) / 'manifest.json', {})
            frozen = old.read(G / 'prepare' / point / 'grad_mean.json', {})
            if preparation.get('status') != 'complete' or frozen.get('status') != 'completed':
                raise ValueError('Collect requires completed verified own-point preparation')
        if frozen.get('point') != info['collector_point'] or frozen.get('source_checkpoint') != str(checkpoint):
            raise ValueError('Frozen state must belong to this external point and exact checkpoint')
        required_path = G / 'prepare' / point / ('frozen_' + info['collector_point'] + '.pt')
        if Path(frozen['frozen_path']).resolve() != required_path:
            raise ValueError('Frozen file aliases another point')
        expected.update(gm_frozen_path=str(required_path), gm_frozen_sha256=frozen['frozen_sha256'],
                        gm_batches=50, gm_score_sample=0)
        for key in ('gm_frozen_path', 'gm_frozen_sha256', 'gm_batches', 'gm_score_sample'):
            overrides.append('+train.' + key + '=' + str(expected[key]))
    return dict(run_id=run_id, point=point, external_point_id=point,
                collector_point=info['collector_point'], group=group, step=step, seed=seed,
                alias_explanation=plan()['internal_alias_explanation'],
                checkpoint=str(checkpoint), checkpoint_itr=info['itr'], update_number=info['updates'],
                logdir=str(logdir), expected_settings=expected,
                parent_commit=SOURCE_COMMIT, source_path=str(old.V6_SOURCE),
                source_repository=str(old.V6_REPO), variant_ids=plan()['variants'],
                console_log=str(HERE / 'logs' / (run_id + '.out')),
                overrides=overrides, cap_seconds=CAP, gpu_count=1, gpu_type='a6000',
                cpus=40, memory_gib=64, monitor_interval_seconds=1800,
                command=[str(PROJECT / 'results/repro_halfcheetah/env/bin/python'), '-B',
                    'script/run.py', '--config-dir=cfg/gym/finetune/halfcheetah-v2',
                    '--config-name=ft_ppo_diffusion_mlp'] + overrides)


def verify(spec):
    result = old.verify(spec)
    meta = old.read(Path(spec['logdir']) / ('grad_mean.json' if spec['step'] == 'prepare' else 'meta.json'))
    if meta['point'] != spec['collector_point'] or meta['source_checkpoint'] != spec['checkpoint']:
        raise ValueError('External point does not match its own checkpoint and internal alias')
    result.update(external_point_id=spec['point'], collector_point=spec['collector_point'],
                  checkpoint=spec['checkpoint'])
    return result


def worker(args):
    spec = old.read(args.spec)
    if spec != make_spec(spec['run_id']):
        raise ValueError('Worker spec differs from the exact approved command')
    provenance = old.validate_v6_gate(GATE, SOURCE_COMMIT)
    manifest_path = HERE / 'workers' / spec['run_id'] / 'manifest.json'
    if any(p.exists() for p in (manifest_path, Path(spec['console_log']), Path(spec['logdir']))):
        raise ValueError('Refusing to overwrite an existing run')
    manifest = dict(spec, status='running', started_utc=old.utc(),
                    scheduler_job_id=os.environ.get('SLURM_JOB_ID'), source_provenance=provenance,
                    monitor_status='active', environment={key: os.environ.get(key) for key in
                      ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'DPPO_DATA_DIR', 'DPPO_LOG_DIR')}, python=sys.version)
    old.atomic(manifest_path, manifest)
    Path(spec['console_log']).parent.mkdir(parents=True, exist_ok=True)
    started, next_monitor, previous, stalled = time.monotonic(), 0, None, 0
    child, stop = None, []
    signal.signal(signal.SIGTERM, lambda sig, frame: stop.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stop.append(sig))
    failure_deadline = None
    try:
        inventory = old.command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])['stdout']
        if not inventory or any('A6000' not in line for line in inventory.splitlines()):
            raise ValueError('An A6000 allocation is required')
        manifest['gpu_inventory'] = inventory
        with Path(spec['console_log']).open('x') as log:
            child = subprocess.Popen(spec['command'], cwd=str(old.V6_SOURCE), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                observation = old.observe(spec)
                if observation['meta_nonfinite'] or observation['status'] == 'failed':
                    if failure_deadline is None:
                        failure_deadline = min(started + CAP - 30, time.monotonic() + 60)
                        manifest['native_failure_observation'] = observation
                    if time.monotonic() >= failure_deadline:
                        raise RuntimeError('Native failure; clean-exit grace expired')
                if stop or time.monotonic() - started >= CAP - 30:
                    raise RuntimeError('Owned worker stopped or its one-hour cap reached')
                if time.monotonic() >= next_monitor:
                    counter = (observation['progress'].get('phase'), observation['completed_batches'])
                    stalled = stalled + 1 if counter == previous else 0
                    previous = counter
                    observation['suspected_stall_confirmed'] = stalled >= 2
                    old.append(manifest_path.parent / 'monitor_history.jsonl', observation)
                    manifest['last_observation'] = observation
                    old.atomic(manifest_path, manifest)
                    next_monitor = time.monotonic() + 1800
                time.sleep(15)
            members = old.process_group_members(child.pid)
            manifest.update(subprocess_exitcode=child.returncode, subprocess_pid=child.pid,
                            remaining_process_group_members=members)
            if child.returncode or failure_deadline is not None or members:
                raise RuntimeError('Native subprocess failed or left children')
        old.validate_v6_gate(GATE, SOURCE_COMMIT)
        manifest.update(status='complete', verification=verify(spec))
        exitcode = 0
    except Exception as error:
        if child is not None:
            try:
                manifest['failure_cleanup_survivors'] = old.terminate_owned_group(child)
            except Exception as cleanup_error:
                manifest['failure_cleanup_error'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
        manifest.update(status='failed', failure_reason=type(error).__name__ + ': ' + str(error))
        exitcode = 1
    manifest.update(ended_utc=old.utc(), elapsed_seconds=time.monotonic() - started,
                    last_observation=old.observe(spec), monitor_status='paused')
    old.atomic(manifest_path, manifest)
    return exitcode


def record(state, monitor=False):
    state['updated_utc'] = old.utc()
    entries = state['entries']
    actual = sum(e.get('allocated_gpu_seconds', 0) for e in entries if e.get('scheduler_state') in old.TERMINAL)
    reserved = sum(CAP for e in entries if e.get('scheduler_state') not in old.TERMINAL)
    state['ledger'] = dict(base_gpu_seconds=BASE_GPU_SECONDS, new_actual_gpu_seconds=actual,
                          outstanding_reserved_gpu_seconds=reserved,
                          aggregate_actual_plus_reserved=BASE_GPU_SECONDS + actual + reserved,
                          aggregate_gpu_seconds_cap=BUDGET_GPU_SECONDS)
    old.atomic(STATE, state)
    if monitor:
        old.append(HERE / 'monitor_history.jsonl', dict(utc=old.utc(), status=state['status'],
                   entries=[{k:e.get(k) for k in ('run_id', 'job_id', 'scheduler_state', 'status', 'progress')} for e in entries],
                   ledger=state['ledger']))
    summary = ['# C1/C2 collection for clean figures version3', '',
               'Status: **' + state['status'] + '**. Updated ' + state['updated_utc'] + '.', '',
               'Twenty50-batch V1-V6 collect jobs and two5-batch critic preparations; no actor updates. '
               'The collector uses internal labels D1/D2 for external C1/C2; all paths, checkpoints and seeds are separate.', '',
               'Monitor interval:30minutes. Estimate:60–70minutes plus scheduling, based on completed V6 jobs. '
               'MAX_PAR=5; each worker1A6000/40CPUs/64GiB with a1-hour cap; inherited aggregate cap125GPU-hours.', '',
               '| Run | Scheduler job | State | Progress |', '|---|---|---|---|']
    for e in entries:
        summary.append('| {run_id} | {job_id} | {status} | {progress} |'.format(
            run_id=e['run_id'], job_id=e['job_id'], status=e.get('status', e.get('scheduler_state')),
            progress=e.get('progress', '—')))
    summary += ['', 'Next step: ' + ('regenerate the authorized clean figures from completed measurements.' if state['status']=='complete'
                    else 'finish only the fixed C1/C2 jobs; no adaptive rounds or automatic replacement jobs.'), '']
    (HERE / 'experiment.md').write_text('\n'.join(summary))
    # Maintain only this run's dedicated marker. Never rewrite original G state.
    path = G / 'experiment.md'
    start, end = '<!-- clean-v3-status-start -->', '<!-- clean-v3-status-end -->'
    complete = sum(e.get('status') == 'complete' for e in entries)
    block = (start + '\n**Clean figures version3 collection (' + state['updated_utc'] + '): '
             + state['status'] + '.** ' + str(complete) + '/22 jobs verified. '
             'The 30-minute monitor is ' + state.get('monitor_status', 'active') + '. '
             '[Collection record](runs/clean_v3_collection_20261012/experiment.md).\n' + end)
    for _ in range(3):
        text = path.read_text()
        revised = (text[:text.index(start)] + block + text[text.index(end) + len(end):]
                   if start in text and end in text else text.rstrip() + '\n\n' + block + '\n')
        temporary = path.with_name('.experiment.clean-v3.' + str(os.getpid()))
        temporary.write_text(revised)
        if path.read_text() != text:
            temporary.unlink()
            continue
        os.replace(str(temporary), str(path))
        break
    else:
        raise RuntimeError('Concurrent experiment.md edit; no overwrite performed')


def update_entries(state):
    for entry in state['entries']:
        if entry.get('status') in ('complete', 'failed'):
            continue
        status = old.accounting(entry['job_id'])
        entry.update(status)
        manifest = old.read(HERE / 'workers' / entry['run_id'] / 'manifest.json', {})
        if manifest:
            obs = old.observe(old.read(HERE / 'specs' / (entry['run_id'] + '.json')))
            entry['progress'] = obs['completed_batches']
        if status.get('scheduler_state') in old.TERMINAL:
            if status['scheduler_state'] == 'COMPLETED' and status.get('exit_code') == '0:0' and manifest.get('status') == 'complete':
                entry['verification'] = verify(old.read(HERE / 'specs' / (entry['run_id'] + '.json')))
                entry['status'] = 'complete'
            else:
                entry['status'] = 'failed'
                entry['failure_reason'] = manifest.get('failure_reason', 'Scheduler or worker did not complete successfully')


def ready(state):
    entries = {e['run_id']: e for e in state['entries']}
    for run_id in RUN_IDS:
        if run_id in entries:
            continue
        point, group = identify(run_id)
        if group is None or entries.get('prepare_' + point, {}).get('status') == 'complete':
            yield run_id


def submit(state, run_id):
    spec = make_spec(run_id)
    path = HERE / 'specs' / (run_id + '.json')
    old.once(path, spec)
    for candidate in (HERE / 'workers' / run_id / 'manifest.json', Path(spec['logdir']), Path(spec['console_log'])):
        if candidate.exists():
            raise ValueError('Refusing to launch over existing output: ' + str(candidate))
    argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue',
            '--export=PATH,HOME,USER,LOGNAME,LANG', '--time=60', '--signal=B:TERM@30',
            '--job-name=gm-v3-' + run_id, '--chdir=' + str(old.V6_SOURCE),
            '--output=' + str(HERE / 'logs' / (run_id + '-%j.out')),
            str(HERE / 'code/run_worker.sh'), '--spec', str(path)]
    state['pending_submission_intent'] = dict(utc=old.utc(), run_id=run_id, command=argv)
    record(state)
    result = old.command(argv, check=False)
    job = result['stdout'].strip().split(';')[0]
    if result['returncode'] or not job.isdigit():
        state['submission_failure'] = result
        record(state)
        raise RuntimeError('Submission failed or ambiguous; no automatic resubmission')
    state['entries'].append(dict(run_id=run_id, job_id=job, submitted_utc=old.utc(), scheduler_state='PENDING',
                                status='submitted', cap_seconds=CAP, reserved_gpu_count=1, command=argv))
    state.pop('pending_submission_intent')
    record(state)


def run(args):
    lock = (HERE / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = old.read(STATE)
    if existing:
        raise ValueError('Controller run already exists; explicit audited recovery required')
    prior = old.read(G / 'runs/campaign.json', {})
    if prior.get('status') != 'complete' or prior.get('ledger', {}).get('actual_gpu_seconds') != BASE_GPU_SECONDS:
        raise ValueError('Old collection changed; recompute the aggregate budget before launch')
    old.once(HERE / 'plan.json', plan())
    provenance = old.validate_v6_gate(GATE, SOURCE_COMMIT)
    if BASE_GPU_SECONDS + 22 * CAP > BUDGET_GPU_SECONDS:
        raise ValueError('All fixed jobs must fit within the remaining aggregate budget')
    state = dict(status='running', started_utc=old.utc(), controller_job_id=os.environ.get('SLURM_JOB_ID'),
                 monitor_status='active', entries=[], plan=plan(), source_provenance=provenance)
    started, next_monitor = time.monotonic(), 0
    stop = []
    signal.signal(signal.SIGTERM, lambda sig, frame: stop.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stop.append(sig))
    try:
        while True:
            update_entries(state)
            record(state, monitor=time.monotonic() >= next_monitor)
            failed = [e for e in state['entries'] if e.get('status') == 'failed']
            if failed:
                raise RuntimeError('Terminal job failure; new admissions stopped, no retries: ' +
                                   ', '.join(e['run_id'] for e in failed))
            if time.monotonic() >= next_monitor:
                next_monitor = time.monotonic() + 1800
            if stop or time.monotonic() - started >= plan()['controller_timeout_seconds'] - 30:
                raise RuntimeError('Controller stopped or capped; running workers are left unchanged')
            entries = state['entries']
            terminal = all(e.get('status') in ('complete', 'failed') for e in entries)
            blocked = {e['run_id'][8:] for e in entries if e['run_id'].startswith('prepare_') and e.get('status') == 'failed'}
            expected = [r for r in RUN_IDS if not any(r.startswith(p + '_j') for p in blocked)]
            if len(entries) == len(expected) and terminal:
                state.update(status='failed' if blocked or any(e['status']=='failed' for e in entries) else 'complete',
                             ended_utc=old.utc(), monitor_status='paused')
                record(state, monitor=True)
                return 0 if state['status']=='complete' else 1
            jobs = old.scheduler_jobs()
            nodes = old.command(['scontrol', 'show', 'nodes', '-o'])['stdout']
            slots = old.queue_slots(entries, jobs, max_parallel=MAX_PAR, free_slots=old.free_a6000_slots(nodes))
            for run_id in list(ready(state))[:slots]:
                record(state)
                if state['ledger']['aggregate_actual_plus_reserved'] + CAP > BUDGET_GPU_SECONDS:
                    raise RuntimeError('Aggregate compute cap prevents further admission')
                submit(state, run_id)
            time.sleep(15)
    except BaseException as error:
        state.update(status='blocked', monitor_status='paused', ended_utc=old.utc(),
                     failure_reason=type(error).__name__ + ': ' + str(error))
        record(state, monitor=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['validate', 'run', 'worker'])
    parser.add_argument('--spec')
    args = parser.parse_args()
    (HERE / 'logs').mkdir(parents=True, exist_ok=True)
    if args.mode == 'validate':
        result = old.validate_v6_gate(GATE, SOURCE_COMMIT)
        old.once(HERE / 'plan.json', plan())
        for point in POINTS:
            old.once(HERE / 'specs' / ('prepare_' + point + '.json'), make_spec('prepare_' + point))
        old.atomic(HERE / 'source_validation.json', result)
        print(json.dumps(dict(source=result, plan=plan()), indent=2))
        return 0
    return worker(args) if args.mode == 'worker' else run(args)


if __name__ == '__main__':
    sys.exit(main())
