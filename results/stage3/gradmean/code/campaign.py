"""Approved Addendum G runtime: bounded admissions, immutable runs, adaptive rounds.

The only scientific aggregation is delegated to the independently gated
gradmean_math module. This module neither trains a policy nor edits its source.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PARENT / 'results/stage3/gradmean'
SOURCE = PARENT / 'results/stage3_gradmean/dppo'
STATE = ROOT / 'runs/campaign.json'
JOURNAL = ROOT / 'runs/commands.jsonl'
POINTS = ('P0', 'D1', 'D2', 'N1', 'N2')
PRIORITY = ('P0', 'D2', 'N2', 'D1', 'N1')
TARGETS = (500, 1400, 2300, 3200, 4100)
CAP, BUDGET, MAX_PAR = 3600, 125 * 3600, 9
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
COMMON = ['train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
          'model.randn_clip_value=100', '+model.clamp_logprob=false', '+train.mode=grad_mean']


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(tmp), str(path))


def once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing immutable record differs: ' + str(path))
    else:
        atomic(path, value)


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def command(argv, check=True):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    record = dict(utc=utc(), command=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    append(JOURNAL, record)
    if check and result.returncode:
        raise RuntimeError('Command failed: ' + repr(argv) + ': ' + result.stderr)
    return record


def files(directory):
    result = {}
    for path in directory.rglob('*'):
        relative = path.relative_to(directory)
        if any(x in ('__pycache__', '.pytest_cache') for x in relative.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        if '.git' in relative.parts or path.is_symlink():
            raise ValueError('Scientific snapshot must consist of ordinary files: ' + str(path))
        if path.is_file():
            result[str(relative)] = path
    return result


def validate_gate(path, commit, full=False):
    gate = read(path, {})
    if gate.get('passed') is not True or gate.get('stage') not in ('offline', 'full'):
        raise ValueError('A passed offline/full Addendum G gate is required')
    if full and (gate.get('stage') != 'full' or gate.get('live_gate_passed') is not True):
        raise ValueError('Production requires the full passed G1-G5 gate')
    if gate.get('parent_commit') != commit:
        raise ValueError('Gate reference commit mismatch')
    branch = command(['git', '-C', str(PARENT), 'branch', '--show-current'])['stdout'].strip()
    if branch != 'stage3-gradmean':
        raise ValueError('Parent branch must be stage3-gradmean')
    current_commit = command(['git', '-C', str(PARENT), 'rev-parse', 'HEAD'])['stdout'].strip()
    if command(['git', '-C', str(PARENT), 'merge-base', '--is-ancestor', commit, current_commit], check=False)['returncode']:
        raise ValueError('Current parent revision must descend from the gate reference')
    snapshot = Path(gate.get('source_snapshot', '')).resolve()
    if ROOT not in snapshot.parents or snapshot == SOURCE:
        raise ValueError('Gate must preserve a separate ordinary source snapshot under the output folder')
    actual, expected = files(SOURCE), files(snapshot)
    if not expected or actual.keys() != expected.keys():
        raise ValueError('Source file set differs from passed gate')
    for name, previous in expected.items():
        if actual[name].read_bytes() != previous.read_bytes() or (actual[name].stat().st_mode & 0o111) != (previous.stat().st_mode & 0o111):
            raise ValueError('Source changed after gate: ' + name)
    prefix = str(SOURCE.relative_to(PARENT)) + '/'
    listing = command(['git', '-C', str(PARENT), 'ls-tree', '-rz', commit, '--', prefix])['stdout']
    entries = {}
    for entry in listing.split('\0'):
        if not entry:
            continue
        metadata, name = entry.split('\t', 1)
        mode, kind, oid = metadata.split()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('Committed source contains a nonordinary file')
        entries[name[len(prefix):]] = (mode, oid)
    if entries.keys() != expected.keys():
        raise ValueError('Reference commit does not contain the complete gated source')
    ordered = sorted(entries)
    payload = ''.join(entries[name][1] + '\n' for name in ordered).encode()
    result = subprocess.run(['git', '-C', str(PARENT), 'cat-file', '--batch'], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise ValueError('Cannot read reference source blobs')
    offset = 0
    for name in ordered:
        end = result.stdout.index(b'\n', offset)
        header = result.stdout[offset:end].split()
        size = int(header[2])
        data = result.stdout[end + 1:end + 1 + size]
        offset = end + 2 + size
        if data != expected[name].read_bytes() or (entries[name][0] == '100755') != bool(expected[name].stat().st_mode & 0o100):
            raise ValueError('Reference commit differs from gate source: ' + name)
    return dict(gate_path=str(Path(path).resolve()), reference_parent_commit=commit, current_parent_commit=current_commit,
                source_snapshot=str(snapshot), files_compared=len(expected), stage=gate['stage'])


def approval():
    value = read(ROOT / 'provenance/approval.json', {})
    required = {'approved': True, 'gpu_hour_cap': 125, 'max_parallel_gpu_jobs': 9,
                'cpus_per_gpu_job': 40, 'memory_gib_per_gpu_job': 64,
                'prepare_collect_wall_limit_seconds': 3600, 'gate_wall_limit_seconds': 7200,
                'batches_per_collect_job': 50, 'round_targets': list(TARGETS),
                'at_most_one_replacement_per_failed_collect': True}
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError('Approval does not match the bounded Addendum G runtime')
    return value


def parse_run(run_id):
    match = re.fullmatch(r'prepare_(P0|D1|D2|N1|N2)', run_id)
    if match:
        return match[1], None, False
    match = re.fullmatch(r'(P0|D1|D2|N1|N2)_j(\d{2})(_retry1)?', run_id)
    if not match or not 0 <= int(match[2]) <= 81:
        raise ValueError('Run outside approved seed/group slots: ' + run_id)
    return match[1], int(match[2]), bool(match[3])


def seed_for(point, group=None, retry=False):
    return (1000000 if retry else 100000) + 10000 * POINTS.index(point) + 100 * (90 if group is None else group)


def run_spec(run_id, commit):
    point, group, retry = parse_run(run_id)
    registry = read(ROOT / 'provenance/checkpoints.json')[point]
    checkpoint = Path(registry['checkpoint'])
    if not checkpoint.is_file():
        raise ValueError('Approved policy checkpoint missing: ' + str(checkpoint))
    step = 'prepare' if group is None else 'collect'
    seed = seed_for(point, group, retry)
    directory = ROOT / ('prepare' if group is None else 'collect') / (point if group is None else run_id)
    overrides = COMMON + ['seed=' + str(seed), 'logdir=' + str(directory), 'base_policy_path=' + str(checkpoint),
                         '+train.gm_step=' + step, '+train.gm_point=' + point, '+train.gm_parent_commit=' + commit]
    expected = dict(mode='grad_mean', gm_step=step, gm_point=point, seed=seed,
                    environment_seeds=list(range(seed, seed + 40)), gm_parent_commit=commit)
    if group is None:
        overrides += ['+train.gm_warmup_batches=' + str(registry['warmup_batches'])]
        expected['gm_warmup_batches'] = registry['warmup_batches']
    else:
        prepare_manifest = read(ROOT / 'runs' / ('prepare_' + point) / 'manifest.json', {})
        frozen = read(ROOT / 'prepare' / point / 'grad_mean.json', {})
        if prepare_manifest.get('status') != 'complete' or frozen.get('status') != 'completed':
            raise ValueError('Collection requires verified completed preparation: ' + point)
        score_sample = 200 if point == 'P0' and group == 0 else 0
        expected.update(gm_frozen_path=frozen['frozen_path'], gm_frozen_sha256=frozen['frozen_sha256'],
                        gm_batches=50, gm_score_sample=score_sample)
        overrides += ['+train.' + key + '=' + str(expected[key]) for key in ('gm_frozen_path', 'gm_frozen_sha256', 'gm_batches', 'gm_score_sample')]
    return dict(run_id=run_id, point=point, group=group, retry=retry, step=step, seed=seed,
                parent_commit=commit, logdir=str(directory), expected_settings=expected,
                console_log=str(ROOT / (run_id + '.out')), overrides=overrides,
                cap_seconds=CAP, gpu_count=1, gpu_type='a6000', cpus=40, memory_gib=64,
                monitor_interval_seconds=1800, estimate_seconds=900,
                command=[sys.executable, '-B', 'script/run.py', '--config-dir=cfg/gym/finetune/halfcheetah-v2',
                         '--config-name=ft_ppo_diffusion_mlp'] + overrides)


def accounting(job):
    output = command(['sacct', '--allocations', '--noheader', '--parsable2', '--jobs=' + str(job),
                      '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%180'])['stdout']
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) >= 4 and fields[0] == str(job):
            tres = dict(x.split('=', 1) for x in fields[3].split(',') if '=' in x)
            count = int(tres.get('gres/gpu', '0'))
            return dict(job_id=str(job), scheduler_state=fields[1].split()[0].rstrip('+'),
                        elapsed_seconds=int(fields[2]), gpu_count=count,
                        allocated_gpu_seconds=int(fields[2]) * count, allocated_tres=fields[3])
    return dict(job_id=str(job), scheduler_state=None)


def ledger(entries):
    actual = sum(x.get('allocated_gpu_seconds', 0) for x in entries if x.get('scheduler_state') in TERMINAL)
    reserved = sum(x['cap_seconds'] * x.get('reserved_gpu_count', 1) for x in entries if x.get('scheduler_state') not in TERMINAL)
    return dict(actual_gpu_seconds=actual, outstanding_reserved_gpu_seconds=reserved,
                actual_plus_reserved_gpu_seconds=actual + reserved, budget_gpu_seconds=BUDGET,
                remaining_unreserved_gpu_seconds=BUDGET - actual - reserved)


def queue_slots(entries, jobs):
    owned = {str(x['job_id']) for x in entries if x.get('scheduler_state') not in TERMINAL}
    others = [j for j in jobs if j['job_id'] not in owned]
    return max(0, min(MAX_PAR - len(owned), 12 - len(owned) - sum(j['gpus'] for j in others),
                      12 - len(jobs), 20 - len(jobs)))


def scheduler_jobs():
    output = command(['squeue', '--noheader', '--user=' + os.environ.get('USER', 'zbao7'), '--format=%i|%T|%b|%j'])['stdout']
    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        job, state, gres, name = line.split('|', 3)
        gpus = sum(int(m.group(1)) for token in gres.split(',') if 'gpu' in token for m in [re.search(r'[:=](\d+)(?:\([^)]*\))?$', token)] if m)
        jobs.append(dict(job_id=job.strip(), state=state, gpus=gpus, name=name))
    return jobs


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, dict):
        return all(finite_json(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_json(x) for x in value)
    return True


def observe(spec):
    directory = Path(spec['logdir'])
    progress = read(directory / 'stage3_progress.json', {})
    meta = read(directory / ('grad_mean.json' if spec['step'] == 'prepare' else 'meta.json'), {})
    return dict(utc=utc(), progress=progress, status=meta.get('status'), error=meta.get('error'),
                completed_batches=meta.get('completed_batches', meta.get('completed_warmup_batches', 0)),
                meta_nonfinite=not finite_json(meta))


def verify(spec):
    import numpy as np
    directory = Path(spec['logdir'])
    meta = read(directory / ('grad_mean.json' if spec['step'] == 'prepare' else 'meta.json'), {})
    if meta.get('status') != 'completed' or not finite_json(meta):
        raise ValueError('Output not complete and finite')
    expected = spec['expected_settings']
    if any(meta.get('settings', {}).get(key) != val for key, val in expected.items()):
        raise ValueError('Effective grad_mean settings differ from exact approved run')
    if 'GRADMEAN_SETTINGS' not in Path(spec['console_log']).read_text(errors='replace'):
        raise ValueError('Missing startup GRADMEAN_SETTINGS printout')
    if meta.get('actor_before_sha256') != meta.get('actor_after_sha256') or not meta.get('actor_before_sha256'):
        raise ValueError('Actor changed during measurement')
    if meta.get('actor_optimizer_steps') != 0:
        raise ValueError('Unexpected actor optimizer update')
    if spec['step'] == 'prepare':
        if meta.get('completed_warmup_batches') != expected['gm_warmup_batches'] or not Path(meta['frozen_path']).is_file():
            raise ValueError('Preparation incomplete or frozen state absent')
    else:
        if meta.get('completed_batches') != 50 or meta.get('gm_batches') != 50 or meta.get('parameter_count') != 568392:
            raise ValueError('Collection dimensions/count differ')
        if meta.get('frozen_sha256') != expected['gm_frozen_sha256']:
            raise ValueError('Frozen state identity differs')
        for key in ('critic', 'scaler'):
            if not meta.get(key + '_before_sha256') or meta[key + '_before_sha256'] != meta.get(key + '_after_sha256'):
                raise ValueError(key + ' changed during collection')
        for variant in range(1, 6):
            vector = np.load(str(directory / ('sums_V%d.npy' % variant)), mmap_mode='r')
            if vector.shape != (568392,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                raise ValueError('Stored group sum has invalid dimensions/dtype/values')
        with (directory / 'batches.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 50:
            raise ValueError('Collection CSV must contain fifty batches')
        cross = np.asarray(meta.get('gradient_cross_sum'))
        if cross.shape != (5, 5) or not np.isfinite(cross).all():
            raise ValueError('Missing finite five-estimator cross sums')
    return dict(verified_utc=utc(), metadata_path=str(directory / ('grad_mean.json' if spec['step'] == 'prepare' else 'meta.json')))


def worker(args):
    spec = read(args.spec)
    approval()
    source = validate_gate(args.gate, args.commit)
    if spec != run_spec(spec['run_id'], args.commit):
        raise ValueError('Worker spec differs from approved exact command')
    if source['stage'] == 'offline' and spec['run_id'] not in ('prepare_P0', 'P0_j00', 'P0_j01'):
        raise ValueError('Offline gate permits only P0 preparation and two live-gate groups')
    manifest_path = ROOT / 'runs' / spec['run_id'] / 'manifest.json'
    if manifest_path.exists() or Path(spec['console_log']).exists() or Path(spec['logdir']).exists():
        raise ValueError('Refusing to overwrite an existing run')
    manifest = dict(spec, source_provenance=source, gate=str(args.gate), status='running', started_utc=utc(),
                    scheduler_job_id=os.environ.get('SLURM_JOB_ID'), monitor_status='active')
    atomic(manifest_path, manifest)
    started = time.monotonic()
    child = None
    stop = []
    def handle(signum, frame):
        stop.append('worker received signal ' + str(signum))
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    previous = None
    stalled = 0
    next_monitor = started
    try:
        with Path(spec['console_log']).open('x') as log:
            child = subprocess.Popen(spec['command'], cwd=str(SOURCE), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                observation = observe(spec)
                if observation['meta_nonfinite'] or observation['status'] == 'failed':
                    raise RuntimeError('Scientific collection failed: ' + str(observation))
                if stop or time.monotonic() - started >= CAP - 30:
                    raise RuntimeError(stop[0] if stop else 'Approved one-hour allocation cap reached')
                if time.monotonic() >= next_monitor:
                    counter = (observation['progress'].get('phase'), observation['completed_batches'])
                    stalled = stalled + 1 if counter == previous else 0
                    previous = counter
                    observation['suspected_stall_confirmed'] = stalled >= 2
                    append(manifest_path.parent / 'monitor_history.jsonl', observation)
                    manifest.update(last_observation=observation)
                    atomic(manifest_path, manifest)
                    next_monitor = time.monotonic() + 1800
                time.sleep(15)
            if child.returncode:
                raise RuntimeError('Measurement subprocess exit ' + str(child.returncode))
        validate_gate(args.gate, args.commit)
        manifest.update(status='complete', verification=verify(spec))
        returncode = 0
    except Exception as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        manifest.update(status='failed', failure_reason=type(error).__name__ + ': ' + str(error))
        returncode = 1
    manifest.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                    last_observation=observe(spec), monitor_status='paused')
    atomic(manifest_path, manifest)
    return returncode
