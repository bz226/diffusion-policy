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


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    return value


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


def close_status(rho, target):
    if target not in TARGETS:
        raise ValueError('Closure is permitted only at an approved round boundary')
    resolved = isinstance(rho, (int, float)) and math.isfinite(rho) and rho >= .68
    return 'closed_rho' if resolved else ('closed_cap' if target == 4100 else 'open')


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
        score_sample = 200 if group == 0 else 0
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
    return dict(utc=utc(), progress=safe(progress), status=meta.get('status'), error=meta.get('error'),
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
    cleanup = read(directory / 'cleanup.json', {})
    if (cleanup.get('status') != 'passed' or cleanup.get('all_children_exited') is not True or
            cleanup.get('remaining_active_children') != [] or cleanup.get('remaining_direct_children') != [] or
            cleanup.get('errors') != [] or cleanup.get('primary_error') is not None or
            any(x.get('exitcode') != 0 or x.get('alive') is not False for x in cleanup.get('workers', []))):
        raise ValueError('Missing or failed clean vector-environment/child shutdown')
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
        if expected['gm_score_sample'] == 200:
            with (directory / 'decision_scores.csv').open() as stream:
                if len(list(csv.DictReader(stream))) != 200:
                    raise ValueError('First group of each point requires two hundred decision-score samples')
        cross = np.asarray(meta.get('gradient_cross_sum'))
        if cross.shape != (5, 5) or not np.isfinite(cross).all():
            raise ValueError('Missing finite five-estimator cross sums')
    return dict(verified_utc=utc(), cleanup_path=str(directory / 'cleanup.json'),
                metadata_path=str(directory / ('grad_mean.json' if spec['step'] == 'prepare' else 'meta.json')))


def process_group_members(group):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if os.getpgid(int(path.name)) == group:
                members.append(int(path.name))
        except (ProcessLookupError, PermissionError):
            continue
    return sorted(members)


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
                    scheduler_job_id=os.environ.get('SLURM_JOB_ID'), monitor_status='active',
                    environment=dict(python=sys.version, executable=sys.executable,
                                     CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                     OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
                                     MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
                                     OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
                                     DPPO_DATA_DIR=os.environ.get('DPPO_DATA_DIR'), DPPO_LOG_DIR=os.environ.get('DPPO_LOG_DIR')))
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
        manifest['gpu_inventory'] = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])['stdout']
        if not manifest['gpu_inventory'] or any('A6000' not in line for line in manifest['gpu_inventory'].splitlines()):
            raise ValueError('Allocation does not match the approved A6000 GPU type')
        atomic(manifest_path, manifest)
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
            members = process_group_members(child.pid)
            manifest.update(subprocess_exitcode=child.returncode, subprocess_pid=child.pid,
                            remaining_process_group_members=members)
            if members:
                raise RuntimeError('Measurement subprocess exited but owned process-group children remain: ' + str(members))
        validate_gate(args.gate, args.commit)
        manifest.update(status='complete', verification=verify(spec))
        returncode = 0
    except Exception as error:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=15)
        manifest.update(status='failed', failure_reason=type(error).__name__ + ': ' + str(error))
        if child is not None:
            manifest.update(subprocess_exitcode=child.returncode, subprocess_pid=child.pid,
                            remaining_process_group_members=process_group_members(child.pid))
        returncode = 1
    manifest.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                    last_observation=observe(spec), monitor_status='paused')
    atomic(manifest_path, manifest)
    return returncode


class Campaign:
    def __init__(self, args):
        self.args = args
        approval()
        (ROOT / 'runs').mkdir(parents=True, exist_ok=True)
        (ROOT / 'setup').mkdir(exist_ok=True)
        self.lock = (ROOT / 'runs/campaign.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.source = validate_gate(args.gate, args.commit, full=args.action == 'run')
        self.state = read(STATE, dict(created_utc=utc(), status='prepared', submitted=[],
                                    external=[], points={p: {'status': 'open', 'rounds': []} for p in POINTS}))
        if self.state.get('pending_submission_intent'):
            raise RuntimeError('Ambiguous prior submission: inspect Slurm and resolve the recorded intent before resuming')
        self.state.update(source_provenance=self.source, gate=str(args.gate), monitor_status='active')
        self.started = time.monotonic()
        self.next_monitor = self.started
        self.stopping = False
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        self.load_external()
        self.record()

    def handle_signal(self, signum, frame):
        self.stopping = True

    def load_external(self):
        path = ROOT / 'provenance/external_allocations.json'
        if not path.exists():
            raise ValueError('Root must record external GPU gate allocations before runtime submissions')
        data = read(path)
        entries = data['allocations'] if isinstance(data, dict) else data
        external = []
        previous = {str(x['job_id']): x for x in self.state['external']}
        for row in entries:
            if int(row.get('gpu_count', 0)) == 0:
                continue
            if int(row['gpu_count']) != 1 or not 0 < int(row['cap_seconds']) <= 7200:
                raise ValueError('External GPU allocation differs from bounded one-GPU gate')
            job = str(row['job_id'])
            if not job.isdigit():
                raise ValueError('External gate needs its actual Slurm job ID')
            record = dict(previous.get(job, {}), job_id=job, cap_seconds=int(row['cap_seconds']),
                          reserved_gpu_count=1, purpose=row.get('purpose', 'GPU gate'), external=True)
            external.append(record)
        if sum(x['cap_seconds'] for x in external) > 7200 or len({x['job_id'] for x in external}) != len(external):
            raise ValueError('External gate reservations exceed the approved aggregate two hours or duplicate a job')
        if not set(previous).issubset({x['job_id'] for x in external}):
            raise ValueError('Previously recorded external GPU allocations cannot disappear')
        self.state['external'] = external

    def record(self):
        allocations = self.state['external'] + self.state['submitted']
        if self.state.get('pending_submission_intent'):
            allocations = allocations + [dict(cap_seconds=CAP, reserved_gpu_count=1)]
        self.state.update(updated_utc=utc(), ledger=ledger(allocations))
        atomic(STATE, self.state)

    def inspect(self):
        self.load_external()
        for entry in self.state['external'] + self.state['submitted']:
            if entry.get('scheduler_state') in TERMINAL and entry.get('terminal_recorded'):
                continue
            status = accounting(entry['job_id'])
            if status['scheduler_state'] is None:
                entry.setdefault('unknown_since', time.time())
                if time.time() - entry['unknown_since'] > 300:
                    raise RuntimeError('Owned allocation unobservable for five minutes: ' + entry['job_id'])
                continue
            entry.pop('unknown_since', None)
            entry.update(status)
            if status['allocated_gpu_seconds'] > entry['cap_seconds'] * entry.get('reserved_gpu_count', 1):
                raise RuntimeError('Allocation exceeded its approved reservation: ' + entry['job_id'])
            if status['scheduler_state'] not in TERMINAL:
                continue
            if entry.get('external'):
                entry['terminal_recorded'] = True
                continue
            spec = read(ROOT / 'runs/specs' / (entry['run_id'] + '.json'))
            path = ROOT / 'runs' / entry['run_id'] / 'manifest.json'
            manifest = read(path, dict(spec, status='failed', failure_reason='Allocation ended before worker manifest'))
            try:
                if status['scheduler_state'] != 'COMPLETED' or manifest.get('status') != 'complete':
                    raise ValueError('Scheduler/worker did not both complete successfully')
                manifest['final_verification'] = verify(spec)
            except Exception as error:
                manifest.update(status='failed', terminal_failure_reason=str(error), last_observation=observe(spec))
            manifest.update(scheduler_observation=status, monitor_status='paused')
            atomic(path, manifest)
            entry.update(terminal_recorded=True, run_status=manifest['status'])
        self.record()

    def entry(self, run_id):
        return next((x for x in self.state['submitted'] if x['run_id'] == run_id), None)

    def successful_group(self, point, group):
        original = '%s_j%02d' % (point, group)
        for name in (original, original + '_retry1'):
            entry = self.entry(name)
            if entry and entry.get('terminal_recorded') and entry.get('run_status') == 'complete':
                return name
        return None

    def slots(self):
        return queue_slots(self.state['external'] + self.state['submitted'], scheduler_jobs())

    def submit(self, run_id):
        if self.entry(run_id):
            raise ValueError('Run already admitted: ' + run_id)
        source = validate_gate(self.args.gate, self.args.commit, full=self.args.action == 'run')
        if source['stage'] == 'offline' and run_id not in ('prepare_P0', 'P0_j00', 'P0_j01'):
            raise ValueError('Offline gate authorizes only P0 preparation and the first two collection groups')
        if source['stage'] == 'offline' and any(x.get('run_status') == 'failed' for x in self.state['submitted']):
            raise ValueError('A live-gate job failed; further collection is stopped')
        point, group, retry = parse_run(run_id)
        if retry:
            first = self.entry('%s_j%02d' % (point, group))
            if not first or not first.get('terminal_recorded') or first.get('run_status') != 'failed':
                raise ValueError('Replacement requires its preserved failed original')
        slots = self.dispatch_slots if hasattr(self, 'dispatch_slots') else self.slots()
        if slots <= 0:
            return False
        budget = ledger(self.state['external'] + self.state['submitted'])
        if budget['actual_plus_reserved_gpu_seconds'] + CAP > BUDGET:
            self.state['budget_admission_wait'] = dict(utc=utc(), requested_run=run_id, ledger=budget)
            self.record()
            return False
        if any(x.get('unknown_since') for x in self.state['external'] + self.state['submitted']):
            return False
        spec = run_spec(run_id, self.args.commit)
        if (ROOT / 'runs' / run_id / 'manifest.json').exists() or Path(spec['logdir']).exists() or Path(spec['console_log']).exists():
            raise ValueError('Unregistered prior output cannot be overwritten: ' + run_id)
        spec_path = ROOT / 'runs/specs' / (run_id + '.json')
        once(spec_path, spec)
        argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue',
                '--export=PATH,HOME,USER,LOGNAME,LANG', '--time=60', '--signal=B:TERM@30',
                '--job-name=gm-' + run_id, '--chdir=' + str(SOURCE),
                '--output=' + str(ROOT / 'setup' / (run_id + '-%j.out')),
                str(ROOT / 'code/run_worker.sh'), '--spec', str(spec_path), '--commit', self.args.commit,
                '--gate', str(self.args.gate)]
        self.state['pending_submission_intent'] = dict(utc=utc(), run_id=run_id, command=argv, reserved_gpu_seconds=CAP)
        self.record()
        result = command(argv, check=False)
        job = result['stdout'].strip().split(';')[0]
        if result['returncode'] or not job.isdigit():
            self.state['submission_failure'] = result
            self.record()
            raise RuntimeError('Submission failed or ambiguous; intent retained and automatic resubmission forbidden')
        self.state['submitted'].append(dict(run_id=run_id, job_id=job, cap_seconds=CAP, reserved_gpu_count=1,
                                            scheduler_state='PENDING', submitted_utc=utc(), command=argv,
                                            source_provenance=source))
        self.state.pop('pending_submission_intent')
        if hasattr(self, 'dispatch_slots'):
            self.dispatch_slots -= 1
        self.state.pop('budget_admission_wait', None)
        self.record()
        return True

    def summarize(self, point, target):
        names = [self.successful_group(point, group) for group in range(target // 50)]
        if any(name is None for name in names):
            raise ValueError('Round summary requires every logical group to complete')
        for name in names:
            verify(read(ROOT / 'runs/specs' / (name + '.json')))
        sys.path.insert(0, str(ROOT))
        from gradmean_math import summarize_group_files
        summary = summarize_group_files([ROOT / 'collect' / name for name in names], split_seed=8675310)
        path = ROOT / 'runs/rounds' / ('%s_%04d.json' % (point, target))
        once(path, summary)
        estimate = summary['variants']['V4']['estimate']
        scale = summary['variants']['V4'].get('scale_reporting', {})
        rho = estimate.get('rho')
        record = dict(target_batches=target, summary=str(path), rho_V4=rho, groups=names,
                      scale_reporting=scale, forecast_batches=estimate.get('forecast_batches'),
                      forecast_batches_lower_bound=estimate.get('forecast_batches_lower_bound'), recorded_utc=utc())
        forecast = scale.get('K_needed', estimate.get('forecast_batches'))
        if target >= 1400 and forecast is not None and forecast > 12300:
            record['forecast_exceeds_12300_flag_continue'] = True
        self.state['points'][point]['rounds'].append(record)
        self.state['points'][point]['status'] = close_status(rho, target)
        self.record()
        self.monitor(reason='round boundary ' + point + ' ' + str(target))

    def dispatch(self):
        self.dispatch_slots = self.slots()
        # Preparation failures do not authorize an implicit rerun.
        for point in PRIORITY:
            state = self.state['points'][point]
            if state['status'] != 'open':
                continue
            prepare = self.entry('prepare_' + point)
            if not prepare:
                self.submit('prepare_' + point)
            elif prepare.get('terminal_recorded') and prepare.get('run_status') != 'complete':
                state.update(status='failed_prepare', failure_run=prepare['run_id'])
        # Global round barriers preserve the approved per-round priority.
        unfinished = [p for p in POINTS if self.state['points'][p]['status'] == 'open']
        if not unfinished:
            self.record()
            return
        round_index = min(len(self.state['points'][p]['rounds']) for p in unfinished)
        target = TARGETS[round_index]
        for point in PRIORITY:
            state = self.state['points'][point]
            prepare = self.entry('prepare_' + point)
            if state['status'] != 'open' or len(state['rounds']) != round_index or not prepare or not prepare.get('terminal_recorded') or prepare.get('run_status') != 'complete':
                continue
            complete = True
            for group in range(target // 50):
                if self.successful_group(point, group):
                    continue
                complete = False
                original = '%s_j%02d' % (point, group)
                entry = self.entry(original)
                if not entry:
                    self.submit(original)
                elif entry.get('terminal_recorded') and entry.get('run_status') != 'complete':
                    replacement = self.entry(original + '_retry1')
                    if replacement is None:
                        self.submit(original + '_retry1')
                    elif replacement.get('terminal_recorded') and replacement.get('run_status') != 'complete':
                        state.update(status='failed_collection', failure_run=replacement['run_id'], failed_group=group)
                        break
            if complete:
                self.summarize(point, target)
        self.record()

    def monitor(self, reason='periodic', terminal=False):
        observations = []
        for entry in self.state['submitted']:
            spec = read(ROOT / 'runs/specs' / (entry['run_id'] + '.json'))
            observations.append(dict(run_id=entry['run_id'], job_id=entry['job_id'],
                                     scheduler_state=entry.get('scheduler_state'), run_status=entry.get('run_status'),
                                     observation=observe(spec)))
        record = dict(utc=utc(), reason=reason, status=self.state['status'], monitor_status='paused' if terminal else 'active',
                      ledger=self.state['ledger'], points=self.state['points'], runs=observations,
                      next_step='Review terminal records; no automatic expansion' if terminal else 'Complete admitted groups and evaluate the next approved round boundary')
        append(ROOT / 'runs/campaign_monitor_history.jsonl', record)
        atomic(ROOT / 'runs/campaign_progress.json', record)
        report = ROOT / 'experiment.md'
        start, end = '<!-- gradmean-status-start -->', '<!-- gradmean-status-end -->'
        if report.exists():
            content = report.read_text()
            status = '; '.join(p + ': ' + self.state['points'][p]['status'] + ', ' +
                               str(sum(self.successful_group(p, j) is not None for j in range(82)) * 50) + ' batches' for p in POINTS)
            block = start + '\nLast monitor: ' + record['utc'] + '. **' + record['status'] + '**. ' + status + '. '
            block += 'GPU hours used: %.3f; outstanding reserved: %.3f; cap:125. ' % (record['ledger']['actual_gpu_seconds'] / 3600, record['ledger']['outstanding_reserved_gpu_seconds'] / 3600)
            block += 'Monitor ' + record['monitor_status'] + '. Next: ' + record['next_step'] + '.\n' + end
            if start in content and end in content:
                content = content[:content.index(start)] + block + content[content.index(end) + len(end):]
            else:
                content = content.rstrip() + '\n\n' + block + '\n'
            tmp = report.with_name('.experiment.monitor.tmp')
            tmp.write_text(content)
            os.replace(str(tmp), str(report))
        self.next_monitor = time.monotonic() + 9600

    def run(self):
        for run_id in ('prepare_P0', 'P0_j00', 'P0_j01'):
            entry = self.entry(run_id)
            if not entry or not entry.get('terminal_recorded') or entry.get('run_status') != 'complete':
                raise ValueError('Production must adopt all three verified live-gate allocations: ' + run_id)
            verify(read(ROOT / 'runs/specs' / (run_id + '.json')))
        self.state.update(status='running', monitor_status='active', controller_job_id=os.environ.get('SLURM_JOB_ID'))
        try:
            while True:
                if self.stopping or time.monotonic() - self.started >= self.args.timeout_seconds - 60:
                    raise RuntimeError('Controller interrupted/capped; new admissions stopped; existing allocations retain their own caps')
                self.inspect()
                self.dispatch()
                active = [x for x in self.state['submitted'] if not x.get('terminal_recorded')]
                closed = all(x['status'] != 'open' for x in self.state['points'].values())
                exhausted = self.state['ledger']['remaining_unreserved_gpu_seconds'] < CAP and not active
                if (closed or exhausted) and not active:
                    failed = any(x['status'].startswith('failed') for x in self.state['points'].values())
                    self.state.update(status='budget_exhausted' if exhausted and not closed else ('complete_with_failures' if failed else 'complete'),
                                      ended_utc=utc(), monitor_status='paused')
                    self.record()
                    self.monitor(reason='verified terminal', terminal=True)
                    return 0 if self.state['status'] == 'complete' else 2
                if time.monotonic() >= self.next_monitor:
                    self.monitor()
                time.sleep(30)
        except Exception as error:
            self.state.update(status='blocked', failure_reason=type(error).__name__ + ': ' + str(error), monitor_status='paused', ended_utc=utc())
            self.record()
            self.monitor(reason='controller blocked', terminal=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'submit', 'status', 'run', 'worker'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=86400)
    args = parser.parse_args()
    args.gate = args.gate.resolve()
    if not 60 <= args.timeout_seconds <= 86400:
        raise ValueError('Controller cap must be between one minute and twenty-four hours')
    if args.action == 'worker':
        if args.spec is None:
            parser.error('worker requires --spec')
        return worker(args)
    if args.action == 'validate':
        approval()
        print(json.dumps(validate_gate(args.gate, args.commit), indent=2))
        return 0
    campaign = Campaign(args)
    campaign.inspect()
    if args.action == 'status':
        if campaign.source['stage'] == 'offline':
            failed = any(x.get('run_status') == 'failed' for x in campaign.state['submitted'])
            ready = all(campaign.entry(name) and campaign.entry(name).get('terminal_recorded') and
                        campaign.entry(name).get('run_status') == 'complete'
                        for name in ('prepare_P0', 'P0_j00', 'P0_j01'))
            campaign.state['status'] = 'live_gate_failed' if failed else ('live_gate_ready' if ready else 'live_gate_running')
            campaign.state['monitor_status'] = 'paused' if failed or ready else 'active'
            campaign.record()
            campaign.monitor(reason='live-gate status check', terminal=failed or ready)
        print(json.dumps(campaign.state, indent=2))
        return 0
    if args.action == 'submit':
        if not args.run_id:
            parser.error('submit requires --run-id')
        if campaign.source['stage'] != 'offline':
            raise ValueError('Manual submissions are only the three offline-gated P0 jobs; use run for production')
        admitted = campaign.submit(args.run_id)
        campaign.state['status'] = 'live_gate_running'
        campaign.record()
        campaign.monitor(reason='manual live-gate admission')
        print(json.dumps(dict(admitted=admitted, run_id=args.run_id, ledger=campaign.state['ledger']), indent=2))
        return 0 if admitted else 3
    return campaign.run()


if __name__ == '__main__':
    sys.exit(main())
