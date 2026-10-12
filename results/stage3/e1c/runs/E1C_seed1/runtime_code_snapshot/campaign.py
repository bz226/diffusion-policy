"""E1C-only queue, adapted from the Stage-3 critic runner. Python 3.8.

No actor computation is changed. Stopping this controller does not cancel Slurm
workers. Every attempt has its own output, 4-hour cap and 30-minute monitor.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/e1c'
PARENT = PROJECT / '.codex-runtime/worktrees/stage3-e1c'
SOURCE = PARENT / 'results/stage3_e1c/dppo'
PYTHON = PROJECT / 'results/repro_halfcheetah/env/bin/python'
GATE = ROOT / 'runs/gate/gate_summary.json'
STATE = ROOT / 'runs/campaign.json'
BUDGET = 16 * 3600
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
COMMON = ['model.clip_ploss_coef=1e6', 'model.clip_ploss_coef_base=1e6',
          'train.target_kl=null', '+model.clamp_logprob=false',
          'model.randn_clip_value=100', '+train.actor_single_step=true',
          '+model.actor_loss=score', '+train.stage3_diag=true']
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if hasattr(value, 'item'):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(finite_json(value), allow_nan=False) + '\n')


def once(path, value):
    path = Path(path)
    text = value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Refusing to overwrite immutable record: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(text)


def command(argv, check=True):
    result = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    record = dict(utc=utc(), command=[str(x) for x in argv], returncode=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    append(ROOT / 'runs/commands.jsonl', record)
    if check and result.returncode:
        raise RuntimeError('Command failed: %r: %s' % (argv, result.stderr))
    return record




def accounting(job):
    text = command(['sacct', '--allocations', '--noheader', '--parsable2', '--jobs=' + str(job),
                    '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%160,ExitCode'])['stdout']
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) >= 5 and fields[0] == str(job):
            state = fields[1].split()[0].rstrip('+')
            tres = dict(x.split('=', 1) for x in fields[3].split(',') if '=' in x)
            count = int(tres.get('gres/gpu', '0'))
            return dict(job_id=str(job), scheduler_state=state, elapsed_seconds=int(fields[2]),
                        gpu_count=count, allocated_gpu_seconds=int(fields[2]) * count,
                        allocated_tres=fields[3], exit_code=fields[4])
    return dict(job_id=str(job), scheduler_state=None)


def jobs():
    text = command(['squeue', '--noheader', '--user=' + os.environ.get('USER', 'zbao7'),
                    '--format=%i|%T|%b|%j|%P'])['stdout']
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        job, state, gres, name, partition = line.split('|', 4)
        gpu = sum(int(match.group(1)) for token in gres.split(',') if 'gpu' in token
                  for match in [re.search(r'[:=](\d+)(?:\([^)]*\))?$', token)] if match)
        entry = dict(job_id=job.strip(), state=state, gpus=gpu, name=name, gres=gres, partition=partition)
        entry['a6000_gpus'] = a6000_count(entry)
        if gpu and not re.search(r'gpu:[A-Za-z][A-Za-z0-9_]*:', gres):
            # Generic GPU requests are not proof of a non-A6000 allocation.
            # Resolve running jobs from typed AllocTRES; pending/unknown types
            # conservatively consume A6000 headroom.
            detail = command(['scontrol', 'show', 'job', job.strip(), '-o'])['stdout']
            match = re.search(r'\bAllocTRES=(\S+)', detail)
            typed = re.findall(r'gres/gpu:([^=,]+)=(\d+)', match.group(1)) if match else []
            if sum(int(count) for _, count in typed) == gpu:
                entry['a6000_gpus'] = sum(int(count) for kind, count in typed if kind == 'a6000')
                entry['gpu_type_resolution'] = dict(source='scontrol AllocTRES', allocated_types=typed)
            else:
                entry['gpu_type_resolution'] = dict(source='unresolved; reserve as A6000')
        result.append(entry)
    return result


def a6000_count(job):
    if 'a6000_gpus' in job:
        return job['a6000_gpus']
    typed = re.findall(r'gpu:([A-Za-z][A-Za-z0-9_]*):(\d+)', job['gres'])
    if sum(int(count) for _, count in typed) == job['gpus']:
        return sum(int(count) for kind, count in typed if kind == 'a6000')
    return job['gpus']




def source_files(directory):
    excluded = {'.git', '__pycache__', '.pytest_cache'}
    return {str(p.relative_to(directory)): p for p in directory.rglob('*')
            if p.is_file() and not excluded.intersection(p.relative_to(directory).parts)
            and p.suffix != '.pyc'}




def nonfinite(value, prefix=''):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite(v, prefix + '.' + str(k))]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in nonfinite(v, prefix + '[%d]' % i)]
    if hasattr(value, 'item'):
        return nonfinite(value.item(), prefix)
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    return []


def observe(spec):
    directory = Path(spec['logdir'])
    try:
        progress = read(directory / 'stage3_progress.json', {})
    except json.JSONDecodeError:
        progress = dict(write_in_progress=True)
    result = dict(utc=utc(), run_id=spec['run_id'], progress=progress)
    try:
        result['native_failure'] = read(directory / 'failure.json', {})
    except json.JSONDecodeError:
        result['native_failure_write_in_progress'] = True
    if spec['mode'] == 'train' and (directory / 'result.pkl').exists():
        try:
            with (directory / 'result.pkl').open('rb') as stream:
                rows = pickle.load(stream)
            result.update(rows=len(rows), last=rows[-1] if rows else {},
                          last_itr=rows[-1]['itr'] if rows else None,
                          nonfinite_paths=nonfinite(rows))
            measured = [float(r['time']) for r in rows if 'train_episode_reward' in r and 'time' in r]
            eval_times = [float(r['time']) for r in rows if 'eval_episode_reward' in r and 'time' in r]
            if measured:
                per_train = statistics.median(measured[-20:])
                per_eval = statistics.median(eval_times) if eval_times else 15.0
                last_itr = rows[-1]['itr']
                remaining = sum(per_eval if itr % 10 == 0 else per_train for itr in range(last_itr + 1, 140))
                result['seconds_per_training_iteration'] = per_train
                result['remaining_seconds_estimate'] = remaining
                result['estimated_finish_utc'] = (datetime.datetime.now(datetime.timezone.utc) +
                    datetime.timedelta(seconds=remaining)).isoformat()
            first = next((r.get('eval_episode_reward') for r in rows if r['itr'] == 0), None)
            result['collapse_eval_iterations'] = [r['itr'] for r in rows if first is not None
                                                 and r.get('eval_episode_reward', first) < 0.5 * first]
        except (EOFError, pickle.UnpicklingError):
            result['result_read_in_progress'] = True
    console = Path(spec['console_log'])
    if console.is_file():
        with console.open('rb') as stream:
            stream.seek(max(0, console.stat().st_size - 32768))
            result['console_tail'] = stream.read().decode('utf-8', errors='replace').splitlines()[-25:]
    return finite_json(result)


def collapsed(observation):
    return bool(observation.get('collapse_eval_iterations') or observation.get('nonfinite_paths') or
                str(observation.get('native_failure', {}).get('reason', '')).startswith('nonfinite'))




def group_members(group):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                if os.getpgid(int(path.name)) == group:
                    result.append(int(path.name))
            except (ProcessLookupError, PermissionError):
                pass
    return sorted(result)


def terminate_owned_group(child):
    for signum, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
        child.poll()
        if not group_members(child.pid):
            break
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            child.poll()
            if not group_members(child.pid):
                break
            time.sleep(.2)
    return group_members(child.pid)


def worker(args):
    approval()
    provenance = validate_gate(args.gate)
    spec = read(args.spec)
    if spec != run_spec(spec['seed'], spec.get('attempt', 0)):
        raise ValueError('Worker spec differs from approved exact command')
    path = ROOT / 'runs' / spec['run_id'] / 'manifest.json'
    if path.exists() or Path(spec['logdir']).exists() or Path(spec['console_log']).exists():
        raise ValueError('Refusing to overwrite a previous run')
    manifest = dict(spec, status='running', started_utc=utc(), source_provenance=provenance,
                    scheduler_job_id=os.environ.get('SLURM_JOB_ID'), monitor_status='active')
    manifest['environment'] = {key: os.environ.get(key) for key in (
        'CUDA_VISIBLE_DEVICES', 'DPPO_DATA_DIR', 'DPPO_LOG_DIR', 'OMP_NUM_THREADS',
        'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONPATH', 'LD_LIBRARY_PATH')}
    manifest['python'] = sys.version
    manifest['normalization'] = 'per 50000-pair chunk, matching actual E1; explicitly user approved'
    runtime_snapshot = path.parent / 'runtime_code_snapshot'
    runtime_snapshot.mkdir(parents=True)
    for code in sorted((ROOT / 'code').iterdir()):
        if code.is_file() and code.suffix in ('.py', '.sh'):
            shutil.copy2(str(code), str(runtime_snapshot / code.name))
    manifest['runtime_code_snapshot'] = str(runtime_snapshot)
    atomic(path, manifest)
    child, stopped, result = None, [], 1
    started, next_monitor, previous, stalled = time.monotonic(), 0, None, 0
    def stop(signum, frame):
        stopped.append('Worker received signal%d' % signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        inventory = command(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader'])['stdout']
        if not inventory.strip() or any('A6000' not in line for line in inventory.splitlines()):
            raise ValueError('Approved A6000 allocation required')
        manifest['hardware'] = dict(gpu_inventory=inventory, cpu_affinity_count=len(os.sched_getaffinity(0)))
        if len(os.sched_getaffinity(0)) < 40:
            raise ValueError('Approved40 CPU allocation required')
        atomic(path, manifest)
        Path(spec['console_log']).parent.mkdir(parents=True, exist_ok=True)
        with Path(spec['console_log']).open('x') as log:
            child = subprocess.Popen(spec['command'], cwd=str(SOURCE), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                observation = observe(spec)
                if observation.get('nonfinite_paths'):
                    manifest['collapse'] = True
                    manifest['failure_observation'] = observation
                    raise FloatingPointError('Non-finite run metrics; other runs may continue')
                if stopped or time.monotonic() - started >= spec['cap_seconds'] - 40:
                    raise RuntimeError(stopped[0] if stopped else 'Approved run wall cap reached')
                if time.monotonic() >= next_monitor:
                    counter = (observation.get('last_itr'), observation['progress'].get('phase'),
                               observation['progress'].get('completed'))
                    stalled = stalled + 1 if counter == previous else 0
                    previous = counter
                    observation['suspected_stall_confirmed'] = stalled >= 2
                    append(path.parent / 'monitor_history.jsonl', observation)
                    manifest['last_observation'] = observation
                    atomic(path, manifest)
                    next_monitor = time.monotonic() + 1800
                if observation.get('last_itr', -1) is not None and observation.get('last_itr', -1) >= 70:
                    interim_comparison(spec)
                time.sleep(10)
        manifest.update(subprocess_exitcode=child.returncode, subprocess_pid=child.pid)
        if child.returncode:
            raise RuntimeError('Native process exited%d' % child.returncode)
        if group_members(child.pid):
            raise RuntimeError('Native process left child processes behind')
        validate_gate(args.gate)
        manifest.update(status='complete', verification=verify(spec), remaining_process_group_members=[])
        derive_baseline_energy(spec)
        result = 0
    except Exception as error:
        manifest.update(status='failed', failure_reason=type(error).__name__ + ': ' + str(error))
        if child is not None:
            manifest.update(failure_cleanup_survivors=terminate_owned_group(child),
                            subprocess_exitcode=child.returncode, subprocess_pid=child.pid)
    manifest.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                    last_observation=observe(spec), monitor_status='paused')
    if collapsed(manifest['last_observation']):
        manifest['collapse'] = True
        manifest['failure_iteration'] = manifest['last_observation'].get('native_failure', {}).get('itr')
    atomic(path, manifest)
    return result




def approval():
    item = read(ROOT / 'provenance/approval.json', {})
    if item.get('approved') is not True or item.get('gpu_hour_cap') != 16:
        raise ValueError('E1C needs its explicit 16 GPU-hour approval')
    required = dict(max_parallel_training_jobs=3, training_wall_limit_seconds=14400,
                    combined_gate_wall_limit_seconds=3600, cpus_per_gpu_job=40, memory_gib_per_gpu_job=64,
                    training_iterations=140, save_model_freq=35, max_retries_per_seed=1)
    for key, value in required.items():
        if item.get(key) != value:
            raise ValueError('E1C approval mismatch: ' + key)
    if item.get('seeds') != [0, 1, 2]:
        raise ValueError('Only E1C seeds 0, 1, 2 are approved')
    return item


def validate_gate(path):
    gate = read(path, {})
    if gate.get('passed') is not True:
        raise ValueError('A1-A5 must pass before production')
    for key in ('A1', 'A2', 'A3', 'A4', 'A5'):
        value = gate.get(key, gate.get('checks', {}).get(key))
        if not isinstance(value, dict) or value.get('passed') is not True:
            raise ValueError('Missing passed gate check ' + key)
    current = command(['git', '-C', PARENT, 'rev-parse', 'HEAD'])['stdout'].strip()
    commit = gate.get('parent_commit')
    if not commit:
        raise ValueError('Gate must record the source revision')
    command(['git', '-C', PARENT, 'merge-base', '--is-ancestor', commit, current])
    snapshot = Path(gate['source_snapshot']).resolve()
    if snapshot == SOURCE.resolve() or ROOT not in snapshot.parents:
        raise ValueError('Gate requires a separate source snapshot')
    actual, expected = source_files(SOURCE), source_files(snapshot)
    if not expected or actual.keys() != expected.keys():
        raise ValueError('Scientific source file set changed after gate')
    for name, previous in expected.items():
        if (actual[name].read_bytes() != previous.read_bytes() or
                ((actual[name].stat().st_mode ^ previous.stat().st_mode) & 0o111)):
            raise ValueError('Scientific source changed after gate: ' + name)
    job = gate.get('slurm_job_id')
    status = accounting(job) if job else {'scheduler_state': 'COMPLETED', 'exit_code': '0:0'}
    if status.get('scheduler_state') != 'COMPLETED' or status.get('exit_code') != '0:0':
        raise ValueError('Gate allocation must finish successfully before training')
    return dict(gate=str(path), parent_commit=commit, current_parent_commit=current,
                source_snapshot=str(snapshot), files_compared=len(expected), gate_allocation=status)


def run_spec(seed, attempt=0):
    if seed not in (0, 1, 2) or attempt not in (0, 1):
        raise ValueError('Only three seeds and at most one corrected same-seed retry are allowed')
    base = 'E1C_seed%d' % seed
    run_id = base if attempt == 0 else base + '_retry1'
    logdir = ROOT / base if attempt == 0 else ROOT / 'retries' / run_id
    overrides = ['train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
                 'seed=' + str(seed), 'logdir=' + str(logdir)] + COMMON + [
                 '+train.adv_estimator=mc_critic', '+train.baseline_diag=true']
    return dict(run_id=run_id, condition='E1C', seed=seed, attempt=attempt, mode='train',
                cap_seconds=14400, estimate_seconds=9120, monitor_interval_seconds=1800,
                gpu_count=1, gpu_type='a6000', cpus=40, memory_gib=64,
                logdir=str(logdir), console_log=str(logdir) + '.out', overrides=overrides,
                command=[str(PYTHON), '-B', str(ROOT / 'code/entrypoint.py'),
                         '--config-dir=' + str(SOURCE / 'cfg/gym/finetune/halfcheetah-v2'),
                         '--config-name=ft_ppo_diffusion_mlp'] + overrides)


def prepare_specs():
    approval()
    for seed in (0, 1, 2):
        spec = run_spec(seed)
        once(ROOT / 'runs/specs' / (spec['run_id'] + '.json'), spec)
    once(ROOT / 'jobs_e1c.txt', '# NAME SEEDS OVERRIDES (Stage-3 COMMON supplied by runner)\n'
         'E1C 0,1,2 +train.adv_estimator=mc_critic +train.baseline_diag=true\n')
    return {'status': 'prepared', 'seeds': [0, 1, 2], 'normalization': 'per 50000-pair chunk'}


def verify(spec):
    directory = Path(spec['logdir'])
    cleanup = read(directory / 'cleanup.json', {})
    if cleanup.get('status') != 'passed' or cleanup.get('all_children_exited') is not True:
        raise ValueError('Explicit vector environment cleanup did not pass')
    workers = cleanup.get('workers', [])
    if len(workers) != 40 or any(w.get('exitcode') != 0 or w.get('alive') for w in workers):
        raise ValueError('All forty environment workers must exit cleanly')
    if cleanup.get('remaining_active_children') or cleanup.get('remaining_direct_children'):
        raise ValueError('Child processes remain after cleanup')
    observation = observe(spec)
    if observation.get('nonfinite_paths'):
        raise FloatingPointError('Non-finite run measurements')
    with (directory / 'result.pkl').open('rb') as stream:
        rows = pickle.load(stream)
    if [r['itr'] for r in rows] != list(range(140)) or rows[-1]['step'] != 10080000:
        raise ValueError('Training did not finish the exact 140-iteration schedule')
    for itr in (0, 35, 70, 105, 139):
        if not (directory / ('checkpoint/state_%d.pt' % itr)).is_file():
            raise ValueError('Missing requested checkpoint ' + str(itr))
    train = [row for row in rows if 'train_episode_reward' in row]
    if len(train) != 126 or any(row.get('actor_optimizer_steps') != 1 for row in train):
        raise ValueError('Expected exactly 126 single actor steps')
    if any(row.get('actor_step_ratio') != 1 or row.get('actor_step_clipfrac') != 0 for row in train):
        raise ValueError('Single-step ratio/clip invariant failed')
    for row in train:
        if row.get('actor_lr') != 1e-4:
            raise ValueError('E1C actor learning rate changed')
        for kind in ('critic', 'loo'):
            if len(row.get('baseline_residual_mean_' + kind, [])) != 250:
                raise ValueError('Missing full per-decision residual moments')
    return dict(verified_utc=utc(), cleanup=str(directory / 'cleanup.json'), observation=observation)


def unweighted_energy(row):
    """E[residual^2] = population variance + mean^2, without new sampling."""
    result = {'itr': row['itr'], 'step': row['step'],
              'energy_definition': 'sum_k E_episode[(G_k - b_k)^2], no gamma weights'}
    for name in ('critic', 'loo'):
        means, stds = row['baseline_residual_mean_' + name], row['baseline_residual_std_' + name]
        if len(means) != 250 or len(stds) != 250 or row['baseline_residual_std_ddof'] != 0:
            raise ValueError('Raw per-k population moments are required')
        energy = [float(a) ** 2 + float(b) ** 2 for a, b in zip(means, stds)]
        result['baseline_unweighted_energy_' + name] = sum(energy)
        result['baseline_unweighted_bin_energy_' + name] = [sum(energy[k:k + 50]) for k in range(0, 250, 50)]
    den = result['baseline_unweighted_energy_loo']
    result['baseline_unweighted_energy_ratio'] = result['baseline_unweighted_energy_critic'] / den if den > 0 else None
    result['baseline_unweighted_bin_energy_ratio'] = [a / b if b > 0 else None for a, b in zip(
        result['baseline_unweighted_bin_energy_critic'], result['baseline_unweighted_bin_energy_loo'])]
    return result


def derive_baseline_energy(spec):
    import csv
    with (Path(spec['logdir']) / 'result.pkl').open('rb') as stream:
        rows = pickle.load(stream)
    metrics = [unweighted_energy(row) for row in rows if 'train_episode_reward' in row]
    output = ROOT / 'runs' / spec['run_id'] / 'derived_baseline_energy.json'
    once(output, {'source': str(Path(spec['logdir']) / 'result.pkl'),
                  'original_keys': 'baseline_energy_* remain gamma^(2k)-weighted; never overwritten',
                  'rows': metrics})
    return str(output)


def interim_comparison(spec):
    path = ROOT / 'runs' / spec['run_id'] / 'interim_itr70.json'
    if path.exists():
        return
    records = {}
    for condition in ('E0', 'E1', 'E1C'):
        source = (Path(spec['logdir']) if condition == 'E1C' else
                  PROJECT / ('results/stage3/%s_seed%d' % (condition, spec['seed']))) / 'result.pkl'
        if not source.is_file():
            continue
        try:
            with source.open('rb') as stream:
                rows = pickle.load(stream)
        except (EOFError, pickle.UnpicklingError):
            return
        selected = [row for row in rows if row['itr'] in (69, 70)]
        if len(selected) != 2:
            return
        records[condition] = dict(source=str(source), rows=[{k: row[k] for k in (
            'itr', 'step', 'eval_episode_reward', 'train_episode_reward', 'J_disc_train',
            'J_disc_eval') if k in row} for row in selected])
    once(path, {'utc': utc(), 'seed': spec['seed'], 'note': 'iteration70 evaluates; preceding train iteration69',
                'comparison': records})
    append(ROOT / 'runs/events.jsonl', {'utc': utc(), 'event': 'interim_itr70_ready',
                                      'run_id': spec['run_id'], 'record': str(path)})


def ledger(entries):
    actual = sum(e.get('allocated_gpu_seconds', 0) for e in entries if e.get('scheduler_state') in TERMINAL)
    reserved = sum(e['cap_seconds'] * e.get('reserved_gpu_count', 1) for e in entries if e.get('scheduler_state') not in TERMINAL)
    return dict(actual_gpu_seconds=actual, outstanding_reserved_gpu_seconds=reserved,
                actual_plus_reserved_gpu_seconds=actual + reserved, budget_gpu_seconds=BUDGET,
                remaining_unreserved_gpu_seconds=BUDGET - actual - reserved)


def capacity(all_jobs, own_active, count):
    if len(own_active) + count > 3:
        return False, 'three-seed allocation cap'
    if len(all_jobs) + count > 12 or sum(j['gpus'] for j in all_jobs) + count > 12:
        return False, 'user job/GPU capacity'
    if sum(a6000_count(j) for j in all_jobs) + count > 10:
        return False, 'A6000 capacity'
    return True, 'available'


class Campaign:
    def __init__(self, args):
        self.args = args
        approval()
        (ROOT / 'runs').mkdir(parents=True, exist_ok=True)
        self.lock = (ROOT / 'runs/campaign.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = read(STATE, dict(created_utc=utc(), status='prepared', submitted=[], external=[]))
        if self.state.get('pending_submission_intent'):
            raise RuntimeError('Ambiguous prior submission; inspect before resuming')
        self.source = validate_gate(args.gate)
        self.started, self.next_monitor, self.stop_requested = time.monotonic(), 0, False
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        external = read(ROOT / 'provenance/external_allocations.json', {'allocations': []})['allocations']
        if sum(e['cap_seconds'] * e.get('reserved_gpu_count', e.get('gpu_count', 0)) for e in external) > 3600:
            raise ValueError('GPU gate allocation exceeds one-hour cap')
        self.state['external'] = external
        prepare_specs()

    def stop(self, signum, frame):
        # Slurm jobs are separate allocations. Never cancel them here.
        self.stop_requested = True

    def record(self):
        entries = self.state['submitted'] + self.state['external']
        intent = self.state.get('pending_submission_intent')
        if intent:
            entries += [dict(cap_seconds=intent['reserved_gpu_seconds'], reserved_gpu_count=1)]
        self.state.update(updated_utc=utc(), ledger=ledger(entries), source_provenance=self.source)
        if self.state['ledger']['actual_plus_reserved_gpu_seconds'] > BUDGET:
            raise ValueError('Actual use plus reservations exceeds16 GPU-hours')
        atomic(STATE, self.state)

    def refresh(self):
        for entry in self.state['external'] + self.state['submitted']:
            if entry.get('scheduler_state') not in TERMINAL:
                entry.update(accounting(entry['job_id']))
            if entry.get('allocated_gpu_seconds', 0) > entry['cap_seconds'] * entry.get('reserved_gpu_count', 1):
                raise RuntimeError('Approved GPU allocation cap exceeded')
            if entry in self.state['external'] or entry.get('terminal_recorded') or entry.get('scheduler_state') not in TERMINAL:
                continue
            spec = read(ROOT / 'runs/specs' / (entry['run_id'] + '.json'))
            path = ROOT / 'runs' / entry['run_id'] / 'manifest.json'
            manifest = read(path, {'status': 'failed', 'failure_reason': 'No worker manifest'})
            good = entry['scheduler_state'] == 'COMPLETED' and entry.get('exit_code') == '0:0' and manifest.get('status') == 'complete'
            if good:
                try:
                    manifest['final_verification'] = verify(spec)
                    derive_baseline_energy(spec)
                except Exception as error:
                    good = False
                    manifest['failure_reason'] = repr(error)
            entry['run_status'] = 'complete' if good else 'failed'
            entry['terminal_recorded'] = True
            manifest.update(status=entry['run_status'], scheduler_observation=entry)
            atomic(path, manifest)
            append(ROOT / 'runs/events.jsonl', {'utc': utc(), 'event': 'run_terminal',
                                              'run_id': entry['run_id'], 'status': entry['run_status']})
        self.record()

    def submit(self, spec):
        if any(e['run_id'] == spec['run_id'] for e in self.state['submitted']):
            raise ValueError('Duplicate attempt')
        validate_gate(self.args.gate)
        once(ROOT / 'runs/specs' / (spec['run_id'] + '.json'), spec)
        argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue',
                '--export=PATH,HOME,USER,LOGNAME,LANG', '--time=240', '--signal=B:TERM@30',
                '--job-name=e1c-' + spec['run_id'], '--chdir=' + str(SOURCE),
                '--output=' + str(ROOT / 'setup' / (spec['run_id'] + '-%j.out')),
                str(ROOT / 'code/run_worker.sh'), '--spec',
                str(ROOT / 'runs/specs' / (spec['run_id'] + '.json')), '--gate', str(self.args.gate)]
        self.state['pending_submission_intent'] = dict(utc=utc(), run_id=spec['run_id'], command=argv,
                                                      reserved_gpu_seconds=spec['cap_seconds'])
        self.record()
        job = command(argv)['stdout'].strip().split(';', 1)[0]
        if not job.isdigit():
            raise RuntimeError('Ambiguous sbatch response')
        self.state['submitted'].append(dict(run_id=spec['run_id'], seed=spec['seed'], attempt=spec['attempt'],
            job_id=job, scheduler_state='PENDING', reserved_gpu_count=1, gpu_count=1,
            cap_seconds=spec['cap_seconds'], submitted_utc=utc(), command=argv))
        del self.state['pending_submission_intent']
        self.record()

    def next_cohort(self):
        missing = [run_spec(seed) for seed in (0, 1, 2) if not any(
            e['seed'] == seed for e in self.state['submitted'])]
        if missing:
            return missing
        fixes = read(ROOT / 'provenance/retry_fixes.json', {'fixes': []})['fixes']
        retries = []
        for seed in (0, 1, 2):
            attempts = [e for e in self.state['submitted'] if e['seed'] == seed]
            if len(attempts) == 1 and attempts[0].get('run_status') == 'failed':
                fix = next((x for x in fixes if x.get('seed') == seed and x.get('resolved') is True
                            and x.get('failed_job_id') == attempts[0]['job_id']), None)
                if fix and fix.get('cause') and fix.get('fix') and fix.get('scientific_configuration_unchanged') is True:
                    retries.append(run_spec(seed, 1))
        return retries

    def monitor(self, reason):
        runs = []
        for entry in self.state['submitted']:
            spec = read(ROOT / 'runs/specs' / (entry['run_id'] + '.json'))
            obs = observe(spec)
            if obs.get('last_itr') is not None and obs['last_itr'] >= 70:
                interim_comparison(spec)
            runs.append(dict(entry, observation=obs))
        record = dict(utc=utc(), reason=reason, status=self.state['status'], ledger=self.state['ledger'],
                      waiting_reason=self.state.get('waiting_reason'), runs=runs)
        atomic(ROOT / 'runs/campaign_progress.json', record)
        append(ROOT / 'runs/campaign_monitor_history.jsonl', record)
        path = ROOT / 'experiment.md'
        text = path.read_text() if path.exists() else '# E1C\n'
        start, end = '<!-- E1C runtime monitor -->', '<!-- end E1C runtime monitor -->'
        text = re.sub(re.escape(start) + '.*?' + re.escape(end), '', text, flags=re.S).rstrip()
        block = ('\n\n%s\nLast monitor: %s. Campaign: **%s**. Seeds complete: %d/3. '
                 'GPU-hours used: %.3f; reserved: %.3f; cap16. '
                 '[Current counters](runs/campaign_progress.json). Next: %s.\n%s\n') % (
                     start, utc(), self.state['status'], len({e['seed'] for e in self.state['submitted']
                     if e.get('run_status') == 'complete'}), self.state['ledger']['actual_gpu_seconds']/3600,
                     self.state['ledger']['outstanding_reserved_gpu_seconds']/3600,
                     self.state.get('waiting_reason', 'finish admitted runs'), end)
        path.write_text(text + block)

    def final_analysis(self, partial):
        output = ROOT / 'runs/final_analysis'
        program = ROOT / 'code/analyze.py'
        if not program.is_file():
            self.state['analysis'] = dict(status='blocked', reason='Analysis program unavailable')
            self.record()
            return False
        previous = self.state.get('analysis', {})
        if previous.get('status') in ('complete', 'partial'):
            return True
        if output.exists() and not previous:
            self.state['analysis'] = dict(status='blocked', reason='Existing analysis evidence requires inspection')
            self.record()
            return False
        argv = [str(PYTHON), '-B', str(program), '--output', str(output), '--publish-report']
        if partial:
            argv.append('--partial')
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        self.state['analysis'] = dict(status='running', command=argv, started_utc=utc(),
                                     cap_seconds=600, gpu_count=0, cpus=1)
        self.record()
        path = ROOT / 'setup/final_analysis.out'
        try:
            with path.open('x') as stream:
                result = subprocess.run(argv, cwd=str(PROJECT), env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=600)
            record = read(output / 'analysis_status.json', {})
            if result.returncode or record.get('status') not in ('complete', 'partial'):
                raise RuntimeError('Analysis did not finish successfully; preserve console and outputs')
            self.state['analysis'].update(status=record['status'], ended_utc=utc(),
                                           output=str(output), subprocess_exitcode=result.returncode)
            self.record()
            return True
        except Exception as error:
            self.state['analysis'].update(status='failed', reason=repr(error), ended_utc=utc())
            self.record()
            return False

    def run(self):
        self.state.update(status='running', monitor_status='active', controller_pid=os.getpid(),
                          controller_job_id=os.environ.get('SLURM_JOB_ID'))
        try:
            while True:
                if self.stop_requested or time.monotonic() - self.started >= self.args.controller_cap_seconds:
                    raise RuntimeError('Controller stopped; existing workers retain individual caps and monitors')
                self.refresh()
                active = {e['job_id'] for e in self.state['submitted'] if e.get('scheduler_state') not in TERMINAL}
                cohort = self.next_cohort()
                if cohort:
                    allowed, reason = capacity(jobs(), active, len(cohort))
                    if self.state['ledger']['remaining_unreserved_gpu_seconds'] < sum(s['cap_seconds'] for s in cohort):
                        allowed, reason = False, 'remaining budget cannot reserve another whole attempt'
                        if not active:
                            self.state.update(status='budget_limited', monitor_status='paused', waiting_reason=reason)
                            self.record(); self.monitor('budget limit'); return 2
                    if allowed:
                        for spec in cohort:
                            self.submit(spec)
                        self.state['waiting_reason'] = 'finish three admitted E1C seeds; G may use remaining capacity'
                        self.monitor('cohort admitted')
                        self.next_monitor = time.monotonic() + 1800
                    else:
                        self.state['waiting_reason'] = reason
                elif not active:
                    complete = len({e['seed'] for e in self.state['submitted'] if e.get('run_status') == 'complete'})
                    self.state.update(status='training_complete' if complete == 3 else 'failed_needs_cause_fix',
                        monitor_status='paused', waiting_reason='analyze saved E1C/E0/E1 results' if complete == 3 else
                        'record cause and verified fix before any same-seed retry; preserve failed attempt')
                    self.record(); self.monitor('training terminal')
                    self.state.update(status='analyzing', monitor_status='active')
                    self.record()
                    good = self.final_analysis(partial=complete != 3)
                    self.state.update(status=('completed' if complete == 3 else 'partial_needs_cause_fix') if good else 'analysis_failed',
                                      monitor_status='paused', waiting_reason='review candidate analysis; curation needs selection' if good else 'inspect analysis failure')
                    self.record(); self.monitor('analysis terminal')
                    return 0 if good and complete == 3 else 1
                self.record()
                if time.monotonic() >= self.next_monitor:
                    self.monitor('scheduled monitor')
                    self.next_monitor = time.monotonic() + 1800
                time.sleep(30)
        except Exception as error:
            self.state.update(status='blocked', monitor_status='paused', failure_reason=repr(error),
                              waiting_reason='inspect controller blocker; running jobs remain unchanged')
            self.record(); self.monitor('controller blocked')
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'status'))
    parser.add_argument('--gate', default=str(GATE))
    parser.add_argument('--spec')
    parser.add_argument('--controller-cap-seconds', type=int, default=43200)
    args = parser.parse_args()
    if not 0 < args.controller_cap_seconds <= 43200:
        parser.error('Controller cap is12 hours; no run wall caps change')
    if args.action == 'prepare':
        print(json.dumps(prepare_specs(), indent=2)); return 0
    if args.action == 'status':
        print(json.dumps(read(ROOT / 'runs/campaign_progress.json', {}), indent=2)); return 0
    if args.action == 'worker':
        return worker(args)
    return Campaign(args).run()


if __name__ == '__main__':
    sys.exit(main())
