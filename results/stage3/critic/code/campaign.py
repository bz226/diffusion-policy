"""Bounded Addendum C queue. G has priority; failed C runs are never retried.

Python 3.8 only. The controller can wait without a Slurm allocation, then admit
three distinct one-GPU seeds together. Every job owns its process group, output,
wall limit and monitor; stopping the controller never stops another campaign.
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
import subprocess
import sys
import time

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/critic'
PARENT = PROJECT / '.codex-runtime/worktrees/stage3-critic'
SOURCE = PARENT / 'results/stage3_critic/dppo'
G_ROOT = PROJECT / 'results/stage3/gradmean'
PYTHON = PROJECT / 'results/repro_halfcheetah/env/bin/python'
FROZEN = G_ROOT / 'prepare/P0/frozen_P0.pt'
GATE = ROOT / 'runs/gate/gate_summary.json'
STATE = ROOT / 'runs/campaign.json'
KLSTAR = 0.000584517336086918
ETA = 0.00012107235674859004
BUDGET = 60 * 3600
CONDITIONS = ('SGDC3', 'SGDC10', 'SGDC3w', 'SGDG3', 'SGDG10')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
COMMON = ['model.clip_ploss_coef=1e6', 'model.clip_ploss_coef_base=1e6',
          'train.target_kl=null', '+model.clamp_logprob=false',
          'model.randn_clip_value=100', '+train.actor_single_step=true',
          '+model.actor_loss=score', '+train.stage3_diag=true']
CRITIC = ['+train.adv_estimator=mc_critic', '+train.decision_discount=true',
          'model.gamma_denoising=1.0', '+model.logprob_reduce=sum', '+model.norm_adv=false']


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


def approval():
    item = read(ROOT / 'provenance/approval.json', {})
    required = dict(approved=True, gpu_hour_cap=60, max_parallel_training_jobs=3,
                    training_wall_limit_seconds=14400, combined_gate_wall_limit_seconds=7200,
                    calibration_wall_limit_seconds=3600, checkpoint_eval_wall_limit_seconds=900,
                    cpus_per_gpu_job=40, memory_gib_per_gpu_job=64, retries_authorized=False)
    for key, value in required.items():
        if item.get(key) != value:
            raise ValueError('Missing or different approval: ' + key)
    if item.get('conditions') != list(CONDITIONS) or item.get('seeds') != [0, 1, 2]:
        raise ValueError('Condition or seed approval mismatch')
    return item


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


def validate_gate(path):
    gate = read(path, {})
    if gate.get('passed') is not True:
        raise ValueError('All C1-C5 checks must pass before production')
    for key in ('C1', 'C2', 'C3', 'C4', 'C5'):
        value = gate.get(key, gate.get('checks', {}).get(key))
        if not isinstance(value, dict) or value.get('passed') is not True:
            raise ValueError('Missing passed gate check ' + key)
    commit = gate.get('parent_commit')
    if not commit or not str(gate.get('slurm_job_id', '')).isdigit():
        raise ValueError('Gate must identify its source revision and GPU allocation')
    status = accounting(gate['slurm_job_id'])
    if status.get('scheduler_state') != 'COMPLETED' or status.get('exit_code') != '0:0':
        raise ValueError('GPU gate allocation must complete with exit0:0')
    if command(['git', '-C', PARENT, 'branch', '--show-current'])['stdout'].strip() != 'stage3-critic':
        raise ValueError('C parent worktree must remain on stage3-critic')
    current = command(['git', '-C', PARENT, 'rev-parse', 'HEAD'])['stdout'].strip()
    command(['git', '-C', PARENT, 'merge-base', '--is-ancestor', commit, current])
    snapshot = Path(gate['source_snapshot']).resolve()
    if snapshot == SOURCE.resolve() or ROOT not in snapshot.parents:
        raise ValueError('Gate needs its separate immutable ordinary source snapshot')
    actual, expected = source_files(SOURCE), source_files(snapshot)
    if not expected or actual.keys() != expected.keys():
        raise ValueError('Scientific source file set changed after gate')
    for name, previous in expected.items():
        if actual[name].read_bytes() != previous.read_bytes() or ((actual[name].stat().st_mode ^ previous.stat().st_mode) & 0o111):
            raise ValueError('Scientific source changed after gate: ' + name)
    return dict(gate=str(path), parent_commit=commit, current_parent_commit=current,
                source_snapshot=str(snapshot), files_compared=len(expected), gate_allocation=status)


def calibration_eta():
    spec = run_spec('calG', 3001)
    manifest = read(ROOT / 'runs/calG_seed3001/manifest.json', {})
    if manifest.get('status') != 'complete':
        raise ValueError('SGDG requires completed, verified calG')
    result = read(Path(spec['logdir']) / 'calibration.json', {})
    if result.get('status') != 'completed' or len(result.get('batches', [])) != 5 or result.get('kl_target') != KLSTAR:
        raise ValueError('calG did not complete the approved five-batch protocol')
    eta = float(result['eta_star'])
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('Invalid calibrated SGDG learning rate')
    return eta


def run_spec(condition, seed, eta_g=None):
    if condition == 'calG':
        if seed != 3001:
            raise ValueError('calG seed must be3001')
        mode, cap, estimate = 'calibrate', 3600, 120
        extra = ['+train.mode=calibrate', '+train.calib_kl_target=' + repr(KLSTAR),
                 '+train.critic_init_path=' + str(FROZEN)]
    elif condition in CONDITIONS:
        if seed not in (0, 1, 2):
            raise ValueError('Training seeds must be0,1,2')
        mode, cap, estimate = 'train', 14400, 9000
        if condition.startswith('SGDC'):
            eta = ETA * (10 if condition == 'SGDC10' else 3)
            extra = list(CRITIC)
        else:
            if eta_g is None:
                eta_g = calibration_eta()
            eta = eta_g * (10 if condition == 'SGDG10' else 3)
            extra = []
        extra += ['+train.actor_optimizer=sgd', 'train.actor_lr=' + repr(eta),
                  'train.actor_lr_scheduler.min_lr=' + repr(eta), '+train.baseline_diag=true']
        if condition == 'SGDC3w':
            extra += ['+train.critic_init_path=' + str(FROZEN)]
    elif condition.startswith('ckpt_'):
        if seed != 5000:
            raise ValueError('Checkpoint evaluation seed must be5000')
        native = condition[len('ckpt_'):]
        match = re.fullmatch(r'(SGDC3|SGDC10|SGDC3w|SGDG3|SGDG10)_seed([012])', native)
        if not match:
            raise ValueError('Unknown checkpoint source')
        mode, cap, estimate = 'ckpt_eval', 900, 60
        extra = ['+train.mode=ckpt_eval', 'base_policy_path=' + str(ROOT / native / 'checkpoint/state_139.pt')]
    else:
        raise ValueError('Unapproved condition ' + condition)
    name = condition + '_seed' + str(seed)
    overrides = ['train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
                 'seed=' + str(seed), 'logdir=' + str(ROOT / name)] + COMMON + extra
    return dict(run_id=name, condition=condition, seed=seed, mode=mode,
                cap_seconds=cap, estimate_seconds=estimate, monitor_interval_seconds=1800,
                gpu_count=1, gpu_type='a6000', cpus=40, memory_gib=64,
                logdir=str(ROOT / name), console_log=str(ROOT / (name + '.out')),
                overrides=overrides, command=[str(PYTHON), '-B', str(ROOT / 'code/entrypoint.py'),
                    '--config-dir=' + str(SOURCE / 'cfg/gym/finetune/halfcheetah-v2'),
                    '--config-name=ft_ppo_diffusion_mlp'] + overrides)


def write_specs(specs, filename):
    groups = {}
    for spec in specs:
        once(ROOT / 'runs/specs' / (spec['run_id'] + '.json'), spec)
        overrides = [x for x in spec['overrides'] if not x.startswith(('seed=', 'logdir='))]
        groups.setdefault((spec['condition'], tuple(overrides)), []).append(str(spec['seed']))
    lines = ['# NAME SEEDS OVERRIDES. All overrides are explicit; per-run JSON adds seed and absolute logdir.']
    lines += [name + ' ' + ','.join(seeds) + ' ' + ' '.join(overrides)
              for (name, overrides), seeds in groups.items()]
    once(ROOT / filename, '\n'.join(lines) + '\n')


def prepare_specs():
    approval()
    if not FROZEN.is_file():
        raise ValueError('Existing approved frozen P0 critic is unavailable; do not duplicate prepare')
    old = read(PROJECT / 'results/stage3/cal_seed3000/calibration.json')
    if old['eta_star'] != ETA or old['kl_target'] != KLSTAR:
        raise ValueError('Stage3 calibration anchors changed')
    write_specs([run_spec('calG', 3001)], 'jobs_calG.txt')
    write_specs([run_spec(c, s) for c in CONDITIONS[:3] for s in (0, 1, 2)], 'jobs_SGDC.txt')
    try:
        eta_g = calibration_eta()
    except ValueError:
        eta_g = None
    if eta_g is not None:
        write_specs([run_spec(c, s, eta_g) for c in CONDITIONS[3:] for s in (0, 1, 2)], 'jobs_SGDG.txt')
    return dict(status='prepared', eta_star_E2=ETA, KL_star=KLSTAR, eta_star_G=eta_g,
                unresolved_job_files=[] if eta_g is not None else ['jobs_SGDG.txt'],
                explanation='SGDG job file is written only after verified calG; no executable placeholders.')


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
            first = next((r.get('eval_episode_reward') for r in rows if r['itr'] == 0), None)
            result['collapse_eval_iterations'] = [r['itr'] for r in rows if first is not None
                                                 and r.get('eval_episode_reward', first) < 0.5 * first]
        except (EOFError, pickle.UnpicklingError):
            result['result_read_in_progress'] = True
    elif spec['mode'] != 'train':
        filename = 'calibration.json' if spec['mode'] == 'calibrate' else 'ckpt_eval.json'
        try:
            native = read(directory / filename, {})
        except json.JSONDecodeError:
            native = dict(write_in_progress=True)
        result.update(native_status=native.get('status'), nonfinite_paths=nonfinite(native))
    console = Path(spec['console_log'])
    if console.is_file():
        with console.open('rb') as stream:
            stream.seek(max(0, console.stat().st_size - 32768))
            result['console_tail'] = stream.read().decode('utf-8', errors='replace').splitlines()[-25:]
    return finite_json(result)


def collapsed(observation):
    return bool(observation.get('collapse_eval_iterations') or observation.get('nonfinite_paths') or
                str(observation.get('native_failure', {}).get('reason', '')).startswith('nonfinite'))


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
        raise FloatingPointError('Non-finite run measurements: ' + str(observation['nonfinite_paths']))
    if spec['mode'] == 'train':
        if observation.get('rows') != 140 or observation.get('last_itr') != 139:
            raise ValueError('Run did not finish all140 iterations')
        with (directory / 'result.pkl').open('rb') as stream:
            rows = pickle.load(stream)
        if [r['itr'] for r in rows] != list(range(140)) or rows[-1]['step'] != 10080000:
            raise ValueError('Training iteration/step schedule changed')
        for itr in (0, 35, 70, 105, 139):
            if not (directory / ('checkpoint/state_%d.pt' % itr)).is_file():
                raise ValueError('Missing requested checkpoint ' + str(itr))
        if any(r.get('actor_optimizer_steps') != 1 for r in rows if 'train_episode_reward' in r):
            raise ValueError('Expected exactly one actor step per training iteration')
    else:
        filename = 'calibration.json' if spec['mode'] == 'calibrate' else 'ckpt_eval.json'
        native = read(directory / filename, {})
        if native.get('status') != 'completed' or native.get('seed') != spec['seed']:
            raise ValueError('Native mode did not complete')
        if spec['mode'] == 'calibrate':
            if len(native.get('batches', [])) != 5 or native.get('kl_target') != KLSTAR:
                raise ValueError('Calibration sample size or KL target differs')
        else:
            if native.get('repeats_per_sampler') != 1:
                raise ValueError('Exactly one rollout per final-checkpoint sampler required')
            for sampler in ('train', 'eval'):
                if native.get('samplers', {}).get(sampler, {}).get('J_disc', {}).get('n') != 80:
                    raise ValueError('Exactly80 episodes required for each sampler')
    return dict(verified_utc=utc(), cleanup=str(directory / 'cleanup.json'), observation=observation)


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
    if spec != run_spec(spec['condition'], spec['seed']):
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
        inventory = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])['stdout']
        if not inventory.strip() or any('A6000' not in line for line in inventory.splitlines()):
            raise ValueError('Approved A6000 allocation required')
        manifest['hardware'] = dict(gpu_inventory=inventory, cpu_affinity_count=len(os.sched_getaffinity(0)))
        if len(os.sched_getaffinity(0)) < 40:
            raise ValueError('Approved40 CPU allocation required')
        atomic(path, manifest)
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
                time.sleep(10)
        manifest.update(subprocess_exitcode=child.returncode, subprocess_pid=child.pid)
        if child.returncode:
            raise RuntimeError('Native process exited%d' % child.returncode)
        if group_members(child.pid):
            raise RuntimeError('Native process left child processes behind')
        validate_gate(args.gate)
        manifest.update(status='complete', verification=verify(spec), remaining_process_group_members=[])
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


def ledger(entries):
    actual = sum(e.get('allocated_gpu_seconds', 0) for e in entries if e.get('scheduler_state') in TERMINAL)
    reserved = sum(e['cap_seconds'] * e.get('reserved_gpu_count', 1) for e in entries if e.get('scheduler_state') not in TERMINAL)
    return dict(actual_gpu_seconds=actual, outstanding_reserved_gpu_seconds=reserved,
                actual_plus_reserved_gpu_seconds=actual + reserved,
                budget_gpu_seconds=BUDGET, remaining_unreserved_gpu_seconds=BUDGET - actual - reserved)


def gradient_mean_active(all_jobs):
    state = read(G_ROOT / 'runs/campaign.json', {})
    bootstrap = read(G_ROOT / 'runs/bootstrap.json', {})
    ids = {str(e['job_id']) for e in state.get('submitted', [])}
    ids.update(str(e.get('job_id')) for e in state.get('analyses', []))
    ids.update(str(x) for x in (state.get('controller_job_id'), bootstrap.get('slurm_job_id')) if x)
    active = [j for j in all_jobs if j['job_id'] in ids or j['name'].startswith('gm-')]
    # No new C training is admitted on a momentary round-barrier vacancy.
    return bool(active or state.get('status') in ('running', 'prepared', 'waiting', None))


def capacity(all_jobs, own_active, count, g_active):
    if len(own_active) + count > 3:
        return False, 'C total allocation cap3'
    if len(all_jobs) + count > 12 or sum(j['gpus'] for j in all_jobs) + count > 12:
        return False, 'Preserve user job/GPU headroom'
    a6000 = sum(a6000_count(j) for j in all_jobs)
    if a6000 + count > 10:
        return False, 'No extra A6000 capacity'
    if g_active:
        # Only calG can enter this path. Reserve9 G allocations plus all
        # unrelated users' jobs/controllers; never exploit a round vacancy.
        others = [j for j in all_jobs if not j['name'].startswith('gm-') and j['job_id'] not in own_active]
        g_control = sum(j['gpus'] == 0 for j in all_jobs if j['name'].startswith('gm-'))
        if 9 + g_control + len(others) + len(own_active) + count > 12:
            return False, 'Preserve nine G jobs and their controller headroom'
        other_gpu = sum(j['gpus'] for j in others)
        other_a6000 = sum(a6000_count(j) for j in others)
        if 9 + other_gpu + len(own_active) + count > 12 or 9 + other_a6000 + len(own_active) + count > 10:
            return False, 'Preserve nine G GPU slots'
    return True, 'available'


def persistent_controller_conflict(g_active, inherited_job_id, scheduler_jobs):
    """Allow only the explicitly scheduled CPU controller while G is active.

    The controller replaces the expiring interactive allocation. Its live
    scheduler entry remains included in every unchanged capacity calculation.
    An inherited shell job ID is not evidence of this controller allocation.
    """
    if not g_active or not inherited_job_id:
        return False
    matching = [job for job in scheduler_jobs if job['job_id'] == str(inherited_job_id)]
    return not (len(matching) == 1 and matching[0]['name'].strip() == 'critic-controller'
                and matching[0]['gpus'] == 0 and matching[0].get('partition') == 'soal'
                and matching[0].get('state') == 'RUNNING')


def replace_paragraph(text, label, content):
    """Keep the study design intact while replacing a stale status paragraph."""
    pattern = r'(?ms)^' + re.escape('**' + label + '.**') + r'.*?(?=\n\s*\n|\Z)'
    replacement = '**' + label + '.** ' + content
    if re.search(pattern, text):
        return re.sub(pattern, lambda match: replacement, text, count=1)
    return text.rstrip() + '\n\n' + replacement


class Campaign:
    def __init__(self, args):
        self.args = args
        approval()
        self.lock = (ROOT / 'runs/campaign.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = read(STATE, dict(created_utc=utc(), status='prepared', submitted=[], external=[]))
        if self.state.get('pending_submission_intent'):
            raise RuntimeError('Ambiguous prior submission; inspect recorded intent before resuming')
        self.source = validate_gate(args.gate)
        self.started, self.next_monitor, self.stop_requested = time.monotonic(), 0, False
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.external()
        prepare_specs()

    def stop(self, signum, frame):
        self.stop_requested = True

    def external(self):
        record = read(ROOT / 'provenance/external_allocations.json', {})
        records = record.get('allocations', []) if isinstance(record, dict) else record
        if not records or sum(e['cap_seconds'] for e in records) > 7200:
            raise ValueError('Record all gate allocations within the combined two-hour gate cap')
        if len({str(e['job_id']) for e in records}) != len(records):
            raise ValueError('Duplicate gate allocation')
        if any(e.get('gpu_count') != 1 or not 0 < e['cap_seconds'] <= 7200 for e in records):
            raise ValueError('Gate entries must be bounded one-GPU allocations')
        if str(read(self.args.gate)['slurm_job_id']) not in {str(e['job_id']) for e in records}:
            raise ValueError('Passed GPU gate must be charged to the60 GPU-hour budget')
        old = {str(e['job_id']): e for e in self.state['external']}
        if not set(old).issubset({str(e['job_id']) for e in records}):
            raise ValueError('Previously charged allocations cannot disappear')
        self.state['external'] = [dict(old.get(str(e['job_id']), {}), reserved_gpu_count=1, **e) for e in records]

    def refresh(self):
        for entry in self.state['external'] + self.state['submitted']:
            if entry.get('scheduler_state') not in TERMINAL:
                status = accounting(entry['job_id'])
                entry.update(status)
            if entry.get('allocated_gpu_seconds', 0) > entry['cap_seconds']:
                raise RuntimeError('An allocation exceeded its approved reservation')
            if entry in self.state['external'] or entry.get('terminal_recorded') or entry.get('scheduler_state') not in TERMINAL:
                continue
            spec = read(ROOT / 'runs/specs' / (entry['run_id'] + '.json'))
            path = ROOT / 'runs' / entry['run_id'] / 'manifest.json'
            manifest = read(path, dict(run_id=entry['run_id'], status='failed', failure_reason='No worker manifest'))
            if entry['scheduler_state'] == 'COMPLETED' and entry.get('exit_code') == '0:0' and manifest.get('status') == 'complete':
                try:
                    manifest['final_verification'] = verify(spec)
                    entry['run_status'] = 'complete'
                except Exception as error:
                    entry['run_status'] = 'failed'
                    manifest.update(status='failed', failure_reason=str(error))
            else:
                entry['run_status'] = 'failed'
                manifest.update(status='failed', scheduler_failure=entry['scheduler_state'])
            entry['terminal_recorded'] = True
            manifest['scheduler_observation'] = {k: entry.get(k) for k in ('job_id', 'scheduler_state', 'exit_code', 'elapsed_seconds', 'allocated_gpu_seconds')}
            atomic(path, manifest)
            append(ROOT / 'runs/events.jsonl', dict(utc=utc(), event='run_terminal', run_id=entry['run_id'], status=entry['run_status']))
            if entry['run_id'] == 'calG_seed3001' and entry['run_status'] == 'complete':
                eta_g = calibration_eta()
                write_specs([run_spec(c, s, eta_g) for c in CONDITIONS[3:] for s in (0, 1, 2)], 'jobs_SGDG.txt')
        self.record()

    def record(self):
        allocations = self.state['external'] + self.state['submitted']
        if self.state.get('pending_submission_intent'):
            allocations = allocations + [dict(cap_seconds=self.state['pending_submission_intent']['reserved_gpu_seconds'], reserved_gpu_count=1)]
        self.state.update(updated_utc=utc(), ledger=ledger(allocations),
                          source_provenance=self.source)
        if (self.state['ledger']['actual_plus_reserved_gpu_seconds'] > BUDGET and
                self.state.get('status') not in ('blocked', 'budget_limited', 'analysis_failed')):
            raise RuntimeError('Actual plus outstanding reservations exceeded60 GPU-hours')
        atomic(STATE, self.state)

    def submit(self, spec):
        if any(e['run_id'] == spec['run_id'] for e in self.state['submitted']):
            raise ValueError('No repeated submissions or retries')
        validate_gate(self.args.gate)
        once(ROOT / 'runs/specs' / (spec['run_id'] + '.json'), spec)
        argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue', '--nice=1000',
                '--export=PATH,HOME,USER,LOGNAME,LANG', '--time=' + str(spec['cap_seconds'] // 60),
                '--signal=B:TERM@30', '--job-name=crit-' + spec['run_id'], '--chdir=' + str(SOURCE),
                '--output=' + str(ROOT / 'setup' / (spec['run_id'] + '-%j.out')),
                str(ROOT / 'code/run_worker.sh'), '--spec',
                str(ROOT / 'runs/specs' / (spec['run_id'] + '.json')), '--gate', str(self.args.gate)]
        self.state['pending_submission_intent'] = dict(utc=utc(), run_id=spec['run_id'], command=argv,
                                                      reserved_gpu_seconds=spec['cap_seconds'])
        self.record()
        output = command(argv)['stdout'].strip()
        job = output.split(';', 1)[0]
        if not job.isdigit():
            raise RuntimeError('Ambiguous sbatch output: ' + output)
        self.state['submitted'].append(dict(run_id=spec['run_id'], job_id=job, scheduler_state='PENDING',
                                            gpu_count=1, reserved_gpu_count=1, cap_seconds=spec['cap_seconds'], submitted_utc=utc(), command=argv))
        del self.state['pending_submission_intent']
        self.record()

    def next_cohort(self):
        submitted = {e['run_id']: e for e in self.state['submitted']}
        cal = submitted.get('calG_seed3001')
        if cal is None:
            return [run_spec('calG', 3001)]
        cal_failed = cal.get('scheduler_state') in TERMINAL and cal.get('run_status') != 'complete'
        for condition in CONDITIONS:
            if condition.startswith('SGDG') and cal_failed:
                self.state.setdefault('blocked_conditions', {})[condition] = 'calG failed; eta_G unavailable; no retry or substitute'
                continue
            names = [condition + '_seed%d' % seed for seed in (0, 1, 2)]
            present = [name in submitted for name in names]
            if any(present) and not all(present):
                raise RuntimeError('Partial cohort submission; inspect existing allocations before continuing')
            if all(present):
                if any(submitted[n].get('scheduler_state') not in TERMINAL for n in names):
                    return []
                continue
            if condition.startswith('SGDG') and cal.get('run_status') != 'complete':
                return []
            return [run_spec(condition, seed) for seed in (0, 1, 2)]
        evaluations = []
        for condition in CONDITIONS:
            for seed in (0, 1, 2):
                native = '%s_seed%d' % (condition, seed)
                if submitted.get(native, {}).get('run_status') != 'complete':
                    continue
                spec = run_spec('ckpt_' + native, 5000)
                if spec['run_id'] not in submitted:
                    evaluations.append(spec)
        if evaluations:
            write_specs(evaluations, 'jobs_ckpt_eval_remaining_' + str(len(evaluations)) + '.txt')
        return evaluations[:3]

    def monitor(self, reason):
        record = dict(utc=utc(), reason=reason, status=self.state['status'], ledger=self.state['ledger'],
                      waiting_reason=self.state.get('waiting_reason'), runs=[])
        for entry in self.state['submitted']:
            spec = read(ROOT / 'runs/specs' / (entry['run_id'] + '.json'))
            record['runs'].append(dict(run_id=entry['run_id'], job_id=entry['job_id'],
                                       mode=spec['mode'], condition=spec['condition'],
                                       scheduler_state=entry.get('scheduler_state'), run_status=entry.get('run_status'),
                                       observation=observe(spec)))
        atomic(ROOT / 'runs/campaign_progress.json', record)
        append(ROOT / 'runs/campaign_monitor_history.jsonl', record)
        path = ROOT / 'experiment.md'
        text = path.read_text() if path.exists() else '# Critic-baseline study\n'
        marker = '<!-- C runtime monitor -->'
        text = text.split(marker)[0].rstrip()
        completed = sum(r.get('run_status') == 'complete' for r in record['runs'])
        failed = sum(r.get('run_status') == 'failed' for r in record['runs'])
        training_complete = sum(r['mode'] == 'train' and r.get('run_status') == 'complete' for r in record['runs'])
        training_failed = sum(r['mode'] == 'train' and r.get('run_status') == 'failed' for r in record['runs'])
        evaluations_complete = sum(r['mode'] == 'ckpt_eval' and r.get('run_status') == 'complete' for r in record['runs'])
        active_count = sum(r.get('scheduler_state') not in TERMINAL for r in record['runs'])
        status = ('C1–C5 passed. Campaign state: **%s**. Training: %d/15 complete, %d failed; '
                  'final-checkpoint evaluations: %d/15 complete. %d C GPU allocations are pending or running. '
                  'Current source and exact run records are linked from [campaign progress](runs/campaign_progress.json).') % (
                      self.state['status'], training_complete, training_failed, evaluations_complete, active_count)
        if self.state.get('blocked_conditions'):
            status += ' ' + ', '.join(sorted(self.state['blocked_conditions'])) + ' remain unattempted because calG failed; no replacement learning rate is used.'
        text = replace_paragraph(text, 'Status', status)
        text = replace_paragraph(text, 'Next step', self.state.get('waiting_reason', 'Complete admitted runs') +
                                 '. [Results and requested analysis](REPORT.md); curation requires a separate decision.')
        block = '\n\n' + marker + '\nLast monitor: %s. Status: **%s**. Completed allocations: %d; failed: %d. ' % (utc(), self.state['status'], completed, failed)
        block += 'GPU-hours used: %.3f; reserved: %.3f; hard cap:60. Next: %s.\n' % (
            self.state['ledger']['actual_gpu_seconds'] / 3600,
            self.state['ledger']['outstanding_reserved_gpu_seconds'] / 3600,
            self.state.get('waiting_reason', 'complete admitted runs'))
        path.write_text(text + block)
        if self.state['status'] in ('blocked', 'budget_limited', 'analysis_failed'):
            self.blocked_report(record)

    def blocked_report(self, observation):
        """Make the blocker visible, preserving any previously published report."""
        reason = (self.state.get('failure_reason') or self.state.get('final_analysis', {}).get('error') or
                  self.state.get('waiting_reason') or self.state['status'])
        active = [entry['run_id'] for entry in self.state['submitted'] if entry.get('scheduler_state') not in TERMINAL]
        analysis = self.state.get('final_analysis', {})
        if analysis.get('job_id') and analysis.get('scheduler_observation', {}).get('scheduler_state') not in TERMINAL:
            active.append('final CPU analysis ' + str(analysis['job_id']))
        identity = dict(status=self.state['status'], reason=reason, active_jobs=active)
        if self.state.get('blocked_report', {}).get('identity') == identity:
            return
        path = ROOT / 'REPORT.md'
        old = path.read_text() if path.exists() else ''
        now = utc()
        record_dir = ROOT / 'runs/controller_reports' / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        record_dir.mkdir(parents=True)
        if old:
            (record_dir / 'previous_REPORT.md').write_text(old)
        atomic(record_dir / 'status.json', dict(identity, utc=now, ledger=self.state['ledger'], observation=observation))
        count_ok = sum(e.get('run_status') == 'complete' for e in self.state['submitted'])
        count_bad = sum(e.get('run_status') == 'failed' for e in self.state['submitted'])
        message = ('**Current execution status: %s.** %s\n\n'
                   'C1–C5 passed before production. Verified complete allocations: %d; failed allocations: %d. '
                   'Actual GPU use is %.3f hours, with %.3f hours still reserved against the60 GPU-hour cap. '
                   'No retry, replacement learning rate, or new admission is authorized by this status report.\n\n') % (
                       self.state['status'], reason, count_ok, count_bad,
                       self.state['ledger']['actual_gpu_seconds'] / 3600,
                       self.state['ledger']['outstanding_reserved_gpu_seconds'] / 3600)
        if active:
            message += ('Already admitted jobs retain their individual caps and worker monitors: ' + ', '.join(active) +
                        '. Their outputs remain provisional; the existing report below is preserved.\n\n')
            start, end = '<!-- C controller status -->', '<!-- end C controller status -->'
            old = re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\s*', '', old, flags=re.S)
            text = start + '\n' + message + end + '\n\n' + old
        else:
            text = '# Addendum C: execution blocked\n\n' + message
            text += 'No C allocation remains active. Scientific conclusions and Q1–Q5 remain incomplete wherever the required runs or analysis are unavailable. '
            if old:
                text += '[The prior report is preserved](' + str((record_dir / 'previous_REPORT.md').relative_to(ROOT)) + '). '
            text += ('[Gate evidence](runs/gate/gate_summary.json), [saved campaign state](runs/campaign.json), '
                     '[current progress](runs/campaign_progress.json), and [methods](methods.md) retain the evidence and approved scope.\n')
        path.write_text(text)
        self.state['blocked_report'] = dict(identity=identity, record=str(record_dir), utc=now)
        self.record()

    def final_analysis(self, partial=False):
        """CPU Slurm report job; terminal status and output both verified."""
        program = ROOT / 'code/analyze_critic.py'
        output = ROOT / 'runs/final_analysis'
        previous = self.state.get('final_analysis')
        if previous and previous.get('status') in ('complete', 'failed', 'blocked'):
            return previous['status'] == 'complete'
        if not program.is_file():
            self.state['final_analysis'] = dict(status='blocked', reason='Approved analysis program is not ready')
            self.record()
            return False
        if output.exists() and not previous:
            self.state['final_analysis'] = dict(status='blocked', reason='Existing final-analysis directory; preserve it for review')
            self.record()
            return False
        self.state.setdefault('final_analysis', dict(status='prepared'))
        try:
            if not previous:
                if gradient_mean_active(jobs()):
                    raise RuntimeError('Final C analysis must not reduce active G job headroom')
                argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal',
                        '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G',
                        '--time=30', '--no-requeue', '--nice=1000', '--job-name=crit-analysis',
                        '--export=PATH,HOME,USER,LOGNAME,LANG',
                        '--output=' + str(ROOT / 'setup/final-analysis-%j.out'),
                        str(ROOT / 'code/run_analysis.sh'), '--output', str(output), '--publish-report']
                if partial:
                    argv.append('--partial')
                self.state['final_analysis'] = dict(status='submission_intent', command=argv, started_utc=utc(),
                                                    gpu_count=0, cpus=4, memory_gib=16, cap_seconds=1800)
                self.record()
                job = command(argv)['stdout'].strip().split(';', 1)[0]
                if not job.isdigit():
                    raise RuntimeError('Ambiguous analysis sbatch output')
                self.state['final_analysis'].update(status='submitted', job_id=job)
                self.record()
            elif not previous.get('job_id'):
                raise RuntimeError('Ambiguous prior analysis submission; inspect intent before resuming')
            job = self.state['final_analysis']['job_id']
            next_monitor = 0
            while True:
                if self.stop_requested or time.monotonic() - self.started >= self.args.controller_cap_seconds:
                    raise RuntimeError('Controller capped while analysis job retains its own30-minute cap')
                status = accounting(job)
                self.state['final_analysis']['scheduler_observation'] = status
                if status.get('scheduler_state') in TERMINAL:
                    record = read(output / 'analysis_record.json', {})
                    good = (status['scheduler_state'] == 'COMPLETED' and status.get('exit_code') == '0:0'
                            and record.get('status') == 'complete' and (output / 'candidates/REPORT.md').is_file())
                    self.state['final_analysis'].update(status='complete' if good else 'failed', ended_utc=utc())
                    break
                self.record()
                if time.monotonic() >= next_monitor:
                    self.monitor('final CPU analysis pending/running')
                    next_monitor = time.monotonic() + 1800
                time.sleep(30)
        except Exception as error:
            self.state['final_analysis'].update(status='failed', error=type(error).__name__ + ': ' + str(error), ended_utc=utc())
        self.record()
        self.monitor('final CPU analysis terminal')
        return self.state['final_analysis']['status'] == 'complete'

    def run(self, submit_cal_only=False):
        self.state.update(status='running', monitor_status='active', controller_pid=os.getpid(),
                          controller_job_id=os.environ.get('SLURM_JOB_ID'))
        try:
            while True:
                if self.stop_requested or time.monotonic() - self.started >= self.args.controller_cap_seconds:
                    raise RuntimeError('Controller stopped/capped; existing workers retain their own caps')
                self.refresh()
                if submit_cal_only and any(e['run_id'] == 'calG_seed3001' for e in self.state['submitted']):
                    self.state.update(status='awaiting_controller', waiting_reason='calG already admitted; no duplicate submission')
                    self.record()
                    return 0
                all_jobs = jobs()
                g_active = gradient_mean_active(all_jobs)
                if persistent_controller_conflict(g_active, os.environ.get('SLURM_JOB_ID'), all_jobs):
                    raise RuntimeError('While G is active, Slurm must identify this process as the approved zero-GPU critic-controller; inherited or unverified allocations are not accepted')
                active = {str(e['job_id']) for e in self.state['submitted'] if e.get('scheduler_state') not in TERMINAL}
                cohort = self.next_cohort()
                if not cohort and not active:
                    jobs_status = ('completed_with_failures' if self.state.get('blocked_conditions') or
                                   any(e.get('run_status') != 'complete' for e in self.state['submitted']) else 'completed')
                    self.state.update(status='analyzing', scientific_jobs_status=jobs_status,
                                      monitor_status='active', waiting_reason='all requested runs/evaluations terminal; analyze saved results')
                    self.record()
                    self.monitor('scientific jobs terminal')
                    good = self.final_analysis(partial=jobs_status == 'completed_with_failures')
                    self.state.update(status=jobs_status if good else 'analysis_failed', monitor_status='paused',
                                      waiting_reason='review requested results' if good else 'review failed final analysis; raw results preserved')
                    self.record(); self.monitor('campaign terminal')
                    return 0 if good else 1
                reason = 'waiting for admitted runs'
                if cohort:
                    count = len(cohort)
                    allowed, reason = capacity(all_jobs, active, count, g_active)
                    if g_active and cohort[0]['mode'] != 'calibrate':
                        allowed, reason = False, 'gradient-mean campaign retains priority; wait for its terminal state'
                    needed = sum(s['cap_seconds'] for s in cohort)
                    if self.state['ledger']['actual_plus_reserved_gpu_seconds'] + needed > BUDGET:
                        allowed, reason = False, '60 GPU-hour actual-plus-reserved budget'
                        if not active:
                            self.state.update(status='budget_limited', monitor_status='paused', waiting_reason=reason)
                            self.record(); self.monitor('budget limit')
                            self.final_analysis(partial=True)
                            return 2
                    if allowed:
                        self.state['waiting_reason'] = 'admit ' + ', '.join(s['run_id'] for s in cohort)
                        for spec in cohort:
                            self.submit(spec)
                        if submit_cal_only:
                            self.state.update(status='awaiting_controller', monitor_status='worker_active', waiting_reason='calG worker monitored; controller resumes after G')
                            self.record(); self.monitor('standalone calibration submitted')
                            return 0
                        self.monitor('cohort admitted')
                        self.next_monitor = time.monotonic() + 8400
                self.state['waiting_reason'] = reason
                self.record()
                if time.monotonic() >= self.next_monitor:
                    self.monitor('scheduled monitor')
                    self.next_monitor = time.monotonic() + 8400
                if submit_cal_only:
                    # One-shot command never occupies a Slurm controller slot
                    # or turns a failed capacity check into unbounded waiting.
                    return 3
                time.sleep(30)
        except Exception as error:
            self.state.update(status='blocked', monitor_status='paused', failure_reason=type(error).__name__ + ': ' + str(error),
                              waiting_reason='inspect recorded failure; no retries or new admissions')
            self.record(); self.monitor('controller blocked')
            append(ROOT / 'runs/events.jsonl', dict(utc=utc(), event='controller_blocked', reason=str(error)))
            # A blocked controller can still have live jobs; don't race them
            # by generating a terminal analysis while their files change.
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'run', 'submit-cal', 'status'))
    parser.add_argument('--gate', default=str(GATE))
    parser.add_argument('--spec')
    parser.add_argument('--controller-cap-seconds', type=int, default=172800)
    args = parser.parse_args()
    if not 0 < args.controller_cap_seconds <= 172800:
        parser.error('Controller lifetime is capped at48 hours; no scientific run cap changes')
    if args.action == 'prepare':
        print(json.dumps(prepare_specs(), indent=2)); return 0
    if args.action == 'status':
        print(json.dumps(read(ROOT / 'runs/campaign_progress.json', {}), indent=2)); return 0
    if args.action == 'worker':
        if not args.spec:
            parser.error('--spec required for worker')
        return worker(args)
    return Campaign(args).run(submit_cal_only=args.action == 'submit-cal')


if __name__ == '__main__':
    raise SystemExit(main())
