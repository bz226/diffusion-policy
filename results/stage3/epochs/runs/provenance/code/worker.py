"""One bounded scientific command in its own process group (Python 3.8)."""
import argparse
import ctypes
import json
import math
import os
import pickle
import numbers
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        with open(str(path)) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    os.replace(str(temporary), str(path))


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def descendants(parent):
    """Read only our descendant process IDs; unrelated processes are excluded."""
    relations = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            relations[int(entry.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    found, frontier = set(), {parent}
    while frontier:
        frontier = {pid for pid, ppid in relations.items() if ppid in frontier and pid not in found}
        found.update(frontier)
    return sorted(found)


def reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def stop_owned(process):
    """Cleanup only this command's group and this worker's adopted children."""
    for signum, grace in ((signal.SIGTERM, 15.0), (signal.SIGKILL, 5.0)):
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        for pid in descendants(os.getpid()):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if process is not None:
                process.poll()
            reap()
            if not descendants(os.getpid()):
                return []
            time.sleep(0.2)
    return descendants(os.getpid())


def json_check(check):
    """Return (ready, detail); missing paths are readiness waits, not success."""
    path = Path(check['path'])
    if not path.is_file():
        return False, 'missing ' + str(path)
    try:
        value = read_json(path)
        for key in check.get('key', '').split('.'):
            if key:
                value = value[int(key)] if isinstance(value, list) else value[key]
        expected = check.get('equals', True)
        return value == expected, '%s:%s = %r (wanted %r)' % (path, check.get('key', ''), value, expected)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return False, '%s: %s' % (path, error)


def source_files(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): p for p in directory.rglob('*')
            if p.is_file() and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(directory).parts)
            and p.suffix != '.pyc'}


def verify_source(provenance):
    """Bind to the approved parent revision and exact ordinary-source snapshot.

    This intentionally permits main and relevant uncommitted source captured in
    the snapshot. Administrative branch or unrelated-file changes do not change
    scientific identity; changing a scientific file does.
    """
    subprocess.run(['git', '-C', provenance['worktree'], 'cat-file', '-e',
                    provenance['revision'] + '^{commit}'], stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, universal_newlines=True, timeout=30, check=True)
    actual, saved = source_files(provenance['source']), source_files(provenance['snapshot'])
    if not actual or set(actual) != set(saved):
        raise RuntimeError('Scientific source file list differs from gate snapshot')
    changed = [name for name in actual if actual[name].read_bytes() != saved[name].read_bytes()]
    if changed:
        raise RuntimeError('Scientific source contents changed: ' + ', '.join(changed))
    return dict(revision=provenance['revision'], files_compared=len(actual), identical=True)


def nonfinite(value, path=''):
    if isinstance(value, dict):
        return [item for key, val in value.items() for item in nonfinite(val, path + '.' + str(key))]
    if isinstance(value, (list, tuple)):
        return [item for idx, val in enumerate(value) for item in nonfinite(val, path + '[%d]' % idx)]
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        return [path]
    return []


def observe(spec):
    directory = Path(spec['logdir'])
    result = dict(utc=utc())
    if spec['mode'] == 'train' and (directory / 'result.pkl').exists():
        try:
            with (directory / 'result.pkl').open('rb') as handle:
                rows = pickle.load(handle)
            result.update(rows=len(rows), last_itr=rows[-1]['itr'] if rows else None,
                          nonfinite_paths=nonfinite(rows))
            wrong_lr = [r['itr'] for r in rows if r.get('actor_lr') != spec['actor_lr']]
            if wrong_lr:
                result['failure'] = dict(gate='G5', error='Actor LR differs from intended constant',
                                         iterations=wrong_lr)
            if rows:
                result['last'] = {key: rows[-1][key] for key in
                                 ('itr', 'step', 'train_episode_reward', 'eval_episode_reward', 'J_disc_train')
                                 if key in rows[-1]}
        except (EOFError, pickle.UnpicklingError):
            result['write_in_progress'] = True
    else:
        try:
            result['progress'] = read_json(spec['progress_file'], {})
        except json.JSONDecodeError:
            result['write_in_progress'] = True
    if (directory / 'failure.json').is_file():
        result['failure'] = read_json(directory / 'failure.json')
    output = Path(spec['stdout'])
    if output.is_file():
        with output.open('rb') as handle:
            handle.seek(max(0, output.stat().st_size - 16000))
            result['console_tail'] = handle.read().decode('utf-8', errors='replace').splitlines()[-15:]
    return result


def validate_scientific_output(spec):
    directory = Path(spec['logdir'])
    cleanup = read_json(directory / 'cleanup.json', {})
    if cleanup.get('status') != 'passed' or not cleanup.get('all_children_exited'):
        raise RuntimeError('Explicit vector-environment cleanup was not verified')
    with (directory / 'result.pkl').open('rb') as handle:
        rows = pickle.load(handle)
    if len(rows) != 140 or [int(r['itr']) for r in rows] != list(range(140)):
        raise RuntimeError('Training did not save all 140 iterations')
    bad = nonfinite(rows)
    if bad:
        raise RuntimeError('Non-finite training output: ' + ', '.join(bad[:10]))
    train = [r for r in rows if 'train_episode_reward' in r]
    evaluations = [r for r in rows if 'eval_episode_reward' in r]
    if len(train) != 126 or len(evaluations) != 14:
        raise RuntimeError('Incorrect training/evaluation iteration counts')
    for itr in (0, 35, 70, 105, 139):
        if not (directory / 'checkpoint' / ('state_%d.pt' % itr)).is_file():
            raise RuntimeError('Missing requested checkpoint %d' % itr)
    intended_lr = float(spec['actor_lr'])
    epochs = int(spec['actor_update_epochs'])
    if epochs not in (1, 2):
        raise RuntimeError('Unexpected actor epoch condition')
    # G5 is checked on every saved iteration, including evaluation-only rows.
    for row in rows:
        if row.get('actor_lr') != intended_lr:
            raise RuntimeError('G5: actor lr changed at iteration %s: %r != %r' %
                               (row['itr'], row.get('actor_lr'), intended_lr))
    required = ('J_disc_train', 'theta_dist', 'theta_dist_before_update',
                'x0_sat_frac_mean', 'kl_true_per_action', 'ppo_steps',
                'ppo_step_count', 'ppo_clipfrac_mean', 'ppo_approx_kl_mean',
                'ppo_switched_off_frac_mean', 'ppo_mean_abs_ratio_minus_one_mean',
                'ppo_switched_off_frac_last', 'ppo_approx_kl_last',
                'ppo_mean_abs_ratio_minus_one_last', 'actor_optimizer_steps',
                'critic_optimizer_steps', 'actor_adam_step_count', 'actor_update_epochs', 'ppo_kl_early_stop')
    actor_count = 0
    early_stop = []
    for row in rows:
        if 'train_episode_reward' not in row:
            if row.get('actor_adam_step_count') != actor_count:
                raise RuntimeError('Adam step count changed during evaluation')
            continue
        for key in required:
            if key not in row:
                raise RuntimeError('Missing diagnostic ' + key)
        steps = row['ppo_steps']
        if row['actor_update_epochs'] != epochs:
            raise RuntimeError('Configured actor epoch count changed')
        if not 1 <= len(steps) <= 4 * epochs or row['ppo_step_count'] != len(steps):
            raise RuntimeError('Invalid PPO actor-step count')
        if row['actor_optimizer_steps'] != len(steps):
            raise RuntimeError('Actor optimizer count differs from recorded actor steps')
        if any(step['minibatch_size'] != 50000 for step in steps):
            raise RuntimeError('PPO minibatch size changed')
        shortened = len(steps) != 4 * epochs or row['critic_optimizer_steps'] != 20
        if shortened and not row['ppo_kl_early_stop']:
            raise RuntimeError('Short update loop without a recorded KL early stop')
        if row['ppo_kl_early_stop']:
            early_stop.append(dict(itr=row['itr'], actor_steps=len(steps),
                                   critic_steps=row['critic_optimizer_steps']))
        if not 1 <= row['critic_optimizer_steps'] <= 20:
            raise RuntimeError('Invalid critic step count')
        actor_count += len(steps)
        if row['actor_adam_step_count'] != actor_count:
            raise RuntimeError('Actor Adam cumulative count is inconsistent')
        if row['itr'] == 1 and (len(steps) != 4 * epochs or row['critic_optimizer_steps'] != 20):
            raise RuntimeError('G3: the first live update did not take 4k actor and 20 critic steps')
    first = evaluations[0]['eval_episode_reward']
    return dict(training_iterations=126, evaluation_iterations=14,
                actor_update_epochs=epochs, actor_lr=intended_lr,
                g5_constant_lr_every_iteration=True,
                final_actor_adam_step_count=actor_count,
                ppo_step_counts=[r['ppo_step_count'] for r in train],
                early_stop_iterations=early_stop,
                collapse_eval_iterations=[r['itr'] for r in evaluations if r['eval_episode_reward'] < .5 * first])


def run(spec_path):
    spec = read_json(spec_path)
    root = Path(spec['run_root'])
    directory = root / 'workers' / spec['id']
    manifest_path = directory / 'manifest.json'
    existing = read_json(manifest_path, {})
    if existing.get('status'):
        raise RuntimeError('Refusing to restart an existing worker: ' + str(manifest_path))
    directory.mkdir(parents=True, exist_ok=True)
    # Reparent grandchildren to this supervisor so clean exits are checkable.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    interrupted = []

    def on_signal(signum, frame):
        interrupted.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)
    manifest = dict(spec=spec, id=spec['id'], status='starting', started_utc=utc(),
                    scheduler_job_id=os.environ.get('SLURM_JOB_ID'), worker_pid=os.getpid(),
                    python=sys.version, command=spec['command'], cwd=spec['cwd'])
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    process, result = None, 1
    try:
        for check in spec.get('startup_checks', []):
            okay, detail = json_check(check)
            if not okay:
                raise RuntimeError('Startup check failed: ' + detail)
        manifest['source_before'] = verify_source(spec['source_provenance'])
        if Path(spec['logdir']).exists():
            raise RuntimeError('Refusing to reuse an existing scientific output directory')
        env = os.environ.copy()
        for key, value in spec.get('env', {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[str(key)] = str(value)
        for name in ('TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'D4RL_DATASET_DIR'):
            if env.get(name):
                Path(env[name]).mkdir(parents=True, exist_ok=True)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20, check=True)
        manifest['hardware'] = dict(gpu=gpu.stdout.strip(), node=os.environ.get('SLURMD_NODENAME'), cpu_affinity=len(os.sched_getaffinity(0)))
        manifest['environment'] = {key: env.get(key) for key in spec.get('env', {})}
        output = Path(spec.get('stdout', str(root / 'logs' / (spec['id'] + '.out'))))
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RuntimeError('Refusing to overwrite console output: ' + str(output))
        with output.open('x') as handle:
            process = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=env,
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            manifest.update(status='running', scientific_pid=process.pid, stdout=str(output))
            atomic_json(manifest_path, manifest)
            next_monitor, next_health = 0.0, 0.0
            # Leave cleanup time before the scheduler's hard job deadline.
            cap = max(1.0, float(spec['cap_seconds']) - 25.0)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if interrupted:
                    raise RuntimeError('Worker received signal %d' % interrupted[0])
                if elapsed >= cap:
                    raise RuntimeError('Scientific-command wall cap reached')
                if time.monotonic() >= next_health:
                    health = observe(spec)
                    if health.get('nonfinite_paths') or health.get('failure'):
                        raise RuntimeError('Scientific non-finite values or recorded failure: %r' % health)
                    next_health = time.monotonic() + 60.0
                if time.monotonic() >= next_monitor:
                    progress = observe(spec)
                    manifest.update(elapsed_seconds=elapsed, observed_utc=utc(), progress=progress)
                    atomic_json(manifest_path, manifest)
                    append_json(directory / 'monitor_history.jsonl',
                                dict(utc=utc(), elapsed_seconds=elapsed, progress=progress))
                    next_monitor = time.monotonic() + float(spec.get('monitor_seconds', 3600))
                time.sleep(1.0)
            returncode = process.returncode
        # A successful scientific program must close and join all workers itself.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            reap()
            if not descendants(os.getpid()):
                break
            time.sleep(0.1)
        remaining = descendants(os.getpid())
        if returncode != 0:
            raise RuntimeError('Scientific command exited with code %d' % returncode)
        if remaining:
            raise RuntimeError('Scientific command left child processes: %r' % remaining)
        for path in spec.get('completion_files', []):
            if not Path(path).is_file():
                raise RuntimeError('Required completed output is missing: ' + path)
        for check in spec.get('completion_checks', []):
            okay, detail = json_check(check)
            if not okay:
                raise RuntimeError('Completion check failed: ' + detail)
        manifest['scientific_validation'] = validate_scientific_output(spec)
        manifest['source_after'] = verify_source(spec['source_provenance'])
        manifest.update(status='complete', exit_code=0, child_processes_after=[])
        result = 0
    except BaseException as error:
        remaining = stop_owned(process)
        manifest.update(status='failed', error='%s: %s' % (type(error).__name__, error),
                        exit_code=process.returncode if process is not None else None,
                        child_processes_after=remaining)
        print(manifest['error'], file=sys.stderr, flush=True)
    finally:
        manifest.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                        monitor_status='paused')
        atomic_json(manifest_path, manifest)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    args = parser.parse_args()
    sys.exit(run(args.spec))
