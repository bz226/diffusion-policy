"""Small Python 3.8 runtime helpers for this approved measurement only."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
STUDY = PROJECT / 'results/stage3/noise_without_ppo'
RUN = STUDY / 'runs/campaign'
WORKTREE = PROJECT / '.codex-runtime/worktrees/noise-without-ppo'
SOURCE = WORKTREE / 'results/stage3/noise_without_ppo/dppo'
PYTHON = PROJECT / 'results/repro_halfcheetah/env/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(tmp), str(path))


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def command(argv, timeout=40):
    result = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True,
                            timeout=timeout)
    append(RUN / 'commands.jsonl', dict(utc=utc(), argv=[str(x) for x in argv],
                                      returncode=result.returncode,
                                      stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        raise RuntimeError('Command failed: %r\n%s' % (argv, result.stderr))
    return result.stdout


def environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SOURCE), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               DPPO_DATA_DIR=str(PROJECT / 'results/repro_halfcheetah/dppo/data'),
               DPPO_LOG_DIR=str(PROJECT / 'results/repro_halfcheetah/dppo/log'),
               LD_LIBRARY_PATH='/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia')
    env['PATH'] = str(PYTHON.parent) + ':' + env.get('PATH', '')
    for key, name in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'),
                      ('MPLCONFIGDIR', 'cache/matplotlib'), ('TORCH_HOME', 'cache/torch'),
                      ('TRITON_CACHE_DIR', 'cache/triton'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                      ('D4RL_DATASET_DIR', 'cache/d4rl')]:
        path = STUDY / 'runs' / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.pop(key, None)
    return env


def files(path):
    path = Path(path)
    return {str(p.relative_to(path)): p for p in path.rglob('*')
            if p.is_file() and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.parts)
            and p.suffix != '.pyc'}


def verify_source():
    provenance = read(STUDY / 'provenance/launch_source.json')
    if not provenance:
        raise RuntimeError('Missing committed launch provenance')
    revision = command(['git', '-C', WORKTREE, 'rev-parse', 'HEAD']).strip()
    if revision != provenance['revision']:
        raise RuntimeError('Isolated worktree revision changed')
    relative = str(SOURCE.relative_to(WORKTREE))
    dirty = command(['git', '-C', WORKTREE, 'status', '--porcelain', '--', relative])
    if dirty.strip():
        raise RuntimeError('Uncommitted scientific source changes: ' + dirty)
    actual, expected = files(SOURCE), files(provenance['snapshot'])
    if set(actual) != set(expected):
        raise RuntimeError('Scientific snapshot file list changed')
    for name in actual:
        if actual[name].read_bytes() != expected[name].read_bytes():
            raise RuntimeError('Scientific source changed since snapshot: ' + name)
    return revision


def children(pid):
    relations = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
            relations[int(entry.name)] = int(raw[raw.rfind(')') + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    found, frontier = set(), {pid}
    while frontier:
        frontier = {p for p, parent in relations.items() if parent in frontier and p not in found}
        found.update(frontier)
    return sorted(found)


def reap():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def stop_owned(process):
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        for pid in children(os.getpid()):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            reap()
            if not children(os.getpid()):
                return []
            time.sleep(.1)
    return children(os.getpid())


def bounded_worker(spec_path):
    spec = read(spec_path)
    folder = RUN / 'workers' / spec['id']
    if (folder / 'manifest.json').exists():
        raise RuntimeError('A scientific worker cannot silently restart')
    folder.mkdir(parents=True, exist_ok=True)
    revision = verify_source()
    # Adopt and reap only this worker's own descendants if its child crashes.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('Unable to establish child cleanup supervision')
    manifest = dict(status='running', started_utc=utc(), spec=str(spec_path), revision=revision,
                    job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    command=spec['command'], cap_seconds=spec['cap_seconds'])
    write(folder / 'manifest.json', manifest)
    process, started, interrupted = None, time.monotonic(), []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    try:
        log = folder / 'console.out'
        with log.open('x') as handle:
            process = subprocess.Popen(spec['command'], cwd=str(SOURCE), env=environment(),
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if interrupted:
                    raise RuntimeError('Worker interrupted by signal %s' % interrupted[-1])
                if time.monotonic() - started > spec['cap_seconds'] - 25:
                    raise RuntimeError('Approved worker runtime cap reached; cleaning owned children')
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError('Scientific process exited %s; see %s' % (process.returncode, log))
        reap()
        if children(os.getpid()):
            raise RuntimeError('Scientific process left child processes behind')
        manifest.update(status='complete', exit_code=0, children_remaining=[])
    except BaseException as error:
        remaining = stop_owned(process) if process is not None else children(os.getpid())
        manifest.update(status='failed', error='%s: %s' % (type(error).__name__, error),
                        children_remaining=remaining)
        raise
    finally:
        manifest.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started)
        write(folder / 'manifest.json', manifest)
