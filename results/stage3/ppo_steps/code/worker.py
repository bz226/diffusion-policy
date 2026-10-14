"""One bounded scientific command in its own process group (Python 3.8)."""
import argparse
import ctypes
import json
import math
import os
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
        env = os.environ.copy()
        for key, value in spec.get('env', {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[str(key)] = str(value)
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
            next_monitor = 0.0
            # Leave cleanup time before the scheduler's hard job deadline.
            cap = max(1.0, float(spec['cap_seconds']) - 25.0)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if interrupted:
                    raise RuntimeError('Worker received signal %d' % interrupted[0])
                if elapsed >= cap:
                    raise RuntimeError('Scientific-command wall cap reached')
                if time.monotonic() >= next_monitor:
                    progress = read_json(spec['progress_file'], {}) if spec.get('progress_file') else {}
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
