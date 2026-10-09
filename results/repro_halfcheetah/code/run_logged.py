#!/usr/bin/env python3
"""Record setup/inspection commands without changing the DPPO checkout."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--timeout', type=float, default=120)
p.add_argument('--cwd')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
root = Path(__file__).resolve().parents[1]
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
log = root / 'setup' / (stamp + '_' + a.label + '.log')
record = dict(start_utc=stamp, label=a.label, command=command,
              cwd=str(Path(a.cwd or os.getcwd()).resolve()), timeout_seconds=a.timeout,
              output=str(log))
with (root / 'provenance' / 'commands.jsonl').open('a') as f:
    f.write(json.dumps(dict(event='start', **record)) + '\n')
start = time.monotonic()
proc = subprocess.Popen(command, cwd=a.cwd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, start_new_session=True)
expired = threading.Event()
def stop():
    expired.set()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(10)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
timer = threading.Timer(a.timeout, stop)
timer.daemon = True
timer.start()
with log.open('xb') as f:
    for line in iter(proc.stdout.readline, b''):
        f.write(line)
        f.flush()
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
rc = proc.wait()
timer.cancel()
record.update(exit_code=rc, timed_out=expired.is_set(), elapsed_seconds=time.monotonic()-start,
              end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (root / 'provenance' / 'commands.jsonl').open('a') as f:
    f.write(json.dumps(dict(event='end', **record)) + '\n')
sys.exit(124 if expired.is_set() else rc)
