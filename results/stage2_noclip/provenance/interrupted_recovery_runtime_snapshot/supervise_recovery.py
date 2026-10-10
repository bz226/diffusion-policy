#!/usr/bin/env python3
"""Adapt only two approved recovery IDs to the unchanged Stage 2 supervisor."""
import argparse
import os
import sys
from pathlib import Path

from recovery_scope import CAP, ROOT, RUNS, time_remaining, validate_authorization, worker_args
from supervise_run import Supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, choices=sorted(RUNS))
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    auth = validate_authorization(args.authorization)
    supervisor = Supervisor(worker_args(args.run_id, auth))
    supervisor.manifest['recovery'] = {
        'authorization_path': str(args.authorization.resolve()),
        'replaced_run_id': RUNS[args.run_id][2], 'fresh_from_released_pretrained_policy': True,
        'resume_attempted': False, 'automatic_retries': 0,
        'original_deadline_utc': auth['original_deadline_utc']}
    if (time_remaining(auth) < CAP or os.environ.get('SLURM_MEM_PER_NODE') != '65536'):
        # Route a refused start through the original terminal-recording path.
        reason = ('Recovery allocation began too late for the unchanged 8100-second cap'
                  if time_remaining(auth) < CAP else 'Recovery allocation must have 64 GiB memory')
        supervisor.manifest['recovery']['start_refusal'] = reason
        supervisor.preflight = lambda: (_ for _ in ()).throw(ValueError(reason))
    return supervisor.run()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('Recovery supervisor refused: ' + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
