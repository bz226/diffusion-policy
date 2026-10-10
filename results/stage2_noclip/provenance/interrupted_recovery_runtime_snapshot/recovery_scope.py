"""Strict scope shared by the two authorized fresh replacement runs."""
import argparse
import datetime as dt
import json
from pathlib import Path

from campaign import ROOT, native_command

COMMIT = 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964'
PRIMARY_JOB = '17778495'
CAP = 8100
ESTIMATE = 7500
DEADLINE = '2026-10-10T17:49:34+00:00'
RUNS = {'R0_recovery_seed0': ('R0', 0, 'R0_rerun_seed0'),
        'NC1_recovery_seed1': ('NC1', 1, 'NC1_seed1')}


def now():
    return dt.datetime.now(dt.timezone.utc)


def validate_authorization(path, root=ROOT):
    path = Path(path).resolve()
    if path != (root / 'provenance/interrupted_recovery_authorization.json').resolve():
        raise ValueError('Unexpected recovery authorization path')
    auth = json.loads(path.read_text())
    expected = {'approved': True, 'expected_commit': COMMIT,
                'primary_controller_job_id': PRIMARY_JOB,
                'worker_runtime_cap_seconds': CAP, 'estimate_seconds': ESTIMATE,
                'max_recovery_runs': 2, 'original_deadline_utc': DEADLINE,
                'gpu_type': 'NVIDIA RTX A6000', 'cpus_per_run': 40,
                'memory_gib_per_run': 64, 'train_n_train_itr': 140, 'save_model_freq': 35,
                'launch_only_after_original_campaign_terminal': True,
                'no_training_resumption_from_partial_checkpoint': True,
                'no_automatic_retries': True, 'preserve_original_outputs': True}
    for key, value in expected.items():
        if auth.get(key) != value:
            raise ValueError('Recovery authorization differs: ' + key)
    if not 0 < auth.get('worst_case_total_gpu_hours', 0) <= auth.get('overall_gpu_hours_cap', 0) == 58.5:
        raise ValueError('Recovery exceeds the unchanged GPU-hour budget')
    paths = {'primary_campaign_path': 'runs/campaign_revised.json',
             'original_matrix_path': 'provenance/run_matrix_revised.json',
             'matrix_path': 'provenance/run_matrix_recovery.json',
             'recovery_record_path': 'runs/recovery_campaign.json',
             'fixed_batch_gate': 'runs/verification_full_update/verification.json'}
    for key, suffix in paths.items():
        if Path(auth.get(key, '')).resolve() != (root / suffix).resolve():
            raise ValueError('Recovery authorization path differs: ' + key)
    runs = auth.get('runs')
    if not isinstance(runs, list) or len(runs) != 2 or {r.get('run_id') for r in runs} != set(RUNS):
        raise ValueError('Only the two interrupted slots may be replaced')
    for run in runs:
        condition, seed, previous = RUNS[run['run_id']]
        if (run.get('condition') != condition or run.get('seed') != seed or
                run.get('replaced_run_id') != previous or
                Path(run.get('manifest', '')).resolve() != (root / 'runs' / run['run_id'] / 'manifest.json').resolve()):
            raise ValueError('Recovery run identity or predecessor differs')
    return auth


def time_remaining(auth, at=None):
    return (dt.datetime.fromisoformat(auth['original_deadline_utc']) - (at or now())).total_seconds()


def worker_args(run_id, auth):
    if run_id not in RUNS:
        raise ValueError('Only authorized fresh replacement IDs are accepted')
    condition, seed, _ = RUNS[run_id]
    run = {'run_id': run_id, 'condition': condition, 'seed': seed}
    return argparse.Namespace(run_id=run_id, condition=condition, seed=seed, probe=False,
                              rows=140, expected_commit=auth['expected_commit'],
                              timeout_seconds=CAP, estimate_seconds=ESTIMATE,
                              fixed_batch_gate=auth['fixed_batch_gate'], command=native_command(run))
