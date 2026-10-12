"""Bounded CPU recovery finalizer; preserve all reward and curated artifacts.

No scheduler submissions, sampling, restarts, or scientific-source edits. The
original finalizer supplies terminal-job, frozen-state, source and budget checks.
Only the two V5 gradient figures, table, note and render record are published.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CODE = Path(__file__).resolve().parent
HERE = CODE.parent.parent
GRADIENT_FIGURES = ('gradient_noise_as_run', 'gradient_noise_paper_form')
PUBLISH = tuple(name + suffix for name in GRADIENT_FIGURES for suffix in ('.png', '.pdf', '.caption.md')) + (
    'points_table.csv', 'points_table.md', 'NOTE.md', 'render_record.json')


def protected_files(figures, curated):
    """Read bytes directly; no new integrity hashes or modification of originals."""
    paths = set(path for path in figures.rglob('*') if path.is_file() and 'reward' in path.name.lower())
    paths.update(path for path in curated.rglob('*') if path.is_file())
    return {str(path): path.read_bytes() for path in sorted(paths)}


def require_preserved(before, figures, curated):
    after = protected_files(figures, curated)
    if before.keys() != after.keys():
        raise ValueError('Existing reward/curated file set changed during finalization')
    different = [name for name in before if before[name] != after[name]]
    if different:
        raise ValueError('Existing reward/curated bytes changed: ' + ', '.join(different))
    return dict(passed=True, files=len(before), method='direct byte comparison',
                scope='all existing reward files including history, and every curated file')


def publish_gradients(candidates, figures):
    """Explicit allowlist: never call the renderer's broad publication helper."""
    if {path.name for path in candidates.iterdir() if path.is_file()} != set(PUBLISH):
        raise ValueError('Gradient-only candidates contain missing or unexpected files')
    figures.mkdir(parents=True, exist_ok=True)
    existing = [figures / name for name in PUBLISH if (figures / name).is_file()]
    archive = None
    if existing:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        archive = figures / 'history' / ('gradient_v5_final_' + stamp)
        archive.mkdir(parents=True)
        for path in existing:
            shutil.copy2(str(path), str(archive / path.name))
    for name in PUBLISH:
        temporary = figures / ('.gradient-final-' + str(os.getpid()) + '-' + name)
        with temporary.open('xb') as stream:
            stream.write((candidates / name).read_bytes())
        os.replace(str(temporary), str(figures / name))
        if (figures / name).read_bytes() != (candidates / name).read_bytes():
            raise ValueError('Published gradient artifact differs: ' + name)
    return str(archive) if archive else None


def old_finalizer():
    original = HERE / 'code/finalize_v5.py'
    sys.path.insert(0, str(original.parent))
    spec = importlib.util.spec_from_file_location('validated_original_v5_finalizer', str(original))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(old, argv, timeout):
    return old.analysis_command(argv, timeout)


def bind_read_query_retries(q):
    """Reuse the tested bounded read-query retries; mutations are never retried."""
    source = CODE / 'recover_controller.py'
    spec = importlib.util.spec_from_file_location('recovery_read_queries_for_finalizer', str(source))
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    if recovery.q is not q or recovery.BASE_COMMAND is not q.old.command:
        raise ValueError('Read-query retry adapter did not capture the original command function')
    q.old.command = recovery.query_command
    return recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    old = old_finalizer()
    q = old.q
    if q.HERE.resolve() != HERE:
        raise ValueError('Recovery wrapper resolved a different authorized campaign')
    q.COMMIT = args.commit
    bind_read_query_retries(q)
    q.bind_supervision()
    lock = (HERE / 'finalizer.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = q.old.read(q.STATE, {})
    ledger = old.require_complete(state)
    normalization = q.normalization()
    source = q.validate_source()
    final = HERE / 'final_analysis'
    final.mkdir(exist_ok=False)
    figures, curated = q.G / 'figs_clean', q.G / 'curated'
    before = protected_files(figures, curated)
    record = dict(status='verifying', started_utc=q.old.utc(), job_id=os.environ.get('SLURM_JOB_ID'),
                  scientific_jobs_launched=False, gpu_count=0, curated=False,
                  source_provenance=source, ledger=ledger, normalization=normalization,
                  artifacts_scope='gradient figures and table only',
                  protected_files=[dict(path=name, bytes=len(value)) for name, value in before.items()],
                  recovery_finalizer=str(Path(__file__).resolve()))
    q.old.atomic(final / 'manifest.json', record)
    try:
        record['verified_jobs'] = old.verify_terminal_jobs(state)
        snapshot = final / 'source'
        shutil.copytree(str(CODE / 'analysis_snapshot'), str(snapshot))
        for name in ('render_gradients_only.py', 'finalize_gradients_v5.py', 'recover_controller.py',
                     'run_finalizer.sh', 'source_snapshot.json'):
            shutil.copy2(str(CODE / name), str(snapshot / name))
        shutil.copy2(str(q.STATE), str(final / 'completed_campaign.json'))
        shutil.copy2(str(HERE / 'normalization_selection.json'), str(final / 'normalization_selection.json'))
        record.update(status='rendering_candidates', analysis_source_snapshot=str(snapshot))
        q.old.atomic(final / 'manifest.json', record)
        rendered = command(old, [sys.executable, '-B', str(snapshot / 'render_gradients_only.py'),
                                '--root', str(q.G), '--snapshot', str(snapshot)], 900)
        output = json.loads(rendered['stdout'])
        candidates = Path(output['run']) / 'candidates'
        if (output.get('status') != 'complete' or output.get('pending_points') != [] or
                candidates.parent.parent != q.G / 'runs'):
            raise ValueError('Gradient renderer did not produce the complete authorized cohort')
        record.update(candidate_directory=str(candidates), status='checking_candidates')
        q.old.atomic(final / 'manifest.json', record)
        command(old, [sys.executable, '-B', str(snapshot / 'check_clean_v5_outputs.py'),
                      '--output-dir', str(candidates), '--report', str(final / 'artifact_check.json'),
                      '--require-complete'], 600)
        check = q.old.read(final / 'artifact_check.json')
        if check.get('passed') is not True or check.get('complete') is not True:
            raise ValueError('Independent gradient checks did not pass')
        require_preserved(before, figures, curated)
        record['previous_gradient_archive'] = publish_gradients(candidates, figures)
        record['preservation_check'] = require_preserved(before, figures, curated)
        record.update(status='updating_completion_records', artifacts=list(PUBLISH),
                      published_directory=str(figures), artifact_check=str(final / 'artifact_check.json'))
        q.old.atomic(final / 'manifest.json', record)
        record['completed_catalog'] = old.update_catalog()
        (HERE / 'experiment.md').write_text(
            '# Exact-critic gradient measurements\n\nStatus: **complete**. '
            'All five preparations, the two-batch normalization check and fifty 50-batch collections '
            'passed saved-output, scheduler and cleanup checks. Every point has 500 fresh batches '
            '(40,000 episodes); no actor or critic updates occurred. The final divisor used the '
            'approved five-batch fit.\n\n'
            '[Gradient figures, table and interpretation](../../figs_clean/NOTE.md) · '
            '[Independent checks](final_analysis/artifact_check.json) · '
            '[Methods](../../methods.md#clean-figures-version-5).\n\n'
            'Normalization was **' + normalization + '**, frozen before collection. '
            'Aggregate GPU time is %.3f hours against the 125-hour cap. '
            'Existing single-seed and three-seed reward artifacts and all curated files were '
            'preserved byte for byte. No outputs were automatically curated.\n' %
            (ledger['aggregate_actual_plus_reserved'] / 3600.0))
        q.update_main_marker('**Clean figures version 5: complete.** All 56 jobs and final gradient figures/table '
                             'passed checks. [Results](figs_clean/NOTE.md). Reward and curated artifacts were preserved; '
                             'the collection monitor is paused.')
        old.update_main_completion(ledger, normalization)
        record['preservation_check'] = require_preserved(before, figures, curated)
        record.update(status='complete', completed_utc=q.old.utc())
        q.old.atomic(final / 'manifest.json', record)
        print(json.dumps(dict(status='complete', output=str(figures), record=str(final / 'manifest.json'),
                              artifacts=list(PUBLISH), rewards_preserved=True, curated=False), indent=2))
        return 0
    except BaseException as error:
        record.update(status='blocked', failed_utc=q.old.utc(), failure_reason=type(error).__name__ + ': ' + str(error))
        q.old.atomic(final / 'manifest.json', record)
        raise


if __name__ == '__main__':
    sys.exit(main())
