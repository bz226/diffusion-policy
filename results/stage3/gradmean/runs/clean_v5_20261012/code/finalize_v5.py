"""Verify the fixed V5 collection, render candidates, then publish checked outputs.

This CPU-only finalizer cannot launch, restart, cancel or modify scientific jobs.
It refuses incomplete collections and preserves older figures and curated files.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import queue_v5 as q

FINAL = q.HERE / 'final_analysis'
PINNED_ANALYSIS = q.HERE / 'finalize_source_snapshot'
FIGURES = ('eval_reward', 'train_reward', 'gradient_noise_as_run', 'gradient_noise_paper_form')
EXPECTED = tuple(name + suffix for name in FIGURES for suffix in ('.png', '.pdf', '.caption.md')) + (
    'points_table.csv', 'points_table.md', 'NOTE.md', 'render_record.json')
SOURCES = ('plot_clean.py', 'clean_v5_stats.py', 'clean_v5_rewards.py',
           'clean_v3_stats.py', 'clean_v3_rewards.py', 'check_clean_v5_outputs.py',
           'analysis_core.py', 'analysis_v6.py', 'gradmean_math.py')


def require_complete(state):
    if state.get('status') != 'complete' or state.get('monitor_status') != 'paused':
        raise ValueError('The bounded collection and its monitor must first finish successfully')
    entries = state.get('entries', [])
    if len(entries) != len(q.RUN_IDS) or {e.get('run_id') for e in entries} != set(q.RUN_IDS):
        raise ValueError('Finalization requires exactly five prepares, one normalization check and fifty collect jobs')
    for entry in entries:
        if (entry.get('status') != 'complete' or entry.get('scheduler_state') != 'COMPLETED' or
                entry.get('exit_code') != '0:0'):
            raise ValueError('An admitted job is not verified complete: ' + entry['run_id'])
    if state.get('source_provenance', {}).get('source_commit') != q.COMMIT:
        raise ValueError('Finalizer source revision does not match the collection')
    ledger = q.ledger(state)
    if ledger['outstanding_reserved_gpu_seconds'] or ledger['aggregate_actual_plus_reserved'] > q.TOTAL_CAP:
        raise ValueError('Final accounting is incomplete or exceeds the authorized aggregate cap')
    return ledger


def verify_terminal_jobs(state):
    verified = []
    for entry in state['entries']:
        scheduler = q.old.accounting(entry['job_id'])
        if (scheduler.get('scheduler_state') != 'COMPLETED' or scheduler.get('exit_code') != '0:0' or
                scheduler.get('gpu_count') != 1):
            raise ValueError('Live scheduler accounting disagrees with completed job ' + entry['run_id'])
        manifest = q.old.read(q.HERE / 'workers' / entry['run_id'] / 'manifest.json', {})
        if (manifest.get('status') != 'complete' or manifest.get('subprocess_exitcode') != 0 or
                manifest.get('remaining_process_group_members') != []):
            raise ValueError('Worker exit or process cleanup failed: ' + entry['run_id'])
        spec = q.old.read(q.HERE / 'specs' / (entry['run_id'] + '.json'))
        expected = q.make_spec(entry['run_id'])
        if spec != expected:
            raise ValueError('Saved job configuration differs from the frozen plan: ' + entry['run_id'])
        native = q.verify(spec)
        verified.append(dict(run_id=entry['run_id'], job_id=entry['job_id'], scheduler=scheduler, native=native))
    return verified


def load_module(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis_command(argv, timeout):
    process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=timeout)
    result = dict(utc=q.old.utc(), command=argv, returncode=process.returncode,
                  stdout=process.stdout, stderr=process.stderr, timeout_seconds=timeout)
    q.old.append(q.HERE / 'commands.jsonl', result)
    if process.returncode:
        raise RuntimeError('Analysis command failed: ' + process.stderr)
    return result


def update_catalog():
    path = q.PROJECT / 'complete_experiment.md'
    experiment_title = '### Stage 3 exact-critic gradient measurements along without_PPO_and_GAE'
    comparison_title = '### DPPO, without_PPO and without_PPO_and_GAE: rewards and exact-critic gradient noise'
    folder = './results/stage3/gradmean/runs/clean_v5_20261012/'
    design = './results/stage3/gradmean/methods.md#clean-figures-version-5'
    if not (q.G / 'methods.md').is_file() or not (q.G / 'figs_clean/render_record.json').is_file():
        raise ValueError('Completion links are missing')
    for _ in range(3):
        before = path.read_text()
        if '## Comparison analyses' not in before:
            raise ValueError('Completed-experiment catalog structure changed')
        after = before
        if experiment_title not in after:
            block = (experiment_title + '\n\n- Experiment folder: [Exact-critic measurements](' + folder + ')\n'
                     '- Experiment design: [Version 5 methods](' + design + ')\n- Curated: —\n\n')
            after = after.replace('## Comparison analyses', block + '## Comparison analyses', 1)
        if comparison_title not in after:
            block = (comparison_title + '\n\n- Analysis folder: [Verified clean figures](./results/stage3/gradmean/figs_clean/)\n'
                     '- Analysis design: [Version 5 methods](' + design + ')\n- Curated: —\n\n')
            after = after.replace('## Comparison analyses\n\n', '## Comparison analyses\n\n' + block, 1)
        temporary = path.with_name('.complete_experiment.v5.' + str(os.getpid()))
        temporary.write_text(after)
        if path.read_text() != before:
            temporary.unlink()
            continue
        os.replace(str(temporary), str(path))
        return str(path)
    raise RuntimeError('Concurrent catalog edit; no overwrite performed')


def update_main_completion(ledger, normalization):
    path = q.G / 'experiment.md'
    heading = "# Rewards and gradient noise with the run's own critic"
    replacements = {
        'Status': ('**Status.** Complete: the five exact-critic points each have 500 fresh batches '
                   '(40,000 episodes). CPU gates, normalization selection, all 56 jobs, worker cleanup '
                   'and the independently checked figures/table passed. '
                   '[Results and limitations](figs_clean/NOTE.md); earlier outputs remain preserved.'),
        'Resources and next step': ('**Resources and next step.** Aggregate gradient-mean use is %.4f GPU-hours '
            'against the approved 125-hour cap. The normalization choice was **%s**, frozen before the '
            '50 collection groups. The bounded queue is complete and its monitor is paused. '
            'The new figures and table await artifact review; nothing was automatically curated. '
            '[Verification record](runs/clean_v5_20261012/final_analysis/manifest.json). '
            '[Earlier curated results](curated/INDEX.md).') %
            (ledger['aggregate_actual_plus_reserved'] / 3600.0, normalization)}
    for _ in range(3):
        before = path.read_text()
        if not before.startswith(heading + '\n'):
            raise ValueError('Main experiment record has moved beyond version 5; preserve it for manual reconciliation')
        after = before
        for prefix, replacement in replacements.items():
            pattern = r'^\*\*' + re.escape(prefix) + r'\.\*\*.*?(?=\n\n|\Z)'
            after, count = re.subn(pattern, lambda match: replacement, after, count=1, flags=re.M | re.S)
            if count != 1:
                raise ValueError('Expected version-5 experiment paragraph is missing: ' + prefix)
        temporary = path.with_name('.experiment.v5-final.' + str(os.getpid()))
        temporary.write_text(after)
        if path.read_text() != before:
            temporary.unlink()
            continue
        os.replace(str(temporary), str(path))
        return
    raise RuntimeError('Concurrent experiment record edit; no overwrite performed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    q.COMMIT = args.commit
    q.bind_supervision()
    lock = (q.HERE / 'finalizer.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = q.old.read(q.STATE, {})
    ledger = require_complete(state)
    normalization = q.normalization()
    source = q.validate_source()
    if any(not (PINNED_ANALYSIS / name).is_file() for name in SOURCES):
        raise ValueError('The analysis source must be pinned before the collection is launched')
    FINAL.mkdir(exist_ok=False)
    record = dict(status='verifying', started_utc=q.old.utc(), job_id=os.environ.get('SLURM_JOB_ID'),
                  scientific_jobs_launched=False, gpu_count=0, source_provenance=source,
                  normalization=normalization, ledger=ledger, curated=False,
                  analysis_source_snapshot=str(PINNED_ANALYSIS))
    q.old.atomic(FINAL / 'manifest.json', record)
    try:
        record['verified_jobs'] = verify_terminal_jobs(state)
        snapshot = FINAL / 'source'
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copy2(str(PINNED_ANALYSIS / name), str(snapshot / name))
        shutil.copy2(__file__, str(snapshot / 'finalize_v5.py'))
        shutil.copy2(str(Path(__file__).with_name('queue_v5.py')), str(snapshot / 'queue_v5.py'))
        shutil.copy2(str(q.STATE), str(FINAL / 'completed_campaign.json'))
        shutil.copy2(str(q.HERE / 'normalization_selection.json'), str(FINAL / 'normalization_selection.json'))
        record['status'] = 'rendering_candidates'
        q.old.atomic(FINAL / 'manifest.json', record)
        rendered = analysis_command([sys.executable, '-B', str(snapshot / 'plot_clean.py'),
                                     '--root', str(q.G), '--analysis-source', str(snapshot), '--candidates-only'], 900)
        output = json.loads(rendered['stdout'])
        candidates = Path(output['run']) / 'candidates'
        if output.get('status') != 'complete' or output.get('pending_points') != []:
            raise ValueError('Renderer did not produce the five complete point estimates')
        if candidates.parent.parent != q.G / 'runs' or any(not (candidates / name).is_file() for name in EXPECTED):
            raise ValueError('Renderer candidates are missing or outside the authorized run directory')
        record['candidate_directory'] = str(candidates)
        record['status'] = 'checking_candidates'
        q.old.atomic(FINAL / 'manifest.json', record)
        analysis_command([sys.executable, '-B', str(snapshot / 'check_clean_v5_outputs.py'),
                          '--output-dir', str(candidates), '--report', str(FINAL / 'artifact_check.json'),
                          '--require-complete'], 600)
        check = q.old.read(FINAL / 'artifact_check.json')
        if check.get('passed') is not True or check.get('complete') is not True:
            raise ValueError('Independent raw-data and artifact checks did not pass')
        saved_renderer = load_module(candidates.parent / 'plot_clean.py', 'verified_v5_publisher')
        saved_renderer.publish(candidates, q.G / 'figs_clean')
        for name in EXPECTED:
            if (candidates / name).read_bytes() != (q.G / 'figs_clean' / name).read_bytes():
                raise ValueError('Published artifact differs from verified candidate: ' + name)
        record.update(status='complete', completed_utc=q.old.utc(), artifacts=list(EXPECTED),
                      published_directory=str(q.G / 'figs_clean'), artifact_check=str(FINAL / 'artifact_check.json'))
        q.old.atomic(FINAL / 'manifest.json', record)
        record['completed_catalog'] = update_catalog()
        q.old.atomic(FINAL / 'manifest.json', record)
        (q.HERE / 'experiment.md').write_text(
            '# Exact-critic gradient measurements\n\n'
            'Status: **complete**. All five preparations, the two-batch normalization check and '
            'fifty 50-batch collections passed their saved-output, scheduler and cleanup checks. '
            'Every point contains 500 fresh batches (40,000 episodes). '
            'The critic was never warmed or updated; the final divisor used the authorized five-batch fit.\n\n'
            '[Figures, table and interpretation](../../figs_clean/NOTE.md) · '
            '[Independent artifact checks](final_analysis/artifact_check.json) · '
            '[Methods](../../methods.md#clean-figures-version-5).\n\n'
            'The normalization choice was **' + normalization + '**, frozen before collection. '
            'Aggregate charged GPU time is %.3f hours against the 125-hour cap. '
            'No outputs were curated.\n' % (ledger['aggregate_actual_plus_reserved'] / 3600.0))
        q.update_main_marker('**Clean figures version 5: complete.** All 56 jobs and the final figures/table '
                             'passed completion checks. [Results and limitations](figs_clean/NOTE.md). '
                             'The collection monitor is paused; artifacts remain uncurated.')
        update_main_completion(ledger, normalization)
        print(json.dumps(dict(status='complete', output=record['published_directory'],
                              record=str(FINAL / 'manifest.json'), curated=False), indent=2))
        return 0
    except BaseException as error:
        record.update(status='blocked', failed_utc=q.old.utc(), failure_reason=type(error).__name__ + ': ' + str(error))
        q.old.atomic(FINAL / 'manifest.json', record)
        raise


if __name__ == '__main__':
    sys.exit(main())
