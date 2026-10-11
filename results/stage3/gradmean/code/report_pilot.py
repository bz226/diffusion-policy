"""Read verified pilot artifacts; report timing without changing admissions."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import statistics

from campaign import ROOT, POINTS, read, utc, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def timing_metrics(rows, metadata, manifest, cleanup):
    components = ('rollout_seconds', 'gradient_seconds', 'diagnostic_seconds')
    values = {key: [float(row[key]) for row in rows] for key in components + ('seconds',)}
    require(len(rows) == 50 and all(math.isfinite(x) and x >= 0 for series in values.values() for x in series),
            'Pilot requires fifty finite nonnegative batch timings')
    score = float(metadata['score_sample']['seconds'])
    require(metadata['score_sample']['count'] == 200 and math.isfinite(score) and score >= 0,
            'Pilot must contain the approved two-hundred-score timing')
    allocation = float(manifest['scheduler_observation']['elapsed_seconds'])
    component_total = sum(sum(values[key]) for key in components)
    recurring = (component_total - score) / 50
    require(recurring > 0, 'Measured recurring batch work must be positive')
    outside = allocation - component_total
    started = datetime.fromisoformat(manifest['started_utc']).timestamp()
    startup_estimate = float(cleanup['started_unix']) - float(metadata['elapsed_seconds']) - started
    maximum_batches, maximum_groups, scored_groups, max_parallel = 20500, 410, 5, 9
    core_seconds = maximum_batches * recurring + scored_groups * score
    total_seconds = core_seconds + maximum_groups * outside
    return {
        'batch_count': 50,
        'components_mean_seconds': {key: statistics.mean(values[key]) for key in components},
        'measured_components_mean_seconds': component_total / 50,
        'row_wall_mean_seconds': statistics.mean(values['seconds']),
        'row_wall_median_seconds': statistics.median(values['seconds']),
        'row_wall_min_seconds': min(values['seconds']),
        'row_wall_max_seconds': max(values['seconds']),
        'one_time_score_diagnostic_seconds': score,
        'score_sample_count': 200,
        'recurring_components_seconds_per_batch_excluding_score_sample': recurring,
        'allocation_seconds': allocation,
        'worker_seconds': manifest['elapsed_seconds'],
        'native_collection_seconds': metadata['elapsed_seconds'],
        'cleanup_seconds': cleanup['elapsed_seconds'],
        'startup_before_collection_seconds_estimate': startup_estimate,
        'startup_estimate_definition': 'cleanup.started_unix - native_collection_elapsed - worker_started_utc; includes initialization and small terminal-write timing differences',
        'allocation_seconds_outside_measured_batch_components': outside,
        'projection': {
            'maximum_batches': maximum_batches, 'maximum_collection_jobs': maximum_groups,
            'first_group_score_samples': scored_groups, 'max_parallel_A6000_jobs': max_parallel,
            'collection_batch_work_gpu_hours': core_seconds / 3600,
            'collection_batch_work_ideal_wall_hours_at9': core_seconds / 3600 / max_parallel,
            'collection_including_observed_allocation_overhead_gpu_hours': total_seconds / 3600,
            'collection_including_observed_allocation_overhead_ideal_wall_hours_at9': total_seconds / 3600 / max_parallel,
            'formula': '20500 * mean(rollout+gradient+diagnostic, excluding the one pilot score sample) + 5 * score_sample_seconds + 410 * observed allocation overhead',
            'scope': 'Maximum collection only; assumes pilot P0 timing transfers to all points, no early closing and no retries. Preparations, offline gates, CPU analysis barriers, queue delays and reduced GPU availability are excluded.',
            'approved_gpu_hour_cap': 125,
            'admission_policy_changed': False,
            'cap_applies_to_actual_allocation_time_plus_outstanding_full_reservations': True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pilot_review')
    args = parser.parse_args()
    output = args.output.resolve()
    require(ROOT in output.parents and not output.exists(), 'Use a new output directory under the existing gradmean study')
    manifest_path = ROOT / 'runs/P0_j00/manifest.json'
    manifest = read(manifest_path, {})
    require(manifest.get('status') == 'complete' and manifest.get('subprocess_exitcode') == 0 and
            manifest.get('remaining_process_group_members') == [], 'P0_j00 must first exit zero with no surviving child processes')
    scheduler = manifest.get('scheduler_observation', {})
    require(scheduler.get('scheduler_state') == 'COMPLETED' and scheduler.get('exit_code') == '0:0',
            'Wait for the controller to record verified Slurm COMPLETED0:0 for P0_j00')
    verification = verify(manifest)
    directory = Path(manifest['logdir'])
    metadata = read(directory / 'meta.json')
    cleanup = read(directory / 'cleanup.json')
    with (directory / 'batches.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    timing = timing_metrics(rows, metadata, manifest, cleanup)
    preparation = {}
    for point in POINTS:
        path = ROOT / 'prepare' / point / 'grad_mean.json'
        saved = read(path, {})
        if not saved:
            continue
        preparation[point] = dict(path=str(path), status=saved.get('status'),
                                  completed_warmup_batches=saved.get('completed_warmup_batches'),
                                  warmup_explained_variance_preupdate=[row.get('explained_var_preupdate') for row in saved.get('warmup', [])],
                                  frozen_critic_extra_batch_explained_variance=saved.get('frozen_ev_batch', {}).get('explained_variance'),
                                  native_prepare_elapsed_seconds=saved.get('elapsed_seconds'))
    record = dict(status='complete', recorded_utc=utc(), run_id='P0_j00', manifest=str(manifest_path),
                  source_parent_commit=manifest['parent_commit'], verification=verification,
                  timing=timing, available_prepare_explained_variances=preparation,
                  interpretation='Observed one-group timing forecast, not a compute-budget increase or a scientific result about gradient direction.')
    projection = timing['projection']
    ev_lines = []
    for point, value in preparation.items():
        ev = value['frozen_critic_extra_batch_explained_variance']
        ev_lines.append('- %s: %s warmup batches recorded; frozen-critic EV %s (%s).' %
                        (point, value['completed_warmup_batches'], 'not yet available' if ev is None else '%.6g' % ev, value['status']))
    text = '# Pilot timing readout\n\n'
    text += ('Verified P0_j00 completed fifty batches, exited zero, and reaped all forty environment workers. '
             'Its mean rollout + five-gradient + diagnostic work was **%.2f seconds per batch**; '
             'the recurring mean after removing the one-time score sample was **%.2f seconds**.\n\n') % (
                 timing['measured_components_mean_seconds'], timing['recurring_components_seconds_per_batch_excluding_score_sample'])
    text += ('At the maximum 20,500 batches, this projects **%.2f GPU-hours** of batch work, or **%.2f GPU-hours** including the pilot allocation overhead repeated for 410 collection jobs. '
             'Ideal wall time with nine A6000s is **%.2f hours** for the latter estimate. '
             'Preparations, gates, CPU analysis barriers, queue delays and any retries are additional; point-specific timings may differ and early closing reduces work.\n\n') % (
                 projection['collection_batch_work_gpu_hours'], projection['collection_including_observed_allocation_overhead_gpu_hours'],
                 projection['collection_including_observed_allocation_overhead_ideal_wall_hours_at9'])
    text += ('The pilot spent %.2f seconds on the 200-score diagnostic. Estimated startup before collection was %.2f seconds; '
             'recorded cleanup was %.2f seconds. The allocation took %.2f seconds total. '
             'These components are reported separately; startup is an estimate from saved timestamps.\n\n') % (
                 timing['one_time_score_diagnostic_seconds'], timing['startup_before_collection_seconds_estimate'],
                 timing['cleanup_seconds'], timing['allocation_seconds'])
    text += '**The approved 125 GPU-hour cap and MAX_PAR=9 remain unchanged.** Actual allocated time plus outstanding job reservations controls admission.\n\n'
    text += 'Available preparation checks:\n\n' + '\n'.join(ev_lines) + '\n\n'
    text += '[Full timing definitions and source paths](pilot_review.json). Explained variance uses each preparation’s separate frozen-critic evaluation batch.\n'
    output.mkdir(parents=True)
    (output / 'pilot_review.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    (output / 'pilot_review.md').write_text(text)
    shutil.copy2(__file__, output / 'report_pilot_source.py')
    print(json.dumps(dict(status='complete', output=str(output), recurring_seconds_per_batch=timing['recurring_components_seconds_per_batch_excluding_score_sample'],
                          projected_collection_gpu_hours=projection['collection_including_observed_allocation_overhead_gpu_hours'],
                          ideal_wall_hours_at9=projection['collection_including_observed_allocation_overhead_ideal_wall_hours_at9']), allow_nan=False))


if __name__ == '__main__':
    main()
