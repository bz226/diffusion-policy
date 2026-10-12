"""Render the fixed Round 2 tables; never import training code or collect samples."""
import csv
import datetime
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN.parent / 'round_1400' / 'candidates'
OUT = RUN / 'candidates'
POINTS = ('P0', 'D1', 'D2', 'N1', 'N2')
COLORS = {'DPPO': '#2766A5', 'NC4': '#BE5B26'}
LABELS = {'P0': 'Pretrained', 'D1': 'DPPO middle', 'D2': 'DPPO final',
          'N1': 'NC4 middle', 'N2': 'NC4 final'}


def rows(name):
    with (SOURCE / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError('Nonfinite source estimate: ' + key)
    return value


def main():
    started = time.monotonic()
    if (RUN / 'manifest.json').exists():
        raise ValueError('Preserve completed outputs; choose a new review run for a revision.')
    status = json.loads((SOURCE.parent / 'analysis_status.json').read_text())
    assert status['status'] == 'complete' and status['new_scientific_samples'] == 0
    estimates = rows('main_results.csv')
    assert len(estimates) == 25
    assert {(r['point'], r['variant']) for r in estimates} == {
        (point, variant) for point in POINTS for variant in ('V1', 'V2', 'V3', 'V4', 'V5')}
    assert all(int(r['K']) == 1400 and int(r['groups']) == 28 and
               int(r['episodes']) == 112000 for r in estimates)
    v4 = {r['point']: r for r in estimates if r['variant'] == 'V4'}
    policy = {r['point']: r for r in rows('per_point.csv')}
    inventory = [r for r in rows('run_inventory.csv') if r['step'] == 'collect']
    assert len(inventory) == 140 and len({r['run_id'] for r in inventory}) == 140
    assert all(r['status'] == 'complete' and r['scheduler_state'] == 'COMPLETED'
               and r['exit_code'] == '0:0' and int(r['completed_batches']) == 50
               for r in inventory)
    for point in POINTS:
        assert sum(r['point'] == point for r in inventory) == 28
        assert number(v4[point], 'u_sq_low') > 0
        assert v4[point]['gradient_dimensioned_units'] == 'paper episode-ascent'
        assert math.isclose(number(v4[point], 'theta_distance'),
                            number(policy[point], 'theta_distance'), rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(number(v4[point], 'B_ep'),
                            number(v4[point], 'trace_ep') / number(v4[point], 'u_sq'), rel_tol=1e-12)
        assert math.isclose(number(v4[point], 'expected_batch_cos'),
                            (1 + number(v4[point], 'B_ep') / 80) ** -.5, rel_tol=1e-12)

    metrics = ('u_sq', 'u_norm', 'trace_ep', 'B_ep', 'rho', 'c', 'expected_batch_cos', 'K_needed')
    columns = ['point', 'policy', 'K', 'groups', 'episodes', 'theta_distance', 'gradient_units']
    columns += [metric + suffix for metric in metrics + ('J_disc', 'return')
                for suffix in ('', '_low', '_high')]
    table = []
    for point in POINTS:
        row = dict(point=point, policy=LABELS[point], K=1400, groups=28, episodes=112000,
                   theta_distance=policy[point]['theta_distance'], gradient_units='paper episode-ascent')
        for metric in metrics + ('J_disc', 'return'):
            origin = policy[point] if metric in ('J_disc', 'return') else v4[point]
            for suffix in ('', '_low', '_high'):
                row[metric + suffix] = origin[metric + suffix]
        table.append(row)
    with (OUT / 'round2_estimates.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelsize': 11, 'axes.titlesize': 12,
                         'pdf.fonttype': 42, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0))
    fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.69, wspace=.29)
    fig.suptitle('Round 2 | Smaller mean gradients, persistently large noise',
                 x=.065, y=.965, ha='left', fontsize=18, fontweight='bold')
    fig.text(.065, .895, 'V4: discounted-return score with a leave-one-out baseline  |  1,400 batches per policy',
             fontsize=11.5, color='#444444')
    handles = [Line2D([0], [0], color='#333333', marker='o', linestyle='none', label='P0: pretrained'),
               Line2D([0], [0], color=COLORS['DPPO'], marker='o', label='DPPO: P0 → D1 → D2'),
               Line2D([0], [0], color=COLORS['NC4'], marker='s', label='NC4: P0 → N1 → N2')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .858), ncol=3,
               frameon=False, fontsize=10.5, columnspacing=2)
    panels = [
        ('u_norm', 1e3, 'Mean gradient magnitude', 'Gradient norm (×10³, paper units)',
         [.85, 2.35], [1, 1.5, 2], ['1.0', '1.5', '2.0']),
        ('trace_ep', 1e11, 'Gradient-noise variance', 'Episode-equivalent trace (×10¹¹)',
         [.7, 2.6], [.8, 1, 1.5, 2], ['0.8', '1.0', '1.5', '2.0']),
        ('B_ep', 1e3, 'Noise relative to the signal', 'Noise scale B (thousand episodes)',
         [45, 135], [50, 75, 100, 125], ['50', '75', '100', '125'])]
    for index, (axis, panel) in enumerate(zip(axes, panels)):
        metric, divisor, title, ylabel, limits, ticks, ticklabels = panel
        for name, points in [('DPPO', ('P0', 'D1', 'D2')), ('NC4', ('P0', 'N1', 'N2'))]:
            axis.plot([number(policy[p], 'theta_distance') for p in points],
                      [number(v4[p], metric) / divisor for p in points],
                      color=COLORS[name], linewidth=1.7, zorder=2)
        for point in POINTS:
            color = '#333333' if point == 'P0' else COLORS['DPPO' if point.startswith('D') else 'NC4']
            row = v4[point]
            y, lo, hi = [number(row, metric + suffix) / divisor for suffix in ('', '_low', '_high')]
            assert 0 < lo <= y <= hi
            x = number(policy[point], 'theta_distance')
            axis.errorbar([x], [y], yerr=[[y - lo], [hi - y]], color=color,
                          fmt='s' if point.startswith('N') else 'o', markersize=5.5,
                          capsize=3.5, linewidth=1.5, zorder=3)
            offset = (7, 5)
            if point == 'N1':
                offset = (-16, 14) if index == 0 else (7, 7)
            elif point == 'N2':
                offset = (8, -15) if index == 0 else (8, -5)
            elif point == 'D2':
                offset = (6, 7)
            axis.annotate(point, (x, y), xytext=offset, textcoords='offset points',
                          color=color, fontsize=10)
        axis.set_title(title, loc='left', pad=11, fontweight='bold')
        axis.set_yscale('log')
        axis.set_ylim(limits)
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FixedFormatter(ticklabels))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_xlim(-.25, 7.5)
        axis.set_xticks([0, 1, 3, 5, 7])
        axis.set_ylabel(ylabel)
        axis.set_xlabel('Actor distance from pretrained policy (L2)', labelpad=10)
        axis.grid(axis='y', color='#D9DEE4', linewidth=.7)
        axis.grid(axis='x', color='#E9EDF1', linewidth=.6)
        axis.set_axisbelow(True)
    fig.text(.065, .056, '90% group-jackknife intervals  •  28 groups × 50 batches × 80 episodes per point  •  Logarithmic y-axes',
             color='#4A4A4A', fontsize=10)
    fig.text(.065, .015, 'Each path uses training seed 0. Connecting lines join saved checkpoints; uncertainty is conditional on those checkpoints.',
             color='#4A4A4A', fontsize=9.5)
    figure_base = OUT / 'round2_signal_noise'
    fig.savefig(str(figure_base) + '.png', dpi=190)
    fig.savefig(str(figure_base) + '.pdf')
    plt.close(fig)

    caption = (
        'Round 2 measures the V4 mean discounted-return gradient and its noise at five fixed policies: '
        'P0 is pretrained; D1/D2 and N1/N2 are the middle/final DPPO and NC4 seed-0 checkpoints. '
        'The x-axis is actor-parameter L2 distance from P0; the logarithmic y-axes show gradient norm, '
        'episode-equivalent covariance trace (80 times whole-batch variance), and noise scale '
        'B = trace / squared signal, with V4 moments converted to paper gradient units. '
        'Each point uses 1,400 fresh 80-episode batches, grouped into 28 independent groups of 50; '
        'bars are the existing 90% delete-one-group jackknife intervals, and lines only connect measured checkpoints. '
        'Mean gradients and noise are smaller at the later policies, while B remains 58,000–94,000 episodes, '
        'far above the 80-episode training batch. '
        'The estimated mean-direction cosine is only 0.75–0.81, and these conditional single-path measurements '
        'do not establish that noise caused a training outcome.\n')
    (OUT / 'round2_signal_noise.caption.md').write_text(caption)
    table_caption = (
        'This table copies V4 estimates and 90% group-jackknife endpoints from the saved Round 2 analysis, '
        'adding raw discounted and undiscounted train-sampler returns and checkpoint distances from the same snapshot. '
        'u_sq is the noise-corrected squared mean-gradient norm; u_norm is its square root; trace_ep is '
        '80 times the covariance trace of an 80-episode batch gradient, in paper units. '
        'B_ep = trace_ep / u_sq; c estimates the full measurement-mean direction cosine, whereas expected_batch_cos '
        'describes one 80-episode training gradient; K_needed forecasts total measurement batches. '
        'All entries condition on the selected seed-0 checkpoints; uncertainty comes from the 28 groups, not independent training seeds.\n')
    (OUT / 'round2_estimates.caption.md').write_text(table_caption)
    record = dict(status='complete', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_run='round_1400', source_analysis_job=status['slurm_job_id'],
                  source_parent_commit=status['parent_commit'], inputs=[str(SOURCE / name) for name in
                  ('main_results.csv', 'per_point.csv', 'run_inventory.csv')],
                  new_scientific_samples=0, gpu_seconds=0, python=sys.version,
                  numpy=np.__version__, matplotlib=matplotlib.__version__,
                  verified=dict(unique_complete_collect_groups=140, groups_per_point=28,
                                batches_per_point=1400, episodes_per_point=112000,
                                all_v4_signals_resolved=True, unit_and_metric_identities=True),
                  code_snapshot=str(Path(__file__).resolve()), elapsed_seconds=time.monotonic() - started,
                  artifacts=[str(path) for path in sorted(OUT.iterdir())], curated=False)
    (RUN / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'elapsed_seconds': record['elapsed_seconds'],
                      'verified': record['verified'], 'candidates': str(OUT)}))


if __name__ == '__main__':
    main()
