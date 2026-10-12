"""Presentation revision from saved V5 groups only; compatible with Python 3.8."""
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from clean_v3_stats import _load_core, jackknife


def restore_point_estimates(statistics, root, analysis_source=None):
    """Retain ratio points; joint t-jackknife intervals with the requested open end.

    A denominator interval touching zero makes the upper B endpoint infinite.
    The lower endpoint is the joint ratio jackknife endpoint intersected with
    the physical domain B >= 0. This is not the earlier plug-in lower bound.
    """
    result = copy.deepcopy(statistics)
    source = analysis_source or statistics['runmetadata']['analysis_source']
    core, _ = _load_core(root, source)
    variants = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    for point in result['points'].values():
        if (point['K'], point['groups'], point['m']) != (500, 10, 40000):
            raise ValueError('Expected ten 50-batch groups at every point')
        data = core.PointData(point['collector_point'],
                              [group['path'] for group in point['selected_jobs']], variants=variants)
        for variant in ('V2', 'V6'):
            estimates = point['estimators'][variant]
            index = data.variants.index(variant)
            signal = estimates['u_sq']
            if signal['value'] <= 0:
                raise ValueError('Nonpositive squared-mean point estimate: ' + point['point'] + '/' + variant)
            multiplier = estimates['gradient_multiplier'] ** 2
            raw_signal = float(data.full_moments['signal'][index]) * multiplier
            np.testing.assert_allclose(raw_signal, signal['value'], rtol=1e-12, atol=0)
            raw_trace = float(data.full_moments['trace'][index]) * 80 * multiplier
            np.testing.assert_allclose(raw_trace, estimates['trace_ep']['value'], rtol=1e-12, atol=0)
            deleted = []
            for moment in data.deleted_moments:
                denominator = float(moment['signal'][index])
                if denominator <= 0:
                    raise ValueError('Undefined positive ratio in a delete-group replicate')
                deleted.append(float(80 * moment['trace'][index] / denominator))
            ratio = estimates['trace_ep']['value'] / signal['value']
            interval = jackknife(ratio, deleted)
            interval.update(signal_estimator=variant, variance_estimator=variant,
                            interval_method='joint delete-one-group Student-t; df=9',
                            unmodified_jackknife_lo=interval['lo'],
                            unmodified_jackknife_hi=interval['hi'])
            if signal['lo'] <= 0:
                interval.update(lo=max(0.0, interval['lo']), hi='inf',
                                status='unbounded_above',
                                reason='squared_signal_interval_reaches_zero')
            estimates['B_ep'] = interval
            per_update = copy.deepcopy(interval)
            for key in ('value', 'lo', 'hi', 'se', 'unmodified_jackknife_lo', 'unmodified_jackknife_hi'):
                if isinstance(per_update.get(key), (int, float)):
                    per_update[key] /= 80.0
            estimates['B_update'] = per_update
        point['predicted_consecutive_cos'] = {
            'value': 1.0 / (1.0 + point['estimators']['V2']['B_ep']['value'] / 80.0),
            'lo': None, 'hi': None, 'se': None, 'status': 'point_estimate',
            'definition': '1 / (1 + B_V2 / 80) using the ratio point estimate'}
    result['interval']['bound_note'] = (
        'B uses joint delete-group ratio t intervals. When the squared-signal interval reaches zero, '
        'the upper endpoint is infinite and the lower endpoint is intersected with B >= 0.')
    return result


def write_note(stats, output, renderer):
    f = renderer.fmt
    unresolved = {}
    for variant in ('V2', 'V6'):
        unresolved[variant] = [str(p['update_number']) for p in stats['points'].values()
                               if p['estimators'][variant]['u_sq']['lo'] <= 0]
    lines = ['# Gradient-noise figures: point estimates', '',
             'Both figures now show five filled-circle point estimates with identical two-decade '
             'log spans in every panel. All ten squared-mean point estimates are positive. '
             'The collection is unchanged: 500 batches / 40,000 episodes / ten groups per checkpoint.', '',
             '**Uncertainty.** The 90% squared-mean intervals reach zero at updates ' +
             ', '.join(unresolved['V2']) + ' for as run, and ' + ', '.join(unresolved['V6']) +
             ' for paper’s form. Their B estimates remain plotted; all six B intervals are [0, ∞). '
             'The table reports squared-mean intervals directly, including negative endpoints. '
             'Intervals use the same delete-one-group jackknife and t(0.95, 9) = 1.833113. '
             'For an unresolved denominator, the upper ratio endpoint is infinite and the joint-ratio '
             'lower endpoint is intersected with B ≥ 0; the previous positive plug-in bounds are not used.', '',
             '| Updates | Logged consecutive-gradient cosine ± s.e. | n | Prediction 1/(1 + B/80) | Measured critic EV (90% interval) | Logged critic EV |',
             '|---:|---:|---:|---:|---:|---:|']
    for p in stats['points'].values():
        log = p['logged_consecutive_cos']
        lines.append('| %d | %s ± %s | %d | %s | %s | %s |' %
                     (p['update_number'], f(log['mean'], 7), f(log['sample_se'], 7), log['n'],
                      f(p['predicted_consecutive_cos']['value'], 7),
                      renderer.metric_cell(p['explained_var']), f(p['logged_explained_var']['value'])))
    lines += ['',
              'Each prediction lies within one logged standard error. Windows are ±10 pre-rollout '
              'update indices, truncated at the endpoints; the first undefined cosine is excluded. '
              'These standard errors are descriptive because successive gradients can be dependent '
              'and training changes the policy and critic. Measured EV uses the training GAE-target '
              'definition; update 126 has no next logged rollout.', '',
              '**Scope and limitations.** Checkpoints state_0/35/70/105/139 correspond to updates '
              '0/32/63/95/126. The final checkpoint retains its authorized five-batch fitted reward '
              'scale; its gradient intervals condition on that scale. Full-batch normalization '
              'retains the previously passed comparison with the training quarters. '
              'Only B is comparable between the two estimators’ different units. '
              'No jobs were launched, no batches were added, and reward code, reward artifacts, '
              'raw collection results and curated copies were preserved. No deviations from this request.', '',
              '[As run](gradient_noise_as_run.png) · [Paper’s form](gradient_noise_paper_form.png) · '
              '[Estimates table](points_table.md) · [Selected batches and calculation record](render_record.json).']
    (output / 'NOTE.md').write_text('\n'.join(lines) + '\n')


def render(args, renderer):
    root = args.root.resolve()
    source_record = root / 'figs_clean/render_record.json'
    saved = json.loads(source_record.read_text())
    if set(saved['statistics']['points']) != set(renderer.POINTS):
        raise ValueError('Expected all five completed points')
    stats = restore_point_estimates(saved['statistics'], root, args.analysis_source)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run = root / 'runs' / ('gradient_point_estimates_' + stamp)
    candidates = run / 'candidates'
    candidates.mkdir(parents=True)
    shutil.copy2(str(source_record), str(run / 'source_record.json'))
    for path in (Path(__file__), Path(renderer.__file__), root / 'clean_v3_stats.py'):
        shutil.copy2(str(path), str(run / path.name))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white'})
    checks = []
    original_save = renderer.save_figure

    def checked_save(fig, output, name, caption, plotting):
        assert len(fig.axes) == 3
        for axis in fig.axes:
            assert axis.get_yscale() == 'log'
            assert len(axis.lines) == 1 and not axis.collections and not axis.containers
            line = axis.lines[0]
            assert line.get_marker() == 'o' and line.get_markerfacecolor() == renderer.GREEN
            np.testing.assert_array_equal(line.get_xdata(), renderer.UPDATES)
            assert len(line.get_ydata()) == 5 and np.all(np.asarray(line.get_ydata()) > 0)
            assert math.isclose(math.log10(axis.get_ylim()[1] / axis.get_ylim()[0]), 2.0, abs_tol=1e-12)
            expected_center = (np.log10(line.get_ydata()).min() + np.log10(line.get_ydata()).max()) / 2
            assert math.isclose(np.log10(axis.get_ylim()).mean(), expected_center, abs_tol=1e-12)
        fig.canvas.draw()
        boxes = [axis.get_tightbbox(fig.canvas.get_renderer()) for axis in fig.axes]
        assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:])), [box.bounds for box in boxes]
        checks.append({'figure': name, 'five_positive_filled_points_per_panel': True,
                       'error_bars_absent': True, 'centered_two_decade_spans': True,
                       'adjacent_axis_text_disjoint': True})
        original_save(fig, output, name, caption, plotting)

    renderer.save_figure = checked_save
    try:
        for variant in ('V2', 'V6'):
            renderer.plot_noise(stats, variant, candidates, plt)
    finally:
        renderer.save_figure = original_save
    rows = renderer.build_table(stats, saved['rewards'])
    renderer.write_tables(rows, candidates)
    write_note(stats, candidates, renderer)
    record = dict(saved)
    record.update(run=str(run), utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  statistics=stats, artifacts_scope='gradient point estimates, table and note only',
                  revision='point-estimates-only', source_record=str(run / 'source_record.json'),
                  source_record_original_path=str(source_record),
                  plot_source_snapshot=[str(run / Path(renderer.__file__).name), str(run / Path(__file__).name)],
                  presentation_checks=checks, reward_artifacts_rendered=False, curated=False)
    (candidates / 'render_record.json').write_text(json.dumps(renderer.json_safe(record), indent=2, allow_nan=False) + '\n')
    output = args.output.resolve() if args.output else root / 'figs_clean'
    if not args.candidates_only:
        renderer.publish(candidates, output)
    print(json.dumps({'status': 'complete', 'run': str(run), 'candidates': str(candidates),
                      'published': not args.candidates_only, 'checks': checks}, indent=2))
