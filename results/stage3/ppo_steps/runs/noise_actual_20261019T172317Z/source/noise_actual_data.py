"""Read saved estimates and convert per-episode quantities; Python 3.8."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GREEN = '#009E73'
KEYS = ('mean_norm', 'variance', 'relative_variance')


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=5e-13, abs_tol=1e-20):
        raise ValueError('Stored estimates disagree: %r versus %r' % (a, b))


def finish(row):
    if row['K'] != 500 or row['groups'] != 10:
        raise ValueError('Expected 500 batches in ten groups')
    if row['signal_sq'] <= 0:
        raise ValueError('Nonpositive mean-squared estimate')
    close(row['mean_norm'], math.sqrt(row['signal_sq']))
    close(row['relative_variance'], row['variance'] / row['signal_sq'])
    for key in KEYS:
        if not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError('Invalid log-axis point: ' + key)
    row['mean_resolved'] = row['signal_sq_lo'] > 0
    row['relative_variance_lower_bound'] = None
    if row['mean_resolved']:
        row['expected_cosine'] = (1 + row['relative_variance']) ** -0.5
        row['expected_cosine_relation'] = '='
    else:
        if row['signal_sq_hi'] <= 0:
            raise ValueError('Cannot form requested positive-denominator bound')
        bound = row['variance'] / row['signal_sq_hi']
        row['relative_variance_lower_bound'] = bound
        row['expected_cosine'] = (1 + bound) ** -0.5
        row['expected_cosine_relation'] = '<='
    return row


def load_rows():
    gm = ROOT / 'results/stage3/gradmean/figs_clean'
    ppo = ROOT / 'results/stage3/ppo_steps/candidates/final/ppo_steps_table.csv'
    record = json.loads((gm / 'render_record.json').read_text())
    table = list(csv.DictReader((gm / 'points_table.csv').open()))
    rows = []
    for r in table:
        p = record['statistics']['points'][r['point']]
        e = p['estimators']['V2']
        for field, prefix in [('u_sq', 'u_sq'), ('trace_ep', 'trace_ep'), ('B_ep', 'B_ep')]:
            for suffix, key in [('', 'value'), ('_lo', 'lo'), ('_hi', 'hi')]:
                a, b = float(r['as_run_' + prefix + suffix]), float(e[field][key])
                if math.isinf(a) and math.isinf(b):
                    continue
                close(a, b)
        close(r['as_run_u_norm'], e['u_norm']['value'])
        row = dict(method='without_PPO_and_GAE', update=int(r['update_number']), step=None,
                   checkpoint=r['checkpoint_file'], K=int(r['K']), groups=int(r['groups']),
                   mean_norm=float(r['as_run_u_norm']),
                   mean_norm_lo=float(e['u_norm']['lo']), mean_norm_hi=float(e['u_norm']['hi']),
                   signal_sq=float(r['as_run_u_sq']), signal_sq_lo=float(r['as_run_u_sq_lo']),
                   signal_sq_hi=float(r['as_run_u_sq_hi']),
                   scale_fallback=r['scale_fallback'] == 'True',
                   norm_interval_source='stored norm jackknife interval')
        for dest, src in [('variance', 'trace_ep'), ('relative_variance', 'B_ep')]:
            for suffix in ('', '_lo', '_hi'):
                row[dest + suffix] = float(r['as_run_' + src + suffix]) / 80
        for suffix in ('', '_lo', '_hi'):
            a, b = row['relative_variance' + suffix], float(r['as_run_B_update' + suffix])
            if not (math.isinf(a) and math.isinf(b)):
                close(a, b)
        rows.append(finish(row))
    source_rows = list(csv.DictReader(ppo.open()))
    for r in source_rows:
        if r['part'] != 'C':
            continue
        if r['status'] != 'complete' or r['method'] != 'DPPO':
            raise ValueError('Incomplete Part C point')
        row = dict(method='DPPO', update=int(r['updates']), step=int(r['step']),
                   checkpoint=r['checkpoint_file'], K=int(r['K']), groups=int(r['groups']),
                   mean_norm=float(r['u_norm']), mean_norm_lo=float(r['u_norm_lo']),
                   mean_norm_hi=float(r['u_norm_hi']), signal_sq=float(r['u_sq']),
                   signal_sq_lo=float(r['u_sq_lo']), signal_sq_hi=float(r['u_sq_hi']),
                   scale_fallback=False, norm_interval_source='stored transformed squared-mean interval')
        for dest, src in [('variance', 'trace_ep'), ('relative_variance', 'B')]:
            for suffix in ('', '_lo', '_hi'):
                row[dest + suffix] = float(r[src + suffix]) / 80
        rows.append(finish(row))
    if [r['update'] for r in rows if r['method'] != 'DPPO'] != [0, 32, 63, 95, 126]:
        raise ValueError('Unexpected as-run points')
    for update in (32, 63, 95):
        if sorted(r['step'] for r in rows if r['method'] == 'DPPO' and r['update'] == update) != list(range(1, 21)):
            raise ValueError('Missing or duplicate DPPO steps')
    if len(rows) != 65:
        raise ValueError('Expected 65 points')
    provenance = dict(sources=[str(gm / 'points_table.csv'), str(gm / 'render_record.json'), str(ppo)],
                      direct_batch_covariance_stored=False,
                      conversion='variance = trace_ep / 80; relative_variance = B / 80',
                      stored_B_update_check='all five points and interval endpoints agree',
                      t_critical=1.8331129326536335, df=9, interval_recomputed=False)
    return rows, provenance


def shared_limits(rows):
    result = {}
    for key in KEYS:
        low, high = min(r[key] for r in rows), max(r[key] for r in rows)
        bounds = [10.0 ** math.floor(math.log10(low) - .25),
                  10.0 ** math.ceil(math.log10(high) + .25)]
        assert math.log10(low / bounds[0]) >= .25 - 1e-12
        assert math.log10(bounds[1] / high) >= .25 - 1e-12
        result[key] = bounds
    return result


def fmt(value):
    if value is None:
        return '–'
    if math.isinf(value):
        return '∞' if value > 0 else '−∞'
    return '%.6g' % value


def unresolved_caption(rows):
    groups = {}
    for r in rows:
        if not r['mean_resolved']:
            groups.setdefault(r['update'], []).append(r)
    if not groups:
        return ' All displayed means are resolved by their 90% squared-mean intervals.'
    texts = []
    for update, entries in sorted(groups.items()):
        if entries[0]['step'] is None:
            texts.append('update %d: R ≥ %s' % (update, fmt(entries[0]['relative_variance_lower_bound'])))
        else:
            text = ', '.join('step %d: R ≥ %s' % (r['step'], fmt(r['relative_variance_lower_bound'])) for r in entries)
            texts.append('at %d updates, %s' % (update, text))
    return (' Means are not resolved at ' + '; '.join(texts) +
            '. The plotted ratios remain point estimates; only the listed lower bounds are supported when the mean is unresolved.')


def captions(rows):
    own = [r for r in rows if r['method'] != 'DPPO']
    dppo = [r for r in rows if r['method'] == 'DPPO']
    name = 'without_ppo_and_gae_noise_actual'
    result = {name: ("The gradient without_PPO_and_GAE passed to AdamW, at its own checkpoints: "
                    "its mean norm, the trace of its covariance across batches, and their ratio, "
                    "the relative variance of one 80-episode update (B/80 in the per-episode convention). "
                    "500 batches of 80 episodes per point, in training-loss units. "
                    "Points are estimates; 90% intervals are in noise_actual_table." + unresolved_caption(own) +
                    ' Update 126 retains the previously fitted reward scale; intervals condition on that scale. '
                    'The saved collection uses full-batch normalization, previously checked against the training quarters.')}
    for update in (None, 32, 63, 95):
        state = 'after 32, 63 and 95 updates (rows)' if update is None else 'after %d updates' % update
        name = 'dppo_per_step_noise_actual_grid' if update is None else 'dppo_per_step_noise_actual_u%d' % update
        selection = dppo if update is None else [r for r in dppo if r['update'] == update]
        result[name] = ("DPPO's gradient at each of the 20 PPO steps on one batch, from the run's own state "
                        + state + ": its mean norm, the trace of its covariance across batches, and their ratio, "
                        "the relative variance of one step's gradient (B_s/80). "
                        "500 batches of 80 episodes per checkpoint (10 groups), in training-loss units. "
                        "Points are estimates; 90% intervals are in noise_actual_table." + unresolved_caption(selection))
    result['dppo_per_step_noise_actual_grid_ref'] = result['dppo_per_step_noise_actual_grid'] + (
        ' Dashed: without_PPO_and_GAE\'s one-update value at the same update count, from its own run (different parameters).'
        + unresolved_caption([r for r in own if r['update'] in (32, 63, 95)]))
    return result


def table(rows, folder):
    fields = ['method', 'update', 'step', 'checkpoint', 'K', 'groups',
              'mean_norm', 'mean_norm_lo', 'mean_norm_hi', 'signal_sq', 'signal_sq_lo', 'signal_sq_hi',
              'variance', 'variance_lo', 'variance_hi', 'relative_variance', 'relative_variance_lo',
              'relative_variance_hi', 'mean_resolved', 'relative_variance_lower_bound',
              'expected_cosine', 'expected_cosine_relation', 'scale_fallback', 'norm_interval_source']
    with (folder / 'noise_actual_table.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out['step'] = '–' if row['step'] is None else row['step']
            writer.writerow(out)
    lines = ['# Gradient noise per optimizer gradient', '',
             'Each point uses 500 fresh batches of 80 episodes, grouped into ten jobs. '
             'Brackets retain the saved 90% delete-one-group jackknife intervals, with t(0.95,9) = 1.833113. '
             'Norms are in training-loss units, covariance traces in their squared units, and relative variance R is unitless. '
             'Variance and R intervals are the saved per-episode intervals divided by 80; no gradients or intervals were recomputed.', '',
             '| Run | Updates | PPO step | Mean norm [90% interval] | tr Cov(g) [90% interval] | R [90% interval] | Mean resolved? | R lower bound | Expected cosine / upper bound |',
             '|---|---:|---:|---|---|---|---|---|---|']
    for row in rows:
        def interval(key):
            return '%s [%s, %s]' % (fmt(row[key]), fmt(row[key + '_lo']), fmt(row[key + '_hi']))
        cosine = ('≤ ' if row['expected_cosine_relation'] == '<=' else '') + fmt(row['expected_cosine'])
        lines.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s |' % (
            row['method'], row['update'], '–' if row['step'] is None else row['step'],
            interval('mean_norm'), interval('variance'), interval('relative_variance'),
            'yes' if row['mean_resolved'] else 'no', fmt(row['relative_variance_lower_bound']), cosine))
    lines += ['',
              'Resolution uses the signed interval for the squared mean (CSV), not the norm interval. '
              'The saved as-run norm intervals are direct norm jackknife intervals; the saved DPPO norm intervals transform the squared-mean endpoints. '
              'Both are retained, including the negative signed norm lower endpoint at update 126.', '',
              'For unresolved means, the requested plug-in bound is R ≥ tr Cov(g) / (squared-mean upper endpoint). '
              'It treats variance at its point estimate and is separate from the joint ratio interval, which remains [0, ∞). '
              'Because cosine decreases with R, this gives an upper bound on (1+R)^(-1/2), not a resolved cosine estimate.', '',
              'The without_PPO_and_GAE point at update 126 retains its fitted reward scale; its intervals condition on that scale. '
              'The saved collection uses full-batch normalization, previously checked against the training quarters. '
              'Cross-method reference lines compare different policies at equal update counts, not identical parameters.', '']
    (folder / 'noise_actual_table.md').write_text('\n'.join(lines))
