#!/usr/bin/env python
"""DL1's approved 2x2 comparison, reading saved results only (Python 3.8).

This script never launches jobs, samples policies, imports training code, or
writes reference results. Final publishing requires all new runs to be verified.
Missing historical measurements remain missing; older reruns are not substituted.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import re
import shutil
import sys

import numpy as np

METHODS = ('DPPO', 'without_PPO', 'without_PPO_and_GAE', 'without_GAE')
COLORS = dict(zip(METHODS, ('#0072B2', '#D55E00', '#009E73', '#CC79A7')))
CHECKPOINTS = (0, 35, 70, 105, 139)
METRICS = (
    'eval_936M', 'eval_last5', 'train_last10', 'J_disc_gain_last10',
    'theta_dist_final', 'mean_kl_true_per_action', 'saturation_final',
    'mean_clipfrac', 'mean_switched_off_fraction', 'mean_approx_kl')
HEADINGS = {
    'eval_936M': 'Eval at 9.36M', 'eval_last5': 'Last five evals',
    'train_last10': 'Train last 10 updates', 'J_disc_gain_last10': 'J_disc gain, last 10',
    'theta_dist_final': 'Final distance from own start',
    'mean_kl_true_per_action': 'Mean true-KL diagnostic per update',
    'saturation_final': 'Final rollout saturation', 'mean_clipfrac': 'Mean clipfrac',
    'mean_switched_off_fraction': 'Mean switched-off fraction',
    'mean_approx_kl': 'Mean PPO approximate KL per update'}
CONTRAST_METRICS = ('eval_last5', 'train_last10', 'J_disc_gain_last10')


def finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def number(value):
    return float(value) if finite(value) else None


def fmt(value, digits=5):
    return ('%.' + str(digits) + 'g') % value if finite(value) else 'not logged'


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def write_json(path, value):
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(safe(rows))


def markdown(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def before(itr):
    return sum(i % 10 != 0 for i in range(int(itr)))


def strict_mean(values):
    return float(np.mean(values, dtype=np.float64)) if values and all(finite(x) for x in values) else None


def sample_stats(values):
    if len(values) != 3 or not all(finite(x) for x in values):
        return None, None
    return float(np.mean(values, dtype=np.float64)), float(np.std(values, ddof=1, dtype=np.float64))


def logdir(project, root, method, seed):
    if method == 'without_GAE':
        return root / ('DL1_seed%d' % seed)
    if method == 'without_PPO_and_GAE':
        return project / 'results/stage3/e1c' / ('E1C_seed%d' % seed)
    if method == 'without_PPO':
        return project / 'results/stage2_noclip/runs' / ('NC4_lr1e-4_seed%d' % seed) / 'native'
    run = 'seed0_handoff' if seed == 0 else 'seed%d' % seed
    manifest = read_json(project / 'results/repro_halfcheetah/runs' / run / 'manifest.json', {})
    if 'logdir' not in manifest:
        raise ValueError('Missing original DPPO manifest for seed %d' % seed)
    return Path(manifest['logdir'])


def read_dppo_step_kl(directory):
    """Original DPPO logs every pre-step approx_kl, then its iteration summary."""
    path = directory / 'run.log'
    if not path.is_file():
        return {}, {}
    values, by_itr, counts = [], {}, {}
    scalar = re.compile(r'approx_kl: ([-+0-9.eE]+), update_epoch: (\d+), num_batch: (\d+)')
    summary = re.compile(r' - (\d+): step\s+\d+\s*\|')
    for line in path.read_text(errors='replace').splitlines():
        match = scalar.search(line)
        if match:
            values.append(float(match.group(1)))
        match = summary.search(line)
        if match:
            itr = int(match.group(1))
            if itr in by_itr:
                raise ValueError('Duplicate original DPPO iteration in ' + str(path))
            if values:
                by_itr[itr] = strict_mean(values)
                counts[itr] = len(values)
            values = []
    return by_itr, counts


def overlay_sidecar(directory, rows):
    """Optional inert DL1 measurements stored separately from native results."""
    path = directory / 'dl1_diagnostics.jsonl'
    if not path.is_file():
        return
    by_itr = {int(row['itr']): row for row in rows}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        itr = int(record['itr'])
        if itr in by_itr:
            for key, value in record.items():
                if key in ('itr', 'step', 'train_episode_reward', 'eval_episode_reward'):
                    continue
                by_itr[itr][key] = value


def canonical_metrics(row):
    """Keep within-update averages distinct from the last minibatch values."""
    steps = row.get('ppo_steps') or row.get('dl1_ppo_steps')
    if isinstance(steps, list) and steps:
        for name, alternatives in {
                'ppo_clipfrac_mean': ('clipfrac',),
                'ppo_switched_off_mean': ('switched_off_fraction', 'switched_off_frac'),
                'ppo_approx_kl_mean': ('approx_kl',),
                'ppo_mean_abs_ratio_minus_one': ('mean_abs_ratio_minus_one',)}.items():
            vals = [next((step[k] for k in alternatives if k in step), None) for step in steps]
            row[name] = strict_mean(vals)
        row['ppo_steps_recorded'] = len(steps)
    aliases = {
        'ppo_clipfrac_mean': ('dl1_clipfrac_mean', 'ppo_clipfrac', 'clipfrac_mean', 'clipfrac'),
        'ppo_switched_off_mean': ('dl1_switched_off_mean', 'ppo_switched_off_fraction',
                                  'switched_off_fraction_mean', 'switched_off_fraction', 'switched_off_frac'),
        'ppo_approx_kl_mean': ('dl1_approx_kl_mean', 'approx_kl_mean'),
        'theta_dist_before_update': ('theta_dist_before',)}
    for key, alternatives in aliases.items():
        if key not in row:
            for other in alternatives:
                if other in row:
                    row[key] = row[other]
                    break


def load_run(project, root, method, seed):
    directory = logdir(project, root, method, seed)
    run = dict(method=method, seed=seed, run_dir=str(directory), result_path=str(directory / 'result.pkl'),
               rows=[], training=[], evaluation=[], status='missing', errors=[], checkpoints={})
    path = directory / 'result.pkl'
    if not path.is_file():
        return run
    try:
        with path.open('rb') as stream:
            rows = pickle.load(stream)
    except (EOFError, OSError, pickle.UnpicklingError) as error:
        run.update(status='temporarily_unreadable', errors=[repr(error)])
        return run
    if not isinstance(rows, list) or [r.get('itr') for r in rows] != list(range(len(rows))) or len(rows) > 140:
        raise ValueError('Invalid iteration sequence: ' + str(path))
    rows = [dict(row) for row in rows]
    if method == 'without_GAE':
        overlay_sidecar(directory, rows)
    kl, step_counts = read_dppo_step_kl(directory) if method == 'DPPO' else ({}, {})
    prev_distance = 0.0
    for row in rows:
        itr = int(row['itr'])
        train = 'train_episode_reward' in row
        if train != (itr % 10 != 0):
            raise ValueError('Unexpected training/evaluation schedule: ' + str(path))
        applied = before(itr)
        if row['step'] != 80000 * (applied + int(train)):
            raise ValueError('Environment-step/update mapping mismatch: ' + str(path))
        row['updates_before_rollout'] = applied
        row['updates_after_iteration'] = applied + int(train)
        canonical_metrics(row)
        if method == 'DPPO' and itr in kl:
            row['ppo_approx_kl_mean'] = kl[itr]
            row['ppo_steps_recorded'] = step_counts[itr]
        if train:
            # A rollout uses theta before the step; never pair it with same-row
            # theta_dist, which is measured after that step.
            row['rollout_theta_dist'] = number(row.get('theta_dist_before_update', prev_distance))
            prev_distance = number(row.get('theta_dist'))
        else:
            row['rollout_theta_dist'] = prev_distance
        required = ('train_episode_reward',) if train else ('eval_episode_reward',)
        for key in required:
            if not finite(row.get(key)):
                run['errors'].append('itr%d nonfinite %s' % (itr, key))
    checkpoints = {str(itr): str(directory / 'checkpoint' / ('state_%d.pt' % itr))
                   for itr in CHECKPOINTS if (directory / 'checkpoint' / ('state_%d.pt' % itr)).is_file()}
    native_complete = len(rows) == 140 and len(checkpoints) == 5
    worker_path = root / 'runs/campaign_training/workers' / ('DL1_seed%d' % seed) / 'manifest.json'
    worker = read_json(worker_path, {}) if method == 'without_GAE' else {}
    verified = native_complete
    if method == 'without_GAE':
        verified = (native_complete and worker.get('status') == 'complete' and worker.get('exit_code') == 0
                    and worker.get('child_processes_after') == [])
    status = 'complete' if verified else ('awaiting_verification' if native_complete else 'partial')
    if worker.get('status') in ('failed', 'timeout', 'cancelled') or (directory / 'failure.json').is_file():
        status = 'failed'
    run.update(rows=rows, training=[r for r in rows if r['itr'] % 10],
               evaluation=[r for r in rows if not r['itr'] % 10], status=status,
               checkpoints=checkpoints, verified_complete=verified,
               last_itr=rows[-1]['itr'] if rows else None, worker_manifest=str(worker_path) if worker else None,
               worker=worker)
    return run


def checkpoint_distances(run, records):
    """Read CPU weights, subtract this run's state_0 in float64; no inference."""
    if not run['checkpoints'] or '0' not in run['checkpoints']:
        return {}
    import torch
    torch.set_num_threads(1)
    def actor(path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = checkpoint['model']
        state = {name[len('actor_ft.'):]: tensor for name, tensor in model.items()
                 if name.startswith('actor_ft.')}
        if sum(t.numel() for t in state.values()) != 568392:
            raise ValueError('Unexpected actor geometry: ' + path)
        return state
    origin = actor(run['checkpoints']['0'])
    distances = {}
    for itr_text, path in run['checkpoints'].items():
        itr = int(itr_text)
        state = origin if itr == 0 else actor(path)
        if set(state) != set(origin):
            raise ValueError('Actor parameter names differ from state_0')
        total = 0.0
        for name in sorted(state):
            if state[name].shape != origin[name].shape:
                raise ValueError('Actor geometry changed')
            difference = state[name].double() - origin[name].double()
            total += float(torch.sum(difference * difference).item())
        distance = math.sqrt(total)
        if not finite(distance):
            raise ValueError('Nonfinite checkpoint distance: ' + path)
        update = before(itr) + int(itr % 10 != 0)
        distances[update] = distance
        records.append(dict(method=run['method'], seed=run['seed'], checkpoint=path,
                            checkpoint_itr=itr, update=update, theta_dist=distance,
                            reference_checkpoint=run['checkpoints']['0']))
    return distances


def seed_metrics(run, distances):
    metrics = {key: None for key in METRICS}
    training = run['training']
    first = training[0] if training else {}
    metrics['initial_J_disc_train'] = number(first.get('J_disc_train'))
    if run['status'] == 'complete' and len(run['rows']) == 140:
        eval_by_itr = {row['itr']: row for row in run['evaluation']}
        metrics['eval_936M'] = number(eval_by_itr[130].get('eval_episode_reward'))
        metrics['eval_last5'] = strict_mean([row.get('eval_episode_reward') for row in run['evaluation'][-5:]])
        metrics['train_last10'] = strict_mean([row.get('train_episode_reward') for row in training[-10:]])
        last_j = strict_mean([row.get('J_disc_train') for row in training[-10:]])
        if finite(last_j) and finite(metrics['initial_J_disc_train']):
            metrics['J_disc_gain_last10'] = last_j - metrics['initial_J_disc_train']
        metrics['theta_dist_final'] = distances.get(126, number(training[-1].get('theta_dist')))
        metrics['mean_kl_true_per_action'] = strict_mean([row.get('kl_true_per_action') for row in training])
        metrics['saturation_final'] = number(training[-1].get('x0_sat_frac_mean'))
        if run['method'] in ('DPPO', 'without_GAE'):
            for dest, key in (('mean_clipfrac', 'ppo_clipfrac_mean'),
                              ('mean_switched_off_fraction', 'ppo_switched_off_mean'),
                              ('mean_approx_kl', 'ppo_approx_kl_mean')):
                metrics[dest] = strict_mean([row.get(key) for row in training])
    return dict(method=run['method'], seed=run['seed'], status=run['status'],
                last_itr=run.get('last_itr'), **metrics)


def condition_metrics(seed_rows):
    result = []
    for method in METHODS:
        rows = [r for r in seed_rows if r['method'] == method]
        output = dict(method=method)
        for metric in METRICS:
            values = [r[metric] for r in rows]
            mean, sd = sample_stats(values)
            output.update({metric: mean, metric + '_sd': sd, metric + '_seeds': values})
        result.append(output)
    return result


def contrasts(condition_rows):
    by_method = {row['method']: row for row in condition_rows}
    pairs = (
        ('GAE under reuse', {'DPPO': 1, 'without_GAE': -1}),
        ('GAE without reuse', {'without_PPO': 1, 'without_PPO_and_GAE': -1}),
        ('reuse under GAE', {'DPPO': 1, 'without_PPO': -1}),
        ('reuse under Monte Carlo', {'without_GAE': 1, 'without_PPO_and_GAE': -1}),
        ('interaction', {'DPPO': 1, 'without_PPO': -1, 'without_GAE': -1, 'without_PPO_and_GAE': 1}))
    output = []
    for name, coefficients in pairs:
        for metric in CONTRAST_METRICS:
            available = all(finite(by_method[m][metric]) and finite(by_method[m][metric + '_sd'])
                            for m in coefficients)
            difference = sum(coef * by_method[m][metric] for m, coef in coefficients.items()) if available else None
            # Interaction: pooled SD of two independent reuse contrasts,
            # sqrt(sum of four seed variances / 2), with no paired covariance.
            pooled = math.sqrt(sum(by_method[m][metric + '_sd'] ** 2 for m in coefficients) / 2) if available else None
            interpretation = ('unresolved' if abs(difference) < 2 * pooled else 'exceeds reading-rule threshold') if available else 'not logged'
            output.append(dict(contrast=name, metric=HEADINGS[metric], metric_key=metric,
                               difference=difference, pooled_seed_sd=pooled,
                               difference_se=pooled * math.sqrt(2 / 3) if available else None,
                               reading=interpretation,
                               formula=' + '.join(('%+d × %s' % (c, m)) for m, c in coefficients.items())))
    return output


def series(run, kind):
    key = {'eval': 'eval_episode_reward', 'train': 'train_episode_reward', 'J': 'J_disc_train',
           'clip': 'ppo_clipfrac_mean', 'off': 'ppo_switched_off_mean', 'kl': 'ppo_approx_kl_mean'}[kind]
    records = run['evaluation'] if kind == 'eval' else run['training']
    return {row['updates_before_rollout']: float(row[key]) for row in records if finite(row.get(key))}


def distance_series(run, checkpoint_values):
    result = {}
    for row in run['training']:
        update = row['updates_before_rollout']
        if not finite(row.get('J_disc_train')):
            continue
        distance = row.get('rollout_theta_dist')
        if not finite(distance):
            distance = checkpoint_values.get(update)
        if finite(distance):
            result[update] = (float(distance), float(row['J_disc_train']))
    return result


def across_seeds(runs, method, getter, three_seed):
    seeds = (0, 1, 2) if three_seed else (0,)
    data = [getter(runs[(method, seed)]) for seed in seeds]
    common = sorted(set.intersection(*(set(row) for row in data))) if data else []
    return common, [np.mean([row[x] for row in data], axis=0) for x in common]


def save_figure(fig, folder, name, caption):
    fig.savefig(str(folder / (name + '.png')), dpi=300, bbox_inches='tight')
    fig.savefig(str(folder / (name + '.pdf')), bbox_inches='tight', metadata={'Creator': 'DL1 saved-results analysis'})
    (folder / (name + '.caption.md')).write_text(caption + '\n')


def plots(runs, distances, output, interim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    folder = output / 'figs'
    folder.mkdir()
    inventory = []
    def style(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.5)
        ax.set_axisbelow(True)
    for three_seed in (False, True):
        suffix = '3seed' if three_seed else 'seed0'
        fig, axes = plt.subplots(3, 1, figsize=(6.1, 7.5), sharex=True)
        for ax, kind, title, unit in zip(axes, ('eval', 'train', 'J'),
                ('Evaluation reward', 'Training return', 'Discounted training return'),
                ('Undiscounted return', 'Undiscounted return', 'J_disc (raw reward units)')):
            for method in METHODS:
                xs, ys = across_seeds(runs, method, lambda run: series(run, kind), three_seed)
                if xs:
                    ax.plot(xs, ys, color=COLORS[method], marker='o', markersize=2.5,
                            linewidth=1.05, label=method)
            ax.set_title(title, fontsize=9)
            style(ax, 'Updates (one batch = 80 episodes)', unit)
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        sampling = 'Three-seed means' if three_seed else 'Seed 0 of each method'
        caption = (sampling + '; filled circles are point estimates, with no uncertainty shown. '
                   'The horizontal axis counts actor updates before the rollout, each consuming 80 episodes; '
                   'evaluation keeps the DDPM noise except at the last denoising step, while training uses a final-step noise floor of 0.1. '
                   'J_disc is the raw-reward sum discounted by 0.99 per four-environment-step decision. '
                   'DPPO and without_PPO did not log J_disc, so they have no discounted-return curve; all reward curves use the original runs. '
                   'No smoothing or replacement runs are used.' +
                   (' This is an interim display; three-seed curves end at the latest update shared by all three seeds.' if interim else ''))
        name = 'ladder2x2_curves_' + suffix
        save_figure(fig, folder, name, caption)
        plt.close(fig)
        inventory.append(name)

        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        for method in METHODS:
            def get(run):
                return distance_series(run, distances[(run['method'], run['seed'])])
            xs, pairs = across_seeds(runs, method, get, three_seed)
            if pairs:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], color=COLORS[method],
                        marker='o', markersize=2.8, linewidth=1.05, label=method)
        style(ax, 'Distance from own start ||θ − θ₀|| (parameter units)', 'J_disc (raw reward units)')
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        caption = (sampling + '; no uncertainty shown. Each point pairs the rollout\'s discounted training return '
                   'with the actor distance before that rollout; the same iteration\'s post-update distance is not used. '
                   'For a checkpoint-only distance record, only an exactly matching rollout update is eligible. '
                   'DPPO and without_PPO have saved checkpoints but no training J_disc, so neither can be placed on this axis. '
                   'Consequently this figure cannot establish the direction-versus-movement comparison across all four methods. '
                   'For three-seed curves both coordinates are averaged at matching update indices, rather than interpolating returns at matched distances.')
        name = 'ladder2x2_distance_' + suffix
        save_figure(fig, folder, name, caption)
        plt.close(fig)
        inventory.append(name)

    fig, axes = plt.subplots(3, 1, figsize=(6.1, 7.2), sharex=True)
    for ax, kind, title, unit in zip(axes, ('clip', 'off', 'kl'),
            ('Ratio clip fraction', 'Pairs switched off by the clip', 'Approximate KL'),
            ('Fraction of pairs', 'Fraction of pairs', 'PPO approximate KL (dimensionless)')):
        for method in ('without_GAE', 'DPPO'):
            xs, ys = across_seeds(runs, method, lambda run: series(run, kind), False)
            if xs:
                ax.plot(xs, ys, color=COLORS[method], marker='o', markersize=2.6,
                        linewidth=1.05, label=method)
        style(ax, 'Updates (one batch = 80 episodes)', unit)
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    caption = ('Seed 0; each point averages the logged minibatch quantities within one update, normally 20 steps. '
               'A pair is switched off when its normalized, denoising-weighted advantage is positive and its ratio exceeds 1+ε, '
               'or its advantage is negative and its ratio falls below 1−ε. '
               'DPPO retained per-step approximate KL in its original console log, but neither clipfrac nor switched-off fraction was saved with W&B disabled; '
               'those two reference curves are therefore absent. Approximate KL uses PPO\'s clamped, coordinate-mean log-prob ratio and is distinct from the post-update true-KL diagnostic. '
               'Points only; no smoothing or uncertainty shown.')
    save_figure(fig, folder, 'without_gae_ppo_steps', caption)
    plt.close(fig)
    inventory.append('without_gae_ppo_steps')
    return inventory


def write_tables(seed_rows, conditions, comparison, output):
    rows = []
    missing_labels = {method: ('not logged' if all(
        row['status'] == 'complete' and row.get('last_itr') == 139
        for row in seed_rows if row['method'] == method) else 'pending completion')
        for method in METHODS}
    for method in METHODS:
        for row in [r for r in seed_rows if r['method'] == method]:
            for metric in METRICS:
                rows.append(dict(method=method, aggregation='seed %d' % row['seed'], metric=HEADINGS[metric],
                                 value=row[metric], sd=None, seed0=None, seed1=None, seed2=None,
                                 status='available' if finite(row[metric]) else
                                 ('not logged' if row['status'] == 'complete' and row.get('last_itr') == 139 else 'pending completion')))
        cond = next(r for r in conditions if r['method'] == method)
        for metric in METRICS:
            vals = cond[metric + '_seeds']
            rows.append(dict(method=method, aggregation='three-seed mean', metric=HEADINGS[metric],
                             value=cond[metric], sd=cond[metric + '_sd'], seed0=vals[0], seed1=vals[1], seed2=vals[2],
                             status='available' if finite(cond[metric]) else missing_labels[method]))
    write_csv(output / 'ladder2x2_table.csv', rows)
    write_csv(output / 'seed_metrics.csv', seed_rows)
    write_csv(output / 'contrasts.csv', comparison)
    lines = ['# Two-by-two reward comparison', '',
             'Each cell is the mean ± sample SD (ddof=1) over three training seeds, followed by seeds 0, 1, 2. '
             'Missing measurements are not replaced by other runs. Evaluation at 9.36M is iteration 130; '
             'the last five evals are iterations 90–130, and the last ten training updates are iterations 129 and 131–139. '
             'Each discounted-return gain subtracts that seed\'s first training batch (iteration 1, before any actor update).', '']
    # Short metric tables avoid one very wide, unreadable display.
    for metric in METRICS:
        lines += ['## ' + HEADINGS[metric], '', markdown(['Method', 'Mean ± SD; seed 0 / 1 / 2'], [
            [row['method'], (fmt(row[metric]) + ' ± ' + fmt(row[metric + '_sd']) + '; ' +
                            ' / '.join(fmt(v) for v in row[metric + '_seeds']))
             if finite(row[metric]) else missing_labels[row['method']]] for row in conditions]), '']
    lines += ['True-KL diagnostics are sampled coordinate-summed, unclamped post-update transition KL estimates; '
              'PPO approximate KL is the within-update, clamped coordinate-mean statistic. Their columns are intentionally separate. '
              'Final saturation comes from the last training rollout (before update 126), while final parameter distance is after update 126. '
              'Checkpoint distances use float64 differences from each run\'s own state_0.', '']
    (output / 'ladder2x2_table.md').write_text('\n'.join(lines))
    c_lines = ['# Contrasts', '',
               'For each ordinary contrast, pooled seed SD is sqrt((s_A²+s_B²)/2). '
               'For the interaction it is sqrt((s_DPPO²+s_without_PPO²+s_without_GAE²+s_without_PPO_and_GAE²)/2), '
               'the pooled SD of two independent reuse contrasts; no paired-seed covariance is assumed. '
               'A difference smaller in magnitude than twice this SD is called unresolved. '
               'SE is also recorded in CSV as pooled SD × sqrt(2/3), without an additional significance test.', '',
               markdown(['Contrast', 'Metric', 'Difference', 'Pooled seed SD', 'Reading'],
                        [[r['contrast'], r['metric'], fmt(r['difference']), fmt(r['pooled_seed_sd']), r['reading']]
                         for r in comparison]), '']
    (output / 'contrasts.md').write_text('\n'.join(c_lines))


def progress_table(runs):
    rows = []
    for method in METHODS:
        for seed in range(3):
            run = runs[(method, seed)]
            last_train = run['training'][-1] if run['training'] else {}
            last_eval = run['evaluation'][-1] if run['evaluation'] else {}
            rows.append(dict(method=method, seed=seed, status=run['status'], last_itr=run.get('last_itr'),
                             last_train_update=last_train.get('updates_before_rollout'),
                             train_return=number(last_train.get('train_episode_reward')),
                             J_disc=number(last_train.get('J_disc_train')),
                             last_eval_update=last_eval.get('updates_before_rollout'),
                             eval_return=number(last_eval.get('eval_episode_reward'))))
    return rows


def gate_record(root):
    records = []
    for path in sorted((root / 'runs/gates').glob('**/gate_summary.json')):
        value = read_json(path, {})
        if value.get('phase') == 'merge' and value.get('passed'):
            records.append((path, value))
    if not records:
        return {'passed': False, 'status': 'not available', 'checks': {}}
    path, value = records[-1]
    return dict(value, summary_path=str(path))


def complete_bookkeeping(root, project, conditions):
    """Administrative completion only; never creates a curated folder."""
    required = [root / name for name in ('REPORT.md', 'analysis_record.json', 'methods.md',
                                        'ladder2x2_table.csv', 'ladder2x2_table.md')]
    if not all(path.is_file() for path in required):
        raise ValueError('Required final artifacts missing; refusing completion bookkeeping')
    path = root / 'experiment.md'
    text = path.read_text()
    current = next(row for row in conditions if row['method'] == 'without_GAE')
    summary = ('**Status: completed.** Three training seeds finished all 140 iterations and 126 fresh-data updates; '
               'the final read-only comparison is complete. without_GAE evaluation at 9.36M is %s ± %s and '
               'last-ten-update training return is %s ± %s (mean ± sample SD over seeds). '
               '[Report](REPORT.md) · [Results table](ladder2x2_table.md).') % (
                   fmt(current['eval_936M']), fmt(current['eval_936M_sd']),
                   fmt(current['train_last10']), fmt(current['train_last10_sd']))
    text = re.sub(r'(?m)^\*\*Status[^\n]*(?:\n(?!\n)[^\n]*)*', summary, text, count=1)
    text = re.sub(r'(?m)^\*\*Next step\.\*\*[^\n]*',
                  '**Next step.** Review the five figure candidates, table and report for keep/revise/leave-uncurated decisions. No additional sampling is authorized.', text, count=1)
    path.write_text(text)
    checklist = project / 'complete_experiment.md'
    text = checklist.read_text()
    title = '### DPPO with Monte Carlo critic advantages (without_GAE)'
    if title not in text:
        curated = ('[Artifact index](./results/stage3/dl1/curated/INDEX.md)'
                   if (root / 'curated/INDEX.md').is_file() else '—')
        block = (title + '\n\n- Experiment folder: [without_GAE](./results/stage3/dl1/)\n'
                 '- Experiment design: [Methods and execution protocol](./results/stage3/dl1/methods.md)\n'
                 '- Curated results: ' + curated + '\n\n')
        marker = '## Comparison analyses'
        if marker not in text:
            raise ValueError('Completed-experiment checklist has no comparison section')
        text = text.replace(marker, block + marker, 1)
    comparison_title = '### PPO pieces and advantage estimator: two-by-two comparison'
    if comparison_title not in text:
        block = (comparison_title + '\n\n'
                 '- Analysis folder: [Original references and without_GAE comparison](./results/stage3/dl1/)\n'
                 '- Analysis design: [Comparison methods](./results/stage3/dl1/methods.md)\n'
                 '- Curated results: ' + ('[Artifact index](./results/stage3/dl1/curated/INDEX.md)'
                    if (root / 'curated/INDEX.md').is_file() else '—') + '\n\n')
        text = text.replace('## Comparison analyses\n\n', '## Comparison analyses\n\n' + block, 1)
    checklist.write_text(text)


def gate_differences(name, value):
    if name == 'G2':
        raw = value.get('dl1_formula', {}).get('raw_comparison', {})
        return [fmt(raw.get('max_abs')), fmt(raw.get('relative_l2')) + ' (relative L2)']
    return [fmt(value.get('max_abs')), fmt(value.get('max_rel'))]


def report(root, output, runs, conditions, comparison, interim, inventory):
    by_method = {r['method']: r for r in conditions}
    gates = gate_record(root)
    revision = read_json(root / 'provenance/execution_revision.json', {})
    def comparison_text(name, metric):
        row = next(r for r in comparison if r['contrast'] == name and r['metric_key'] == metric)
        return ('difference %s; pooled seed SD %s; %s' %
                (fmt(row['difference']), fmt(row['pooled_seed_sd']), row['reading']))
    lines = ['# DPPO with the Monte Carlo advantage', '',
             '**Question.** Does replacing GAE with the Monte Carlo return minus the critic change performance when the PPO pieces are retained?', '',
             '**Status.** ' + ('Interim; the requested training seeds are not all complete and verified.' if interim else
             'All three new training seeds are complete and verified. The original nine reference runs were reused without new sampling.'), '',
             'The reward comparisons fill the missing two-by-two cell. Historical logging limits the mechanistic conclusions: '
             'DPPO and without_PPO have no training J_disc, and original DPPO has neither saved clipfrac nor the switched-off fraction. '
             'No missing quantity is inferred from an unrequested rerun or from an unmatched sampler.', '',
             '[Full per-seed table](ladder2x2_table.md) · [Contrasts](contrasts.md) · [Data and commands](analysis_record.json)', '',
             '![Seed-0 reward curves](figs/ladder2x2_curves_seed0.png)', '',
             '## Q1. GAE under reuse', '',
             'DPPO minus without_GAE: last-five-evaluation ' + comparison_text('GAE under reuse', 'eval_last5') +
             '; training return ' + comparison_text('GAE under reuse', 'train_last10') +
             '. Discounted-return gain cannot be compared because DPPO did not log it.', '',
             '## Q2. Reuse and the interaction', '',
             'Reuse under Monte Carlo: last-five-evaluation ' + comparison_text('reuse under Monte Carlo', 'eval_last5') +
             '; training return ' + comparison_text('reuse under Monte Carlo', 'train_last10') + '.', '',
             'For discounted-return gain, without_GAE minus without_PPO_and_GAE: ' +
             comparison_text('reuse under Monte Carlo', 'J_disc_gain_last10') + '.', '',
             'Reuse under GAE: last-five-evaluation ' + comparison_text('reuse under GAE', 'eval_last5') +
             '; training return ' + comparison_text('reuse under GAE', 'train_last10') + '.', '',
             'The interaction is reuse under GAE minus reuse under Monte Carlo: last-five-evaluation ' +
             comparison_text('interaction', 'eval_last5') + '; training return ' + comparison_text('interaction', 'train_last10') +
             '. Its J_disc counterpart is not logged. “Reuse” here changes the full PPO bundle (step reuse, ratio clip, '
             'KL stop, log-prob clamp and sampling-noise truncation); the contrast does not isolate reuse alone.', '',
             '## Q3. Direction versus movement', '',
             'The complete four-method question is not identifiable from the existing records: original DPPO and without_PPO '
             'did not save J_disc. Their parameter distances can be recovered from checkpoints, but distance alone cannot supply '
             'their missing vertical coordinate. The distance plots show only genuinely logged J_disc paired with the pre-rollout '
             'parameter distance, without interpolating or substituting measurements. Mean final distances are ' +
             '; '.join('%s %s' % (m, fmt(by_method[m]['theta_dist_final'])) for m in METHODS) + '.', '',
             '## Q4. How hard the PPO clip works', '',
             'For without_GAE, mean clipfrac is %s and mean switched-off fraction is %s. '
             'Both historical DPPO quantities are not logged, preventing that direct comparison. '
             'Mean within-update PPO approximate KL is %s for without_GAE versus %s for DPPO; '
             'the separate true-KL column must not be compared to this PPO statistic.' %
             (fmt(by_method['without_GAE']['mean_clipfrac']), fmt(by_method['without_GAE']['mean_switched_off_fraction']),
              fmt(by_method['without_GAE']['mean_approx_kl']), fmt(by_method['DPPO']['mean_approx_kl'])), '',
             '## Setup and validity', '',
             'Three seeds per method, 140 iterations, 126 fresh-data updates, 80 episodes per batch. '
             'without_GAE changes only the actor advantage to raw Monte Carlo return minus the pre-update critic converted '
             'to raw units; critic targets remain GAE λ=0.95. The PPO loop, noise clip, log-prob clamp, '
             'ratio clip and KL early stop retain the DPPO settings. All averages and sample SDs use independent training seeds.', '',
             'Checkpoint state_0/35/70/105/139 contains 0/32/63/95/126 updates. Evaluation iteration 130 is at update 117 '
             '(9.36M environment steps). Training rows log their post-batch step counter but the plotted rollout uses the '
             'pre-update count. θ₀ for gain is each seed\'s own iteration-1 training J_disc, rather than the earlier shared pretrained anchor.', '',
             '[Methods and commands](' + str(root / 'methods.md') + ') · [Gate records](' + str(root / 'runs/gates') + ') · [Training job file](' + str(root / 'jobs_dl1.txt') + '). '
             'Exact source revisions, per-run commands, completion evidence, resource sharing and filenames are retained in '
             '[analysis_record.json](analysis_record.json). No final-checkpoint sampling was requested or performed by this analysis.', '',
             '## Gates, revision and allocation', '',
             markdown(['Gate', 'Result', 'Max absolute difference', 'Relative difference'],
                      [[name, 'passed' if value.get('passed') else 'not passed'] + gate_differences(name, value)
                       for name, value in sorted(gates.get('checks', {}).items())]), '',
             'Full numerical evidence: [' + str(gates.get('summary_path', 'pending')) + '](' + str(gates.get('summary_path', 'pending')) + '). '
             'Execution branch `%s`, revision `%s`, based on `%s`.' % (revision.get('branch', 'not recorded'),
                 revision.get('revision', 'not recorded'), revision.get('base_revision', 'not recorded')), '',
             markdown(['Seed', 'Scheduler job', 'Node', 'GPU allocation', 'Wall seconds'],
                      [[seed, runs[('without_GAE', seed)].get('worker', {}).get('scheduler_job_id', 'pending'),
                        runs[('without_GAE', seed)].get('worker', {}).get('hardware', {}).get('node', 'pending'),
                        runs[('without_GAE', seed)].get('worker', {}).get('hardware', {}).get('gpu', 'pending'),
                        fmt(runs[('without_GAE', seed)].get('worker', {}).get('elapsed_seconds'))]
                       for seed in range(3)]), '',
             'Each training process receives one allocated GPU and 40 CPUs. Allocations do not terminate or alter other jobs. '
             'Submission times and GPU identities in the saved worker manifests document resource sharing.', '',
             '## Figures', '']
    for name in inventory:
        lines += ['[' + name.replace('_', ' ') + '](figs/' + name + '.png) · [PDF](figs/' + name + '.pdf)', '',
                  (output / 'figs' / (name + '.caption.md')).read_text().strip(), '']
    if interim:
        cutoff = next((r.get('display_cutoff_itr') for r in runs.values() if r.get('display_cutoff_itr') is not None), None)
        if cutoff is not None:
            lines += ['All displayed curves and the progress comparison are capped at iteration %d for every method; native results remain unchanged. Latest train/J values refer to the last training iteration at or before that cutoff.' % cutoff, '']
        lines += ['## Current progress', '', markdown(['Method', 'Seed', 'Status', 'Last iteration', 'Latest eval', 'Latest train', 'Latest J_disc'],
                  [[r['method'], r['seed'], r['status'], r['last_itr'], fmt(r['eval_return']), fmt(r['train_return']), fmt(r['J_disc'])]
                   for r in progress_table(runs)]), '']
    # Relative links resolve after final publishing. Candidate preview retains
    # an explicit canonical-study link for provenance paths outside this folder.
    lines += ['Canonical study folder: [' + str(root) + '](' + str(root) + ').', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Main results/stage3/dl1 directory')
    parser.add_argument('--target-itr', type=int, help='Interim display cutoff applied identically to every method')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--interim', action='store_true')
    group.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    project = root.parents[2]
    runs = {(method, seed): load_run(project, root, method, seed) for method in METHODS for seed in range(3)}
    missing = ['%s seed %d: %s' % (method, seed, run['status']) for (method, seed), run in runs.items()
               if run['status'] != 'complete' or run['errors']]
    if args.target_itr is not None and (not args.interim or not 0 <= args.target_itr <= 139):
        parser.error('--target-itr requires --interim and an iteration between 0 and 139')
    if args.final and not gate_record(root).get('passed'):
        raise SystemExit('Final analysis requires the merged passed G1–G6 gate record')
    if args.final and missing:
        raise SystemExit('Final analysis requires complete verified runs: ' + '; '.join(missing))
    if args.interim and not any(runs[('without_GAE', seed)]['rows'] for seed in range(3)):
        raise SystemExit('No new training measurements yet; no candidate analysis generated')
    if args.target_itr is not None:
        for run in runs.values():
            run['actual_last_itr'] = run.get('last_itr')
            run['display_cutoff_itr'] = args.target_itr
            run['rows'] = [row for row in run['rows'] if row['itr'] <= args.target_itr]
            run['training'] = [row for row in run['rows'] if row['itr'] % 10]
            run['evaluation'] = [row for row in run['rows'] if not row['itr'] % 10]
            run['last_itr'] = run['rows'][-1]['itr'] if run['rows'] else None
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'candidates' / (('final_' if args.final else 'interim_') + timestamp)
    output.mkdir(parents=True, exist_ok=False)
    distance_records, distances = [], {}
    for key, run in runs.items():
        distances[key] = checkpoint_distances(run, distance_records)
    write_csv(output / 'checkpoint_distances.csv', distance_records)
    seeds = [seed_metrics(run, distances[key]) for key, run in runs.items()]
    conditions = condition_metrics(seeds)
    comparison = contrasts(conditions)
    write_tables(seeds, conditions, comparison, output)
    write_csv(output / 'progress.csv', progress_table(runs))
    write_json(output / 'condition_metrics.json', conditions)
    inventory = plots(runs, distances, output, args.interim)
    report(root, output, runs, conditions, comparison, args.interim, inventory)
    source_records = []
    for (method, seed), run in runs.items():
        worker = run.get('worker', {})
        source_records.append(dict(method=method, seed=seed, run_dir=run['run_dir'], result_path=run['result_path'],
                                   status=run['status'], last_itr=run.get('last_itr'), checkpoints=run['checkpoints'],
                                   worker_manifest=run.get('worker_manifest'), errors=run['errors'],
                                   command=worker.get('command'), source_before=worker.get('source_before'),
                                   source_after=worker.get('source_after'), hardware=worker.get('hardware'),
                                   scheduler_job_id=worker.get('scheduler_job_id'), elapsed_seconds=worker.get('elapsed_seconds')))
    record = dict(status='complete' if args.final else 'interim', observed_utc=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv, candidate_dir=str(output), missing=missing, runs=source_records,
                  display_cutoff_itr=args.target_itr, execution_revision=read_json(root / 'provenance/execution_revision.json', {}),
                  gate_record=gate_record(root),
                  figure_names=inventory, final_publication=bool(args.final), new_sampling=False,
                  uncertainty='sample SD across 3 independent training seeds, ddof=1; figures show no uncertainty',
                  contrast_rule='absolute difference below twice pooled seed SD is unresolved',
                  interaction_sd='sqrt(sum of four condition seed variances / 2); no paired covariance',
                  J_anchor='each seed iteration1 J_disc_train, or not logged',
                  historical_limits={'DPPO': ['J_disc', 'true KL', 'clipfrac', 'switched-off fraction', 'saturation'],
                                     'without_PPO': ['J_disc', 'saturation']})
    write_json(output / 'analysis_record.json', record)
    if args.final:
        (root / 'figs').mkdir(exist_ok=True)
        for path in output.iterdir():
            if path.is_file():
                shutil.copy2(str(path), str(root / path.name))
        for path in (output / 'figs').iterdir():
            shutil.copy2(str(path), str(root / 'figs' / path.name))
        complete_bookkeeping(root, project, conditions)
    print(json.dumps({'status': record['status'], 'candidate_dir': str(output), 'missing': missing,
                      'published': bool(args.final)}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
