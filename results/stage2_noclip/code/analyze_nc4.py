#!/usr/bin/env python3
"""Analyze the completed NC4 subset at the user's request; never touch training.

Reuses Stage 2's validated loaders and predeclared summaries. New outputs only.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys

from analyze_stage2 import load_run, summarize_run, complete_case_stats, metric_series

STUDY = Path(__file__).resolve().parents[1]
BASELINE = STUDY.parent / 'repro_halfcheetah'
GROUPS = ['baseline', 'NC4_lr1e-4', 'NC4_lr1e-3', 'NC4_lr3e-3']
LABELS = ['Stage 1 baseline', 'NC4 · LR 1e−4', 'NC4 · LR 1e−3', 'NC4 · LR 3e−3']
COLORS = ['#343a40', '#0072B2', '#009E73', '#D55E00']
RATES = {'NC4_lr1e-4': 1e-4, 'NC4_lr1e-3': 1e-3, 'NC4_lr3e-3': 3e-3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir).resolve()
    require(STUDY in destination.parents and destination.name == 'candidates',
            'Use a new candidates directory inside this study')
    require(not destination.exists(), 'Refusing to overwrite candidate artifacts')
    require(not (destination.parent / 'analysis_record.json').exists(),
            'Refusing to overwrite analysis evidence')

    runs = [load_run(BASELINE / 'runs' / name / 'manifest.json', 'baseline', seed)
            for seed, name in enumerate(['seed0_handoff', 'seed1', 'seed2'])]
    for condition in GROUPS[1:]:
        for seed in range(3):
            run_id = condition + '_seed' + str(seed)
            runs.append(load_run(STUDY / 'runs' / run_id / 'manifest.json',
                                 condition, seed, run_id))
    checks = []
    summaries = []
    for run in runs:
        rows, manifest = run['rows'], run['manifest']
        require(run['status'] == 'complete' and len(rows) == 140, 'Incomplete input')
        require(all(math.isfinite(v) for row in rows for v in row.values()),
                'Nonfinite input; report before generating this completed-run subset')
        summary = summarize_run(run)
        train = [row for row in rows if 'train_episode_reward' in row]
        evals = [row for row in rows if 'eval_episode_reward' in row]
        require(len(train) == 126 and len(evals) == 14, 'Unexpected measurement counts')
        summary['train_return_itr1'] = rows[1]['train_episode_reward']
        summary['train_return_itr139'] = rows[139]['train_episode_reward']
        summary['first_collapse_eval_itr'] = next((row['itr'] for row in evals
            if row['eval_episode_reward'] < evals[0]['eval_episode_reward'] / 2), None)
        summary['result_path'] = run['result_path']
        summary['actor_lr'] = RATES.get(run['condition'])
        if run['condition'] != 'baseline':
            expected = {'actor_optimizer_steps': 1, 'critic_optimizer_steps': 20,
                        'actor_step_ratio': 1.0, 'actor_step_clipfrac': 0.0,
                        'ratio': 1.0, 'clipfrac': 0.0, 'approx_kl': 0.0,
                        'actor_lr': RATES[run['condition']]}
            observed = {key: sorted({row[key] for row in train}) for key in expected}
            require(all(observed[key] == [value] for key, value in expected.items()),
                    'NC4 invariant failed: ' + run['run_id'])
            require(manifest['effective_startup_flags'] ==
                    {'clamp_logprob': False, 'logprob_reduce': 'mean',
                     'actor_single_step': True}, 'Startup flags do not match NC4')
            require(manifest['checkpoint_tensors_finite'] is True,
                    'Missing supervisor finite-checkpoint verification')
            summary['max_kl_true_per_action'] = max(row['kl_true_per_action'] for row in train)
            summary['max_logratio_p99'] = max(row['logratio_p99'] for row in train)
            summary['single_step_checks_passed'] = True
            checks.append({'run_id': run['run_id'], 'training_rows_checked': len(train),
                           'observed_unique_values': observed,
                           'startup_flags': manifest['effective_startup_flags']})
        summaries.append(summary)

    grouped = {c: [r for r in runs if r['condition'] == c] for c in GROUPS}
    aggregates = {}
    for condition in GROUPS:
        members = [s for s in summaries if s['condition'] == condition]
        aggregates[condition] = {'n_seeds': 3, 'collapse_count': sum(s['collapse'] for s in members)}
        for key in ['eval_return_itr0', 'eval_return_itr70', 'eval_return_itr130',
                    'train_return_itr1', 'train_return_itr139']:
            values = [s[key] for s in members]
            aggregates[condition][key] = {'mean': statistics.mean(values),
                                          'population_std': statistics.pstdev(values)}

    os.environ.setdefault('MPLCONFIGDIR', str(STUDY / 'cache' / 'matplotlib'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold',
                         'axes.labelcolor': '#343a40', 'text.color': '#232a31',
                         'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                         'pdf.fonttype': 42})
    destination.mkdir(parents=True)
    log_omissions = {'zero_observations': 0, 'band_positions_with_nonpositive_lower_bound': 0}

    def plot_group(axis, condition, metric, horizontal, logarithmic=False):
        color = COLORS[GROUPS.index(condition)]
        for run in grouped[condition]:
            series = metric_series(run, metric, horizontal)
            x = np.array(sorted(series), dtype=float)
            y = np.array([series[item] for item in x])
            if horizontal == 'step':
                x /= 1e6
            if logarithmic:
                log_omissions['zero_observations'] += int(np.count_nonzero(y == 0))
                y = np.where(y > 0, y, np.nan)
            axis.plot(x, y, color=color, alpha=0.25, lw=0.85, zorder=2)
        x, mean, std = complete_case_stats(grouped[condition], metric, horizontal)
        x, mean, std = np.asarray(x, dtype=float), np.asarray(mean), np.asarray(std)
        if horizontal == 'step':
            x /= 1e6
        lower, upper = mean - std, mean + std
        valid = np.ones(len(x), dtype=bool)
        if logarithmic:
            valid = lower > 0
            log_omissions['band_positions_with_nonpositive_lower_bound'] += int((~valid).sum())
        axis.fill_between(x, lower, upper, where=valid, color=color, alpha=0.12, lw=0)
        axis.plot(x, np.where(mean > 0, mean, np.nan) if logarithmic else mean,
                  color=color, lw=2.25, label=LABELS[GROUPS.index(condition)], zorder=3)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    for axis, metric, title in zip(axes, ['eval_episode_reward', 'train_episode_reward'],
                                  ['Evaluation return', 'Training return · exploration noise']):
        for condition in GROUPS:
            plot_group(axis, condition, metric, 'step')
        axis.set(title=title, xlabel='Training environment steps (millions)', xlim=(0, 10.15))
        axis.set_xticks([0, 2.5, 5, 7.5, 10])
        axis.grid(alpha=0.18)
    axes[0].set_ylabel('Undiscounted episode return')
    figure.suptitle('NC4: one actor update per batch', fontsize=16, fontweight='bold', y=0.985)
    figure.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
                  bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False, fontsize=10)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.18, wspace=0.12)
    figure.text(0.5, 0.035, 'Three seeds per condition · bold mean ± population SD · faint individual seeds\n'
                'Last evaluation: 9.36M steps; training ends at 10.08M. No smoothing or extrapolation.',
                ha='center', fontsize=9, color='#525b64')
    for ext in ['png', 'pdf']:
        figure.savefig(destination / ('nc4_returns.' + ext), dpi=200)
    plt.close(figure)

    metrics = [('kl_true_per_action', 'Post-update KL estimate', 'Estimated KL per action'),
               ('logratio_p99', 'Post-update log-ratio tail', '99th percentile of |d|'),
               ('clamp_hit_frac', 'Coordinates outside [−5, 2]', 'Fraction of coordinates (%)'),
               ('approx_kl', 'DPPO approximate KL · before actor step', 'Last critic-loop minibatch'),
               ('clipfrac', 'DPPO clip fraction · before actor step', 'Minibatch mean')]
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.8))
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        for condition in GROUPS[1:]:
            plot_group(axis, condition, metric, 'itr', metric == 'kl_true_per_action')
        axis.set(title=title, xlabel='Training iteration', ylabel=ylabel, xlim=(0, 140))
        if metric == 'kl_true_per_action':
            axis.set_yscale('log')
        if metric == 'clamp_hit_frac':
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        if metric in ['approx_kl', 'clipfrac']:
            axis.set_ylim(-0.03, 0.12)
            axis.set_yticks([0, 0.05, 0.1])
            axis.text(0.5, 0.68, 'All three learning rates: exactly 0',
                      transform=axis.transAxes, ha='center', fontsize=10)
        axis.grid(alpha=0.18)
    notes = axes.flat[-1]
    notes.axis('off')
    notes.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper left', frameon=False)
    note = ('Three seeds · bold mean ± population SD\nFaint lines show each seed; no smoothing.\n\n'
            'Actor pre-step ratio = 1 in every update.\nZero pre-step KL does not constrain post-step KL.\n\n'
            'Log panel omits %d exact-zero observations;\n%d nonpositive lower SD bounds are not drawn.\n'
            'No zeros are replaced with an epsilon.\n\n'
            'Stage 1 did not record these diagnostics;\nR0 is still running and is not included.'
            % (log_omissions['zero_observations'],
               log_omissions['band_positions_with_nonpositive_lower_bound']))
    notes.text(0.02, 0.56, note, va='top', fontsize=9, linespacing=1.45, transform=notes.transAxes)
    figure.suptitle('NC4 diagnostics: changes after one actor step', fontsize=16,
                    fontweight='bold', y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    for ext in ['png', 'pdf']:
        figure.savefig(destination / ('nc4_diagnostics.' + ext), dpi=200)
    plt.close(figure)

    fields = ['condition', 'seed', 'run_id', 'actor_lr', 'status', 'completed_iterations',
              'eval_return_itr0', 'eval_step_itr0', 'eval_return_itr70', 'eval_step_itr70',
              'eval_return_itr130', 'eval_step_itr130', 'minimum_eval_return',
              'collapse', 'first_collapse_eval_itr', 'median_kl_true_per_action',
              'max_kl_true_per_action', 'max_logratio_p99', 'train_return_itr1',
              'train_return_itr139', 'single_step_checks_passed', 'result_path', 'manifest']
    with (destination / 'nc4_seed_results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: 'NA' if row.get(key) is None else row[key] for key in fields}
                         for row in summaries)
    captions = {
        'nc4_returns': 'Evaluation (left) and exploratory training (right) undiscounted 1000-step episode returns compare the unchanged Stage 1 baseline with NC4 at three constant actor learning rates. The x-axis is cumulative training environment steps in millions; faint lines show seeds 0, 1 and 2, bold lines show their arithmetic mean, and bands show ± one population standard deviation across seeds, not confidence intervals. All twelve plotted runs completed 140 iterations; evaluation ends at 9.36 million steps and training at 10.08 million, with no smoothing, interpolation or extrapolation. LR 1e-4 preserves high returns but ends below the baseline mean, LR 1e-3 degrades, and all LR 3e-3 seeds meet the predefined collapse criterion; this three-seed comparison does not isolate sample reuse from total actor-update magnitude.',
        'nc4_diagnostics': 'Five logged diagnostics are shown against iteration for the nine completed NC4 runs, with faint seed traces and bold arithmetic means ± population standard deviation over three seeds per learning rate. The logarithmic KL panel uses the prescribed post-update statistic 10 × mean[expm1(d) − d] on 20,000 sampled denoising pairs, where d sums the raw log-probability difference over 24 coordinates; this is not exact final-action KL, while the p99 panel shows |d| and the outside-range panel uses percentages of coordinates. Four exact-zero KL observations and nonpositive lower band bounds are omitted from the log panel without numerical replacement; evaluation-only iterations have no measurements. DPPO approximate KL and clip fraction are exactly zero before the actor update, yet post-update changes grow with learning rate and the largest rate collapses; outside-range fractions describe where clamping would have applied because NC4 disables it. Stage 1 has no diagnostic measurements and the ongoing R0 is excluded.',
        'nc4_seed_results.csv': 'One row per seed for the Stage 1 baseline and each NC4 learning rate records evaluation returns at 0, 5.04 million and 9.36 million training steps, minimum evaluation return, first observed collapse, median/maximum diagnostic KL, and training returns at iterations 1 and 139. Collapse means any saved evaluation below half that seed initial evaluation or any scientific NaN/inf; all twelve runs here are complete and finite, so this subset has no missing outcomes. Values are individual-run measurements, without uncertainty intervals; NA denotes diagnostics or NC4 checks that were not recorded or applicable to Stage 1. None of the nine NC4 runs violated its single-step invariant; absence of the collapse flag at LR 1e-3 does not mean baseline-quality performance.'
    }
    with (destination / 'captions.md').open('x') as stream:
        stream.write('\n\n'.join('**' + key + '**\n\n' + value for key, value in captions.items()) + '\n')
    sources = [{'condition': r['condition'], 'seed': r['seed'], 'run_id': r['run_id'],
                'manifest': r['manifest_path'], 'result_path': r['result_path'],
                'logdir': r['manifest'].get('logdir'),
                'checkpoints': r['manifest'].get('checkpoints', r['manifest'].get('checkpoints_found', [])),
                'status': r['status'], 'source_after': r['manifest'].get('source_after'),
                'wall_seconds': r['manifest'].get('subprocess_wall_seconds')} for r in runs]
    record = {'status': 'completed_nc4_subset_analysis', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'authorization': 'User: do the analysis and ploting for the NC4 runs',
              'command': shlex.join([sys.executable] + sys.argv),
              'scientific_commit': 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964',
              'baseline_commit': 'cc7234ad7ff39a8f32de3af903606723a16f0648',
              'analysis_code': str(Path(__file__).resolve()),
              'shared_analysis_code': str(STUDY / 'code' / 'analyze_stage2.py'),
              'python_version': sys.version, 'numpy_version': np.__version__,
              'matplotlib_version': matplotlib.__version__,
              'aggregation': 'Arithmetic mean and population SD across seeds0,1,2; no smoothing or extrapolation',
              'summaries': summaries, 'aggregates': aggregates, 'invariant_checks': checks,
              'sources': sources, 'log_axis_omissions': log_omissions,
              'artifacts': ['nc4_returns.png', 'nc4_returns.pdf', 'nc4_diagnostics.png',
                            'nc4_diagnostics.pdf', 'nc4_seed_results.csv'],
              'captions': str(destination / 'captions.md'),
              'curation_status': 'Awaiting explicit artifact/name approval; no curated outputs',
              'scope': 'Completed NC4 subset and reused Stage1 only. R0/NC1 replacements and full-campaign finalizer remain untouched.'}
    with (destination.parent / 'analysis_record.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'candidate_directory': str(destination), 'aggregates': aggregates,
                      'log_axis_omissions': log_omissions, 'validated_nc4_runs': len(checks)}, indent=2))


if __name__ == '__main__':
    main()
