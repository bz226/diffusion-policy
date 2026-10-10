#!/usr/bin/env python3
"""Plot the completed NC2/NC3 subset without changing training or existing outputs."""
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
GROUPS = ['baseline', 'NC2', 'NC3']
COLORS = {'baseline': '#343a40', 'NC2': '#D55E00', 'NC3': '#009E73'}
LABELS = {'baseline': 'Stage 1 baseline', 'NC2': 'NC2 · coordinate mean',
          'NC3': 'NC3 · coordinate sum'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    destination = Path(parser.parse_args().output_dir).resolve()
    require(STUDY in destination.parents and destination.name == 'candidates',
            'Output must be a candidates directory inside this study')
    require(not destination.exists() and not (destination.parent / 'analysis_record.json').exists(),
            'Refusing to overwrite existing evidence')
    runs = [load_run(BASELINE / 'runs' / name / 'manifest.json', 'baseline', seed)
            for seed, name in enumerate(['seed0_handoff', 'seed1', 'seed2'])]
    for condition in GROUPS[1:]:
        for seed in range(3):
            run_id = condition + '_seed' + str(seed)
            runs.append(load_run(STUDY / 'runs' / run_id / 'manifest.json', condition, seed, run_id))

    summaries, checks, sources = [], [], []
    for run in runs:
        rows, manifest = run['rows'], run['manifest']
        require(run['status'] == 'complete' and len(rows) == 140, 'Incomplete input')
        require(all(math.isfinite(v) for row in rows for v in row.values()), 'Nonfinite input')
        train = [row for row in rows if 'train_episode_reward' in row]
        evals = [row for row in rows if 'eval_episode_reward' in row]
        require(len(train) == 126 and len(evals) == 14, 'Unexpected measurement count')
        summary = summarize_run(run)
        summary.update(result_path=run['result_path'], train_return_itr1=rows[1]['train_episode_reward'],
                       train_return_itr139=rows[139]['train_episode_reward'],
                       first_collapse_eval_itr=next((r['itr'] for r in evals
                           if r['eval_episode_reward'] < evals[0]['eval_episode_reward'] / 2), None))
        checkpoints = manifest.get('checkpoints', manifest.get('checkpoints_found', []))
        require(len(checkpoints) == 5 and all(Path(p).is_file() for p in checkpoints),
                'Missing checkpoint: ' + str(run['run_id']))
        if run['condition'] != 'baseline':
            expected = {'actor_optimizer_steps': 20, 'critic_optimizer_steps': 20,
                        'actor_lr': 1e-4}
            observed = {key: sorted({row[key] for row in train}) for key in expected}
            require(all(observed[key] == [value] for key, value in expected.items()),
                    'Update invariant failed: ' + run['run_id'])
            expected_flags = {'clamp_logprob': False, 'actor_single_step': False,
                              'logprob_reduce': 'mean' if run['condition'] == 'NC2' else 'sum'}
            require(manifest['effective_startup_flags'] == expected_flags, 'Startup flag mismatch')
            require(manifest['checkpoint_tensors_finite'] is True, 'Missing finite-checkpoint verification')
            require(manifest['source_after']['tracked_files_clean'] and
                    manifest['source_after']['commit'] == 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964',
                    'Scientific source mismatch')
            require(manifest['fixed_batch_gate']['passed'], 'Fixed-batch gate did not pass')
            summary.update(max_kl_true_per_action=max(r['kl_true_per_action'] for r in train),
                           max_logratio_p99=max(r['logratio_p99'] for r in train),
                           max_clamp_hit_frac=max(r['clamp_hit_frac'] for r in train),
                           max_clipfrac=max(r['clipfrac'] for r in train),
                           nonzero_clipfrac_iterations=[r['itr'] for r in train if r['clipfrac'] != 0],
                           update_checks_passed=True)
            checks.append({'run_id': run['run_id'], 'training_rows_checked': len(train),
                           'observed_unique_values': observed, 'startup_flags': expected_flags,
                           'nonzero_clipfrac_rows': [r for r in train if r['clipfrac'] != 0]})
        summaries.append(summary)
        sources.append({'condition': run['condition'], 'seed': run['seed'], 'run_id': run['run_id'],
                        'manifest': run['manifest_path'], 'result_path': run['result_path'],
                        'logdir': manifest.get('logdir'), 'checkpoints': checkpoints,
                        'status': run['status'], 'source_after': manifest.get('source_after')})
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
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'font.family': 'DejaVu Sans',
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    destination.mkdir(parents=True)
    omissions = {'zero_observations': 0, 'band_positions_with_nonpositive_lower_bound': 0}

    def plot_group(axis, condition, metric, horizontal, logarithmic=False):
        color = COLORS[condition]
        for run in grouped[condition]:
            series = metric_series(run, metric, horizontal)
            x = np.array(sorted(series), dtype=float)
            y = np.array([series[item] for item in x])
            if horizontal == 'step':
                x /= 1e6
            if logarithmic:
                omissions['zero_observations'] += int(np.count_nonzero(y == 0))
                y = np.where(y > 0, y, np.nan)
            axis.plot(x, y, color=color, alpha=0.25, lw=0.85, zorder=2)
        x, mean, std = complete_case_stats(grouped[condition], metric, horizontal)
        x, mean, std = np.asarray(x, dtype=float), np.asarray(mean), np.asarray(std)
        if horizontal == 'step':
            x /= 1e6
        lower, upper = mean - std, mean + std
        valid = lower > 0 if logarithmic else np.ones(len(x), dtype=bool)
        if logarithmic:
            omissions['band_positions_with_nonpositive_lower_bound'] += int((~valid).sum())
        axis.fill_between(x, lower, upper, where=valid, color=color, alpha=0.12, lw=0)
        axis.plot(x, np.where(mean > 0, mean, np.nan) if logarithmic else mean,
                  color=color, lw=2.25, label=LABELS[condition], zorder=3)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    for axis, metric, title in zip(axes, ['eval_episode_reward', 'train_episode_reward'],
                                  ['Evaluation return', 'Training return · exploration noise']):
        for condition in GROUPS:
            plot_group(axis, condition, metric, 'step')
        axis.set(title=title, xlabel='Training environment steps (millions)', xlim=(0, 10.15))
        axis.set_xticks([0, 2.5, 5, 7.5, 10])
        axis.grid(alpha=0.18)
    axes[0].set_ylabel('Undiscounted episode return')
    figure.suptitle('NC2 and NC3: fine-tuning after clipping removal', fontsize=16, fontweight='bold', y=0.985)
    figure.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
                  bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=10)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.18, wspace=0.12)
    figure.text(0.5, 0.035, 'Three seeds per condition · bold mean ± population SD · faint individual seeds\n'
                'Last evaluation: 9.36M steps; training ends at 10.08M. No smoothing or extrapolation.',
                ha='center', fontsize=9, color='#525b64')
    for ext in ['png', 'pdf']:
        figure.savefig(destination / ('nc23_returns.' + ext), dpi=200)
    plt.close(figure)

    metrics = [('kl_true_per_action', 'Post-update KL estimate', 'Estimated KL per action'),
               ('logratio_p99', 'Post-update log-ratio tail', '99th percentile of |d|'),
               ('clamp_hit_frac', 'Coordinates outside [−5, 2]', 'Fraction of coordinates (%)'),
               ('approx_kl', 'DPPO approximate KL · last minibatch', 'Native approximate KL'),
               ('clipfrac', 'DPPO clip fraction · minibatch mean', 'Fraction of samples')]
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.8))
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        for condition in GROUPS[1:]:
            plot_group(axis, condition, metric, 'itr', metric == 'kl_true_per_action')
        axis.set(title=title, xlabel='Training iteration', ylabel=ylabel, xlim=(0, 140))
        if metric == 'kl_true_per_action':
            axis.set_yscale('log')
        if metric == 'clamp_hit_frac':
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        if metric == 'clipfrac':
            axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            axis.text(0.47, 0.72, 'One nonzero row: NC3 seed 1, itr 31\nClip fraction ≈ 1e−6; all others 0',
                      transform=axis.transAxes, ha='left', fontsize=9)
        axis.grid(alpha=0.18)
    notes = axes.flat[-1]
    notes.axis('off')
    notes.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper left', frameon=False)
    notes.text(0.02, 0.65,
               'Three seeds · bold mean ± population SD\nFaint lines show each seed; no smoothing.\n\n'
               'Raw summed log-probs define post-update d\nfor both conditions; native KL uses mean/sum.\n'
               'Outside-range fractions are hypothetical:\nlog-prob clamping is disabled in both.\n\n'
               'Log panel omits %d exact-zero observations;\n%d nonpositive lower SD bounds are not drawn.\n'
               'No zeros are replaced with an epsilon.\n\n'
               'Stage 1 did not record these diagnostics;\nR0 will be included in the full Stage 2 analysis.'
               % (omissions['zero_observations'], omissions['band_positions_with_nonpositive_lower_bound']),
               va='top', fontsize=9, linespacing=1.45, transform=notes.transAxes)
    figure.suptitle('NC2 and NC3: update diagnostics', fontsize=16, fontweight='bold', y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    for ext in ['png', 'pdf']:
        figure.savefig(destination / ('nc23_diagnostics.' + ext), dpi=200)
    plt.close(figure)

    fields = ['condition', 'seed', 'run_id', 'status', 'completed_iterations',
              'eval_return_itr0', 'eval_step_itr0', 'eval_return_itr70', 'eval_step_itr70',
              'eval_return_itr130', 'eval_step_itr130', 'minimum_eval_return', 'collapse',
              'first_collapse_eval_itr', 'median_kl_true_per_action', 'max_kl_true_per_action',
              'max_logratio_p99', 'max_clamp_hit_frac', 'max_clipfrac', 'train_return_itr1', 'train_return_itr139',
              'update_checks_passed', 'result_path', 'manifest']
    with (destination / 'nc23_seed_results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: 'NA' if row.get(key) is None else row[key] for key in fields}
                         for row in summaries)
    captions = {
        'nc23_returns': 'Evaluation (left) and exploratory training (right) undiscounted 1000-step episode returns compare the unchanged Stage 1 baseline with NC2 and NC3, each using seeds 0, 1 and 2. The x-axis is cumulative training environment steps in millions; faint lines show individual seeds, bold lines their arithmetic mean, and bands ± one population standard deviation across seeds, not confidence intervals. NC2 sets the ratio-clip bound to 1e6 and disables KL early stopping, log-probability clamping and practical sampling-noise clipping; NC3 additionally replaces the 24-coordinate mean by a sum when forming the PPO ratio. All nine runs completed 140 iterations, with last evaluation at 9.36 million steps and training ending at 10.08 million; no smoothing or extrapolation is used. NC3 ends higher than NC2 but all six ablation seeds crossed the predefined collapse threshold; three seeds and the joint removals in NC2 do not isolate each mechanism.',
        'nc23_diagnostics': 'The six NC2/NC3 runs contribute three seeds per condition to faint individual traces, bold arithmetic means and ± population-SD bands versus training iteration. Post-update diagnostic KL uses a logarithmic axis and the prescribed 10 × mean[expm1(d) − d] on 20,000 sampled denoising pairs, where d sums raw log-probability differences over 24 coordinates; it estimates a denoising-transition quantity, not exact final-action marginal KL, and p99 denotes the 99th percentile of |d|. The outside-range panel is the percentage of coordinates that would have been clamped to [−5, 2], since clamping is disabled; native approximate KL is from the last update minibatch and uses different mean/sum reductions in NC2/NC3, while native clip fraction is zero except NC3 seed 1 iteration 31 (approximately 1e-6), showing that the prescribed finite bound was reached. The log panel omits %d exact-zero measurements and %d band positions with nonpositive lower bounds, retaining zeros in all arithmetic summaries without epsilon replacement; evaluation-only iterations have no diagnostics. Early NC2 changes subside after returns deteriorate, whereas NC3 retains intermittent large updates; the partial condition comparison excludes the still-separate R0 diagnostic baseline.' % (omissions['zero_observations'], omissions['band_positions_with_nonpositive_lower_bound']),
        'nc23_seed_results.csv': 'One row per seed for the Stage 1 baseline, NC2 and NC3 reports saved evaluation returns at 0, 5.04 million and 9.36 million steps, minimum evaluation return, first observed collapse, median/maximum post-update diagnostic KL, maximum p99 and outside-range fraction, and first/final training returns. Collapse means any evaluation below half its own initial return or any scientific NaN/inf; all nine runs are complete and finite, and all six ablation seeds meet the reward-collapse criterion. Values are individual-run measurements without uncertainty intervals; NA denotes a diagnostic or ablation check unavailable for Stage 1, or no observed collapse. Every ablation training iteration records 20 actor steps, 20 critic steps and LR 1e-4; clip fraction is zero in 755 of 756 training rows, with the sole approximately 1e-6 exception in NC3 seed 1 iteration 31; high final NC3 returns relative to NC2 do not undo earlier collapse.'}
    with (destination / 'captions.md').open('x') as stream:
        stream.write('\n\n'.join('**' + key + '**\n\n' + value for key, value in captions.items()) + '\n')
        stream.write('\nReproduction: [source paths and command](../analysis_record.json), '
                     '[subset methods](../../../methods.md#nc2-and-nc3-plots-before-full-stage-2-analysis).\n')
    record = {'status': 'completed_nc23_subset_plotting', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'authorization': 'User: do the plotting first before the full stage 2 analysis',
              'command': shlex.join([sys.executable] + (['-B'] if sys.dont_write_bytecode else []) + sys.argv),
              'scientific_commit': 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964',
              'baseline_commit': 'cc7234ad7ff39a8f32de3af903606723a16f0648',
              'analysis_code': str(Path(__file__).resolve()),
              'shared_analysis_code': str(STUDY / 'code' / 'analyze_stage2.py'),
              'python_version': sys.version, 'numpy_version': np.__version__,
              'matplotlib_version': matplotlib.__version__, 'aggregation': 'Mean and population SD across seeds 0,1,2',
              'summaries': summaries, 'aggregates': aggregates, 'invariant_checks': checks, 'sources': sources,
              'log_axis_omissions': omissions, 'artifacts': ['nc23_returns.png', 'nc23_returns.pdf',
                  'nc23_diagnostics.png', 'nc23_diagnostics.pdf', 'nc23_seed_results.csv'],
              'captions': str(destination / 'captions.md'),
              'curation_status': 'Awaiting explicit artifact/name approval; candidates only',
              'scope': 'Completed NC2/NC3 and unchanged Stage 1; no full Stage 2 analysis or training changes.'}
    with (destination.parent / 'analysis_record.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'candidate_directory': str(destination), 'aggregates': aggregates,
                      'log_axis_omissions': omissions, 'validated_ablation_runs': len(checks)}, indent=2))


if __name__ == '__main__':
    main()
