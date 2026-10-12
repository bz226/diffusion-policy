"""Approved E1C comparisons; saved measurements only, Python 3.8 compatible."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/e1c'
STAGE3 = ROOT.parent
LABELS = {'E0': 'without_PPO (GAE + critic)', 'E1C': 'MC + critic', 'E1': 'MC + LOO'}
COLORS = {'E0': '#596573', 'E1C': '#c75d21', 'E1': '#167c91'}
FIELDS = ['eval_936M', 'train_last10', 'J_disc_gain_last10', 'median_KL',
          'theta_dist_final', 'saturation_final']


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [number(v) for v in values]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def stats(values, require_three=True):
    values = [number(v) for v in values]
    if any(v is None for v in values) or len(values) < (3 if require_three else 2):
        return None, None
    if require_three and len(values) != 3:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def complete_values(records, key, count):
    values = [number(r.get(key)) for r in records]
    return values if len(values) == count and all(v is not None for v in values) else None


def strict_mean(records, key, count):
    values = complete_values(records, key, count)
    return float(np.mean(values)) if values is not None else None


def nonfinite_paths(value, prefix=''):
    if isinstance(value, dict):
        return [p for key, child in value.items() for p in nonfinite_paths(child, prefix + '.' + str(key))]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [p for i, child in enumerate(value) for p in nonfinite_paths(child, prefix + '[%d]' % i)]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return [prefix]
    return []


def fmt(value, digits=3):
    return 'N/A' if value is None else ('{:.%dg}' % digits).format(value)


def cell(pair):
    return fmt(pair[0]) + ' ± ' + fmt(pair[1]) if pair[0] is not None else 'N/A'


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def enrich(records):
    output, applied = [], 0
    for index, original in enumerate(records):
        row = dict(original)
        if row.get('itr') != index:
            raise ValueError('Run records must be an uninterrupted iteration prefix beginning at zero')
        training = 'train_episode_reward' in row
        if training != (index % 10 != 0):
            raise ValueError('Unexpected training/evaluation schedule')
        row['updates_before_rollout'] = applied
        if training:
            if row.get('actor_optimizer_steps', 1) != 1:
                raise ValueError('E0/E1/E1C require exactly one actor update per training record')
            row['update_number'] = applied + 1
            applied += 1
        else:
            row['update_number'] = None
        energies = {}
        for base in ('critic', 'loo'):
            a = row.get('baseline_residual_mean_' + base)
            b = row.get('baseline_residual_std_' + base)
            if a is not None and b is not None:
                if row.get('baseline_residual_std_ddof') != 0:
                    raise ValueError('Unweighted energy requires population residual standard deviations')
                a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
                if a.shape != (250,) or b.shape != (250,):
                    raise ValueError('Unexpected residual moment shape')
                with np.errstate(over='ignore', invalid='ignore'):
                    energy = a * a + b * b
                energies[base] = energy
                row['unweighted_energy_' + base] = number(energy.sum())
                for k in range(5):
                    row['unweighted_energy_%s_bin%d' % (base, k)] = number(energy[k*50:(k+1)*50].sum())
        if len(energies) == 2:
            for suffix in [''] + ['_bin%d' % k for k in range(5)]:
                a, b = row['unweighted_energy_critic' + suffix], row['unweighted_energy_loo' + suffix]
                row['unweighted_energy_ratio' + suffix] = a / b if a is not None and b is not None and b > 0 else None
        output.append(row)
    return output


def e1c_selection():
    """Use immutable run specs; retries live under ROOT/retries/, unlike first attempts."""
    inventory = []
    state_path = ROOT / 'runs/campaign.json'
    state = read(state_path) if state_path.exists() else {}
    submitted = state.get('submitted', [])
    for seed in range(3):
        entries = [item for item in submitted if item.get('seed') == seed and
                   str(item.get('run_id', '')).startswith('E1C_seed')]
        if not entries:
            entries = [{'run_id': 'E1C_seed%d' % seed, 'seed': seed, 'attempt': 0}]
        for item in entries:
            run_id = item['run_id']
            spec_path = ROOT / 'runs/specs' / (run_id + '.json')
            manifest_path = ROOT / 'runs' / run_id / 'manifest.json'
            spec = read(spec_path) if spec_path.exists() else {}
            manifest = read(manifest_path) if manifest_path.exists() else {}
            attempt = int(spec.get('attempt', item.get('attempt', 0)))
            fallback = ROOT / run_id if attempt == 0 else ROOT / 'retries' / run_id
            directory = Path(spec.get('logdir', manifest.get('logdir', item.get('logdir', fallback))))
            verified = (item.get('run_status') == 'complete' and item.get('scheduler_state') == 'COMPLETED'
                        and item.get('exit_code') == '0:0' and manifest.get('status') == 'complete'
                        and manifest.get('subprocess_exitcode') == 0
                        and bool(manifest.get('verification') or manifest.get('final_verification'))
                        and manifest.get('remaining_process_group_members') == [])
            inventory.append({'condition': 'E1C', 'seed': seed, 'run_id': run_id, 'attempt': attempt,
                              'result_path': str(directory / 'result.pkl'), 'result_exists': (directory / 'result.pkl').is_file(),
                              'spec_path': str(spec_path), 'manifest_path': str(manifest_path),
                              'scheduler_job_id': item.get('job_id'), 'scheduler_state': item.get('scheduler_state'),
                              'scheduler_exit_code': item.get('exit_code'), 'run_status': item.get('run_status'),
                              'manifest_status': manifest.get('status'), 'verified_complete': verified,
                              'failure_reason': manifest.get('failure_reason'), 'failure_iteration': manifest.get('failure_iteration'),
                              'elapsed_seconds': number(manifest.get('elapsed_seconds')),
                              'source_provenance': manifest.get('source_provenance'), 'command': spec.get('command'),
                              'selected': False})
    selected = {}
    for seed in range(3):
        attempts = [r for r in inventory if r['seed'] == seed]
        winner = max(attempts, key=lambda r: (r['verified_complete'], r['result_exists'], r['attempt']))
        winner['selected'] = True
        selected[seed] = winner
    return selected, inventory


def e1c_paths():
    return {seed: Path(row['result_path']) for seed, row in e1c_selection()[0].items()}


def load_runs():
    runs = []
    e1c, _ = e1c_selection()
    for condition in LABELS:
        paths = e1c_paths() if condition == 'E1C' else {
            s: STAGE3 / ('%s_seed%d' % (condition, s)) / 'result.pkl' for s in range(3)}
        for seed, path in sorted(paths.items()):
            if not path.exists():
                runs.append(dict(condition=condition, seed=seed, path=str(path), records=[], complete=False))
                continue
            with path.open('rb') as f:
                raw = pickle.load(f)
            records = enrich(raw)
            rows_complete = len(records) == 140 and [r['itr'] for r in records] == list(range(140))
            complete = rows_complete and (condition != 'E1C' or e1c[seed]['verified_complete'])
            if complete:
                train = [r for r in records if r['update_number'] is not None]
                assert len(train) == 126 and train[-1]['update_number'] == 126
                assert next(r for r in records if r['itr'] == 130)['updates_before_rollout'] == 117
            runs.append(dict(condition=condition, seed=seed, path=str(path), records=records, complete=complete,
                             rows_complete=rows_complete, attempt=e1c[seed] if condition == 'E1C' else None))
    return runs


def summarize(run, j0):
    records = run['records']
    train = [r for r in records if r['update_number'] is not None]
    last = train[-10:] if run['complete'] else []
    ev = next((r for r in records if r['itr'] == 130), {})
    end = train[-1] if train and run['complete'] else {}
    first_eval = records[0].get('eval_episode_reward') if records else None
    nonfinite = [(r['itr'], path) for r in records for path in nonfinite_paths(r)]
    collapsed = bool(nonfinite) or any(r.get('eval_episode_reward', float('inf')) < first_eval / 2
                                      for r in records) if first_eval is not None else bool(nonfinite)
    kls = complete_values(train, 'kl_true_per_action', 126) if run['complete'] else None
    jd = strict_mean(last, 'J_disc_train', 10)
    early = {key: complete_values(train[:20], key, 20) for key in
             ('kl_true_per_action', 'J_disc_train', 'x0_sat_frac_mean')}
    return dict(condition=run['condition'], seed=run['seed'], complete=run['complete'],
                result_path=run['path'], iterations=len(records), collapse=collapsed,
                nonfinite=nonfinite, eval_936M=number(ev.get('eval_episode_reward')),
                train_last10=strict_mean(last, 'train_episode_reward', 10),
                J_disc_gain_last10=jd-j0 if jd is not None else None,
                median_KL=float(np.median(kls)) if kls is not None else None,
                theta_dist_final=number(end.get('theta_dist')),
                saturation_final=number(end.get('x0_sat_frac_mean')),
                early_updates_observed=min(20, len(train)),
                early_max_KL=max(early['kl_true_per_action']) if early['kl_true_per_action'] is not None else None,
                early_min_J_disc=min(early['J_disc_train']) if early['J_disc_train'] is not None else None,
                early_max_saturation=max(early['x0_sat_frac_mean']) if early['x0_sat_frac_mean'] is not None else None)


def curves(ax, runs, key, log=False):
    for condition in LABELS:
        series = []
        for run in runs:
            if run['condition'] != condition:
                continue
            row = {r['updates_before_rollout']: number(r.get(key)) for r in run['records'] if key in r}
            row = {x: y for x, y in row.items() if y is not None and (not log or y > 0)}
            if row:
                ax.plot(list(row), list(row.values()), color=COLORS[condition], lw=.6, alpha=.22)
            series.append(row)
        xs = sorted(set.intersection(*(set(s) for s in series))) if len(series) == 3 else []
        if xs:
            a = np.asarray([[s[x] for x in xs] for s in series])
            mu, sd = a.mean(0), a.std(0, ddof=1)
            ax.plot(xs, mu, color=COLORS[condition], lw=1.7, label=LABELS[condition])
            low = np.where(mu-sd > 0, mu-sd, np.nan) if log else mu-sd
            ax.fill_between(xs, low, mu+sd, color=COLORS[condition], alpha=.13)
    if log:
        ax.set_yscale('log')
    ax.set_xlabel('Actor updates before rollout')
    ax.grid(alpha=.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / (name + '.png'), dpi=170)
    fig.savefig(directory / (name + '.pdf'))
    plt.close(fig)


def contrasts(grouped):
    out = []
    for a, b in [('E0', 'E1C'), ('E1C', 'E1')]:
        for key in FIELDS[:3]:
            ma, sa = grouped[a][key]
            mb, sb = grouped[b][key]
            valid = all(x is not None for x in [ma, sa, mb, sb])
            delta = mb-ma if valid else None
            threshold = 2*math.sqrt((sa*sa+sb*sb)/2) if valid else None
            out.append(dict(before=a, after=b, metric=key, delta=delta, twice_pooled_sd=threshold,
                            interpretation=('unresolved' if abs(delta) < threshold else 'exceeds reading-rule threshold')
                            if valid else 'incomplete'))
    return out


def noise_snapshot():
    choices = []
    for p in (STAGE3 / 'gradmean/runs/analyses').glob('round_*/candidates/main_results.csv'):
        try:
            target = int(p.parent.parent.name.split('_')[1])
        except ValueError:
            continue
        status = p.parent.parent / 'analysis_status.json'
        if not status.exists() or read(status).get('status') not in ('complete', 'completed'):
            continue
        rows = list(csv.DictReader(p.open()))
        picked = [r for r in rows if r['point'] in ('P0', 'N1', 'N2') and r['variant'] in ('V1', 'V2', 'V3')]
        if len(picked) == 9 and len({(r['point'], r['variant']) for r in picked}) == 9:
            choices.append((target, p, picked))
    return max(choices, key=lambda x: x[0]) if choices else (None, None, [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish-report', action='store_true')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / 'analysis_status.json').exists():
        raise RuntimeError('Analysis output already exists; preserve it and select a new output directory')
    candidates = out / 'candidates'
    figures = candidates / 'figs'
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    complete = all(r['complete'] for r in runs)
    if not complete and not args.partial:
        raise RuntimeError('All three E1C seeds must finish before full analysis')
    j0_file = STAGE3 / 'ckpt_theta0_seed5000/ckpt_eval.json'
    j0 = read(j0_file)['samplers']['train']['J_disc']['mean']
    summaries = [summarize(run, j0) for run in runs]
    grouped = {c: {k: stats([s[k] for s in summaries if s['condition'] == c]) for k in FIELDS}
               for c in LABELS}
    csv_write(candidates / 'seed_results.csv', summaries)
    rows = []
    for c in LABELS:
        row = {'condition': c, 'label': LABELS[c]}
        for k, (mu, sd) in grouped[c].items():
            row[k+'_mean'], row[k+'_sd'] = mu, sd
        rows.append(row)
    csv_write(candidates / 'condition_results.csv', rows)
    long = [dict(condition=run['condition'], seed=run['seed'], **row) for run in runs for row in run['records']]
    csv_write(candidates / 'all_runs_long.csv', long)
    differences = contrasts(grouped)
    csv_write(candidates / 'contrasts.csv', differences)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, key, title in zip(axs, ['eval_episode_reward', 'train_episode_reward', 'J_disc_train'],
                            ['Evaluation return', 'Training return', 'Discounted training return']):
        curves(ax, runs, key)
        ax.set_title(title)
    references = read(STAGE3 / 'analysis_record.json')['reference_sources']['Stage 1 DPPO']
    for ax, key in zip(axs[:2], ['eval_episode_reward', 'train_episode_reward']):
        vals = []
        for p in references:
            with open(p, 'rb') as f:
                records = pickle.load(f)
            selected = [r[key] for r in records if key in r]
            vals.append(float(np.mean(selected[-1:] if key.startswith('eval') else selected[-10:])))
        ax.axhline(np.mean(vals), ls='--', lw=1, color='#888888', alpha=.55, label='DPPO final reference')
    axs[0].legend(fontsize=8)
    save_figure(fig, figures, 'returns')
    for name, panels in [
        ('updates', [('kl_true_per_action', 'KL per update', True), ('theta_dist', 'Actor distance from pretrained', False)]),
        ('saturation_direction', [('x0_sat_frac_mean', 'Pre-update x0 saturation fraction', False), ('grad_cos_prev', 'Cosine of consecutive gradients', False)]),
        ('baseline_quality', [('unweighted_energy_ratio', 'Unweighted critic / LOO residual energy', True), ('explained_var', 'Critic explained variance', False)])]:
        fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3))
        for ax, (key, title, log) in zip(axs, panels):
            curves(ax, runs, key, log)
            ax.set_title(title)
        axs[0].legend(fontsize=8)
        save_figure(fig, figures, name)
    windows = [(1, 10), (60, 70), (117, 126)]
    bins = []
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (lo, hi) in zip(axs, windows):
        arrays = []
        for run in runs:
            if run['condition'] != 'E1C':
                continue
            selected = [r for r in run['records'] if r['update_number'] is not None and lo <= r['update_number'] <= hi]
            vals = []
            for b in range(5):
                ec = mean([number(r.get('unweighted_energy_critic_bin%d' % b)) for r in selected])
                el = mean([number(r.get('unweighted_energy_loo_bin%d' % b)) for r in selected])
                ratio = ec/el if ec is not None and el is not None and el > 0 else None
                bins.append(dict(seed=run['seed'], updates='%d-%d' % (lo, hi), bin_start=50*b,
                                 critic_energy=ec, loo_energy=el, ratio=ratio))
                vals.append(ratio)
            if all(v is not None for v in vals):
                arrays.append(vals)
                ax.plot(range(5), vals, color=COLORS['E1C'], lw=.6, alpha=.25)
        if len(arrays) == 3:
            a = np.asarray(arrays)
            ax.errorbar(range(5), a.mean(0), yerr=a.std(0, ddof=1), color=COLORS['E1C'], marker='o', capsize=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['0–49', '50–99', '100–149', '150–199', '200–249'], rotation=25)
        ax.set_xlabel('Within-episode decision index k')
        ax.set_title('MC + critic: updates %d–%d' % (lo, hi))
        ax.set_yscale('log')
        ax.axhline(1, color='gray', lw=.7)
    axs[0].set_ylabel('Unweighted critic / LOO residual energy')
    save_figure(fig, figures, 'baseline_bins')
    csv_write(candidates / 'baseline_bins.csv', bins)
    target, noise_path, noise = noise_snapshot()
    csv_write(candidates / 'noise_reference.csv', noise)
    captions = '''# Figure definitions

All E0/E1C/E1 curves show three-seed mean ± one sample standard deviation (ddof=1), with thin individual seeds; means require all three available values. The x-axis counts actor updates applied before each rollout: training itr1 is x=0 and eval itr130 is x=117. A training diagnostic describes the update after that rollout, while returns and saturation concern its pre-update policy. Partial runs end where observed; no endpoint is extrapolated.

**Returns:** undiscounted evaluation/training episode returns and raw discounted train-sampler return, gamma=0.99 per four-action decision, horizon250. Faint dashed DPPO references are its three-seed final scheduled eval and last-ten-training-batch means, not an invented single-step optimizer history. Stage1 did not record discounted return or the other Stage3 diagnostics, so no DPPO values are fabricated in those panels.

**Updates:** true summed-log-prob KL per action (log axis) and post-update actor L2 distance from pretrained parameters. **Saturation/direction:** fraction of coordinates outside the unchanged x0 clamp before the update and cosine with the previous training gradient. Undefined first-gradient cosines are gaps.

**Baseline quality:** MC+critic's unweighted residual-energy ratio (log axis), derived as sum_k(mean_k²+population_std_k²) separately for critic and LOO; the original weighted energy fields are not used. Explained variance is DPPO's scaled-target critic diagnostic, not variance explained in raw Monte Carlo returns. **Baseline bins:** ratio of time-averaged residual energies within each k-bin and specified update window; error bars are sample sd across three seeds, not episode uncertainty.

These plots condition on three training seeds; small differences are unresolved under the user's twice-pooled-sd rule. Fixed-policy V1–V3 noise measurements use full-batch normalization and warmed/frozen critics and are proxies for the training estimators. Nonpositive log-axis values and lower SD bounds are omitted, not clipped to arbitrary positive constants.
'''
    (figures / 'captions.md').write_text(captions)
    lines = ['# E1C: Monte Carlo returns with a critic under AdamW', '',
             'Does the E0→E1 gain come from the return estimate or the baseline?', '',
             '**Status:** ' + ('All three E1C runs completed.' if complete else 'Partial results; at least one seed is incomplete.') +
             ' Differences below twice the pooled seed SD remain unresolved.', '',
             '| Condition | Eval 9.36M | Train last10 | J_disc gain last10 | Median KL | Final distance | Final saturation |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for c in LABELS:
        lines.append('| ' + LABELS[c] + ' | ' + ' | '.join(cell(grouped[c][k]) for k in FIELDS) + ' |')
    lines += ['', 'Each cell is mean ± sample SD over seeds. KL first takes the median across each seed’s training updates. J_disc gain subtracts the common saved pretrained train-sampler mean %.6f (240 episodes); that reference uncertainty is not a training-seed SD. Final saturation is the final training rollout’s pre-update saturation, not a new final-checkpoint evaluation.' % j0,
              '', '![Returns](figs/returns.png)', '', '[Figure definitions and limitations](figs/captions.md).', '', '**Q1 — Decomposing the gain.**']
    for d in differences:
        lines.append('- %s→%s, %s: difference %s; twice pooled SD %s; **%s**.' %
                     (LABELS[d['before']], LABELS[d['after']], d['metric'], fmt(d['delta']), fmt(d['twice_pooled_sd']), d['interpretation']))
    lines += ['', '**Q2 — Critic quality.**']
    for lo, hi in windows:
        values = []
        for b in range(5):
            values.append(cell(stats([r['ratio'] for r in bins if r['updates'] == '%d-%d' % (lo, hi) and r['bin_start'] == 50*b])))
        lines.append('- Updates %d–%d: critic/LOO energy by increasing k-bin: %s.' % (lo, hi, '; '.join(values)))
    lines += ['Settling is assessed descriptively from these windows and the full curve; no new settling threshold was introduced.',
              '', '**Q3 — First20updates.**']
    for c in ['E1C', 'E1']:
        selected = [s for s in summaries if s['condition'] == c]
        lines.append('- %s: maximum KL %s; minimum J_disc %s; maximum saturation %s (mean±SD of per-seed extrema).' %
                     (LABELS[c], cell(stats([s['early_max_KL'] for s in selected])),
                      cell(stats([s['early_min_J_disc'] for s in selected])), cell(stats([s['early_max_saturation'] for s in selected]))))
    lines += ['', '**Q4 — Fixed-policy noise.** Latest available complete analysis round: %s. V1/V2/V3 are frozen-policy proxies, not exact training gradients.' % target]
    for point in ['P0', 'N1', 'N2']:
        selected = [r for r in noise if r['point'] == point]
        entries = []
        for r in selected:
            bound = r.get('B_ep_type') != 'point'
            value = number(r.get('B_ep_lower_bound') if bound else r.get('B_ep'))
            entries.append('%s %s%s episodes (K=%s)' % (r['variant'], '≥' if bound else '', fmt(value), r['K']))
        lines.append('- %s: %s.' % (point, '; '.join(entries) or 'unavailable'))
    lines += ['', '[KL and distance](figs/updates.png) · [Saturation and direction](figs/saturation_direction.png) · [Baseline quality](figs/baseline_quality.png) · [Baseline bins](figs/baseline_bins.png).',
              '[Per-seed table](seed_results.csv) · [Condition table](condition_results.csv) · [Contrasts](contrasts.csv) · [Noise inputs](noise_reference.csv).',
              '', 'The methods, gate evidence, commands, source revisions and resource-sharing record are linked from the study record. The approved per-chunk normalization corrects the original addendum; unweighted energies are reconstructed from saved moments. No reference runs, checkpoint evaluations or extra experiments were added. These are candidate artifacts awaiting review.']
    report = candidates / 'REPORT.md'
    report.write_text('\n'.join(lines) + '\n')
    shutil.copy2(__file__, out / 'analysis_source.py')
    gate = ROOT / 'runs/gate/gate_summary.json'
    evidence = {'status': 'complete' if complete else 'partial', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_results': [r['path'] for r in runs], 'pretrained_reference': str(j0_file),
                'pretrained_J_disc': j0, 'noise_reference': str(noise_path) if noise_path else None,
                'gate': str(gate), 'sample_sd_ddof': 1, 'reading_rule': 'absolute difference < twice pooled seed SD is unresolved',
                'new_scientific_samples': 0, 'curated': False, 'report': str(report), 'source_snapshot': str(out/'analysis_source.py')}
    dump(out / 'analysis_status.json', evidence)
    if args.publish_report:
        destination = ROOT / 'REPORT.md'
        if destination.exists():
            shutil.copy2(destination, out / 'previous_REPORT.md')
        text = report.read_text()
        for name in ['figs/', 'seed_results.csv', 'condition_results.csv', 'contrasts.csv', 'noise_reference.csv']:
            prefix = os.path.relpath(candidates, ROOT) + '/'
            text = text.replace('](' + name, '](' + prefix + name)
        text += '\n[Study record](experiment.md) · [Methods](methods.md) · [Gate results](runs/gate/gate_summary.json) · [Commands and resource records](runs/campaign.json) · [Job file](jobs_e1c.txt) · [Revisions](provenance/revisions.json).\n'
        destination.write_text(text)
    print(json.dumps({'status': evidence['status'], 'report': str(report)}))


if __name__ == '__main__':
    main()
