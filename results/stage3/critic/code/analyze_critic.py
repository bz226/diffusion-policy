#!/usr/bin/env python
"""Addendum C analysis of native saved results; no sampling or curation."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent
NEW = ('SGDC3', 'SGDC10', 'SGDC3w', 'SGDG3', 'SGDG10')
REF = ('SGD3', 'SGD10', 'E0', 'E2')
CONDITIONS = NEW + REF
COLORS = dict(zip(CONDITIONS, ('#0072b2', '#d55e00', '#009e73', '#cc79a7', '#e69f00', '#555555', '#999999', '#56b4e9', '#882255')))
PAIRS = (('SGDC3', 'SGD3'), ('SGDC10', 'SGD10'))
WINDOWS = (('early_1_10', 1, 10), ('middle_60_70', 60, 70), ('late_130_140', 130, 140))
PANELS = (('eval_episode_reward', 'Eval return', False), ('train_episode_reward', 'Train return', False),
          ('J_disc_train', 'Discounted training return', False), ('kl_true_per_action', 'True transition KL per action', True),
          ('theta_dist', 'Distance from initial actor', False), ('B_env', 'Window noise scale B_env', True),
          ('grad_cos_prev', 'Consecutive gradient cosine', False), ('x0_sat_frac_mean', 'Saturation / action range', False))


def require(value, message):
    if not value:
        raise ValueError(message)


def helper():
    path = REFERENCE / 'code/analyze_stage3.py'
    module = types.ModuleType('critic_stage3_analysis')
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    module.CONDITIONS = list(CONDITIONS)
    return module


def specifications(root, names, a):
    result = []
    for name in names:
        for seed in range(3):
            run_id = '%s_seed%d' % (name, seed)
            spec = a.read_json(root / 'runs/specs' / (run_id + '.json'), {})
            if not spec:
                campaign = a.read_json(root / 'runs/campaign.json', {})
                spec = next((s for s in campaign.get('runs', []) if s.get('run_id') == run_id), {})
            if not spec:
                spec = dict(run_id=run_id, condition=name, seed=seed, mode='train', logdir=str(root / run_id))
            require(spec.get('condition') == name and spec.get('seed') == seed, 'Unexpected run specification: ' + run_id)
            result.append(dict(spec, mode='train'))
    return result


def mean(values, a):
    return float(np.mean(values)) if values and all(a.finite(x) for x in values) else None


def ratio(x, y, a):
    value = x / y if a.finite(x) and a.finite(y) and y > 0 else None
    return value if a.finite(value) else None


def load_checked(root, specs, a):
    """Render terminal failures without losing their paths or fabricating results."""
    runs = []
    for spec in specs:
        try:
            run = a.load_runs(root, {'runs': [spec]}, True)[0]
        except (ValueError, OSError, EOFError) as error:
            manifest_path = root / 'runs' / spec['run_id'] / 'manifest.json'
            manifest = a.read_json(manifest_path, {})
            evaluation_id = 'ckpt_' + spec['run_id'] + '_seed5000'
            run = dict(spec=spec, manifest=manifest, manifest_path=str(manifest_path),
                status='failed', rows=[], windows=a.window_noise([]),
                result_path=str(Path(spec['logdir']) / 'result.pkl'), read_error=str(error),
                checkpoints=[str(Path(spec['logdir']) / 'checkpoint' / ('state_%d.pt' % i)) for i in (0,35,70,105,139)],
                checkpoint_eval={}, checkpoint_eval_status='unavailable',
                checkpoint_eval_path=str(root / evaluation_id / 'ckpt_eval.json'))
            run['analysis_validation_errors'] = [type(error).__name__ + ': ' + str(error)]
        runs.append(run)
    return runs


def validate(runs, a):
    for run in runs:
        errors = run.setdefault('analysis_validation_errors', [])
        if a.numeric_bad(run.get('checkpoint_eval', {})):
            run['checkpoint_nonfinite_evidence'] = a.numeric_bad(run['checkpoint_eval'])
            errors.append('Checkpoint evaluation contains nonfinite evidence; final sampler values withheld')
            run['checkpoint_eval'] = {}
            run['checkpoint_eval_status'] = 'invalid_nonfinite'
        if run['status'] != 'complete':
            continue
        error_count_before_training = len(errors)
        rows = run['rows']
        if a.numeric_bad(rows):
            errors.append('Claimed complete run has nonfinite saved evidence')
        if sum('train_episode_reward' in row for row in rows) != 126:
            errors.append('Claimed complete run does not contain 126 training updates')
        if run['spec']['condition'] in NEW:
            for row in rows:
                if 'train_episode_reward' not in row:
                    continue
                for key in ('baseline_energy_critic', 'baseline_energy_loo', 'baseline_divisor'):
                    if not a.finite(row.get(key)):
                        errors.append('Missing/nonfinite baseline field %s at itr%s' % (key, row.get('itr')))
                for key in ('baseline_bin_energy_critic', 'baseline_bin_energy_loo'):
                    value = row.get(key)
                    if not isinstance(value, (list, tuple)) or len(value) != 5 or not all(a.finite(x) for x in value):
                        errors.append('Invalid baseline bins %s at itr%s' % (key, row.get('itr')))
        if len(errors) > error_count_before_training:
            # Preserve the immutable manifest and observed prefix. Only analysis
            # status changes, so absent endpoints cannot enter condition means.
            run['status'] = 'failed'


def extended_summaries(runs, theta0, a):
    result = []
    for run in runs:
        s = a.summarize(run, theta0)
        if run.get('checkpoint_nonfinite_evidence'):
            s.update(collapse=True, nonfinite=True,
                     checkpoint_nonfinite_evidence=run['checkpoint_nonfinite_evidence'])
        train = [row for row in run['rows'] if 'train_episode_reward' in row]
        first = train[:20]
        first_complete = len(first) == 20
        full_complete = run['status'] == 'complete' and len(run['rows']) == 140
        s.update(first20_update_count=len(first), first20_last_itr=first[-1]['itr'] if first else None,
                 first20_complete=first_complete,
                 first20_max_kl=max((r['kl_true_per_action'] for r in first), default=None) if first_complete and all(a.finite(r.get('kl_true_per_action')) for r in first) else None,
                 first20_min_J_disc=min((r['J_disc_train'] for r in first), default=None) if first_complete and all(a.finite(r.get('J_disc_train')) for r in first) else None,
                 first20_start_J_disc=first[0].get('J_disc_train') if first_complete else None,
                 first20_saturation=mean([r.get('x0_sat_frac_mean') for r in first], a) if first_complete else None,
                 first20_energy_ratio=mean([r.get('baseline_energy_ratio') for r in first], a) if first_complete else None,
                 explained_var_mean=mean([r.get('explained_var') for r in train], a) if full_complete else None,
                 baseline_energy_ratio_mean=mean([r.get('baseline_energy_ratio') for r in train], a) if full_complete else None,
                 analysis_validation_errors=run.get('analysis_validation_errors', []))
        if first and a.finite(s['first20_min_J_disc']) and a.finite(s['first20_start_J_disc']):
            s['first20_J_dip_from_start'] = s['first20_min_J_disc'] - s['first20_start_J_disc']
        else:
            s['first20_J_dip_from_start'] = None
        result.append(s)
    return result


def diagnostic_tables(runs, a):
    windows, bins, matched = [], [], []
    indexed = {(r['spec']['condition'], r['spec']['seed']): r for r in runs}
    for run in runs:
        name, seed = run['spec']['condition'], run['spec']['seed']
        for label, low, high in WINDOWS:
            rows = [r for r in run['rows'] if low <= r['itr'] <= high and 'train_episode_reward' in r]
            expected = [i for i in range(low, min(high, 139) + 1) if i % 10]
            complete = [r['itr'] for r in rows] == expected
            record = dict(condition=name, seed=seed, window=label, training_iterations=len(rows), expected_iterations=len(expected), complete=complete)
            for key in ('kl_true_per_action', 'baseline_energy_critic', 'baseline_energy_loo', 'baseline_energy_ratio', 'explained_var', 'J_disc_train'):
                record[key] = mean([r.get(key) for r in rows], a) if complete else None
            record['split_trace'] = mean([20 * r['split_diff_sq'] for r in rows if a.finite(r.get('split_diff_sq'))], a) if complete and all(a.finite(r.get('split_diff_sq')) for r in rows) else None
            windows.append(record)
            if name in NEW:
                for kbin in range(5):
                    record_bin = dict(condition=name, seed=seed, window=label, k_start=50*kbin, k_stop=50*(kbin+1), complete=complete)
                    for baseline in ('critic', 'loo'):
                        key = 'baseline_bin_energy_' + baseline
                        arrays = [r.get(key) for r in rows]
                        valid_arrays = all(isinstance(x, (list, tuple)) and len(x) == 5 for x in arrays)
                        record_bin['energy_' + baseline] = mean([x[kbin] for x in arrays], a) if complete and valid_arrays else None
                    record_bin['energy_ratio'] = ratio(record_bin['energy_critic'], record_bin['energy_loo'], a)
                    bins.append(record_bin)
    for left, right in PAIRS:
        for seed in range(3):
            lrun, rrun = indexed[(left, seed)], indexed[(right, seed)]
            right_rows = {r['itr']: r for r in rrun['rows'] if 'train_episode_reward' in r}
            for row in lrun['rows']:
                if 'train_episode_reward' not in row or row['itr'] not in right_rows:
                    continue
                ref = right_rows[row['itr']]
                matched.append(dict(condition=left, reference=right, seed=seed, itr=row['itr'], step=row['step'],
                    kl_critic=row.get('kl_true_per_action'), kl_loo=ref.get('kl_true_per_action'),
                    kl_ratio=ratio(row.get('kl_true_per_action'), ref.get('kl_true_per_action'), a),
                    split_trace_ratio=ratio(row.get('split_diff_sq'), ref.get('split_diff_sq'), a),
                    residual_energy_ratio=row.get('baseline_energy_ratio')))
    comparisons = []
    lookup = {(r['condition'], r['seed'], r['window']): r for r in windows}
    for left, right in PAIRS:
        for label, _, _ in WINDOWS:
            for seed in range(3):
                lrow, rrow = lookup[(left, seed, label)], lookup[(right, seed, label)]
                comparisons.append(dict(condition=left, reference=right, seed=seed, window=label,
                    kl_ratio=ratio(lrow['kl_true_per_action'], rrow['kl_true_per_action'], a),
                    split_trace_ratio=ratio(lrow['split_trace'], rrow['split_trace'], a),
                    residual_energy_ratio=ratio(lrow['baseline_energy_critic'], lrow['baseline_energy_loo'], a)))
    return windows, bins, matched, comparisons


def line_plot(ax, values, key, xkey, condition, color, a, log=False):
    indexed = {seed: {r[xkey]: r.get(key) for r in values if r.get('seed') == seed and a.finite(r.get(xkey))} for seed in range(3)}
    for seed in range(3):
        xs = sorted(indexed[seed])
        ys = [indexed[seed][x] if a.finite(indexed[seed][x]) and (not log or indexed[seed][x] > 0) else np.nan for x in xs]
        ax.plot(xs, ys, color=color, alpha=.22, lw=.7)
    xs = sorted(set.intersection(*(set(x) for x in indexed.values())))
    means, sds = [], []
    for x in xs:
        stat = a.mean_sd([indexed[seed].get(x) for seed in range(3)])
        m, s = stat['mean'], stat['sd']
        means.append(m if a.finite(m) and (not log or m > 0) else np.nan)
        sds.append(s if a.finite(s) else np.nan)
    xs, m, s = np.array(xs), np.array(means), np.array(sds)
    ax.plot(xs, m, label=condition, color=color, lw=1.6)
    low, high = m-s, m+s
    if log:
        low = np.where(low > 0, low, np.nan)
    ax.fill_between(xs, low, high, color=color, alpha=.12)


def figures(runs, windows, bins, matched, destination, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs, captions = [], []
    destination.mkdir()
    common = ('Thin lines are seeds0–2; bold lines and bands are the mean ± sample SD(ddof1), requiring all three finite seed values at each observation. '
              'No smoothing, imputation or survivor-only averages are used. Nonpositive values on log axes and nonpositive lower SD bounds are omitted, without epsilon substitution. '
              'The last scheduled evaluation is itr130 at9.36M training environment steps; training ends at10.08M. Three seeds leave small differences unresolved.')
    def save(fig, name, caption):
        for axis in fig.axes:
            if axis.get_yscale() == 'log' and (not np.isfinite(axis.dataLim.minposy) or axis.dataLim.minposy <= 0):
                # An empty plot is a normal terminal-failure outcome. These are
                # display limits only, never substituted numerical observations.
                axis.set_ylim(.1, 10.)
                axis.text(.5, .5, 'No positive measurements available', ha='center',
                          va='center', transform=axis.transAxes, fontsize=8)
        fig.tight_layout()
        for ext in ('png','pdf'):
            path = destination / (name+'.'+ext)
            fig.savefig(path, dpi=160, metadata={'Description':caption} if ext=='png' else {'Subject':caption})
            outputs.append(str(path))
        plt.close(fig)
        captions.append('## '+name+'\n\n'+caption+'\n')
    for family, names in (('critic_vs_loo', ('SGDC3','SGDC10','SGDC3w','SGD3','SGD10','E2')), ('gae_sgd', ('SGDG3','SGDG10','E0','SGD3','SGD10','E2'))):
        fig, axes = plt.subplots(2,4,figsize=(18,8))
        for ax,(metric,title,log) in zip(axes.flat,PANELS):
            for name in names:
                selected=[r for r in runs if r['spec']['condition']==name]
                a.plot_group(ax,selected,metric,'step',name,COLORS[name],logscale=log)
                if metric=='x0_sat_frac_mean':
                    a.plot_group(ax,selected,'action_oor_frac','step',name+' action',COLORS[name],linestyle='--')
                if metric=='B_env':
                    for run in selected:
                        bad=[w['step']/1e6 for w in run['windows'] if w['reason']=='nonpositive_split_dot' and a.finite(w.get('step'))]
                        ax.scatter(bad,[.025]*len(bad),marker='x',s=12,color=COLORS[name],transform=ax.get_xaxis_transform())
            if log: ax.set_yscale('log')
            ax.set_title(title); ax.set_xlabel('Training environment steps (millions)'); ax.grid(alpha=.18)
        axes.flat[0].legend(fontsize=8)
        save(fig,family,common+' B_env=10 sum(split_diff_sq)/sum(split_dot) over complete ten-iteration windows; nonpositive-denominator windows are crosses/gaps. Saturation is solid and executed action |a|>1 is dashed, using training rows only. Split estimates are approximate for coupled batches.')
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for name in NEW:
        selected=[r for r in runs if r['spec']['condition']==name]
        a.plot_group(axes[0],selected,'baseline_energy_ratio','itr',name,COLORS[name],logscale=True)
        a.plot_group(axes[1],selected,'explained_var','itr',name,COLORS[name])
    axes[0].set_yscale('log'); axes[0].axhline(1,color='gray',ls=':'); axes[0].set_ylabel('Critic / LOO residual energy'); axes[1].set_ylabel('Critic explained variance'); axes[1].set_xlabel('Iteration (training only)'); axes[0].legend(ncol=3)
    save(fig,'residual_energy_and_critic',common+' Energy is the mean over80 episodes of sum_k gamma^(2k)(G_k−b_k)^2, gamma0.99, using raw returns and the iteration-start scaled critic. LOO uses the same batch. Explained variance retains DPPO targets and pre-update values; no settling threshold is imposed.')
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
    for ax,(label,_,_) in zip(axes,WINDOWS):
        for name in NEW:
            subset=[r for r in bins if r['window']==label and r['condition']==name]
            line_plot(ax,subset,'energy_ratio','k_start',name,COLORS[name],a,True)
        ax.set_yscale('log'); ax.axhline(1,color='gray',ls=':'); ax.set_title(label); ax.set_xlabel('Decision-index bin start (width50)'); ax.grid(alpha=.18)
    axes[0].set_ylabel('Critic / LOO bin residual energy'); axes[0].legend(fontsize=8)
    save(fig,'residual_energy_bins',common+' Each seed value is the ratio of its summed/averaged critic and LOO energies in the indicated iteration window and50-decision bin, not the mean of individual ratios. Evaluation-only rows are excluded; each window contains nine training rows. Energy weights are gamma^(2k).')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for column,(left,right) in enumerate(PAIRS):
        for name in (left,right):
            selected=[r for r in runs if r['spec']['condition']==name]
            a.plot_group(axes[0,column],selected,'kl_true_per_action','itr',name,COLORS[name],logscale=True)
        line_plot(axes[1,column],[r for r in matched if r['condition']==left],'kl_ratio','itr',left+'/'+right,COLORS[left],a,True)
        for row in range(2): axes[row,column].set_yscale('log'); axes[row,column].grid(alpha=.18); axes[row,column].legend(fontsize=8)
        axes[1,column].axhline(1,color='gray',ls=':'); axes[1,column].set_xlabel('Training iteration')
    axes[0,0].set_ylabel('KL per action'); axes[1,0].set_ylabel('Index-matched seed KL ratio')
    save(fig,'matched_step_kl',common+' The pairs use the same fixed actor learning rate. Ratios match seed labels and iteration labels algebraically; original Gym initial observations are not guaranteed paired. Ratios are descriptive and can vary when the denominator is small.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,label in zip(axes,('early_1_10','late_130_140')):
        for n,name in enumerate(('SGDC3','SGD3','SGDC10','SGD10','SGDC3w')):
            ys=[next(r['split_trace'] for r in windows if r['condition']==name and r['seed']==seed and r['window']==label) for seed in range(3)]
            stat=a.mean_sd(ys)
            for y in ys:
                if a.finite(y) and y>0: ax.scatter(n,y,color=COLORS[name],alpha=.5,s=16)
            if a.finite(stat['mean']) and stat['mean']>0:
                lo=stat['sd'] if stat['mean']-stat['sd']>0 else 0.
                ax.errorbar(n,stat['mean'],yerr=np.array([[lo],[stat['sd']]]),fmt='o',color=COLORS[name],capsize=3)
        ax.set_xticks(range(5)); ax.set_xticklabels(('SGDC3','SGD3','SGDC10','SGD10','SGDC3w'),rotation=25); ax.set_yscale('log'); ax.set_title(label); ax.grid(alpha=.18)
    axes[0].set_ylabel('20 × mean split_diff_sq (code gradient units²)')
    save(fig,'early_late_split_noise',common+' The displayed trace proxy is20 times split_diff_sq averaged over each seed\'s nine training rows in the early/late window. It follows the prescribed split-half conversion; LOO/normalization coupling makes it approximate rather than an independently estimated covariance trace.')
    (destination/'captions.md').write_text('# Figure captions\n\n'+'\n'.join(captions))
    return outputs


def generate_report(a,grouped,windows,bins,comparisons,gate,calibration,runs,candidates=None):
    def cell(c,m): return a.cell(grouped,c,m)
    def table(headers,rows): return a.md_table(headers,rows)
    def condition_metric(c,m): return cell(c,m)
    lines=['# Plain SGD with a critic baseline','',
      '**Question.** At fixed SGD rates, what does a learned state-only baseline cost or buy relative to LOO, and does DPPO\'s estimator work without Adam?','',
      '**Status.** %d of15 new training runs completed; %d failed. Missing results remain unavailable, and no survivor-only condition means are reported.' % (sum(r['status']=='complete' for r in runs if r['spec']['condition'] in NEW),sum(r['status']=='failed' for r in runs if r['spec']['condition'] in NEW)),'',
      '**Interpretation.** Means and sample SDs use three seeds. Differences smaller than twice the pooled seed SD are unresolved; no further significance tests are performed. The normalized SGDG estimator has different units from E2/SGDC. Critic/LOO noise comparisons at equal SGD learning rates use identical E2 score coefficients apart from the baseline.','',
      '## Gate and calibration','', 'C1–C5 status: **%s**. [Full gate record](%s).' % (gate.get('status','unavailable'),
          os.path.relpath(ROOT/'runs/gate/gate_summary.json',candidates) if candidates else '../../../runs/gate/gate_summary.json'),'',
      'Existing E2 eta*=0.00012107235674859004; KL*=0.000584517336086918. calG eta*_G=%s; max/min spread=%s; exponents=%s. Calibration warnings: %s.' % (a.fmt(calibration.get('eta_star')),a.fmt(calibration.get('eta_spread_max_over_min')),', '.join(a.fmt(r.get('exponent')) for r in calibration.get('batches',[])),calibration.get('warnings',[])+['batch%d: %s'%(r.get('batch',-1),w) for r in calibration.get('batches',[]) for w in r.get('warnings',[])]),
      'The warmed critic/scaler are frozen during five calibration batches. The actual per-shuffled-chunk-normalized SGDG gradient is used; no universal eta/2500 conversion applies to it.','',
      '## Main results','',table(['Condition','Eval itr0','Eval itr130','Last10 train','Last10 J_disc','Final train-sampler J gain','Median KL','Distance'],[[c]+[cell(c,m) for m in ('eval_itr0','eval_itr130','train_last10','J_disc_last10','delta_train_J_disc','median_kl','theta_dist_final')] for c in CONDITIONS]),'',
      '[Complete seed/condition table](condition_results.csv) includes both checkpoint samplers, saturation, update-zero fractions, B_env, timing, collapse and all required per-seed values. [Seed records and output paths](seed_results.csv).','',
      '## Q1. Critic versus LOO at equal step sizes','']
    for left,right in PAIRS:
        lines.append('%s versus %s: final train-sampler discounted gain %s versus %s; eval %s versus %s; median KL %s versus %s; distance %s versus %s; mean x0 saturation %s versus %s.' % (left,right,cell(left,'delta_train_J_disc'),cell(right,'delta_train_J_disc'),cell(left,'eval_itr130'),cell(right,'eval_itr130'),cell(left,'median_kl'),cell(right,'median_kl'),cell(left,'theta_dist_final'),cell(right,'theta_dist_final'),cell(left,'x0_saturation_mean'),cell(right,'x0_saturation_mean')))
        for metric in ('delta_train_J_disc','eval_itr130'):
            v=a.comparison(grouped,right,left,metric)
            lines.append('%s difference=%s; two-pooled-SD boundary=%s: %s.'%(metric,a.fmt(v['difference']),a.fmt(v['threshold']),v['interpretation']))
    lines+=['','![Critic versus LOO](figs/critic_vs_loo.png)','','## Q2. Noise and residual energy','',
      'The following ratios compare each SGDC seed with the same-index SGD reference; they are descriptive ratios, not paired causal estimates. Energy compares critic/LOO on the same new batch. These proxies need not be numerically equal because score magnitude and residuals can be correlated.','']
    rows=[]
    for left,right in PAIRS:
        for label,_,_ in WINDOWS:
            selection=[r for r in comparisons if r['condition']==left and r['window']==label]
            rows.append([left+'/'+right,label]+[a.estimate_text(a.mean_sd([next(r[k] for r in selection if r['seed']==s) for s in range(3)])) for k in ('kl_ratio','split_trace_ratio','residual_energy_ratio')])
    lines += [table(['Comparison','Window','KL ratio','Split trace ratio','Residual energy ratio'],rows),'',
      '[Per-bin residual ratios](baseline_bins.csv) identify where the critic exceeds LOO; energies sum gamma^(2k) residual² within each bin. [Per-k residual mean/std](all_runs_long.csv) retain the episode aggregation before binning.','',
      '![Residual bins](figs/residual_energy_bins.png)','','## Q3. Cold start','',table(['Condition','Max KL first20','J dip from first update','Mean saturation first20','Mean energy ratio first20'],[[c]+[cell(c,k) for k in ('first20_max_kl','first20_J_dip_from_start','first20_saturation','first20_energy_ratio')] for c in ('SGDC3','SGDC3w')]),'',
      'First20 counts actor updates, excluding evaluation iterations. No settling-time threshold was prescribed; the early/middle/late energy rows and full trajectory show stabilization or continued drift without inventing one.','',
      '![Residual energy and explained variance](figs/residual_energy_and_critic.png)','','## Q4. DPPO estimator with SGD','']
    for c in ('SGDG3','SGDG10','E0','SGD3','SGD10'):
        lines.append('%s: actor lr %s; eval %s; final train-sampler J gain %s; median KL %s; collapse flags %s.'%(c,cell(c,'actor_lr'),cell(c,'eval_itr130'),cell(c,'delta_train_J_disc'),cell(c,'median_kl'),grouped[c]['collapse_per_seed']))
    for c in ('SGDG3','SGDG10'):
        for ref in ('E0','SGD3','SGD10'):
            v=a.comparison(grouped,ref,c,'delta_train_J_disc')
            lines.append('%s−%s discounted gain: %s; boundary %s; %s.'%(c,ref,a.fmt(v['difference']),a.fmt(v['threshold']),v['interpretation']))
    lines += ['','![GAE with SGD](figs/gae_sgd.png)','','## Q5. Deviations table addition','',
      table(['Choice','DPPO / E0','Paper','Addendum C','Measured effect'],[['Actor baseline','Critic inside GAE, AdamW','None; a state-only baseline preserves the score mean','SGDC: raw MC−cV under SGD; SGDG: original GAE under SGD; SGDC3w: warm start',
          'Final discounted gains: SGDC3 %s vs SGD3 %s; SGDC10 %s vs SGD10 %s. DPPO estimator: SGDG3 %s and SGDG10 %s vs E0 %s. E0 versus E2 changes more than the baseline, so an isolated baseline effect under Adam was not measured.' % tuple(cell(c,'delta_train_J_disc') for c in ('SGDC3','SGD3','SGDC10','SGD10','SGDG3','SGDG10','E0'))]]),'',
      '## Artifacts, execution and limits','',
      '[All per-iteration data](all_runs_long.csv) · [Cold-start and seed summaries](seed_results.csv) · [Window diagnostics](baseline_windows.csv) · [Matched-step ratios](matched_step_ratios.csv) · [Table definitions](table_captions.md) · [Figure captions](figs/captions.md) · [Commands and provenance](../analysis_record.json).','',
      'The main workspace stayed on stage3-gradmean; source and commits are isolated on stage3-critic. G retained priority. CalG preceded three-seed cohorts; its allocation could not overlap a full cohort under the three-GPU cap. Existing warmed P0 and all reference runs were reused. Diagnostics use raw rewards and the iteration-start critic divisor; critic updates stay unchanged. Final evaluation reuses seed5000, but inherited Gym initialization does not ensure paired initial states. All artifacts remain uncurated. During queue verification, a synthetic status fixture reached the live campaign path before any production admission; it was preserved as test evidence and the test isolation was corrected and verified. The scientific gate and source were unaffected.']
    failures = [(r['spec']['run_id'],r['status'],r.get('read_error'),r.get('analysis_validation_errors',[]),
                 r['manifest'].get('failure_reason')) for r in runs
                if r['status']=='failed' or r.get('read_error') or r.get('analysis_validation_errors')]
    if failures:
        lines += ['', '## Incomplete or invalid evidence', '', table(
            ['Run','Analysis status','Read error','Validation findings','Recorded failure'], failures)]
    if gate.get('passed') is not True:
        lines.insert(4, '**Gate not passed.** No new campaign result is established. This is a partial failure/status report; reference data are shown only as existing context.\n')
    return '\n'.join(lines)+'\n'


def maintain_completed_catalog(runs, a, gate_passed, partial, publish_report):
    """Add the finished study once; never modify existing entries or curate."""
    if partial or not publish_report or not gate_passed:
        return {'status':'not_eligible','reason':'requires a published nonpartial analysis and passed gate'}
    selected = [r for r in runs if r['spec']['condition'] in NEW]
    expected = {(name,seed) for name in NEW for seed in range(3)}
    actual = [(r['spec']['condition'],r['spec']['seed']) for r in selected]
    if len(actual)!=15 or set(actual)!=expected:
        return {'status':'not_eligible','reason':'requires exactly the fifteen requested training runs'}
    for run in selected:
        evaluation = run.get('checkpoint_eval',{})
        if (run['status']!='complete' or run['manifest'].get('status')!='complete'
                or len(run['rows'])!=140 or a.numeric_bad(run['rows'])
                or run.get('read_error') or run.get('analysis_validation_errors')
                or not Path(run['result_path']).is_file()
                or len(run['checkpoints'])!=5 or not all(Path(p).is_file() for p in run['checkpoints'])
                or run['checkpoint_eval_status']!='complete'
                or not Path(run['checkpoint_eval_path']).is_file()
                or evaluation.get('status')!='completed' or a.numeric_bad(evaluation)):
            return {'status':'not_eligible','reason':'training or final evaluation not verified complete: '+run['spec']['run_id']}
        for sampler in ('train','eval'):
            result=evaluation.get('samplers',{}).get(sampler,{})
            for metric in ('undiscounted_return','J_disc'):
                summary=result.get(metric,{})
                if summary.get('n')!=80 or not all(a.finite(summary.get(k)) for k in ('mean','se')):
                    return {'status':'not_eligible','reason':'missing eighty-episode sampler result: '+run['spec']['run_id']}
    project=ROOT.parents[2]
    catalog=project/'complete_experiment.md'
    require(ROOT==project/'results/stage3/critic','Unexpected study location for completed-experiment links')
    require(catalog.is_file() and (ROOT/'methods.md').is_file() and (ROOT/'experiment.md').is_file(),
            'Completed catalog, experiment record or study methods unavailable')
    title='### Stage 3 plain SGD with a critic baseline (Addendum C)'
    entry=(title+'\n\n'
           '- Experiment record: [stage3/critic](./results/stage3/critic/experiment.md)\n'
           '- Experiment design: [Critic-baseline methods](./results/stage3/critic/methods.md)\n'
           '- Curated: —\n\n')
    # Lock the existing file while preserving every byte outside the inserted
    # entry. No separate lock file, replacement catalog, or artifact is created.
    import fcntl
    with catalog.open('r+',encoding='utf-8') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        text=stream.read()
        if title in text or re.search(r'\]\((?:\./)?results/stage3/critic(?:/|\))',text):
            return {'status':'already_present','path':str(catalog)}
        experiments=re.search(r'^## Experiments\s*$',text,re.MULTILINE)
        comparison=re.search(r'^## Comparison analyses\s*$',text,re.MULTILINE)
        require(experiments is not None and comparison is not None and experiments.start()<comparison.start(),
                'Completed catalog lacks the expected experiment/comparison structure')
        updated=text[:comparison.start()]+entry+text[comparison.start():]
        stream.seek(0);stream.write(updated);stream.truncate();stream.flush();os.fsync(stream.fileno())
    return {'status':'added','path':str(catalog),'curated':False,
            'verified_training_runs':15,'verified_checkpoint_evaluations':15}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--publish-report',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    require(ROOT in output.parents and not output.exists(),'Use a fresh analysis output directory inside this study')
    a=helper()
    specs=specifications(ROOT,NEW,a)
    refs=specifications(REFERENCE,REF,a)
    runs=load_checked(ROOT,specs,a)+load_checked(REFERENCE,refs,a)
    validate(runs,a)
    theta0=a.read_json(REFERENCE/'ckpt_theta0_seed5000/ckpt_eval.json',{})
    gate=a.read_json(ROOT/'runs/gate/gate_summary.json',{})
    if theta0.get('status')!='completed' or a.numeric_bad(theta0):
        theta0={}
    calibration=a.read_json(ROOT/'calG_seed3001/calibration.json',{})
    summaries=extended_summaries(runs,theta0,a)
    extra=('first20_max_kl','first20_min_J_disc','first20_J_dip_from_start','first20_saturation','first20_energy_ratio','explained_var_mean','baseline_energy_ratio_mean')
    a.METRICS=list(a.METRICS)+list(extra)
    grouped=a.aggregate(summaries)
    windows,bins,matched,comparisons=diagnostic_tables(runs,a)
    output.mkdir(parents=True)
    candidates=output/'candidates';candidates.mkdir()
    record={'status':'running','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'command':[sys.executable]+sys.argv,
      'partial':args.partial,'curated':False,'reference_root':str(REFERENCE),'source_policy':'native evidence only; reference code copied below',
      'gate_passed':gate.get('passed') is True,'initial_policy_evaluation_available':bool(theta0),
      'runs':[dict(spec=r['spec'],manifest=r['manifest'],analysis_status=r['status'],
          analysis_validation_errors=r.get('analysis_validation_errors',[]),result_path=r['result_path'],checkpoint_eval_path=r['checkpoint_eval_path']) for r in runs]}
    a.write_json(output/'analysis_record.json',record)
    try:
        source=output/'source';source.mkdir()
        shutil.copy2(__file__,source/'analyze_critic.py');shutil.copy2(REFERENCE/'code/analyze_stage3.py',source/'analyze_stage3_reference.py')
        flat=[]
        for condition,g in grouped.items():
            row={'condition':condition,'status_per_seed':g['statuses'],'collapse_per_seed':g['collapse_per_seed'],
                 'wall_span_seconds':g['wall_span_seconds']}
            for key,v in g['metrics'].items():
                row.update({key+'_mean':v['mean'],key+'_sd':v['sd'],key+'_per_seed':v['per_seed'],key+'_n':v['n_available']})
            flat.append(row)
        tables={'seed_results':summaries,'condition_results':flat,'baseline_windows':windows,'baseline_bins':bins,'matched_step_ratios':matched,'window_ratios':comparisons,
          'all_runs_long':[dict(run=r['spec']['run_id'],condition=r['spec']['condition'],seed=r['spec']['seed'],**row) for r in runs for row in r['rows']],
          'noise_windows':[dict(run=r['spec']['run_id'],condition=r['spec']['condition'],seed=r['spec']['seed'],**row) for r in runs for row in r['windows']]}
        for name,rows in tables.items():a.write_csv(candidates/(name+'.csv'),rows)
        a.write_json(candidates/'results.json',{'conditions':grouped,'summaries':summaries,'window_ratios':comparisons,'calibration':calibration})
        (candidates/'table_captions.md').write_text('''# Table definitions

All condition means and sample SDs require three finite seed values; missing or failed runs retain observed prefixes without survivor means. Seed results use the existing Stage3 definitions: last10 means the last10 training rows of a complete140-row run, collapse is eval below half initial eval or any recorded nonfinite, and final checkpoint columns come from80 episodes per sampler after iteration139. Delta J subtracts the saved pretrained sampler mean. See native result/checkpoint paths in seed_results.csv and exact manifests in analysis_record.json.

Baseline windows use training iteration labels1–10,60–70,130–140, excluding evaluation-only rows; each complete window has nine training updates. Energies are episode means of weighted squared residual sums, and per-bin energy ratios divide window-averaged energies. Split trace is20 times mean split_diff_sq in code gradient units squared, an approximate split-half conversion. Matched-step ratios align seed/iteration labels but do not assert paired environment trajectories. Zero or missing denominators give unavailable ratios, never artificial finite values. Cold-start statistics use the first20 actual updates, ending at iteration22. Per-k residual means/std are over episodes at fixed k; bin summaries retain their stated separate aggregation.
''')
        artifacts=figures(runs,windows,bins,matched,candidates/'figs',a)
        report=generate_report(a,grouped,windows,bins,comparisons,gate,calibration,runs,candidates)
        (candidates/'REPORT.md').write_text(report)
        if args.publish_report:
            old=ROOT/'REPORT.md'
            if old.exists():shutil.copy2(old,output/'previous_REPORT.md')
            prefix=str(candidates.relative_to(ROOT))+'/'
            rebased=re.sub(r'\]\(([^)]+)\)',lambda m:']('+os.path.relpath((candidates/m.group(1)).resolve(),ROOT)+')' if not re.match(r'^[a-z]+://',m.group(1)) else m.group(0),report)
            temporary=ROOT/'.REPORT.analysis.tmp';temporary.write_text(rebased);temporary.replace(old)
        record.update(status='complete',ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),figure_artifacts=artifacts,
          all_new_runs_completed=all(r['status']=='complete' for r in runs if r['spec']['condition'] in NEW),
          all_new_checkpoint_evaluations_completed=all(r['checkpoint_eval_status']=='complete' for r in runs if r['spec']['condition'] in NEW))
        record['completed_catalog']=maintain_completed_catalog(runs,a,gate.get('passed') is True,
                                                               args.partial,args.publish_report)
    except BaseException as error:
        record.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:a.write_json(output/'analysis_record.json',record)
    print(json.dumps({'status':record['status'],'output':str(output),'partial':args.partial}))

if __name__=='__main__':main()
