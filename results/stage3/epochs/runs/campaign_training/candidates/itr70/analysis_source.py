#!/usr/bin/env python
"""Read-only EP comparison; no rollouts or training (Python 3.8)."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import re
import shutil
import time
from datetime import datetime, timezone

import numpy as np

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/epochs'
HELPER = PROJECT / 'results/stage3/dl1/code/analyze.py'
spec = importlib.util.spec_from_file_location('dl1_saved_results', str(HELPER))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
CONDITIONS = {
    'EP1': ('DPPO, 1 epoch', 1, 1e-4),
    'EP1M': ('DPPO, 1 epoch, lr 5e-4', 1, 5e-4),
    'EP2': ('DPPO, 2 epochs', 2, 1e-4),
    'EP2M': ('DPPO, 2 epochs, lr 2.5e-4', 2, 2.5e-4),
}
COLORS = {'DPPO': '#084594', 'without_PPO': '#D55E00', 'without_PPO, lr 1e-3': '#D55E00'}
for ident, (label, k, lr) in CONDITIONS.items():
    COLORS[label] = '#6BAED6' if k == 1 else '#2171B5'
SAME = ['DPPO', CONDITIONS['EP1'][0], CONDITIONS['EP2'][0], 'without_PPO']
MATCHED = ['DPPO', CONDITIONS['EP1M'][0], CONDITIONS['EP2M'][0], 'without_PPO, lr 1e-3']
METRICS = ('eval_936M', 'eval_last5', 'train_last10', 'J_disc_gain_last10',
           'theta_dist_final', 'mean_last_actor_approx_kl', 'mean_last_actor_switched_off',
           'mean_last_actor_abs_ratio', 'saturation_final')


def read_rows(path):
    # result.pkl is written non-atomically by the unchanged native trainer.
    for attempt in range(3):
        try:
            with path.open('rb') as stream:
                rows = pickle.load(stream)
            break
        except (OSError, EOFError, pickle.UnpicklingError):
            if attempt == 2:
                raise
            time.sleep(0.2)
    if [r['itr'] for r in rows] != list(range(len(rows))) or len(rows) > 140:
        raise ValueError('Invalid iteration sequence: ' + str(path))
    previous_distance = 0.0
    for row in rows:
        train = bool(row['itr'] % 10)
        before = h.before(row['itr'])
        if row['step'] != 80000 * (before + int(train)):
            raise ValueError('Incorrect rollout/update mapping: ' + str(path))
        row['updates_before_rollout'] = before
        row['updates_after_iteration'] = before + int(train)
        row['rollout_theta_dist'] = row.get('theta_dist_before_update', previous_distance)
        previous_distance = row.get('theta_dist', previous_distance if not train else None)
    return rows


def own_run(ident):
    label, k, lr = CONDITIONS[ident]
    directory = ROOT / (ident + '_seed0')
    path = directory / 'result.pkl'
    read_error = None
    try:
        rows = read_rows(path) if path.is_file() else []
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        rows, read_error = [], repr(error)
    manifest_path = ROOT / 'runs/campaign_training/workers' / (ident + '_seed0') / 'manifest.json'
    manifest = h.read_json(manifest_path, {})
    checkpoints = {str(i): str(directory / 'checkpoint' / ('state_%d.pt' % i))
                   for i in h.CHECKPOINTS if (directory / 'checkpoint' / ('state_%d.pt' % i)).is_file()}
    complete = (len(rows) == 140 and len(checkpoints) == 5 and manifest.get('status') == 'complete'
                and manifest.get('exit_code') == 0 and manifest.get('child_processes_after') == [])
    return dict(method=label, seed=0, rows=rows, training=[r for r in rows if r['itr'] % 10],
                evaluation=[r for r in rows if not r['itr'] % 10], run_dir=str(directory),
                result_path=str(path), status='temporarily_unreadable' if read_error else ('complete' if complete else manifest.get('status', 'pending')),
                read_error=read_error,
                last_itr=rows[-1]['itr'] if rows else None, checkpoints=checkpoints,
                worker_manifest=str(manifest_path), worker=manifest, actor_epochs=k, actor_lr=lr,
                actor_steps=4*k, movement=k*lr/(5e-4), movement_assumption='first-order k*lr match')


def last_kl(run):
    """The native console logs every minibatch; retain the last value, not mean."""
    file = Path(run['run_dir']) / 'run.log'
    if not file.is_file():
        return
    scalar = re.compile(r'approx_kl: ([-+0-9.eE]+), update_epoch: (\d+), num_batch: (\d+)')
    summary = re.compile(r' - (\d+): step\s+\d+\s*\|')
    pending, found = [], {}
    for line in file.read_text(errors='replace').splitlines():
        m = scalar.search(line)
        if m:
            pending.append(float(m.group(1)))
        m = summary.search(line)
        if m:
            if pending:
                found[int(m.group(1))] = pending[-1]
            pending = []
    for row in run['training']:
        if row['itr'] in found:
            row['ppo_approx_kl_last'] = found[row['itr']]


def references():
    dppo = [h.load_run(PROJECT, PROJECT/'results/stage3/dl1', 'DPPO', seed) for seed in (0,1,2)]
    wp = h.load_run(PROJECT, ROOT, 'without_PPO', 0)
    for run in dppo:
        run.update(actor_epochs=5, actor_steps=20, actor_lr=1e-4, movement=1.0,
                   movement_assumption='first-order k*lr match')
        last_kl(run)
    wp.update(actor_epochs=None, actor_steps=1, actor_lr=1e-4, movement=0.1,
              movement_assumption='assumes quarter-batch sqrt(v) is twice full-batch sqrt(v)')
    directory = PROJECT / 'results/stage2_noclip/runs/NC4_lr1e-3_seed0/native'
    extra = None
    if (directory/'result.pkl').is_file():
        rows = read_rows(directory/'result.pkl')
        checkpoints = {str(i): str(directory/'checkpoint'/('state_%d.pt'%i)) for i in h.CHECKPOINTS
                       if (directory/'checkpoint'/('state_%d.pt'%i)).is_file()}
        extra = dict(method='without_PPO, lr 1e-3', seed=0, run_dir=str(directory),
                     result_path=str(directory/'result.pkl'), rows=rows, training=[r for r in rows if r['itr']%10],
                     evaluation=[r for r in rows if not r['itr']%10], checkpoints=checkpoints,
                     status='complete' if len(rows)==140 and len(checkpoints)==5 else 'incomplete',
                     actor_epochs=None, actor_steps=1, actor_lr=1e-3, movement=1.0,
                     movement_assumption=wp['movement_assumption'], last_itr=rows[-1]['itr'])
    return dppo, wp, extra


def metrics(run, distances):
    result = h.seed_metrics(run, distances)
    for output, key in (('mean_last_actor_approx_kl', 'ppo_approx_kl_last'),
                        ('mean_last_actor_switched_off', 'ppo_switched_off_frac_last'),
                        ('mean_last_actor_abs_ratio', 'ppo_mean_abs_ratio_minus_one_last')):
        result[output] = h.strict_mean([r.get(key) for r in run['training']]) if run['status']=='complete' else None
    result.update(actor_epochs=run['actor_epochs'], actor_steps_per_batch=run['actor_steps'],
                  actor_lr=run['actor_lr'], movement_relative_to_dppo=run['movement'],
                  movement_assumption=run['movement_assumption'],
                  minimum_eval=h.number(min([r['eval_episode_reward'] for r in run['evaluation']], default=float('nan'))),
                  initial_eval=h.number(run['evaluation'][0]['eval_episode_reward']) if run['evaluation'] else None)
    return result


def get_series(run, key, eval_only=False, after=False):
    records = run['evaluation'] if eval_only else run['training']
    return [(r['updates_after_iteration' if after else 'updates_before_rollout'], r[key])
            for r in records if h.finite(r.get(key))]


def figures(runs, distances, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    folder = output/'figs'
    folder.mkdir()
    def style(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(axis='y',color='#dddddd',linewidth=.5); ax.set_axisbelow(True)
    def line(ax, method, points):
        if points:
            ax.plot([p[0] for p in points],[p[1] for p in points],color=COLORS[method],
                    marker='o',markersize=2,linewidth=1,linestyle='--' if ', lr' in method else '-',label=method)
    fig, axes = plt.subplots(3,2,figsize=(8.0,8.0),sharex=True)
    for col, methods in enumerate((SAME, MATCHED)):
        for row,(key,ylabel) in enumerate((('eval_episode_reward','Eval reward (undiscounted return)'),
                                         ('train_episode_reward','Train return (undiscounted return)'),
                                         ('J_disc_train','J_disc (raw reward units)'))):
            ax=axes[row,col]
            for method in methods:
                if method in runs:
                    line(ax,method,get_series(runs[method],key,row==0))
            style(ax,'Updates (one update = 80 episodes)',ylabel)
            if row==0:
                ax.set_title(('Same lr (1e-4)','Matched movement per batch')[col])
                ax.legend(frameon=False,fontsize=6)
    fig.tight_layout()
    caption=("Seed 0. DPPO with its actor trained for 1, 2 or 5 epochs per batch (5 = DPPO). "
             "Left: all at DPPO's lr (1e-4), so fewer epochs also means less movement per batch. "
             "Right: lr scaled so that the first-order movement per batch matches DPPO's; only the reuse of pairs differs. "
             "x = updates; each update uses 80 new episodes. "
             "without_PPO, lr 1e-3 (Stage 2) is a one-step run with about DPPO's movement per batch; it also lacks the log-prob clamp and the noise clip. "
             "Historical DPPO and without_PPO did not log J_disc; those curves are absent. No uncertainty is shown.")
    h.save_figure(fig,folder,'epochs_curves_seed0',caption);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.4))
    for ax,methods,title in zip(axes,(SAME,MATCHED),('Same lr (1e-4)','Matched movement per batch')):
        for method in methods:
            if method in runs:
                values=h.distance_series(runs[method],distances[method])
                line(ax,method,[values[x] for x in sorted(values)])
        style(ax,'Distance from own start ‖θ − θ₀‖ (parameter units)','J_disc (raw reward units)')
        ax.set_title(title);ax.legend(frameon=False,fontsize=6)
    fig.tight_layout()
    caption=("Seed 0. J_disc against each run's distance from its own starting parameters: runs on the same curve differ only in how far they moved. "
             "Distances correspond to the policy before the plotted rollout. DPPO and without_PPO lack J_disc and are omitted; their checkpoints alone cannot reconstruct this comparison. "
             "New runs log distances every iteration. No uncertainty is shown.")
    h.save_figure(fig,folder,'epochs_distance_seed0',caption);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.2))
    for ax,key,title,ylabel in zip(axes,
            ('ppo_switched_off_frac_last','ppo_approx_kl_last','ppo_mean_abs_ratio_minus_one_last'),
            ('Pairs switched off','Approximate KL','Mean |ρ − 1|'),
            ('Fraction of pairs','PPO approximate KL (no units)','Mean |ρ − 1| (no units)')):
        for method in ['DPPO']+[v[0] for v in CONDITIONS.values()]:
            line(ax,method,get_series(runs[method],key,after=True))
        style(ax,'Updates (one update = 80 episodes)',ylabel);ax.set_title(title)
    axes[1].legend(frameon=False,fontsize=6)
    fig.tight_layout()
    caption=("Seed 0. Clip activity at the last actor step of each update. Historical DPPO retained last-step approximate KL in its console log, "
             "but not switched-off fraction or mean ratio displacement; those reference curves are absent. "
             "Points are estimates, without uncertainty shown; dashed lines identify the matched learning rates. Diagnostics use the pre-step policy at the last applied actor step.")
    h.save_figure(fig,folder,'epochs_ppo_diagnostics_seed0',caption);plt.close(fig)


def interim_note(runs, output):
    """Report the exact requested checkpoints, never silently the latest row."""
    rows = []
    for method, run in runs.items():
        by_itr = {r['itr']: r for r in run['rows']}
        evaluation, training = by_itr.get(70, {}), by_itr.get(69, {})
        rows.append(dict(method=method, evaluation_itr=70, training_itr=69,
                         eval_itr70=h.number(evaluation.get('eval_episode_reward')),
                         train_itr69=h.number(training.get('train_episode_reward')),
                         J_disc_itr69=h.number(training.get('J_disc_train')),
                         available_last_itr=run.get('last_itr'), read_error=run.get('read_error'),
                         run_dir=run['run_dir'], result_path=run['result_path']))
    h.write_csv(output/'interim_itr70.csv', rows)
    header=['Method', 'Eval, itr 70', 'Train return, itr 69', 'J_disc, itr 69']
    table=h.markdown(header, [[r['method'], h.fmt(r['eval_itr70']), h.fmt(r['train_itr69']),
                              h.fmt(r['J_disc_itr69'])] for r in rows])
    lines=['# Seed-0 interim comparison', '',
           'Evaluation is from iteration 70 (63 prior updates); training return and J_disc are from iteration 69 (62 prior updates). '
           'Every value comes from that exact iteration. Historical DPPO and without_PPO did not log J_disc.', '', table, '']
    for reference in ('DPPO', 'without_PPO'):
        old=next(r for r in rows if r['method']==reference)
        pieces=[]
        for ident, (label, _, _) in CONDITIONS.items():
            new=next(r for r in rows if r['method']==label)
            delta_eval=new['eval_itr70']-old['eval_itr70'] if h.finite(new['eval_itr70']) and h.finite(old['eval_itr70']) else None
            delta_train=new['train_itr69']-old['train_itr69'] if h.finite(new['train_itr69']) and h.finite(old['train_itr69']) else None
            pieces.append('%s: eval gap %s, train gap %s' % (label, h.fmt(delta_eval), h.fmt(delta_train)))
        lines += ['Against %s at the same iterations: %s.' % (reference, '; '.join(pieces)), '']
    errors=[r for r in rows if r['read_error']]
    if errors:
        lines += ['Read warning: after three attempts spaced by 0.2 s, the following live results were still unreadable; '
                  'their cells are unavailable, not zero: '+ '; '.join(r['method']+': '+r['read_error'] for r in errors)+'.', '']
    lines += ['These are interim observations from one seed; they do not determine the final comparison.']
    (output/'INTERIM.md').write_text('\n'.join(lines)+'\n')
    return rows


def execution_records(own):
    records=[]
    for run in own:
        worker=run.get('worker', {})
        provenance=worker.get('spec', {}).get('source_provenance', {})
        hardware=worker.get('hardware', {})
        elapsed=h.number(worker.get('elapsed_seconds'))
        times=[r.get('time') for r in run['training'] if h.finite(r.get('time'))]
        records.append(dict(method=run['method'], scheduler_job_id=worker.get('scheduler_job_id'),
                            node=hardware.get('node'), gpu=hardware.get('gpu'),
                            cpu_affinity=hardware.get('cpu_affinity'),
                            started_utc=worker.get('started_utc'), ended_utc=worker.get('ended_utc'),
                            wall_seconds=elapsed, seconds_per_iteration_wall=elapsed/140 if elapsed is not None else None,
                            median_training_iteration_seconds=float(np.median(times)) if times else None,
                            revision=provenance.get('revision'), branch=provenance.get('branch'),
                            source=provenance.get('source'), source_snapshot=provenance.get('snapshot'),
                            relevant_diff=provenance.get('relevant_diff'), command=worker.get('command'),
                            worker_manifest=run.get('worker_manifest'),
                            source_before=worker.get('source_before'), source_after=worker.get('source_after')))
    return records


def matched_verdict(by, runs, gaps):
    label=CONDITIONS['EP1M'][0]
    comparisons=[r for r in gaps if r['method']==label and r['metric'] in ('eval_936M','eval_last5','train_last10')
                 and h.finite(r['difference_to_DPPO']) and h.finite(r['dppo_seed_sd'])]
    negative=[r for r in comparisons if r['difference_to_DPPO'] < -2*r['dppo_seed_sd']]
    positive=[r for r in comparisons if r['difference_to_DPPO'] > 2*r['dppo_seed_sd']]
    if negative and positive:
        conclusion='The seed-0 evidence is mixed across reward metrics, so neither proposed reuse verdict is resolved.'
    elif negative:
        conclusion=('The matched one-epoch run is worse beyond the stated yardstick on %s; this one-seed result is consistent '
                    'with reuse buying something that a larger fresh-data step does not.' % ', '.join(r['metric'] for r in negative))
    elif comparisons and all(abs(r['difference_to_DPPO']) < 2*r['dppo_seed_sd'] for r in comparisons):
        conclusion=('All measured matched-one-epoch reward gaps are unresolved by the stated yardstick. '
                    'The movement-only outcome remains compatible with this one seed, but equivalence is not established.')
    elif comparisons:
        conclusion=('The matched one-epoch run has no resolved reward deficit by the stated yardstick; this one seed provides '
                    'no evidence that reuse buys a reward advantage beyond the movement match.')
    else:
        conclusion='The matched one-epoch verdict is unresolved because the required reward contrasts are unavailable.'
    labels={'eval_936M':'final eval', 'eval_last5':'last-five eval', 'train_last10':'last-ten train'}
    evidence='; '.join('%s gap %s versus threshold 2 sd = %s' %
                      (labels[r['metric']],h.fmt(r['difference_to_DPPO']),h.fmt(2*r['dppo_seed_sd'])) for r in comparisons)
    run=runs[label]
    rewards=[r['train_episode_reward'] for r in run['training']]
    minimum_window=min((float(np.mean(rewards[i:i+10])) for i in range(len(rewards)-9)),default=None)
    values=[r[key] for r in run['rows'] for key in ('eval_episode_reward','train_episode_reward','J_disc_train') if key in r]
    finite=bool(values) and all(h.finite(x) for x in values)
    stability=('Saved reward/J_disc observations are %s; minimum eval %s versus initial %s, and minimum ten-update '
               'mean training return %s. No additional collapse cutoff is imposed.' %
               ('finite' if finite else 'not all finite',h.fmt(by[label]['minimum_eval']),
                h.fmt(by[label]['initial_eval']),h.fmt(minimum_window)))
    return ' '.join([conclusion,evidence+'.',stability,
                    'Adam momentum means k·lr does not guarantee identical realized movement. Missing historical J_disc prevents a discounted-objective verdict.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interim',action='store_true')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite analysis evidence')
    own=[own_run(ident) for ident in CONDITIONS]
    if not args.interim and any(r['status']!='complete' for r in own):
        raise ValueError('All four seed-0 runs must be verified complete before final analysis')
    originals,wp,extra=references()
    if any(r['status']!='complete' for r in originals+[wp]+([extra] if extra else [])):
        raise ValueError('Reference run missing or incomplete')
    args.output.mkdir(parents=True)
    shutil.copy2(str(Path(__file__).resolve()), str(args.output/'analysis_source.py'))
    shutil.copy2(str(HELPER), str(args.output/'shared_analysis_source.py'))
    runs={r['method']:r for r in [originals[0],wp]+([extra] if extra else [])+own}
    evidence=[];distances={}
    for name,run in runs.items():
        distances[name]=h.checkpoint_distances(run,evidence)
    rows=[metrics(run,distances[name]) for name,run in runs.items()]
    # Three original seeds define only the user-requested metric yardsticks.
    base=[]
    for run in originals:
        base.append(metrics(run,h.checkpoint_distances(run,evidence)))
    yardsticks={key:h.sample_stats([r.get(key) for r in base])[1] for key in METRICS}
    ref=next(r for r in rows if r['method']=='DPPO')
    gaps=[]
    for row in rows:
        if row['method']=='DPPO':
            continue
        for key in METRICS:
            available=h.finite(row.get(key)) and h.finite(ref.get(key))
            gap=row[key]-ref[key] if available else None
            sd=yardsticks[key]
            reading=('unresolved' if abs(gap)<2*sd else 'exceeds twice the DPPO seed SD') if available and h.finite(sd) else 'not logged'
            gaps.append(dict(method=row['method'],metric=key,difference_to_DPPO=gap,
                             dppo_seed_sd=sd,reading=reading))
    h.write_csv(args.output/'epochs_table.csv',rows)
    h.write_csv(args.output/'gaps_to_dppo.csv',gaps)
    h.write_csv(args.output/'checkpoint_distances.csv',evidence)
    headings={'eval_936M':'Eval at 9.36M','eval_last5':'Mean last five evals','train_last10':'Train last ten updates',
              'J_disc_gain_last10':'J_disc gain last ten updates','theta_dist_final':'Final parameter distance',
              'mean_last_actor_approx_kl':'Mean last-step approx_kl','mean_last_actor_switched_off':'Mean last-step switched-off fraction',
              'mean_last_actor_abs_ratio':'Mean last-step |ratio − 1|','saturation_final':'Final saturation'}
    table=['# Actor-epoch comparison, seed 0','','One fresh-data update consumes 80 episodes. Missing values are not logged; no historical run was repeated.','',
           '## Setup','',
           h.markdown(['Method','Actor epochs','Actor steps/batch','Actor lr','Movement / DPPO'],
                      [[r['method'],r['actor_epochs'] if r['actor_epochs'] else 'single full-batch step',r['actor_steps_per_batch'],h.fmt(r['actor_lr']),h.fmt(r['movement_relative_to_dppo'])] for r in rows]),'',
           'Movement for without_PPO assumes quarter-batch √v is twice full-batch √v. The remaining matches use k·lr/(5·1e-4).','']
    returns=('eval_936M','eval_last5','train_last10','J_disc_gain_last10')
    mechanics=('theta_dist_final','mean_last_actor_approx_kl','mean_last_actor_switched_off',
               'mean_last_actor_abs_ratio','saturation_final')
    for title, keys in (('Returns', returns), ('Update diagnostics', mechanics)):
        table += ['## '+title,'',h.markdown(['Method']+[headings[key] for key in keys],
                    [[r['method']]+[h.fmt(r.get(key)) for key in keys] for r in rows]),'']
    table += ['J_disc gain subtracts each run’s own first training-batch value. Final saturation uses the last training rollout, before update 126; final distance uses the checkpoint after update 126.','',
              'Stage-1 three-seed sample SD (the reading-rule threshold is twice this): '+
              '; '.join(headings[key]+': '+h.fmt(yardsticks[key]) for key in METRICS)+'.']
    (args.output/'epochs_table.md').write_text('\n'.join(table)+'\n')
    figures(runs,distances,args.output)
    execution=execution_records(own)
    h.write_json(args.output/'execution_records.json',execution)
    interim_rows=interim_note(runs,args.output) if args.interim else None
    record=dict(utc=datetime.now(timezone.utc).isoformat(),status='interim' if args.interim else 'complete',
                runs=[{k:r.get(k) for k in ('method','seed','run_dir','result_path','status','worker_manifest','checkpoints')} for r in runs.values()],
                source_analysis=str(Path(__file__).resolve()),shared_analysis_source=str(HELPER),
                source_analysis_snapshot=str(args.output.resolve()/'analysis_source.py'),
                shared_analysis_snapshot=str(args.output.resolve()/'shared_analysis_source.py'),
                execution_records=execution,interim_rows=interim_rows,
                launch_source=h.read_json(ROOT/'provenance/launch_source.json',{}),
                job_file=str(ROOT/'jobs_epochs.txt'),
                yardsticks=yardsticks,gaps=gaps,gate=h.read_json(ROOT/'runs/gates/complete/gate_summary.json',{}))
    h.write_json(args.output/'analysis_record.json',record)
    if not args.interim:
        by={r['method']:r for r in rows}
        def summary(ident):
            label=CONDITIONS[ident][0];r=by[label]
            ge=next(x for x in gaps if x['method']==label and x['metric']=='eval_936M')
            gt=next(x for x in gaps if x['method']==label and x['metric']=='train_last10')
            return ('%s: eval %s (gap to DPPO %s; %s), last-ten train %s (gap %s; %s), J_disc gain %s.'%
                    (label,h.fmt(r['eval_936M']),h.fmt(ge['difference_to_DPPO']),ge['reading'],h.fmt(r['train_last10']),h.fmt(gt['difference_to_DPPO']),gt['reading'],h.fmt(r['J_disc_gain_last10'])))
        report=['# DPPO actor epochs at fixed and matched learning rates','','**Question.** Does reuse buy more than first-order movement per batch?','',
                '**Status.** The four seed-0 conditions are complete. Reward contrasts are judged against twice the original DPPO three-seed sample SD. Historical DPPO and without_PPO lack J_disc, so their discounted-gain and matched-distance comparisons remain unidentified.','',
                '[Table](epochs_table.md) · [Exact inputs and gate evidence](analysis_record.json) · [Methods](methods.md)','',
                '## Q1. Same learning rate','',summary('EP1')+' '+summary('EP2'),'',
                '## Q2. Matched first-order movement','',summary('EP1M')+' '+summary('EP2M'),'']
        r=by[CONDITIONS['EP1M'][0]]
        report += ['Matched one-epoch minimum eval %s, initial eval %s. '%(h.fmt(r['minimum_eval']),h.fmt(r['initial_eval']))+
                   ('without_PPO, lr 1e-3 final eval %s and last-ten train %s.'%(h.fmt(by['without_PPO, lr 1e-3']['eval_936M']),h.fmt(by['without_PPO, lr 1e-3']['train_last10'])) if extra else 'The 1e-3 reference is unavailable.'),'',
                   '## Q3. Distance','',
                   'The full comparison with DPPO cannot be made: its saved checkpoints provide parameter distances but no J_disc. The distance figure shows only observed new-run returns at their pre-rollout distances; it cannot establish whether DPPO buys a better direction.','',
                   '## Q4. Clip','',
                   'Matched one epoch: last-actor-step means are switched-off %s, approx_kl %s and |ratio−1| %s. DPPO last-step approx_kl is %s; the other two historical statistics were not logged.'%
                   (h.fmt(r['mean_last_actor_switched_off']),h.fmt(r['mean_last_actor_approx_kl']),h.fmt(r['mean_last_actor_abs_ratio']),h.fmt(ref['mean_last_actor_approx_kl'])),'',
                   '## Verdict','',
                   matched_verdict(by,runs,gaps),'',
                   '## Gates and provenance','',
                   'The launch requires G1–G5; final workers verify constant learning rates, actual actor/critic counts, checkpoints and clean exit. Any baseline KL early stop is recorded without changing its rule. The gate summaries, code revisions, commands, allocation identities and elapsed times are linked in the analysis record and campaign manifests. No additional seeds or reference sampling were run.','',
                   '## Figures','']
        report[-2:]=['## Execution','',
            h.markdown(['Method','Slurm job','Node','Wall time (s)','Median training iteration (s)'],
                       [[r['method'],r['scheduler_job_id'],r['node'],h.fmt(r['wall_seconds']),
                         h.fmt(r['median_training_iteration_seconds'])] for r in execution]),'',
            'Each run requested one A6000, 40 CPU cores and 64 GiB. GPU identities, UTC start/end times and exact commands '
            'are in [execution_records.json](execution_records.json); no running external job was stopped or changed.','',
            'Execution revisions: '+ '; '.join('%s: `%s` (branch `%s`)' %
                                               (r['method'],r['revision'],r['branch']) for r in execution)+'.','',
            '[Job file](jobs_epochs.txt) · [Launch provenance](provenance/launch_source.json) · '
            '[Campaign and allocation record](runs/campaign_training/campaign.json)','', '## Figures','']
        for name in ('epochs_curves_seed0','epochs_distance_seed0','epochs_ppo_diagnostics_seed0'):
            report += ['![%s](figs/%s.png)'%(name.replace('_',' '),name),'', (args.output/'figs'/(name+'.caption.md')).read_text().strip(),'']
        (args.output/'REPORT.md').write_text('\n'.join(report)+'\n')
    if args.publish:
        if args.interim:
            raise ValueError('Interim outputs cannot replace final deliverables')
        for name in ('epochs_table.csv','epochs_table.md','gaps_to_dppo.csv','checkpoint_distances.csv','analysis_record.json','execution_records.json','REPORT.md'):
            destination=ROOT/name
            if destination.exists():
                raise ValueError('Refuse to replace existing final evidence '+str(destination))
            shutil.copy2(str(args.output/name),str(destination))
        if (ROOT/'figs').exists():
            raise ValueError('Refuse to replace existing figures')
        shutil.copytree(str(args.output/'figs'),str(ROOT/'figs'))
        experiment = ROOT/'experiment.md'
        text = experiment.read_text()
        old = ('**Status: approved; implementation and verification in progress.** Four seed-0 runs are authorized. '
               'Full training starts only after G1–G5 pass.')
        new = ('**Status: completed.** All four seed-0 runs finished 140 iterations, and their saved outputs and '
               'clean exits were verified. G1–G5 passed; the requested comparison is available in '
               '[REPORT.md](REPORT.md) and [epochs_table.md](epochs_table.md).')
        if old in text:
            experiment.write_text(text.replace(old, new, 1))
        checklist = PROJECT/'complete_experiment.md'
        title = '### DPPO actor epochs at fixed and matched learning rates'
        text = checklist.read_text()
        if title not in text:
            section = (title+'\n\n'
                       '- Experiment folder: [Actor-epoch comparison](./results/stage3/epochs/).\n'
                       '- Experiment design: [Methods](./results/stage3/epochs/methods.md).\n'
                       '- Curated: —\n\n')
            marker = '## Comparison analyses'
            if marker not in text:
                raise ValueError('Completed-experiment checklist structure changed; review before adding entry')
            checklist.write_text(text.replace(marker, section+marker, 1))
    print(json.dumps({'status':record['status'],'output':str(args.output),'runs':{k:v['status'] for k,v in runs.items()},
                      'interim_note':str(args.output/'INTERIM.md') if args.interim else None,
                      'transient_read_errors':{k:v['read_error'] for k,v in runs.items() if v.get('read_error')}}))


if __name__=='__main__':
    main()
