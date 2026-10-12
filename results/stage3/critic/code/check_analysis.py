#!/usr/bin/env python
"""Small deterministic checks using synthetic records; no experimental reads."""
import argparse
import ast
import copy
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
import tempfile
import types

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('critic_analysis_check_target', HERE / 'analyze_critic.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
# The imported helper supplies pure math and formatting functions only.
a = c.helper()

def rows(condition, count=140):
    result = []
    for itr in range(count):
        row = {'itr':itr,'step':(itr-itr//10)*80000,'time':2.}
        if itr%10 == 0:
            row['eval_episode_reward']=4500.+itr
        else:
            row.update(train_episode_reward=4000.+itr,J_disc_train=1800.-itr,
                kl_true_per_action=6. if condition.startswith('SGDC') else 2.,
                split_diff_sq=10. if condition.startswith('SGDC') else 5.,split_dot=1.,
                baseline_energy_critic=float(itr+2),baseline_energy_loo=float(itr+1),
                baseline_energy_ratio=float(itr+2)/(itr+1),baseline_divisor=20.,
                baseline_bin_energy_critic=[float(itr+2)/5]*5,
                baseline_bin_energy_loo=[float(itr+1)/5]*5,
                theta_dist=.01*itr,update_norm=.01,x0_sat_frac_mean=.1,x0_sat_frac_last=.2,
                action_oor_frac=.3,explained_var=.4,grad_cos_prev=.5,proj_active=False,actor_lr=.001)
        result.append(row)
    return result

def run(condition,seed=0,count=140,status='complete'):
    rr=rows(condition,count)
    return {'spec':dict(condition=condition,seed=seed,run_id=condition+'_seed'+str(seed),logdir='/synthetic'),
        'manifest':{},'status':status,'rows':rr,'windows':a.window_noise(rr),
        'manifest_path':'/synthetic/manifest.json','result_path':'/synthetic/result.pkl',
        'read_error':None,'checkpoints':[],'checkpoint_eval':{},
        'checkpoint_eval_status':'unavailable','checkpoint_eval_path':'/synthetic/ckpt_eval.json'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require=lambda x: (_ for _ in ()).throw(AssertionError()) if not x else None
    checks=[]
    def check(name,ok,**extra):
        require(ok);checks.append(dict(name=name,passed=True,**extra))
    ast.parse((HERE/'analyze_critic.py').read_text(),feature_version=8)
    check('Python38_AST',True)
    runs=[run(name,seed) for name in c.CONDITIONS for seed in range(3)]
    saved=copy.deepcopy(runs)
    windows,bins,matched,comparisons=c.diagnostic_tables(runs,a)
    e=next(x for x in comparisons if x['condition']=='SGDC3' and x['seed']==0 and x['window']=='early_1_10')
    check('KL_ratio_of_window_means',e['kl_ratio']==3.)
    check('split_trace_ratio_prescribed20',e['split_trace_ratio']==2. and
          next(x for x in windows if x['condition']=='SGDC3' and x['seed']==0)['split_trace']==200.)
    check('residual_ratio_of_energies_not_mean_ratios',np.isclose(e['residual_energy_ratio'],7./6.) and
          not np.isclose(e['residual_energy_ratio'],np.mean([(i+2.)/(i+1.) for i in range(1,10)])))
    check('all_windows_nine_updates',all(x['training_iterations']==9 and x['complete'] for x in windows))
    check('bin_energies_ratio',np.isclose(bins[0]['energy_ratio'],7./6.))
    check('raw_records_inert',runs==saved)
    first=c.extended_summaries([run('SGDC3',count=23,status='running')],{},a)[0]
    incomplete=c.extended_summaries([run('SGDC3',count=22,status='failed')],{},a)[0]
    check('cold_twenty_actual_updates',first['first20_update_count']==20 and first['first20_last_itr']==22
          and first['first20_min_J_disc']==1778.)
    check('cold_incomplete_withheld',incomplete['first20_update_count']==19 and
          incomplete['first20_max_kl'] is None and incomplete['first20_saturation'] is None)
    check('incomplete_full_run_mean_withheld',first['explained_var_mean'] is None)
    corrupted=run('SGDC3');corrupted['rows'][1]['kl_true_per_action']=float('inf')
    c.validate([corrupted],a)
    check('nonfinite_saved_prefix_preserved',corrupted['status']=='failed' and len(corrupted['rows'])==140
          and c.extended_summaries([corrupted],{},a)[0]['collapse'] is True)
    badbin=run('SGDC3');badbin['rows'][1]['baseline_bin_energy_critic']=[1.,2.]
    c.validate([badbin],a)
    bw,bb,bm,bc=c.diagnostic_tables([badbin]+[r for r in runs if (r['spec']['condition'],r['spec']['seed'])!=('SGDC3',0)],a)
    check('malformed_bins_safe_unavailable',badbin['status']=='failed' and bb[0]['energy_critic'] is None)
    checkpoint_bad=run('SGDC3');checkpoint_bad['checkpoint_eval']={'bad':float('nan')}
    c.validate([checkpoint_bad],a)
    cs=c.extended_summaries([checkpoint_bad],{},a)[0]
    check('checkpoint_nonfinite_separate_from_training',checkpoint_bad['status']=='complete' and cs['collapse']
          and checkpoint_bad['checkpoint_eval_status']=='invalid_nonfinite')
    with tempfile.TemporaryDirectory(prefix='critic-analysis-check-') as temp:
        base=Path(temp); rr=c.load_checked(base,[dict(mode='train',condition='SGDC3',seed=0,run_id='SGDC3_seed0',logdir=str(base/'native'))],a)
        check('missing_run_safe',rr[0]['rows']==[] and rr[0]['status']=='planned')
        manifest=base/'runs/SGDC3_seed0';manifest.mkdir(parents=True)
        (manifest/'manifest.json').write_text('{"status":"complete"}')
        rr=c.load_checked(base,[dict(mode='train',condition='SGDC3',seed=0,run_id='SGDC3_seed0',logdir=str(base/'native'))],a)
        check('false_completion_reported',rr[0]['status']=='failed' and rr[0]['analysis_validation_errors'])
    all_s=c.extended_summaries(runs,{},a)
    a.METRICS=list(a.METRICS)+['first20_max_kl','first20_J_dip_from_start','first20_saturation','first20_energy_ratio']
    grouped=a.aggregate(all_s)
    candidate=c.ROOT/'runs/analyses/final/candidates'
    report=c.generate_report(a,grouped,windows,bins,comparisons,{'passed':False,'status':'failed'}, {},runs,candidate)
    check('failed_gate_report_explicit','Gate not passed' in report and 'C1–C5 status: **failed**' in report)
    gate_link=re.search(r'\[Full gate record\]\(([^)]+)\)',report).group(1)
    check('gate_link_resolves',(candidate/gate_link).resolve()==c.ROOT/'runs/gate/gate_summary.json')
    check('no_survivor_mean',a.mean_sd([1.,2.,None])['mean'] is None)
    check('unsafe_derived_ratio_unavailable',c.ratio(1e308,1e-308,a) is None and c.ratio(1.,0.,a) is None)
    original_root, original_reference, original_argv = c.ROOT, c.REFERENCE, sys.argv[:]
    with tempfile.TemporaryDirectory(prefix='critic-catalog-check-') as temp:
        try:
            project=Path(temp);c.ROOT=project/'results/stage3/critic';c.ROOT.mkdir(parents=True)
            (c.ROOT/'methods.md').write_text('Synthetic methods only.\n')
            (c.ROOT/'experiment.md').write_text('Synthetic completed experiment record only.\n')
            catalog=project/'complete_experiment.md'
            original=('# Completed experiments\n\n## Experiments\n\n### Existing study\n\n'
                      '- Existing design: [Keep](./existing/methods.md)\n\n'
                      '## Comparison analyses\n\n### Existing comparison\n\n'
                      '- Existing curated: [Keep](./existing/curated/INDEX.md)\n')
            catalog.write_text(original)
            finished=[run(name,seed) for name in c.NEW for seed in range(3)]
            for value in finished:
                native=c.ROOT/value['spec']['run_id'];native.mkdir()
                result=native/'result.pkl';result.write_text('Synthetic fixture marker, not experimental data.\n')
                value['result_path']=str(result);value['manifest']['status']='complete'
                value['checkpoints']=[]
                for itr in (0,35,70,105,139):
                    path=native/('state_%d.pt'%itr);path.write_text('Synthetic marker\n');value['checkpoints'].append(str(path))
                evaluation={'status':'completed','samplers':{s:{k:{'mean':1.,'se':.1,'n':80}
                            for k in ('undiscounted_return','J_disc')} for s in ('train','eval')}}
                path=native/'ckpt_eval.json';path.write_text(json.dumps(evaluation))
                value.update(checkpoint_eval=evaluation,checkpoint_eval_path=str(path),checkpoint_eval_status='complete')
            for flags in ((True,True,True),(False,False,True),(True,False,False)):
                result=c.maintain_completed_catalog(finished,a,*flags)
                check('catalog_guard_'+str(flags),result['status']=='not_eligible' and catalog.read_text()==original)
            failed=copy.deepcopy(finished);failed[0]['status']='failed'
            incomplete=copy.deepcopy(finished);incomplete[0]['checkpoint_eval_status']='unavailable'
            wrong_count=finished[:-1]
            wrong_sampler=copy.deepcopy(finished);wrong_sampler[0]['checkpoint_eval']['samplers']['eval']['J_disc']['n']=79
            for label,values in (('failed',failed),('evaluation_missing',incomplete),('fourteen_runs',wrong_count),('wrong_sampler_count',wrong_sampler)):
                result=c.maintain_completed_catalog(values,a,True,False,True)
                check('catalog_guard_'+label,result['status']=='not_eligible' and catalog.read_text()==original)
            result=c.maintain_completed_catalog(finished,a,True,False,True)
            inserted=catalog.read_text();start=inserted.index('### Stage 3 plain SGD');end=inserted.index('## Comparison analyses')
            entry=inserted[start:end]
            check('catalog_complete_inserts_under_experiments',result['status']=='added' and
                  inserted[:start]+inserted[end:]==original and '- Curated: —' in entry)
            links=re.findall(r'\]\(([^)]+)\)',entry)
            check('catalog_links_resolve_to_files',len(links)==2 and all((project/link).is_file() for link in links)
                  and './results/stage3/critic/experiment.md' in links)
            second=c.maintain_completed_catalog(finished,a,True,False,True)
            check('catalog_idempotent',second['status']=='already_present' and catalog.read_text()==inserted)
            curated=inserted.replace('- Curated: —','- Curated: [Approved index](./approved/INDEX.md)')
            catalog.write_text(curated)
            c.maintain_completed_catalog(finished,a,True,False,True)
            check('catalog_preserves_later_curated_link',catalog.read_text()==curated)
        finally:
            c.ROOT=original_root
    with tempfile.TemporaryDirectory(prefix='critic-analysis-render-') as temp:
        try:
            c.ROOT=Path(temp)/'stage3/critic';c.REFERENCE=c.ROOT.parent
            (c.REFERENCE/'code').mkdir(parents=True)
            shutil.copy2(original_reference/'code/analyze_stage3.py',c.REFERENCE/'code/analyze_stage3.py')
            gate=c.ROOT/'runs/gate/gate_summary.json';gate.parent.mkdir(parents=True)
            gate.write_text('{"passed":false,"status":"failed","reason":"synthetic gate failure"}')
            gate_before=gate.read_bytes()
            (c.ROOT/'REPORT.md').write_text('previous synthetic report\n')
            output=c.ROOT/'runs/analyses/final'
            sys.argv=[str(HERE/'analyze_critic.py'),'--output',str(output),'--partial','--publish-report']
            c.main()
            record=json.loads((output/'analysis_record.json').read_text())
            published=(c.ROOT/'REPORT.md').read_text()
            check('empty_failed_campaign_renders',record['status']=='complete' and not record['gate_passed']
                  and not record['all_new_runs_completed'] and 'Gate not passed' in published)
            check('report_previous_evidence_preserved',(output/'previous_REPORT.md').read_text()=='previous synthetic report\n'
                  and gate.read_bytes()==gate_before)
            link=re.search(r'\[Full gate record\]\(([^)]+)\)',published).group(1)
            check('published_gate_link_resolves',(c.ROOT/link).resolve()==gate)
        finally:
            c.ROOT, c.REFERENCE, sys.argv = original_root, original_reference, original_argv
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,checks=checks,scope='Synthetic software records only; no experiment analyzed'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),output=str(args.output))))
if __name__=='__main__':main()
