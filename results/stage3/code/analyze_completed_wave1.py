#!/usr/bin/env python
"""Analyze all 27 completed Wave-1 runs, preserving prior curated evidence."""
import datetime as dt
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import analyze_stage3 as a
from analyze_completed_five import describe, fmt, KEYS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/completed_wave1_analysis/candidates'
NAMES = ['E0', 'E1', 'E2', 'E2_beta0', 'E2_nobase', 'SGD03', 'SGD1', 'SGD3', 'SGD10']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record_path = OUT.parent / 'analysis_record.json'
    if record_path.exists():
        raise ValueError('Analysis already exists; preserve previous evidence')
    campaign = a.read_json(ROOT / 'runs/campaign.json')
    subset = dict(campaign, runs=[r for r in campaign['runs'] if r['mode'] == 'train' and r['condition'] in NAMES])
    runs = a.load_runs(ROOT, subset, False)
    assert len(runs) == 27 and all(r['status'] == 'complete' for r in runs)
    summaries = []
    keys = KEYS + ['eta_theory', 'median_kl_over_target']
    kl_target = a.read_json(ROOT / 'cal_seed3000/calibration.json')['kl_target']
    for run in runs:
        rows = run['rows']; train = [x for x in rows if 'train_episode_reward' in x]
        assert len(rows) == 140 and len(train) == 126 and not a.numeric_bad(rows)
        assert [x['itr'] for x in train[-10:]] == [129] + list(range(131, 140))
        summary = a.summarize(run, {})
        summary.update(eval_gain=summary['eval_itr130'] - summary['eval_itr0'],
                       first_train_return=train[0]['train_episode_reward'], first_train_J_disc=train[0]['J_disc_train'],
                       grad_cos_prev_mean=float(np.mean([x['grad_cos_prev'] for x in train if x['grad_cos_prev'] is not None])),
                       J_disc_eval_itr130=rows[130]['J_disc_eval'],
                       eta_theory=summary['actor_lr'] / 2500 if run['spec']['condition'].startswith('SGD') else None,
                       median_kl_over_target=summary['median_kl'] / kl_target)
        summaries.append(summary)
    grouped = {}
    for name in NAMES:
        seeds = sorted([s for s in summaries if s['condition'] == name], key=lambda s: s['seed'])
        assert [s['seed'] for s in seeds] == [0, 1, 2]
        grouped[name] = {key: describe([s[key] for s in seeds]) for key in keys if all(a.finite(s[key]) for s in seeds)}
    references = a.reference_runs(ROOT)
    ref_summary = {name: {
        'eval_itr130': describe([next(x['eval_episode_reward'] for x in seed['rows'] if x['itr'] == 130) for seed in seeds]),
        'train_last10': describe([np.mean([x['train_episode_reward'] for x in seed['rows'] if 'train_episode_reward' in x][-10:]) for seed in seeds])}
        for name, seeds in references.items()}
    comparisons = []; all_groups = dict(grouped, **ref_summary)
    pairs = list(itertools.combinations(NAMES, 2)) + [('Stage 2 NC4 1e-4', 'E0'), ('Stage 1 DPPO', 'E1')]
    for left, right in pairs:
        for metric in ['eval_itr130', 'train_last10', 'J_disc_last10']:
            if metric not in all_groups[left] or metric not in all_groups[right]:
                continue
            x, y = all_groups[left][metric], all_groups[right][metric]
            delta = y['mean'] - x['mean']; cutoff = 2 * math.sqrt((x['sd']**2 + y['sd']**2) / 2)
            comparisons.append({'from': left, 'to': right, 'metric': metric, 'difference': delta,
                                'twice_pooled_sample_sd': cutoff, 'unresolved_by_prespecified_rule': abs(delta) < cutoff})
    a.write_csv(OUT / 'seed_results.csv', summaries)
    a.write_csv(OUT / 'condition_results.csv', [dict(condition=name, **{f'{metric}_{stat}': value for metric, entry in grouped[name].items() for stat, value in entry.items()}) for name in NAMES])
    a.write_csv(OUT / 'comparisons.csv', comparisons)
    a.write_csv(OUT / 'all_runs_long.csv', [dict(run=r['spec']['run_id'], condition=r['spec']['condition'], seed=r['spec']['seed'], **row) for r in runs for row in r['rows']])
    windows = [dict(run=r['spec']['run_id'], condition=r['spec']['condition'], seed=r['spec']['seed'], **row) for r in runs for row in r['windows']]
    a.write_csv(OUT / 'noise_windows.csv', windows)
    a.write_json(OUT / 'summary.json', {'conditions': grouped, 'references': ref_summary, 'comparisons': comparisons,
                                      'kl_target': kl_target, 'projection_decisions': campaign['projection_decisions']})
    display = [[name, fmt(grouped[name]['eval_itr130']), fmt(grouped[name]['train_last10']), fmt(grouped[name]['J_disc_last10'])] for name in NAMES]
    report = [
        '# Completed Wave-1 runs: estimator and SGD comparison', '',
        '**Question.** Which estimator and optimizer choices improve HalfCheetah returns without clipping or sample reuse?', '',
        '**Answer.** All 27 Wave-1 runs completed without collapse, non-finite values or zero actor updates. E1 has the highest mean return; removing LOO hurts performance, while removing Adam momentum leaves endpoint returns unresolved relative to E2. SGD10 remains stable, but its apparent improvement over SGD3 is unresolved.', '',
        a.md_table(['Condition', 'Eval @9.36M', 'Train, last 10', 'Discounted train J'], display), '',
        'Mean ± sample SD (ddof=1) across three independent seeds. Eval is iteration 130; training averages iterations 129 and 131–139, ending at 10.08M steps. J discounts raw four-step decision rewards by 0.99. The comparison rule calls differences smaller than twice pooled seed SD unresolved; it is not a significance test.', '',
        '[Return curves](performance.png) · [Update diagnostics](diagnostics.png) · [Noise and saturation](noise_saturation.png) · [Captions](captions.md) · [Condition table](condition_results.csv) · [Per-seed table](seed_results.csv) · [Table definitions](table_caption.md)', '',
        '**Estimator changes.** E1 exceeds E0 in eval, train and J. E2 lowers both return measures relative to E1, offering no evidence of a discounted-up/undiscounted-down tradeoff; the E1–E2 eval contrast only narrowly exceeds the rule. These are bundled interventions, not an isolated test of decision discounting.', '',
        '**Momentum and baseline.** E2_beta0 versus E2 is unresolved on all three return endpoints, despite approximately 14.7× larger mean seed-median KL (0.00555 versus 0.000377). Removing LOO reduces eval by 236 and J by 97.9, both beyond the descriptive threshold; it does not cause collapse.', '',
        '**SGD sweep.** Mean training J is 1,399, 1,413, 1,450 and 1,480 at η*/3, η*, 3η* and 10η*, respectively. SGD10 versus SGD3 and E2 remains unresolved on all three endpoints. Its mean seed-median KL is 0.0214, 36.6× the calibration target, yet its final parameter distance is only 0.242 versus E2’s 0.876. Thus the initial calibration does not keep realized training KL matched. No SGD run collapsed or froze.', '',
        '**Diagnostic limits.** Split-noise denominators are nonpositive in 207/378 seed windows; only nine of 126 condition windows support a complete three-seed B_env mean. Conditional medians from the remaining positive windows cannot establish a noise-scale ordering. Mean x0 saturation spans 6.64–12.18%, but no zero updates occur and saturation alone does not identify a cause of return differences.', '',
        '**Scope.** AdamW uses 1e-4; η*=0.0001210723567 for SGD, with η_theory=η/2500. Calibration’s 3.40× spread triggered the prescribed warning. BATCH4 is ongoing; PROJ3/PROJ10 were skipped by the specified distance rule. Final-checkpoint evaluations are pending, so no claim about final-policy discounted improvement is made. Known Phase-1 bootstrap confidence-interval problems do not enter these seed-SD comparisons. Earlier curated results and all native outputs remain unchanged.', '',
        '**Decision.** Review these expanded candidates for curation. [Methods](../../../methods.md) · [Reproduction and sources](../analysis_record.json).', ''
    ]
    text = '\n'.join(report)
    assert len(text.split()) <= 600, len(text.split())
    (OUT / 'analysis.md').write_text(text)
    table_caption = '''# Table definitions

The condition table gives each seed value, the arithmetic mean and sample SD (ddof=1); the seed table retains all 27 completed runs, including all endpoints and diagnostics. Eval is measured at iterations 0 and 130 (9.36M steps), while train/J summaries average the last ten training iterations, 129 and 131–139. KL is summarized as a median within each seed, then mean ± SD across those three medians; this differs from pooling all 378 iterations. Actor distance is measured at iteration 139, and saturation, out-of-range executed actions and zero-update fractions summarize all training iterations.

The comparisons table applies the pre-specified twice-pooled-sample-SD rule without hypothesis tests; all-pairs entries are descriptive. Noise windows use complete ten-iteration blocks with nine training updates and B_env=10×sum(split_diff_sq)/sum(split_dot); nonpositive denominators remain undefined and are counted. Seed-level median B_env uses only defined windows, and the plotting mean requires all three seeds at the same window; these conditional values cannot establish a reliable ordering. The long table retains all 3,780 native iteration rows without smoothing or filtering; structured summaries include the six earlier Stage-1/NC4 reference runs.

Final checkpoint evaluations and incomplete BATCH4 runs are not substituted or extrapolated. Phase-1 bootstrap confidence intervals are not used. See [analysis provenance](../analysis_record.json), [independent checks](../independent_verification.json) and [plot provenance](../plot_record.json).
'''
    (OUT / 'table_caption.md').write_text(table_caption)
    record = {'status': 'completed_interim_analysis', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'command': [sys.executable, '-B', str(Path(__file__).resolve())], 'scientific_source_commit': campaign['commit'],
              'conditions': NAMES, 'sample_sd_ddof': 1, 'training_average_iterations': [129] + list(range(131, 140)),
              'checkpoint_evaluations': 'pending; not used', 'scope': 'All 27 completed Wave-1 training runs; excludes ongoing BATCH4 and prospectively skipped projection conditions.',
              'projection_decisions': campaign['projection_decisions'],
              'validation': {'runs': 27, 'rows_per_run': 140, 'training_rows_per_run': 126, 'all_checkpoint_paths_exist': True,
                             'all_values_finite': True, 'collapse_runs': [s['run'] for s in summaries if s['collapse']],
                             'nonpositive_noise_windows': sum(w['reason'] == 'nonpositive_split_dot' for w in windows)},
              'inputs': [{'run': r['spec']['run_id'], 'result': r['result_path'], 'manifest': r['manifest_path'], 'checkpoints': r['checkpoints']} for r in runs],
              'reference_inputs': {name: [s['path'] for s in seeds] for name, seeds in references.items()},
              'artifacts': [str(p.relative_to(ROOT)) for p in sorted(OUT.iterdir()) if p.is_file()],
              'curation': 'unreviewed; previous five-condition curated comparison preserved'}
    a.write_json(record_path, record)
    print(a.md_table(['Condition', 'Eval @9.36M', 'Train, last 10', 'Discounted train J'], display))
    print('Saved', record_path)


if __name__ == '__main__':
    main()
