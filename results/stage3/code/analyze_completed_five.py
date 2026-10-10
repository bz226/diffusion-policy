#!/usr/bin/env python
"""Read completed E0/E1/E2/SGD1/SGD3 evidence; no training or curation."""
import datetime as dt
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import analyze_stage3 as a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/completed_five_analysis/candidates'
NAMES = ['E0', 'E1', 'E2', 'SGD1', 'SGD3']
KEYS = ['eval_itr0', 'eval_itr130', 'eval_gain', 'train_last10', 'J_disc_last10',
        'median_kl', 'theta_dist_final', 'minimum_eval_return', 'median_B_env',
        'defined_noise_windows', 'nonpositive_noise_windows', 'zero_update_fraction',
        'x0_saturation_mean', 'x0_saturation_last', 'action_oor_mean',
        'seconds_per_training_iteration', 'actor_lr', 'first_train_return',
        'first_train_J_disc', 'grad_cos_prev_mean', 'J_disc_eval_itr130']


def describe(values):
    assert len(values) == 3 and all(a.finite(x) for x in values)
    return {'seed0': float(values[0]), 'seed1': float(values[1]), 'seed2': float(values[2]),
            'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}


def fmt(value, digits=0):
    return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record_path = OUT.parent / 'analysis_record.json'
    if record_path.exists():
        raise ValueError('Completed analysis evidence exists; do not overwrite it')
    campaign = a.read_json(ROOT / 'runs/campaign.json')
    subset = dict(campaign, runs=[r for r in campaign['runs'] if r['mode'] == 'train' and r['condition'] in NAMES])
    runs = a.load_runs(ROOT, subset, False)
    assert len(runs) == 15 and all(r['status'] == 'complete' for r in runs)
    summaries = []
    for run in runs:
        rows = run['rows']; train = [x for x in rows if 'train_episode_reward' in x]
        assert len(rows) == 140 and len(train) == 126 and not a.numeric_bad(rows)
        assert [x['itr'] for x in train[-10:]] == [129] + list(range(131, 140))
        summary = a.summarize(run, {})
        summary.update(eval_gain=summary['eval_itr130'] - summary['eval_itr0'],
                       first_train_return=train[0]['train_episode_reward'],
                       first_train_J_disc=train[0]['J_disc_train'],
                       grad_cos_prev_mean=float(np.mean([x['grad_cos_prev'] for x in train if x['grad_cos_prev'] is not None])),
                       J_disc_eval_itr130=rows[130]['J_disc_eval'])
        summaries.append(summary)
    grouped = {}
    for name in NAMES:
        seeds = sorted([s for s in summaries if s['condition'] == name], key=lambda s: s['seed'])
        assert [s['seed'] for s in seeds] == [0, 1, 2]
        grouped[name] = {key: describe([s[key] for s in seeds]) for key in KEYS}
    references = a.reference_runs(ROOT)
    ref_summary = {}
    for name, seeds in references.items():
        ref_summary[name] = {
            'eval_itr130': describe([next(x['eval_episode_reward'] for x in seed['rows'] if x['itr'] == 130) for seed in seeds]),
            'train_last10': describe([np.mean([x['train_episode_reward'] for x in seed['rows'] if 'train_episode_reward' in x][-10:]) for seed in seeds])}
    comparisons = []
    all_groups = dict(grouped, **ref_summary)
    pairs = list(itertools.combinations(NAMES, 2)) + [('Stage 2 NC4 1e-4', 'E0'), ('Stage 1 DPPO', 'E1')]
    for left, right in pairs:
        for metric in ['eval_itr130', 'train_last10', 'J_disc_last10']:
            if metric not in all_groups[left] or metric not in all_groups[right]:
                continue
            x, y = all_groups[left][metric], all_groups[right][metric]
            delta = y['mean'] - x['mean']
            cutoff = 2 * math.sqrt((x['sd']**2 + y['sd']**2) / 2)
            comparisons.append({'from': left, 'to': right, 'metric': metric,
                                'difference': delta, 'twice_pooled_sample_sd': cutoff,
                                'unresolved_by_prespecified_rule': abs(delta) < cutoff})
    a.write_csv(OUT / 'seed_results.csv', summaries)
    a.write_csv(OUT / 'condition_results.csv', [dict(condition=name, **{f'{metric}_{stat}': value for metric, entry in grouped[name].items() for stat, value in entry.items()}) for name in NAMES])
    a.write_csv(OUT / 'comparisons.csv', comparisons)
    a.write_csv(OUT / 'all_runs_long.csv', [dict(run=r['spec']['run_id'], condition=r['spec']['condition'], seed=r['spec']['seed'], **row) for r in runs for row in r['rows']])
    a.write_csv(OUT / 'noise_windows.csv', [dict(run=r['spec']['run_id'], condition=r['spec']['condition'], seed=r['spec']['seed'], **row) for r in runs for row in r['windows']])
    a.write_json(OUT / 'summary.json', {'conditions': grouped, 'references': ref_summary, 'comparisons': comparisons})
    rows = [[name, fmt(grouped[name]['eval_itr130']), fmt(grouped[name]['train_last10']),
             fmt(grouped[name]['J_disc_last10']), f"{grouped[name]['median_kl']['mean']:.2g}",
             f"{grouped[name]['theta_dist_final']['mean']:.3f}"] for name in NAMES]
    report = [
        '# Completed E0, E1, E2, SGD1 and SGD3: interim analysis', '',
        '**Question.** How do the estimator ladder and calibrated SGD change return and stability relative to NC4?', '',
        '**Answer.** E1 has the highest final scheduled evaluation and last-ten-iteration training returns among these five conditions. E2 does not show the proposed discounted-versus-undiscounted tradeoff: both measures are lower than E1. All 15 runs finished without non-finite values, reward-collapse flags or zero actor updates; final-checkpoint evaluation is still pending.', '',
        a.md_table(['Condition', 'Eval @9.36M', 'Train, last 10', 'Discounted train J', 'Mean seed median KL', 'Mean final distance'], rows), '',
        'Returns show mean ± sample SD (ddof=1) across three independent training seeds. Eval is iteration 130; training averages use iterations 129 and 131–139, ending at 10.08M environment steps. J uses raw rewards discounted by 0.99 per four-step decision. KL averages each seed’s median sampled transition KL; distance is the actor parameter norm relative to the pretrained actor.', '',
        '[Performance curves](performance.png) · [Update diagnostics](diagnostics.png) · [Figure captions](captions.md) · [Per-seed values](seed_results.csv) · [Condition table](condition_results.csv) · [Table definitions](table_caption.md)', '',
        '**Evidence.** E0→E1 improves eval by 137, train return by 126 and J by 43.3; all exceed twice the pooled seed SD. E1→E2 reduces eval by 128 and J by 43.1; the eval comparison only narrowly exceeds that rule (threshold 122). These are joint estimator interventions: E1 changes returns, baseline and actor reward units; E2 adds decision discounting while changing normalization, denoising weights and log-prob reduction.', '',
        'SGD1→SGD3 raises J by 37.9, exceeding the same rule, while the 74-point eval difference remains unresolved. E2 exceeds SGD1 in eval and J; E2 versus SGD3 is unresolved in eval but E2 has higher J. SGD1/3 travel only 0.030/0.088 in parameter space versus E2’s 0.876. Their mean seed-median KLs are 0.58×/4.46× the calibration target, so matching initial calibration does not enforce matched KL throughout training.', '',
        'E0’s eval difference from NC4 1e-4 (4,604 ± 19) remains inside the prescribed two-pooled-SD sanity band. E1’s difference from Stage-1 DPPO (4,758 ± 108) is unresolved under the same descriptive rule; this does not establish equivalence.', '',
        '**Setup and limits.** AdamW uses 1e-4. SGD1 uses η=0.0001210723567 (η_theory=4.8429e-8); SGD3 uses three times that. Calibration’s 3.40× batch spread triggered the prescribed warning; the median was retained. All use one actor step per fresh batch, with unchanged critic training. Three seeds and training-sampler J are insufficient to claim a final-policy objective improvement; checkpoint evaluation will provide that comparison. No significance tests, smoothing, resampling or extra scientific sampling were added. The projection conditions and remaining campaign runs are outside this interim analysis.', '',
        '**Decision.** Review these candidates for curation; no artifact has been curated. [Methods](../../../methods.md) · [Reproduction and sources](../analysis_record.json).', ''
    ]
    text = '\n'.join(report)
    assert len(text.split()) <= 600
    (OUT / 'analysis.md').write_text(text)
    record = {'status': 'completed_interim_analysis', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'command': [sys.executable, '-B', str(Path(__file__).resolve())],
              'scientific_source_commit': campaign['commit'], 'conditions': NAMES,
              'sample_sd_ddof': 1, 'training_average_iterations': [129] + list(range(131, 140)),
              'checkpoint_evaluations': 'not used; pending at analysis time',
              'scope': 'Native completed training evidence only; Phase-1 bootstrap uncertainty is not used.',
              'validation': {'runs': 15, 'rows_per_run': 140, 'training_rows_per_run': 126, 'all_checkpoint_paths_exist': True,
                             'all_values_finite': True, 'collapse_runs': [s['run'] for s in summaries if s['collapse']]},
              'inputs': [{'run': r['spec']['run_id'], 'result': r['result_path'], 'manifest': r['manifest_path'], 'checkpoints': r['checkpoints']} for r in runs],
              'reference_inputs': {name: [s['path'] for s in seeds] for name, seeds in references.items()},
              'artifacts': [str(p.relative_to(ROOT)) for p in sorted(OUT.iterdir()) if p.is_file()],
              'curation': 'unreviewed; keep/revise/leave and human-readable names require user decision'}
    a.write_json(record_path, record)
    print(a.md_table(['Condition', 'Eval @9.36M', 'Train, last 10', 'Discounted train J', 'Mean seed median KL', 'Mean final distance'], rows))
    print('Saved', record_path)


if __name__ == '__main__':
    main()
