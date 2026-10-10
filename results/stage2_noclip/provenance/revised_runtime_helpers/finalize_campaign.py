#!/usr/bin/env python
"""Generate only the approved candidate outputs after the revised campaign ends."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess

from analyze_stage2 import generate, load_campaign, CONDITIONS
from fixed_batch_gate import validate_gate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'repro_halfcheetah'
COMMIT = 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964'
BASELINES = [BASE / 'runs' / name / 'manifest.json'
             for name in ('seed0_handoff', 'seed1', 'seed2')]


def current_report(record, gate):
    summaries = record['summaries']
    active = [row for row in summaries if row['condition'] != 'baseline']
    complete = sum(row['status'] == 'complete' for row in active)
    collapsed = sum(row['collapse'] is True for row in active)
    r0 = next(row for row in active if row['condition'] == 'R0')
    # The manifest is authoritative if an observer filename changes.
    manifest = json.loads((ROOT / 'runs/R0_rerun_seed0/manifest.json').read_text())
    red_flags = manifest.get('trajectory_red_flags', [])
    lines = ['# Stage 2: DPPO clipping ablations', '',
        'Which clipping mechanisms support stable HalfCheetah-v2 fine-tuning, and does their effect depend on reusing a batch for multiple actor updates?', '',
        '**Executed results:** %d of 19 new full runs completed; %d met the prescribed observed-collapse criterion. R0 ended with status `%s`. The fixed-batch differential gate passed before launch; trajectories were not used to test code equivalence.' % (complete, collapsed, r0['status']), '',
        '| Condition | Completed | Final eval mean ± std | Observed collapses |',
        '|---|---:|---:|---:|']
    for condition in ['baseline', 'R0'] + CONDITIONS:
        group = [row for row in summaries if row['condition'] == condition]
        values = [row['eval_return_itr130'] for row in group]
        if all(isinstance(value, (float, int)) for value in values):
            value = ('%.1f ± %.1f' % (statistics.mean(values), statistics.pstdev(values))) if len(values) == 3 else '%.1f (one run)' % values[0]
        else:
            value = 'NA: unavailable endpoint'
        unknown = sum(row['collapse'] is None for row in group)
        collapse_text = '%d/%d' % (sum(row['collapse'] is True for row in group), len(group))
        if unknown:
            collapse_text += ' (%d unknown)' % unknown
        lines.append('| %s | %d/%d | %s | %s |' % (
            condition, sum(row['status'] == 'complete' for row in group), len(group),
            value, collapse_text))
    raw_gate = json.loads(Path(gate['path']).read_text())
    differences = []
    for device in ('cpu', 'gpu'):
        checks = list(raw_gate[device]['full_update']['comparisons'].values())
        checks += list(raw_gate[device]['sampling']['comparisons'].values())
        differences.append('%s %.6g / %.6g' % (device.upper(),
            max(float(check['max_abs']) for check in checks),
            max(float(check['max_rel']) for check in checks)))
    lines += ['',
        'Final evaluation means iteration 130 at 9.36 million training steps; completed training ends at 10.08 million. Bands use population standard deviation over all three requested seeds, without survivor averaging or endpoint substitution. Collapse means an observed evaluation below half the initial return or any recorded scientific NaN/inf. A partial run without an observed collapse does not establish stability.', '',
        'NC1 removes ratio clipping and KL stopping; NC2 additionally removes log-probability and noise clamps; NC3 additionally sums log-probabilities. NC4 uses NC2 with one actor step per batch at the three specified constant learning rates. The table reports their measured endpoints; missing endpoints limit comparisons. [Complete per-seed results and captions](runs/final_analysis/candidates/) retain every failure.', '',
        '[CPU/GPU gate results](runs/verification_full_update/verification.json) include each minibatch loss, actor/critic gradients, final parameters, fixed-noise sampling and RNG isolation. Maximum absolute / relative differences: ' + '; '.join(differences) + '. CPU requires bitwise equality; GPU tolerance was fixed at `atol=1e-7, rtol=1e-5`. [Flag verification](runs/verification/verification.json) additionally checks the NC4 ratio-one/clip-zero invariant.', '',
        'R0 produced %d non-stopping evaluation red flags against the three Stage 1 seeds\' same-step mean ± five sample standard deviations; see its [manifest](runs/R0_rerun_seed0/manifest.json). P1/P2 and the previously stopped R0 were excluded from condition averages and remain preserved.' % len(red_flags), '',
        'Candidate artifacts: [evaluation returns](runs/final_analysis/candidates/eval_return_vs_env_steps.png), [training returns](runs/final_analysis/candidates/train_return_vs_env_steps.png), [diagnostics](runs/final_analysis/candidates/diagnostics_vs_iteration.png), and [condition × seed table](runs/final_analysis/candidates/condition_seed_results.csv). [Analysis records](runs/final_analysis/analysis_record.json) provide native log/result/checkpoint paths, failures, last diagnostics, per-run and per-condition times, and observed GPU sharing.', '',
        'Source remains `ab46b150fa34b5a5b457cd4062cd4c5ad830d964` on `stage2-noclip`; [diff](provenance/dppo_changes.patch), [methods](methods.md), [dependencies](provenance/pip_freeze.txt). Scientific settings and Stage 1 outputs are unchanged. The KL diagnostic is a sampled transition estimate, not the exact final-action KL.', '',
        '**Decision boundary:** all runs are terminal and monitors paused. Candidate artifacts are generated for review, with no curation or further experiments authorized. Keep/revise/leave-uncurated decisions and final names remain for the user.']
    text = '\n'.join(lines) + '\n'
    if len(text.split()) > 600:
        raise ValueError('Generated report exceeds the study report budget')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=ROOT / 'provenance/run_matrix_revised.json')
    parser.add_argument('--campaign', type=Path, default=ROOT / 'runs/campaign_revised.json')
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    if campaign.get('status') != 'runs_terminal_analysis_pending' or len(campaign.get('submitted_jobs', [])) != 18:
        raise ValueError('Campaign has not finished all 18 ablation submissions')
    gate = validate_gate(ROOT / 'runs/verification_full_update/verification.json', COMMIT)
    repo = ROOT / 'dppo'
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Source revision changed before analysis')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Source tree changed before analysis')
    runs = load_campaign(args.matrix, BASELINES)  # Refuses any nonterminal run.
    for run in runs:
        if run['condition'] != 'baseline':
            progress_path = Path(run['manifest_path']).with_name('progress.json')
            if progress_path.exists() and json.loads(progress_path.read_text()).get('monitor_status') != 'paused':
                raise ValueError('Run monitor is still active: ' + run['run_id'])
    output = ROOT / 'runs/final_analysis/candidates'
    record_path = generate(args.matrix, BASELINES, output)
    record = json.loads(record_path.read_text())
    report = current_report(record, gate)
    previous = ROOT / 'runs/final_analysis/experiment_before_finalization.md'
    with previous.open('x') as handle:
        handle.write((ROOT / 'experiment.md').read_text())
    pending = ROOT / 'experiment.md.pending'
    pending.write_text(report)
    pending.replace(ROOT / 'experiment.md')
    with (ROOT / 'runs/final_analysis/finalization.json').open('x') as handle:
        json.dump({'status': 'candidate_artifacts_ready_for_review', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'analysis_record': str(record_path), 'gate': gate, 'curated': False,
                   'stage1_modified': False, 'report_words': len(report.split())}, handle, indent=2)
    catalog = ROOT.parent.parent / 'complete_experiment.md'
    old_catalog = catalog.read_text()
    title = '### DPPO clipping ablations (HalfCheetah-v2)'
    if title not in old_catalog:
        entry = ('\n' + title + '\n\n'
                 '- Experiment folder: [stage2_noclip](results/stage2_noclip/)\n'
                 '- Experiment design: [Methods](results/stage2_noclip/methods.md)\n'
                 '- Curated: —\n\n'
                 '#### Analysis: clipping ablation returns and update diagnostics\n\n'
                 '- Analysis folder: [final_analysis](results/stage2_noclip/runs/final_analysis/)\n'
                 '- Analysis design: [Measurements and aggregation](results/stage2_noclip/methods.md#analysis-and-deliverables)\n'
                 '- Curated: —\n\n')
        if '\n## Comparison analyses\n' not in old_catalog:
            raise ValueError('Completed-experiment catalog structure changed; review required')
        (ROOT / 'runs/final_analysis/complete_experiment_before_finalization.md').write_text(old_catalog)
        catalog.write_text(old_catalog.replace('\n## Comparison analyses\n', entry + '## Comparison analyses\n', 1))
    campaign.update(status='candidate_artifacts_ready_for_review', analysis_record=str(record_path),
                    finalization_record=str(ROOT / 'runs/final_analysis/finalization.json'))
    args.campaign.write_text(json.dumps(campaign, indent=2) + '\n')
    print('Stage 2 candidate artifacts generated; awaiting user review: ' + str(record_path), flush=True)


if __name__ == '__main__':
    main()
