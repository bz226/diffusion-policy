#!/usr/bin/env python
"""Generate only the approved candidate outputs after the revised campaign ends."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess

from analyze_stage2 import generate, load_campaign, summarize_run, resources, CONDITIONS
from fixed_batch_gate import validate_gate
from source_git import source_git_command

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'repro_halfcheetah'
COMMIT = 'ab46b150fa34b5a5b457cd4062cd4c5ad830d964'
BASELINES = [BASE / 'runs' / name / 'manifest.json'
             for name in ('seed0_handoff', 'seed1', 'seed2')]
REPLACEMENTS = {('R0', 0): ('R0_rerun_seed0', 'R0_recovery_seed0'),
                ('NC1', 1): ('NC1_seed1', 'NC1_recovery_seed1')}


def recovery_authorization():
    path = ROOT / 'provenance/interrupted_recovery_authorization.json'
    if not path.exists():
        return None
    authorization = json.loads(path.read_text())
    if authorization.get('approved') is not True:
        raise ValueError('Recovery authorization is not approved')
    expected = {(condition, seed, old, new) for (condition, seed), (old, new) in REPLACEMENTS.items()}
    observed = [(run.get('condition'), run.get('seed'), run.get('replaced_run_id'), run.get('run_id'))
                for run in authorization.get('runs', [])]
    if len(observed) != 2 or set(observed) != expected:
        raise ValueError('Recovery authorization must select exactly the two interrupted runs')
    if (authorization.get('expected_commit') != COMMIT or
            authorization.get('worker_runtime_cap_seconds') != 8100 or
            authorization.get('max_recovery_runs') != 2):
        raise ValueError('Recovery authorization changed source, cap, or number of runs')
    paths = {'original_matrix_path': ROOT / 'provenance/run_matrix_revised.json',
             'matrix_path': ROOT / 'provenance/run_matrix_recovery.json',
             'recovery_record_path': ROOT / 'runs/recovery_campaign.json',
             'primary_campaign_path': ROOT / 'runs/campaign_revised.json'}
    for key, expected_path in paths.items():
        if Path(authorization.get(key, '')).resolve() != expected_path.resolve():
            raise ValueError('Recovery authorization path mismatch: ' + key)
    for run in authorization['runs']:
        if Path(run.get('manifest', '')).resolve() != (ROOT / 'runs' / run['run_id'] / 'manifest.json').resolve():
            raise ValueError('Recovery authorization manifest path mismatch')
    return authorization


def defer_for_recovery(campaign_path, campaign, authorization, recovery_record_path):
    """The original controller finalizer must not emit partial final artifacts."""
    if not authorization or recovery_record_path is not None:
        return False
    campaign.update(status='runs_terminal_recovery_pending',
                    recovery_record=authorization['recovery_record_path'],
                    recovery_matrix=authorization['matrix_path'],
                    finalization_deferred_reason='Two authorized fresh replacements remain pending')
    Path(campaign_path).write_text(json.dumps(campaign, indent=2) + '\n')
    print('Original runs are terminal; final analysis waits for the two authorized replacement attempts.', flush=True)
    return True


def validate_recovery_matrix(matrix_path, original_matrix_path):
    original_matrix_path, matrix_path = Path(original_matrix_path), Path(matrix_path)
    original = json.loads(original_matrix_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if not isinstance(matrix, list) or len(matrix) != 19 or len(original) != 19:
        raise ValueError('Recovery matrix must retain exactly the original 19 condition/seed slots')
    expected = []
    for entry in original:
        new = dict(entry)
        key = (entry['condition'], entry['seed'])
        if key in REPLACEMENTS:
            old_id, new_id = REPLACEMENTS[key]
            if entry['run_id'] != old_id:
                raise ValueError('Original interrupted run ID changed')
            new.update(run_id=new_id, manifest=str(ROOT / 'runs' / new_id / 'manifest.json'),
                       runtime_cap_seconds=8100)
        expected.append(new)
    if matrix != expected:
        raise ValueError('Recovery matrix changed more than the two approved IDs, manifest paths, and caps')
    return matrix


def recovery_selection(matrix_path, recovery_record_path, authorization):
    if not authorization:
        raise ValueError('Recovery finalization requires explicit authorization')
    if (Path(matrix_path).resolve() != Path(authorization['matrix_path']).resolve() or
            Path(recovery_record_path).resolve() != Path(authorization['recovery_record_path']).resolve()):
        raise ValueError('Recovery finalization paths differ from authorization')
    validate_recovery_matrix(matrix_path, authorization['original_matrix_path'])
    recovery = json.loads(Path(recovery_record_path).read_text())
    jobs = recovery.get('submitted_jobs', [])
    if (recovery.get('status') != 'runs_terminal_analysis_pending' or
            recovery.get('monitor_status') != 'paused' or
            len(jobs) != 2 or {job.get('run_id') for job in jobs} != {pair[1] for pair in REPLACEMENTS.values()} or
            any(not job.get('job_id') for job in jobs) or
            recovery.get('retries') != 0 or recovery.get('worker_runtime_cap_seconds') != 8100 or
            recovery.get('commit') != COMMIT):
        raise ValueError('Both authorized recovery attempts must be terminal with monitors paused')
    original_runs = load_campaign(authorization['original_matrix_path'], BASELINES)
    verify_paused(original_runs)
    excluded = [run for run in original_runs if (run['condition'], run['seed']) in REPLACEMENTS]
    if {run['run_id'] for run in excluded} != {pair[0] for pair in REPLACEMENTS.values()}:
        raise ValueError('Interrupted-run provenance is incomplete')
    selected = {'authorization': str(ROOT / 'provenance/interrupted_recovery_authorization.json'),
                'recovery_record': str(recovery_record_path),
                'selection_rule': 'Use the one authorized fresh replacement for each interrupted slot regardless of outcome; never splice or select the better trajectory.',
                'original_matrix': authorization['original_matrix_path'],
                'selected_matrix': str(matrix_path),
                'replacements': authorization['runs'],
                'excluded_interrupted_runs': [dict(summarize_run(run), result_path=run['result_path'],
                    logdir=run['manifest'].get('logdir'), checkpoints=run['manifest'].get('checkpoints', run['manifest'].get('checkpoints_found', []))) for run in excluded],
                'excluded_interrupted_resources': [row for row in resources(original_runs)['runs']
                    if row['run_id'] in {pair[0] for pair in REPLACEMENTS.values()}]}
    return recovery, selected


def verify_paused(runs):
    for run in runs:
        if run['condition'] != 'baseline':
            progress_path = Path(run['manifest_path']).with_name('progress.json')
            if progress_path.exists() and json.loads(progress_path.read_text()).get('monitor_status') != 'paused':
                raise ValueError('Run monitor is still active: ' + run['run_id'])


def current_report(record, gate):
    summaries = record['summaries']
    active = [row for row in summaries if row['condition'] != 'baseline']
    complete = sum(row['status'] == 'complete' for row in active)
    collapsed = sum(row['collapse'] is True for row in active)
    r0 = next(row for row in active if row['condition'] == 'R0')
    # The manifest is authoritative if an observer filename changes.
    r0_source = next(source for source in record['sources'] if source['condition'] == 'R0')
    manifest = json.loads(Path(r0_source['manifest']).read_text())
    r0_manifest_link = str(Path(r0_source['manifest']).relative_to(ROOT))
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
        'R0 produced %d non-stopping evaluation red flags against the three Stage 1 seeds\' same-step mean ± five sample standard deviations; see its [manifest](%s). P1/P2 and the previously stopped R0 remain preserved outside condition averages.' % (len(red_flags), r0_manifest_link), '',
        'Candidate artifacts: [evaluation returns](runs/final_analysis/candidates/eval_return_vs_env_steps.png), [training returns](runs/final_analysis/candidates/train_return_vs_env_steps.png), [diagnostics](runs/final_analysis/candidates/diagnostics_vs_iteration.png), and [condition × seed table](runs/final_analysis/candidates/condition_seed_results.csv). [Analysis records](runs/final_analysis/analysis_record.json) provide native log/result/checkpoint paths, failures, last diagnostics, per-run and per-condition times, and observed GPU sharing.', '',
        'Source remains `ab46b150fa34b5a5b457cd4062cd4c5ad830d964` on `stage2-noclip`; [diff](provenance/dppo_changes.patch), [methods](methods.md), [dependencies](provenance/pip_freeze.txt). Scientific settings and Stage 1 outputs are unchanged. The KL diagnostic is a sampled transition estimate, not the exact final-action KL.', '',
        '**Decision boundary:** all runs are terminal and monitors paused. Candidate artifacts are generated for review, with no curation or further experiments authorized. Keep/revise/leave-uncurated decisions and final names remain for the user.']
    if (ROOT / 'provenance/parallel_handoff_failure.json').exists():
        resolution = (' The authorized fresh replacements `R0_recovery_seed0` and `NC1_recovery_seed1` now occupy those two analysis slots regardless of outcome; original trajectories are excluded, not spliced or averaged together.'
                      if record.get('recovery_selection') else '')
        lines[4:4] = [
            'The agent’s scheduler handoff interrupted [R0](runs/R0_rerun_seed0/manifest.json) and [NC1 seed 1](runs/NC1_seed1/manifest.json) at 138/140 iterations, leaving their final two training rows and checkpoint missing. These orchestration failures remain [preserved](provenance/parallel_handoff_failure.json).' + resolution, '']
    hardware_manifest = ROOT / 'runs/NC1_seed2/manifest.json'
    if hardware_manifest.exists():
        hardware = json.loads(hardware_manifest.read_text()).get('hardware_exception')
        if hardware:
            lines[-2:-2] = [
                'NC1 seed 2 used H200 NVL after a passing [saved-batch H200 check](runs/verification_h200/verification.json); other runs used A6000. This approved hardware difference is a comparison limitation. [Authorization](provenance/h200_seed2_authorization.json) and run manifests retain resource details.', '']
    text = '\n'.join(lines) + '\n'
    if len(text.split()) > 600:
        raise ValueError('Generated report exceeds the study report budget')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=ROOT / 'provenance/run_matrix_revised.json')
    parser.add_argument('--campaign', type=Path, default=ROOT / 'runs/campaign_revised.json')
    parser.add_argument('--recovery-record', type=Path)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    if campaign.get('status') not in ('runs_terminal_analysis_pending', 'runs_terminal_recovery_pending') or len(campaign.get('submitted_jobs', [])) != 18:
        raise ValueError('Campaign has not finished all 18 ablation submissions')
    authorization = recovery_authorization()
    if defer_for_recovery(args.campaign, campaign, authorization, args.recovery_record):
        return
    recovery = selection = None
    expected_r0 = 'R0_rerun_seed0'
    if args.recovery_record is not None:
        recovery, selection = recovery_selection(args.matrix, args.recovery_record, authorization)
        expected_r0 = 'R0_recovery_seed0'
    elif campaign.get('status') == 'runs_terminal_recovery_pending':
        raise ValueError('Deferred recovery authorization is unavailable')
    gate = validate_gate(ROOT / 'runs/verification_full_update/verification.json', COMMIT)
    repo = ROOT / 'dppo'
    if subprocess.check_output(source_git_command(repo, 'rev-parse', 'HEAD'), text=True).strip() != COMMIT:
        raise ValueError('Source revision changed before analysis')
    if subprocess.check_output(source_git_command(repo, 'status', '--porcelain'), text=True).strip():
        raise ValueError('Source tree changed before analysis')
    runs = load_campaign(args.matrix, BASELINES, expected_r0)  # Refuses any nonterminal run.
    verify_paused(runs)
    output = ROOT / 'runs/final_analysis/candidates'
    record_path = generate(args.matrix, BASELINES, output, expected_r0, selection)
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
                   'stage1_modified': False, 'report_words': len(report.split()),
                   'recovery_record': str(args.recovery_record) if args.recovery_record else None,
                   'selected_r0_run_id': expected_r0}, handle, indent=2)
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
    if recovery is not None:
        recovery.update(status='candidate_artifacts_ready_for_review', analysis_record=str(record_path),
                        finalization_record=str(ROOT / 'runs/final_analysis/finalization.json'))
        args.recovery_record.write_text(json.dumps(recovery, indent=2) + '\n')
    print('Stage 2 candidate artifacts generated; awaiting user review: ' + str(record_path), flush=True)


if __name__ == '__main__':
    main()
