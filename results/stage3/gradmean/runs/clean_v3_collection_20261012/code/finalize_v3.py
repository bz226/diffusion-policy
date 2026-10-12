"""Publish completion bookkeeping only after the fixed collection and checks pass."""
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/gradmean'
RUN = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def replace_paragraph(text, label, replacement):
    pattern = r'\*\*' + re.escape(label) + r'\.\*\*.*?(?=\n\n|\Z)'
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Missing current study paragraph: ' + label)
    return updated


def main():
    state = read(RUN / 'campaign.json')
    record = read(ROOT / 'figs_clean/render_record.json')
    if state['status'] != 'complete' or len(state['entries']) != 22:
        raise ValueError('The fixed 22-job collection is not complete')
    if any(e.get('status') != 'complete' or e.get('scheduler_state') != 'COMPLETED'
           or e.get('exit_code') != '0:0' for e in state['entries']):
        raise ValueError('A worker lacks verified scheduler completion')
    for point in ('C1', 'C2'):
        groups = [e for e in state['entries'] if e['run_id'].startswith(point + '_j')]
        if len(groups) != 10:
            raise ValueError('Wrong number of new collection jobs')
    if record['version'] != 3 or record['status'] != 'complete' or record['pending_points']:
        raise ValueError('Final seven-point figures are not complete')
    render = Path(record['run'])
    checks = [read(p) for p in sorted(render.glob('checks*.json'))]
    verified = [c for c in checks if c.get('passed') is True and c.get('require_all') is True
                and c.get('compared_numeric_cells') == 1008]
    if not verified:
        raise ValueError('Final independent table/artifact verification has not passed')
    if state['ledger']['aggregate_actual_plus_reserved'] > state['ledger']['aggregate_gpu_seconds_cap']:
        raise ValueError('Aggregate approved compute budget exceeded')
    utc = datetime.now(timezone.utc).isoformat()
    result = dict(status='complete', verified_utc=utc, collection_controller=state['controller_job_id'],
                  collection_jobs=[e['job_id'] for e in state['entries']], collection_status=str(RUN / 'campaign.json'),
                  render_run=str(render), verification=verified[-1], figures=str(ROOT / 'figs_clean'),
                  K_per_point=500, episodes_per_point=40000, points=list(record['statistics']['points']),
                  new_gpu_hours=state['ledger']['new_actual_gpu_seconds'] / 3600.,
                  aggregate_gpu_hours=state['ledger']['aggregate_actual_plus_reserved'] / 3600.,
                  curated=False, monitor_status='paused', next_step='User artifact review; no further collection.')
    destination = RUN / 'finalization.json'
    if destination.exists():
        raise ValueError('Refusing to overwrite finalization evidence')
    catalog = PROJECT / 'complete_experiment.md'
    catalog_text = catalog.read_text()
    if '## Comparison analyses\n' not in catalog_text:
        raise ValueError('Missing comparison analysis section')
    study = ROOT / 'experiment.md'
    shutil.copy2(str(study), str(RUN / 'experiment_before_v3_completion.md'))
    text = study.read_text()
    text = replace_paragraph(text, 'Status',
        '**Status.** Version 3 is complete: all seven policies have matched 500-batch cohorts, and the five requested figures '
        'and checkpoint table passed independent recomputation and artifact checks. The two new policies used exactly '
        'two preparations and twenty collection jobs, with no actor updates. '
        '[Figures, table and limitations](figs_clean/NOTE.md) · '
        '[Completion evidence](runs/clean_v3_collection_20261012/finalization.json). '
        'Earlier approved [version-2 artifacts](curated/INDEX.md#clean-figures-v2) remain preserved.')
    text = replace_paragraph(text, 'Next step',
        '**Next step.** Review the five version-3 figures and checkpoint table for keep, revise or leave uncurated. '
        'The existing filenames are suggested names; prior curated versions will not be overwritten without approval. '
        'Collection and monitoring have stopped. No new scientific sampling is queued.')
    text += ('\n<!-- clean-v3-analysis-start -->\nVerified ' + utc + ': all seven points use K=500 / m=40,000; '
             '1,008 numerical table values independently recomputed. New collection used %.3f GPU-hours; '
             'aggregate gradient-mean use is %.3f of 125 GPU-hours. '
             'Version-3 artifacts are uncurated candidates.\n<!-- clean-v3-analysis-end -->\n' %
             (result['new_gpu_hours'], result['aggregate_gpu_hours']))
    study.write_text(text)
    shutil.copy2(str(catalog), str(RUN / 'catalog_before_v3_completion.md'))
    text = catalog_text
    heading = '### DPPO, without_PPO and without_PPO_and_GAE: returns and gradient noise'
    if heading not in text:
        entry = ('\n' + heading + '\n\n'
                 '- Analysis folder: [Clean figures version 3](./results/stage3/gradmean/figs_clean/)\n'
                 '- Analysis design: [Matched-sample protocol](./results/stage3/gradmean/methods.md#clean-figures-version-3)\n'
                 '- Curated: —\n')
        marker = '## Comparison analyses\n'
        if marker not in text:
            raise ValueError('Missing comparison analysis section')
        text = text.replace(marker, marker + entry, 1)
    # The older comparison must keep pointing to its own preserved candidates.
    text = text.replace('- Analysis folder: [Clean figures and tables](./results/stage3/gradmean/figs_clean/)',
        '- Analysis folder: [Clean figures version 2](./results/stage3/gradmean/figs_clean/history/20261012T161233.062570Z/)')
    catalog.write_text(text)
    child = RUN / 'experiment.md'
    child_text = child.read_text().replace(
        'Next step: regenerate the authorized clean figures from completed measurements.',
        'Next step: review the verified version-3 figures and table for curation; no further sampling is queued.')
    child_text += '\nFinal rendering and independent verification completed: [completion evidence](finalization.json).\n'
    child.write_text(child_text)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'new_gpu_hours', 'aggregate_gpu_hours', 'figures', 'curated')}, indent=2))


if __name__ == '__main__':
    main()
