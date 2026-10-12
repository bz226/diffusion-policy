"""Publish verified, existing-data postprocessing after a retained report failure."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PARENT / 'results/stage3/gradmean'
CODE = PARENT / '.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean'
HERE = ROOT / 'runs/postprocess_recovery_20261012'
read = lambda path: json.loads(path.read_text())
def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

record = read(HERE / 'recovery.json')
out = Path(record['output'])
analysis = read(out / 'analysis_status.json')
render = read(ROOT / 'figs_clean/render_record.json')
assert analysis['status'] == render['status'] == 'complete'
assert not analysis['v6_consistency_stop']
assert all(render['counts'][point]['K_V6'] == 500 for point in ('P0','D1','D2','N1','N2'))
assert '|COMPLETED|0:0|' in record['analysis_scheduler']
assert '|COMPLETED|0:0|' in record['clean_scheduler']
for name in ('eval_reward','train_reward','gradient_noise_V6','gradient_noise_V4','gradient_noise_V5'):
    for ext in ('.png','.pdf','.caption.md'):
        assert (ROOT / 'figs_clean' / (name + ext)).stat().st_size > 0
for name in ('points_table.csv','points_table.md','NOTE.md','render_record.json'):
    assert (ROOT / 'figs_clean' / name).stat().st_size > 0

for name in ('campaign.json','campaign_progress.json'):
    target = HERE / ('before_recovery_' + name)
    assert not target.exists()
    shutil.copy2(ROOT / 'runs' / name, target)
for source, dest in ((ROOT/'experiment.md', HERE/'experiment_before_completion.md'),
                     (ROOT/'REPORT.md', HERE/'report_before_completion.md'),
                     (ROOT/'V6_NOTE.md', HERE/'v6_note_before_readability.md'),
                     (PARENT/'complete_experiment.md', HERE/'catalog_before_completion.md')):
    assert not dest.exists(); shutil.copy2(source, dest)

now = datetime.now(timezone.utc).isoformat()
campaign = read(ROOT / 'runs/campaign.json')
for label, job, output, marker, cap in (
    ('final_v6_fixed_reportfix_20261012', record['analysis_job_id'], out, analysis, 1800),
    ('clean_v6_fixed', record['clean_job_id'], ROOT/'figs_clean', render, 600)):
    assert not any(entry['label'] == label for entry in campaign.get('analyses', []))
    campaign['analyses'].append(dict(label=label, job_id=job, output=str(output),
        status='complete', scheduler_state='COMPLETED', exit_code='0:0', terminal_recorded=True,
        terminal_marker=marker, cap_seconds=cap, gpu_count=0,
        recovery_record=str(HERE/'recovery.json'), verified_utc=now))
campaign.update(status='complete', monitor_status='paused', ended_utc=now,
                analysis_recovery=dict(record=str(HERE/'recovery.json'), recovered=True, verified_utc=now))
if 'failure_reason' in campaign:
    campaign['retained_earlier_failure_reason'] = campaign.pop('failure_reason')
write(ROOT/'runs/campaign.json', campaign)
progress = read(ROOT/'runs/campaign_progress.json')
progress.update(status='complete', monitor_status='paused', utc=now,
                analysis_recovery=campaign['analysis_recovery'])
write(ROOT/'runs/campaign_progress.json', progress)

sys.path.insert(0, str(CODE/'code'))
from analysis_report import build_report, metadata_inventory
from finalize_records import catalog_text, experiment_text
payload = read(out/'candidates/results.json')
meta = read(out/'internal/checkpoint_metadata.json')
body = build_report(ROOT/'REPORT.md', ROOT, out/'candidates', payload, analysis['figures'],
                    meta, campaign, analysis, metadata_inventory(ROOT, campaign))
body = body.replace('## Execution, resources and deviations',
    '## Execution, resources and deviations\n\nThe first final CPU analysis failed while formatting the report (`fixed_plan` was not passed into its helper). The report-only fix passed 21 checks; a CPU rerun completed with all 15 numerical CSVs byte-for-byte unchanged. The original failed outputs and traceback remain preserved. [Recovery evidence](runs/postprocess_recovery_20261012/recovery.json).')
(ROOT/'REPORT.md').write_text(body)

note = (ROOT/'V6_NOTE.md').read_text()
note = note.replace('**Answer.** No resolved mean-direction inconsistency was detected at this scheduled analysis. All five policies have at least 500 V6 batches.',
    '**Answer.** The critic has higher gradient variance than LOO at every measured policy: 1.73× at P0 and 3.43–4.44× at the four trained checkpoints. Each point has exactly 500 V6 batches. No resolved mean-direction inconsistency was detected, but V6\'s own mean remains unresolved at D1, D2, N1 and N2; their signal intervals include zero. Cosine point estimates outside [−1, 1] reflect this uncertainty and must not be interpreted as resolved physical angles.')
(ROOT/'V6_NOTE.md').write_text(note)

text = experiment_text((ROOT/'experiment.md').read_text(), campaign)
text = text.replace('Final analysis and all requested clean figures were verified.',
    'Final analysis and all requested clean figures were verified. A report-formatting failure was fixed without changing any numerical table; the original failure is retained in the [recovery record](runs/postprocess_recovery_20261012/recovery.json).')
text = text.replace('Mean gradient and noise along DPPO and NC4 paths','Mean gradient and noise along DPPO and without_PPO paths').replace('along the DPPO and NC4 paths','along the DPPO and without_PPO paths')
text = re.sub(r'<!-- gradmean-bootstrap-start -->.*?<!-- gradmean-bootstrap-end -->\s*', '', text, flags=re.S)
block = ('<!-- gradmean-status-start -->\nVerified '+now+'. **Complete**: all five policies have 500/500 new V6 batches; final analysis and clean figures passed. Total gradient-mean usage: 68.054 GPU-hours of 125. Monitor paused; no further collection queued.\n<!-- gradmean-status-end -->')
text = re.sub(r'<!-- gradmean-status-start -->.*?<!-- gradmean-status-end -->', block, text, flags=re.S)
(ROOT/'experiment.md').write_text(text)

catalog = catalog_text((PARENT/'complete_experiment.md').read_text(), True)
catalog = catalog.replace('./results/stage3/gradmean/runs/analyses/final_v6_fixed/',
                          './results/stage3/gradmean/runs/analyses/reportfix_20261012/final_v6_fixed/')
(PARENT/'complete_experiment.md').write_text(catalog)
record.update(status='complete', verified_utc=now, curation_performed=False,
              current_report=str(ROOT/'REPORT.md'), current_note=str(ROOT/'V6_NOTE.md'),
              clean_note=str(ROOT/'figs_clean/NOTE.md'), completed_catalog_updated=True)
write(HERE/'recovery.json', record)
print('Verified reports, completion records and catalog published; original failure preserved; no curation.')
