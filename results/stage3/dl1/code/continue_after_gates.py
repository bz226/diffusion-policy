"""Approved DL1 gate continuation and scheduled monitor; Python 3.8.

Never reruns a gate, cancels another job, or admits training on a failed gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import campaign
from worker import atomic_json, append_json, read_json, utc, verify_source

ROOT = campaign.ROOT
STATE = ROOT / 'runs/gate_continuation/state.json'
HISTORY = ROOT / 'runs/gate_continuation/monitor_history.jsonl'


def publish(state, force=False):
    state['updated_utc'] = utc()
    atomic_json(STATE, state)
    if force:
        append_json(HISTORY, state)
        path = ROOT / 'experiment.md'
        text = path.read_text()
        start, end = '<!-- DL1 gate monitor -->', '<!-- end DL1 gate monitor -->'
        block = '\n'.join([start, '**Gate status (%s):** %s. %s' % (
            state['updated_utc'], state['status'], state.get('detail', '')),
            '[Continuation record](runs/gate_continuation/state.json).', end])
        if start in text:
            text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
        else:
            text = text.rstrip() + '\n\n' + block + '\n'
        path.write_text(text)


def wait_for(job_id, summary, state, phase):
    last_counter, unchanged, next_monitor = None, 0, 0.0
    while True:
        info = campaign.scheduler_rows([job_id]).get(job_id)
        gate = read_json(summary, {})
        counter = gate.get('active_worker') or gate.get('status')
        state.update(status='waiting_' + phase, scheduler=info, gate_status=gate.get('status'),
                     detail='Job %s; active gate worker: %s.' % (job_id, counter or 'initializing'))
        if time.monotonic() >= next_monitor:
            unchanged = unchanged + 1 if counter and counter == last_counter else 0
            last_counter = counter
            if unchanged == 2:
                state['stall_warning'] = 'No gate-worker transition in two 30-minute checks; inspect live logs.'
            publish(state, True)
            next_monitor = time.monotonic() + 1800
        if info and info['state'] in campaign.TERMINAL:
            if info['state'] != 'COMPLETED' or info['exit_code'] != '0:0' or not gate.get('passed'):
                raise RuntimeError('%s gate failed or exited unsuccessfully: %r; %s' % (phase, info, gate.get('error')))
            return info, gate
        if gate.get('status') == 'failed':
            raise RuntimeError('%s gate failed: %s' % (phase, gate.get('error')))
        publish(state)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-job', required=True)
    args = parser.parse_args()
    if STATE.exists():
        raise RuntimeError('Continuation already exists; do not start a duplicate controller')
    state = dict(status='started', started_utc=utc(), controller_job_id=os.environ.get('SLURM_JOB_ID'),
                 cpu_job=args.cpu_job, monitor_status='active', monitor_seconds=1800)
    publish(state, True)
    try:
        source = read_json(ROOT / 'provenance/launch_source.json')
        verify_source(source)
        cpu_path = ROOT / 'runs/gates/cpu/gate_summary.json'
        cpu_info, cpu = wait_for(args.cpu_job, cpu_path, state, 'cpu')
        if max(cpu_info['elapsed_seconds'], cpu['cpu_gate_seconds']) > 7200:
            raise RuntimeError('CPU gate cap exceeded')
        verify_source(source)
        command = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
                   '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--time=60', '--no-requeue',
                   '--job-name=dl1-live-gate', '--output=' + str(ROOT / 'runs/logs/live_gate.scheduler.out'),
                   '--error=' + str(ROOT / 'runs/logs/live_gate.scheduler.err'), str(ROOT / 'code/run_gate.sh'), 'live']
        state.update(status='submitting_live_gate', live_submission_command=command)
        publish(state, True)
        reply = campaign.command(command).strip().split(';')[0]
        if not reply.isdigit():
            raise RuntimeError('Ambiguous live-gate submission; inspect scheduler before any retry')
        state['live_job'] = reply
        atomic_json(ROOT / 'provenance/live_gate_submission.json', dict(job_id=reply, command=command, utc=utc()))
        publish(state, True)
        live_path = ROOT / 'runs/gates/live/gate_summary.json'
        live_info, live = wait_for(reply, live_path, state, 'live')
        verify_source(source)
        if max(live_info['elapsed_seconds'], live['gpu_gate_seconds']) > 3600:
            raise RuntimeError('GPU gate cap exceeded')
        merge = [str(campaign.PYTHON), '-B', str(ROOT / 'code/gate_dl1.py'), '--phase', 'merge',
                 '--source', source['source'], '--output', str(ROOT / 'runs/gates/complete'),
                 '--cpu-result', str(cpu_path), '--live-result', str(live_path)]
        subprocess.run(merge, check=True, timeout=120)
        complete_path = ROOT / 'runs/gates/complete/gate_summary.json'
        complete = read_json(complete_path)
        complete['phase_wall_seconds'] = dict(cpu=complete['cpu_gate_seconds'], gpu=complete['gpu_gate_seconds'])
        complete['cpu_gate_seconds'] = max(cpu_info['elapsed_seconds'], complete['cpu_gate_seconds'])
        complete['gpu_gate_seconds'] = max(live_info['elapsed_seconds'], complete['gpu_gate_seconds'])
        complete['scheduler_accounting'] = dict(cpu=cpu_info, gpu=live_info)
        atomic_json(complete_path, complete)
        state.update(status='gates_passed', detail='G1–G6 passed; building the three-seed DPPO launch plan.')
        publish(state, True)
        command = [str(campaign.PYTHON), '-B', str(ROOT / 'code/campaign.py'), 'build-training',
                   '--revision', source['revision'], '--gate', str(complete_path)]
        subprocess.run(command, check=True, timeout=120)
        state.update(status='training_controller_active', detail='All gates passed; training admissions and 30-minute monitoring are active.')
        publish(state, True)
        # The isolated Stage-3 runner applies DPPO only when explicitly selected.
        environment = os.environ.copy()
        environment['SKELETON'] = 'dppo'
        runner = Path(source['worktree']) / 'results/stage3/run_queue.sh'
        command = ['bash', str(runner), 'run', str(ROOT / 'runs/plan_training.json')]
        state['training_controller_command'] = command
        publish(state)
        subprocess.run(command, check=True, env=environment)
        completed = read_json(ROOT / 'runs/campaign_training/campaign.json', {})
        if completed.get('status') != 'complete' or completed.get('analysis_status') != 'complete':
            raise RuntimeError('Training controller exited before verified training and final analysis completion')
        state.update(status='complete', monitor_status='paused', ended_utc=utc(),
                     detail='Training and its requested final analysis finished; see campaign status.')
        publish(state, True)
    except BaseException as error:
        state.update(status='blocked', monitor_status='paused', error=repr(error), traceback=traceback.format_exc(),
                     detail='Admission stopped. No scientific retry or configuration change was made.')
        publish(state, True)
        raise


if __name__ == '__main__':
    main()
