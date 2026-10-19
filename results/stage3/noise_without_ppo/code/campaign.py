"""Bounded Slurm admission, verification and 30-minute monitoring; Python 3.8."""
import argparse
import csv
import fcntl
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime import (PROJECT, STUDY, RUN, WORKTREE, SOURCE, PYTHON, utc, read, write,
                     append, command, environment, verify_source, bounded_worker)

CAP = 30 * 3600
MAX_PAR = 5
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
POINTS = [(0, 0), (35, 32), (70, 63), (105, 95), (139, 126)]
START, END = '<!-- noise monitor -->', '<!-- end noise monitor -->'


def build():
    if (RUN / 'plan.json').exists():
        raise RuntimeError('Refusing to replace the approved executable plan')
    revision = verify_source()
    jobs, points, collect_paths = {}, [], []
    for p, (itr, updates) in enumerate(POINTS):
        point = 'WP%03d' % itr
        checkpoint = PROJECT / 'results/stage2_noclip/runs/NC4_lr1e-4_seed0/native/checkpoint' / ('state_%d.pt' % itr)
        if not checkpoint.is_file():
            raise RuntimeError('Original requested checkpoint is missing')
        frozen = RUN / 'prepare' / point / 'prepare.json'
        entry = dict(point=point, point_index=p, checkpoint_itr=itr, updates=updates,
                     checkpoint=str(checkpoint), frozen_path=str(frozen))
        points.append(entry)
        for kind, index in [('prepare', 99)] + [('collect', j) for j in range(10)]:
            identifier = 'prepare_' + point if kind == 'prepare' else point + '_j%02d' % index
            out = RUN / kind / (point if kind == 'prepare' else identifier)
            spec = dict(entry, operation=kind, job_index=index, out=str(out),
                        batches=50, seed_base=50000000, device='cuda:0',
                        parent_revision=revision,
                        normalization_path=str(PROJECT / 'results/repro_halfcheetah/dppo/data/gym/halfcheetah-medium-v2/normalization.npz'))
            spec_path = RUN / 'specs' / (identifier + '.json')
            write(spec_path, spec)
            if kind == 'collect':
                collect_paths.append(str(spec_path))
            jobs[identifier] = dict(id=identifier, phase=kind, out=str(out),
                command=[str(PYTHON), '-B', '-m', 'agent.finetune.noise_without_ppo', '--spec', str(spec_path)],
                cap_seconds=600 if kind == 'prepare' else 1800, gpu_count=1, spec=str(spec_path))
    study_spec = RUN / 'study_spec.json'
    write(study_spec, dict(points=points, collect_specs=collect_paths, seed_base=50000000))
    cpu_summary = STUDY / 'runs/gates/cpu/gate_summary.json'
    for phase, cap, gpu in [('cpu', 7200, 0), ('live', 1800, 1)]:
        identifier = 'gate_' + phase
        argv = [str(PYTHON), '-B', str(STUDY / 'code/gate_noise.py'), '--phase', phase,
                '--source', str(SOURCE), '--parent-source', str(PROJECT / '.codex-runtime/worktrees/stage3-gradmean-v5/results/stage3_gradmean/dppo'),
                '--output', str(STUDY / 'runs/gates' / phase)]
        if phase == 'live':
            argv += ['--cpu-result', str(cpu_summary), '--study-spec', str(study_spec)]
        jobs[identifier] = dict(id=identifier, phase=identifier, out=str(STUDY / 'runs/gates' / phase),
                               command=argv, cap_seconds=cap, gpu_count=gpu)
    plan = dict(revision=revision, max_parallel=MAX_PAR, gpu_seconds_cap=CAP,
                monitor_seconds=1800, poll_seconds=30, points=points, jobs=jobs,
                expected_gpu_hours=10.5, expected_collection_hours=2.5,
                estimate_basis='50 preceding clean-v5 jobs: mean735.84s/job; total10.22GPUh; rollouts dominate.',
                source_snapshot=str(STUDY / 'runs/source_snapshot'),
                automatic_retries=False, bounded_retry_policy='Only after a documented cause fix, new seeds, remaining cap; no silent rerun.')
    if sum(j['cap_seconds'] * j['gpu_count'] for j in jobs.values()) > CAP:
        raise RuntimeError('Complete approved plan cannot fit its reservation cap')
    write(RUN / 'plan.json', plan)
    append(STUDY / 'runs/runtime_estimates.jsonl', dict(utc=utc(), expected_gpu_hours=10.5,
            expected_collection_hours=2.5, monitor_seconds=1800, basis=plan['estimate_basis']))
    return plan


def scheduler(ids):
    if not ids:
        return {}
    output = command(['sacct', '-n', '-P', '--allocations', '-j', ','.join(ids),
                      '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%160,NodeList'])
    rows = {}
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) >= 6 and fields[0] in ids:
            tres = dict(x.split('=', 1) for x in fields[4].split(',') if '=' in x)
            rows[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'), exit_code=fields[2],
                                   elapsed_seconds=int(fields[3] or 0),
                                   gpu_count=int(tres.get('gres/gpu', '0')), node=fields[5])
    missing = [identifier for identifier in ids if identifier not in rows]
    if missing:
        output = command(['squeue', '-h', '-j', ','.join(missing), '-o', '%i|%T|%N|%S'])
        for line in output.splitlines():
            fields = line.split('|')
            if len(fields) >= 4 and fields[0] in missing:
                rows[fields[0]] = dict(state=fields[1], exit_code=None, elapsed_seconds=0,
                                       node=fields[2], predicted_start=fields[3])
    return rows


def costs(plan, state):
    actual, reserved = 0, 0
    for name, record in state['jobs'].items():
        job = plan['jobs'][name]
        if record['status'] in ('complete', 'failed'):
            actual += record.get('scheduler', {}).get('elapsed_seconds', job['cap_seconds']) * job['gpu_count']
        else:
            reserved += job['cap_seconds'] * job['gpu_count']
    return actual, reserved


def submit(plan, state, name):
    if name in state['jobs']:
        raise RuntimeError('Admission already recorded: ' + name)
    job = plan['jobs'][name]
    actual, reserved = costs(plan, state)
    if actual + reserved + job['gpu_count'] * job['cap_seconds'] > CAP:
        raise RuntimeError('Aggregate GPU cap would be exceeded')
    worker_spec = RUN / 'workers' / name / 'spec.json'
    write(worker_spec, job)
    logs = RUN / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=40', '--mem=64G', '--no-requeue', '--time=%d' % math.ceil(job['cap_seconds'] / 60),
            '--job-name=wpnoise-' + name, '--output=' + str(logs / (name + '.out')),
            '--error=' + str(logs / (name + '.err'))]
    if job['gpu_count']:
        argv += ['--gres=gpu:a6000:1']
    argv += [str(STUDY / 'code/run_admin.sh'), 'worker', '--spec', str(worker_spec)]
    state['jobs'][name] = dict(status='submission_unknown', submitted_utc=utc(), command=argv)
    write(RUN / 'state.json', state)
    answer = command(argv).strip().split(';')[0]
    if not answer.isdigit():
        raise RuntimeError('Ambiguous submission; never submit a duplicate automatically')
    state['jobs'][name].update(job_id=answer, status='pending')
    write(RUN / 'state.json', state)


def verify_job(job):
    folder = Path(job['out'])
    if job['phase'].startswith('gate_'):
        result = read(folder / 'gate_summary.json', {})
        if not result.get('passed'):
            raise RuntimeError('Gate failed: ' + str(result.get('error', folder)))
        return result
    result = read(folder / ('prepare.json' if job['phase'] == 'prepare' else 'meta.json'), {})
    cleanup = read(folder / 'cleanup.json', {})
    if result.get('status') != 'completed' or not result.get('clean_exit'):
        raise RuntimeError('Scientific result is incomplete: ' + str(folder))
    if cleanup.get('status') != 'passed' or not cleanup.get('all_children_exited'):
        raise RuntimeError('Explicit child cleanup not verified: ' + str(folder))
    if job['phase'] == 'collect':
        import numpy as np
        if result.get('K') != 50:
            raise RuntimeError('Group does not contain exactly50batches')
        with (folder / 'batches.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 50 or [int(r['batch']) for r in rows] != list(range(50)):
            raise RuntimeError('Missing or repeated batch index')
        for key in ('norm_sq_V1', 'norm_sq_V2', 'cross_V1_V2'):
            if not all(math.isfinite(float(r[key])) for r in rows):
                raise RuntimeError('Nonfinite saved gradient scalar')
        for variant in (1, 2):
            vector = np.load(str(folder / ('sums_V%d.npy' % variant)), allow_pickle=False)
            if vector.dtype != np.float64 or vector.shape != (568392,) or not np.isfinite(vector).all():
                raise RuntimeError('Invalidfloat64group sum')
    return result


def refresh(plan, state):
    active = {name: row for name, row in state['jobs'].items() if row['status'] not in ('complete', 'failed')}
    rows = scheduler([row['job_id'] for row in active.values() if row.get('job_id')])
    failures = []
    for name, row in active.items():
        status = rows.get(row.get('job_id'))
        if not status:
            row['missing_polls'] = row.get('missing_polls', 0) + 1
            if row['missing_polls'] >= 6:
                failures.append('Scheduler cannot locate ' + name)
            continue
        row.update(scheduler=status, missing_polls=0)
        folder = Path(plan['jobs'][name]['out'])
        row['progress'] = read(folder / 'progress.json', read(folder / 'gate_summary.json', {}))
        if status['state'] in TERMINAL:
            manifest = read(RUN / 'workers' / name / 'manifest.json', {})
            try:
                if status['state'] != 'COMPLETED' or status['exit_code'] != '0:0' or manifest.get('status') != 'complete':
                    raise RuntimeError('%s/%s; worker=%s' % (status['state'], status['exit_code'], manifest.get('error', manifest.get('status'))))
                result = verify_job(plan['jobs'][name])
                row.update(status='complete', verified_utc=utc(), K=result.get('K'))
                append(RUN / 'events.jsonl', dict(utc=utc(), event='job_verified', job=name, job_id=row['job_id']))
            except Exception as error:
                row.update(status='failed', error=str(error))
                failures.append(name + ': ' + str(error))
        else:
            row['status'] = 'running' if status['state'] == 'RUNNING' else 'pending'
    return failures


def publish(plan, state, due=False):
    actual, reserved = costs(plan, state)
    done = sum(row['status'] == 'complete' and plan['jobs'][name]['phase'] == 'collect'
               for name, row in state['jobs'].items())
    state.update(updated_utc=utc(), complete_collection_jobs=done,
                 used_gpu_hours=actual / 3600., reserved_gpu_hours=reserved / 3600.)
    write(RUN / 'state.json', state)
    if due:
        append(RUN / 'monitor_history.jsonl', state)
        path = STUDY / 'experiment.md'
        text = path.read_text()
        block = '\n'.join([START, '**Execution status (%s):** %s; %d/50 collection jobs verified.' %
                           (state['updated_utc'], state['status'], done), '',
                           'Completed allocations: %.3f GPU-hours; reserved: %.3f; cap:30. Monitor:%s.' %
                           (actual / 3600., reserved / 3600., state['monitor_status']), '',
                           '[Current execution record](runs/campaign/state.json). ' + state.get('detail', ''), END])
        if START in text and END in text:
            text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
        else:
            text = text.rstrip() + '\n\n' + block + '\n'
        path.write_text(text)


def preserved_figures():
    before = read(STUDY / 'runs/figure_hashes_before.json')
    after = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in before}
    write(STUDY / 'runs/figure_hashes_after.json', after)
    if before != after:
        raise RuntimeError('A preexisting figure changed')
    return len(after)


def controller():
    plan = read(RUN / 'plan.json')
    if not plan or plan['gpu_seconds_cap'] != CAP or plan['max_parallel'] != MAX_PAR:
        raise RuntimeError('Missing approved immutable plan')
    lock = (RUN / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(RUN / 'state.json')
    if state is not None:
        raise RuntimeError('Existing controller record; inspect before restarting')
    state = dict(status='cpu_gates', started_utc=utc(), controller_job_id=os.environ.get('SLURM_JOB_ID'),
                 monitor_status='active', jobs={}, notifications=[])
    last_monitor, last_counter, stagnant = -1800, None, 0
    started = time.monotonic()
    try:
        verify_source()
        while True:
            failures = refresh(plan, state)
            if failures:
                raise RuntimeError('; '.join(failures))
            done = {name for name, row in state['jobs'].items() if row['status'] == 'complete'}
            active = [name for name, row in state['jobs'].items() if row['status'] not in ('complete', 'failed')]
            if 'gate_cpu' not in done:
                wanted = ['gate_cpu']
                state['status'] = 'cpu_gates'
            else:
                prepares = [name for name, job in plan['jobs'].items() if job['phase'] == 'prepare']
                if not set(prepares).issubset(done):
                    wanted, state['status'] = prepares, 'prepare_states'
                elif 'gate_live' not in done:
                    wanted, state['status'] = ['gate_live'], 'live_gates'
                else:
                    complete_gate = STUDY / 'runs/gates/complete/gate_summary.json'
                    if not complete_gate.exists():
                        command([PYTHON, '-B', STUDY / 'code/gate_noise.py', '--phase', 'merge',
                                 '--source', SOURCE, '--output', complete_gate.parent,
                                 '--cpu-result', STUDY / 'runs/gates/cpu/gate_summary.json',
                                 '--live-result', STUDY / 'runs/gates/live/gate_summary.json'], timeout=120)
                    if not read(complete_gate).get('passed'):
                        raise RuntimeError('Merged gate did not pass')
                    wanted = [name for name, job in plan['jobs'].items() if job['phase'] == 'collect']
                    state['status'] = 'collecting'
                    if set(wanted).issubset(done):
                        break
            for name in wanted:
                if name not in state['jobs'] and len(active) < MAX_PAR:
                    verify_source()
                    submit(plan, state, name)
                    active.append(name)
            due = time.monotonic() - started - last_monitor >= 1800
            if due:
                counter = [(name, state['jobs'][name]['status'],
                            state['jobs'][name].get('progress', {}).get('completed'),
                            state['jobs'][name].get('progress', {}).get('active_worker')) for name in sorted(state['jobs'])]
                running = [name for name in active if state['jobs'][name]['status'] == 'running']
                stagnant = stagnant + 1 if counter == last_counter and running else 0
                last_counter = counter
                if stagnant >= 2:
                    state['detail'] = 'Possible stall: no progress across two scheduled checks; inspect active logs.'
                    append(RUN / 'events.jsonl', dict(utc=utc(), event='stall_requires_inspection', jobs=active))
                last_monitor = time.monotonic() - started
            publish(plan, state, due)
            time.sleep(30)
        state['status'] = 'analyzing'
        publish(plan, state, True)
        env = environment()
        candidate = RUN / 'candidates'
        argv = [str(PYTHON), '-B', str(STUDY / 'code/analyze.py'), '--collect-root', str(RUN / 'collect'),
                '--output', str(candidate), '--require-complete', '--write-report']
        append(RUN / 'commands.jsonl', dict(utc=utc(), argv=argv))
        with (RUN / 'analysis.out').open('x') as log:
            subprocess.run(argv, cwd=str(STUDY), env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=600, check=True)
        state.update(status='complete', monitor_status='paused', ended_utc=utc(),
                     preserved_figure_count=preserved_figures(), detail='All50groups and candidate analysis verified. Curation awaits a separate decision.')
        append(RUN / 'events.jsonl', dict(utc=utc(), event='study_complete', candidates=str(candidate)))
        publish(plan, state, True)
    except BaseException as error:
        state.update(status='blocked', monitor_status='paused', ended_utc=utc(),
                     detail=str(error), traceback=traceback.format_exc())
        append(RUN / 'events.jsonl', dict(utc=utc(), event='terminal_failure', error=str(error)))
        publish(plan, state, True)
        report = STUDY / 'REPORT.md'
        if not report.exists():
            report.write_text('# without_PPO gradient-noise measurement\n\n**Blocked:** ' + str(error) +
                              '\n\nNo further jobs were admitted after this failure. Already-running jobs are left to finish. '
                              'See [execution record](runs/campaign/state.json) and [methods](methods.md).\n')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'controller', 'worker', 'status'))
    parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    if args.action == 'build':
        build()
    elif args.action == 'worker':
        if args.spec is None:
            raise ValueError('Worker requires a saved spec')
        if read(args.spec)['gpu_count'] == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        bounded_worker(args.spec)
    elif args.action == 'controller':
        controller()
    else:
        print(read(RUN / 'state.json', {}))


if __name__ == '__main__':
    main()
