"""Bounded EP study admissions and monitoring (Python 3.8).

Four seed-0 DPPO-skeleton runs share one 20 GPU-hour accounting boundary, including the live gate.
No automatic restarts, retries, source changes, or cancellation of other jobs.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import signal
import statistics
import subprocess
import time

from worker import atomic_json, append_json, json_check, read_json, source_files, utc, verify_source

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/epochs'
WORKTREE = PROJECT / '.codex-runtime/worktrees/stage3-epochs'
SOURCE = WORKTREE / 'results/stage3/epochs/dppo'
PYTHON = PROJECT / 'results/repro_halfcheetah/env/bin/python'
CAP = 20 * 3600
CONDITIONS = {
    'EP1M': (1, 5e-4, ['+train.actor_update_epochs=1', 'train.actor_lr=5e-4', 'train.actor_lr_scheduler.min_lr=5e-4']),
    'EP2M': (2, 2.5e-4, ['+train.actor_update_epochs=2', 'train.actor_lr=2.5e-4', 'train.actor_lr_scheduler.min_lr=2.5e-4']),
    'EP1': (1, 1e-4, ['+train.actor_update_epochs=1']),
    'EP2': (2, 1e-4, ['+train.actor_update_epochs=2']),
}
LAUNCH_ORDER = ('EP1M', 'EP2M', 'EP1', 'EP2')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
START = '<!-- EP runtime monitor -->'
END = '<!-- end EP runtime monitor -->'


def command(argv, readonly=False):
    error = None
    for attempt in range(2 if readonly else 1):
        try:
            run = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
            if run.returncode == 0:
                return run.stdout
            error = RuntimeError('%r exited %d: %s' % (argv, run.returncode, run.stderr[-2000:]))
        except subprocess.TimeoutExpired as caught:
            error = caught
        if readonly and attempt < 1:
            time.sleep(3)
    raise error


def env():
    cache = ROOT / 'cache'
    return dict(EP_SOURCE=str(SOURCE), PYTHONPATH=str(SOURCE),
                PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(cache),
                MPLCONFIGDIR=str(cache / 'matplotlib'), TORCH_HOME=str(cache / 'torch'),
                TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
                D4RL_DATASET_DIR=str(cache / 'd4rl'),
                LD_LIBRARY_PATH='/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia',
                DPPO_DATA_DIR=str(PROJECT / 'results/repro_halfcheetah/dppo/data'),
                DPPO_LOG_DIR=str(PROJECT / 'results/repro_halfcheetah/dppo/log'),
                OMP_NUM_THREADS=None, MKL_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None)


def source_snapshot(revision):
    """Bind runs to ordinary source contents and an explicit parent revision.

    A local source snapshot also captures approved edits when the parent index is
    read-only. Branch names are descriptive and may be main.
    """
    existing = read_json(ROOT / 'provenance/launch_source.json')
    if existing:
        if existing['revision'] != revision:
            raise ValueError('Launch provenance revision differs from requested revision')
        verify_source(existing)
        return existing
    destination = ROOT / 'runs/provenance/source_snapshot'
    provenance = dict(worktree=str(WORKTREE), revision=revision, source=str(SOURCE),
                      source_relative='results/stage3/epochs/dppo', snapshot=str(destination),
                      branch=command(['git', '-C', WORKTREE, 'branch', '--show-current'], True).strip(),
                      source_identity='ordinary-file snapshot',
                      allow_relevant_source_diff=True)
    if command(['git', '-C', WORKTREE, 'rev-parse', 'HEAD'], True).strip() != revision:
        raise ValueError('Requested parent revision differs from source worktree HEAD')
    if destination.exists():
        raise ValueError('An existing source snapshot requires its saved launch provenance')
    if any(p.name == '.git' for p in SOURCE.rglob('*')):
        raise ValueError('Scientific source must contain ordinary files, not nested Git metadata')
    paths = source_files(SOURCE)
    if not paths:
        raise ValueError('Scientific source is empty')
    for name, source in paths.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
    diff = command(['git', '-C', WORKTREE, 'diff', revision, '--', provenance['source_relative']], True)
    diff_path = ROOT / 'runs/provenance/source.diff'
    diff_path.write_text(diff)
    provenance['relevant_diff'] = str(diff_path)
    provenance['files_compared'] = len(paths)
    verify_source(provenance)
    atomic_json(ROOT / 'provenance/launch_source.json', provenance)
    return provenance


def native_job(condition):
    epochs, actor_lr, scientific_overrides = CONDITIONS[condition]
    name = condition + '_seed0'
    out = ROOT / name
    overrides = ['train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
                 'seed=0', 'logdir=' + str(out)] + scientific_overrides
    return dict(id=name, condition=condition, seed=0, mode='train',
                cap_seconds=14400, estimate_seconds=9000, gpu_count=1,
                cpus=40, memory_gib=64, cwd=str(SOURCE),
                command=[str(PYTHON), '-B', str(ROOT / 'code/entrypoint.py'),
                         '--config-dir=' + str(SOURCE / 'cfg/gym/finetune/halfcheetah-v2'),
                         '--config-name=ft_ppo_diffusion_mlp'] + overrides,
                env=env(), logdir=str(out), stdout=str(ROOT / 'runs/logs' / (name + '.out')),
                progress_file=str(out / 'stage3_progress.json'), depends_on=[], ready_files=[],
                completion_files=[str(out / 'result.pkl')] + [str(out / 'checkpoint' / ('state_%d.pt' % i))
                                                                 for i in (0, 35, 70, 105, 139)],
                completion_checks=[], overrides=overrides, actor_lr=actor_lr,
                actor_update_epochs=epochs)


def build(args):
    check = dict(path=str(Path(args.gate).resolve()), key='passed', equals=True)
    okay, detail = json_check(check)
    if not okay:
        raise ValueError('G1-G5 must pass before constructing a launch plan: ' + detail)
    approval = read_json(ROOT / 'provenance/approval.json')
    if not approval.get('approved') or approval['budget_gpu_hours'] != 20:
        raise ValueError('Approved budget missing')
    gate = read_json(args.gate)
    spent = float(gate['gpu_gate_seconds'])
    if spent > 3600 or float(gate['cpu_gate_seconds']) > 7200:
        raise ValueError('Gate allocation cap exceeded')
    provenance = source_snapshot(args.revision)
    jobs = [native_job(condition) for condition in LAUNCH_ORDER]
    plan = dict(campaign_id='epochs_training', phase='training', skeleton='dppo',
                run_root=str(ROOT / 'runs/campaign_training'), python=str(PYTHON),
                source_provenance=provenance, budget_gpu_hours=20, prior_gpu_seconds=spent,
                max_parallel=int(os.environ.get("MAX_PAR", "4")), monitor_seconds=1800, poll_seconds=30, controller_cap_seconds=259200,
                account='soal', partition='soal', global_checks=[check], jobs=jobs,
                automatic_retries=False, max_same_seed_retries_after_review=1)
    validate(plan)
    target = ROOT / 'runs/plan_training.json'
    if target.exists():
        raise ValueError('Refusing to overwrite an existing launch plan')
    atomic_json(target, plan)
    lines = ['# DPPO skeleton. Complete overrides for each seed; COMMON is not applied.']
    lines += ['%s 0 %s' % (j['condition'], ' '.join(CONDITIONS[j['condition']][2])) for j in jobs]
    (ROOT / 'jobs_epochs.txt').write_text('\n'.join(lines) + '\n')
    print(str(target))


def validate(plan):
    if plan['budget_gpu_hours'] != 20 or not 1 <= plan['max_parallel'] <= 4 or plan['monitor_seconds'] != 1800:
        raise ValueError('Approved cap is 20 GPU-hours, up to four workers, with 30-minute monitoring')
    if plan.get('skeleton') != 'dppo':
        raise ValueError('EP requires the DPPO skeleton')
    if [(j['condition'], j['seed']) for j in plan['jobs']] != [(condition, 0) for condition in LAUNCH_ORDER]:
        raise ValueError('Exactly four seed-0 conditions, matched-learning-rate first, are approved')
    for job in plan['jobs']:
        if (job['gpu_count'], job['cpus'], job['memory_gib'], job['cap_seconds']) != (1, 40, 64, 14400):
            raise ValueError('Scientific allocations require one A6000, 40 CPUs, 64GiB and a 4-hour cap')
        if job['overrides'] != native_job(job['condition'])['overrides']:
            raise ValueError('Run overrides changed from the approved DPPO configuration')
    if float(plan['prior_gpu_seconds']) + sum(j['cap_seconds'] for j in plan['jobs']) > CAP:
        raise ValueError('Cannot reserve the complete plan within 20 GPU-hours')
    return plan


def scheduler_rows(ids):
    if not ids:
        return {}
    output = command(['sacct', '-n', '-P', '--allocations', '-j', ','.join(ids),
                      '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%160'], True)
    rows = {}
    for line in output.splitlines():
        f = line.split('|')
        if len(f) >= 5 and f[0] in ids:
            tres = dict(x.split('=', 1) for x in f[4].split(',') if '=' in x)
            rows[f[0]] = dict(state=f[1].split()[0].rstrip('+'), exit_code=f[2],
                              elapsed_seconds=int(f[3] or 0), gpu_count=int(tres.get('gres/gpu', '0')))
    missing = [x for x in ids if x not in rows]
    if missing:
        output = command(['squeue', '-h', '-j', ','.join(missing), '-o', '%i|%T|%N|%S'], True)
        for line in output.splitlines():
            f = line.split('|')
            if len(f) >= 4 and f[0] in missing:
                rows[f[0]] = dict(state=f[1], exit_code=None, elapsed_seconds=0,
                                  gpu_count=1, node=f[2], predicted_start=f[3])
    return rows


def costs(plan, state):
    actual, reserved = float(plan['prior_gpu_seconds']), 0.0
    for j in plan['jobs']:
        row = state['jobs'].get(j['id'], {})
        if row.get('status') in ('complete', 'failed'):
            actual += row.get('elapsed_seconds', j['cap_seconds']) * max(1, row.get('gpu_count', 1))
        elif row.get('scheduler_job_id') or row.get('status') == 'submission_unknown':
            reserved += j['cap_seconds']
    return actual, reserved


def notify(state, root, kind, detail):
    record = dict(utc=utc(), kind=kind, detail=detail)
    append_json(root / 'events.jsonl', record)
    state.setdefault('notifications', []).append(record)
    print(json.dumps(record), flush=True)


def train_rows(job):
    try:
        with (Path(job['logdir']) / 'result.pkl').open('rb') as handle:
            return pickle.load(handle)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return []


def progress(job):
    if job['mode'] != 'train':
        try:
            return read_json(job['progress_file'], {})
        except json.JSONDecodeError:
            return dict(write_in_progress=True)
    rows = train_rows(job)
    if not rows:
        return {}
    last = rows[-1]
    value = {key: last[key] for key in ('itr', 'step', 'train_episode_reward', 'eval_episode_reward',
                                      'J_disc_train', 'kl_true_per_action', 'ppo_step_count', 'actor_optimizer_steps', 'critic_optimizer_steps', 'actor_lr', 'actor_adam_step_count') if key in last}
    value['rows'] = len(rows)
    times = [r['time'] for r in rows if 'train_episode_reward' in r and 'time' in r]
    if times:
        rate = statistics.median(times[-20:])
        eval_times = [r['time'] for r in rows if 'eval_episode_reward' in r and 'time' in r]
        erate = statistics.median(eval_times) if eval_times else 14
        remaining = sum(erate if i % 10 == 0 else rate for i in range(last['itr'] + 1, 140))
        value.update(seconds_per_training_iteration=rate, remaining_seconds_estimate=remaining,
                     estimated_finish_utc=(datetime.datetime.now(datetime.timezone.utc) +
                                           datetime.timedelta(seconds=remaining)).isoformat())
    return value


def submit(plan, job, state):
    root = Path(plan['run_root'])
    spec = dict(job, run_root=str(root), monitor_seconds=1800, source_provenance=plan['source_provenance'])
    path = root / 'workers' / job['id'] / 'spec.json'
    if path.exists():
        raise RuntimeError('Refusing to reuse existing worker specification')
    atomic_json(path, spec)
    (ROOT / 'runs/logs').mkdir(parents=True, exist_ok=True)
    argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue',
            '--time=%d' % math.ceil(job['cap_seconds'] / 60), '--job-name=epochs-' + job['id'],
            '--output=' + str(ROOT / 'runs/logs' / (job['id'] + '.scheduler.out')),
            '--error=' + str(ROOT / 'runs/logs' / (job['id'] + '.scheduler.err')),
            str(ROOT / 'code/run_worker.sh'), str(PYTHON), str(ROOT / 'code/worker.py'), str(path)]
    row = dict(status='submission_unknown', submitted_utc=utc(), argv=argv, spec=str(path),
               cap_seconds=job['cap_seconds'], gpu_count=1)
    state['jobs'][job['id']] = row
    atomic_json(root / 'campaign.json', state)
    append_json(root / 'commands.jsonl', dict(utc=utc(), action='submit', argv=argv))
    answer = command(argv).strip().split(';')[0]
    if not answer.isdigit():
        raise RuntimeError('Ambiguous scheduler submission; never resubmit automatically: ' + answer)
    row.update(status='pending', scheduler_job_id=answer)
    atomic_json(root / 'campaign.json', state)


def refresh(plan, state):
    active = {k: v for k, v in state['jobs'].items() if v.get('scheduler_job_id') and v['status'] not in ('complete', 'failed')}
    scheduler = scheduler_rows([v['scheduler_job_id'] for v in active.values()])
    lookup, failures = {j['id']: j for j in plan['jobs']}, []
    for name, row in active.items():
        info = scheduler.get(row['scheduler_job_id'])
        if not info:
            row['missing_checks'] = row.get('missing_checks', 0) + 1
            if row['missing_checks'] >= 3:
                failures.append('Scheduler cannot locate admitted job ' + name)
            continue
        row.update(scheduler=info, missing_checks=0, observed_utc=utc(), progress=progress(lookup[name]))
        manifest = read_json(Path(plan['run_root']) / 'workers' / name / 'manifest.json', {})
        row['hardware'] = manifest.get('hardware', {})
        for result_row in train_rows(lookup[name]):
            if result_row.get('ppo_kl_early_stop'):
                itr = int(result_row['itr'])
                seen = row.setdefault('reported_kl_early_stops', [])
                if itr not in seen:
                    notify(state, Path(plan['run_root']), 'kl_early_stop',
                           dict(run=name, itr=itr, actor_steps=result_row.get('actor_optimizer_steps'),
                                critic_steps=result_row.get('critic_optimizer_steps')))
                    seen.append(itr)
        if manifest.get('status') == 'failed':
            failures.append(name + ': ' + manifest.get('error', 'worker failed'))
        if info['state'] in TERMINAL:
            row.update(elapsed_seconds=info['elapsed_seconds'], gpu_count=max(1, info['gpu_count']), ended_utc=utc())
            okay = info['state'] == 'COMPLETED' and info['exit_code'] == '0:0' and manifest.get('status') == 'complete'
            row['status'] = 'complete' if okay else 'failed'
            if okay:
                notify(state, Path(plan['run_root']), 'job_complete', name)
            else:
                failures.append('%s: %s/%s; worker=%s' % (name, info['state'], info['exit_code'], manifest.get('status')))
        else:
            row['status'] = 'running' if info['state'] == 'RUNNING' else 'pending'
    return failures


def update_status(plan, state, document=False):
    actual, reserved = costs(plan, state)
    state.update(updated_utc=utc(), actual_gpu_hours=actual / 3600, reserved_gpu_hours=reserved / 3600,
                 remaining_unreserved_gpu_hours=(CAP - actual - reserved) / 3600)
    atomic_json(Path(plan['run_root']) / 'campaign.json', state)
    atomic_json(ROOT / 'runs/status.json', state)
    if not document or not (ROOT / 'experiment.md').exists():
        return
    path = ROOT / 'experiment.md'
    text = path.read_text()
    count = sum(r['status'] == 'complete' for r in state['jobs'].values())
    active = [k for k, v in state['jobs'].items() if v['status'] in ('pending', 'running', 'submission_unknown')]
    block = '\n'.join([START, '**Execution status (%s):** %s; %d/%d jobs verified complete in %s.' %
                       (state['updated_utc'], state['status'], count, len(plan['jobs']), plan['phase']), '',
                       'GPU time: %.3f hours used; %.3f hours reserved; 20-hour combined cap. Active or queued: %s.' %
                       (actual / 3600, reserved / 3600, ', '.join(active) or 'none'), '',
                       '[Live status](runs/status.json) · [Execution record](runs/campaign_%s/campaign.json).' % plan['phase'], END])
    if START not in text and END not in text:
        text = text.rstrip() + '\n\n' + block + '\n'
    elif text.count(START) == text.count(END) == 1 and text.index(START) < text.index(END):
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        raise ValueError('Malformed runtime monitor markers')
    temporary = path.with_name(path.name + '.monitor.tmp.%d' % os.getpid())
    temporary.write_text(text)
    os.replace(str(temporary), str(path))


def interim(plan, state):
    if plan['phase'] != 'training' or state.get('itr70_note_ready'):
        return
    jobs = [j for j in plan['jobs'] if j['mode'] == 'train' and j['seed'] == 0]
    if not all(any(r['itr'] == 70 for r in train_rows(j)) for j in jobs):
        return
    state['itr70_note_ready'] = True
    notify(state, Path(plan['run_root']), 'interim_ready', 'Seed 0 reached evaluation iteration 70')
    target = ROOT / 'code/analyze.py'
    if target.exists():
        argv = [str(PYTHON), '-B', str(target), '--output', str(ROOT / 'runs/campaign_training/candidates/itr70'), '--interim']
        output = ROOT / 'runs/logs/interim_analysis.out'
        analysis_env = os.environ.copy()
        analysis_env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'))
        with output.open('x') as handle:
            result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=analysis_env, timeout=600)
        state['interim_analysis'] = dict(command=argv, returncode=result.returncode, output=str(output), ended_utc=utc())
        notify(state, Path(plan['run_root']), 'interim_analysis_ready' if result.returncode == 0 else 'interim_analysis_failed', str(output))


def final_analysis(plan, state):
    target = ROOT / 'code/analyze.py'
    if not target.exists():
        state['analysis_status'] = 'pending_entrypoint'
        return
    argv = [str(PYTHON), '-B', str(target), '--output', str(ROOT / 'runs/campaign_training/candidates/final'), '--publish']
    path = ROOT / 'runs/logs/final_analysis.out'
    analysis_env = os.environ.copy()
    analysis_env.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'))
    with path.open('x') as output:
        result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, env=analysis_env, timeout=600)
    state['final_analysis'] = dict(command=argv, returncode=result.returncode, output=str(path), ended_utc=utc())
    state['analysis_status'] = 'complete' if result.returncode == 0 else 'failed'


def run(path):
    plan = validate(read_json(path))
    root = Path(plan['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    # One study-wide controller lock prevents simultaneous phases double-spending.
    lock = (ROOT / 'runs/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'campaign.json').exists():
        raise RuntimeError('Existing campaign requires explicit reviewed continuation; no automatic restart')
    for check in plan['global_checks']:
        okay, detail = json_check(check)
        if not okay:
            raise RuntimeError('Launch gate not passed: ' + detail)
    verify_source(plan['source_provenance'])
    state = dict(campaign_id=plan['campaign_id'], status='running', jobs={}, started_utc=utc(),
                 monitor_status='active', controller_job_id=os.environ.get('SLURM_JOB_ID'),
                 source_provenance=plan['source_provenance'], plan_path=str(Path(path).resolve()))
    atomic_json(root / 'plan_snapshot.json', plan)
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    started, next_monitor = time.monotonic(), 0.0
    try:
        while True:
            if interrupted or time.monotonic() - started > plan['controller_cap_seconds'] - 60:
                raise RuntimeError('Controller stopped; independent worker allocations retain their own caps')
            scheduler_ok = True
            try:
                failures = refresh(plan, state)
                if state.pop('scheduler_errors', 0):
                    notify(state, root, 'scheduler_recovered', 'Read-only scheduler queries recovered')
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                scheduler_ok, failures = False, []
                state['scheduler_errors'] = state.get('scheduler_errors', 0) + 1
                state['scheduler_error'] = str(error)
                if state['scheduler_errors'] == 1:
                    notify(state, root, 'scheduler_unavailable',
                           'Admissions paused; independent workers retain their caps: ' + str(error))
            if failures:
                if state['status'] != 'draining_after_failure':
                    notify(state, root, 'failure', failures)
                state.update(status='draining_after_failure', failures=failures)
            active = [r for r in state['jobs'].values() if r['status'] in ('running', 'pending', 'submission_unknown')]
            if state['status'] == 'running' and scheduler_ok:
                for job in plan['jobs']:
                    if len(active) >= plan['max_parallel']:
                        break
                    if job['id'] in state['jobs']:
                        continue
                    if any(state['jobs'].get(dep, {}).get('status') != 'complete' for dep in job['depends_on']):
                        continue
                    if not all(Path(p).is_file() for p in job['ready_files']):
                        continue
                    actual, reserved = costs(plan, state)
                    if actual + reserved + job['cap_seconds'] > CAP:
                        if not active:
                            raise RuntimeError('The remaining budget cannot reserve another full job')
                        continue
                    submit(plan, job, state)
                    active.append(state['jobs'][job['id']])
            interim(plan, state)
            due = time.monotonic() >= next_monitor
            if due:
                for name, row in state['jobs'].items():
                    if row['status'] != 'running':
                        continue
                    p = row.get('progress', {})
                    counter = {k: p[k] for k in ('itr', 'phase', 'completed', 'done', 'substage') if k in p}
                    if counter:
                        row['unchanged_checks'] = row.get('unchanged_checks', 0) + 1 if counter == row.get('last_counter') else 0
                        row['last_counter'] = counter
                        if row['unchanged_checks'] == 2:
                            notify(state, root, 'suspected_stall', name + ': no progress over two 30-minute checks')
                append_json(root / 'monitor_history.jsonl', dict(utc=utc(), status=state['status'],
                            jobs={k: dict(status=v['status'], progress=v.get('progress')) for k, v in state['jobs'].items()}))
                next_monitor = time.monotonic() + 1800
            if len(state['jobs']) == len(plan['jobs']) and all(r['status'] == 'complete' for r in state['jobs'].values()):
                state.update(status='complete', monitor_status='paused', ended_utc=utc())
                if plan['phase'] == 'training':
                    final_analysis(plan, state)
                update_status(plan, state, True)
                atomic_json(ROOT / 'runs' / ('completion_' + plan['phase'] + '.json'),
                            dict(verified=True, jobs=len(plan['jobs']), ended_utc=state['ended_utc'],
                                 actual_gpu_hours=state['actual_gpu_hours']))
                notify(state, root, 'campaign_complete', 'All admitted outputs verified; monitor paused')
                update_status(plan, state, True)
                return 0
            update_status(plan, state, due or bool(failures))
            if state['status'] == 'draining_after_failure' and not active:
                raise RuntimeError('Admissions stopped after failure; same-seed retry requires reviewed fix')
            time.sleep(plan['poll_seconds'])
    except BaseException as error:
        state.update(status='blocked', error='%s: %s' % (type(error).__name__, error), monitor_status='paused', ended_utc=utc())
        notify(state, root, 'controller_blocked', state['error'])
        update_status(plan, state, True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    item = sub.add_parser('build-training')
    item.add_argument('--revision', required=True)
    item.add_argument('--gate', required=True)
    for name in ('run', 'validate', 'status'):
        item = sub.add_parser(name)
        item.add_argument('plan')
    args = parser.parse_args()
    if args.action == 'build-training':
        build(args)
    elif args.action == 'run':
        run(args.plan)
    elif args.action == 'validate':
        print(json.dumps(dict(valid=True, jobs=len(validate(read_json(args.plan))['jobs']))))
    else:
        plan = validate(read_json(args.plan))
        print(json.dumps(read_json(Path(plan['run_root']) / 'campaign.json', {}), indent=2))
