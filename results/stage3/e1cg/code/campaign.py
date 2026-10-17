"""Approved E1CG admissions and monitor. Python 3.8; no scientific auto-retries.

Only this study's jobs are admitted. Stopping the controller never cancels a
worker allocation; every worker has its own hard limit and cleanup supervisor.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

from worker import atomic_json, append_json, json_check, read_json, utc, verify_source

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/e1cg'
WORKTREE = PROJECT / '.codex-runtime/worktrees/stage3-e1cg'
SOURCE = WORKTREE / 'results/stage3/e1cg/dppo'
PYTHON = PROJECT / 'results/repro_halfcheetah/env/bin/python'
E1C = PROJECT / 'results/stage3/e1c'
COMMON = ['model.clip_ploss_coef=1e6', 'model.clip_ploss_coef_base=1e6',
          'train.target_kl=null', '+model.clamp_logprob=false',
          'model.randn_clip_value=100', '+train.actor_single_step=true',
          '+model.actor_loss=score', '+train.stage3_diag=true']
CONDITIONS = {
    'E1CG': ['+train.adv_estimator=mc_critic', '+train.baseline_diag=true',
             '+train.decision_discount=true', '+train.decision_discount_post_norm=true'],
    'E2C': ['+train.adv_estimator=mc_critic', '+train.baseline_diag=true',
            '+train.decision_discount=true', 'model.gamma_denoising=1.0',
            '+model.logprob_reduce=sum', '+model.norm_adv=false'],
    'E1C': ['+train.adv_estimator=mc_critic', '+train.baseline_diag=true'],
}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
START_MARKER = '<!-- E1CG runtime monitor -->'
END_MARKER = '<!-- end E1CG runtime monitor -->'


def command(argv, readonly=False):
    error = None
    for attempt in range(3 if readonly else 1):
        try:
            result = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
            error = RuntimeError('%r exited %d: %s' % (argv, result.returncode, result.stderr[-2000:]))
        except subprocess.TimeoutExpired as caught:
            error = caught
        if readonly and attempt < 2:
            time.sleep((2, 5)[attempt])
    raise error


def source_paths(directory):
    return {str(p.relative_to(directory)): p for p in Path(directory).rglob('*')
            if p.is_file() and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(directory).parts)
            and p.suffix != '.pyc'}


def environment():
    cache = ROOT / 'cache'
    return dict(E1CG_SOURCE=str(SOURCE), PYTHONPATH=str(SOURCE),
                PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(cache),
                MPLCONFIGDIR=str(cache / 'matplotlib'), TORCH_HOME=str(cache / 'torch'),
                TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
                D4RL_DATASET_DIR=str(cache / 'd4rl'),
                LD_LIBRARY_PATH='/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia',
                DPPO_DATA_DIR=str(PROJECT / 'results/repro_halfcheetah/dppo/data'),
                DPPO_LOG_DIR=str(PROJECT / 'results/repro_halfcheetah/dppo/log'),
                OMP_NUM_THREADS=None, MKL_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None)


def native_job(name, condition, seed, mode, checkpoint=None, repeats=2, samplers='train'):
    out = ROOT / (name if mode == 'train' else 'evaluations/' + name)
    extra = list(CONDITIONS[condition]) if condition in CONDITIONS else []
    if mode != 'train':
        extra += ['+train.mode=ckpt_eval', '+train.ckpt_segments=true',
                  '+train.ckpt_eval_repeats=%d' % repeats, '+train.ckpt_samplers=' + samplers]
    if checkpoint is not None:
        extra.append('base_policy_path=' + str(checkpoint))
    overrides = ['train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
                 'seed=%d' % seed, 'logdir=' + str(out)] + COMMON + extra
    argv = [str(PYTHON), '-B', str(ROOT / 'code/entrypoint.py'),
            '--config-dir=' + str(SOURCE / 'cfg/gym/finetune/halfcheetah-v2'),
            '--config-name=ft_ppo_diffusion_mlp'] + overrides
    job = dict(id=name, condition=condition, seed=seed, mode=mode,
               cap_seconds=14400 if mode == 'train' else 600 if repeats == 6 else 300,
               estimate_seconds=8800 if mode == 'train' else 110 if repeats == 6 else 80,
               gpu_count=1, cpus=40, memory_gib=64, cwd=str(SOURCE), command=argv,
               env=environment(), logdir=str(out), stdout=str(ROOT / 'runs/logs' / (name + '.out')),
               progress_file=str(out / 'stage3_progress.json'),
               completion_files=[str(out / 'result.pkl'), str(out / 'checkpoint/state_139.pt')]
                   if mode == 'train' else [str(out / 'ckpt_eval.json'), str(out / 'ckpt_eval_rollouts.json')],
               depends_on=[], ready_files=[], completion_checks=[],
               overrides=overrides, optional=condition == 'E2C')
    if mode != 'train':
        selected = ['train', 'eval'] if samplers == 'both' else [samplers]
        job.update(repeats=repeats, samplers=selected)
        job['completion_checks'] = [dict(path=str(out / 'ckpt_eval.json'), key='status', equals='completed'),
                                     dict(path=str(out / 'ckpt_eval.json'), key='repeats_per_sampler', equals=repeats)]
        for sampler in selected:
            job['completion_checks'].append(dict(path=str(out / 'ckpt_eval.json'),
                         key='samplers.%s.J_disc.n' % sampler, equals=80 * repeats))
    return job


def build(args):
    snapshot = ROOT / 'runs/provenance/source_snapshot'
    revision = command(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], readonly=True).strip()
    if revision != args.revision:
        raise ValueError('Requested source revision does not match the worktree HEAD')
    if snapshot.exists():
        raise ValueError('Source snapshot already exists; do not silently replace provenance')
    paths = source_paths(SOURCE)
    prefix = 'results/stage3/e1cg/dppo/'
    tracked = {line[len(prefix):] for line in command(['git', '-C', str(WORKTREE), 'ls-files', '--', prefix], readonly=True).splitlines()
               if line.startswith(prefix)}
    if not set(paths).issubset(tracked):
        raise ValueError('Commit every scientific source file before building the plan')
    dirty = command(['git', '-C', str(WORKTREE), 'status', '--porcelain', '--untracked-files=no', '--', prefix], readonly=True)
    if dirty.strip():
        raise ValueError('Scientific source has uncommitted changes')
    for relative, path in paths.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(target))
    provenance = dict(worktree=str(WORKTREE), revision=revision, source=str(SOURCE),
                      source_relative='results/stage3/e1cg/dppo', snapshot=str(snapshot),
                      branch=command(['git', '-C', str(WORKTREE), 'branch', '--show-current'], readonly=True).strip())
    verify_source(provenance)
    primary = [native_job('E1CG_seed%d' % seed, 'E1CG', seed, 'train') for seed in range(3)]
    optional = [native_job('E2C_seed%d' % seed, 'E2C', seed, 'train') for seed in range(3)]
    evaluations = []
    for condition in ('E1C', 'E1CG'):
        for seed in range(3):
            run = (E1C if condition == 'E1C' else ROOT) / ('%s_seed%d' % (condition, seed))
            for itr in (35, 70, 105, 139):
                cp = run / 'checkpoint' / ('state_%d.pt' % itr)
                job = native_job('segments_%s_s%d_i%d' % (condition, seed, itr), condition, 5000,
                                 'ckpt_eval', cp, samplers='both' if itr == 139 else 'train')
                job.update(source_seed=seed, checkpoint_itr=itr,
                           depends_on=[p['id'] for p in primary], ready_files=[str(cp)])
                evaluations.append(job)
    theta = native_job('segments_theta0', 'theta0', 5000, 'ckpt_eval', repeats=6)
    theta['depends_on'] = [p['id'] for p in primary]
    evaluations.append(theta)
    for seed in range(3):
        cp = ROOT / ('E2C_seed%d/checkpoint/state_139.pt' % seed)
        job = native_job('final_E2C_seed%d' % seed, 'E2C', 5000, 'ckpt_eval', cp, samplers='both')
        job.update(source_seed=seed, checkpoint_itr=139,
                   depends_on=['E2C_seed%d' % seed], ready_files=[str(cp)])
        evaluations.append(job)
    plan = dict(campaign_id='e1cg', run_root=str(ROOT / 'runs/campaign'),
                python=str(PYTHON), source_provenance=provenance,
                budget_gpu_hours=32, max_parallel=9, monitor_seconds=1800, poll_seconds=30,
                controller_cap_seconds=172800, account='soal', partition='soal',
                optional_enabled=True,
                global_checks=[dict(path=str(Path(args.gate).resolve()), key=args.gate_key, equals=True),
                               dict(path=str(ROOT / 'runs/gates/launch_greenlight.json'), key='approved', equals=True)],
                optional_checks=[dict(path=str(Path(args.gate).resolve()), key=args.optional_gate_key, equals=True)],
                jobs=primary + optional + evaluations)
    validate(plan)
    target = ROOT / 'runs/plan.json'
    if target.exists():
        raise ValueError('Plan already exists')
    atomic_json(target, plan)
    print(str(target))


def validate(plan):
    if float(plan['budget_gpu_hours']) != 32 or int(plan['max_parallel']) > 9:
        raise ValueError('The approved cap is 32 GPU-hours, at most nine simultaneous workers')
    if int(plan['monitor_seconds']) != 1800:
        raise ValueError('Monitor interval must remain 30 minutes')
    ids = [j['id'] for j in plan['jobs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job ID')
    for job in plan['jobs']:
        if int(job['gpu_count']) != 1 or int(job['cpus']) != 40 or int(job['memory_gib']) != 64:
            raise ValueError('Each scientific job uses one A6000 / 40 CPUs / 64 GiB')
        cap = 14400 if job['mode'] == 'train' else 600 if job['repeats'] == 6 else 300
        if not 0 < int(job['cap_seconds']) <= cap or not set(job['depends_on']).issubset(ids):
            raise ValueError('Invalid cap/dependency')
        if not all(isinstance(x, str) for x in job['command']):
            raise ValueError('Commands must be argument arrays')
    primary = [j for j in plan['jobs'] if j['condition'] == 'E1CG' and j['mode'] == 'train']
    optional = [j for j in plan['jobs'] if j['condition'] == 'E2C' and j['mode'] == 'train']
    if [j['seed'] for j in primary] != [0, 1, 2] or [j['seed'] for j in optional] != [0, 1, 2]:
        raise ValueError('Exactly three planned seeds per condition')
    # No uncontrolled new collection or duplicated intermediate eval sampler.
    required = [j for j in plan['jobs'] if j['mode'] != 'train' and not j['optional']]
    train_rollouts = sum(j['repeats'] for j in required if 'train' in j['samplers'])
    eval_rollouts = sum(j['repeats'] for j in required if 'eval' in j['samplers'])
    if len(required) != 25 or (train_rollouts, eval_rollouts) != (54, 12):
        raise ValueError('Part C must contain exactly 54 train and 12 final-eval rollouts')
    return plan


def scheduler_rows(ids):
    if not ids:
        return {}
    out = command(['sacct', '-n', '-P', '--allocations', '-j', ','.join(ids),
                   '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%160'], readonly=True)
    result = {}
    for line in out.splitlines():
        fields = line.split('|')
        if len(fields) >= 5 and fields[0] in ids:
            tres = dict(t.split('=', 1) for t in fields[4].split(',') if '=' in t)
            result[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'), exit_code=fields[2],
                                    elapsed_seconds=int(fields[3] or 0), gpu_count=int(tres.get('gres/gpu', '0')),
                                    allocated_tres=fields[4])
    missing = [x for x in ids if x not in result]
    if missing:
        out = command(['squeue', '-h', '-j', ','.join(missing), '-o', '%i|%T|%N|%S'], readonly=True)
        for line in out.splitlines():
            fields = line.split('|')
            if len(fields) == 4 and fields[0] in missing:
                result[fields[0]] = dict(state=fields[1], exit_code=None, elapsed_seconds=0,
                                        gpu_count=1, nodes=fields[2], predicted_start=fields[3])
    return result


def free_a6000():
    """Read only; do not allocate GPUs or infer availability from nvidia-smi."""
    out = command(['scontrol', 'show', 'nodes', 'soal-10,soal-11', '-o'], readonly=True)
    total, nodes = 0, []
    for line in out.splitlines():
        values = dict(x.split('=', 1) for x in line.split() if '=' in x)
        cfg = dict(x.split('=', 1) for x in values.get('CfgTRES', '').split(',') if '=' in x)
        alloc = dict(x.split('=', 1) for x in values.get('AllocTRES', '').split(',') if '=' in x)
        available = max(0, int(cfg.get('gres/gpu:a6000', 0)) - int(alloc.get('gres/gpu:a6000', 0)))
        if any(token in values.get('State', '') for token in ('DOWN', 'DRAIN', 'FAIL', 'NOT_RESPONDING')):
            available = 0
        # A GPU slot must also have the approved CPUs and memory.
        available = min(available, max(0, int(values.get('CPUEfctv', 0)) - int(values.get('CPUAlloc', 0))) // 40,
                        max(0, int(values.get('RealMemory', 0)) - int(values.get('AllocMem', 0))) // (64 * 1024))
        nodes.append(dict(node=values.get('NodeName'), free_a6000_slots=available))
        total += available
    return dict(utc=utc(), free_a6000_slots=total, nodes=nodes)


def costs(plan, state):
    actual, reserved = 0.0, 0.0
    for job in plan['jobs']:
        row = state['jobs'].get(job['id'], {})
        if row.get('status') in ('complete', 'failed'):
            actual += float(row.get('elapsed_seconds', job['cap_seconds'])) * max(1, int(row.get('gpu_count', 1)))
        elif row.get('scheduler_job_id') or row.get('status') == 'submission_unknown':
            reserved += job['cap_seconds']
    return actual, reserved


def ready(job, state):
    for dep in job['depends_on']:
        if state['jobs'].get(dep, {}).get('status') != 'complete':
            return False
    return all(Path(p).is_file() for p in job.get('ready_files', []))


def submit(plan, job, state):
    root = Path(plan['run_root'])
    spec = dict(job, run_root=str(root), monitor_seconds=1800,
                source_provenance=plan['source_provenance'],
                startup_checks=plan['global_checks'] + (plan['optional_checks'] if job['optional'] else []))
    spec_path = root / 'workers' / job['id'] / 'spec.json'
    if spec_path.exists():
        raise RuntimeError('No automatic retry or overwrite: ' + job['id'])
    atomic_json(spec_path, spec)
    logs = ROOT / 'runs/logs'
    logs.mkdir(parents=True, exist_ok=True)
    argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue',
            '--time=%d' % int(math.ceil(job['cap_seconds'] / 60.0)),
            '--job-name=e1cg-' + job['id'], '--output=' + str(logs / (job['id'] + '.scheduler.out')),
            '--error=' + str(logs / (job['id'] + '.scheduler.err')),
            str(ROOT / 'code/run_worker.sh'), plan['python'], str(ROOT / 'code/worker.py'), str(spec_path)]
    state['jobs'][job['id']] = dict(status='submission_unknown', submitted_utc=utc(), argv=argv,
                                  cap_seconds=job['cap_seconds'], gpu_count=1)
    atomic_json(root / 'campaign.json', state)
    append_json(root / 'commands.jsonl', dict(utc=utc(), action='submit', job=job['id'], argv=argv))
    # A timeout is ambiguous, so never repeat this mutating call automatically.
    response = command(argv).strip()
    sid = response.split(';')[0]
    if not sid.isdigit():
        raise RuntimeError('Unrecognized sbatch response: ' + response)
    state['jobs'][job['id']].update(status='pending', scheduler_job_id=sid, spec=str(spec_path))
    atomic_json(root / 'campaign.json', state)


def training_rows(job):
    path = Path(job['logdir']) / 'result.pkl'
    if not path.is_file():
        return []
    try:
        with path.open('rb') as handle:
            return pickle.load(handle)
    except (EOFError, pickle.UnpicklingError):
        return []


def progress(job):
    if job['mode'] != 'train':
        try:
            return read_json(job['progress_file'], {})
        except json.JSONDecodeError:
            return dict(write_in_progress=True)
    rows = training_rows(job)
    if not rows:
        return {}
    last = rows[-1]
    times = [float(r['time']) for r in rows if 'train_episode_reward' in r and 'time' in r]
    eval_times = [float(r['time']) for r in rows if 'eval_episode_reward' in r and 'time' in r]
    value = {key: last[key] for key in ('itr', 'step', 'train_episode_reward', 'eval_episode_reward',
                                     'J_disc_train', 'J_disc_eval', 'kl_true_per_action') if key in last}
    value['rows'] = len(rows)
    if times:
        rate = statistics.median(times[-20:])
        erate = statistics.median(eval_times) if eval_times else 14.0
        remaining = sum(erate if i % 10 == 0 else rate for i in range(int(last['itr']) + 1, 140))
        value.update(seconds_per_training_iteration=rate, remaining_seconds_estimate=remaining,
                     estimated_finish_utc=(datetime.datetime.now(datetime.timezone.utc) +
                                           datetime.timedelta(seconds=remaining)).isoformat())
    return value


def notify(state, root, kind, detail):
    record = dict(utc=utc(), kind=kind, detail=detail)
    append_json(root / 'events.jsonl', record)
    state.setdefault('notifications', []).append(record)
    print(json.dumps(record, sort_keys=True), flush=True)


def interim_comparison(plan, root, state):
    if state.get('itr70_note_ready'):
        return
    jobs = [j for j in plan['jobs'] if j['condition'] == 'E1CG' and j['mode'] == 'train']
    observed = []
    for job in jobs:
        rows = training_rows(job)
        if not any(int(r['itr']) == 70 for r in rows):
            return
        with (E1C / ('E1C_seed%d/result.pkl' % job['seed'])).open('rb') as handle:
            reference = pickle.load(handle)
        pair = dict(seed=job['seed'])
        for label, data in (('without_PPO_and_GAE + gamma^k', rows), ('without_PPO_and_GAE', reference)):
            evaluation = next(r for r in data if int(r['itr']) == 70)
            training = next(r for r in data if int(r['itr']) == 69)
            pair[label] = dict(eval_itr70=evaluation['eval_episode_reward'],
                               train_itr69=training['train_episode_reward'],
                               J_disc_train_itr69=training['J_disc_train'])
        observed.append(pair)
    record = dict(utc=utc(), status='ready', comparison='eval at 70; most recent training rollout at 69',
                  per_seed=observed)
    atomic_json(ROOT / 'runs/itr70_comparison.json', record)
    state['itr70_note_ready'] = True
    notify(state, root, 'interim_ready', str(ROOT / 'runs/itr70_comparison.json'))


def refresh(plan, root, state):
    active = {name: value for name, value in state['jobs'].items()
              if value.get('scheduler_job_id') and value.get('status') not in ('complete', 'failed')}
    sched = scheduler_rows([value['scheduler_job_id'] for value in active.values()])
    failures = []
    lookup = {j['id']: j for j in plan['jobs']}
    for name, record in active.items():
        sid = record['scheduler_job_id']
        if sid not in sched:
            record['missing_scheduler_checks'] = record.get('missing_scheduler_checks', 0) + 1
            if record['missing_scheduler_checks'] >= 3:
                failures.append('Scheduler cannot locate admitted job ' + name)
            continue
        row = sched[sid]
        record.update(scheduler=row, observed_utc=utc(), missing_scheduler_checks=0)
        manifest = read_json(root / 'workers' / name / 'manifest.json', {})
        record['progress'] = progress(lookup[name])
        record['hardware'] = manifest.get('hardware', {})
        if row['state'] == 'RUNNING':
            record['ever_running'] = True
        if manifest.get('status') == 'failed':
            failures.append(name + ': ' + manifest.get('error', 'worker failed'))
        if row['state'] in TERMINAL:
            record.update(elapsed_seconds=row['elapsed_seconds'], gpu_count=max(1, row['gpu_count']),
                          ended_utc=utc())
            if row['state'] == 'COMPLETED' and row['exit_code'] == '0:0' and manifest.get('status') == 'complete':
                record['status'] = 'complete'
                notify(state, root, 'job_complete', name)
            else:
                record['status'] = 'failed'
                failures.append('%s: %s/%s, worker=%s' % (name, row['state'], row['exit_code'], manifest.get('status')))
        else:
            record['status'] = 'running' if row['state'] == 'RUNNING' else 'pending'
    return failures


def update_status(plan, root, state, document=False):
    actual, reserved = costs(plan, state)
    state.update(updated_utc=utc(), actual_gpu_hours=actual / 3600, reserved_gpu_hours=reserved / 3600,
                 remaining_unreserved_gpu_hours=32 - (actual + reserved) / 3600)
    atomic_json(root / 'campaign.json', state)
    simple = {key: state.get(key) for key in ('status', 'updated_utc', 'actual_gpu_hours', 'reserved_gpu_hours',
              'remaining_unreserved_gpu_hours', 'monitor_status', 'optional_decision', 'error', 'itr70_note_ready')}
    simple['jobs'] = {name: {key: record.get(key) for key in ('status', 'scheduler_job_id', 'progress', 'hardware')}
                      for name, record in state['jobs'].items()}
    atomic_json(ROOT / 'runs/status.json', simple)
    if not document:
        return
    path = ROOT / 'experiment.md'
    if not path.exists():
        return
    text = path.read_text()
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError('Expected exactly one E1CG runtime monitor marker pair')
    complete = sum(r.get('status') == 'complete' for r in state['jobs'].values())
    active = [name for name, r in state['jobs'].items() if r.get('status') in ('pending', 'running')]
    block = [START_MARKER, '**Execution status (%s):** %s. %d jobs verified complete.' %
             (state['updated_utc'], state['status'].replace('_', ' '), complete), '',
             'GPU time: %.3f hours completed; %.3f hours reserved; 32-hour cap. Active or queued: %s.' %
             (actual / 3600, reserved / 3600, ', '.join(active) or 'none'), '',
             'Next: ' + ('review completed measurements and generate the approved final analysis.'
                        if state['status'] == 'complete' else 'continue only the approved gated jobs; no automatic retries.'), '',
             '[Live status](runs/status.json) · [Execution record](runs/campaign/campaign.json).', END_MARKER]
    first, last = text.index(START_MARKER), text.index(END_MARKER)
    if last < first:
        raise ValueError('Reversed monitor markers')
    replacement = text[:first] + '\n'.join(block) + text[last + len(END_MARKER):]
    temporary = path.with_name(path.name + '.monitor.tmp.%d' % os.getpid())
    temporary.write_text(replacement)
    os.replace(str(temporary), str(path))


def run_analysis(plan, root, state, final=False):
    phase = 'final' if final else 'interim'
    target = ROOT / 'code/analyze.py'
    if not target.is_file():
        raise RuntimeError('Approved analysis entrypoint is not present: ' + str(target))
    argv = [plan['python'], '-B', str(target), '--root', str(ROOT)]
    if final:
        argv.append('--final')
    record = dict(status='running', command=argv, started_utc=utc(), cap_seconds=600)
    state[phase + '_analysis'] = record
    update_status(plan, root, state)
    append_json(root / 'commands.jsonl', dict(utc=utc(), action=phase + '_analysis', argv=argv))
    output = ROOT / 'runs/logs' / (phase + '_analysis.out')
    output.parent.mkdir(parents=True, exist_ok=True)
    analysis_env = os.environ.copy()
    analysis_env['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
    analysis_env['PYTHONDONTWRITEBYTECODE'] = '1'
    Path(analysis_env['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, timeout=600, env=analysis_env)
    record.update(ended_utc=utc(), returncode=result.returncode,
                  status='complete' if result.returncode == 0 else 'failed', stdout=str(output))
    if result.returncode:
        raise RuntimeError(phase + ' analysis failed; scientific jobs were not changed')
    notify(state, root, phase + '_analysis_ready', str(output))


def run(plan_path):
    plan = validate(read_json(plan_path))
    root = Path(plan['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'campaign.json').exists():
        raise RuntimeError('Existing campaign: reviewed continuation is required; no automatic resume')
    for check in plan['global_checks']:
        okay, detail = json_check(check)
        if not okay:
            raise RuntimeError('Launch gate not satisfied: ' + detail)
    verify_source(plan['source_provenance'])
    state = dict(campaign_id='e1cg', status='running', jobs={}, started_utc=utc(),
                 monitor_status='active', controller_job_id=os.environ.get('SLURM_JOB_ID'),
                 optional_decision='waiting_for_primary_running_and_three_free_slots',
                 plan_path=str(Path(plan_path).resolve()), source_provenance=plan['source_provenance'])
    atomic_json(root / 'plan_snapshot.json', plan)
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    started, next_monitor = time.monotonic(), 0.0
    primary_ids = ['E1CG_seed%d' % i for i in range(3)]
    optional_ids = ['E2C_seed%d' % i for i in range(3)]
    try:
        while True:
            if interrupted or time.monotonic() - started > plan['controller_cap_seconds'] - 60:
                raise RuntimeError('Controller stopped; worker allocations retain independent caps')
            failures = refresh(plan, root, state)
            if failures:
                if state['status'] != 'draining_after_failure':
                    notify(state, root, 'failure', failures)
                state.update(status='draining_after_failure', failures=failures)
            primary = [state['jobs'].get(name, {}) for name in primary_ids]
            if state['status'] == 'running' and state['optional_decision'].startswith('waiting'):
                if all(r.get('status') == 'running' for r in primary):
                    capacity = free_a6000()
                    state['last_optional_capacity_check'] = capacity
                    if capacity['free_a6000_slots'] >= 3:
                        for check in plan['optional_checks']:
                            okay, detail = json_check(check)
                            if not okay:
                                raise RuntimeError('Optional-condition gate failed: ' + detail)
                        state['optional_decision'] = 'admit_three_seeds'
                        notify(state, root, 'optional_admitted', capacity)
                elif any(r.get('status') == 'complete' for r in primary):
                    state['optional_decision'] = 'skipped_no_three_extra_slots_while_all_primary_running'
                    for job in plan['jobs']:
                        if job['optional']:
                            state['jobs'][job['id']] = dict(status='skipped', reason=state['optional_decision'])
                    notify(state, root, 'optional_skipped', state['optional_decision'])
            active = [r for r in state['jobs'].values() if r.get('status') in ('running', 'pending', 'submission_unknown')]
            if state['status'] == 'running':
                for job in plan['jobs']:
                    if len(active) >= plan['max_parallel']:
                        break
                    if job['id'] in state['jobs'] or not ready(job, state):
                        continue
                    if job['optional'] and state['optional_decision'] != 'admit_three_seeds':
                        continue
                    actual, reserved = costs(plan, state)
                    if actual + reserved + job['cap_seconds'] > 32 * 3600:
                        if not active:
                            raise RuntimeError('Remaining global budget cannot reserve the next complete job')
                        continue
                    submit(plan, job, state)
                    active.append(state['jobs'][job['id']])
            interim_comparison(plan, root, state)
            seed0_progress = state['jobs'].get('E1CG_seed0', {}).get('progress', {})
            if (not state.get('interim_analysis') and int(seed0_progress.get('itr', -1)) >= 70
                    and state['status'] == 'running'):
                run_analysis(plan, root, state, final=False)
            done = sum(r.get('status') in ('complete', 'skipped') for r in state['jobs'].values())
            monitor_due = time.monotonic() >= next_monitor
            if monitor_due:
                for job in plan['jobs']:
                    record = state['jobs'].get(job['id'], {})
                    if record.get('status') != 'running':
                        continue
                    p = record.get('progress', {})
                    keys = ('itr',) if job['mode'] == 'train' else ('phase', 'completed', 'done', 'substage')
                    counter = {key: p[key] for key in keys if key in p}
                    if not counter:
                        continue
                    record['unchanged_checks'] = record.get('unchanged_checks', 0) + 1 if counter == record.get('last_counter') else 0
                    record['last_counter'] = counter
                    if record['unchanged_checks'] == 2:
                        notify(state, root, 'suspected_stall', job['id'] + ': no progress over two 30-minute checks')
                append_json(root / 'monitor_history.jsonl', dict(utc=utc(), status=state['status'],
                            jobs={k: dict(status=v['status'], progress=v.get('progress')) for k, v in state['jobs'].items()}))
                next_monitor = time.monotonic() + 1800
            if done == len(plan['jobs']):
                update_status(plan, root, state)
                if not state.get('final_analysis'):
                    run_analysis(plan, root, state, final=True)
                state.update(status='complete', monitor_status='paused', ended_utc=utc())
                notify(state, root, 'campaign_complete', 'All admitted outputs verified; no further sampling')
                update_status(plan, root, state, document=True)
                return 0
            update_status(plan, root, state, document=monitor_due or bool(failures))
            if state['status'] == 'draining_after_failure' and not active:
                raise RuntimeError('Admissions stopped after a verified failure; review is required')
            time.sleep(plan['poll_seconds'])
    except BaseException as error:
        state.update(status='blocked', error='%s: %s' % (type(error).__name__, error),
                     monitor_status='paused', ended_utc=utc())
        notify(state, root, 'controller_blocked', state['error'])
        update_status(plan, root, state, document=True)
        return 1
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    p.add_argument('--revision', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--gate-key', default='passed')
    p.add_argument('--optional-gate-key', default='checks.C1.passed')
    for action in ('validate', 'run'):
        p = sub.add_parser(action)
        p.add_argument('plan')
    args = parser.parse_args()
    if args.action == 'build':
        build(args)
    elif args.action == 'validate':
        plan = validate(read_json(args.plan))
        print(json.dumps(dict(valid=True, jobs=len(plan['jobs']), budget_gpu_hours=32)))
    else:
        sys.exit(run(args.plan))
