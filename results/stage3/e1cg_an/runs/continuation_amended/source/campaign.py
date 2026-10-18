"""Bounded AdaGrad-Norm study admissions and monitoring (Python 3.8).

Separate calibration/training plans share one 40 GPU-hour accounting boundary.
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

from worker import atomic_json, append_json, json_check, read_json, utc, verify_source

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
ROOT = PROJECT / 'results/stage3/e1cg_an'
WORKTREE = PROJECT / '.codex-runtime/worktrees/stage3-e1cg-an'
SOURCE = WORKTREE / 'results/stage3/e1cg_an/dppo'
PYTHON = PROJECT / 'results/repro_halfcheetah/env/bin/python'
REFERENCE = PROJECT / 'results/stage3/e1cg'
KL_TARGET = 0.000584517336086918
AMENDED_ETAS = {'AN1': 0.00263955, 'AN3': 0.00791865, 'AN10': 0.0263955}
AMENDED_B0 = 0.00000322129
CAP = 40 * 3600
COMMON = ['model.clip_ploss_coef=1e6', 'model.clip_ploss_coef_base=1e6',
          'train.target_kl=null', '+model.clamp_logprob=false',
          'model.randn_clip_value=100', '+train.actor_single_step=true',
          '+model.actor_loss=score', '+train.stage3_diag=true',
          '+train.adv_estimator=mc_critic', '+train.baseline_diag=true',
          '+train.decision_discount=true', '+train.decision_discount_post_norm=true']
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
START = '<!-- AN runtime monitor -->'
END = '<!-- end AN runtime monitor -->'


def command(argv, readonly=False):
    error = None
    for attempt in range(3 if readonly else 1):
        try:
            run = subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
            if run.returncode == 0:
                return run.stdout
            error = RuntimeError('%r exited %d: %s' % (argv, run.returncode, run.stderr[-2000:]))
        except subprocess.TimeoutExpired as caught:
            error = caught
        if readonly and attempt < 2:
            time.sleep((2, 5)[attempt])
    raise error


def env():
    cache = ROOT / 'cache'
    return dict(E1CG_AN_SOURCE=str(SOURCE), PYTHONPATH=str(SOURCE),
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
    destination = ROOT / 'runs/provenance/source_snapshot'
    provenance = dict(worktree=str(WORKTREE), revision=revision, source=str(SOURCE),
                      source_relative='results/stage3/e1cg_an/dppo', snapshot=str(destination),
                      branch=command(['git', '-C', WORKTREE, 'branch', '--show-current'], True).strip())
    if command(['git', '-C', WORKTREE, 'rev-parse', 'HEAD'], True).strip() != revision:
        raise ValueError('Requested parent revision differs from source worktree HEAD')
    if destination.exists():
        verify_source(provenance)
        return provenance
    prefix = provenance['source_relative'] + '/'
    paths = {str(p.relative_to(SOURCE)): p for p in SOURCE.rglob('*') if p.is_file()
             and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(SOURCE).parts)
             and p.suffix != '.pyc'}
    tracked = {p[len(prefix):] for p in command(['git', '-C', WORKTREE, 'ls-files', '--', prefix], True).splitlines()
               if p.startswith(prefix)}
    if not set(paths).issubset(tracked):
        raise ValueError('Commit all scientific source files before making a launch plan')
    if command(['git', '-C', WORKTREE, 'status', '--porcelain', '--untracked-files=no', '--', prefix], True).strip():
        raise ValueError('Source must match its committed parent revision')
    for name, source in paths.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
    verify_source(provenance)
    return provenance


def native_job(name, condition, seed, mode, eta=None, b0=None, checkpoint=None):
    out = ROOT / ('evaluations/' + name if mode == 'ckpt_eval' else name)
    extra = []
    if mode == 'calibrate':
        extra += ['+train.mode=calibrate', '+train.calib_kl_target=%.17g' % KL_TARGET]
    else:
        extra += ['+train.actor_optimizer=adagrad_norm', 'train.actor_lr=%.17g' % eta,
                  'train.actor_lr_scheduler.min_lr=%.17g' % eta, '+train.adagrad_norm_b0=%.17g' % b0]
    if mode == 'ckpt_eval':
        # Existing E1CG mode permits repeated rollouts via its inert segment flag.
        extra += ['+train.mode=ckpt_eval', '+train.ckpt_segments=true',
                  '+train.ckpt_eval_repeats=2', '+train.ckpt_samplers=both',
                  'base_policy_path=' + str(checkpoint)]
    overrides = ['train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
                 'seed=%d' % seed, 'logdir=' + str(out)] + COMMON + extra
    job = dict(id=name, condition=condition, seed=seed, mode=mode,
               cap_seconds={'train': 14400, 'calibrate': 3600, 'ckpt_eval': 600}[mode],
               estimate_seconds={'train': 8700, 'calibrate': 180, 'ckpt_eval': 80}[mode],
               gpu_count=1, cpus=40, memory_gib=64, cwd=str(SOURCE),
               command=[str(PYTHON), '-B', str(ROOT / 'code/entrypoint.py'),
                        '--config-dir=' + str(SOURCE / 'cfg/gym/finetune/halfcheetah-v2'),
                        '--config-name=ft_ppo_diffusion_mlp'] + overrides,
               env=env(), logdir=str(out), stdout=str(ROOT / 'runs/logs' / (name + '.out')),
               progress_file=str(out / 'stage3_progress.json'), depends_on=[], ready_files=[],
               completion_checks=[], overrides=overrides, actor_lr=eta, adagrad_norm_b0=b0)
    if mode == 'train':
        job['completion_files'] = [str(out / 'result.pkl')] + [str(out / 'checkpoint' / ('state_%d.pt' % i))
                                                            for i in (0, 35, 70, 105, 139)]
    elif mode == 'calibrate':
        job.update(kl_target=KL_TARGET, completion_files=[str(out / 'calibration.json')])
    else:
        job.update(repeats=2, samplers=['train', 'eval'], ready_files=[str(checkpoint)],
                   completion_files=[str(out / 'ckpt_eval.json'), str(out / 'ckpt_eval_rollouts.json')])
        job['completion_checks'] = [dict(path=str(out / 'ckpt_eval.json'), key='status', equals='completed')]
    return job


def calibration_summary(path):
    value = read_json(path)
    if value.get('status') != 'completed' or len(value.get('batches', [])) != 5:
        raise ValueError('A completed five-batch calibration is required')
    if value.get('seed') != 3000 or value.get('kl_target') != KL_TARGET:
        raise ValueError('Calibration seed or KL target differs from the approved protocol')
    norms = [float(b['gradient_norm']) for b in value['batches']]
    lengths = [float(b['eta_b']) * n for b, n in zip(value['batches'], norms)]
    if any(not math.isfinite(x) or x <= 0 for x in norms + lengths):
        raise ValueError('Calibration gradients and distances must be positive and finite')
    return dict(calibration_path=str(Path(path).resolve()), kl_target=KL_TARGET, d_b=lengths,
                d_star=statistics.median(lengths), spread=max(lengths) / min(lengths),
                median_gradient_norm=statistics.median(norms), b0=1e-3 * statistics.median(norms))


def amended_calibration_summary(path, amendment_path):
    """Preserve the measured calibration; bind run values to the user's amendment."""
    raw = calibration_summary(path)
    amendment = read_json(amendment_path)
    if amendment.get('approved') is not True:
        raise ValueError('The exact-value amendment has not been approved')
    if amendment.get('eta_by_condition') != AMENDED_ETAS or amendment.get('b0') != AMENDED_B0:
        raise ValueError('Amendment must name the exact three approved eta values and b0')
    summary = dict(raw)
    summary.update(raw_calibrated_d_star=raw['d_star'], raw_calibrated_b0=raw['b0'],
                   d_star=AMENDED_ETAS['AN1'], b0=AMENDED_B0,
                   eta_by_condition=dict(AMENDED_ETAS),
                   amendment_path=str(Path(amendment_path).resolve()),
                   run_values_source='User-approved exact values in the G3 amendment; calibration not rerun.')
    return summary


def build(args):
    gate = dict(path=str(Path(args.gate).resolve()), key=args.gate_key, equals=True)
    okay, detail = json_check(gate)
    if not okay:
        raise ValueError('Required gate not passed: ' + detail)
    phase = 'calibration' if args.action == 'build-calibration' else 'training'
    provenance = source_snapshot(args.revision)
    spent = 0.0
    if phase == 'calibration':
        jobs = [native_job('cal_seed3000', 'cal', 3000, 'calibrate')]
        summary = None
    else:
        old = read_json(ROOT / 'runs/campaign_calibration/campaign.json', {})
        if old.get('status') != 'complete' or old.get('reserved_gpu_hours') != 0:
            raise ValueError('Calibration campaign must finish before training is planned')
        if old['source_provenance'] != provenance:
            raise ValueError('Calibration and training must use the identical source snapshot')
        spent = float(old['actual_gpu_hours']) * 3600
        amendment_path = getattr(args, 'amendment', None)
        summary = (amended_calibration_summary(args.calibration, amendment_path)
                   if amendment_path else calibration_summary(args.calibration))
        jobs = []
        # Each condition's seed 0 is submitted before any seed 1 or 2.
        for seed in (0, 1, 2):
            for condition, multiplier in (('AN1', 1), ('AN3', 3), ('AN10', 10)):
                eta = (summary['eta_by_condition'][condition] if 'eta_by_condition' in summary
                       else multiplier * summary['d_star'])
                jobs.append(native_job('%s_seed%d' % (condition, seed), condition, seed, 'train',
                                       eta, summary['b0']))
        for train in list(jobs):
            checkpoint = Path(train['logdir']) / 'checkpoint/state_139.pt'
            job = native_job('final_' + train['id'], train['condition'], 5000, 'ckpt_eval',
                             train['actor_lr'], train['adagrad_norm_b0'], checkpoint)
            job.update(source_seed=train['seed'], depends_on=[j['id'] for j in jobs if j['mode'] == 'train'])
            jobs.append(job)
    plan = dict(campaign_id='e1cg_an_' + phase, phase=phase,
                run_root=str(ROOT / 'runs' / ('campaign_' + phase)), python=str(PYTHON),
                source_provenance=provenance, budget_gpu_hours=40, prior_gpu_seconds=spent,
                max_parallel=9, monitor_seconds=1800, poll_seconds=30, controller_cap_seconds=259200,
                account='soal', partition='soal', global_checks=[gate], jobs=jobs,
                calibration=summary, automatic_retries=False, max_same_seed_retries_after_review=1)
    validate(plan)
    target = ROOT / 'runs' / ('plan_' + phase + '.json')
    if target.exists():
        raise ValueError('Plan already exists; do not overwrite admitted provenance')
    atomic_json(target, plan)
    if summary is not None:
        derived_name = 'calibration_derived_amended.json' if 'amendment_path' in summary else 'calibration_derived.json'
        atomic_json(ROOT / 'runs' / derived_name, summary)
        lines = ['# Exact expanded overrides; each seed has its own line.']
        lines += ['%s %d %s' % (j['condition'], j['seed'], ' '.join(j['overrides']))
                  for j in jobs if j['mode'] == 'train']
        (ROOT / 'jobs_e1cg_an.txt').write_text('\n'.join(lines) + '\n')
    print(str(target))


def validate(plan):
    if plan['budget_gpu_hours'] != 40 or not 1 <= plan['max_parallel'] <= 9 or plan['monitor_seconds'] != 1800:
        raise ValueError('Approved cap is 40 GPU-hours, up to nine workers, with 30-minute monitoring')
    ids = [j['id'] for j in plan['jobs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job identifiers')
    for j in plan['jobs']:
        if (j['gpu_count'], j['cpus'], j['memory_gib']) != (1, 40, 64):
            raise ValueError('Each scientific allocation uses one A6000, 40 CPUs and 64 GiB')
        if j['cap_seconds'] != {'train': 14400, 'calibrate': 3600, 'ckpt_eval': 600}[j['mode']]:
            raise ValueError('Unexpected job time cap')
        if not set(j['depends_on']).issubset(ids):
            raise ValueError('Unknown dependency')
    if plan['phase'] == 'calibration':
        if len(plan['jobs']) != 1 or plan['jobs'][0]['mode'] != 'calibrate':
            raise ValueError('Exactly one five-batch calibration is approved')
    else:
        training = [j for j in plan['jobs'] if j['mode'] == 'train']
        evaluation = [j for j in plan['jobs'] if j['mode'] == 'ckpt_eval']
        expected = [(name, seed) for seed in (0, 1, 2) for name in ('AN1', 'AN3', 'AN10')]
        if [(j['condition'], j['seed']) for j in training] != expected or len(evaluation) != 9:
            raise ValueError('Exactly nine training runs in approved order and nine final evaluations are required')
        if any(j['repeats'] != 2 or j['samplers'] != ['train', 'eval'] for j in evaluation):
            raise ValueError('Final evaluations use two rollouts per sampler')
        if (plan.get('calibration') or {}).get('amendment_path'):
            summary = plan['calibration']
            if summary.get('eta_by_condition') != AMENDED_ETAS or summary.get('b0') != AMENDED_B0:
                raise ValueError('The approved exact optimizer settings changed')
            for job in training + evaluation:
                if job['actor_lr'] != AMENDED_ETAS[job['condition']] or job['adagrad_norm_b0'] != AMENDED_B0:
                    raise ValueError('A job differs from the approved exact optimizer settings')
    if float(plan['prior_gpu_seconds']) + sum(j['cap_seconds'] for j in plan['jobs']) > CAP:
        raise ValueError('Full planned allocations cannot fit the remaining 40 GPU-hour cap')
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
                                      'J_disc_train', 'kl_true_per_action', 'adagrad_b') if key in last}
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
            '--time=%d' % math.ceil(job['cap_seconds'] / 60), '--job-name=e1cg-an-' + job['id'],
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
                       'GPU time: %.3f hours used; %.3f hours reserved; 40-hour combined cap. Active or queued: %s.' %
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
    reference = dict(logdir=str(REFERENCE / 'E1CG_seed0'))
    rows = []
    for name, data in [('AdamW (lr 1e-4)', train_rows(reference))] + [(j['condition'], train_rows(j)) for j in jobs]:
        evaluation = next(r for r in data if r['itr'] == 70)
        training = next(r for r in data if r['itr'] == 69)
        rows.append(dict(run=name, eval_itr70=evaluation['eval_episode_reward'],
                         train_itr69=training['train_episode_reward'], J_disc_itr69=training['J_disc_train'],
                         kl_itr69=training['kl_true_per_action']))
    path = ROOT / 'runs/itr70_comparison.json'
    atomic_json(path, dict(utc=utc(), seed=0, rows=rows))
    state['itr70_note_ready'] = True
    notify(state, Path(plan['run_root']), 'interim_ready', str(path))
    target = ROOT / 'code/analyze.py'
    if target.exists():
        argv = [str(PYTHON), '-B', str(target), '--root', str(ROOT), '--interim', '--target-itr', '70']
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
    argv = [str(PYTHON), '-B', str(target), '--root', str(ROOT), '--final']
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
            failures = refresh(plan, state)
            if failures:
                if state['status'] != 'draining_after_failure':
                    notify(state, root, 'failure', failures)
                state.update(status='draining_after_failure', failures=failures)
            active = [r for r in state['jobs'].values() if r['status'] in ('running', 'pending', 'submission_unknown')]
            if state['status'] == 'running':
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
    for name in ('build-calibration', 'build-training'):
        item = sub.add_parser(name)
        item.add_argument('--revision', required=True)
        item.add_argument('--gate', required=True)
        item.add_argument('--gate-key', default='passed')
        if name == 'build-training':
            item.add_argument('--calibration', required=True)
            item.add_argument('--amendment', help='Approved exact-value amendment; raw calibration is retained')
    for name in ('run', 'validate', 'status'):
        item = sub.add_parser(name)
        item.add_argument('plan')
    args = parser.parse_args()
    if args.action.startswith('build-'):
        build(args)
    elif args.action == 'run':
        run(args.plan)
    elif args.action == 'validate':
        print(json.dumps(dict(valid=True, jobs=len(validate(read_json(args.plan))['jobs']))))
    else:
        plan = validate(read_json(args.plan))
        print(json.dumps(read_json(Path(plan['run_root']) / 'campaign.json', {}), indent=2))
