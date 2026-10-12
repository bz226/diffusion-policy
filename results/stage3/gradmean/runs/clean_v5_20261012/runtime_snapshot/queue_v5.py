"""Version 5 fixed-plan administration; source and normalization gates precede collection.

No prior campaign is reopened. Native collection/estimator code lives only in
the separately gated stage3-gradmean-v5 source worktree.
"""
import argparse
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path('/soalnas/share/data/zbao7/diffusion_policy')
G = PROJECT / 'results/stage3/gradmean'
HERE = G / 'runs/clean_v5_20261012'
V3 = G / 'runs/clean_v3_collection_20261012'
SPEC = importlib.util.spec_from_file_location('v3_process_supervision', str(V3 / 'code/collection.py'))
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)
old = v3.old
old.JOURNAL = HERE / 'commands.jsonl'
REPO = PROJECT / '.codex-runtime/worktrees/stage3-gradmean-v5'
SOURCE = REPO / 'results/stage3_gradmean/dppo'
GATE = HERE / 'gate_cpu/gate_summary.json'
STATE = HERE / 'campaign.json'
CAP, MAX_PAR, TOTAL_CAP = 1800, 5, 450000
BASE_REVISION = 'eabff898011503ad2635b0efdbc59e0b90c8dc5c'
POINTS = {
    'W0_000': dict(checkpoint_itr=0, update_number=0, index=7, native='P0'),
    'W0_035': dict(checkpoint_itr=35, update_number=32, index=8, native='D1'),
    'W0_070': dict(checkpoint_itr=70, update_number=63, index=9, native='D2'),
    'W0_105': dict(checkpoint_itr=105, update_number=95, index=10, native='N1'),
    'W0_139': dict(checkpoint_itr=139, update_number=126, index=11, native='N2')}
PREPARES = ['prepare_' + point for point in POINTS]
NORM = 'norm_W0_000'
COLLECTS = [point + '_j%02d' % j for point in POINTS for j in range(10)]
RUN_IDS = PREPARES + [NORM] + COLLECTS
COMMIT = None


def raw_checkpoint(point):
    audit = old.read(HERE / 'log_audit.json')
    record = audit['checkpoints'][str(POINTS[point]['checkpoint_itr'])]
    if record['update_number'] != POINTS[point]['update_number'] or not Path(record['checkpoint_path']).is_file():
        raise ValueError('Checkpoint/update audit differs from the authorized five points')
    return record


def plan():
    return dict(version=5, points=POINTS, checkpoint_audit=str(HERE / 'log_audit.json'),
                run_ids=RUN_IDS, scientific_source=str(SOURCE), source_commit=COMMIT,
                prepare_warmup=0, collect_groups_per_point=10, batches_per_collect=50,
                episodes_per_batch=80, final_divisor_fallback_batches=5,
                normalization_check=dict(point='W0_000', batches=2, seed=179100,
                    selection='full unless the authorized comparison selects quarter; frozen before first collect'),
                source_flags=dict(gm_warmup=0, gm_v2_raw=True),
                cap_seconds_per_job=CAP, max_parallel=MAX_PAR, gpu_type='RTX A6000',
                cpus_per_job=40, memory_gib=64, aggregate_gpu_seconds_cap=TOTAL_CAP,
                initial_reserved_gpu_seconds=len(RUN_IDS) * CAP,
                monitor_interval_seconds=1800, controller_cap_seconds=14400,
                estimate_wall_hours=2.1, estimate_gpu_hours=10.2,
                estimate_basis='The previous 50 V6 groups averaged 724.88 seconds; the new arithmetic must be gated first.',
                zero_actor_updates=True, zero_critic_updates=True, automatic_retries=False,
                source_gate='CPU F1-F3 must pass, source bytes must equal their gate snapshot and supplied commit',
                point_aliases='External W0 IDs use P0/D1/D2/N1/N2 native labels solely for unchanged point bookkeeping; outputs and checkpoints are separate.',
                supersedes='Version3 new admissions stopped; all previously running jobs retained; no old collection restarted.')


def identify(run_id):
    if run_id in PREPARES:
        return run_id[8:], 'prepare', None
    if run_id == NORM:
        return 'W0_000', 'norm_check', None
    if run_id in COLLECTS:
        return run_id[:-4], 'collect', int(run_id[-2:])
    raise ValueError('Run outside the exact version5 job list')


def normalization():
    chosen = old.read(HERE / 'normalization_selection.json')
    if not chosen or chosen.get('source_commit') != COMMIT or chosen.get('selected_normalization') not in ('full', 'quarter'):
        raise ValueError('Production collection requires the frozen normalization selection')
    return chosen['selected_normalization']


def make_spec(run_id, frozen=None, norm=None):
    point, step, group = identify(run_id)
    info = POINTS[point]
    record = raw_checkpoint(point)
    seed = 100000 + 10000 * info['index'] + 100 * (group if group is not None else 91 if step == 'norm_check' else 90)
    directory = HERE / ('prepare' if step == 'prepare' else 'normalization' if step == 'norm_check' else 'collect') / (point if step == 'prepare' else run_id)
    scale = record['divisor']
    if (point == 'W0_139') != (scale is None):
        raise ValueError('Only the final checkpoint permits the authorized five-batch scale fallback')
    settings = dict(mode='grad_mean', gm_step=step, gm_point=info['native'], seed=seed,
                    environment_seeds=list(range(seed, seed + 40)), gm_parent_commit=COMMIT,
                    gm_include_critic=True, gm_warmup=0, gm_scaler_c=scale, gm_v2_raw=True,
                    gm_normalization=(norm or normalization()) if step == 'collect' else 'full')
    overrides = old.COMMON + ['seed=' + str(seed), 'logdir=' + str(directory), 'base_policy_path=' + record['checkpoint_path']]
    for key in ('gm_step', 'gm_point', 'gm_parent_commit', 'gm_include_critic', 'gm_warmup', 'gm_scaler_c', 'gm_v2_raw', 'gm_normalization'):
        value = settings[key]
        literal = 'null' if value is None else str(value).lower() if isinstance(value, bool) else str(value)
        overrides.append('+train.' + key + '=' + literal)
    if step != 'prepare':
        if frozen is None:
            manifest = old.read(HERE / 'workers' / ('prepare_' + point) / 'manifest.json', {})
            frozen = old.read(HERE / 'prepare' / point / 'grad_mean.json', {})
            if manifest.get('status') != 'complete' or frozen.get('status') != 'completed':
                raise ValueError('A verified own-checkpoint W0 preparation is required')
        wanted = HERE / 'prepare' / point / ('frozen_' + info['native'] + '.pt')
        if frozen.get('point') != info['native'] or frozen.get('source_checkpoint') != record['checkpoint_path'] or Path(frozen.get('frozen_path', '')).resolve() != wanted:
            raise ValueError('Frozen state aliases a different checkpoint or preparation')
        settings.update(gm_frozen_path=str(wanted), gm_frozen_sha256=frozen['frozen_sha256'], gm_batches=50, gm_score_sample=0)
        for key in ('gm_frozen_path', 'gm_frozen_sha256', 'gm_batches', 'gm_score_sample'):
            overrides.append('+train.' + key + '=' + str(settings[key]))
    return dict(run_id=run_id, point=point, external_point_id=point, collector_point=info['native'],
                step=step, group=group, seed=seed, logdir=str(directory), checkpoint=record['checkpoint_path'],
                checkpoint_itr=info['checkpoint_itr'], update_number=info['update_number'],
                raw_unit_divisor=scale, divisor_source=record.get('divisor_source'),
                alias_explanation=plan()['point_aliases'], expected_settings=settings,
                source_path=str(SOURCE), source_repository=str(REPO), parent_commit=COMMIT,
                variant_ids=['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                console_log=str(HERE / 'logs' / (run_id + '.out')), overrides=overrides,
                cap_seconds=CAP, gpu_count=1, gpu_type='a6000', cpus=40, memory_gib=64,
                command=[str(PROJECT / 'results/repro_halfcheetah/env/bin/python'), '-B', 'script/run.py',
                         '--config-dir=cfg/gym/finetune/halfcheetah-v2', '--config-name=ft_ppo_diffusion_mlp'] + overrides)


def validate_source(*unused):
    gate = old.read(GATE, {})
    if gate.get('passed') is not True or gate.get('device') != 'cpu' or any(gate.get(key, {}).get('passed') is not True for key in ('F1', 'F2', 'F3')):
        raise ValueError('Version5 CPU F1-F3 must all pass before any admissions')
    status = old.accounting(str(gate.get('slurm_job_id', '')))
    if status.get('scheduler_state') != 'COMPLETED' or status.get('exit_code') != '0:0' or status.get('gpu_count') != 0:
        raise ValueError('Version5 gate allocation must finish successfully on CPU only')
    branch = old.command(['git', '-C', str(REPO), 'branch', '--show-current'])['stdout'].strip()
    if branch not in ('main', 'stage3-gradmean-v5'):
        raise ValueError('Unexpected isolated source branch')
    current = old.command(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])['stdout'].strip()
    for ancestor, descendant in ((BASE_REVISION, COMMIT), (COMMIT, current)):
        if not ancestor or not descendant or old.command(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', ancestor, descendant], check=False)['returncode']:
            raise ValueError('Source revision must descend from the authorized baseline and contain the gated edits')
    snapshot = Path(gate.get('source_snapshot', '')).resolve()
    if HERE not in snapshot.parents or snapshot == SOURCE:
        raise ValueError('A separate gate source snapshot under this run is required')
    actual, expected = old.files(SOURCE), old.files(snapshot)
    if not expected or actual.keys() != expected.keys():
        raise ValueError('Scientific source file set changed after the gate')
    for name in expected:
        if actual[name].read_bytes() != expected[name].read_bytes() or (actual[name].stat().st_mode & 0o111) != (expected[name].stat().st_mode & 0o111):
            raise ValueError('Scientific source changed after CPU gate: ' + name)
    prefix = str(SOURCE.relative_to(REPO)) + '/'
    listing = old.command(['git', '-C', str(REPO), 'ls-tree', '-rz', COMMIT, '--', prefix])['stdout']
    entries = {}
    for row in listing.split('\0'):
        if not row:
            continue
        metadata, name = row.split('\t', 1)
        mode, kind, oid = metadata.split()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('Scientific source must be ordinary committed files')
        entries[name[len(prefix):]] = (mode, oid)
    if entries.keys() != expected.keys():
        raise ValueError('Supplied revision lacks the complete gated source')
    ordered = sorted(entries)
    process = subprocess.run(['git', '-C', str(REPO), 'cat-file', '--batch'],
        input=''.join(entries[name][1] + '\n' for name in ordered).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if process.returncode:
        raise ValueError('Cannot inspect committed source contents')
    offset = 0
    for name in ordered:
        end = process.stdout.index(b'\n', offset)
        size = int(process.stdout[offset:end].split()[2])
        content = process.stdout[end + 1:end + 1 + size]
        offset = end + 2 + size
        if content != expected[name].read_bytes() or (entries[name][0] == '100755') != bool(expected[name].stat().st_mode & 0o100):
            raise ValueError('Source commit differs from CPU gate snapshot: ' + name)
    return dict(gate_path=str(GATE), cpu_gate_job_id=gate['slurm_job_id'], source_repository=str(REPO),
                source_path=str(SOURCE), source_snapshot=str(snapshot), source_commit=COMMIT,
                current_parent_commit=current, files_compared=len(expected), stage='v5')


def meta_path(spec):
    return Path(spec['logdir']) / ('grad_mean.json' if spec['step'] == 'prepare' else 'normalization_check.json' if spec['step'] == 'norm_check' else 'meta.json')


def observe(spec):
    progress = old.read(Path(spec['logdir']) / 'stage3_progress.json', {})
    meta = old.read(meta_path(spec), {})
    return dict(utc=old.utc(), progress=old.safe(progress), status=meta.get('status'), error=meta.get('error'),
                completed_batches=meta.get('completed_batches', meta.get('batch_count', meta.get('completed_warmup_batches', 0))),
                meta_nonfinite=not old.finite_json(meta))


def verify(spec):
    meta = old.read(meta_path(spec), {})
    if meta.get('status') != 'completed' or not old.finite_json(meta):
        raise ValueError('Native output is incomplete or nonfinite')
    if any(meta.get('settings', {}).get(k) != v for k, v in spec['expected_settings'].items()):
        raise ValueError('Effective W0/raw-V2/normalization settings differ from the exact spec')
    if meta.get('point') != spec['collector_point'] or meta.get('source_checkpoint') != spec['checkpoint']:
        raise ValueError('Wrong own-checkpoint point mapping')
    if meta.get('actor_optimizer_steps') != 0 or meta.get('critic_optimizer_steps') != 0:
        raise ValueError('W0 diagnostics must not update actor or critic')
    for key in ('actor', 'critic'):
        if not meta.get(key + '_before_sha256') or meta[key + '_before_sha256'] != meta.get(key + '_after_sha256'):
            raise ValueError('W0 ' + key + ' parameters changed')
    cleanup = old.read(Path(spec['logdir']) / 'cleanup.json', {})
    if (cleanup.get('status') != 'passed' or cleanup.get('all_children_exited') is not True or
            cleanup.get('remaining_active_children') != [] or cleanup.get('remaining_direct_children') != [] or
            cleanup.get('errors') != [] or cleanup.get('primary_error') is not None or
            len(cleanup.get('workers', [])) != 40 or
            any(w.get('exitcode') != 0 or w.get('alive') is not False for w in cleanup.get('workers', []))):
        raise ValueError('Vector-environment workers did not close cleanly')
    if spec['step'] == 'collect':
        # The existing verifier checks all six vectors, fifty rows, frozen hashes,
        # estimator cross moments, finite values, zero updates and clean shutdown.
        old.verify(spec)
    elif spec['step'] == 'prepare':
        if meta.get('completed_warmup_batches') != 0 or not Path(meta.get('frozen_path', '')).is_file():
            raise ValueError('W0 preparation changed the critic or omitted frozen output')
        effective = meta.get('effective_scaler_c')
        if not isinstance(effective, (int, float)) or not math.isfinite(effective) or effective <= 0:
            raise ValueError('W0 preparation has no finite positive raw-unit divisor')
        if spec['point'] == 'W0_139':
            fits = meta.get('scaler_fit_batches', [])
            if meta.get('scaler_c_source') != 'least_squares_five_batches' or meta.get('scaler_c_fallback_batches') != 5 or len(fits) != 5:
                raise ValueError('Final point requires exactly the authorized five-batch frozen-critic fit')
            expected_c = sum(x['sum_VG'] for x in fits) / sum(x['sum_V2'] for x in fits)
            if effective != expected_c:
                raise ValueError('Final divisor differs from the recorded through-origin fit')
        elif effective != spec['raw_unit_divisor'] or meta.get('scaler_c_source') != 'provided' or meta.get('scaler_c_fallback_batches') != 0:
            raise ValueError('Provided divisor must exactly match the training log, without a fallback')
    else:
        rows = meta.get('rows', [])
        if meta.get('batch_count') != 2 or len(rows) != 2 or meta.get('selected_normalization') not in ('full', 'quarter'):
            raise ValueError('Normalization comparison must finish exactly two batches and select full or quarter')
        if any(row.get('state_and_rng_unchanged') is not True for row in rows):
            raise ValueError('Normalization comparison changed frozen state or global random streams')
        expected_choice = 'full' if all(row['cosine'] >= .999 and abs(row['quarter_over_full_norm'] - 1) <= .01 for row in rows) else 'quarter'
        if expected_choice != meta['selected_normalization']:
            raise ValueError('Normalization selection differs from the authorized thresholds')
    if spec['step'] != 'prepare':
        if meta.get('frozen_sha256') != spec['expected_settings']['gm_frozen_sha256']:
            raise ValueError('Loaded frozen identity differs from its own preparation')
        if not meta.get('scaler_before_sha256') or meta['scaler_before_sha256'] != meta.get('scaler_after_sha256'):
            raise ValueError('Frozen reward scaler changed')
    return dict(verified_utc=old.utc(), metadata_path=str(meta_path(spec)), cleanup_path=str(Path(spec['logdir']) / 'cleanup.json'),
                external_point_id=spec['point'], collector_point=spec['collector_point'], checkpoint=spec['checkpoint'])


def bind_supervision():
    # In-process rebinding affects only this new wrapper, never existing workers.
    v3.HERE, v3.STATE, v3.GATE = HERE, STATE, GATE
    v3.CAP, v3.MAX_PAR, v3.SOURCE_COMMIT = CAP, MAX_PAR, COMMIT
    v3.make_spec, v3.verify = make_spec, verify
    old.V6_SOURCE, old.V6_REPO = SOURCE, REPO
    old.validate_v6_gate, old.observe = validate_source, observe


def ledger(state):
    finished = [e for e in state['entries'] if e.get('scheduler_state') in old.TERMINAL]
    actual = sum(e.get('allocated_gpu_seconds', 0) for e in finished)
    reserved = CAP * (len(state['entries']) - len(finished))
    return dict(base_gpu_seconds=state['base_gpu_seconds'], new_actual_gpu_seconds=actual,
                outstanding_reserved_gpu_seconds=reserved,
                aggregate_actual_plus_reserved=state['base_gpu_seconds'] + actual + reserved,
                aggregate_gpu_seconds_cap=TOTAL_CAP)


def record(state, monitor=False):
    state.update(updated_utc=old.utc(), ledger=ledger(state))
    old.atomic(STATE, state)
    if monitor:
        old.append(HERE / 'monitor_history.jsonl', dict(utc=old.utc(), status=state['status'], ledger=state['ledger'],
            entries=[{k:e.get(k) for k in ('run_id', 'job_id', 'status', 'scheduler_state', 'progress')} for e in state['entries']]))
    completed = sum(e.get('status') == 'complete' for e in state['entries'])
    text = ('# Version 5 own-critic gradient collection\n\nStatus: **' + state['status'] + '**; ' + str(completed) + '/56 jobs verified. '
            'The five points use their exact checkpoint critic without warm-up; only the final divisor uses the authorized five-batch fit. '
            'The full/quarter normalization choice is frozen after the two-batch comparison, before collection.\n\n'
            'MAX_PAR=5; 30-minute job caps and 125 aggregate GPU-hours. Scheduled monitor: every 30 minutes. '
            'Failures stop admissions; already-running jobs are left unchanged. No automatic retries.\n\n'
            '[Plan](plan.json) · [Current ledger](campaign.json) · [Checkpoint/scaler audit](log_audit.json).\n')
    (HERE / 'experiment.md').write_text(text)
    marker = ('**Clean figures version 5 collection (' + state['updated_utc'] + '): '
              + state['status'] + '.** ' + str(completed) + '/56 fixed jobs verified; '
              '30-minute monitor ' + state.get('monitor_status', 'active') + '. '
              '[Current collection record](runs/clean_v5_20261012/experiment.md).')
    update_main_marker(marker)


def update_main_marker(message):
    path = G / 'experiment.md'
    archive = HERE / 'experiment_before_v5_monitor.md'
    if not archive.exists():
        with archive.open('x') as target:
            target.write(path.read_text())
    start, end = '<!-- clean-v5-status-start -->', '<!-- clean-v5-status-end -->'
    block = start + '\n' + message + '\n' + end
    for _ in range(3):
        before = path.read_text()
        after = (before[:before.index(start)] + block + before[before.index(end) + len(end):]
                 if start in before and end in before else before.rstrip() + '\n\n' + block + '\n')
        temporary = path.with_name('.experiment.clean-v5.' + str(os.getpid()))
        temporary.write_text(after)
        if path.read_text() != before:
            temporary.unlink()
            continue
        os.replace(str(temporary), str(path))
        return
    raise RuntimeError('Concurrent experiment.md edit; no overwrite performed')


def update(state):
    for entry in state['entries']:
        if entry.get('status') in ('complete', 'failed'):
            continue
        status = old.accounting(entry['job_id'])
        entry.update(status)
        manifest = old.read(HERE / 'workers' / entry['run_id'] / 'manifest.json', {})
        if manifest:
            entry['progress'] = observe(old.read(HERE / 'specs' / (entry['run_id'] + '.json')))['completed_batches']
        if status.get('scheduler_state') in old.TERMINAL:
            if status['scheduler_state'] == 'COMPLETED' and status.get('exit_code') == '0:0' and manifest.get('status') == 'complete':
                entry['verification'] = verify(old.read(HERE / 'specs' / (entry['run_id'] + '.json')))
                entry['status'] = 'complete'
            else:
                entry.update(status='failed', failure_reason=manifest.get('failure_reason', 'Scheduler or native job failed'))


def freeze_normalization(state):
    entry = next((e for e in state['entries'] if e['run_id'] == NORM and e.get('status') == 'complete'), None)
    if entry is None:
        return
    spec = old.read(HERE / 'specs' / (NORM + '.json'))
    result = old.read(meta_path(spec))
    decision = dict(source_commit=COMMIT, selected_normalization=result['selected_normalization'],
                    comparison_path=str(meta_path(spec)), scheduler_job_id=entry['job_id'],
                    frozen_before_first_collect=True, comparison_seed=179100, comparison_batches=2)
    old.once(HERE / 'normalization_selection.json', decision)


def ready(state):
    entries = {e['run_id']:e for e in state['entries']}
    for name in PREPARES:
        if name not in entries:
            yield name
    if not all(entries.get(name, {}).get('status') == 'complete' for name in PREPARES):
        return
    if NORM not in entries:
        yield NORM
        return
    if entries[NORM].get('status') != 'complete':
        return
    normalization()
    for name in COLLECTS:
        if name not in entries:
            yield name


def submit(state, name):
    spec = make_spec(name)
    spec_path = HERE / 'specs' / (name + '.json')
    old.once(spec_path, spec)
    if any(p.exists() for p in (HERE / 'workers' / name / 'manifest.json', Path(spec['logdir']), Path(spec['console_log']))):
        raise ValueError('Refusing to overwrite existing outputs: ' + name)
    argv = ['sbatch', '--parsable', '--account=soal', '--partition=soal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=40', '--mem=64G', '--gres=gpu:a6000:1', '--no-requeue',
            '--export=PATH,HOME,USER,LOGNAME,LANG', '--time=30', '--signal=B:TERM@30',
            '--job-name=gm-v5-' + name, '--chdir=' + str(SOURCE),
            '--output=' + str(HERE / 'logs' / (name + '-%j.out')), str(HERE / 'code/run_worker_v5.sh'),
            '--commit', COMMIT, '--spec', str(spec_path)]
    state['pending_submission_intent'] = dict(utc=old.utc(), run_id=name, command=argv)
    record(state)
    result = old.command(argv, check=False)
    job = result['stdout'].strip().split(';')[0]
    if result['returncode'] or not job.isdigit():
        state['submission_failure'] = result
        record(state)
        raise RuntimeError('Submission failed or ambiguous; automatic retry forbidden')
    state['entries'].append(dict(run_id=name, job_id=job, submitted_utc=old.utc(), scheduler_state='PENDING',
        status='submitted', cap_seconds=CAP, reserved_gpu_count=1, command=argv))
    state.pop('pending_submission_intent')
    record(state)


def run():
    lock = (HERE / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        raise ValueError('An existing version5 controller record cannot be restarted implicitly')
    winddown = old.read(HERE / 'v3_final_ledger.json', {})
    if not winddown.get('all_workers_verified') or not winddown.get('ledger', {}).get('is_final'):
        raise ValueError('All admitted old jobs must finish and final accounting must be verified first')
    base = winddown['ledger']['aggregate_actual_gpu_seconds_as_observed']
    if base + len(RUN_IDS) * CAP > TOTAL_CAP:
        raise ValueError('All fixed reservations must fit in the aggregate budget')
    provenance = validate_source()
    old.once(HERE / 'plan.json', plan())
    snapshot = HERE / 'runtime_snapshot'
    snapshot.mkdir(exist_ok=False)
    for path in (HERE / 'code').glob('*'):
        if path.is_file():
            shutil.copy2(str(path), str(snapshot / path.name))
    state = dict(status='running', monitor_status='active', started_utc=old.utc(), entries=[],
        controller_job_id=os.environ.get('SLURM_JOB_ID'), base_gpu_seconds=base,
        source_provenance=provenance, plan=plan(), runtime_snapshot=str(snapshot),
        administration_parent_commit=old.command(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'])['stdout'].strip())
    started, next_monitor, stop = time.monotonic(), 0, []
    signal.signal(signal.SIGTERM, lambda sig, frame: stop.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stop.append(sig))
    try:
        while True:
            update(state)
            freeze_normalization(state)
            record(state, monitor=time.monotonic() >= next_monitor)
            if time.monotonic() >= next_monitor:
                next_monitor = time.monotonic() + 1800
            failures = [e['run_id'] for e in state['entries'] if e.get('status') == 'failed']
            if failures:
                raise RuntimeError('Terminal job failure; no further admissions: ' + ', '.join(failures))
            if stop or time.monotonic() - started >= plan()['controller_cap_seconds'] - 30:
                raise RuntimeError('Controller stopped/capped; running workers left unchanged')
            if len(state['entries']) == len(RUN_IDS) and all(e.get('status') == 'complete' for e in state['entries']):
                state.update(status='complete', ended_utc=old.utc(), monitor_status='paused')
                record(state, monitor=True)
                return 0
            jobs = old.scheduler_jobs()
            nodes = old.command(['scontrol', 'show', 'nodes', '-o'])['stdout']
            slots = old.queue_slots(state['entries'], jobs, max_parallel=MAX_PAR, free_slots=old.free_a6000_slots(nodes))
            for name in list(ready(state))[:slots]:
                if ledger(state)['aggregate_actual_plus_reserved'] + CAP > TOTAL_CAP:
                    raise RuntimeError('Aggregate budget prevents another admission')
                submit(state, name)
            time.sleep(15)
    except BaseException as error:
        state.update(status='blocked', ended_utc=old.utc(), monitor_status='paused',
                     failure_reason=type(error).__name__ + ': ' + str(error))
        record(state, monitor=True)
        raise


def main():
    global COMMIT
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('validate', 'run', 'worker'))
    parser.add_argument('--commit', required=True)
    parser.add_argument('--spec')
    args = parser.parse_args()
    COMMIT = args.commit
    bind_supervision()
    (HERE / 'logs').mkdir(parents=True, exist_ok=True)
    if args.mode == 'validate':
        record = validate_source()
        old.atomic(HERE / 'source_validation.json', record)
        for point in POINTS:
            old.once(HERE / 'specs' / ('prepare_' + point + '.json'), make_spec('prepare_' + point))
        old.once(HERE / 'plan.json', plan())
        print('Version5 source gate and fixed preparation specifications verified', flush=True)
        return 0
    return v3.worker(args) if args.mode == 'worker' else run()


if __name__ == '__main__':
    sys.exit(main())
