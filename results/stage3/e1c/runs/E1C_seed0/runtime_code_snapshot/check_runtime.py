"""Mechanical runner checks in a temporary sandbox; no external jobs or sampling."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for path in (here / 'campaign.py', here / 'entrypoint.py'):
        compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('e1c_runtime_checks', str(here / 'campaign.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    results = []
    def check(name, truth):
        if not truth:
            raise AssertionError(name)
        results.append(name)
    first = [module.run_spec(seed) for seed in (0, 1, 2)]
    for job in first:
        check('exact E1C flags seed%d' % job['seed'], set(job['overrides']) == set([
            'train.n_train_itr=140', 'train.save_model_freq=35', 'wandb=null',
            'seed=%d' % job['seed'], 'logdir=' + job['logdir']] + module.COMMON + [
            '+train.adv_estimator=mc_critic', '+train.baseline_diag=true']))
        check('four-hour one A6000 forty CPU cap seed%d' % job['seed'], job['cap_seconds'] == 14400 and
              job['gpu_count'] == 1 and job['cpus'] == 40 and job['memory_gib'] == 64)
        check('absolute output outside isolated source seed%d' % job['seed'],
              Path(job['logdir']).is_absolute() and module.PARENT not in Path(job['logdir']).parents)
    check('source is isolated E1C ordinary snapshot', module.SOURCE == module.PARENT / 'results/stage3_e1c/dppo')
    check('retry preserves prior seed output', module.run_spec(0, 1)['logdir'] != first[0]['logdir'])
    bad = False
    try:
        module.run_spec(0, 2)
    except ValueError:
        bad = True
    check('second retry forbidden', bad)
    budget = module.ledger([dict(scheduler_state='COMPLETED', allocated_gpu_seconds=3600, cap_seconds=3600),
                           dict(scheduler_state='PENDING', reserved_gpu_count=1, cap_seconds=14400)])
    check('pending reservations charged at full wall cap', budget['actual_plus_reserved_gpu_seconds'] == 18000)
    check('16 GPU-hour hard cap', budget['budget_gpu_seconds'] == 57600)
    check('nonfinite detection survives sanitization', module.nonfinite({'a': float('inf')}) == ['.a'] and
          module.finite_json({'a': float('inf')}) == {'a': None})
    fake_row = {'itr': 1, 'step': 80000, 'baseline_residual_std_ddof': 0}
    for name, mean, sd in [('critic', 2, 3), ('loo', 1, 1)]:
        fake_row['baseline_residual_mean_' + name] = [mean] * 250
        fake_row['baseline_residual_std_' + name] = [sd] * 250
    energy = module.unweighted_energy(fake_row)
    check('unweighted energy identity from per-k population moments', energy['baseline_unweighted_energy_critic'] == 3250
          and energy['baseline_unweighted_energy_loo'] == 500 and energy['baseline_unweighted_energy_ratio'] == 6.5)
    check('unweighted bin energy identity', energy['baseline_unweighted_bin_energy_ratio'] == [6.5] * 5)
    jobs = [dict(job_id=str(i), gpus=1, gres='gpu:a6000:1') for i in range(8)]
    check('three seeds wait without three A6000 slots', not module.capacity(jobs, set(), 3)[0])
    check('three seeds fit beside five G workers', module.capacity(jobs[:5], set(), 3)[0])
    check('four E1C allocations forbidden', not module.capacity([], {'fixture'}, 3)[0])
    originals = {k: getattr(module, k) for k in ('PROJECT', 'ROOT', 'PARENT', 'SOURCE', 'PYTHON', 'GATE', 'STATE')}
    canonical_before = originals['STATE'].read_bytes() if originals['STATE'].exists() else None
    with tempfile.TemporaryDirectory(prefix='e1c-runtime-') as temp:
        sandbox = Path(temp)
        for key in originals:
            setattr(module, key, sandbox / key.lower())
        module.GATE = module.ROOT / 'runs/gate/gate_summary.json'
        module.STATE = module.ROOT / 'runs/campaign.json'
        guard = {'active': True, 'denied': 0}
        def audit(event, args):
            if not guard['active']:
                return
            if event == 'subprocess.Popen':
                raise PermissionError('Runtime checks cannot start scheduler or other processes')
            paths = []
            if event == 'open' and ((isinstance(args[1], str) and any(c in args[1] for c in 'wax+')) or
                    (isinstance(args[2], int) and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))):
                paths = [args[0]]
            elif event == 'os.mkdir':
                paths = [args[0]]
            elif event == 'os.rename':
                paths = args[:2]
            for path in paths:
                if isinstance(path, int):
                    continue
                actual = Path(os.fsdecode(path)).resolve()
                if actual != sandbox and sandbox not in actual.parents:
                    guard['denied'] += 1
                    raise PermissionError('Write outside temporary test sandbox')
        sys.addaudithook(audit)
        module.command = lambda *a, **k: (_ for _ in ()).throw(PermissionError('No scheduler in checks'))
        try:
            failed = False
            try:
                module.atomic(originals['STATE'], {'must_not_write': True})
            except PermissionError:
                failed = True
            check('canonical state protected by write guard', failed and guard['denied'] >= 1)
            value = sandbox / 'record.json'
            module.once(value, {'old': 1}); module.once(value, {'old': 1})
            failed = False
            try:
                module.once(value, {'new': 1})
            except ValueError:
                failed = True
            check('old evidence never overwritten', failed and json.loads(value.read_text()) == {'old': 1})
            campaign = module.Campaign.__new__(module.Campaign)
            campaign.state = {'submitted': [], 'external': []}
            check('all three seeds admitted as one first cohort', [s['seed'] for s in campaign.next_cohort()] == [0, 1, 2])
            campaign.state['submitted'] = [dict(seed=s, run_status='failed', job_id='job%d' % s) for s in (0, 1, 2)]
            check('failed jobs not blindly retried', campaign.next_cohort() == [])
            module.atomic(module.ROOT / 'provenance/retry_fixes.json', {'fixes': [dict(seed=1, failed_job_id='job1',
                resolved=True, cause='node failed', fix='different available node', scientific_configuration_unchanged=True)]})
            retry = campaign.next_cohort()
            check('only corrected same-seed retry is offered', len(retry) == 1 and retry[0]['seed'] == 1 and retry[0]['attempt'] == 1)
            campaign.stop_requested = False
            campaign.stop(15, None)
            check('stopping controller only toggles its flag', campaign.stop_requested)
        finally:
            guard['active'] = False
    after = originals['STATE'].read_bytes() if originals['STATE'].exists() else None
    check('canonical campaign unchanged', after == canonical_before)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Preserve previous check output')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'passed': True, 'checks': results, 'count': len(results),
                     'python': sys.version, 'scientific_runs_started': 0,
                     'subprocesses_denied': True, 'filesystem_isolated': True}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'count': len(results), 'output': str(output)}))


if __name__ == '__main__':
    main()
