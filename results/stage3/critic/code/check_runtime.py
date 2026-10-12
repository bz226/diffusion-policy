"""Mechanical runtime tests only; never starts Slurm jobs or scientific runs."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


@contextlib.contextmanager
def isolated_paths(module, directory):
    """Redirect every path constant and deny writes outside this test sandbox."""
    names = ('PROJECT', 'ROOT', 'PARENT', 'SOURCE', 'G_ROOT', 'PYTHON', 'FROZEN', 'GATE', 'STATE')
    originals = {name: getattr(module, name) for name in names}
    original_command = module.command
    sandbox = Path(directory).resolve()
    paths = dict(PROJECT=sandbox / 'project', ROOT=sandbox / 'C', G_ROOT=sandbox / 'G',
                 PARENT=sandbox / 'parent', SOURCE=sandbox / 'source', PYTHON=sandbox / 'python',
                 FROZEN=sandbox / 'G/prepare/P0/frozen_P0.pt', GATE=sandbox / 'C/runs/gate/gate_summary.json',
                 STATE=sandbox / 'C/runs/campaign.json')
    guard = dict(active=True, denied_writes=0)
    def allowed(path):
        if isinstance(path, int):
            return
        target = Path(os.fsdecode(path)).resolve()
        if target != sandbox and sandbox not in target.parents:
            guard['denied_writes'] += 1
            raise PermissionError('Runtime test attempted an out-of-sandbox write: ' + str(target))
    def audit(event, args):
        if not guard['active']:
            return
        if event == 'open':
            mode, flags = args[1], args[2]
            writable = (isinstance(mode, str) and any(mark in mode for mark in ('w', 'a', 'x', '+')))
            writable = writable or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)))
            if writable:
                allowed(args[0])
        elif event == 'os.mkdir':
            allowed(args[0])
        elif event == 'os.rename':
            allowed(args[0]); allowed(args[1])
        elif event == 'subprocess.Popen':
            raise PermissionError('Runtime tests cannot start external commands or jobs')
    def no_command(*args, **kwargs):
        raise PermissionError('Runtime tests cannot invoke scheduler or shell commands')
    sys.addaudithook(audit)
    for name, value in paths.items():
        setattr(module, name, value)
    module.command = no_command
    try:
        yield guard
    finally:
        guard['active'] = False
        for name, value in originals.items():
            setattr(module, name, value)
        module.command = original_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for path in (here / 'campaign.py', here / 'entrypoint.py'):
        compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('critic_campaign_checks', str(here / 'campaign.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = []
    def check(name, truth):
        if not truth:
            raise AssertionError(name)
        checked.append(name)
    cal = module.run_spec('calG', 3001)
    check('calG exact five-batch mode and warmed critic flags', '+train.mode=calibrate' in cal['overrides'] and
          '+train.critic_init_path=' + str(module.FROZEN) in cal['overrides'])
    rates = {'SGDC3': 3 * module.ETA, 'SGDC10': 10 * module.ETA, 'SGDC3w': 3 * module.ETA,
             'SGDG3': 3 * .1, 'SGDG10': 10 * .1}
    for name, eta in rates.items():
        run = module.run_spec(name, 0, eta_g=.1)
        check(name + ' exact LR and constant schedule', 'train.actor_lr=' + repr(eta) in run['overrides'] and
              'train.actor_lr_scheduler.min_lr=' + repr(eta) in run['overrides'])
        check(name + ' fixed140 iterations and baseline diagnostics', 'train.n_train_itr=140' in run['overrides'] and
              '+train.baseline_diag=true' in run['overrides'] and run['cap_seconds'] == 14400)
        check(name + ' explicit decision estimator', ('+train.adv_estimator=mc_critic' in run['overrides']) == name.startswith('SGDC'))
    eval_run = module.run_spec('ckpt_SGDC3w_seed2', 5000)
    check('final evaluation does not overwrite checkpoint critic with warmed critic', not any(
        x.startswith('+train.critic_init_path=') for x in eval_run['overrides']))
    check('final evaluation name and cap', eval_run['run_id'] == 'ckpt_SGDC3w_seed2_seed5000' and eval_run['cap_seconds'] == 900)
    check('source path is isolated from running G', str(module.SOURCE).startswith(str(module.PARENT)) and
          'stage3-critic' in str(module.PARENT) and 'stage3_gradmean' not in str(module.SOURCE))
    entries = [dict(scheduler_state='COMPLETED', allocated_gpu_seconds=629, cap_seconds=7200),
               dict(scheduler_state='PENDING', gpu_count=0, reserved_gpu_count=1, cap_seconds=14400)]
    value = module.ledger(entries)
    check('pending unallocated GPU still reserves full approved cap', value['actual_plus_reserved_gpu_seconds'] == 15029)
    check('actual plus reservations is bounded by60 GPU-hours', value['budget_gpu_seconds'] == 216000)
    check('NaN and infinity detected before JSON conversion', module.nonfinite({'a': float('nan'), 'b': float('inf')}) == ['.a', '.b'])
    check('JSON never emits nonstandard NaN', module.finite_json({'a': float('nan')}) == {'a': None})
    check('native nonfinite gradient marks collapse without nonfinite pickle values', module.collapsed(
        {'nonfinite_paths': [], 'native_failure': {'itr': 17, 'reason': 'nonfinite', 'last_diagnostics': {'kl': '0.001'}}}))
    check('ordinary crash is distinct from nonfinite collapse', not module.collapsed(
        {'native_failure': {'reason': 'alignment'}}))
    def job(number, gpu=1, name=None):
        return dict(job_id=str(number), name=name or 'gm-j%d' % number, gpus=gpu,
                    gres='gres/gpu:a6000:1' if gpu else 'N/A')
    gs = [job(i) for i in range(9)] + [job(20, 0, 'gm-bootstrap')]
    check('three C jobs never displace nine G GPU slots', module.capacity(gs, set(), 3, True)[0] is False)
    check('calG can use genuinely spare tenth GPU', module.capacity(gs, set(), 1, True)[0] is True)
    check('extra controllers preserve G job headroom', module.capacity(gs + [job(21, 0, 'other-controller'),
          job(22, 0, 'other-controller2')], set(), 1, True)[0] is False)
    check('total C cap includes calibration', module.capacity([job(30, name='crit-calG')], {'30'}, 3, False)[0] is False)
    check('three seeds together after G', module.capacity([], set(), 3, False)[0] is True)
    cpu_controller = dict(job_id='controller-fixture', name='critic-controller', gpus=0,
                          gres='N/A', partition='soal', state='RUNNING')
    check('verified zero-GPU critic controller may admit calG while G remains active',
          not module.persistent_controller_conflict(True, 'controller-fixture', [cpu_controller]))
    check('inherited interactive bash allocation is rejected as a persistent C controller',
          module.persistent_controller_conflict(True, '17810633', [dict(job_id='17810633', name='bash',
              gpus=1, gres='gres/gpu:titanv:1', partition='soal-interactive', state='RUNNING')]))
    check('persistent controller is allowed after G is terminal',
          not module.persistent_controller_conflict(False, 'controller-fixture', []))
    check('controller classification never bypasses G capacity checks',
          module.capacity(gs + [job(21, 0, 'other-controller'), job(22, 0, 'other-controller2')],
                          set(), 1, True)[0] is False)
    check('missing scheduler controller entry is rejected',
          module.persistent_controller_conflict(True, 'controller-fixture', []))
    check('GPU allocation masquerading as critic controller is rejected',
          module.persistent_controller_conflict(True, 'controller-fixture', [dict(cpu_controller, gpus=1)]))
    check('other newly submitted CPU controller is rejected',
          module.persistent_controller_conflict(True, 'controller-fixture', [dict(cpu_controller, name='other-controller')]))
    check('pending controller entry cannot impersonate a running controller allocation',
          module.persistent_controller_conflict(True, 'controller-fixture', [dict(cpu_controller, state='PENDING')]))
    check('nine G jobs plus bootstrap plus verified C controller can fit calG after interactive exit',
          module.capacity(gs + [cpu_controller], set(), 1, True)[0] is True)
    check('generic unresolved GPU conservatively reserves A6000', module.a6000_count(
        dict(gpus=1, gres='gres/gpu:1')) == 1)
    check('typed TitanV does not reserve A6000', module.a6000_count(dict(gpus=1, gres='gres/gpu:titanv:1')) == 0)
    check('scontrol resolved generic TitanV does not reserve A6000', module.a6000_count(
        dict(gpus=1, gres='gres/gpu:1', a6000_gpus=0)) == 0)
    canonical_state = module.STATE
    canonical_before = canonical_state.read_bytes() if canonical_state.exists() else None
    with tempfile.TemporaryDirectory(prefix='critic-runtime-') as directory, isolated_paths(module, directory) as guard:
        check('every imported path constant is redirected into the temporary sandbox', all(
            Path(directory) in getattr(module, name).parents for name in
            ('PROJECT', 'ROOT', 'PARENT', 'SOURCE', 'G_ROOT', 'PYTHON', 'FROZEN', 'GATE', 'STATE')))
        denied = False
        try:
            module.atomic(canonical_state, {'must_never_write': True})
        except PermissionError:
            denied = True
        check('write guard blocks canonical state before filesystem mutation', denied and guard['denied_writes'] >= 1)
        path = Path(directory) / 'immutable.json'
        module.once(path, {'x': 1}); module.once(path, {'x': 1})
        failed = False
        try:
            module.once(path, {'x': 2})
        except ValueError:
            failed = True
        check('old evidence cannot be overwritten', failed and json.loads(path.read_text()) == {'x': 1})
        module.atomic(module.G_ROOT / 'runs/campaign.json', dict(status='running', submitted=[]))
        check('G round vacancy cannot admit C training', module.gradient_mean_active([]))
        module.atomic(module.G_ROOT / 'runs/campaign.json', dict(status='completed', submitted=[]))
        check('G terminal status permits capacity review', not module.gradient_mean_active([]))
        native = module.ROOT / 'SGDC3_seed0'
        module.atomic(native / 'failure.json', dict(itr=17, reason='nonfinite', last_diagnostics={'KL': '0.001'}))
        observation = module.observe(dict(logdir=str(native), console_log=str(native) + '.out',
                                          run_id='SGDC3_seed0', mode='train'))
        check('fast native failure iteration and last diagnostics survive worker poll race',
              observation['native_failure']['itr'] == 17 and observation['native_failure']['last_diagnostics']['KL'] == '0.001'
              and module.collapsed(observation))
        campaign = module.Campaign.__new__(module.Campaign)
        campaign.state = dict(submitted=[dict(run_id='calG_seed3001', scheduler_state='FAILED', run_status='failed')])
        check('failed calG preserves independent SGDC cohort', [s['run_id'] for s in campaign.next_cohort()] ==
              ['SGDC3_seed0', 'SGDC3_seed1', 'SGDC3_seed2'])
        campaign.state['submitted'] += [dict(run_id='%s_seed%d' % (c, s), scheduler_state='COMPLETED', run_status='complete')
                                         for c in module.CONDITIONS[:3] for s in (0, 1, 2)]
        next_jobs = campaign.next_cohort()
        check('failed calG blocks only SGDG and still evaluates finished SGDC',
              set(campaign.state['blocked_conditions']) == {'SGDG3', 'SGDG10'} and len(next_jobs) == 3
              and next_jobs[0]['run_id'] == 'ckpt_SGDC3_seed0_seed5000')
        text = ('# Study\n\n**Question.** Keep this question.\n\n**Status.** Gate queued.\n\n'
                '**Design.** Keep this design.\n\n**Resources.** Keep this budget.\n\n**Next step.** Run gate.\n')
        (module.ROOT / 'experiment.md').write_text(text)
        (module.ROOT / 'REPORT.md').write_text('# Prior report\n\nExisting recorded evidence.\n')
        campaign.state = dict(status='blocked', submitted=[], external=[], ledger=module.ledger([]),
                              failure_reason='Mechanical fixture blocker', waiting_reason='inspect fixture')
        campaign.source = {}
        campaign.monitor('software check')
        rendered = (module.ROOT / 'experiment.md').read_text()
        check('current status replaces stale queued gate without altering question/design/resources',
              'Gate queued' not in rendered and 'C1–C5 passed' in rendered and
              all(phrase in rendered for phrase in ('Keep this question.', 'Keep this design.', 'Keep this budget.')))
        blocked_report = (module.ROOT / 'REPORT.md').read_text()
        previous = Path(campaign.state['blocked_report']['record']) / 'previous_REPORT.md'
        check('blocked idle campaign writes concise report and preserves prior evidence',
              'Mechanical fixture blocker' in blocked_report and 'No C allocation remains active' in blocked_report
              and 'Existing recorded evidence.' in previous.read_text())
        campaign.state['submitted'] = [dict(run_id='SGDC3_seed0', scheduler_state='RUNNING', job_id='fixture',
                                           cap_seconds=14400, reserved_gpu_count=1)]
        campaign.blocked_report(dict(runs=[]))
        check('blocked controller with active jobs preserves report and states continuing caps',
              'Already admitted jobs retain their individual caps' in (module.ROOT / 'REPORT.md').read_text()
              and blocked_report in (module.ROOT / 'REPORT.md').read_text())
        check('Campaign.record wrote only the sandbox STATE target', module.STATE.is_file() and
              json.loads(module.STATE.read_text())['submitted'][0]['job_id'] == 'fixture')
    canonical_after = canonical_state.read_bytes() if canonical_state.exists() else None
    check('canonical campaign state is unchanged by the tests', canonical_before == canonical_after)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Do not overwrite prior check evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(passed=True, checks=checked, check_count=len(checked),
                                      scientific_runs_started=0, isolated_filesystem=True,
                                      canonical_state_unchanged=True), indent=2) + '\n')
    print(json.dumps(dict(passed=True, checks=len(checked), output=str(output))))


if __name__ == '__main__':
    main()
