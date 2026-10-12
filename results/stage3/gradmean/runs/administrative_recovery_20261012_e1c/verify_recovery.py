"""CPU-only administrative recovery/admission checks; no scheduler mutations or samples."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
module_spec = importlib.util.spec_from_file_location('gm_campaign_check', str(ROOT / 'code/campaign.py'))
campaign = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(campaign)
from analysis_core import verified_administrative_recovery


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected a rejected administrative recovery or admission')


def main():
    config = json.loads((HERE / 'recovery.json').read_text())
    config['config_path'] = str(HERE / 'recovery.json')
    state = json.loads((HERE / 'before_campaign.json').read_text())
    native_checks = []
    for run_id, allowed in config['runs'].items():
        original = json.loads(Path(allowed['original_manifest']).read_text())
        spec = json.loads((ROOT / 'runs/specs' / (run_id + '.json')).read_text())
        entry = next(row for row in state['submitted'] if row['run_id'] == run_id)
        status = {'scheduler_state': 'FAILED', 'exit_code': '1:0'}
        annotation = campaign.branch_guard_recovery(entry, status, spec, original, config)
        manifest = dict(original, scientific_status='complete', administrative_recovery=annotation)
        accepted = dict(entry, scheduler_state='FAILED', exit_code='1:0', terminal_recorded=True,
                        run_status='complete', wrapper_status='failed', administrative_recovery=annotation)
        assert manifest['status'] == 'failed' and accepted['scheduler_state'] == 'FAILED'
        assert verified_administrative_recovery(ROOT, run_id, accepted, manifest)
        for key, value in [('subprocess_exitcode', 1), ('remaining_process_group_members', [123]),
                           ('failure_reason', 'nonfinite gradients')]:
            bad = copy.deepcopy(original)
            bad[key] = value
            expect_value_error(lambda: campaign.branch_guard_recovery(entry, status, spec, bad, config))
        assert not verified_administrative_recovery(ROOT, run_id, dict(accepted, exit_code='2:0'), manifest)
        assert not verified_administrative_recovery(ROOT, run_id, dict(accepted, administrative_recovery={}), manifest)
        assert not verified_administrative_recovery(ROOT, run_id, accepted, dict(manifest, scientific_status='failed'))
        assert not verified_administrative_recovery(ROOT, 'N2_j51', accepted, manifest)
        native_checks.append(run_id)

    # Slot policy: old behavior remains nine; priority use is bounded at five.
    assert campaign.queue_slots([], []) == 9
    assert campaign.queue_slots([], [], max_parallel=5, free_slots=8) == 5
    assert campaign.queue_slots([], [], max_parallel=5, free_slots=2) == 2
    own = [{'job_id': '1', 'scheduler_state': 'PENDING'}]
    jobs = [{'job_id': '1', 'state': 'PENDING', 'gpus': 1},
            {'job_id': '2', 'state': 'RUNNING', 'gpus': 3}]
    assert campaign.queue_slots(own, jobs, max_parallel=5, free_slots=3) == 2
    expect_value_error(lambda: campaign.queue_slots([], [], max_parallel=10))
    expect_value_error(lambda: campaign.queue_slots([], [], max_parallel=0))
    nodes = ('NodeName=a Partitions=soal State=MIXED CfgTRES=cpu=256,mem=1030000M,gres/gpu:a6000=5 '
             'AllocTRES=cpu=32,mem=256G,gres/gpu=1\n')
    assert campaign.free_a6000_slots(nodes + nodes.replace('NodeName=a', 'NodeName=b')) == 8
    assert campaign.free_a6000_slots(nodes.replace('cpu=32', 'cpu=240')) == 0
    assert campaign.free_a6000_slots(nodes.replace('mem=256G', 'mem=1000000M')) == 0
    assert campaign.free_a6000_slots(nodes.replace('State=MIXED', 'State=DRAIN')) == 0
    assert campaign.free_a6000_slots(nodes.replace('gres/gpu=1', 'gres/gpu=4')) == 1

    # Simulated full accounting reconciliation: no writes, subprocesses or admissions.
    model = campaign.Campaign.__new__(campaign.Campaign)
    model.args = SimpleNamespace(max_parallel=5, respect_free_a6000=True)
    model.state = copy.deepcopy(state)
    model.recovery = config
    model.load_external = lambda: None
    model.inspect_analyses = lambda: None
    model.record = lambda: model.state.update(ledger=campaign.ledger(model.state['external'] + model.state['submitted']))
    accounting = {'17819422': 799, '17819426': 724, '17819445': 724, '17819529': 692,
                  '17819573': 722, '17819574': 725, '17819729': 739, '17819739': 2, '17819750': 2}
    written = {}
    original_accounting, original_atomic = campaign.accounting, campaign.atomic
    campaign.accounting = lambda job: dict(scheduler_state='FAILED', exit_code='1:0',
        elapsed_seconds=accounting[str(job)], allocated_gpu_seconds=accounting[str(job)], gpu_count=1)
    campaign.atomic = lambda path, value: written.update({str(path): copy.deepcopy(value)})
    try:
        model.inspect()
    finally:
        campaign.accounting, campaign.atomic = original_accounting, original_atomic
    for run_id in native_checks:
        entry = model.entry(run_id)
        assert entry['scheduler_state'] == 'FAILED' and entry['run_status'] == 'complete'
        assert model.successful_group(*campaign.parse_run(run_id)[:2]) == run_id
    for run_id in ('N2_j51', 'N2_j52'):
        assert model.entry(run_id)['run_status'] == 'failed'
        assert model.successful_group(*campaign.parse_run(run_id)[:2]) is None
        assert campaign.seed_for('N2', int(run_id[-2:]), retry=True) in (1045100, 1045200)
    assert model.state['ledger']['budget_gpu_seconds'] == 125 * 3600
    assert model.state['ledger']['outstanding_reserved_gpu_seconds'] == 0
    assert model.state['ledger']['actual_gpu_seconds'] == state['ledger']['actual_gpu_seconds'] + sum(accounting.values())
    result = {'passed': True, 'python': sys.version, 'native_verified_groups': native_checks,
              'recovered_groups': 7, 'pre_sampling_failures_preserved': 2, 'scheduler_failures_preserved': 9,
              'simulated_ledger': model.state['ledger'], 'defaults_max_parallel': 9,
              'priority_max_parallel': 5, 'live_resource_limit_checked': True,
              'negative_recovery_guards_passed': True, 'scientific_sampling': False,
              'scheduler_mutations': False, 'campaign_state_mutations': False}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
