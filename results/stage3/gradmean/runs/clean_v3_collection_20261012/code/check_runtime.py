"""Mechanical checks only: no environment, model or scheduler allocations."""
import ast
import importlib.util
import json
from pathlib import Path

path = Path(__file__).with_name('collection.py')
ast.parse(path.read_text())
spec = importlib.util.spec_from_file_location('clean_v3_runtime', str(path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
m = module
checks = []


def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


check('exact_twenty_two_run_ids', len(m.RUN_IDS) == len(set(m.RUN_IDS)) == 22)
specs = []
for point in m.POINTS:
    prep = m.make_spec('prepare_' + point)
    specs.append(prep)
    frozen = dict(point=m.POINTS[point]['collector_point'], source_checkpoint=prep['checkpoint'],
                  frozen_path=str(m.G / 'prepare' / point / ('frozen_' + m.POINTS[point]['collector_point'] + '.pt')),
                  frozen_sha256='0' * 64)
    for j in range(10):
        specs.append(m.make_spec(point + '_j%02d' % j, frozen=frozen))
    wrong = dict(frozen, source_checkpoint='original_DPPO_checkpoint')
    try:
        m.make_spec(point + '_j00', frozen=wrong)
    except ValueError:
        checks.append('reject_other_checkpoint_' + point)
    else:
        raise AssertionError('Wrong checkpoint accepted')
    wrong = dict(frozen, frozen_path=str(m.G / 'prepare/D1/frozen_D1.pt'))
    try:
        m.make_spec(point + '_j00', frozen=wrong)
    except ValueError:
        checks.append('reject_original_D_frozen_path_' + point)
    else:
        raise AssertionError('Original D frozen path accepted')
check('unique_seeds', len(set(s['seed'] for s in specs)) == 22)
check('nonoverlapping_environment_ranges', not any(a != b and abs(a['seed'] - b['seed']) < 40 for a in specs for b in specs))
check('external_internal_aliases', all(s['point'] in ('C1', 'C2') and s['collector_point'] in ('D1', 'D2') for s in specs))
check('separate_C_output_paths', all(s['point'] in s['logdir'] for s in specs))
check('native_label_matches_alias', all(s['expected_settings']['gm_point'] == s['collector_point'] for s in specs))
check('fifty_batches_per_collect', all(s['expected_settings'].get('gm_batches') == 50 for s in specs if s['step'] == 'collect'))
check('five_critic_warmups', all(s['expected_settings'].get('gm_warmup_batches') == 5 for s in specs if s['step'] == 'prepare'))
check('no_optional_score_sampling', all(s['expected_settings'].get('gm_score_sample') == 0 for s in specs if s['step'] == 'collect'))
check('V6_and_wandb_disabled', all('+train.gm_include_critic=true' in s['overrides'] and 'wandb=null' in s['overrides'] for s in specs))
check('initial_full_reservation_under_budget', m.BASE_GPU_SECONDS + 22 * m.CAP < m.BUDGET_GPU_SECONDS)
check('only_prepares_admitted_first', list(m.ready({'entries': []})) == ['prepare_C1', 'prepare_C2'])
prepares = [{'run_id': 'prepare_C1', 'status': 'complete'}, {'run_id': 'prepare_C2', 'status': 'complete'}]
check('collect_exact_fixed_order', list(m.ready({'entries': prepares})) == m.RUN_IDS[2:])
failed = [{'run_id': 'prepare_C1', 'status': 'failed'}, {'run_id': 'prepare_C2', 'status': 'complete'}]
check('failed_prepare_does_not_admit_dependents', not any(x.startswith('C1_j') for x in m.ready({'entries': failed})))
try:
    m.identify('P0_j00')
except ValueError:
    checks.append('reject_original_point_admission')
else:
    raise AssertionError('Original point admitted')
old_seeds = [json.loads(q.read_text())['seed'] for q in (m.G / 'runs/specs').glob('*.json')]
check('disjoint_from_existing_environment_streams', not any(abs(s['seed'] - seed) < 40 for s in specs for seed in old_seeds))
result = dict(passed=True, checks=checks, check_count=len(checks), python38_compatible=True,
              configs=[{k:s[k] for k in ('run_id', 'point', 'collector_point', 'seed', 'checkpoint_itr', 'update_number')} for s in specs],
              total_collect_batches=sum(s['expected_settings'].get('gm_batches', 0) for s in specs))
m.old.atomic(m.HERE / 'runtime_checks.json', result)
print(json.dumps(result, indent=2))
