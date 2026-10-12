"""Bounded-administration checks without models, environments or allocations."""
import ast
import importlib.util
import json
from pathlib import Path

path = Path(__file__).with_name('queue_v5.py')
ast.parse(path.read_text())
spec = importlib.util.spec_from_file_location('v5_queue_check', str(path))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.COMMIT = 'PENDING_PASSED_GATE_COMMIT'
checks = []


def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


rows = []
for point in q.POINTS:
    prep = q.make_spec('prepare_' + point)
    rows.append(prep)
    frozen = dict(point=q.POINTS[point]['native'], source_checkpoint=prep['checkpoint'],
                  frozen_path=str(q.HERE / 'prepare' / point / ('frozen_' + q.POINTS[point]['native'] + '.pt')),
                  frozen_sha256='0' * 64)
    for j in range(10):
        rows.append(q.make_spec(point + '_j%02d' % j, frozen=frozen, norm='full'))
    if point == 'W0_000':
        rows.append(q.make_spec(q.NORM, frozen=frozen))
    bad = dict(frozen, frozen_path=str(q.G / 'prepare/P0/frozen_P0.pt'))
    try:
        q.make_spec(point + '_j00', frozen=bad, norm='full')
    except ValueError:
        checks.append('reject_old_preparation_' + point)
    else:
        raise AssertionError('Old preparation accepted')
check('exact_fifty_six_jobs', len(q.RUN_IDS) == len(set(q.RUN_IDS)) == len(rows) == 56)
check('exact_fifty_collects', len([x for x in rows if x['step'] == 'collect']) == 50)
check('exact_ten_collects_each_point', all(sum(x['point'] == p and x['step'] == 'collect' for x in rows) == 10 for p in q.POINTS))
check('exact_fifty_batches_per_collect', all(x['expected_settings']['gm_batches'] == 50 for x in rows if x['step'] == 'collect'))
check('normalization_fixed_point_and_seed', next(x for x in rows if x['step'] == 'norm_check')['seed'] == 179100)
check('all_zero_warmup', all(x['expected_settings']['gm_warmup'] == 0 for x in rows))
check('all_raw_V2', all(x['expected_settings']['gm_v2_raw'] is True for x in rows))
check('only_final_explicit_null_scale', all(('+train.gm_scaler_c=null' in x['overrides']) == (x['point'] == 'W0_139') for x in rows))
check('noise_and_clamp_match_authorized_measurements', all('model.randn_clip_value=100' in x['overrides'] and '+model.clamp_logprob=false' in x['overrides'] for x in rows))
check('unique_seeds', len(set(x['seed'] for x in rows)) == 56)
check('nonoverlapping_new_environment_streams', not any(a != b and abs(a['seed'] - b['seed']) < 40 for a in rows for b in rows))
old_seeds = []
for base in (q.G / 'runs/specs', q.V3 / 'specs'):
    for p in base.glob('*.json'):
        old_seeds.append(json.loads(p.read_text())['seed'])
check('disjoint_old_environment_streams', not any(abs(x['seed'] - s) < 40 for x in rows for s in old_seeds))
check('all_outputs_separate_from_old_collection', all(q.HERE in Path(x['logdir']).parents for x in rows))
check('only_prepares_first', list(q.ready({'entries': []})) == q.PREPARES)
prepares = [{'run_id':x, 'status':'complete'} for x in q.PREPARES]
check('normalization_after_all_prepares', list(q.ready({'entries': prepares})) == [q.NORM])
check('no_collection_while_normalization_running', list(q.ready({'entries': prepares + [{'run_id':q.NORM, 'status':'submitted'}]})) == [])
original_normalization = q.normalization
q.normalization = lambda: 'full'
check('exact_fixed_collect_order_after_norm', list(q.ready({'entries': prepares + [{'run_id':q.NORM, 'status':'complete'}]})) == q.COLLECTS)
q.normalization = original_normalization
try:
    q.identify('P0_j00')
except ValueError:
    checks.append('old_points_cannot_be_admitted')
else:
    raise AssertionError('Old point accepted')
check('thirty_minute_caps', q.CAP == 1800 and all(x['cap_seconds'] == 1800 for x in rows))
check('five_parallel_maximum', q.MAX_PAR == 5)
winddown = q.old.read(q.HERE / 'v3_final_ledger.json')
check('old_jobs_fully_verified', winddown['all_workers_verified'] is True and winddown['ledger']['is_final'] is True)
base = winddown['ledger']['aggregate_actual_gpu_seconds_as_observed']
check('all_initial_reservations_within_aggregate_cap', base + len(rows) * q.CAP <= q.TOTAL_CAP)
check('no_automatic_retries', q.plan()['automatic_retries'] is False)
result = dict(passed=True, check_count=len(checks), checks=checks, python38_compatible=True,
              production_source_gate_not_yet_validated=True, source_commit_pending=True,
              cumulative_prior_gpu_hours=base / 3600, fixed_reserved_gpu_hours=len(rows) * q.CAP / 3600,
              maximum_aggregate_gpu_hours=(base + len(rows) * q.CAP) / 3600)
q.old.atomic(q.HERE / 'queue_checks.json', result)
q.old.atomic(q.HERE / 'queue_preview.json', dict(plan=q.plan(), specifications=rows, all_configs_preview_only=True))
print(json.dumps(result, indent=2))
