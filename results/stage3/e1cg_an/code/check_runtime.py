"""Read-only, no-Slurm tests of approved scope, launch order and hard accounting."""
import ast
import json
from pathlib import Path
import sys

import campaign as c


def run():
    root = Path(__file__).resolve().parent
    for path in root.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    job = c.native_job('cal_seed3000', 'cal', 3000, 'calibrate')
    base = dict(phase='calibration', budget_gpu_hours=40, max_parallel=9,
                monitor_seconds=1800, prior_gpu_seconds=0, jobs=[job])
    c.validate(base)
    assert job['cap_seconds'] == 3600
    assert not any('adagrad_norm' in item for item in job['overrides'])
    jobs = []
    for seed in (0, 1, 2):
        for name, multiplier in (('AN1', 1), ('AN3', 3), ('AN10', 10)):
            jobs.append(c.native_job('%s_seed%d' % (name, seed), name, seed,
                                     'train', multiplier * .01, .000001))
    for train in list(jobs):
        job = c.native_job('final_' + train['id'], train['condition'], 5000, 'ckpt_eval',
                           train['actor_lr'], train['adagrad_norm_b0'],
                           Path(train['logdir']) / 'checkpoint/state_139.pt')
        job['depends_on'] = [j['id'] for j in jobs if j['mode'] == 'train']
        jobs.append(job)
    plan = dict(base, phase='training', prior_gpu_seconds=3600, jobs=jobs)
    c.validate(plan)
    assert [j['id'] for j in jobs[:3]] == ['AN1_seed0', 'AN3_seed0', 'AN10_seed0']
    assert c.costs(plan, {'jobs': {}}) == (3600, 0)
    state = {'jobs': {j['id']: dict(status='pending', scheduler_job_id='1') for j in jobs[:9]}}
    assert c.costs(plan, state) == (3600, 129600)
    state['jobs'][jobs[0]['id']] = dict(status='complete', elapsed_seconds=8700, gpu_count=1)
    assert c.costs(plan, state) == (12300, 115200)
    assert sum(j['cap_seconds'] for j in jobs) + 3600 == 138600
    for invalid in (dict(plan, prior_gpu_seconds=10000), dict(plan, jobs=jobs[1:])):
        try:
            c.validate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Expected scope or budget rejection')
    for job in jobs[9:]:
        assert job['repeats'] == 2 and job['samplers'] == ['train', 'eval'] and job['seed'] == 5000
        assert '+train.ckpt_segments=true' in job['overrides']
    return dict(passed=True, python=sys.version.split()[0], syntax='passed',
                native_plan_scope='passed', launch_order='passed', shared_budget='passed',
                scope_and_budget_rejections='passed', reservations_gpu_hours=38.5,
                scientific_jobs_submitted=0)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
