"""Pure administrative recovery guards; no allocations or scientific work."""
import ast
import copy
import errno
import json
from pathlib import Path

import recover_controller as r

r.q.COMMIT = r.COMMIT
state = r.q.old.read(r.q.STATE)
checks = []
ast.parse(Path(r.__file__).read_text(), feature_version=(3, 8))
recovered = r.validate_original(state)
assert recovered == state and recovered is not state
assert recovered['entries'] is not state['entries']
assert len(recovered['entries']) == 11
checks.append('preserve_all_eleven_admissions_without_mutating_original')


def reject(name, mutate):
    sample = copy.deepcopy(state)
    mutate(sample)
    try:
        r.validate_original(sample)
    except ValueError:
        checks.append(name)
    else:
        raise AssertionError(name)


reject('reject_pending_intent', lambda x:x.update(pending_submission_intent={}))
reject('reject_wrong_failure_origin', lambda x:x.update(failure_reason='RuntimeError: scientific failure'))
reject('reject_scientific_failure', lambda x:x['entries'][-1].update(status='failed'))
reject('reject_scientific_scheduler_failure', lambda x:x['entries'][-1].update(scheduler_state='FAILED'))
reject('reject_prior_recovery', lambda x:x.update(controller_recoveries=[{}]))
reject('reject_missing_admission', lambda x:x['entries'].pop())
reject('reject_wrong_source', lambda x:x['source_provenance'].update(source_commit='wrong'))
reject('reject_changed_budget', lambda x:x['plan'].update(max_parallel=6))
calls = []
def transient(path, default):
    calls.append(path)
    if len(calls) == 1:
        raise OSError(errno.ESTALE, 'test')
    return {'progress': 3}
assert r.read_progress('progress', reader=transient, sleeper=lambda delay: None) == {'progress':3}
assert len(calls) == 2
checks.append('one_transient_ESTale_reopens_and_succeeds')
for code, expected in ((errno.ESTALE, 3), (errno.EIO, 1)):
    calls = []
    def fail(path, default):
        calls.append(path)
        raise OSError(code, 'test')
    try:
        r.read_progress('progress', reader=fail, sleeper=lambda delay: None)
    except OSError as error:
        assert error.errno == code and len(calls) == expected
    else:
        raise AssertionError('Unexpected read success')
    checks.append('bounded_ESTale_three_attempts' if code == errno.ESTALE else 'non_ESTale_not_retried')
remaining = [name for name in r.q.RUN_IDS if name not in {e['run_id'] for e in recovered['entries']}]
assert remaining == r.q.COLLECTS[5:] and len(remaining) == 45
checks.append('only_the_forty_five_unadmitted_jobs_remain')
result = dict(passed=True, checks=checks, count=len(checks), scientific_jobs_launched=False,
              scientific_source_changed=False, original_controller_files_changed=False)
(r.HERE / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
