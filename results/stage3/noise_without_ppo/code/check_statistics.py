"""Deterministic arithmetic check of aggregation; no rollouts or random draws."""
import json
from pathlib import Path
import numpy as np
from analyze import summarize_point


def main():
    t = np.arange(500, dtype=np.float64)
    a = np.stack((2 + np.sin(t), 3 + np.cos(t / 7), 4 + t / 1000), axis=1)
    b = np.stack((1 + np.cos(t), 4 + np.sin(t / 8), 3 + t / 2000), axis=1)
    groups = []
    for j in range(10):
        x, y = a[j * 50:(j + 1) * 50], b[j * 50:(j + 1) * 50]
        groups.append(dict(directory='arithmetic_fixture_%d' % j, count=50,
            vectors=[x.sum(0), y.sum(0)], Q=np.array([(x*x).sum(), (y*y).sum()]),
            within_sum=float(np.sum(np.sum(x*y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))),
            meta=dict(checkpoint_itr=0, source_checkpoint='arithmetic_fixture', scaler_c=1,
                      scaler_c_source='arithmetic fixture only', scaler_approximation=False)))
    result = summarize_point(groups, 0)
    errors = {}
    for v, data in enumerate((a, b)):
        inner = data @ data.T
        signal = (inner.sum() - np.trace(inner)) / (500 * 499)
        variance = np.var(data, axis=0, ddof=1).sum()
        row = result['rows'][v]
        for key, expected in [('signal_sq', signal), ('variance', variance),
                              ('relative_variance', variance / signal)]:
            error = abs(row[key] - expected) / abs(expected)
            errors['%d_%s' % (v, key)] = float(error)
            assert error < 1e-11, (key, error)
    mean_dot = sum(float(np.dot(groups[g]['vectors'][0], groups[h]['vectors'][1]))
                   for g in range(10) for h in range(10) if g != h) / (90 * 50 * 50)
    errors['paired_group_mean_dot'] = abs(result['paired']['mean_dot']['value'] - mean_dot) / abs(mean_dot)
    assert errors['paired_group_mean_dot'] < 1e-11
    path = Path(__file__).resolve().parents[1] / 'runs/statistics_arithmetic_check.json'
    path.write_text(json.dumps(dict(passed=True, errors=errors,
        scope='Deterministic explicit batch-pair and group-pair sums versus aggregation; no RNG, model or rollout.'), indent=2) + '\n')
    print('Statistics arithmetic passed; maximum relative difference %.3g' % max(errors.values()))


if __name__ == '__main__':
    main()
