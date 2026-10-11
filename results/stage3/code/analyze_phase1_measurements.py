#!/usr/bin/env python
"""Report existing Phase-1 point estimates without reusing invalid uncertainty."""
import csv
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/phase1_analysis'
OUT = RUN / 'candidates'
POINTS = [('theta0', 'ns_theta0_seed1000', 20), ('theta_ckpt', 'ns_ckpt_seed2000', 5)]
VARIANTS = ['nc4', 'mc_critic', 'mc_loo', 'theory_loo', 'theory_nobase']
LABELS = ['V1 NC4', 'V2 MC critic', 'V3 MC LOO', 'V4 theory LOO', 'V5 theory no baseline']
FIELDS = ['u_sq', 'tr_sigma', 'B_env', 'B_ep', 'snr_training_batch', 'expected_batch_cosine']


def read(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(right, left):
    return right / left if right is not None and left is not None and left != 0 else None


def display(value, places=0):
    return '—' if value is None else ('{:,.%df}' % places).format(value)


def main():
    owned = ['point_estimates.csv', 'direction_cosines.csv', 'contrasts.csv',
             'summary.json', 'analysis.md', 'table_caption.md']
    if (RUN / 'analysis_record.json').exists() or any((OUT / name).exists() for name in owned):
        raise FileExistsError('Preserve the previous Phase-1 analysis')
    OUT.mkdir(parents=True, exist_ok=True)
    audit = read(ROOT / 'runs/noise_ci_review.json')
    assert audit['status'] == 'uncertainty_unreliable'
    rows, directions, metadata, inputs, originals = [], [], {}, [], {}
    data = {}
    for point, run_id, warmup in POINTS:
        path = ROOT / run_id / 'noise_scale.json'
        manifest_path = ROOT / 'runs' / run_id / 'manifest.json'
        manifest, value = read(manifest_path), read(path)
        progress_path = ROOT / run_id / 'stage3_progress.json'
        progress = read(progress_path)
        measurements_path = ROOT / run_id / 'noise_measurements.json'
        assert manifest['status'] == 'complete' and value['status'] == 'completed'
        assert progress['status'] == 'complete' and progress['completed'] == progress['target'] == 25
        assert value['n_batches'] == value['measurement_batches'] == 25
        assert value['n_env_rollouts'] == 1000 and value['warmup_batches'] == warmup
        assert len(value['warmup']) == warmup and len(value['measurements']) == 25
        assert len(read(measurements_path)) == 25
        assert all(b['critic_optimizer_steps'] == 20 for b in value['warmup'])
        assert value['variant_order'] == VARIANTS
        data[point] = value
        metadata[point] = {
            'run': run_id, 'seed': value['seed'], 'warmup_batches': warmup,
            'measurement_batches': 25, 'environment_rollouts': 1000, 'episodes': 2000,
            'critic_explained_var_first_measurement': value['critic_explained_var_after_warmup'],
            'critic_explained_var_mean_measurement': value['critic_explained_var_mean_measurement'],
            'elapsed_seconds': value['elapsed_seconds'], 'sampling': value['sampling'],
            'gradient_units': value['gradient_units'], 'source_commit': manifest['source_commit']}
        for v, row in enumerate(value['variants']):
            assert row['variant'] == VARIANTS[v]
            assert math.isfinite(row['u_sq']) and row['tr_sigma'] > 0
            derived = {k: row[k] for k in FIELDS}
            if row['u_sq'] > 0:
                expected = [row['tr_sigma'] / row['u_sq'], 2 * row['tr_sigma'] / row['u_sq'],
                            40 * row['u_sq'] / row['tr_sigma'],
                            (1 + row['tr_sigma'] / (40 * row['u_sq'])) ** -0.5]
                for key, expected_value in zip(FIELDS[2:], expected):
                    assert row[key] is not None and math.isclose(row[key], expected_value, rel_tol=1e-13)
                reason = 'positive signed point; uncertainty unvalidated'
            else:
                assert all(row[k] is None for k in FIELDS[2:])
                reason = 'undefined: nonpositive signed signal estimate'
            rows.append(dict(policy=point, run=run_id, variant_id='V%d' % (v + 1),
                             variant=row['variant'], **derived, point_status=reason))
        for v, first in enumerate(VARIANTS):
            for w, second in enumerate(VARIANTS):
                cosine = value['cosine_matrix'][v][w]
                u, other = value['variants'][v]['u_sq'], value['variants'][w]['u_sq']
                cross = value['cross_signal'][v][w]
                if u > 0 and other > 0:
                    assert cosine is not None and math.isclose(cosine, cross / math.sqrt(u * other), rel_tol=1e-13, abs_tol=1e-13)
                else:
                    assert cosine is None
                reason = ('undefined: nonpositive signal' if cosine is None else
                          'outside [-1,1]: invalid as a population cosine' if abs(cosine) > 1 + 1e-12 else
                          'provisional; no validated uncertainty')
                directions.append(dict(policy=point, from_variant=first, to_variant=second,
                                       corrected_inner_product=cross, cosine=cosine, status=reason))
        for file in [path, manifest_path, progress_path, measurements_path]:
            originals[file] = file.read_bytes()
        inputs.append({'policy': point, 'summary': str(path), 'manifest': str(manifest_path),
                       'measurements': str(measurements_path),
                       'gram': str(ROOT / run_id / 'noise_batch_gram.npy'),
                       'scalar_cross_sums': str(ROOT / run_id / 'noise_batch_scalar_cross_sums.npy')})
    assert len({v['source_commit'] for v in metadata.values()}) == 1
    contrasts = []
    for point in data:
        value = data[point]
        for v in range(4):
            left, right = value['variants'][v:v+2]
            cosine = value['cosine_matrix'][v][v+1]
            cosine_status = ('undefined: nonpositive signal' if cosine is None else
                             'outside [-1,1]: invalid as a population cosine' if abs(cosine) > 1 + 1e-12 else
                             'provisional; no validated uncertainty')
            contrasts.append(dict(comparison_type='adjacent_variants', policy=point,
                                  from_variant=left['variant'], to_variant=right['variant'],
                                  B_env_ratio=ratio(right['B_env'], left['B_env']),
                                  noise_trace_ratio=right['tr_sigma'] / left['tr_sigma'],
                                  corrected_cosine=cosine, cosine_status=cosine_status,
                                  interpretation='descriptive point only; no valid confidence interval'))
    for v, variant in enumerate(VARIANTS):
        left, right = data['theta0']['variants'][v], data['theta_ckpt']['variants'][v]
        contrasts.append(dict(comparison_type='between_policies', policy='theta0_to_theta_ckpt',
                              from_variant=variant, to_variant=variant,
                              B_env_ratio=ratio(right['B_env'], left['B_env']),
                              noise_trace_ratio=right['tr_sigma'] / left['tr_sigma'],
                              corrected_cosine=None, cosine_status='not estimated across policies',
                              interpretation='separate measurement seeds; no valid uncertainty on policy difference'))
    write_csv(OUT / 'point_estimates.csv', rows)
    write_csv(OUT / 'direction_cosines.csv', directions)
    write_csv(OUT / 'contrasts.csv', contrasts)
    summary = {'scope': 'Existing two-policy Phase-1 measurements; point estimates only',
               'metadata': metadata, 'points': rows, 'direction_cosines': directions, 'contrasts': contrasts,
               'uncertainty': 'Saved percentile CIs, confidence flags and lower bounds excluded as unreliable; no replacement method computed',
               'nonpositive_signals': [r for r in rows if r['u_sq'] <= 0],
               'out_of_range_unique_direction_pairs': {point: sum(
                   data[point]['cosine_matrix'][v][w] is not None and abs(data[point]['cosine_matrix'][v][w]) > 1 + 1e-12
                   for v in range(5) for w in range(v + 1, 5)) for point in data}}
    write_json(OUT / 'summary.json', summary)
    table = ['| Variant | θ₀: B_ep | θ₀: batch cosine | Checkpoint: B_ep | Checkpoint: batch cosine |',
             '| --- | ---: | ---: | ---: | ---: |']
    for v, label in enumerate(LABELS):
        left, right = data['theta0']['variants'][v], data['theta_ckpt']['variants'][v]
        table.append('| %s | %s | %s | %s | %s |' % (label, display(left['B_ep']),
                     display(left['expected_batch_cosine'], 3), display(right['B_ep']),
                     display(right['expected_batch_cosine'], 3)))
    report = '''# Phase 1: fixed-policy gradient noise in HalfCheetah-v2

**Question.** At the pretrained policy and the final NC4 policy, how noisy are V1–V5 at the training batch of 80 episodes, and which estimator changes alter their mean directions?

**Answer.** The point estimates suggest noise-dominated gradients, but estimator rankings and cross-policy changes remain inconclusive. The saved percentile-bootstrap intervals are unreliable; they and confidence-based bounds are excluded. Two squared-signal estimates are negative and seven unique direction-cosine estimates exceed 1, showing that the corrected signal and direction estimates are not reliably resolved.

TABLE

B_ep is estimated noise trace / squared mean-gradient norm, expressed in episodes; batch cosine is the approximation (1+B_ep/80)^−1/2. Values are provisional points, not uncertainty-qualified conclusions. Dashes retain nonpositive-signal cases. [Noise-scale figure](noise_scale.png) · [Direction figure](direction_cosines.png) · [Captions](figure_captions.md).

**Noise at θ₀.** V5’s B_ep≈114,292 and batch cosine≈0.026 suggest a very weak directional signal at 80 episodes; its estimated SNR is 0.000700. V1/V3/V4 give cosines 0.044/0.043/0.032. V2’s signed squared-signal estimate is −6.38×10⁻⁸, so its noise ratio is unavailable. These measurements do not establish a required batch size.

**Estimator changes.** At θ₀, V3→V4 raises the point noise ratio 1.81×, and V4→V5 raises it 1.46×; dropping LOO raises the raw noise trace 27.4×, but also changes estimated signal. Their corrected direction cosines are 0.709 and −0.269, respectively, with no valid uncertainty. V1→V2 and V2→V3 are undefined at θ₀. A largest-effect ranking and claims of alignment or opposition are therefore unsupported. Raw gradient norms across V1–V5 also use different coefficient/reduction scales.

**Policy change.** From θ₀ to the checkpoint, point B_ep falls from 41,253 to 15,571 for V1 and 114,292 to 24,396 for V5, while V3 rises from 43,263 to 234,465. Checkpoint V4 has signed signal −0.319 and no defined noise ratio. These mixed point changes cannot establish how the underlying noise scale evolved. First-measurement critic explained variance is 0.644/0.273, with 25-batch averages 0.473/0.197; thus the two measurement points include different critic quality.

**Setup.** θ₀ used seed 1000 and 20 critic warm-up batches; NC4 1e-4 seed-0’s final checkpoint used measurement seed 2000 and five warm-ups. Each then measured 25 fresh batches of 40 environment rollouts (two episodes each), totaling 1,000 gradients per variant and 2,000 episodes. Actor, critic and reward scaler were fixed during measurement. All five variants share each batch; normalization and LOO introduce within-batch dependence, making the prescribed per-environment correction approximate. See [methods](../../../methods.md#fixed-policy-noise-measurements-and-uncertainty).

**Boundary.** Saved measurements and the original estimates are preserved. No training, fresh rollouts, bootstrap rerun or replacement uncertainty estimator was used. Resolving uncertainty requires a separate methodological decision, not interpreting missing estimates as zero. [Audit](../../noise_ci_review.json) · [All point moments](point_estimates.csv) · [Directions](direction_cosines.csv) · [Contrasts](contrasts.csv) · [Table definitions](table_caption.md) · [Verification and sources](../analysis_record.json).

**Decision.** Review these candidates for curation; the uncertainty limitation must accompany exports.
'''.replace('TABLE', '\n'.join(table))
    assert len(report.split()) <= 600, len(report.split())
    (OUT / 'analysis.md').write_text(report)
    (OUT / 'table_caption.md').write_text('''# Phase-1 table definitions

`point_estimates.csv` retains all ten policy/variant combinations: signed squared-signal estimate `u_sq`, per-environment covariance trace `tr_sigma`, B_env, B_ep=2B_env, training-batch SNR=40/B_env and expected cosine (1+B_env/40)^−1/2. Gradients are negative mean-loss gradients per 5,000-pair environment rollout; theory-variant gradients equal minus the episode-average estimator divided by 2,500, so their signal and variance both scale by 2,500² when expressed in paper units, while noise ratios and cosines are invariant. Units differ across estimator variants; raw norm magnitudes alone do not rank estimator quality. Each policy has 25 fresh batches, 40 environment rollouts and 80 episodes per batch, with shared normalization/LOO coupling where applicable.

`direction_cosines.csv` includes all 50 matrix entries and their corrected inner products; null cells retain nonpositive signal, and out-of-range corrected cosines remain numerically visible and explicitly invalid as population cosines. `contrasts.csv` gives the eight adjacent-variant and five cross-policy descriptive comparisons, with explicit status labels for invalid/undefined cosines; a missing ratio is never replaced by zero, and the cross-policy rows do not estimate cross-policy direction cosines. The saved percentile confidence intervals, positivity flags and lower bounds are deliberately absent, with no replacement confidence calculation or error bars. All point comparisons remain provisional; see the [uncertainty audit](../../noise_ci_review.json), [analysis record](../analysis_record.json) and [independent reconstruction](../independent_verification.json).
''')
    assert all(p.read_bytes() == content for p, content in originals.items())
    write_json(RUN / 'analysis_record.json', {
        'status': 'completed_point_analysis_pending_figure_review',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'command': [sys.executable, '-B', str(Path(__file__).resolve())],
        'working_directory': str(Path.cwd()), 'source_script': str(Path(__file__).resolve()),
        'scientific_source_commit': metadata['theta0']['source_commit'],
        'inputs': inputs, 'metadata': metadata, 'uncertainty_review': 'runs/noise_ci_review.json',
        'scope': 'Existing Phase-1 measurements only; no new estimator or resampling',
        'versions': {'python': sys.version, 'numpy': np.__version__},
        'validation': {'completed_policy_points': 2, 'variants_per_policy': 5,
                       'measurement_batches_per_policy': 25, 'warmup_batches': [20, 5],
                       'point_formulas_checked': True, 'source_records_unchanged': True,
                       'nonpositive_signal_count': 2, 'out_of_range_unique_direction_pairs': 7},
        'new_scientific_samples': 0, 'replacement_uncertainty_method': None,
        'artifacts': [str((OUT / name).relative_to(ROOT)) for name in owned],
        'curation': 'unreviewed'})
    print('\n'.join(table))
    print('Report words:', len(report.split()))


if __name__ == '__main__':
    main()
