#!/usr/bin/env python3
"""Replot saved optimizer-gradient noise; no sampling, Python 3.8."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from noise_actual_data import ROOT, GREEN, KEYS, captions, fmt, load_rows, shared_limits, table

FOLDERS = [ROOT / 'results/stage3/gradmean/figs_clean', ROOT / 'results/stage3/ppo_steps/figs']


def hashes(paths=None):
    if paths is None:
        paths = [p for folder in FOLDERS for p in sorted(folder.rglob('*'))
                 if p.is_file() and p.suffix.lower() in ('.png', '.pdf')]
    return {os.path.relpath(str(p), str(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_existing(run):
    before = json.loads((run / 'figure_hashes_before.json').read_text())
    after = hashes([ROOT / p for p in before])
    if before != after:
        raise RuntimeError('A pre-existing figure changed')
    (run / 'figure_hashes_after.json').write_text(json.dumps(after, indent=2) + '\n')
    return dict(algorithm='SHA-256, explicitly requested', files=len(before), unchanged=True,
                before=str(run / 'figure_hashes_before.json'), after=str(run / 'figure_hashes_after.json'))


def render_as_run(rows, limits, out, caption):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    points = [r for r in rows if r['method'] == 'without_PPO_and_GAE']
    x = [r['update'] for r in points]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(3.6, 2.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=.025, h_pad=.04, wspace=.02)
    titles = ('Mean gradient\nnorm', 'Gradient\nvariance\n(one update)', 'Relative\nvariance\n(one update)')
    labels = (r'$\|\mathrm{E}\,g\|$' + '\n(training-loss units)',
              r'$\mathrm{tr}\,\mathrm{Cov}(g)$' + '\n(training-loss units²)',
              r'$\mathrm{tr}\,\mathrm{Cov}(g)/\|\mathrm{E}\,g\|^2$' + '\n(no units)')
    checks = []
    for ax, key, title, label in zip(axes, KEYS, titles, labels):
        y = [r[key] for r in points]
        line, = ax.plot(x, y, 'o-', color=GREEN, markersize=2.4, linewidth=.9,
                        markerfacecolor=GREEN, markeredgecolor=GREEN)
        ax.set_yscale('log')
        ax.set_ylim(limits[key])
        ax.set_xlim(-5, 131)
        ax.set_xticks([0, 32, 63, 95, 126])
        for label_artist in ax.get_xticklabels():
            label_artist.set_rotation(45)
            label_artist.set_ha('right')
        ax.tick_params(axis='both', labelsize=6.2, length=2.5, pad=1.5)
        ax.tick_params(axis='y', which='minor', length=0)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=7, pad=5)
        ax.set_ylabel(label, fontsize=6.5, labelpad=1.5)
        ax.grid(axis='y', which='major', color='0.90', linewidth=.5)
        ax.grid(axis='x', visible=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        assert list(line.get_xdata()) == x and list(line.get_ydata()) == y
        assert list(ax.get_ylim()) == limits[key]
        assert len(ax.lines) == 1 and len(ax.collections) == 0
        checks.append(dict(quantity=key, x=x, y=y, limits=limits[key], filled_circles_only=True))
    fig.suptitle('without_PPO_and_GAE, as run\neach point: 500 batches of 80 episodes', fontsize=8)
    fig.supxlabel('Updates (one update = 80 episodes)', fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in list(fig.texts) + [text for ax in axes for text in
                                    [ax.title, ax.yaxis.label] + ax.get_xticklabels()]:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
    name = 'without_ppo_and_gae_noise_actual'
    for ext in ('png', 'pdf'):
        meta = {'Description': caption} if ext == 'png' else {'Subject': caption}
        fig.savefig(str(out / (name + '.' + ext)), dpi=300, metadata=meta)
    (out / (name + '.caption.md')).write_text(caption + '\n')
    plt.close(fig)
    return dict(name=name, checks=checks)


def note(rows, limits, run):
    lines = ['# Gradient noise per optimizer gradient', '',
             'The six figures retain every saved mean-norm point estimate. Covariance traces and relative variances '
             'are divided by 80, including their saved intervals. No sampling, jobs, gradients or jackknife estimates were recomputed.', '',
             '| Quantity | Lower y-limit | Upper y-limit |', '|---|---:|---:|']
    for key in KEYS:
        lines.append('| %s | %s | %s |' % (key, fmt(limits[key][0]), fmt(limits[key][1])))
    lines += ['', 'All new figures share these limits by quantity, at whole decades with at least 0.25 decade of margin.', '',
              '**Unresolved means.** without_PPO_and_GAE: updates 32, 95 and 126. DPPO: steps 1–2 at update 32, '
              'steps 1–5 at update 63, and steps 1–4 at update 95. Positive ratio point estimates remain plotted. '
              'The table gives the requested plug-in R lower bounds and therefore upper bounds on expected cosine. '
              'Those bounds hold variance at its estimate; they do not replace the saved joint ratio intervals [0, ∞).', '',
              '| Updates | DPPO step-1 covariance / without_PPO_and_GAE update covariance |', '|---:|---:|']
    ratios = {}
    for update in (32, 63, 95):
        a = next(r for r in rows if r['method'] == 'DPPO' and r['update'] == update and r['step'] == 1)
        b = next(r for r in rows if r['method'] == 'without_PPO_and_GAE' and r['update'] == update)
        ratios[update] = a['variance'] / b['variance']
        lines.append('| %d | %.8g |' % (update, ratios[update]))
    lines += ['',
              '**Interpretation and inherited limitations.** These are different policies at matched update counts. '
              'The normalization choices put both collections in training-loss units, but the saved without_PPO_and_GAE '
              'collection uses full-batch normalization that previously passed its check against training quarters; it is not '
              'a bitwise replay of those quarters. Its update-126 scale is the previously authorized five-batch least-squares fit, '
              'and its intervals condition on that scale.', '',
              'Norm intervals retain each source convention: direct norm jackknife for without_PPO_and_GAE '
              '(including its signed negative lower endpoint at update 126), and transformed squared-mean endpoints for DPPO. '
              'Resolution always uses the squared-mean interval. No direct batch-covariance field was stored; traces use /80. '
              'The saved B_update fields agree with B/80 for all five as-run points and interval endpoints.', '',
              '**Preservation.** All 88 existing PNG/PDF files in the two figure folders, including their history subfolders, '
              'match their explicitly requested before/after SHA-256 values. The hash records are in ' +
              '[the render record folder](' + os.path.relpath(str(run), str(FOLDERS[1])) + '/). '
              'Existing code, tables and figures were not edited. New tables are copied identically into both figure folders '
              'so captions can refer to a local noise_actual_table. Long x-axis labels are shared once per figure for legibility. '
              'There are no scientific changes.', '',
              '[Table](noise_actual_table.md) · [CSV](noise_actual_table.csv)', '']
    return '\n'.join(lines), ratios


def publish(run):
    mapping = [(run / 'candidates/gradmean', FOLDERS[0]), (run / 'candidates/ppo_steps', FOLDERS[1])]
    for source, target in mapping:
        for p in source.iterdir():
            if (target / p.name).exists():
                raise RuntimeError('Refuse to overwrite existing output: ' + str(target / p.name))
    check_existing(run)
    for source, target in mapping:
        for p in source.iterdir():
            shutil.copy2(str(p), str(target / p.name))
    return check_existing(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish-existing', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError('600-second render cap')))
    signal.alarm(600)
    if args.publish_existing:
        result = publish(run)
        record = json.loads((run / 'render_record.json').read_text())
        record.update(published=True, existing_figures=result)
        (run / 'render_record.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Published six figure sets and identical tables; %d original figures unchanged.' % result['files'])
        return
    started = time.monotonic()
    if (run / 'candidates').exists():
        raise RuntimeError('Refuse to overwrite candidates')
    before = run / 'figure_hashes_before.json'
    if not before.exists():
        run.mkdir(parents=True, exist_ok=True)
        before.write_text(json.dumps(hashes(), indent=2) + '\n')
    check_existing(run)
    rows, provenance = load_rows()
    limits, cap = shared_limits(rows), captions(rows)
    gm, ppo = run / 'candidates/gradmean', run / 'candidates/ppo_steps'
    gm.mkdir(parents=True)
    ppo.mkdir(parents=True)
    checks = [render_as_run(rows, limits, gm, cap['without_ppo_and_gae_noise_actual'])]
    source = ROOT / 'results/stage3/ppo_steps/code/plot_noise_actual.py'
    spec = importlib.util.spec_from_file_location('dppo_noise_actual_renderer', str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks += module.render(rows, limits, ppo, cap)
    table(rows, ppo)
    for ext in ('csv', 'md'):
        shutil.copy2(str(ppo / ('noise_actual_table.' + ext)), str(gm / ('noise_actual_table.' + ext)))
    note_text, ratios = note(rows, limits, run)
    (ppo / 'NOISE_ACTUAL_NOTE.md').write_text(note_text)
    (gm / 'NOISE_ACTUAL_NOTE.md').write_text(note_text.replace(
        os.path.relpath(str(run), str(FOLDERS[1])), os.path.relpath(str(run), str(FOLDERS[0]))))
    source_folder = run / 'source'
    source_folder.mkdir()
    for p in (Path(__file__), Path(__file__).with_name('noise_actual_data.py'), source):
        name = 'dppo_' + p.name if p == source else p.name
        shutil.copy2(str(p), str(source_folder / name))
    record = dict(status='complete', published=False, command=[sys.executable] + sys.argv,
                  K=500, groups=10, points=len(rows), limits=limits, sources=provenance,
                  covariance_ratios=ratios, figure_checks=checks,
                  unresolved=[dict(method=r['method'], update=r['update'], step=r['step'],
                                   R_lower_bound=r['relative_variance_lower_bound'])
                              for r in rows if not r['mean_resolved']],
                  new_sampling=False, gradients_recomputed=False,
                  elapsed_seconds=time.monotonic() - started, existing_figures=check_existing(run))
    (run / 'render_record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(candidates=str(run / 'candidates'), points=len(rows), limits=limits,
                          covariance_ratios=ratios, existing_figures_unchanged=True)))


if __name__ == '__main__':
    main()
