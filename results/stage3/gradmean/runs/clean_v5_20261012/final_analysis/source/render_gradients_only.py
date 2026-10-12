"""Render complete V5 gradients only, using the frozen validated analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys


def relink_external_note(note, root):
    """External evidence links work from candidates and the published folder."""
    replacements = {
        '../runs/clean_v5_20261012/gate_cpu/': root / 'runs/clean_v5_20261012/gate_cpu/gate_summary.json',
        '../runs/clean_v5_20261012/': root / 'runs/clean_v5_20261012/experiment.md',
        'eval_reward.png': root / 'figs_clean/eval_reward.png',
        'train_reward.png': root / 'figs_clean/train_reward.png',
        'REWARD_3SEED_NOTE.md': root / 'figs_clean/REWARD_3SEED_NOTE.md',
    }
    for target, path in replacements.items():
        if not path.is_file():
            raise ValueError('Required existing reward/evidence link is missing: ' + str(path))
        note = note.replace('](' + target + ')', '](' + str(path.resolve()) + ')')
    return note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    source = args.snapshot.resolve()
    sys.path.insert(0, str(source))
    import plot_clean as renderer
    root = args.root.resolve()
    rewards = renderer.load_reward_data(root.parents[2])
    statistics = renderer.build_statistics(root, analysis_source=source)
    if set(statistics['points']) != set(renderer.POINTS):
        raise ValueError('All five verified points must be complete before final rendering')
    if any(point['K'] != 500 or point['groups'] != 10 for point in statistics['points'].values()):
        raise ValueError('The final gradient figures require exactly ten 50-batch groups per point')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run = root / 'runs' / ('clean_v5_gradient_final_' + stamp)
    candidates = run / 'candidates'
    candidates.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix == '.py':
            shutil.copy2(str(path), str(run / path.name))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    for variant in ('V2', 'V6'):
        if renderer.plot_noise(statistics, variant, candidates, plt) is not True:
            raise ValueError('A complete gradient figure was not rendered')
    rows = renderer.build_table(statistics, rewards)
    renderer.write_tables(rows, candidates)
    renderer.write_note(rewards, statistics, rows, candidates)
    note = candidates / 'NOTE.md'
    text = (note.read_text() + '\nThe existing reward artifacts were preserved byte for byte. '
        'Single-seed curves retain each run\'s individual starting value; '
        '[the three-seed reward comparison](REWARD_3SEED_NOTE.md) retains mean ± sample-SD bands. '
        'This finalization rendered and published only the gradient figures, table and this note.\n')
    note.write_text(relink_external_note(text, root))
    record = dict(version=5, status='complete', run=str(run), utc=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv, rewards=rewards, statistics=statistics, pending_points=[],
                  artifacts_scope='gradient figures and table only',
                  reward_artifacts_rendered=False, simulation_launched_by_renderer=False,
                  curated=False, analysis_source_snapshot=str(source))
    (candidates / 'render_record.json').write_text(json.dumps(renderer.json_safe(record), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', run=str(run), pending_points=[],
                         artifacts_scope=record['artifacts_scope'])))


if __name__ == '__main__':
    main()
