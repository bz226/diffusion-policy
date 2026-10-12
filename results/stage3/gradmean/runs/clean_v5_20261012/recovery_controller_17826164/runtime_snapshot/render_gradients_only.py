"""Render complete V5 gradients only, using the frozen validated analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys


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
    note.write_text(note.read_text() + '\nThe existing reward artifacts were preserved byte for byte. '
        'Single-seed curves retain each run\'s individual starting value; '
        '[the three-seed reward comparison](REWARD_3SEED_NOTE.md) retains mean ± sample-SD bands. '
        'This finalization rendered and published only the gradient figures, table and this note.\n')
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
