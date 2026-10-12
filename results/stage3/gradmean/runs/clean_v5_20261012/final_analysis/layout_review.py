"""Presentation-only revision from the verified saved final statistics."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
import plot_clean as renderer
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = HERE.parents[2]
manifest = json.loads((HERE / 'manifest.json').read_text())
record = json.loads((Path(manifest['candidate_directory']) / 'render_record.json').read_text())
output = HERE / 'layout_candidates'
output.mkdir(exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'savefig.facecolor':'white'})
original = renderer.save_figure
checks = []
def save(fig, path, name, caption, plotting):
    original_lines = [[(line.get_xdata().copy(), line.get_ydata().copy()) for line in axis.lines] for axis in fig.axes]
    fig.set_size_inches(7.25, 3.45)
    for axis in fig.axes:
        axis.set_xlabel('Updates\n(one update = 80 episodes)', fontsize=8)
        axis.tick_params(which='both', labelsize=7.5)
        axis.yaxis.label.set_size(8)
    fig.subplots_adjust(left=.085, right=.99, bottom=.225, top=.70, wspace=.95)
    fig.canvas.draw()
    drawing = fig.canvas.get_renderer()
    boxes = [axis.get_tightbbox(drawing) for axis in fig.axes]
    assert all(a.x1 < b.x0 for a,b in zip(boxes,boxes[1:])), [box.bounds for box in boxes]
    import numpy as np
    for axis,before in zip(fig.axes,original_lines):
        assert len(before)==len(axis.lines)
        for line,(x,y) in zip(axis.lines,before):
            np.testing.assert_array_equal(line.get_xdata(),x)
            np.testing.assert_array_equal(line.get_ydata(),y)
    variant = 'V2' if name.endswith('as_run') else 'V6'
    unresolved = [str(point['update_number']) for point in record['statistics']['points'].values()
                  if not point['estimators'][variant]['u_norm']['signal_resolved']]
    caption += ' The squared-mean signal is unresolved at updates '+', '.join(unresolved)+'; a positive norm point estimate does not establish a resolved signal.'
    original(fig,path,name,caption,plotting)
    checks.append({'figure':name,'adjacent_axis_text_boxes_disjoint':True,'all_plotted_values_unchanged':True,
                   'unresolved_signal_updates':unresolved,'png_dpi':300})
renderer.save_figure = save
for variant in ['V2','V6']:
    renderer.plot_noise(record['statistics'],variant,output,plt)
(HERE/'layout_review.json').write_text(json.dumps({'status':'passed','scope':'layout and uncertainty explanation only',
 'source_statistics':str(Path(manifest['candidate_directory'])/'render_record.json'),'candidates':str(output),
 'numeric_analysis_recomputed':False,'reward_code_changed':False,'gradient_plot_code_changed':False,'checks':checks},indent=2)+'\n')
print(json.dumps({'status':'passed','candidates':str(output),'checks':checks},indent=2))
