"""Layout-only replay from the saved table; no estimators are recomputed."""
from pathlib import Path
from datetime import datetime,timezone
import csv,importlib.util,json,shutil
root=Path('/soalnas/share/data/zbao7/diffusion_policy')
script=root/'.codex-runtime/worktrees/stage3-gradmean-v6/results/stage3/gradmean/plot_clean.py'
spec=importlib.util.spec_from_file_location('clean',str(script));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=root/'results/stage3/gradmean/figs_clean'
raw=list(csv.DictReader((out/'points_table.csv').open()))
def value(text):
    if not text:return None
    try:return float(text)
    except ValueError:return text
rows=[{k:value(v) for k,v in row.items()} for row in raw]
for row in rows:
    for k in ('K_signal_V4','K','V1_m','m','m_signal_V4','update_number'):
        if row.get(k) is not None:row[k]=int(row[k])
m.preserve_render(out)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
for variant in ('V6','V4','V5'):m.plot_noise(rows,variant,out,plt)
r=json.loads((out/'render_record.json').read_text());r['label_layout_adjustment']={'utc':datetime.now(timezone.utc).isoformat(),'source':str(script),'command':'runs/postprocess_recovery_20261012/adjust_figure_labels.py','change':'Smaller measurement-count annotations, placed above the higher and below the lower path at each checkpoint','statistics_recomputed':False,'saved_table_unchanged':True};(out/'render_record.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
shutil.copy2(script,root/'results/stage3/gradmean/plot_clean.py')
print('Adjusted three gradient figures from unchanged saved table; previous renders preserved.')
