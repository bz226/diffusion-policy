# Gradient-only recovery finalization

This CPU wrapper completes the already-approved V5 analysis after the bounded
collection has finished. It does not submit, restart, cancel, or change jobs.

Run with the campaign's recorded scientific source revision, after successful
completion of recovery controller 17828455:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
results/repro_halfcheetah/env/bin/python -B \
results/stage3/gradmean/runs/clean_v5_20261012/recovery_controller_17826164/code/finalize_gradients_v5.py \
  --commit <campaign.source_provenance.source_commit>
```

`run_finalizer.sh` requests four CPUs, 64 GiB and 30 minutes, with no GPU request.
The wrapper reuses the tested administrative adapter for at most three attempts
on transient read-only Slurm failures; mutations are never retried. External
reward, gate and collection links use their real absolute paths, so the note
works from both candidate and published locations.

`finalize_gradients_v5.py` retains the original finalizer's completion, source,
frozen-state, cleanup, and aggregate-budget checks. Its explicit publication list
contains only `gradient_noise_as_run` and `gradient_noise_paper_form` PNG/PDF/caption
files, `points_table.csv`, `points_table.md`, `NOTE.md`, and `render_record.json`.
It archives only previous files with those names. All existing reward files,
including their history, and every curated file must remain byte-identical.

`render_gradients_only.py` calls the current renderer's unchanged gradient/table
functions, with the original pinned numerical analysis. It never calls either
reward plotter. The current note includes the individual-starting-value amendment;
the wrapper also points to the preserved three-seed reward comparison.

The copied independent checker differs only in scope: it no longer requires or
recreates reward figures and instead requires the gradient-only scope marker.
Its raw-data recomputation, jackknife calculations, fitted-scale interval checks,
table checks, and gradient-axis checks are unchanged. `source_snapshot.json`
records the source locations; `mechanical_checks.json` records seven successful
Python 3.8 checks using temporary publication fixtures only.

Execution creates `clean_v5_20261012/final_analysis/` and refuses to overwrite it.
The renderer and checker have 900-second and 600-second subprocess limits.
Completion records and catalog entries update only after the real data and
published artifacts pass their checks. No curation is automatic.
