# Administrative recovery and E1C admission priority

Seven collection jobs completed all 50 batches with native exit code 0 and clean environment shutdown, then their wrappers failed the obsolete `stage3-gradmean` branch check. The approved recovery accepts their scientific outputs after the existing verifier passes. Their scheduler state remains `FAILED`, their wrapper manifest remains `failed`, and all original manifests and pre-recovery campaign records are preserved here. This does not add samples or change any estimator, checkpoint, seed, or stopping threshold.

The recovered groups are D2_j62, D2_j63, and N2_j46–50. N2_j51/52 failed before sampling and are not recovered; the controller's existing single-replacement rule assigns seeds 1045100 and 1045200. Recovered groups continue to contribute their original allocated GPU seconds; the campaign cap remains 125 GPU-hours.

New optional controller settings leave defaults unchanged. `--max-parallel 5` lowers admissions while three E1C jobs have priority. `--respect-free-a6000` further limits admissions using live free A6000 GPUs, 40 CPUs and 64 GiB per job across the `soal` partition, including resources occupied by other users. Already running jobs are untouched. After E1C finishes, restoring the default maximum requires restarting only the controller or retaining the conservative limit.

Python 3.8 verification passed: seven real native outputs, positive and negative recovery guards, mocked ledger reconciliation, bounds on concurrent jobs, live-resource parsing, and all 20 existing runtime tests. The scientific source matches the passed gate snapshot. See [verification](verification.json), [original records](original_manifests/), [configuration](recovery.json), [process baseline](process_baseline.json), and [exact administrative diff](administrative_changes.patch).

Controller 17821085 resumed collection after all three E1C seeds were running. It was subsequently replaced by CPU controller 17821138 solely to apply the 50-minute continuation monitor. Actual before/after process inspection confirmed that all five independent GPU wrappers and native processes survived with identical PIDs; no GPU job was stopped. [Monitor verification](monitor_interval_verification.json) and [exact restart command](monitor_restart_submission.json) preserve the evidence. The current command is:

```bash
sbatch --parsable --export=PATH,HOME,USER,LOGNAME,LANG \
  --output=/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/setup/controller_monitor50m-%j.out \
  /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/code/run_queue.sh \
  --gate /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/gate_live_G1/gate_summary.json \
  --commit f425082723799776af35e100983fe20b86888a02 \
  --timeout-seconds 86400 --max-parallel 5 --respect-free-a6000 --monitor-interval-seconds 3000 \
  --administrative-recovery /soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean/runs/administrative_recovery_20261012_e1c/recovery.json
```

The controller updates accounting first, preserves actual failure states, applies the explicit recovery, and resumes the original point and round order. No bootstrap repetition, actor update, or critic-addendum resume is involved.
