"""Small synthetic/metadata checks of clean-v5 analysis; Python 3.8, no rollouts."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import clean_v5_stats as stats
from check_clean_v3_stats import fixture
from check_clean_v5_outputs import verify_point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core, source = stats._load_core(args.root)
    passed = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        passed.append(name)
    def close(a, b):
        return np.allclose(a, b, rtol=2e-11, atol=2e-11)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    rng = np.random.default_rng(542955)
    gradients = rng.normal(size=(10, 50, 6, 7)) + np.array([.3, .5, .8, 1.2, 1.7, 2.1])[None, None, :, None]
    fit = [dict(sum_VG=n, sum_V2=d) for n, d in zip((10., 24., 36., 60., 80.), (1., 2., 3., 4., 5.))]
    fitted = dict(effective_scaler_c=210. / 15., scaler_c_fallback_batches=5, scaler_fit_batches=fit)
    fitted_metric = stats.divisor_estimate(fitted)
    deleted_fits = [(210. - row["sum_VG"]) / (15. - row["sum_V2"]) for row in fit]
    expected_fit_se = np.sqrt(4. / 5. * np.square(np.array(deleted_fits) - np.mean(deleted_fits)).sum())
    check("fitted divisor uses delete-one-fit-batch ratio", close(fitted_metric["se"], expected_fit_se))
    check("fitted divisor has t4 90-percent interval", abs(fitted_metric["critical"] - 2.1318468) < 1e-7 and
          fitted_metric["df"] == 4 and close(fitted_metric["lo"], 14. - fitted_metric["critical"] * expected_fit_se))
    fixed_metric = stats.divisor_estimate(dict(effective_scaler_c=1.000000005, scaler_c_fallback_batches=0))
    check("logged divisor remains fixed without sampling interval", fixed_metric["lo"] is None and fixed_metric["se"] is None and fixed_metric["status"] == "fixed_provided")
    with tempfile.TemporaryDirectory(prefix="clean-v5-check-") as temporary:
        root = Path(temporary)
        run = root / "runs/clean_v5_20261012"
        paths = [fixture(run / "collect" / ("W0_000_j%02d" % i), i, values) for i, values in enumerate(gradients)]
        for path in paths:
            text = (path / "batches.csv").read_text().splitlines()
            (path / "batches.csv").write_text(text[0] + ",explained_variance,frozen_reward_divisor\n" +
                "\n".join(line + ",0.25,1.000000005" for line in text[1:]) + "\n")
        data = core.PointData("P0", paths, variants=stats.VARIANTS)
        result = stats.statistics_from_data(data)
        def independent(ids):
            k = len(ids) * 50
            s = data.sums[:, ids].sum(axis=1)
            q = data.q[ids].sum(axis=0)
            cross = s @ s.T
            u = (cross - q) / (k * (k - 1))
            variance = (np.diag(q) - np.diag(cross) / k) / (k - 1)
            return u, variance
        signal, trace = independent(np.arange(10))
        deletes = [independent(np.delete(np.arange(10), i)) for i in range(10)]
        for variant, index, multiplier in (("V2", 1, 1.0), ("V6", 5, 2500.0)):
            metrics = result["estimators"][variant]
            check(variant + " uses its OWN U-statistic signal", close(metrics["u_sq"]["value"], signal[index, index] * multiplier ** 2))
            check(variant + " norm has correct units", close(metrics["u_norm"]["value"], np.sqrt(signal[index, index]) * multiplier))
            check(variant + " episode trace scale", close(metrics["trace_ep"]["value"], trace[index] * 80 * multiplier ** 2))
            expected_B = 80 * trace[index] / signal[index, index]
            independent_B = stats.jackknife(expected_B, [80 * t[index] / u[index, index] for u, t in deletes])
            check(variant + " own-denominator joint t jackknife", close(metrics["B_ep"]["value"], expected_B) and close(metrics["B_ep"]["se"], independent_B["se"]))
            check(variant + " per-update interval divides by80", close(metrics["B_update"]["hi"], independent_B["hi"] / 80))
        check("V2-V6 cosine excludes same-batch pairs", close(result["cos_V2_V6"]["value"], signal[1, 5] / np.sqrt(signal[1, 1] * signal[5, 5])))
        expected = stats.jackknife(signal[1, 1] / (signal[1, 1] + trace[1]),
            [u[1, 1] / (u[1, 1] + t[1]) for u, t in deletes])
        check("consecutive cosine is not square root", close(result["predicted_consecutive_cos"]["value"], expected["value"]))
        check("consecutive cosine recomputes each deletion", close(result["predicted_consecutive_cos"]["se"], expected["se"]))
        check("ten-group t9 interval", close(result["predicted_consecutive_cos"]["critical"], 1.8331129326536335))
        check("explained variance group aggregation", result["explained_var"]["value"] == .25 and result["explained_var"]["se"] == 0)
        data.full_moments["signal"] = data.full_moments["signal"].copy()
        data.full_moments["signal"][1] = -.01
        for i, moment in enumerate(data.deleted_moments):
            moment["signal"] = moment["signal"].copy()
            moment["signal"][1] = -.01 + .01 * (i - 4.5)
        prediction = stats.predicted_consecutive_cos(data)
        scale = stats._noise_scale(data, "V2", "V2")
        check("unresolved prediction has upper bound only", prediction["value"] is None and prediction["status"] == "upper_bound" and close(prediction["upper_bound"], 1 / (1 + scale["lower_bound"] / 80)))
        check("negative signal norm avoids square root and gives upper bound", stats._point_norm(data, "V2")["value"] is None and stats._point_norm(data, "V2")["upper_bound"] > 0)
        # Verify the reader admits only exact W=0 source/normalization/frozen
        # cohorts after both scheduler and cleanup records are terminal.
        native_checkpoint = run / "original/state_0.pt"
        native_checkpoint.parent.mkdir(); native_checkpoint.write_bytes(b"synthetic-checkpoint")
        point = "W0_000"
        checkpoint = dict(checkpoint_itr=0, update_number=0, training_env_steps=0,
            checkpoint_path=str(native_checkpoint), divisor=1.000000005,
            divisor_source=dict(file="original-result", key="baseline_divisor", itr=1, value=1.000000005),
            grad_cos_prev_window=dict(n=10, mean=.01, sample_se=.02),
            logged_explained_var=dict(value=-.1, itr=1))
        write(run / "log_audit.json", dict(version=5, checkpoints={"0": checkpoint}))
        norm = dict(source_commit="fixture-commit", selected_normalization="quarter", frozen_before_first_collect=True)
        write(run / "normalization_selection.json", norm)
        norm_check = dict(status="completed", selected_normalization="quarter",
                          rows=[dict(batch=i, cosine=.998, quarter_over_full_norm=1.02) for i in range(2)])
        write(run / "normalization/norm_W0_000/normalization_check.json", norm_check)
        gate = dict(passed=True, device="cpu", slurm_job_id="cpu-job", F1=dict(passed=True), F2=dict(passed=True), F3=dict(passed=True))
        write(run / "gate_cpu/gate_summary.json", gate)
        provenance = dict(gate_path=str(run / "gate_cpu/gate_summary.json"), cpu_gate_job_id="cpu-job", source_commit="fixture-commit")
        cleanup = dict(status="passed", all_children_exited=True, remaining_active_children=[], remaining_direct_children=[], errors=[], primary_error=None,
                       workers=[dict(exitcode=0, alive=False) for _ in range(40)])
        prepare_path = run / "prepare" / point
        prepare_path.mkdir(parents=True); frozen = prepare_path / "frozen_P0.pt"; frozen.write_bytes(b"frozen-fixture")
        entries = []
        names = ["prepare_" + point] + [p.name for p in paths]
        for i, name in enumerate(names):
            preparation = i == 0
            settings = dict(gm_warmup=0, gm_v2_raw=True, gm_include_critic=True, gm_normalization="full" if preparation else "quarter",
                            gm_scaler_c=1.000000005, gm_parent_commit="fixture-commit")
            spec = dict(point=point, collector_point="P0", checkpoint=str(native_checkpoint), parent_commit="fixture-commit", expected_settings=settings)
            write(run / "specs" / (name + ".json"), spec)
            manifest = dict(status="complete", subprocess_exitcode=0, remaining_process_group_members=[], external_point_id=point, collector_point="P0",
                            started_utc="2026-10-12T00:%02d:00+00:00" % i)
            write(run / "workers" / name / "manifest.json", manifest)
            entries.append(dict(run_id=name, job_id=str(i), status="complete", scheduler_state="COMPLETED", exit_code="0:0"))
            directory = prepare_path if preparation else paths[i - 1]
            meta = dict(status="completed", point="P0", source_checkpoint=str(native_checkpoint), settings=settings, parent_commit="fixture-commit",
                        actor_optimizer_steps=0, critic_optimizer_steps=0, frozen_path=str(frozen), frozen_sha256="fixture")
            for state in ("actor", "critic", "scaler"):
                meta[state + "_before_sha256"] = meta[state + "_after_sha256"] = "unchanged"
            if preparation:
                meta.update(completed_warmup_batches=0, warmup=[], effective_scaler_c=1.000000005,
                            scaler_c_source="provided", scaler_c_fallback_batches=0, scaler_fit_batches=[])
            else:
                old = json.loads((directory / "meta.json").read_text()); old.update(meta); meta = old
            write(directory / ("grad_mean.json" if preparation else "meta.json"), meta)
            write(directory / "cleanup.json", cleanup)
        campaign = dict(status="running", entries=entries, source_provenance=provenance)
        write(run / "campaign.json", campaign)
        verified = stats.build_statistics(root, analysis_source=source)
        check("verified own-critic cohort admitted, incomplete points omitted", list(verified["points"]) == [point] and verified["availability"]["W0_035"]["status"] == "pending")
        check("actual provided divisor provenance exposed", verified["points"][point]["divisor_source"]["effective_divisor"] == 1.000000005)
        check("gate evidence exposed for renderer", verified["runmetadata"]["gate"] == gate)
        check("actual normalization rows exposed for renderer", verified["runmetadata"]["normalization_check"] == norm_check)
        check("independent raw-moment output checker passes", verify_point(verified["points"][point])["own_V2_and_V6_signals_verified"])
        wrong_numeric = json.loads(json.dumps(verified["points"][point]))
        wrong_numeric["estimators"]["V6"]["B_ep"]["value"] *= 2
        try:
            verify_point(wrong_numeric)
        except AssertionError:
            check("independent checker rejects wrong B denominator", True)
        else:
            raise AssertionError("Incorrect relative variance passed independent recomputation")
        entries[-1]["scheduler_state"] = "RUNNING"
        write(run / "campaign.json", campaign)
        incomplete = stats.build_statistics(root, analysis_source=source)
        check("native completion cannot admit nonterminal scheduler job", not incomplete["points"] and incomplete["availability"][point]["complete_groups"] == 9)
        entries[-1]["scheduler_state"] = "COMPLETED"; write(run / "campaign.json", campaign)
        wrong = json.loads((paths[0] / "meta.json").read_text()); wrong["settings"]["gm_v2_raw"] = False; write(paths[0] / "meta.json", wrong)
        try:
            stats.build_statistics(root, analysis_source=source)
        except ValueError as error:
            check("wrong raw-V2 flag rejected", "settings" in str(error))
        else:
            raise AssertionError("Wrong V2 definition was admitted")
        json.dumps(verified, allow_nan=False)
        check("strict JSON output", True)
    report = dict(passed=True, checks=passed, count=len(passed), python=__import__("sys").version,
                  synthetic_only=True, rollouts=0, analysis_source=str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
