"""Original reward curves and own-critic checkpoint log audit for clean v5.

Python 3.8. No sampling, fitting, model modification or scientific code changes.
``load_reward_data(project_root)`` is a read-only, JSON-serializable interface.
The CLI saves the explicitly requested audit file without overwriting evidence.
"""

import argparse
import json
import math
from pathlib import Path
import pickle

from clean_v3_rewards import load_reward_data as _load_v3_rewards
from clean_v3_rewards import _pointer


CHECKPOINTS = ((0, 0), (35, 32), (70, 63), (105, 95), (139, 126))
OWN_METHOD = "without_PPO_and_GAE"


def _mean_sample_se(values):
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sample_sd": None, "sample_se": None}
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sample_sd": sd,
            "sample_se": sd / math.sqrt(n) if sd is not None else None}


def load_reward_data(project_root):
    """Return three saved seed-0 curves, five checkpoint records, and log windows.

    ``checkpoints`` is keyed by checkpoint iteration strings. Its stable fields
    include checkpoint_itr, update_number, divisor and logged_itr. An unavailable
    final divisor remains None; this function never substitutes or fits one.
    """
    project = Path(project_root).resolve()
    previous = _load_v3_rewards(project)
    checks = previous["final_eval_checks"]
    if any(value["status"] != "complete" for value in checks.values()):
        raise ValueError("Clean v5 requires the three completed original training runs")
    result_path = Path(checks[OWN_METHOD]["result_path"])
    with result_path.open("rb") as stream:
        rows = pickle.load(stream)
    train = [row for row in rows if "train_episode_reward" in row]
    source = project / "results/stage3/e1c/runs/gate/source_snapshot"
    scaler_source = source / "util/reward_scaling.py"
    support_source = source / "agent/finetune/stage3_support.py"
    critic_source = source / "agent/finetune/critic_baseline.py"
    agent_source = source / "agent/finetune/train_ppo_diffusion_agent.py"
    init_note = (
        "The native initial divisor is sqrt(1 + 1e-8) = 1.000000005, as logged at "
        "iteration 1. Use this exact logged value rather than replacing it with "
        "the prompt's rounded value 1.")
    checkpoints = {}
    for itr, updates in CHECKPOINTS:
        checkpoint = result_path.parent / "checkpoint" / ("state_%d.pt" % itr)
        if not checkpoint.is_file():
            raise ValueError("Required original checkpoint missing: " + str(checkpoint))
        native = rows[itr]
        count = sum("train_episode_reward" in row for row in rows if row["itr"] <= itr)
        if count != updates or native["step"] != updates * 80000:
            raise ValueError("Native checkpoint/update mapping differs from the v5 plan")
        next_row = next((row for row in train if int(row["step"]) // 80000 - 1 == updates), None)
        c = float(next_row["baseline_divisor"]) if next_row is not None else None
        if c is not None and (not math.isfinite(c) or c <= 0):
            raise ValueError("Logged baseline divisor is invalid")
        # Windows are centered on the policy before a rollout; evaluation-only
        # iterations are absent. Inclusive +/-10 updates gives up to21 values.
        window = [row for row in train if abs(int(row["step"]) // 80000 - 1 - updates) <= 10]
        selected = []
        for row in window:
            cosine = row.get("grad_cos_prev")
            if cosine is not None and not math.isfinite(float(cosine)):
                raise ValueError("Nonfinite saved gradient cosine")
            selected.append({"itr": int(row["itr"]), "update_before_rollout": int(row["step"]) // 80000 - 1,
                             "grad_cos_prev": float(cosine) if cosine is not None else None,
                             "missing_reason": row.get("grad_cos_prev_reason") if cosine is None else None})
        cosine_values = [row["grad_cos_prev"] for row in selected if row["grad_cos_prev"] is not None]
        cosine = _mean_sample_se(cosine_values)
        cosine.update(requested_window=[updates - 10, updates + 10],
                      observed_window=[selected[0]["update_before_rollout"], selected[-1]["update_before_rollout"]],
                      iteration_span=[selected[0]["itr"], selected[-1]["itr"]],
                      training_rows=len(window), selected_rows=selected,
                      source_file=str(result_path), source_key="grad_cos_prev",
                      window_rule="inclusive +/-10 actor updates, centered on pre-rollout policy",
                      uncertainty_note="Sample sd/sqrt(n) is descriptive; adjacent training gradients may be dependent.")
        ev = float(next_row["explained_var"]) if next_row is not None else None
        checkpoints[str(itr)] = {
            "method": OWN_METHOD, "checkpoint_itr": itr, "update_number": updates,
            "training_env_steps": updates * 80000, "checkpoint_file": checkpoint.name,
            "checkpoint_path": str(checkpoint), "mapping_verified": True,
            "divisor": c, "logged_itr": int(next_row["itr"]) if next_row is not None else None,
            "divisor_status": "exact_next_training_iteration_start" if c is not None else "requires_authorized_five_batch_least_squares_fallback",
            "divisor_source": {"file": str(result_path), "key": "baseline_divisor",
                               "itr": int(next_row["itr"]) if next_row is not None else None,
                               "value": c},
            "divisor_note": init_note if itr == 0 else (
                "Captured before the next batch updates the scaler, with the critic already stored in this checkpoint." if c is not None else
                "No next training batch was logged. Checkpoints save itr and model only, with no reward-scaler state. "
                "Iteration 139's baseline_divisor is pre-final-update and is not this checkpoint's final scaler."),
            "final_pre_update_divisor_not_used": float(native["baseline_divisor"]) if itr == 139 else None,
            "grad_cos_prev_window": cosine,
            "logged_explained_var": {
                "value": ev, "itr": int(next_row["itr"]) if next_row is not None else None,
                "update_before_rollout": updates if next_row is not None else None,
                "source_file": str(result_path), "source_key": "explained_var",
                "status": "exact_checkpoint_critic_on_next_batch" if next_row is not None else "unavailable_no_rollout_from_final_policy",
                "definition": "1 - Var(GAE target - pre-update critic value) / Var(GAE target), using scaled-reward DPPO targets; not raw-MC explained variance.",
            },
        }
    # Preserve the original (unpooled) samples. A renderer may show the approved
    # pooled origin, but no earlier native curve value is silently overwritten.
    common = (
        "One seed per method (seed 0), 80 episodes per point. The x-axis is training "
        "environment steps used before the rollout. The shared starting mean pools "
        "240 original episodes; its episode standard error is unavailable because "
        "episode dispersion was not saved. No seed bands are shown.")
    captions = {
        "eval": common + " " + previous["sampler"]["evaluation_description"] +
                " DPPO's seed-to-seed sd at the final evaluation is about 108 (3 seeds), "
                "so the difference between two single-seed curves has sd of about 150.",
        "train": common + " No smoothing. " + previous["sampler"]["training_description"] +
                 " DPPO retains its original +/-3-sigma truncation of the nominal Gaussian noise.",
    }
    return {
        "version": 5, "methods": previous["methods"], "curves": previous["curves"],
        "pooled_start": previous["pooled_start"], "final_eval_checks": checks,
        "DPPO_measured_seed_is_rerun": previous["DPPO_measured_seed_is_rerun"],
        "sampler": previous["sampler"], "captions": captions,
        "checkpoints": checkpoints, "checkpoint_iterations": [itr for itr, _ in CHECKPOINTS],
        "checkpoint_updates": [updates for _, updates in CHECKPOINTS],
        "divisor_initialization_note": init_note,
        "final_divisor_fit_required": True,
        "final_divisor_fit_scope": "User-authorized five fresh batches with the final checkpoint's critic frozen; no fit performed by this audit.",
        "evidence": {
            "divisor_formula": _pointer(critic_source, "def reward_scaler_divisor"),
            "iteration_start_capture": _pointer(support_source, "def _critic_baseline_iteration_start"),
            "divisor_logged": _pointer(support_source, 'self._stage3_metrics["baseline_divisor"]'),
            "scaler_update": _pointer(scaler_source, "self.ret_rms.update("),
            "scaler_initial_variance": _pointer(scaler_source, "self.var = np.ones(shape)"),
            "critic_preupdate_prediction": _pointer(agent_source, "values = self.model.critic(obs)"),
            "explained_variance": _pointer(agent_source, "y_pred, y_true = values_k"),
            "gradient_capture": _pointer(support_source, "def _stage3_before_step"),
            "checkpoint_contents": _pointer(source / "agent/finetune/train_agent.py", "def save_model"),
        },
        "missing_original_reward_se": True, "scientific_sampling_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = load_reward_data(args.project)
    output = args.output or args.project / "results/stage3/gradmean/runs/clean_v5_20261012/log_audit.json"
    text = json.dumps(audit, indent=2, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text() != text:
            raise ValueError("Refusing to overwrite a different audit: " + str(output))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(text)
    print(json.dumps({"audit": str(output), "checkpoint_updates": audit["checkpoint_updates"],
                      "exact_divisors": 4, "final_divisor_fit_required": True}))


if __name__ == "__main__":
    main()
