"""Read-only reward/provenance loader for clean figures v3 (Python 3.8).

``load_reward_data(project_root)`` returns JSON-serializable curves and metadata.
It never samples an environment, estimates missing episode dispersion, or writes files.
"""

import importlib.util
import json
import math
from pathlib import Path
import pickle
import re


METHODS = ("DPPO", "without_PPO", "without_PPO_and_GAE")
EPISODES = 80
STEPS_PER_UPDATE = 80000


def _pointer(path, needle):
    lines = path.read_text().splitlines()
    for number, line in enumerate(lines, 1):
        if needle in line:
            return str(path) + ":" + str(number)
    raise ValueError("Required source evidence missing: %s in %s" % (needle, path))


def _config_number(text, key):
    match = re.search(r"^\s*" + re.escape(key) + r":\s*([0-9.eE+-]+)\s*$", text, re.M)
    if match is None:
        raise ValueError("Numeric native config key missing: " + key)
    return float(match.group(1))


def _sampler_metadata(project, configs):
    """Evaluate the local schedule in its native float32 precision on CPU."""
    import torch

    source = project / "results/repro_halfcheetah/dppo"
    sampling = source / "model/diffusion/sampling.py"
    spec = importlib.util.spec_from_file_location("clean_v3_local_sampling", str(sampling))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    betas = module.cosine_beta_schedule(20)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    previous = torch.cat((torch.ones(1), alpha_bar[:-1]))
    variance = betas * (1.0 - previous) / (1.0 - alpha_bar)
    std = torch.exp(0.5 * torch.log(torch.clamp(variance, min=1e-20)))
    minimum = float(std[1:].min())
    binding = (std < 0.1).nonzero(as_tuple=False).flatten().tolist()
    if minimum <= 0.001 or round(minimum, 3) != 0.076 or binding != [0, 1]:
        raise ValueError("The native DDPM schedule no longer matches the caption")
    noise_clamps = {}
    for method, cfg in configs.items():
        if _config_number(cfg, "denoising_steps") != 20:
            raise ValueError("Unexpected number of denoising steps: " + method)
        if _config_number(cfg, "min_sampling_denoising_std") != 0.1:
            raise ValueError("Unexpected training sampler floor: " + method)
        noise_clamps[method] = _config_number(cfg, "randn_clip_value")
    vpg = source / "model/diffusion/diffusion_vpg.py"
    diffusion = source / "model/diffusion/diffusion.py"
    wrapper = source / "env/gym_utils/wrapper/mujoco_locomotion_lowdim.py"
    return {
        "schedule": "20-step cosine DDPM; native CPU float32 coefficients",
        "posterior_std_by_t": [float(v) for v in std],
        "min_posterior_std_t_ge_1": minimum,
        "training_floor_binding_t": binding,
        "evaluation_last_step_std": 0.0,
        "training_last_step_std": 0.1,
        "training_noise_clamp_sd": noise_clamps,
        "evaluation_description": (
            'Evaluation uses DPPO\'s "deterministic" mode, which retains the DDPM\'s own '
            'noise sqrt(beta_tilde_t) at every step except the noiseless last step; '
            'the numerical lower bound never binds, since the minimum '
            'sqrt(beta_tilde_t) for t >= 1 is approximately 0.076.'),
        "training_description": (
            "Training raises the noise standard deviation to at least 0.1, which binds "
            "only at the final two denoising steps. At the final step it adds nominal "
            "N(0, 0.1^2) noise to each action coordinate, so training returns sit below "
            "evaluation returns."),
        "noise_truncation_note": (
            "DPPO retains its original clipping of standard-normal draws at +/-3; "
            "the other two methods use +/-100. The Gaussian description is before "
            "this clipping."),
        "reset_seeding_note": (
            "The wrapper's seed() updates NumPy's global generator, while reset() "
            "calls the inner environment without passing a seed; the MuJoCo reset "
            "generator is not seeded by this wrapper."),
        "evidence": {
            "schedule": _pointer(sampling, "def cosine_beta_schedule"),
            "posterior_variance": _pointer(diffusion, "self.ddpm_var ="),
            "sampler": _pointer(vpg, "if deterministic and t == 0:"),
            "noise_truncation": _pointer(vpg, "noise = torch.randn_like(x).clamp_"),
            "seed": _pointer(wrapper, "def seed("),
            "reset": _pointer(wrapper, "raw_obs = self.env.reset()"),
        },
    }


def _read_run(method, path):
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    if not rows or [int(row["itr"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Missing or nonconsecutive native iterations: " + method)
    if method != METHODS[2] and len(rows) != 140:
        raise ValueError("Reference run must be complete: " + method)
    config_path = path.parent / ".hydra/config.yaml"
    cfg = config_path.read_text()
    for key, expected in (("seed", 0), ("n_envs", 40), ("n_steps", 500),
                          ("max_episode_steps", 1000), ("val_freq", 10)):
        if _config_number(cfg, key) != expected:
            raise ValueError("Unexpected %s in %s" % (key, method))
    curves = {"eval": [], "train": []}
    for row in rows:
        itr = int(row["itr"])
        train = "train_episode_reward" in row
        expected = sum(j % 10 != 0 for j in range(itr))
        step = int(row["step"])
        update = step // STEPS_PER_UPDATE - int(train)
        if step % STEPS_PER_UPDATE or update != expected:
            raise ValueError("Native rollout update mismatch: %s itr %d" % (method, itr))
        kind = "train" if train else "eval"
        value = float(row[kind + "_episode_reward"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite reference reward: " + method)
        if "episode_count" in row and int(row["episode_count"]) != EPISODES:
            raise ValueError("Unexpected episode count: " + method)
        curves[kind].append({"itr": itr, "update": update,
                             "env_steps": update * STEPS_PER_UPDATE,
                             "logged_env_steps": step, "reward": value,
                             "episodes": EPISODES})
    log_path = path.parent / "run.log"
    matches = re.findall(r"eval:.*?avg episode reward\s+([-+0-9.eE]+)", log_path.read_text())
    if len(matches) != len(curves["eval"]):
        raise ValueError("Native evaluation log/row count mismatch: " + method)
    for item, logged in zip(curves["eval"], matches):
        if abs(item["reward"] - float(logged)) > 0.00005001:
            raise ValueError("Native evaluation log/reward mismatch: " + method)
    final = next((item for item in curves["eval"] if item["itr"] == 130), None)
    if final and (final["update"] != 117 or final["env_steps"] != 9360000):
        raise ValueError("Final evaluation update mapping is incorrect: " + method)
    checks = {"status": "complete" if len(rows) == 140 else "partial",
              "iterations_saved": len(rows), "last_iteration": int(rows[-1]["itr"]),
              "final_eval": final["reward"] if final else None,
              "final_eval_log_rounded": float(matches[-1]) if final else None,
              "final_eval_matches_log": bool(final), "result_path": str(path),
              "log_path": str(log_path), "config_path": str(config_path)}
    return rows, curves, cfg, checks


def load_reward_data(project_root):
    """Load exact seed-0 curves, point mappings, sampler facts and missing-SE flags."""
    project = Path(project_root).resolve()
    root = project / "results/stage3/gradmean"
    points = json.loads((root / "provenance/checkpoints.json").read_text())
    paths = {METHODS[0]: Path(points["D1"]["checkpoint"]).parent.parent / "result.pkl",
             METHODS[1]: Path(points["N1"]["checkpoint"]).parent.parent / "result.pkl",
             METHODS[2]: project / "results/stage3/e1c/E1C_seed0/result.pkl"}
    if Path(points["D2"]["checkpoint"]).parent.parent / "result.pkl" != paths[METHODS[0]]:
        raise ValueError("The measured DPPO points came from different runs")
    if Path(points["N2"]["checkpoint"]).parent.parent / "result.pkl" != paths[METHODS[1]]:
        raise ValueError("The measured without_PPO points came from different runs")
    curves, rows, configs, checks = {}, {}, {}, {}
    for method in METHODS:
        rows[method], curves[method], configs[method], checks[method] = _read_run(method, paths[method])
    mapping = {"P0": {"method": "shared pretrained policy", "checkpoint_file":
                       Path(points["P0"]["checkpoint"]).name,
                       "checkpoint_path": points["P0"]["checkpoint"],
                       "update_number": 0, "env_steps": 0, "checkpoint_itr": None,
                       "checkpoint_exists": Path(points["P0"]["checkpoint"]).is_file()}}
    for method, prefix in zip(METHODS, ("D", "N", "C")):
        for suffix, itr, expected in (("1", 70, 63), ("2", 139, 126)):
            point = prefix + suffix
            checkpoint = (Path(points[point]["checkpoint"]) if point in points else
                          paths[method].parent / "checkpoint" / ("state_%d.pt" % itr))
            native = next((row for row in rows[method] if row["itr"] == itr), None)
            updates = sum("train_episode_reward" in row for row in rows[method] if row["itr"] <= itr)
            if native and (updates != expected or int(native["step"]) != updates * STEPS_PER_UPDATE):
                raise ValueError("Checkpoint mapping conflict: " + point)
            if native and not checkpoint.is_file():
                raise ValueError("Measured checkpoint missing: " + str(checkpoint))
            mapping[point] = {"method": method, "checkpoint_file": checkpoint.name,
                              "checkpoint_path": str(checkpoint), "checkpoint_itr": itr,
                              "update_number": expected, "env_steps": expected * STEPS_PER_UPDATE,
                              "checkpoint_exists": checkpoint.is_file(),
                              "mapping_verified": native is not None,
                              "result_path": str(paths[method])}
    pooled = {}
    for kind in ("eval", "train"):
        starts = [curves[method][kind][0] for method in METHODS]
        if any(row["update"] != 0 for row in starts):
            raise ValueError("The starting rollouts do not all use the pretrained policy")
        pooled[kind] = {"update": 0, "env_steps": 0,
                        "mean": sum(row["reward"] for row in starts) / len(starts),
                        "n_episodes": EPISODES * len(starts), "se": None,
                        "se_status": "unavailable_original_episode_dispersion_not_saved",
                        "se_reason": ("Original episode returns or undiscounted return SD/sumsq "
                                      "were not saved. Discounted-return SE and SD across three "
                                      "run means are not substitutes."),
                        "source_means": dict(zip(METHODS, [row["reward"] for row in starts]))}
    sampler = _sampler_metadata(project, configs)
    reference = json.loads((project / "results/stage3/analysis_record.json").read_text())
    rerun = str(paths[METHODS[0]]) != reference["reference_sources"]["Stage 1 DPPO"][0]
    caption_common = (
        "One seed per method (seed 0), 80 episodes per point. Each line starts from one "
        "shared mean over the three pretrained-policy batches (240 episodes). Its "
        "episode standard error is unavailable because original episode dispersion "
        "was not saved. The x-axis counts actor updates already applied before the "
        "rollout; the top axis is training environment steps in millions. Dashed "
        "lines mark the measured checkpoints at 63 and 126 updates.")
    return {"methods": list(METHODS), "curves": curves, "checkpoint_mapping": mapping,
            "pooled_start": pooled, "final_eval_checks": checks,
            "DPPO_measured_seed_is_rerun": rerun, "sampler": sampler,
            "captions": {
                "eval": (caption_common + " " + sampler["evaluation_description"] +
                         " DPPO's seed-to-seed sd at the final evaluation is about 108 "
                         "(3 seeds), so the difference between two single-seed curves "
                         "has sd of about 150. " + sampler["noise_truncation_note"]),
                "train": (caption_common + " No smoothing or seed bands. " +
                          sampler["training_description"] + " " + sampler["noise_truncation_note"]),
            },
            "evidence": {
                "checkpoint_source": str(root / "provenance/checkpoints.json"),
                "DPPO_post_update_checkpoint": _pointer(project / "results/repro_halfcheetah/dppo/agent/finetune/train_ppo_diffusion_agent.py", "# Save model"),
                "without_PPO_post_update_checkpoint": _pointer(project / "results/stage2_noclip/dppo/agent/finetune/train_ppo_diffusion_agent.py", "# Save model"),
                "without_PPO_and_GAE_post_update_checkpoint": _pointer(project / "results/stage3/e1c/runs/gate/source_snapshot/agent/finetune/train_ppo_diffusion_agent.py", "# Save model"),
                "missing_raw_rollouts": _pointer(project / "results/repro_halfcheetah/dppo/agent/finetune/train_agent.py", "self.save_trajs ="),
            }}
