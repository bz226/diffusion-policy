"""Fresh-process R2 replay of the captured DPPO iteration-36 update only."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from agent.finetune.ppo_steps_state import native_update, restore, restore_rng, snapshot


def compare(left, right):
    """Relative error uses max(|reference|, |actual|, 1e-12) per coordinate."""
    result = {"max_abs": 0.0, "max_relative": 0.0, "bitwise_equal": True,
              "finite": True, "coordinates": 0, "structural_mismatches": []}
    def visit(a, b, path):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or tuple(a.shape) != tuple(b.shape):
                result["structural_mismatches"].append(path)
                return
            av, bv = a.detach().cpu(), b.detach().cpu()
            result["bitwise_equal"] = result["bitwise_equal"] and torch.equal(av, bv)
            av, bv = av.double(), bv.double()
            result["coordinates"] += av.numel()
            if not bool(torch.isfinite(av).all() and torch.isfinite(bv).all()):
                result["finite"] = False
                return
            if av.numel():
                diff = (av - bv).abs()
                denominator = torch.maximum(av.abs(), bv.abs()).clamp_min(1e-12)
                result["max_abs"] = max(result["max_abs"], float(diff.max()))
                result["max_relative"] = max(result["max_relative"], float((diff / denominator).max()))
        elif isinstance(a, np.ndarray):
            visit(torch.from_numpy(a), torch.from_numpy(b), path)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                result["structural_mismatches"].append(path)
                return
            for key in a:
                visit(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)) or len(a) != len(b):
                result["structural_mismatches"].append(path)
                return
            for index, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, path + "/" + str(index))
        elif a != b:
            result["structural_mismatches"].append(path)
    visit(left, right, "state")
    result["bitwise_equal"] = result["bitwise_equal"] and not result["structural_mismatches"]
    return result


def load(path):
    # These files are our own trusted captures and include NumPy/Python RNG state.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.run_dir.resolve() / "ppo_steps"
    cfg = OmegaConf.create(json.loads((directory / "config.json").read_text()))
    cfg.logdir = str(args.output.resolve())
    cfg.device = args.device
    cfg.model.device = args.device
    cfg.train.ppo_steps_capture = False
    cfg.wandb = None
    state35 = load(directory / "full_state_35.pt")
    saved = load(directory / "replay_itr36_input.pt")
    expected = load(directory / "replay_itr36_expected.pt")
    if state35["itr"] != 35 or saved["itr"] != 36 or expected["itr"] != 36:
        raise ValueError("Replay requires states 35/36 exactly")
    agent, error = None, None
    result = {"gate": "R2", "status": "running", "device": args.device,
              "source_full_state": str(directory / "full_state_35.pt"),
              "source_input": str(directory / "replay_itr36_input.pt"),
              "source_expected": str(directory / "replay_itr36_expected.pt"),
              "maximum_relative_tolerance": 1e-5,
              "relative_definition": "abs(a-b)/max(abs(a),abs(b),1e-12) per actor coordinate",
              "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled()}
    try:
        cls = hydra.utils.get_class(cfg._target_)
        agent = cls.__new__(cls)
        agent.logdir = str(args.output.resolve())
        agent.venv = SimpleNamespace(processes=[], close=lambda: None)
        agent.__init__(cfg)
        restore(agent, state35, restore_rng=False)
        before = snapshot(agent)
        result["restored_pre_update"] = {
            key: compare(saved["pre_update_state"][key], before[key])
            for key in ("model", "actor_optimizer", "critic_optimizer", "actor_scheduler", "critic_scheduler")}
        if not all(value["bitwise_equal"] for value in result["restored_pre_update"].values()):
            raise AssertionError("Restored state35 differs from iteration36 pre-update model/optimizer")
        agent.itr = 36
        agent.model.train()
        restore_rng(saved["pre_loop_rng"])
        result["update"] = native_update(agent, saved["batch"])
        actual = snapshot(agent)
        wanted = expected["state"]
        result["comparisons"] = {key: compare(wanted[key], actual[key]) for key in
                                 ("model", "actor_optimizer", "critic_optimizer", "rng")}
        actor_reference = {key: value for key, value in wanted["model"].items() if key.startswith("actor_ft.")}
        actor_actual = {key: value for key, value in actual["model"].items() if key.startswith("actor_ft.")}
        result["actor_parameters"] = compare(actor_reference, actor_actual)
        actor = result["actor_parameters"]
        if (not actor_reference or not actor["finite"] or actor["structural_mismatches"] or
                actor["max_relative"] > 1e-5):
            raise AssertionError("Actor parameter replay exceeds the approved maximum relative tolerance")
        for key in ("actor_optimizer_steps", "critic_optimizer_steps"):
            if result["update"][key] != expected[key]:
                raise AssertionError("Replay step count differs: " + key)
        result["status"] = "passed"
    except BaseException as caught:
        error = caught
        result.update(status="failed", error=type(caught).__name__ + ": " + str(caught))
        raise
    finally:
        if agent is not None:
            from agent.finetune.grad_mean import cleanup_environment
            try:
                result["cleanup"] = cleanup_environment(agent, primary_error=error)
            except BaseException as cleanup_error:
                result.update(status="failed", cleanup_error=type(cleanup_error).__name__ + ": " + str(cleanup_error))
                (args.output / "replay_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                raise
        (args.output / "replay_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
