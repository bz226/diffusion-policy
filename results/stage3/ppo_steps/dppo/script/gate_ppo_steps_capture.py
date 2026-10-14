"""CPU-only R1 source and state-capture mechanics; no environment or rollouts."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "script"))

import numpy as np
import torch

from agent.finetune import ppo_steps_state as capture
from replay_ppo_steps import compare
from util.reward_scaling import RunningRewardScaler
from util.scheduler import CosineAnnealingWarmupRestarts


class StripCapture(ast.NodeTransformer):
    def visit_If(self, node):
        if (isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name) and
                node.test.value.id == "self" and node.test.attr == "ppo_steps_capture"):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and
               target.value.id == "self" and target.attr == "ppo_steps_capture" for target in node.targets):
            return None
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    relative = Path("agent/finetune/train_ppo_diffusion_agent.py")
    old = ast.parse((args.baseline_source / relative).read_text())
    new = ast.parse((SOURCE / relative).read_text())
    stripped = StripCapture().visit(copy.deepcopy(new))
    identical = ast.dump(old, include_attributes=False) == ast.dump(stripped, include_attributes=False)
    if not identical:
        raise AssertionError("Removing only guarded capture hooks does not reproduce the original native AST")
    tree, path = capture.native_update_source()
    compile(ast.fix_missing_locations(tree), str(path), "exec")

    model = torch.nn.Module()
    model.actor = torch.nn.Linear(2, 2)
    model.actor_ft = torch.nn.Linear(2, 2)
    model.critic = torch.nn.Linear(2, 1)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(False)
    model.ft_denoising_steps_cnt, model.ft_denoising_steps = 36, 10
    actor = torch.optim.AdamW(model.actor_ft.parameters(), lr=1e-4, weight_decay=0)
    critic = torch.optim.AdamW(model.critic.parameters(), lr=1e-3, weight_decay=0)
    actor_scheduler = CosineAnnealingWarmupRestarts(actor, first_cycle_steps=1000,
                                                  max_lr=1e-4, min_lr=1e-4, warmup_steps=10)
    critic_scheduler = CosineAnnealingWarmupRestarts(critic, first_cycle_steps=1000,
                                                   max_lr=1e-3, min_lr=1e-3, warmup_steps=10)
    for optimizer, module in ((actor, model.actor_ft), (critic, model.critic)):
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        optimizer.step()
    agent = SimpleNamespace(model=model, actor_optimizer=actor, critic_optimizer=critic,
                            actor_lr_scheduler=actor_scheduler, critic_lr_scheduler=critic_scheduler,
                            running_reward_scaler=RunningRewardScaler(2), itr=35,
                            diagnostic_generator=torch.Generator(device="cpu"))
    agent.running_reward_scaler(np.array([[1., 2.], [3., 4.]]), np.array([[1., 0.], [1., 0.]]))
    before_rng = capture.rng_state()
    before_state = capture.snapshot(agent)
    with tempfile.TemporaryDirectory(prefix="ppo_capture_gate_") as temporary:
        file = Path(temporary) / "state.pt"
        capture._save(file, before_state)
        loaded = torch.load(str(file), map_location="cpu", weights_only=False)
        capture_rng_inert = compare(before_rng, capture.rng_state())["bitwise_equal"]
        assert capture_rng_inert
        assert all(isinstance(key, int) for key in loaded["actor_optimizer"]["state"])
        assert all(isinstance(key, int) for key in loaded["critic_optimizer"]["state"])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        actor.param_groups[0]["lr"] = 0.2
        agent.running_reward_scaler.ret_rms.var = np.array(123.)
        agent.running_reward_scaler.ret[:] = 12
        agent.itr = 99
        capture.restore(agent, loaded, restore_rng=True)
        restored = capture.snapshot(agent)
        roundtrip = compare(before_state, restored)
        if not roundtrip["bitwise_equal"]:
            raise AssertionError("Full-state round trip was not bitwise identical")
        frozen_copy = compare(loaded, before_state)
        assert frozen_copy["bitwise_equal"]
        # restore_rng=False must leave the caller's independent collection stream intact.
        torch.rand(3)
        stream = capture.rng_state()
        capture.restore(agent, loaded, restore_rng=False)
        restore_preserves_rng = compare(stream, capture.rng_state())["bitwise_equal"]
        assert restore_preserves_rng

    result = {"status": "passed", "gate": "R1",
              "baseline_source": str(args.baseline_source.resolve()),
              "edited_source": str(SOURCE),
              "native_AST_identical_after_removing_only_capture_guards": identical,
              "native_update_AST_compiles": True,
              "snapshot_and_serialization_preserve_global_RNG": capture_rng_inert,
              "optimizer_integer_keys_preserved": True,
              "full_state_roundtrip": roundtrip,
              "snapshot_independent_of_live_mutation": frozen_copy["bitwise_equal"],
              "default_restore_preserves_collection_RNG": restore_preserves_rng,
              "environment_created": False, "scientific_sampling": False,
              "python": sys.version, "torch": torch.__version__}
    (args.output / "capture_gate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
